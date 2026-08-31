"""Evolvable task-orchestration and run-loop wiring.

This file is part of every generation snapshot: the editor may restructure
how a task run is orchestrated (composer dispatch, bundle assembly, module
instantiation, the surrounding lifecycle) — this is where architecture-level
capabilities (persistent memory, subagents, richer context plumbing) get
wired in. The per-turn loop itself lives in `modules/agent_loop/`.

FROZEN BOUNDARY CONTRACT (enforced by the conformance gate; the shim side
lives in the installed, never-snapshotted `agent.py`):

    async def run_task(*, params, instruction, environment, context,
                       logs_dir, logger) -> None

- The keyword parameter NAMES above must not change (the shim calls with
  them; conformance checks them).
- `params` is a plain dict of constructor primitives the shim packs (model
  config, tmux dims, composer_name, staging/modules/trajectory/archive
  paths, …). Evolution may READ any key and may rely on new keys being
  absent-tolerant (`params.get(...)`), but must not require the shim to pack
  new keys — the shim is frozen.
- Everything below the signature is free to evolve, subject to the usual
  gates (conformance, review, probes).

The implementation receives runtime state through ``params`` so generation
snapshots remain independent from the installed shim.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from harbor.agents.terminus_2_modular.composer.editor_static import (
    EditorStaticComposer,
)
from harbor.agents.terminus_2_modular.composer.llm_dynamic import LLMComposer
from harbor.agents.terminus_2_modular.composer.static import StaticComposer
from harbor.agents.terminus_2_modular.library import build_default_library
from harbor.agents.terminus_2_modular.protocols import (
    KernelServices,
    ModuleCtx,
    ModuleSpec,
    ModuleStatsView,
    ObsState,
    RuntimeState,
    SharedResources,
)
from harbor.agents.terminus_2_modular.services import (
    AtifTrajectoryRecorder,
    build_default_services,
)
from harbor.environments.base import BaseEnvironment
from harbor.llms.chat import Chat
from harbor.llms.lite_llm import LiteLLM
from harbor.models.agent.context import AgentContext
from harbor.models.trajectories import FinalMetrics, Step

_logger = logging.getLogger(__name__)

# Solver helper tools: an archived module type with SUBSET arity — a task gets all
# the active helpers at once rather than one winner — so they ride on the tools
# spec's params instead of being a sixth ModuleBundle slot.
_HELPER_TYPE = "tool_helper"
_BUNDLE_TYPES = (
    "agent_loop",
    "observation",
    "context_mgmt",
    "tools",
    "verification",
)


def _composer_cache_file(cache_dir: Path, instruction: str) -> Path:
    digest = hashlib.sha256(instruction.encode("utf-8")).hexdigest()
    return cache_dir / f"{digest}.json"


def _bundle_payload(bundle) -> dict[str, Any]:
    return {
        module_type: {
            "name": getattr(bundle, module_type).name,
            "params": getattr(bundle, module_type).params,
        }
        for module_type in _BUNDLE_TYPES
    }


def _write_composer_cache(
    cache_dir: Path, instruction: str, bundle, session_id: str
) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _composer_cache_file(cache_dir, instruction)
    payload = {
        "format": 1,
        "instruction_sha256": path.stem,
        "bundle": _bundle_payload(bundle),
    }
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{session_id}.tmp")
    tmp.write_text(json.dumps(payload, sort_keys=True) + "\n")
    os.replace(tmp, path)
    return path


def _read_composer_cache(cache_dir: Path, instruction: str):
    path = _composer_cache_file(cache_dir, instruction)
    payload = json.loads(path.read_text())
    if payload.get("format") != 1 or payload.get("instruction_sha256") != path.stem:
        raise ValueError(f"invalid composer cache entry: {path}")
    raw_bundle = payload.get("bundle")
    if not isinstance(raw_bundle, dict):
        raise ValueError(f"composer cache entry has no bundle: {path}")
    specs = {}
    for module_type in _BUNDLE_TYPES:
        raw_spec = raw_bundle.get(module_type)
        if not isinstance(raw_spec, dict) or not raw_spec.get("name"):
            raise ValueError(
                f"composer cache entry has invalid {module_type} spec: {path}"
            )
        params = raw_spec.get("params") or {}
        if not isinstance(params, dict):
            raise ValueError(
                f"composer cache entry has invalid {module_type} params: {path}"
            )
        specs[module_type] = ModuleSpec(name=str(raw_spec["name"]), params=params)
    from harbor.agents.terminus_2_modular.protocols import ModuleBundle

    return ModuleBundle(**specs), path


def build_llm(params: dict[str, Any]) -> LiteLLM:
    """Build LiteLLM with full Terminus2-equivalent kwargs."""
    ctor_kwargs: dict[str, Any] = dict(params.get("llm_kwargs") or {})
    if params.get("temperature") is not None:
        ctor_kwargs["temperature"] = params["temperature"]
    api_key = params.get("api_key")
    if api_key:
        ctor_kwargs["api_key"] = api_key
        os.environ.setdefault("OPENAI_API_KEY", api_key)
    return LiteLLM(
        model_name=params["model_name"],
        api_base=params.get("api_base"),
        collect_rollout_details=params.get("collect_rollout_details", False),
        session_id=params["session_id"],
        max_thinking_tokens=params.get("max_thinking_tokens"),
        reasoning_effort=params.get("reasoning_effort"),
        model_info=params.get("model_info"),
        use_responses_api=params.get("use_responses_api", False),
        **ctor_kwargs,
    )


async def run_task(
    *,
    params: dict[str, Any],
    instruction: str,
    environment: BaseEnvironment | None,
    context: AgentContext,
    logs_dir: Path,
    logger: logging.Logger,
) -> None:
    # Build the real ATIF trajectory recorder up front so modules can
    # append Steps as they go.
    agent_extra: dict[str, Any] = {"parser": params["parser_name"]}
    if params.get("temperature") is not None:
        agent_extra["temperature"] = params["temperature"]
    if params.get("llm_kwargs"):
        agent_extra["llm_kwargs"] = params["llm_kwargs"]
    traj_cfg = params.get("trajectory_config") or {}
    recorder = AtifTrajectoryRecorder(
        logs_dir=Path(logs_dir),
        session_id=params["session_id"],
        agent_name=params["agent_name"],
        agent_version=params.get("agent_version") or "unknown",
        model_name=params["model_name"],
        agent_extra=agent_extra,
        linear_history=bool(traj_cfg.get("linear_history", False)),
        raw_content=bool(traj_cfg.get("raw_content", False)),
        logger=logger,
    )

    services: KernelServices = build_default_services(
        logger, trajectory_recorder=recorder
    )
    shared = SharedResources()
    services.logger.info("terminus-2-modular: starting task")

    state = RuntimeState(
        task_id=params["session_id"],
        gen_id="gen_0",
        env=environment,
        instruction=instruction,
        started_at=time.time(),
        model_name=params["model_name"],
        logs_dir=logs_dir,
        skills_dir=params.get("skills_dir"),
        mcp_servers=params.get("mcp_servers"),
        extra_env=params.get("extra_env"),
        record_terminal_session=params.get("record_terminal_session", True),
        staging_dir=params.get("staging_dir"),
        modules_root=params.get("modules_root"),
        trajectory_root=params.get("trajectory_root"),
        archive_path=params.get("archive_path"),
        locked_module_type=params.get("locked_module_type"),
        composer_scope=params.get("composer_scope") or "locked",
    )
    ctx = ModuleCtx(state=state, services=services, shared=shared)

    # If modules_root is set, load ModuleLibrary from that path (e.g., a
    # candidate gen_N/modules/ directory under self_evo_runs/). Otherwise
    # use the installed package's modules/.
    library = build_default_library(modules_root=params.get("modules_root"))
    # Composer selection (llm_dynamic = per-task solver default,
    # static = fixed default bundle, editor_static = editor mode). Evaluations
    # may pre-compose into an instruction-keyed cache, then replay those exact
    # choices while solver traffic saturates the shared endpoint.
    composer_name = params.get("composer_name", "llm_dynamic")
    cache_mode = str(params.get("composer_cache_mode") or "off").lower()
    cache_dir_raw = params.get("composer_cache_dir")
    cache_dir = Path(cache_dir_raw) if cache_dir_raw else None
    if cache_mode == "read":
        if cache_dir is None:
            raise ValueError("composer cache read mode requires composer_cache_dir")
        bundle, cache_path = _read_composer_cache(cache_dir, instruction)
        services.logger.info("composer cache read: %s", cache_path.name)
    else:
        if composer_name == "editor_static":
            composer = EditorStaticComposer()
        elif composer_name == "llm_dynamic":
            # Per-task LLM selection. Needs the api config (ctx carries only the
            # model name), so build it from the agent's own settings.
            composer = LLMComposer(
                model_name=params["model_name"],
                api_base=params.get("api_base"),
                api_key=params.get("api_key"),
                timeout=int(params.get("composer_timeout", 120)),
                call_kwargs=params.get("composer_llm_kwargs"),
            )
        else:
            composer = StaticComposer()
        bundle = await composer.choose(
            instruction=instruction,
            library=library.list_infos(),
            stats=ModuleStatsView(),
            ctx=ctx,
        )
        if cache_mode == "write":
            if cache_dir is None:
                raise ValueError(
                    "composer cache write mode requires composer_cache_dir"
                )
            cache_path = _write_composer_cache(
                cache_dir, instruction, bundle, params["session_id"]
            )
            services.logger.info("composer cache wrote: %s", cache_path.name)
    services.logger.info(
        "composer chose: loop=%s obs=%s ctx=%s tools=%s verify=%s",
        bundle.agent_loop.name,
        bundle.observation.name,
        bundle.context_mgmt.name,
        bundle.tools.name,
        bundle.verification.name,
    )
    # Record the per-task composition INTO the trajectory (agent.extra.bundle),
    # not just the agent log. When the editor later reflects on this trial it
    # reads trajectory.json — so this is what lets it see WHICH variants
    # actually ran (e.g. whether the variant a past generation wrote was even
    # selected for this task).
    # Solver helpers chosen for this task. Instantiate them from the library here
    # (the tools module has no library handle, and shouldn't need one) and record
    # the NAMES alongside the five picks, so cross-epoch confirm can compare rolls
    # where a helper was present against rolls where it was not.
    helper_names = [str(n) for n in (bundle.tools.params.get("helpers") or [])]
    helper_tools = []
    for hname in helper_names:
        try:
            helper_tools.append(
                library.instantiate(_HELPER_TYPE, ModuleSpec(name=hname))
            )
        except Exception as exc:
            # A broken helper must never take down the solver — drop it and move on.
            _logger.warning(
                "solver helper %r could not be instantiated: %s", hname, exc
            )
    live_helper_names = [
        getattr(h, "name", "") for h in helper_tools if getattr(h, "name", "")
    ]

    recorder.agent_extra["bundle"] = {
        "agent_loop": bundle.agent_loop.name,
        "observation": bundle.observation.name,
        "context_mgmt": bundle.context_mgmt.name,
        "tools": bundle.tools.name,
        "verification": bundle.verification.name,
        _HELPER_TYPE: sorted(live_helper_names),
    }

    if params.get("composer_only", False):
        context.metadata = {
            "composer_only": True,
            "bundle": dict(recorder.agent_extra["bundle"]),
        }
        recorder.dump()
        services.logger.info("composer-only task finished")
        return

    # Override Spec params with agent-level config (Composer's defaults
    # don't know about ctor args).
    agent_loop_spec = ModuleSpec(
        name=bundle.agent_loop.name,
        params={
            "max_iterations": params["max_turns"],
            "llm_call_kwargs": params.get("llm_call_kwargs") or {},
            "raw_content": bool(traj_cfg.get("raw_content", False)),
            **bundle.agent_loop.params,
        },
    )
    tools_spec = ModuleSpec(
        name=bundle.tools.name,
        params={
            "parser_name": params["parser_name"],
            "tmux_pane_width": params["tmux_pane_width"],
            "tmux_pane_height": params["tmux_pane_height"],
            **bundle.tools.params,
            # Instantiated objects, not names — replaces the `helpers` name list
            # the Composer put here.
            "helper_tools": helper_tools,
        },
    )
    context_mgmt_spec = ModuleSpec(
        name=bundle.context_mgmt.name,
        params={
            "enable_summarize": params["enable_summarize"],
            "proactive_summarization_threshold": (
                params["proactive_summarization_threshold"]
            ),
            "linear_history": bool(traj_cfg.get("linear_history", False)),
            **bundle.context_mgmt.params,
        },
    )

    agent_loop = library.instantiate("agent_loop", agent_loop_spec)
    observation = library.instantiate("observation", bundle.observation)
    context_mgmt = library.instantiate("context_mgmt", context_mgmt_spec)
    tools = library.instantiate("tools", tools_spec)
    verification = library.instantiate("verification", bundle.verification)

    # Re-record the helper set from what the tools module ACTUALLY loaded, not
    # what the Composer asked for. A variant may inject helpers of its own during
    # __init__ (observed: an `always_helpers` variant that force-adds four file
    # helpers "regardless of what the Composer selects"). Recording the request
    # would then put rolls where the agent DID have a helper into cross-epoch
    # confirm's "absent" group, contaminating both sides and making the
    # present-vs-absent delta meaningless. Duck-typed: a tools impl with no
    # helper support simply leaves the Composer's list in place.
    _actual = getattr(tools, "_helpers", None)
    if isinstance(_actual, dict):
        recorder.agent_extra["bundle"][_HELPER_TYPE] = sorted(_actual)

    # Kernel-level execution tracing (evaluator-side, resolves to the
    # installed package even under gen injection): wrap every module in a
    # transparent proxy that records one line per Protocol-method call into
    # trial.log + `agent.extra.module_trace`. This is what lets the editor —
    # and the review gate — see whether a variant's declared mechanism ever
    # actually fires at runtime (dead-feature blind writes are invisible to
    # the load/import gates). Never gate on wrap failure: tracing is
    # observability, not behavior.
    try:
        from harbor.agents.terminus_2_modular.tracing import wrap_module

        agent_loop = wrap_module(
            agent_loop, "agent_loop", bundle.agent_loop.name, services
        )
        observation = wrap_module(
            observation, "observation", bundle.observation.name, services
        )
        context_mgmt = wrap_module(
            context_mgmt, "context_mgmt", bundle.context_mgmt.name, services
        )
        tools = wrap_module(tools, "tools", bundle.tools.name, services)
        verification = wrap_module(
            verification, "verification", bundle.verification.name, services
        )
    except Exception as exc:
        services.logger.warning("module tracing disabled (wrap failed): %s", exc)

    llm = build_llm(params)
    chat = Chat(
        model=llm, interleaved_thinking=params.get("interleaved_thinking", False)
    )

    try:
        await tools.setup(ctx)

        initial_obs, _ = await observation.capture(ObsState(), ctx)

        # Build initial prompt: instruction + MCP + skills + template
        instr_aug = instruction
        if hasattr(tools, "build_skills_section"):
            try:
                skills_xml = await tools.build_skills_section(ctx)
                if skills_xml:
                    instr_aug = instruction + skills_xml
            except Exception as exc:
                services.logger.warning("build_skills_section failed: %s", exc)
        initial_prompt = tools.format_initial_prompt(instr_aug, initial_obs.text, ctx)

        # Step 1: initial user prompt (mirrors Terminus2.run lines 1658-1665)
        recorder.append_step(
            Step(
                step_id=1,
                timestamp=datetime.now(timezone.utc).isoformat(),
                source="user",
                message=initial_prompt,
            )
        )

        await agent_loop.run(
            initial_prompt=initial_prompt,
            original_instruction=instruction,
            observation=observation,
            context_mgmt=context_mgmt,
            tools=tools,
            verification=verification,
            chat=chat,
            ctx=ctx,
        )
    finally:
        try:
            await tools.teardown(ctx)
        except Exception as exc:
            services.logger.warning("tools.teardown failed: %s", exc)

        # Populate AgentContext (mirrors Terminus2.run finally block 1677-1702)
        # Merge in any subagent (summarization) metrics from context_mgmt.
        sub_metrics: dict[str, Any] = {}
        try:
            if hasattr(context_mgmt, "get_subagent_metrics"):
                sub_metrics = context_mgmt.get_subagent_metrics()
        except Exception as exc:
            services.logger.warning("context_mgmt.get_subagent_metrics failed: %s", exc)

        try:
            main_in = chat.total_input_tokens
            main_out = chat.total_output_tokens
            main_cache = chat.total_cache_tokens
            main_cost = chat.total_cost

            sub_in = int(sub_metrics.get("total_prompt_tokens", 0) or 0)
            sub_out = int(sub_metrics.get("total_completion_tokens", 0) or 0)
            sub_cache = int(sub_metrics.get("total_cached_tokens", 0) or 0)
            sub_cost = float(sub_metrics.get("total_cost_usd", 0.0) or 0.0)

            context.n_input_tokens = main_in + sub_in
            context.n_output_tokens = main_out + sub_out
            context.n_cache_tokens = main_cache + sub_cache
            total_cost = main_cost + sub_cost
            context.cost_usd = total_cost if total_cost > 0 else None

            # rollout_details = chat.rollout_details + subagent rollouts
            try:
                sub_rollouts = list(sub_metrics.get("rollout_details", []) or [])
                context.rollout_details = chat.rollout_details + sub_rollouts
            except Exception:
                pass

            meta: dict[str, Any] = {
                "summarization_count": int(
                    sub_metrics.get("summarization_count", 0) or 0
                ),
            }
            # Pull api_request_times / n_episodes from agent_loop if it
            # exposes them (BaselineAgentLoop does).
            if hasattr(agent_loop, "get_metrics"):
                try:
                    meta.update(agent_loop.get_metrics())
                except Exception as exc:
                    services.logger.warning("agent_loop.get_metrics failed: %s", exc)
            if params.get("store_all_messages"):
                meta["all_messages"] = chat.messages
            context.metadata = meta
        except Exception as exc:
            services.logger.warning("failed to populate context metrics: %s", exc)

        # Dump ATIF trajectory (with merged final metrics)
        try:
            recorder.set_final_metrics(
                FinalMetrics(
                    total_prompt_tokens=context.n_input_tokens,
                    total_completion_tokens=context.n_output_tokens,
                    total_cached_tokens=context.n_cache_tokens or 0,
                    total_cost_usd=context.cost_usd,
                )
            )
            recorder.dump()
        except Exception as exc:
            services.logger.warning("failed to dump ATIF trajectory: %s", exc)
