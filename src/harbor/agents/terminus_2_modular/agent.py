"""Frozen Harbor entry point for the modular Terminus-2 agent.

The shim translates Harbor constructor arguments into the primitives consumed
by ``kernel/orchestration.py``. Evolution changes module implementations only;
the shim and orchestration boundary remain installed evaluator code.

Forwarding map (unchanged from the pre-split agent):
- LiteLLM ctor   ← api_base, temperature, reasoning_effort, max_thinking_tokens,
                   model_info, use_responses_api, collect_rollout_details,
                   session_id, api_key, llm_kwargs (spread)
- Chat ctor      ← interleaved_thinking
- agent_loop     ← max_turns (as max_iterations), llm_call_kwargs
- context_mgmt   ← enable_summarize, proactive_summarization_threshold
- tools          ← parser_name, tmux_pane_width/height
- trajectory rec ← trajectory_config
- end of run     ← store_all_messages (dumps chat.messages to context.metadata)
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.llms.base import LLMBackend
from harbor.models.agent.context import AgentContext
from harbor.models.agent.trajectory_config import TrajectoryConfig


class Terminus2Modular(BaseAgent):
    """Modular self-evo terminus-2 (shim; orchestration in kernel/)."""

    SUPPORTS_ATIF: bool = True
    SUPPORTS_WINDOWS: bool = False

    def __init__(
        self,
        logs_dir: Path,
        model_name: str | None = None,
        max_turns: int | None = None,
        parser_name: str = "json",
        api_base: str | None = None,
        temperature: float | None = None,
        reasoning_effort: str | None = None,
        collect_rollout_details: bool = False,
        session_id: str | None = None,
        enable_summarize: bool = True,
        proactive_summarization_threshold: int = 8000,
        max_thinking_tokens: int | None = None,
        model_info: dict | None = None,
        trajectory_config: TrajectoryConfig | dict | None = None,
        tmux_pane_width: int = 160,
        tmux_pane_height: int = 40,
        store_all_messages: bool = False,
        record_terminal_session: bool = True,
        interleaved_thinking: bool = False,
        suppress_max_turns_warning: bool = False,
        use_responses_api: bool = False,
        llm_backend: LLMBackend | str = LLMBackend.LITELLM,
        llm_kwargs: dict | None = None,
        llm_call_kwargs: dict[str, Any] | None = None,
        extra_env: dict | None = None,
        # `--ak key=...` from Harbor; treat as api_key
        api_key: str | None = None,
        key: str | None = None,
        # Back-compat with our earlier Phase-2 ctor name
        max_iterations: int | None = None,
        # Editor / self-evo support (Phase 8):
        # - staging_dir: a local filesystem path the editor will edit files in
        # - modules_root: load ModuleLibrary from a specific generation's
        #   modules/ directory; None = use package default
        # - composer_name: pick a registered Composer. Solver default is
        #   "llm_dynamic" (per-task LLM module selection). It makes an LLM call
        #   only when some module type has >1 implementation; with a single impl
        #   per type it is free and yields the default bundle. "static" = always
        #   the default bundle (or an active_bundle.json override) with no call.
        #   "editor_static" = editor mode (forced by the editor subclass).
        staging_dir: str | None = None,
        modules_root: str | None = None,
        # - trajectory_root: editor/self-evo mode only — a path the editor's
        #   file tools may additionally READ (never write), e.g. the batch's
        #   solver output dir, so the editor can investigate raw trajectories.
        trajectory_root: str | None = None,
        # - archive_path: editor mode — the run's archive.json, read by the
        #   read-only `archive` tool for niche/genealogy queries.
        archive_path: str | None = None,
        composer_name: str = "llm_dynamic",
        # Per-endpoint accommodations for the one-shot composer classification.
        # They must cross the shim boundary explicitly; swallowing them in
        # **kwargs leaves orchestration on its 120-second/default-call settings.
        composer_timeout: int = 120,
        composer_llm_kwargs: dict[str, Any] | None = None,
        # Optional two-phase evaluation support. "write" persists the dynamic
        # per-task choice, "read" replays it without an endpoint call, and
        # composer_only returns immediately after writing the choice. Defaults
        # preserve the normal one-shot dynamic-composer path.
        composer_cache_dir: str | None = None,
        composer_cache_mode: str = "off",
        composer_only: bool = False,
        # - locked_module_type: single-module ablation lock (self-evo). When set
        #   (e.g. "agent_loop"), the editor may only write under
        #   modules/<locked>/. None = off.
        # - composer_scope: "locked" (default) = the lock ALSO narrows the
        #   per-task composer to that one type; "all" = the composer picks every
        #   type normally regardless of the lock. Serial lineages use "all" so
        #   the tree they inherited is actually usable. See protocols.RuntimeState.
        locked_module_type: str | None = None,
        composer_scope: str = "locked",
        **kwargs,
    ):
        super().__init__(logs_dir=logs_dir, model_name=model_name, **kwargs)
        if model_name is None:
            raise ValueError("Terminus2Modular requires --model")

        backend_value = (
            llm_backend.value if isinstance(llm_backend, LLMBackend) else llm_backend
        )
        if backend_value != LLMBackend.LITELLM.value:
            raise ValueError(
                f"Unsupported llm_backend {llm_backend!r}; only "
                f"{LLMBackend.LITELLM.value!r} is implemented"
            )
        # max_turns precedence: explicit max_turns > legacy max_iterations >
        # default 1_000_000 (matches Terminus2 behavior).
        if max_turns is not None:
            resolved_max_turns = max_turns
        elif max_iterations is not None:
            resolved_max_turns = max_iterations
        else:
            resolved_max_turns = 1_000_000
        if (
            max_turns is not None or max_iterations is not None
        ) and not suppress_max_turns_warning:
            self.logger.warning(
                "max_turns artificially limited to %d. Consider removing "
                "the limit for better task completion.",
                resolved_max_turns,
            )
        self._max_turns = resolved_max_turns

        # Parser + tmux params (forwarded to tools module)
        self._parser_name = parser_name
        self._tmux_pane_width = tmux_pane_width
        self._tmux_pane_height = tmux_pane_height
        self._record_terminal_session = record_terminal_session
        self._extra_env = extra_env

        # Context-mgmt params (forwarded to context_mgmt module)
        self._enable_summarize = enable_summarize
        self._proactive_summarization_threshold = proactive_summarization_threshold

        # LLM/Chat params
        self._temperature = temperature
        self._reasoning_effort = reasoning_effort
        self._collect_rollout_details = collect_rollout_details
        self._max_thinking_tokens = max_thinking_tokens
        self._model_info = model_info
        self._use_responses_api = use_responses_api
        self._interleaved_thinking = interleaved_thinking
        self._llm_kwargs = dict(llm_kwargs or {})
        self._llm_call_kwargs = dict(llm_call_kwargs or {})
        self._api_base = api_base
        self._api_key = api_key or key

        # End-of-run wiring
        self._store_all_messages = store_all_messages
        self._trajectory_config: TrajectoryConfig = trajectory_config or {}

        # Session id: passed-in wins; otherwise stable per-instance
        self._session_id = session_id or str(uuid.uuid4())

        # Editor / self-evo wiring (None = solver mode, default)
        self._staging_dir = Path(staging_dir) if staging_dir else None
        self._modules_root = Path(modules_root) if modules_root else None
        self._trajectory_root = Path(trajectory_root) if trajectory_root else None
        self._archive_path = Path(archive_path) if archive_path else None
        self._composer_name = composer_name
        if composer_timeout <= 0:
            raise ValueError("composer_timeout must be positive")
        self._composer_timeout = int(composer_timeout)
        self._composer_llm_kwargs = dict(composer_llm_kwargs or {})
        cache_mode = str(composer_cache_mode or "off").lower()
        if cache_mode not in {"off", "read", "write"}:
            raise ValueError("composer_cache_mode must be off, read, or write")
        if cache_mode != "off" and not composer_cache_dir:
            raise ValueError(
                "composer_cache_dir is required when composer_cache_mode is enabled"
            )
        if composer_only and cache_mode != "write":
            raise ValueError("composer_only requires composer_cache_mode=write")
        self._composer_cache_dir = (
            Path(composer_cache_dir) if composer_cache_dir else None
        )
        self._composer_cache_mode = cache_mode
        self._composer_only = bool(composer_only)
        self._locked_module_type = locked_module_type or None
        self._composer_scope = composer_scope or "locked"

    @staticmethod
    def name() -> str:
        return "terminus-2-modular"

    def version(self) -> str:
        return "0.4.0"

    async def setup(self, environment: BaseEnvironment) -> None:
        return None

    def _boot_params(self) -> dict[str, Any]:
        """The primitives dict handed across the shim/orchestration boundary.

        Keys here are part of the FROZEN contract (see
        kernel/orchestration.py's header): evolution may read them and may
        tolerate their absence, but cannot make the frozen shim pack more.
        """
        return {
            "agent_name": self.name(),
            "agent_version": self.version(),
            "model_name": self.model_name,
            "session_id": self._session_id,
            "max_turns": self._max_turns,
            "parser_name": self._parser_name,
            "tmux_pane_width": self._tmux_pane_width,
            "tmux_pane_height": self._tmux_pane_height,
            "record_terminal_session": self._record_terminal_session,
            "extra_env": self._extra_env,
            "enable_summarize": self._enable_summarize,
            "proactive_summarization_threshold": (
                self._proactive_summarization_threshold
            ),
            "temperature": self._temperature,
            "reasoning_effort": self._reasoning_effort,
            "collect_rollout_details": self._collect_rollout_details,
            "max_thinking_tokens": self._max_thinking_tokens,
            "model_info": self._model_info,
            "use_responses_api": self._use_responses_api,
            "interleaved_thinking": self._interleaved_thinking,
            "llm_kwargs": self._llm_kwargs,
            "llm_call_kwargs": self._llm_call_kwargs,
            "api_base": self._api_base,
            "api_key": self._api_key,
            "store_all_messages": self._store_all_messages,
            "trajectory_config": self._trajectory_config,
            "skills_dir": self.skills_dir,
            "mcp_servers": self.mcp_servers,
            "staging_dir": self._staging_dir,
            "modules_root": self._modules_root,
            "trajectory_root": self._trajectory_root,
            "archive_path": self._archive_path,
            "composer_name": self._composer_name,
            "composer_timeout": self._composer_timeout,
            "composer_llm_kwargs": self._composer_llm_kwargs,
            "composer_cache_dir": self._composer_cache_dir,
            "composer_cache_mode": self._composer_cache_mode,
            "composer_only": self._composer_only,
            "locked_module_type": self._locked_module_type,
            "composer_scope": self._composer_scope,
        }

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        from harbor.agents.terminus_2_modular.kernel import orchestration as orch

        await orch.run_task(
            params=self._boot_params(),
            instruction=instruction,
            environment=environment,
            context=context,
            logs_dir=Path(self.logs_dir),
            logger=self.logger,
        )


class Terminus2ModularEditor(Terminus2Modular):
    """Editor variant — same code, different default Composer + tools.

    Use case: self-evo loop. This agent reads a parent generation's
    `modules/` directory (copied to `staging_dir`), proposes edits via
    `<edit_file>` / `<create_file>` actions, and signals `<commit_patch/>`
    when done. The outer evo driver then runs smoke tests on staging and
    promotes it to a new generation if they pass.

    The editor always runs the installed evaluator kernel.
    """

    @staticmethod
    def name() -> str:
        return "terminus-2-modular-editor"

    def version(self) -> str:
        return "0.1.0"

    def __init__(self, *args, **kwargs):
        # Force editor mode unless caller explicitly overrides
        kwargs.setdefault("composer_name", "editor_static")
        # Editor sessions are short; no summarization / no terminal recording
        kwargs.setdefault("enable_summarize", False)
        kwargs.setdefault("record_terminal_session", False)
        super().__init__(*args, **kwargs)
