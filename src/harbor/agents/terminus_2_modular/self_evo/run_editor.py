"""Run the local editor against a staged modules directory."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import weakref

from harbor.agents.terminus_2_modular import Terminus2ModularEditor
from harbor.models.agent.context import AgentContext

# ---------------------------------------------------------------------------
# Every investigator, implementer, reviewer, and repair session shares one
# admission limit. This bounds editor endpoint load independently from solver
# concurrency; queued reflection work is visible through the stale-parent
# metric recorded for each reflection.
# ---------------------------------------------------------------------------

_EDITOR_CONCURRENCY_ENV = "HARBOR_EDITOR_CONCURRENCY"
_DEFAULT_EDITOR_CONCURRENCY = 2
# One semaphore per event loop: pytest (and any embedder) runs each coroutine
# in a fresh loop, and an asyncio primitive must not be shared across loops.
_slots_by_loop: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


def _editor_concurrency_limit() -> int:
    raw = os.environ.get(_EDITOR_CONCURRENCY_ENV, "")
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return _DEFAULT_EDITOR_CONCURRENCY


def _editor_slot() -> asyncio.Semaphore:
    loop = asyncio.get_running_loop()
    limit = _editor_concurrency_limit()
    sem = _slots_by_loop.get(loop)
    if sem is None or getattr(sem, "_admission_limit", None) != limit:
        sem = asyncio.Semaphore(limit)
        sem._admission_limit = limit  # type: ignore[attr-defined]
        _slots_by_loop[loop] = sem
    return sem


@dataclass
class EditorRunOutcome:
    success: bool  # True iff: no exception AND editor emitted commit_patch
    committed: bool  # editor explicitly emitted <commit_patch/>?
    error: str | None
    logs_dir: Path
    trajectory_path: Path | None
    staging_dir: Path
    final_metrics: dict
    n_edits: int = 0  # how many edit_file / create_file actions happened


def resolve_model_info(model_info: dict | None) -> dict | None:
    """Use the lineage-wide model profile for every direct editor LLM.

    Solver and sanity subprocesses receive the same profile through
    ``task_runner.run_harbor_task``. Editors bypass Harbor and construct the
    agent directly, so they must resolve the shared out-of-band value here.
    """
    if model_info is not None:
        return model_info
    raw = os.environ.get("HARBOR_MODEL_INFO", "").strip()
    if not raw:
        return None
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("HARBOR_MODEL_INFO must be a JSON object")
    for key in ("max_input_tokens", "max_output_tokens"):
        if not isinstance(parsed.get(key), int) or parsed[key] <= 0:
            raise ValueError(f"HARBOR_MODEL_INFO.{key} must be a positive integer")
    return parsed


def _trajectory_commit_and_edit_counts(traj_path: Path) -> tuple[bool, int]:
    """Scan trajectory.json for editor actions.

    Returns (committed, n_edits) where:
      committed: did the editor emit a <commit_patch/> action?
      n_edits:   how many edit_file / edit_lines / create_file actions happened?
    """
    import json

    if traj_path is None or not traj_path.exists():
        return False, 0
    try:
        data = json.loads(traj_path.read_text())
    except Exception:
        return False, 0
    committed = False
    n_edits = 0
    for step in data.get("steps", []):
        for tc in step.get("tool_calls") or []:
            args = tc.get("arguments") or {}
            # The editor encodes its action in arguments.keystrokes (JSON
            # payload). action ∈ {read_file, grep, edit_file, edit_lines,
            # create_file, validate, commit_patch}.
            ks = args.get("keystrokes") or ""
            if not isinstance(ks, str):
                continue
            if '"action": "commit_patch"' in ks:
                committed = True
            elif (
                '"action": "edit_file"' in ks
                or '"action": "edit_lines"' in ks
                or '"action": "create_file"' in ks
            ):
                n_edits += 1
    return committed, n_edits


async def run_editor(
    *,
    staging_dir: Path,
    instruction: str,
    model_name: str,
    api_base: str | None = None,
    api_key: str | None = None,
    skills_dir: Path | None = None,
    logs_dir: Path | None = None,
    max_turns: int = 30,
    model_info: dict | None = None,
    temperature: float | None = None,
    interleaved_thinking: bool = False,
    llm_call_kwargs: dict | None = None,
    trajectory_root: Path | None = None,
    archive_path: Path | None = None,
    locked_module_type: str | None = None,
) -> EditorRunOutcome:
    """Run the editor locally against a staged modules directory.

    trajectory_root (optional): a read-only path the editor's file tools may
    additionally read — the batch's solver output dir — so the editor can
    investigate the raw run (trajectories, episode prompt/response, terminal
    panes, result.json) itself instead of only a pre-digested summary.

    """

    if not staging_dir.exists():
        raise FileNotFoundError(f"staging_dir does not exist: {staging_dir}")

    logs_dir = logs_dir or Path("editor_logs") / "run"
    logs_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("terminus-2-modular-editor")

    resolved_model_info = resolve_model_info(model_info)

    agent = Terminus2ModularEditor(
        logs_dir=logs_dir,
        model_name=model_name,
        api_base=api_base,
        api_key=api_key,
        skills_dir=str(skills_dir) if skills_dir else None,
        staging_dir=str(staging_dir),
        trajectory_root=str(trajectory_root) if trajectory_root else None,
        archive_path=str(archive_path) if archive_path else None,
        locked_module_type=locked_module_type,
        max_turns=max_turns,
        model_info=resolved_model_info,
        temperature=temperature,
        interleaved_thinking=interleaved_thinking,
        llm_call_kwargs=llm_call_kwargs,
        # Editor doesn't need terminal recording / summarization (subclass
        # already sets these, but be explicit)
        record_terminal_session=False,
        enable_summarize=False,
        suppress_max_turns_warning=True,
        logger=logger,
    )

    # NB: We do NOT call agent.setup() — that's a no-op anyway, but the
    # `environment` arg is None here so it's irrelevant.

    context = AgentContext()
    error_msg: str | None = None
    ran_clean = False
    sem = _editor_slot()
    if sem.locked():
        logger.debug(
            "editor admission: waiting for a slot (limit=%d)",
            _editor_concurrency_limit(),
        )
    try:
        async with sem:
            await agent.run(
                instruction=instruction,
                environment=None,  # editor doesn't talk to Docker
                context=context,
            )
        ran_clean = True
    except Exception as exc:
        import traceback

        error_msg = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        logger.error("editor run failed: %s", error_msg)

    traj_path = logs_dir / "trajectory.json"
    if not traj_path.exists():
        traj_path_opt: Path | None = None
    else:
        traj_path_opt = traj_path

    # Critical: a clean run (no exception) where the editor never emitted
    # <commit_patch/> means the editor used all its turns without finalizing.
    # That should NOT count as success — driver would otherwise promote an
    # unintended no-op or partial patch.
    committed, n_edits = _trajectory_commit_and_edit_counts(traj_path_opt)
    success = ran_clean and committed
    if ran_clean and not committed:
        error_msg = (
            "editor ran to completion but never emitted <commit_patch/>; "
            "treating as failed editor session"
        )

    return EditorRunOutcome(
        success=success,
        committed=committed,
        error=error_msg,
        logs_dir=logs_dir,
        trajectory_path=traj_path_opt,
        staging_dir=staging_dir,
        final_metrics={
            "n_input_tokens": context.n_input_tokens,
            "n_output_tokens": context.n_output_tokens,
            "n_cache_tokens": context.n_cache_tokens,
            "cost_usd": context.cost_usd,
            "metadata": context.metadata,
        },
        n_edits=n_edits,
    )
