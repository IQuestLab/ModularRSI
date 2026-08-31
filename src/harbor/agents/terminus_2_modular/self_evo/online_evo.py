"""Execution engine for the public Phase0 two-lane workflow.

The engine runs same-bundle K-roll task groups, accumulates contrastive evidence,
implements portfolio-selected proposals in isolated lanes, and promotes only
candidates that pass review, smoke, activation, and crash-sanity gates. Use the
``self_evo.phase0`` module as the public CLI.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import json
import logging
import os
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from harbor.agents.terminus_2_modular.self_evo.task_runner import (
    HarborTaskResult,
    DEFAULT_MODEL_INFO,
    run_harbor_task,
)
from harbor.agents.terminus_2_modular.self_evo.k_roll import (
    KRollGroupResult,
    run_same_bundle_k_rolls,
)
from harbor.agents.terminus_2_modular.self_evo.run_editor import run_editor
from harbor.agents.terminus_2_modular.self_evo.smoke_tests import run_fast_smoke
from harbor.agents.terminus_2_modular.self_evo.staging import (
    discard_staging,
    initialize_gen_0,
    next_gen_number,
    prepare_staging,
)
from harbor.agents.terminus_2_modular import archive as _archive
from harbor.agents.terminus_2_modular.self_evo.trajectory_analysis import (
    TrialSummary,
    changed_module_files,
    extract_editor_intent,
    extract_variant_meta_blocks,
    append_evolution_log,
    build_contrast_investigation_instruction,
    build_diff_text,
    build_efficiency_investigation_instruction,
    build_review_instruction,
    build_sanity_repair_instruction,
    build_smoke_repair_instruction,
    parse_contrast_finding,
    summarize_trial,
)
from harbor.agents.terminus_2_modular.self_evo import (
    action_gate as _action_gate,
)
from harbor.agents.terminus_2_modular.self_evo import (
    backlog as _backlog,
)
from harbor.agents.terminus_2_modular.self_evo import (
    manifest as _manifest,
)
from harbor.agents.terminus_2_modular.self_evo import (
    proposals as _proposals,
)
from harbor.agents.terminus_2_modular.self_evo import (
    evidence_pass as _evidence,
)
from harbor.agents.terminus_2_modular.self_evo import (
    clustering as _clustering,
)
from harbor.agents.terminus_2_modular.self_evo import (
    routing_probe as _routing_probe,
)
from harbor.agents.terminus_2_modular.self_evo import (
    dual_implement as _dual_implement,
)
from harbor.agents.terminus_2_modular.self_evo import (
    two_lane as _two_lane,
)
from harbor.agents.terminus_2_modular.self_evo import editor_memory, review_verdict
from harbor.agents.terminus_2_modular.self_evo.router import (
    Ledger,
    RouterConfig,
    route_batch,
)
from harbor.agents.terminus_2_modular.self_evo.state import (
    OnlineEvoOutcome,
    Promotion as _Promotion,
    ReflectionOutcome as _ReflectionOutcome,
    TaskRunRecord,
)


_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sanity gate config
# ---------------------------------------------------------------------------

# Fixed, fast, reliable tasks used by the runtime sanity gate. These only check
# that the edited modules still RUN without crashing — we do NOT compare reward
# (that would be info leakage / overfitting to the reflected batch). Picked from
# results_backup610/gen_eval: each completes fast (full <200s), has a short
# docker build, exercises 7+ episodes (so a broken module surfaces), reliably
# produced a clean result across runs (0 timeouts/errors), and the four span
# different task types (git / pip-deps / reasoning / text) so different module
# paths get exercised. Override with --sanity-tasks.
DEFAULT_SANITY_TASKS = [
    "fix-git",
    "modernize-scientific-stack",
    "prove-plus-comm",
    "log-summary-date-ranges",
]


def _gen_run_kwargs(root: Path) -> dict:
    """Return the modules directory used to run a generation or staging tree."""
    root = Path(root)
    if (root / "modules").is_dir():
        return {"staging_modules_dir": root / "modules"}
    return {"staging_modules_dir": root}


def _prepare_candidate_staging(current_gen: Path, staging_root: Path) -> Path:
    """Copy the current generation's modules into a fresh staging tree."""
    return prepare_staging(current_gen / "modules", staging_root, fresh=True)


@dataclass
class _E2BSlot:
    """One in-flight sandbox slot on a specific e2b account. key=None means
    'no override' → the harbor subprocess inherits the single global key."""

    key: str | None
    token: str | None


class AccountPool:
    """Spreads e2b sandbox creation across multiple accounts, each capped at
    `per_account_cap` concurrent sandboxes (e2b hard-caps 20/account; we keep
    headroom). A slot is a FIFO token carrying one account's (key, token);
    `slot()` blocks when every account is at its cap — correct back-pressure,
    strictly better than letting harbor hit a 429. FIFO get/put load-balances
    across accounts and enforces the per-account cap (exactly `cap` tokens per
    account exist), so it auto-adapts as task/probe concurrency grow.
    """

    def __init__(self, accounts: list[tuple[str, str]], per_account_cap: int = 18):
        self.per_account_cap = max(1, per_account_cap)
        self.n_accounts = len(accounts)
        self.capacity = self.n_accounts * self.per_account_cap
        self._q: asyncio.Queue[_E2BSlot] = asyncio.Queue()
        # Interleave tokens round-robin ([a,b,c,d, a,b,c,d, …]) rather than
        # account-blocked ([a×cap, b×cap, …]). Both hold exactly `cap` tokens
        # per account, but blocked ordering hands the first `cap` acquisitions
        # all to account a, so at concurrency < cap the whole run piles on one
        # account (b/c/d idle) — 4 accounts stop spreading load/credit and a
        # probe burst pushes a to the 20-cap edge. Interleaving fills a,b,c,d
        # in turn, so load balances even when concurrency never reaches `cap`.
        for _ in range(self.per_account_cap):
            for key, token in accounts:
                self._q.put_nowait(_E2BSlot(key=key, token=token))

    @contextlib.asynccontextmanager
    async def slot(self):
        s = await self._q.get()
        try:
            yield s
        finally:
            self._q.put_nowait(s)


class NullAccountPool:
    """No-op pool for single-account / docker runs: yields a keyless slot so
    ``run_harbor_task`` inherits the one global E2B_API_KEY."""

    n_accounts = 0
    per_account_cap = 0
    capacity = 0

    @contextlib.asynccontextmanager
    async def slot(self):
        yield _E2BSlot(key=None, token=None)


async def _run_same_bundle_battery(
    *,
    source_root: Path,
    run_root: Path,
    tasks: list[str],
    k_repeats: int,
    model_name: str,
    api_base: str | None,
    api_key: str | None,
    max_solver_turns: int,
    timeout_sec: int | None,
    environment: str | None,
    concurrency: int,
    e2b_pool=None,
) -> list[KRollGroupResult]:
    """Run task×K probes with one frozen bundle per task and tree."""

    sem = asyncio.Semaphore(max(1, concurrency))

    async def _one(task_idx: int, task: str):
        group_root = run_root / f"{task_idx:02d}_{task}"

        async def _run_roll(
            roll_idx: int,
            load_root: Path,
            composer_name: str,
            output_dir: Path,
        ) -> HarborTaskResult:
            kwargs = dict(
                task=task,
                **_gen_run_kwargs(load_root),
                model_name=model_name,
                api_base=api_base,
                api_key=api_key,
                output_dir=output_dir,
                max_turns=max_solver_turns,
                timeout_sec=timeout_sec,
                environment=environment,
                composer_scope="all",
                composer_name=composer_name,
            )
            if e2b_pool is None:
                return await asyncio.to_thread(run_harbor_task, **kwargs)
            async with e2b_pool.slot() as slot:
                return await asyncio.to_thread(
                    run_harbor_task,
                    **kwargs,
                    e2b_key=slot.key,
                    e2b_token=slot.token,
                )

        async with sem:
            return await run_same_bundle_k_rolls(
                attempts=k_repeats,
                source_root=source_root,
                group_root=group_root,
                run_roll=_run_roll,
            )

    return list(await asyncio.gather(*(_one(i, task) for i, task in enumerate(tasks))))


_K_ROLL_BUNDLE_PROTOCOL = 1


# Exception types that mean "infrastructure / endpoint / timeout", NOT a code
# break. A sanity task ending with one of these does NOT fail the gate.
_INFRA_OK_EXC = frozenset(
    {
        "AgentTimeoutError",
        "VerifierTimeoutError",
        "ContextLengthExceededError",
        "RewardFileNotFoundError",
        "RewardFileEmptyError",
        "VerifierOutputParseError",
        # e2b / sandbox infrastructure failures — NOT a module-code crash. A
        # cold-start build race (K identical templates building at once) or a
        # concurrency-cap 429 leaves 0 episodes but says nothing about the
        # staged code; treating it as a crash triggers a bogus gate-repair loop.
        "BuildException",
        "RateLimitException",
        "SandboxException",
        "TimeoutException",
    }
)

# Substrings in harbor's error/stderr that mean an e2b/sandbox infra failure
# even when NO trial summary was produced (build failed before any result.json).
_INFRA_ERR_MARKERS = (
    "BuildException",
    "build was cancelled",
    "not in waiting state",
    "RateLimit",
    "concurrent E2B sandboxes",
    "SandboxException",
)


def _is_code_break(
    hr: HarborTaskResult, summary: TrialSummary | None
) -> tuple[bool, str]:
    """Did this sanity run indicate the staged modules are BROKEN (crashed)?

    Crash-only, deliberately lenient — we discard a candidate only on a clear
    code-level failure, never on a task-failure (reward=0) or an infra hiccup
    (endpoint timeout). reward is the strongest "it ran" signal: if the verifier
    produced a reward at all, the agent loop ran end-to-end on the staged code.

    Returns (is_break, reason).
    """
    # Got a real reward (0 or 1) → agent ran AND verifier scored it → not a break.
    if hr.reward is not None:
        return False, ""

    # reward is None — figure out whether it's a code crash or just infra.
    if hr.error == "timeout":
        # Outer subprocess wall-clock cap (infra), not a code break.
        return False, f"{hr.task}: outer timeout (infra)"
    if summary is None:
        # No trial result produced. If harbor's error names an e2b/sandbox infra
        # failure (build race, 429), that's NOT a module crash — don't discard.
        if hr.error and any(m in hr.error for m in _INFRA_ERR_MARKERS):
            return False, f"{hr.task}: infra failure before trial ({hr.error[:80]})"
        # Otherwise harbor/agent couldn't even run — treat as a crash.
        return True, f"{hr.task}: no trial result ({hr.error or 'unknown'})"
    if summary.exception_type in _INFRA_OK_EXC:
        return False, f"{hr.task}: {summary.exception_type} (infra)"
    if (summary.n_episodes or 0) == 0:
        return True, f"{hr.task}: agent ran 0 episodes (module setup likely crashed)"
    if summary.exception_type:
        return True, f"{hr.task}: code exception {summary.exception_type}"
    # reward None, no exception, episodes>0, not a timeout → ambiguous infra;
    # be lenient (don't discard a candidate for an unexplained non-crash).
    return False, f"{hr.task}: reward None but ran {summary.n_episodes} eps (lenient)"


def _sanity_crash_detail(
    trial_dir: Path, task: str, exception_type: str | None
) -> dict:
    """Pull what a repair editor needs from a crashed sanity trial: the traceback
    (exception.txt) and which variant was active (trajectory agent.extra.bundle)."""
    tb = ""
    exc = trial_dir / "exception.txt"
    if exc.exists():
        try:
            tb = exc.read_text()[-3000:]
        except Exception:
            pass
    bundle = None
    tj = trial_dir / "agent" / "trajectory.json"
    if tj.exists():
        try:
            extra = (json.loads(tj.read_text()).get("agent") or {}).get("extra") or {}
            b = extra.get("bundle")
            bundle = b if isinstance(b, dict) else None
        except Exception:
            pass
    return {
        "task": task,
        "exception_type": exception_type,
        "traceback": tb,
        "bundle": bundle,
        "trial_dir": str(trial_dir),
    }


def _collect_sanity_activation(sanity_root: Path) -> dict | None:
    """Aggregate composer selections + kernel module-trace call counts across
    the sanity/probe battery's trials.

    The collector itself is policy-free. Callers use selection plus at least
    one traced call as a hard *activation* check for a privately pinned changed
    variant; call frequency for all other variants remains diagnostic evidence.
    Returns None when no trajectory carried a bundle (old code or total crash)
    so the log shows null, not an empty claim.
    """
    variants: dict[str, dict[str, int]] = {}
    n_trials = 0
    try:
        traj_paths = sorted(Path(sanity_root).rglob("agent/trajectory.json"))
    except Exception:
        return None
    for tj in traj_paths:
        try:
            extra = (json.loads(tj.read_text()).get("agent") or {}).get("extra") or {}
        except Exception:
            continue
        bundle = extra.get("bundle")
        if not isinstance(bundle, dict):
            continue
        n_trials += 1
        for mtype, name in bundle.items():
            names = (
                name if mtype == "tool_helper" and isinstance(name, list) else [name]
            )
            for selected_name in names:
                if not isinstance(selected_name, str):
                    continue
                key = f"{mtype}:{selected_name}"
                variants.setdefault(key, {"selected": 0, "trace_calls": 0})
                variants[key]["selected"] += 1
        trace = extra.get("module_trace")
        if isinstance(trace, list):
            for ev in trace:
                if isinstance(ev, dict) and isinstance(ev.get("module"), str):
                    variants.setdefault(ev["module"], {"selected": 0, "trace_calls": 0})
                    variants[ev["module"]]["trace_calls"] += 1
    if not n_trials:
        return None
    return {"n_trials": n_trials, "variants": variants}


_SANITY_PINNABLE_TYPES = frozenset(
    {"agent_loop", "observation", "context_mgmt", "tools", "verification"}
)


def _module_relpath(changed: str) -> str:
    """Normalize a changed-file entry to its modules-root-relative path."""

    rel = changed[4:] if changed.startswith("NEW:") else changed
    if rel.startswith("modules/"):
        rel = rel[len("modules/") :]
    return rel


def _registered_names_of_file(mod_file: Path, locked: str) -> list[str]:
    """Names the changed file ITSELF registers for `locked` — loaded standalone
    into a throwaway library, the same spec_from_file_location mechanism
    build_default_library uses per file. Smoke already proved the file loads;
    a load/register failure here is still a clean gate rejection."""
    import importlib.util
    import uuid

    from harbor.agents.terminus_2_modular.library import ModuleLibrary

    synthetic = f"_evo_pin_probe_{mod_file.stem}_{uuid.uuid4().hex[:8]}"
    spec = importlib.util.spec_from_file_location(synthetic, mod_file)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load changed module file {mod_file.name}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[synthetic] = mod
    try:
        try:
            spec.loader.exec_module(mod)
        except Exception as exc:
            raise ValueError(
                f"changed module file {mod_file.name} failed to load: {exc}"
            ) from exc
        register_fn = getattr(mod, "register", None)
        if not callable(register_fn):
            raise ValueError(f"changed module file {mod_file.name} has no register()")
        probe = ModuleLibrary()
        try:
            register_fn(probe)
        except Exception as exc:
            raise ValueError(
                f"register() of changed file {mod_file.name} failed: {exc}"
            ) from exc
        return sorted(i.name for i in probe.list_infos() if i.type == locked)
    finally:
        sys.modules.pop(synthetic, None)


def _required_sanity_variant(
    *,
    archive_root: Path,
    staging_root: Path,
    files_changed: list[str],
    variant_meta_text: str,
    locked_module_type: str | None,
) -> tuple[str, str] | None:
    """Resolve the single changed locked variant that sanity must execute.

    Contrastive/serial evolution is deliberately one-module-at-a-time. The
    CODE is the source of truth: the changed file is loaded standalone and the
    name it actually register()s is the pin target. variant_meta is only a
    cross-check — a declaration that names a DIFFERENT (but registered)
    variant used to redirect the pin, so the gate would validate e.g.
    `baseline` while the changed file never ran. Mismatch / zero / multiple
    registrations are rejected instead of silently pinning the wrong thing.
    """

    locked = (locked_module_type or "").strip()
    if not locked:
        return None
    changed_locked = [
        rel
        for rel in (_module_relpath(f) for f in files_changed)
        if rel.startswith(f"{locked}/") and rel.endswith(".py")
    ]
    if not changed_locked:
        return None
    if len(changed_locked) != 1:
        raise ValueError(
            f"forced sanity requires one changed {locked} module file; "
            f"got {changed_locked}"
        )
    if locked not in _SANITY_PINNABLE_TYPES:
        raise ValueError(
            f"changed locked type {locked!r} cannot be pinned by ModuleBundle"
        )

    from harbor.agents.terminus_2_modular.library import build_default_library

    modules_root = (
        staging_root / "modules"
        if (staging_root / "modules").is_dir()
        else staging_root
    )
    library = build_default_library(modules_root=modules_root)

    # The code is the source of truth: what does the changed file ITSELF
    # register under the locked type?
    code_names = _registered_names_of_file(modules_root / changed_locked[0], locked)
    if len(code_names) != 1:
        raise ValueError(
            f"changed file {changed_locked[0]} must register exactly one "
            f"{locked} variant for forced sanity; it registers: "
            f"{', '.join(code_names) or 'none'}"
        )
    name = code_names[0]

    metas = _archive.parse_variant_meta(variant_meta_text or "")
    declared = [
        str(meta.get("name") or "").strip()
        for meta in metas
        if str(meta.get("type") or "").strip() == locked
        and str(meta.get("name") or "").strip()
    ]
    if declared and declared[-1] != name:
        raise ValueError(
            f"variant_meta declares {locked}:{declared[-1]} but the changed "
            f"file {changed_locked[0]} actually registers {locked}:{name}; "
            "forced sanity refuses to pin a variant the edit did not produce "
            "— fix the <variant_meta> name"
        )
    if not library.has(locked, name):
        raise ValueError(
            f"forced sanity target {locked}:{name} is not registered in staging"
        )

    existing = _archive.load_archive(archive_root)
    retired = {
        (entry.type, entry.name): entry.status
        for entry in existing
        if entry.status in ("superseded", "excluded")
    }
    if (locked, name) in retired:
        raise ValueError(
            f"forced sanity target {locked}:{name} is {retired[(locked, name)]}; "
            "the real Composer cannot deploy this edit"
        )
    qual = f"{locked}/{name}"
    if qual in _archive.supersede_targets(metas, existing):
        raise ValueError(
            f"forced sanity target {locked}:{name} would be superseded by its "
            "own variant_meta at promotion; the Composer could not deploy it"
        )
    return locked, name


def _prepare_forced_sanity_source(
    source_root: Path,
    destination: Path,
    target: tuple[str, str],
) -> Path:
    """Copy staging privately and pin target without mutating the candidate."""

    source_root = Path(source_root)
    destination = Path(destination)
    shutil.copytree(source_root, destination)
    modules_root = (
        destination / "modules" if (destination / "modules").is_dir() else destination
    )
    pin_path = modules_root / "active_bundle.json"
    raw: dict[str, Any] = {}
    if pin_path.is_file():
        try:
            loaded = json.loads(pin_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"cannot extend existing sanity bundle pin: {exc}"
            ) from exc
        if not isinstance(loaded, dict):
            raise ValueError("existing sanity bundle pin is not a JSON object")
        raw = loaded

    module_type, name = target
    old_spec = raw.get(module_type)
    spec: dict[str, Any] = {"name": name}
    if (
        isinstance(old_spec, dict)
        and old_spec.get("name") == name
        and isinstance(old_spec.get("params"), dict)
    ):
        spec["params"] = old_spec["params"]
    raw[module_type] = spec
    pin_path.write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n")
    return destination


def _required_activation_failure(
    activation: dict | None, target: tuple[str, str]
) -> str | None:
    """Return why a forced target was not actually selected and executed."""

    key = f"{target[0]}:{target[1]}"
    if not activation:
        return f"required {key}, but no sanity trajectory recorded a bundle"
    evidence = (activation.get("variants") or {}).get(key)
    if not isinstance(evidence, dict) or int(evidence.get("selected") or 0) < 1:
        return f"required {key}, but sanity never selected it"
    if int(evidence.get("trace_calls") or 0) < 1:
        return f"required {key}, but its module trace recorded zero calls"
    return None


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


def _restore_unreflected_progress(
    prior_progress: list[dict],
) -> tuple[list[TrialSummary], int, int | None]:
    """Rebuild the rolling reflection buffer after a process restart.

    ``progress.json`` is written while a background reflection is running, so
    it can contain completed solver groups *after* the latest record carrying a
    reflection outcome.  Those suffix groups were completed but never digested.
    Merely skipping every recorded ``task_idx`` on resume silently loses their
    findings.  Rehydrate their summaries and task cadence so they enter the next
    reflection without rerunning the solver work.

    Returns ``(pending_summaries, valid_task_count, suffix_anchor_index)``.  The
    anchor is the last prior-progress entry in the suffix; callers use it when a
    restored reflection finishes before any new task record exists, ensuring a
    later resume sees the new reflection boundary too.
    """

    last_reflection_idx = -1
    for i, entry in enumerate(prior_progress):
        if entry.get("reflection") is not None:
            last_reflection_idx = i
    suffix = prior_progress[last_reflection_idx + 1 :]
    if not suffix:
        return [], 0, None

    entries_by_task: dict[int, list[dict]] = {}
    for entry in suffix:
        try:
            task_idx = int(entry["task_idx"])
        except (KeyError, TypeError, ValueError):
            continue
        entries_by_task.setdefault(task_idx, []).append(entry)

    valid_tasks = {
        task_idx
        for task_idx, entries in entries_by_task.items()
        if all(not entry.get("k_group_invalid_reason") for entry in entries)
    }
    restored: list[TrialSummary] = []
    for entry in suffix:
        try:
            task_idx = int(entry["task_idx"])
        except (KeyError, TypeError, ValueError):
            continue
        if task_idx not in valid_tasks:
            continue
        raw_summary = entry.get("summary")
        if isinstance(raw_summary, dict):
            try:
                restored.append(TrialSummary(**raw_summary))
            except TypeError as exc:
                _logger.warning(
                    "resume: could not restore summary for task_idx=%s roll=%s: %s",
                    task_idx,
                    entry.get("roll"),
                    exc,
                )
        elif entry.get("error"):
            restored.append(
                TrialSummary(
                    task_name=str(entry.get("task_name") or f"task_{task_idx}"),
                    trial_name=f"failed_{task_idx}_r{int(entry.get('roll') or 0)}",
                    reward=None,
                    exception_type=None,
                    exception_message=str(entry["error"]),
                    n_episodes=0,
                    n_input_tokens=0,
                    n_output_tokens=0,
                    last_step_messages=[],
                    failure_signals=["harbor_error"],
                    trial_dir=(
                        str(entry["trial_dir"]) if entry.get("trial_dir") else None
                    ),
                )
            )

    return restored, len(valid_tasks), len(prior_progress) - 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_latest_gen(archive_root: Path) -> Path | None:
    """Return the path of the highest gen_N/ dir, or None."""
    latest = next_gen_number(archive_root) - 1
    if latest < 0:
        return None
    return archive_root / f"gen_{latest}"


def _seed_archive(archive_root: Path) -> None:
    """Seed <run>/archive.json from gen_0's modules (once, if absent). Best-effort."""
    if _archive.archive_path(archive_root).is_file():
        return
    gen0 = archive_root / "gen_0" / "modules"
    if not gen0.is_dir():
        return
    try:
        from harbor.agents.terminus_2_modular.library import build_default_library

        lib = build_default_library(modules_root=gen0)
        _archive.save_archive(
            archive_root, _archive.seed_from_library(lib, born_gen="gen_0")
        )
    except Exception as exc:  # noqa: BLE001
        _logger.warning("archive seed failed: %s", exc)


def _preflight_pin(
    archive_root: Path, gen_dir: Path, locked_module_type: str | None
) -> None:
    """Validate `<gen>/modules/active_bundle.json` — the SERIAL pin — and abort
    the run if it is broken.

    The pin is how a serial lineage stands on a previous lineage's winner (e.g.
    `agent_loop=planning_with_guard` while evolving `observation` on top). Its
    reader, `bundle_config.load_bundle_overrides`, is deliberately fail-open: a
    bad file logs one warning and degrades to `DEFAULT_BUNDLE`. That is the right
    behavior for a single roll and exactly the WRONG behavior for a run — every
    roll would silently use `baseline` and the lineage would quietly stop being
    serial. So check it once, loudly, before any task starts.

    Rejects: unparseable / names an impl not in the library (both surface as
    "no overrides"), pins the locked type (the experiment would vary nothing),
    or pins a variant the archive has retired (the composer's `_archive_skip`
    would drop it from the catalog while `base` still names it).
    """
    from harbor.agents.terminus_2_modular.composer.bundle_config import (
        BUNDLE_CONFIG_FILENAME,
        load_bundle_overrides,
    )
    from harbor.agents.terminus_2_modular.library import build_default_library

    modules = gen_dir / "modules"
    cfg = modules / BUNDLE_CONFIG_FILENAME
    if not cfg.is_file():
        return  # no pin: a normal (non-serial) lineage

    overrides = load_bundle_overrides(
        modules, build_default_library(modules_root=modules).list_infos()
    )
    if not overrides:
        raise SystemExit(
            f"[fatal] {cfg} exists but yielded no valid pin — unparseable, or it "
            f"names an implementation the library at {modules} does not have. "
            f"Re-run the reader's warning above for the exact reason. Refusing to "
            f"start: every roll would silently fall back to the baseline bundle."
        )

    if locked_module_type and locked_module_type in overrides:
        raise SystemExit(
            f"[fatal] {cfg} pins {locked_module_type!r}, which is also "
            f"--locked-module. The pin freezes it and the lock says it is the one "
            f"type to evolve — the run would vary nothing."
        )

    retired = {
        f"{e.type}/{e.name}"
        for e in _archive.load_archive(archive_root)
        if e.status in ("superseded", "excluded")
    }
    for t, spec in overrides.items():
        if f"{t}/{spec.name}" in retired:
            raise SystemExit(
                f"[fatal] {cfg} pins {t}/{spec.name}, which archive.json marks "
                f"retired. The composer would drop it from the catalog while still "
                f"naming it as the default — a self-contradictory foundation."
            )

    _logger.info(
        "SERIAL PIN (from %s): %s — these types are frozen for this lineage and "
        "never reach the per-task picker",
        cfg,
        ", ".join(f"{t}={s.name}" for t, s in sorted(overrides.items())),
    )


def _sync_archive_after_promote(
    archive_root: Path, promoted_gen: Path, parent_modules: Path, editor_text: str
) -> None:
    """Record the promoted gen's variants into archive.json (D3 write-path).

    New variants get genealogy from the editor's <variant_meta> blocks (D2, best
    effort); existing variants have their niche refreshed. Never blocks promotion.
    """
    try:
        from harbor.agents.terminus_2_modular.library import build_default_library

        new_lib = build_default_library(modules_root=promoted_gen / "modules")
        par_quals = {
            f"{i.type}/{i.name}"
            for i in build_default_library(modules_root=parent_modules).list_infos()
        }
        existing = {e.qual: e for e in _archive.load_archive(archive_root)}
        meta_list = _archive.parse_variant_meta(editor_text or "")
        metas: dict[str, dict] = {}
        for m in meta_list:
            metas[f"{m['type']}/{m['name']}"] = m
            metas.setdefault(m["name"], m)
        updates: list = []
        for info in new_lib.list_infos():
            qual = f"{info.type}/{info.name}"
            niche = dict(info.niche or {})
            if qual in existing:
                e = existing[qual]
                e.niche = niche
                if e.status != "superseded":
                    e.status = "active" if e.solver_selectable else "excluded"
                updates.append(e)
                continue
            meta = metas.get(qual) or metas.get(info.name) or {}
            change = meta.get("change_type") or (
                "add" if qual not in par_quals else "modify"
            )
            e = _archive.ArchiveEntry(
                name=info.name,
                type=info.type,
                niche=niche,
                parent_ids=meta.get("parent_ids", []),
                addresses=meta.get("addresses", ""),
                change_type=change,
                born_gen=promoted_gen.name,
            )
            e.status = "active" if e.solver_selectable else "excluded"
            updates.append(e)
        # Supersede: retire the incumbents a promoted variant subsumes — via an
        # explicit SUPERSEDES line OR as the parents of a CHANGE: merge (merging
        # A+B into C consolidates A,B into C). Seeds/baseline never retire, so the
        # composer keeps its DEFAULT_BUNDLE fallback. (composer/gate then skip them.)
        sup_quals = _archive.supersede_targets(meta_list, list(existing.values()))
        for e in existing.values():
            if e.qual in sup_quals and e.status != "superseded":
                e.status = "superseded"
                updates.append(e)
        _archive.update_archive(archive_root, updates)
    except Exception as exc:  # noqa: BLE001
        _logger.warning("archive sync failed: %s", exc)


def _action_reject(
    *,
    action: str,
    target_variant: str,
    files_changed: list[str],
    variant_meta_text: str,
    parent_modules: Path,
    staged_modules: Path,
) -> str | None:
    """C3 gate, with the two libraries read for it. Best-effort — any error
    passes, because a gate that rejects on its own infra trouble costs a
    generation for a reason that has nothing to do with the change."""
    try:
        from harbor.agents.terminus_2_modular.library import build_default_library

        def _quals(root: Path) -> set[str]:
            return {
                f"{i.type}/{i.name}"
                for i in build_default_library(modules_root=root).list_infos()
            }

        return _action_gate.check_action(
            action=action,
            target_variant=target_variant,
            files_changed=list(files_changed or []),
            variant_meta_text=variant_meta_text,
            parent_quals=_quals(parent_modules),
            staged_quals=_quals(staged_modules),
        )
    except Exception as exc:  # noqa: BLE001
        _logger.warning("action gate could not run (passing): %s", exc)
        return None


def _niche_dedup_reject(
    archive_root: Path, staging_modules: Path, parent_modules: Path, editor_text: str
) -> str | None:
    """S5 dedup gate: reject a NEW staged variant that lands in an occupied niche
    cell (same type+niche_key as an ACTIVE archive variant) without declaring it
    SUPERSEDES that incumbent. Returns a rejection reason, or None to pass.
    Best-effort — any error → pass (never blocks on infra trouble)."""
    try:
        from harbor.agents.terminus_2_modular.library import build_default_library

        parent_quals = {
            f"{i.type}/{i.name}"
            for i in build_default_library(modules_root=parent_modules).list_infos()
        }
        active_by_cell: dict[tuple, list[str]] = {}
        for e in _archive.load_archive(archive_root):
            if e.status == "active" and e.key:
                active_by_cell.setdefault((e.type, e.key), []).append(e.name)
        metas: dict[str, dict] = {}
        for m in _archive.parse_variant_meta(editor_text or ""):
            metas[f"{m['type']}/{m['name']}"] = m
            metas.setdefault(m["name"], m)
        rejects: list[str] = []
        for info in build_default_library(modules_root=staging_modules).list_infos():
            qual = f"{info.type}/{info.name}"
            if qual in parent_quals:
                continue  # not new (baseline / modify-in-place already owns its cell)
            key = _archive._niche.niche_key(info.type, info.niche)
            if not key:
                # Undeclared niche was an escape hatch (no cell → no dedup). A new
                # variant MUST name its cell so the gate can see near-duplicates.
                rejects.append(
                    f"{qual} declares no NICHE — every new variant must name its "
                    "cell (`NICHE = {axis: value}` next to DESCRIPTION)"
                )
                continue
            incumbents = [
                n for n in active_by_cell.get((info.type, key), []) if n != info.name
            ]
            if not incumbents:
                continue
            meta = metas.get(qual) or metas.get(info.name) or {}
            sup = {s.split("/")[-1] for s in meta.get("supersedes", [])}
            if any(inc in sup for inc in incumbents):
                continue  # legitimate supersede
            rejects.append(
                f"{qual} (niche {dict(info.niche)}) collides with active {incumbents}"
            )
        if rejects:
            return (
                "niche gate rejected the candidate. Every new variant must declare "
                "a NICHE naming an EMPTY cell — or MERGE/SUPERSEDES an incumbent — "
                "not spawn a near-duplicate:\n  " + "\n  ".join(rejects)
            )
        return None
    except Exception as exc:  # noqa: BLE001
        _logger.warning("niche dedup check failed: %s", exc)
        return None


def _harbor_results_to_summary(
    output_dir: Path,
) -> TrialSummary | None:
    """After a harbor run, find the trial dir and summarize it."""
    if not output_dir.is_dir():
        return None
    job_dirs = sorted(
        [d for d in output_dir.iterdir() if d.is_dir()],
        key=lambda p: p.stat().st_mtime,
    )
    if not job_dirs:
        return None
    job_dir = job_dirs[-1]
    trial_dirs = sorted([d for d in job_dir.iterdir() if d.is_dir() and "__" in d.name])
    if not trial_dirs:
        return None
    return summarize_trial(trial_dirs[0])


# ---------------------------------------------------------------------------
# Edit self-review (review gate)
# ---------------------------------------------------------------------------


@dataclass
class _ReviewResult:
    passed: bool
    verdict: str | None  # "accept" / "reject" / "skipped" / "blocked"
    reason: str
    # Files the reviewer flagged to revert on a PARTIAL accept (honored only when
    # passed=True): the good files in the bundle promote, these are dropped.
    drop_files: list[str] = field(default_factory=list)
    # Whether the direction was wrong or its implementation was flawed.
    reject_class: str = "none"
    repair_brief: str = ""


def _apply_review_drops(
    staging_modules: Path, parent_modules: Path, drop_files: list[str]
) -> tuple[list[str], list[str]]:
    """Revert the reviewer-flagged files in `staging_modules` to their parent
    state for a PARTIAL accept: a NEW file (absent from parent) is deleted, a
    MODIFIED file is restored to the parent's bytes. Path-traversal safe — a drop
    path resolving outside the modules root is skipped. Returns (applied,
    skipped)."""
    applied: list[str] = []
    skipped: list[str] = []
    sroot = staging_modules.resolve()
    for rel in drop_files:
        rel = rel.strip().lstrip("/")
        if not rel:
            continue
        sp = (staging_modules / rel).resolve()
        try:
            sp.relative_to(sroot)
        except ValueError:
            skipped.append(rel)
            continue
        if not sp.is_file():
            skipped.append(rel)
            continue
        pp = parent_modules / rel
        try:
            if pp.is_file():
                sp.write_bytes(pp.read_bytes())  # MODIFIED → restore parent
            else:
                sp.unlink()  # NEW → delete
            applied.append(rel)
        except OSError:
            skipped.append(rel)
    return applied, skipped


async def review_edit(
    *,
    archive_root: Path,
    parent_modules: Path,
    staging_modules: Path,
    pending_summaries: list[TrialSummary],
    intent: str,
    files_changed: list[str],
    model_name: str,
    api_base: str | None,
    api_key: str | None,
    skills_dir: Path | None,
    step_id: str,
    review_max_turns: int,
    review_timeout_sec: int,
    review_call_timeout_sec: int,
    locked_module_type: str | None = None,
) -> _ReviewResult:
    """Editor self-review of a just-committed change.

    Runs a SECOND editor pass on a THROWAWAY COPY of the candidate staging (so
    any stray edits the reviewer makes can't corrupt the real candidate), with
    read access to the raw solver trajectories. The reviewer re-investigates the
    failures + the diff and emits ``VERDICT: ACCEPT|REJECT``. Pure reasoning — no
    task reward is read, so this does NOT leak the eval signal; it specifically
    targets overfit / ineffective edits.

    On timeout / editor error (e.g. a wedged endpoint) it is LENIENT — returns
    passed=True/"skipped" rather than killing a candidate for infra flakiness.
    """
    review_root = archive_root / "review" / step_id
    review_staging = prepare_staging(staging_modules, review_root, fresh=True)

    diff_text = build_diff_text(parent_modules, staging_modules, files_changed)
    instruction = build_review_instruction(diff_text, intent, pending_summaries)
    instr_dir = archive_root / "auto_instructions"
    instr_dir.mkdir(parents=True, exist_ok=True)
    (instr_dir / f"{step_id}_review.md").write_text(instruction)

    review_logs = archive_root / "editor_logs" / f"{step_id}_review"
    try:
        # run_editor return value unused: the verdict travels via the
        # staging file, not the outcome or trajectory
        await asyncio.wait_for(
            run_editor(
                staging_dir=review_staging,
                archive_path=archive_root,
                instruction=instruction,
                model_name=model_name,
                api_base=api_base,
                api_key=api_key,
                skills_dir=skills_dir,
                logs_dir=review_logs,
                max_turns=review_max_turns,
                trajectory_root=archive_root / "solver",
                llm_call_kwargs={"timeout": review_call_timeout_sec},
                locked_module_type=locked_module_type,
            ),
            timeout=review_timeout_sec,
        )
    except (asyncio.TimeoutError, TimeoutError):
        _logger.warning(
            "review gate timed out after %ds (endpoint likely saturated); "
            "NOT blocking promotion",
            review_timeout_sec,
        )
        discard_staging(review_root)
        return _ReviewResult(
            passed=True, verdict="skipped", reason="review timed out (infra)"
        )
    except Exception as exc:
        _logger.warning("review gate errored (%s); NOT blocking promotion", exc)
        discard_staging(review_root)
        return _ReviewResult(
            passed=True, verdict="skipped", reason=f"review errored: {exc}"
        )

    # The only verdict channel is the executed <review_verdict/> action.
    # action — the tool handler wrote it to a file in the review staging. No
    # text is ever read: the prose stack produced 5 confirmed false rejects,
    # and the first structured version still scanned messages for a tag (a
    # plan quoted in <analysis> completed a session as ACCEPT — reproduced).
    verdict = review_verdict.read_verdict_file(review_staging)
    discard_staging(review_root)
    if verdict is None:
        # No structured submission: pass through ONCE (lenient, like the
        # review-errored path) — but count it, and trip the run-level breaker
        # on 3 consecutive skips or >10% of >=10 reviews.
        health = review_verdict.record_review_outcome(archive_root, skipped=True)
        if review_verdict.breaker_tripped(health):
            _logger.error(
                "review circuit breaker TRIPPED (%d/%d skipped, %d consecutive) "
                "— promotion paused for this candidate; inspect the review "
                "sessions before resuming",
                health["total_skips"],
                health["total_reviews"],
                health["consecutive_skips"],
            )
            return _ReviewResult(
                passed=False,
                verdict="blocked",
                reason=(
                    "review circuit breaker: "
                    f"{health['total_skips']}/{health['total_reviews']} reviews "
                    f"submitted no structured verdict "
                    f"({health['consecutive_skips']} consecutive)"
                ),
            )
        return _ReviewResult(
            passed=True,
            verdict="skipped",
            reason="review_skipped_parse_failure (no structured verdict; "
            "single pass-through)",
        )
    review_verdict.record_review_outcome(archive_root, skipped=False)
    accepted = verdict["decision"] == "accept"
    reason = verdict["reason"]
    if not accepted and verdict["reject_class"] != "none":
        reason = f"[{verdict['reject_class']}] {reason}"
    # File-level partial accept is disabled: dropping files while keeping stale
    # supersession metadata can retire the wrong parent. A partly bad bundle is
    # rejected as an implementation failure instead.
    return _ReviewResult(
        passed=accepted,
        verdict="accept" if accepted else "reject",
        reason=reason,
        drop_files=[],
        reject_class=verdict["reject_class"],
        repair_brief=verdict["repair_brief"],
    )


# ---------------------------------------------------------------------------
# Reflection step (called every K tasks)
# ---------------------------------------------------------------------------


def _ingest_findings_to_backlog(
    archive_root: Path,
    payload: dict,
    diagnose_items: list | None,
) -> None:
    """Mirror this step's findings into the append-only finding ledger.

    The snapshot next to this call is write-only — nothing reads it back, so an
    unselected finding has no way to survive its window. The ledger does, and it
    also stores the citations the finding dict never carried: which router
    bucket the task was in, whether the contrast pass came from history or
    nowhere at all, and where the two trial dirs are.

    Best-effort: ledger persistence must never fail a reflection.
    """
    provenance: dict[str, dict] = {}
    for item in diagnose_items or []:
        pass_roll = getattr(item, "contrast_pass", None)
        fail_roll = getattr(item, "contrast_fail", None)
        provenance[item.task] = {
            "bucket": getattr(item, "bucket", ""),
            "source": getattr(item, "source", ""),
            "fail_trial": getattr(fail_roll, "trial_dir", None),
            "pass_trial": getattr(pass_roll, "trial_dir", None),
        }
    try:
        _backlog.ingest_step(archive_root, payload, provenance=provenance)
    except Exception as exc:  # observer — never blocks the generation
        _logger.warning("backlog ingest failed (findings snapshot is intact): %s", exc)


async def _record_candidate_routing(
    *,
    archive_root: Path,
    staging_modules: Path,
    changed_files: list[str],
    locked_type: str,
    tasks: list[str],
    step_id: str,
    model_name: str,
    api_base: str | None,
    api_key: str | None,
) -> None:
    """Record whether the real composer would select the new variant.

    Uses the same code-as-source-of-truth rule as the activation gate — the
    variant name comes from what the changed file actually ``register()``s, not
    from what the editor declared. Writes one audit row; decides nothing.
    """
    names: list[str] = []
    for rel in changed_files:
        mod_file = staging_modules / rel
        if mod_file.is_file():
            names.extend(_registered_names_of_file(mod_file, locked_type))
    if len(names) != 1:
        # zero = the change did not add a selectable variant; more than one = the
        # activation gate will refuse it anyway. Either way there is nothing
        # single to probe, and guessing would put a name in the audit log that
        # nothing verified.
        return

    result = await _routing_probe.probe_routing(
        target_variant=names[0],
        tasks=tasks,
        instructions=_routing_probe.load_instructions(tasks),
        choose=_routing_probe.make_composer_chooser(
            modules_root=staging_modules,
            archive_root=archive_root,
            locked_type=locked_type,
            model_name=model_name,
            api_base=api_base,
            api_key=api_key,
        ),
    )
    _clustering._audit(
        archive_root,
        {
            "step": step_id,
            "decision": result.verdict,
            "target_variant": result.target_variant,
            "picked_by": result.picked_by,
            "picks": result.picks,
            "errors": result.errors,
            "reason": result.reason,
            "remedy": result.remedy,
        },
    )
    _logger.info("routing probe: %s (%s)", result.verdict, result.reason)


def _sanity_solver_concurrency(requested: int) -> int:
    """Cap concurrent sanity-task rolls.

    These are real solver runs launched DURING reflection, on top of the
    training pool (8) and editor sessions (2): uncapped they ran at
    sanity_concurrency=6, spiking the endpoint to ~16. The cap
    (HARBOR_AUX_SOLVER_CONCURRENCY, default 2) bounds them to the
    reflection-side share of the budget; by the time the battery runs, the
    editor sessions are done, so the total stays ≈ solver + 2. Never RAISES
    the requested concurrency — it only caps. All three knobs
    (TASK_CONCURRENCY / HARBOR_EDITOR_CONCURRENCY / this) are env-tunable per
    run; nothing is hardcoded."""
    raw = os.environ.get("HARBOR_AUX_SOLVER_CONCURRENCY", "")
    try:
        cap = max(1, int(raw))
    except (TypeError, ValueError):
        cap = 2
    return max(1, min(requested, cap))


_HISTORY_PASS_FILENAME = "history_pass.json"


def _record_history_passes(archive_root: Path, summaries: list[TrialSummary]) -> None:
    """Remember the latest passing trial directory per task.

    A later all-fail encounter of the same task hands this to the router's
    `history_pass_lookup` so its investigator gets a pass-vs-fail contrast
    instead of a single-sided guess. Best-effort JSON map; latest pass wins."""
    p = archive_root / _HISTORY_PASS_FILENAME
    try:
        mapping: dict[str, str] = json.loads(p.read_text())
    except Exception:
        mapping = {}
    changed = False
    for s in summaries:
        if s.reward is not None and s.reward >= 1.0 and s.trial_dir:
            mapping[s.task_name] = str(s.trial_dir)
            changed = True
    if changed:
        try:
            p.write_text(json.dumps(mapping, indent=1, ensure_ascii=False))
        except Exception as exc:  # never let bookkeeping kill a reflection
            _logger.warning("history_pass.json write failed: %s", exc)


def _make_history_pass_lookup(archive_root: Path):
    """`history_pass_lookup(task) -> TrialSummary | None` over recorded passes.

    Re-summarizes the stored trial dir on demand; a vanished/unreadable dir
    (cleanup, quarantine) degrades to None — single-sided diagnosis, never a
    crash."""
    p = archive_root / _HISTORY_PASS_FILENAME
    try:
        mapping: dict[str, str] = json.loads(p.read_text())
    except Exception:
        mapping = {}

    def lookup(task: str) -> TrialSummary | None:
        d = mapping.get(task)
        if not d:
            return None
        try:
            summary = summarize_trial(Path(d))
        except Exception:
            return None
        # only a still-verifiably-passing summary is a valid contrast partner
        if summary is None or summary.reward is None or summary.reward < 1.0:
            return None
        return summary

    return lookup


async def _contrastive_investigate_and_consolidate(
    *,
    archive_root: Path,
    current_gen: Path,
    pending_summaries: list[TrialSummary],
    step_id: str,
    model_name: str,
    api_base: str | None,
    api_key: str | None,
    skills_dir: Path | None,
    editor_timeout_sec: int,
    editor_call_timeout_sec: int,
    locked_module_type: str,
    epoch: int = 0,
    lens_max_turns: int = 80,
) -> None:
    """Collect contrastive findings and route them into the proposal backlog."""
    if not locked_module_type:
        _logger.warning("contrastive reflect needs --locked-module; skipping")
        return None

    ledger = Ledger.load(archive_root)
    # Remember this batch's passing trials and hand the router a
    # lookup over ALL past passes — an all-fail task whose earlier epoch had a
    # pass gets that trajectory as its contrast partner (the router's
    # `history_pass_lookup` hook existed but no live caller ever supplied it,
    # so every all-fail encounter was diagnosed single-sided).
    _record_history_passes(archive_root, pending_summaries)
    route = route_batch(
        pending_summaries,
        ledger=ledger,
        epoch=epoch,
        cfg=RouterConfig(),
        history_pass_lookup=_make_history_pass_lookup(archive_root),
    )
    _logger.info(
        "router[%s] → %d task(s) to diagnose", route.summary(), len(route.diagnose)
    )
    if not route.diagnose:
        ledger.save(archive_root)
        return None

    findings_root = archive_root / "findings"
    findings_root.mkdir(parents=True, exist_ok=True)
    instr_dir = archive_root / "auto_instructions"
    instr_dir.mkdir(parents=True, exist_ok=True)

    async def _investigate(item) -> dict:
        # Throwaway staging → read access to the locked module's code; discarded
        # after (read-only in effect — the investigator emits a finding, no edit).
        stage_root = findings_root / step_id / f"_stage_{item.task}"
        stage = _prepare_candidate_staging(current_gen, stage_root)
        if item.source == "efficiency":
            # all_pass_wasteful: diagnose the wasteful passing roll (S10).
            instr = build_efficiency_investigation_instruction(
                locked_module_type, item.task, item.contrast_fail
            )
        else:
            instr = build_contrast_investigation_instruction(
                locked_module_type,
                item.task,
                item.contrast_fail,
                item.contrast_pass,
                pass_from_history=(item.source == "archive_history"),
            )
        (instr_dir / f"{step_id}_contrast_{item.task}.md").write_text(instr)
        logs = archive_root / "editor_logs" / f"{step_id}_contrast_{item.task}"
        try:
            outcome = await asyncio.wait_for(
                run_editor(
                    staging_dir=stage,
                    archive_path=archive_root,
                    instruction=instr,
                    model_name=model_name,
                    api_base=api_base,
                    api_key=api_key,
                    skills_dir=skills_dir,
                    logs_dir=logs,
                    max_turns=lens_max_turns,
                    trajectory_root=archive_root / "solver",
                    llm_call_kwargs={"timeout": editor_call_timeout_sec},
                    locked_module_type=locked_module_type,
                ),
                timeout=editor_timeout_sec,
            )
            finding = parse_contrast_finding(outcome.trajectory_path, item.task)
        except Exception as exc:
            _logger.warning("contrast investigator %s failed: %s", item.task, exc)
            finding = {
                "task": item.task,
                "is_culprit": False,
                "note": f"investigator error: {exc}",
            }
        finally:
            discard_staging(stage_root)
        return finding

    _logger.info(
        "contrastive stage 1: %d per-task investigators (locked=%s)",
        len(route.diagnose),
        locked_module_type,
    )
    findings = list(await asyncio.gather(*[_investigate(it) for it in route.diagnose]))
    snapshot = {"step": step_id, "router": route.buckets, "findings": findings}
    (findings_root / f"{step_id}.json").write_text(
        json.dumps(snapshot, indent=2, ensure_ascii=False)
    )
    _ingest_findings_to_backlog(archive_root, snapshot, route.diagnose)

    try:
        active_variants, archive_entries = _evidence.archive_view(
            archive_root, current_gen / "modules", locked_module_type
        )
        evidence_report = await _evidence.collect_evidence(
            archive_root,
            snapshot=snapshot,
            diagnose_items=route.diagnose,
            active_variants=active_variants,
            archive_entries=archive_entries,
            locked_type=locked_module_type,
            step=step_id,
            ask=_evidence.make_asker(
                model_name=model_name, api_base=api_base, api_key=api_key
            ),
        )
        _logger.info("evidence pass: %s", evidence_report.as_dict())
    except Exception as exc:  # backlog observation must not stop the run
        _logger.warning("evidence pass failed: %s", exc)
    culprits = [f for f in findings if f.get("is_culprit")]
    _logger.info(
        "contrastive findings: %d/%d task(s) implicate %s",
        len(culprits),
        len(findings),
        locked_module_type,
    )

    # Do not charge every diagnosed task as a failed intervention. Only selected
    # proposals whose changed variant actually ran are accounted below.
    ledger.save(archive_root)

    # Efficiency lens (S10) — SECONDARY: only when there is no correctness gap to
    # fix does the single-variant slot open for a "make it leaner" change. This
    # keeps efficiency from ever starving correctness.
    if not culprits and route.efficiency:
        _logger.info(
            "no correctness culprit → efficiency lens on %d wasteful task(s)",
            len(route.efficiency),
        )
        eff = list(await asyncio.gather(*[_investigate(it) for it in route.efficiency]))
        eff_culprits = [f for f in eff if f.get("is_culprit")]
        if eff_culprits:
            findings, culprits = eff, eff_culprits

    return None


async def _gate_and_promote(
    *,
    # --- per-candidate state, produced by the editor stage ---
    editor_outcome,
    staging_root: Path,
    staging_modules: Path,
    diff_base: Path,
    _changed,
    candidate_n: int,
    step_id: str,
    proposal_action: str = "",
    proposal_target: str = "",
    # --- window-level config, passed through verbatim ---
    api_base: str | None,
    api_key: str | None,
    archive_root: Path,
    current_gen: Path,
    e2b_pool=None,
    editor_call_timeout_sec: int = 240,
    editor_timeout_sec: int = 1500,
    environment: str | None = None,
    locked_module_type: str | None = None,
    max_editor_turns: int,
    max_gate_repairs: int = 1,
    max_solver_turns: int = 200,
    model_name: str,
    pending_summaries: list[TrialSummary],
    sanity_concurrency: int = 6,
    review_gate: bool = True,
    review_max_turns: int = 60,
    sanity_gate: bool = True,
    sanity_tasks: list[str] | None = None,
    skills_dir: Path | None,
    solver_timeout_sec: int | None = None,
) -> _ReflectionOutcome:
    """Every gate a candidate must clear, then promote it.

    Extracted verbatim from `_maybe_reflect_inner` so a window with TWO
    candidates can run this once per lane, on its own staging tree, with
    neither lane's gates seeing the other's. The seven arguments above the
    line are exactly the per-candidate state; everything below is window-level
    and identical for both lanes.
    """
    out = _ReflectionOutcome(
        triggered=True,
        promoted_gen=None,
        discard_reason=None,
        editor_n_edits=editor_outcome.n_edits,
        editor_committed=editor_outcome.committed,
    )
    # Capture intent + changed files NOW — staging still exists; a later
    # promote (mv) / discard (rm) removes it. These feed the evolution log.
    out.candidate_gen_n = candidate_n
    out.editor_trajectory_path = editor_outcome.trajectory_path
    out.intent = extract_editor_intent(editor_outcome.trajectory_path)
    out.variant_meta_text = extract_variant_meta_blocks(editor_outcome.trajectory_path)
    out.files_changed = _changed(staging_modules)

    if not editor_outcome.success:
        out.discard_reason = f"editor failed: {editor_outcome.error}"
        discard_staging(staging_root)
        return out

    # Editor committed but made no edits → it explicitly chose "no change"
    if editor_outcome.n_edits == 0:
        _logger.info("editor chose 'no change' (committed without edits)")
        out.discard_reason = "editor chose no change"
        discard_staging(staging_root)
        return out

    # The parent tree the gates read, derived from what this candidate is
    # actually diffed against — NOT from `current_gen`. A rebased second lane
    # sits on the generation the first lane just promoted, so `diff_base` moved
    # and `current_gen` did not; reading `current_gen` here files the OTHER
    # lane's new files as this lane's additions and the niche gate rejects the
    # lane for a cell its sibling occupies.
    parent_modules = diff_base

    # Smoke test the staged changes — if it fails to LOAD (import/syntax/Protocol),
    # try to REPAIR before discarding (a relative-import or a typo is a mechanical
    # bug, not a bad idea). Bounded by max_gate_repairs.
    smoke_repairs = 0
    while True:
        smoke = run_fast_smoke(staging_modules)
        if smoke.passed:
            if smoke_repairs:
                _logger.info("smoke passed after %d repair(s)", smoke_repairs)
            break
        fails = smoke.all_failures()
        if smoke_repairs >= max_gate_repairs:
            out.discard_reason = "smoke failed:\n  " + "\n  ".join(fails)
            discard_staging(staging_root)
            return out
        smoke_repairs += 1
        _logger.info(
            "smoke fail → repair attempt %d/%d: %s",
            smoke_repairs,
            max_gate_repairs,
            "; ".join(fails)[:200],
        )
        diff_text = build_diff_text(
            diff_base, staging_modules, _changed(staging_modules)
        )
        repair_instr = build_smoke_repair_instruction(fails, diff_text)
        repair_logs = (
            archive_root / "editor_logs" / f"{step_id}_smokerepair_{smoke_repairs:02d}"
        )
        try:
            repaired = await asyncio.wait_for(
                run_editor(
                    staging_dir=staging_modules,
                    archive_path=archive_root,
                    instruction=repair_instr,
                    model_name=model_name,
                    api_base=api_base,
                    api_key=api_key,
                    skills_dir=skills_dir,
                    logs_dir=repair_logs,
                    max_turns=max_editor_turns,
                    trajectory_root=archive_root / "solver",
                    llm_call_kwargs={"timeout": editor_call_timeout_sec},
                    locked_module_type=locked_module_type,
                ),
                timeout=editor_timeout_sec,
            )
        except Exception as exc:
            out.discard_reason = f"smoke repair editor failed: {exc}"
            discard_staging(staging_root)
            return out
        if not (repaired.committed and repaired.n_edits > 0):
            out.discard_reason = "smoke repair: editor made no fix"
            discard_staging(staging_root)
            return out
        out.editor_n_edits += repaired.n_edits
        out.files_changed = _changed(staging_modules)
        # loop: re-run smoke on the repaired staging

    # Record whether normal composer routing would select the candidate. A
    # forced sanity pin proves the variant runs but says nothing about whether a
    # normal solver would ever select it, because the pin bypasses DESCRIPTION
    # routing. This asks that second question on the batch's own tasks, for one
    # LLM call each. It is recorded and NEVER gates: "nothing picks it" is at
    # least as likely to be a badly-written DESCRIPTION as a useless variant.
    if locked_module_type:
        try:
            await _record_candidate_routing(
                archive_root=archive_root,
                staging_modules=staging_modules,
                changed_files=out.files_changed,
                locked_type=locked_module_type,
                tasks=[s.task_name for s in pending_summaries],
                step_id=step_id,
                model_name=model_name,
                api_base=api_base,
                api_key=api_key,
            )
        except Exception as exc:  # observer — never costs a generation
            _logger.warning("routing probe failed: %s", exc)

    # ---- Action gate (C3): did it build what it was routed to build? --------
    # The lane/action routing reached the implementer as prose and nothing ever
    # checked the result against it, so an `add` could come back as an edit to
    # the incumbent and a `replace` as more logic piled into the variant it was
    # supposed to retire. Mechanical, no LLM. Runs after smoke because it needs
    # a loadable library on both sides.
    if proposal_action:
        action_reject = _action_reject(
            action=proposal_action,
            target_variant=proposal_target,
            files_changed=out.files_changed,
            variant_meta_text=out.variant_meta_text or "",
            parent_modules=parent_modules,
            staged_modules=staging_modules,
        )
        if action_reject is not None:
            _logger.info("action gate REJECT: %s", action_reject)
            out.discard_reason = action_reject
            discard_staging(staging_root)
            return out

    # ---- Niche dedup gate (S5): the anti-mode-collapse enforcement -----------
    # A new variant that lands in an already-occupied niche cell without declaring
    # it supersedes the incumbent is a near-duplicate → reject. Runs after smoke
    # (needs a loadable library) and before the expensive review/sanity gates.
    niche_reject = _niche_dedup_reject(
        archive_root,
        staging_modules,
        parent_modules,
        out.variant_meta_text or "",
    )
    if niche_reject is not None:
        _logger.info("niche dedup gate REJECT: %s", niche_reject.replace("\n", " "))
        out.discard_reason = niche_reject
        discard_staging(staging_root)
        return out

    # ---- Review gate (editor self-review: effective + not overfit) ---------
    # A SECOND editor pass re-investigates the trajectories + the diff it just
    # made and decides KEEP / REJECT. Pure reasoning (no reward read) → no eval
    # leakage; specifically targets overfit / ineffective edits. Hard gate:
    # reject → discard. Runs before the sanity gate so a rejected change skips
    # the expensive docker sanity run.
    if review_gate:
        review = await review_edit(
            archive_root=archive_root,
            parent_modules=diff_base,
            staging_modules=staging_modules,
            pending_summaries=pending_summaries,
            intent=out.intent,
            files_changed=out.files_changed,
            model_name=model_name,
            api_base=api_base,
            api_key=api_key,
            skills_dir=skills_dir,
            step_id=step_id,
            review_max_turns=review_max_turns,
            review_timeout_sec=editor_timeout_sec,
            review_call_timeout_sec=editor_call_timeout_sec,
            locked_module_type=locked_module_type,
        )
        out.review_passed = review.passed
        out.review_verdict = review.verdict
        out.review_reason = review.reason
        out.review_reject_class = review.reject_class
        out.review_repair_brief = review.repair_brief
        _logger.info("review gate: verdict=%s (%s)", review.verdict, review.reason)
        if not review.passed:
            out.discard_reason = f"review gate (reject): {review.reason}"
            discard_staging(staging_root)
            return out

        # Partial accept: revert ONLY the reviewer-flagged files (an overfit /
        # ineffective file bundled with clean improvements), keep the rest, then
        # re-run smoke so a drop that breaks a dependency is caught before the
        # (expensive) probe battery. Cheap: no task runs here.
        if review.drop_files:
            applied, skipped = _apply_review_drops(
                staging_modules, diff_base, review.drop_files
            )
            if applied:
                remaining = _changed(staging_modules)
                if not remaining:
                    out.discard_reason = (
                        f"review partial-accept: dropping {applied} left no change"
                    )
                    discard_staging(staging_root)
                    return out
                resmoke = run_fast_smoke(staging_modules)
                if not resmoke.passed:
                    out.discard_reason = (
                        f"review partial-accept: dropping {applied} broke smoke: "
                        + "; ".join(resmoke.all_failures())[:200]
                    )
                    discard_staging(staging_root)
                    return out
                out.files_changed = remaining
                out.review_reason = (
                    f"{review.reason} [partial-accept: dropped {applied}]"
                )
                _logger.info(
                    "review partial-accept: dropped %s, kept %s", applied, remaining
                )

    # The sanity battery discards code crashes. Task failure and infrastructure
    # timeouts are not treated as implementation breakage.
    # parent_mean is the batch's IN-SAMPLE reward on the CURRENT gen — recorded
    # for the evolution log as a reference only, never used to gate.
    parent_rewards = [
        float(s.reward) if (s.reward is not None) else 0.0 for s in pending_summaries
    ]
    out.parent_mean_reward = (
        sum(parent_rewards) / len(parent_rewards) if parent_rewards else 0.0
    )

    # NOTE: module selection is now PER-TASK at runtime (LLMComposer), not a
    # generation-level step here — reflection only GROWS the library (the editor
    # adds variants). The solver/sanity runs pick the right variant per task via
    # `composer_name="llm_dynamic"` (the solver default). So there is no compose
    # step in the reflection chain; the sanity gate below runs the real per-task
    # composer on the staged library.

    if sanity_gate:
        try:
            required_activation = _required_sanity_variant(
                archive_root=archive_root,
                staging_root=staging_modules,
                files_changed=out.files_changed,
                variant_meta_text=out.variant_meta_text or "",
                locked_module_type=locked_module_type,
            )
        except ValueError as exc:
            out.sanity_passed = False
            out.sanity_break_reason = str(exc)
            out.discard_reason = f"sanity activation gate: {exc}"
            _logger.warning("sanity activation gate REJECT: %s", exc)
            discard_staging(staging_root)
            return out

        tasks_to_run = sanity_tasks or DEFAULT_SANITY_TASKS
        sanity_timeout = solver_timeout_sec
        k_eff = 1
        repair_attempts = 0
        while True:
            _logger.info(
                "sanity gate: running %d fixed tasks x K=%d on staging: %s",
                len(tasks_to_run),
                k_eff,
                ", ".join(tasks_to_run),
            )
            sanity_root = (
                archive_root
                / "sanity_check"
                / (
                    step_id
                    if repair_attempts == 0
                    else f"{step_id}_repair{repair_attempts}"
                )
            )
            sanity_root.mkdir(parents=True, exist_ok=True)

            # A sanity pass is meaningful only if it executes the code that was
            # just changed. Pin that variant in a PRIVATE staging copy: the
            # candidate itself remains unchanged, and all other module types
            # retain their normal per-task Composer behavior.
            primary_source = staging_modules
            primary_run_root = sanity_root
            forced_run_root: Path | None = None
            if required_activation is not None:
                try:
                    forced_source = _prepare_forced_sanity_source(
                        staging_modules,
                        sanity_root / "_forced_candidate_source",
                        required_activation,
                    )
                except (OSError, ValueError) as exc:
                    out.sanity_passed = False
                    out.sanity_break_reason = str(exc)
                    out.discard_reason = f"sanity activation gate: {exc}"
                    _logger.warning("sanity activation gate REJECT: %s", exc)
                    discard_staging(staging_root)
                    return out

                forced_run_root = sanity_root / "forced_activation"
                # Reuse the K=1 crash checks so activation verification adds no
                # extra solver rolls.
                primary_source = forced_source
                primary_run_root = forced_run_root

            # Each primary task composes once, then reuses a private static
            # bundle for its remaining K-1 rolls. Concurrency is across tasks.
            sanity_groups = await _run_same_bundle_battery(
                source_root=primary_source,
                run_root=primary_run_root,
                tasks=list(tasks_to_run),
                k_repeats=k_eff,
                model_name=model_name,
                api_base=api_base,
                api_key=api_key,
                max_solver_turns=max_solver_turns,
                timeout_sec=sanity_timeout,
                environment=environment,
                concurrency=_sanity_solver_concurrency(sanity_concurrency),
                e2b_pool=e2b_pool,
            )

            break_reasons: list[str] = []
            crashes: list[dict] = []
            per_task: list[tuple[str, list[float | None]]] = []
            bundle_audit: dict[str, dict] = {}
            for task, group in zip(tasks_to_run, sanity_groups, strict=True):
                rewards: list[float | None] = []
                for hr, run_dir in group.rolls:
                    rewards.append(hr.reward)
                    summary = _harbor_results_to_summary(run_dir)
                    broke, reason = _is_code_break(hr, summary)
                    if broke:
                        break_reasons.append(reason)
                        crashes.append(
                            _sanity_crash_detail(
                                run_dir,
                                task,
                                summary.exception_type if summary else None,
                            )
                        )
                rewards.extend([None] * (k_eff - len(rewards)))
                per_task.append((task, rewards))
                bundle_audit[task] = {
                    "bundle_id": group.bundle_id,
                    "invalid_reason": group.invalid_reason,
                }
                if not group.valid:
                    _logger.error(
                        "candidate probe K-group invalid for %s: %s",
                        task,
                        group.invalid_reason,
                    )
            out.sanity_per_task = per_task
            out.sanity_bundles = bundle_audit

            out.sanity_activation = _collect_sanity_activation(sanity_root)
            forced_activation = (
                _collect_sanity_activation(forced_run_root)
                if required_activation is not None and forced_run_root is not None
                else None
            )
            activation_failure: str | None = None
            if required_activation is not None:
                activation_failure = _required_activation_failure(
                    forced_activation, required_activation
                )
                if out.sanity_activation is None:
                    out.sanity_activation = {"n_trials": 0, "variants": {}}
                out.sanity_activation.update(
                    {
                        "required": (
                            f"{required_activation[0]}:{required_activation[1]}"
                        ),
                        "forced_verified": activation_failure is None,
                        "forced_evidence": forced_activation,
                    }
                )

            if out.sanity_activation:
                acts = out.sanity_activation.get("variants", {})
                _logger.info(
                    "sanity activation: %s",
                    " ".join(
                        f"{k}(sel={v['selected']},calls={v['trace_calls']})"
                        for k, v in sorted(acts.items())
                    )[:800],
                )

            if not break_reasons and activation_failure is not None:
                out.sanity_passed = False
                out.sanity_break_reason = activation_failure
                out.discard_reason = f"sanity activation gate: {activation_failure}"
                _logger.warning("sanity activation gate REJECT: %s", activation_failure)
                discard_staging(staging_root)
                return out

            if not break_reasons:
                out.sanity_passed = True
                _logger.info(
                    "sanity gate passed (all %d tasks ran clean%s)",
                    len(tasks_to_run),
                    f", after {repair_attempts} repair(s)" if repair_attempts else "",
                )
                break

            # Crash — but the change already passed review (the idea is sound), so
            # try to REPAIR the bug before discarding, up to max_gate_repairs.
            if repair_attempts >= max_gate_repairs:
                out.sanity_passed = False
                out.sanity_break_reason = "; ".join(break_reasons)
                out.discard_reason = (
                    f"sanity gate (code crash, {repair_attempts} repair(s) tried): "
                    + out.sanity_break_reason
                )
                _logger.warning(
                    "sanity gate FAILED after %d repair(s) — staged modules crash: %s",
                    repair_attempts,
                    out.sanity_break_reason,
                )
                discard_staging(staging_root)
                return out

            repair_attempts += 1
            _logger.info(
                "sanity crash → repair attempt %d/%d: %s",
                repair_attempts,
                max_gate_repairs,
                "; ".join(break_reasons),
            )
            diff_text = build_diff_text(
                diff_base, staging_modules, _changed(staging_modules)
            )
            repair_instr = build_sanity_repair_instruction(crashes, diff_text)
            repair_logs = (
                archive_root / "editor_logs" / f"{step_id}_repair_{repair_attempts:02d}"
            )
            crash_root = (
                Path(crashes[0]["trial_dir"])
                if crashes and crashes[0].get("trial_dir")
                else sanity_root
            )
            try:
                repaired = await asyncio.wait_for(
                    run_editor(
                        staging_dir=staging_modules,
                        archive_path=archive_root,
                        instruction=repair_instr,
                        model_name=model_name,
                        api_base=api_base,
                        api_key=api_key,
                        skills_dir=skills_dir,
                        logs_dir=repair_logs,
                        max_turns=max_editor_turns,
                        trajectory_root=crash_root,
                        llm_call_kwargs={"timeout": editor_call_timeout_sec},
                        locked_module_type=locked_module_type,
                    ),
                    timeout=editor_timeout_sec,
                )
            except Exception as exc:
                out.sanity_passed = False
                out.sanity_break_reason = "; ".join(break_reasons)
                out.discard_reason = f"sanity repair editor failed: {exc}"
                discard_staging(staging_root)
                return out
            if not (repaired.committed and repaired.n_edits > 0):
                out.sanity_passed = False
                out.sanity_break_reason = "; ".join(break_reasons)
                out.discard_reason = "sanity repair: editor made no fix"
                discard_staging(staging_root)
                return out
            if not run_fast_smoke(staging_modules).passed:
                out.sanity_passed = False
                out.discard_reason = "sanity repair broke smoke"
                discard_staging(staging_root)
                return out
            out.editor_n_edits += repaired.n_edits
            out.files_changed = _changed(staging_modules)
            # loop: re-run the sanity tasks on the repaired staging

    # Promotion order belongs to the two-lane window because both candidates
    # share a parent and the second may need to rebase onto the first.
    out.gates_passed = True
    return out


def _task_accounting_hook(archive_root: Path):
    """Charge the ledger only for an intervention that actually ran.

    Two conditions, both from the window, make the replacement honest: the
    proposal reached an implementer (so a task nobody acted on is never
    charged), and the changed variant actually RAN (so a change that never
    executed is not evidence that a task is unfixable).
    """

    def settle(settlement) -> None:
        if not settlement.ran or not settlement.support_tasks:
            return
        ledger = Ledger.load(archive_root)
        for task in settlement.support_tasks:
            ledger.record_reflection(task, made_progress=settlement.made_progress)
        ledger.save(archive_root)

    return settle


def _archive_sync_hook(archive_root: Path):
    """The two-lane window's `on_promote`, bound to this run's archive.

    The legacy single-candidate path calls `_sync_archive_after_promote` inside
    `_gate_and_promote`. The two-lane path promotes in `two_lane.run_window`,
    which must not import this module, so the same call is handed in.

    `parent_gen` comes from the window, NOT from `current_gen`: a rebased lane
    was built on the generation the other lane just created, and measuring it
    against `current_gen` would file the other lane's files as this lane's
    additions.
    """

    def sync(promoted_gen: Path, parent_gen: Path, result: Any) -> None:
        detail = getattr(result, "detail", None)
        _sync_archive_after_promote(
            archive_root,
            promoted_gen,
            parent_gen / "modules",
            getattr(detail, "variant_meta_text", "") or "",
        )

    return sync


async def _maybe_reflect_inner(
    *,
    archive_root: Path,
    current_gen: Path,
    pending_summaries: list[TrialSummary],
    model_name: str,
    api_base: str | None,
    api_key: str | None,
    max_editor_turns: int,
    skills_dir: Path | None,
    review_max_turns: int = 60,
    sanity_tasks: list[str] | None = None,
    environment: str | None = None,
    max_solver_turns: int = 200,
    solver_timeout_sec: int | None = None,
    editor_timeout_sec: int = 1500,
    editor_call_timeout_sec: int = 240,
    max_gate_repairs: int = 2,
    sanity_concurrency: int = 6,
    e2b_pool=None,
    locked_module_type: str | None = None,
    max_lanes: int = 2,
    epoch: int = 0,
) -> _ReflectionOutcome:
    """Collect findings, implement selected proposals, and run the gates."""
    if not any((summary.reward or 0) < 1.0 for summary in pending_summaries):
        _logger.info(
            "all %d pending trials passed; skipping reflection",
            len(pending_summaries),
        )
        return _ReflectionOutcome(
            triggered=False, promoted_gen=None, discard_reason=None
        )
    if not locked_module_type:
        raise ValueError("Phase0 requires a locked module")

    candidate_n = next_gen_number(archive_root)
    step_id = f"gen_{candidate_n}_candidate_{int(time.time())}"

    await _contrastive_investigate_and_consolidate(
        archive_root=archive_root,
        current_gen=current_gen,
        pending_summaries=pending_summaries,
        step_id=step_id,
        model_name=model_name,
        api_base=api_base,
        api_key=api_key,
        skills_dir=skills_dir,
        editor_timeout_sec=editor_timeout_sec,
        editor_call_timeout_sec=editor_call_timeout_sec,
        locked_module_type=locked_module_type,
        epoch=epoch,
    )
    return await _two_lane_reflect(
        archive_root=archive_root,
        current_gen=current_gen,
        step_id=step_id,
        candidate_n=candidate_n,
        max_lanes=max_lanes,
        model_name=model_name,
        api_base=api_base,
        api_key=api_key,
        skills_dir=skills_dir,
        max_editor_turns=max_editor_turns,
        pending_summaries=pending_summaries,
        editor_timeout_sec=editor_timeout_sec,
        editor_call_timeout_sec=editor_call_timeout_sec,
        locked_module_type=locked_module_type,
        review_gate=True,
        review_max_turns=review_max_turns,
        sanity_gate=True,
        sanity_tasks=sanity_tasks,
        environment=environment,
        max_solver_turns=max_solver_turns,
        solver_timeout_sec=solver_timeout_sec,
        max_gate_repairs=max_gate_repairs,
        sanity_concurrency=sanity_concurrency,
        e2b_pool=e2b_pool,
    )


def _lane_tree_paths(
    staging_root: Path,
    parent: Path,
) -> tuple[Path, Path]:
    """(what the editor edits, what the diff is measured against).

    `parent` is the tree this lane actually sits on, which is NOT always the
    generation the window opened with: when both lanes clear their gates the
    incumbent promotes first and the novelty lane is rebased onto the generation
    it just created. Measuring a rebased lane against the window's original
    parent files the incumbent's new files as this lane's changes.

    """
    return staging_root / "modules", parent / "modules"


def _record_lane_implementation(
    archive_root: Path, proposal_id: str, *, step: str, outcome
) -> None:
    """C1's other half: what this lane actually produced, against its proposal.

    The proposal says what was *intended*; the manifest says what was *built* —
    which files, which `<variant_meta>`, which incumbents it claims to retire.
    They arrive at different times and the staging that carried the second one
    is deleted minutes later, so without this there is nothing left to check the
    build against the direction after the fact.

    (The drop half of the manifest — `drop_implementation` and the
    `live_supersede_targets` read — stays unwired: partial accept is disabled,
    so there is no drop to be atomic about yet.)

    Never raises: bookkeeping does not get to fail a change that passed its gates.
    """
    if not proposal_id or outcome is None:
        return
    try:
        meta_text = getattr(outcome, "variant_meta_text", "") or ""
        _manifest.record_implementation(
            archive_root,
            proposal_id=proposal_id,
            step=step,
            files=list(getattr(outcome, "files_changed", []) or []),
            variant_meta_text=meta_text,
            supersede_targets=sorted(
                {
                    ref
                    for meta in _archive.parse_variant_meta(meta_text)
                    for ref in (meta.get("supersedes") or [])
                }
            ),
        )
    except Exception as exc:  # noqa: BLE001
        _logger.warning("could not record what %s built: %s", proposal_id, exc)


def _review_gate_for(gates: tuple[str, ...]) -> bool:
    """Should the re-gate pay for the review LLM pass again?

    Only if the rebase invalidated it. Review is the one verdict a byte-identical
    diff carries over, and it is both the most expensive gate and the largest
    historical source of false rejects — so re-running it when nothing about the
    change moved is paying an LLM call for another chance to kill a good change.
    """
    return "review" in gates


async def _two_lane_reflect(
    *,
    archive_root: Path,
    current_gen: Path,
    step_id: str,
    candidate_n: int,
    max_lanes: int,
    novelty_first: bool = False,
    **gate_kwargs,
) -> _ReflectionOutcome:
    """The two-lane window: build both selected proposals, promote what lands.

    This is assembly, not new policy. The direction each lane implements was
    already decided by the backlog (:mod:`portfolio`), the brief it gets is
    already built (:mod:`dual_implement`), and who promotes first is already
    decided (:mod:`promotion`). All that happens here is plugging the real
    editor and the real gate battery into those seams.

    Note what is NOT here: no consolidator. In the legacy path the editor picks
    its own direction out of the batch, which is exactly the greedy selection
    this phase replaces. Here the direction arrives from the backlog and the
    editor only implements it.
    """

    def _tree_paths(staging_root: Path, parent: Path) -> tuple[Path, Path]:
        return _lane_tree_paths(staging_root, parent)

    def _changer(diff_base: Path):
        return lambda staging: changed_module_files(diff_base, staging)

    async def _gate(
        *,
        staging_root: Path,
        editor_outcome,
        parent: Path,
        lane: str,
        review_gate: bool | None = None,
        proposal=None,
    ) -> _ReflectionOutcome:
        staging_modules, diff_base = _tree_paths(staging_root, parent)
        kwargs = dict(gate_kwargs)
        if review_gate is not None:
            kwargs["review_gate"] = review_gate
        return await _gate_and_promote(
            editor_outcome=editor_outcome,
            staging_root=staging_root,
            staging_modules=staging_modules,
            diff_base=diff_base,
            _changed=_changer(diff_base),
            candidate_n=candidate_n,
            step_id=f"{step_id}_{lane}",
            # C3: the routed action, so the battery can check that this is what
            # was actually built. Only the lane knows which proposal it is.
            proposal_action=getattr(proposal, "action", "") or "",
            proposal_target=getattr(proposal, "target_variant", "") or "",
            archive_root=archive_root,
            current_gen=current_gen,
            **kwargs,
        )

    async def run_lane(brief):
        staging_modules, _ = _tree_paths(brief.staging_dir, current_gen)
        editor_outcome = await run_editor(
            staging_dir=staging_modules,
            archive_path=archive_root,
            instruction=brief.instruction,
            logs_dir=archive_root / "editor_logs" / f"{step_id}_{brief.lane}",
            trajectory_root=archive_root / "solver",
            model_name=gate_kwargs["model_name"],
            api_base=gate_kwargs.get("api_base"),
            api_key=gate_kwargs.get("api_key"),
            skills_dir=gate_kwargs.get("skills_dir"),
            max_turns=gate_kwargs.get("max_editor_turns", 30),
            locked_module_type=gate_kwargs.get("locked_module_type"),
            llm_call_kwargs={
                "timeout": gate_kwargs.get("editor_call_timeout_sec", 240)
            },
        )
        out = await _gate(
            staging_root=brief.staging_dir,
            editor_outcome=editor_outcome,
            parent=current_gen,
            lane=brief.lane,
            proposal=brief.proposal,
        )
        _record_lane_implementation(
            archive_root, brief.proposal_id, step=step_id, outcome=out
        )
        return _dual_implement.LaneResult(
            passed=out.gates_passed, diff_hash="", detail=out
        )

    async def rerun_gates(*, dest, gates, lane, new_parent, prior, proposal_id=""):
        # `prior` carries the first pass's editor session: re-gating must never
        # re-run the editor, and the reused review verdict belongs to it.
        detail = getattr(prior, "detail", None)
        proposal = None
        if proposal_id:
            try:
                proposal = _proposals.get_proposal(archive_root, proposal_id)
            except Exception as exc:  # noqa: BLE001
                _logger.warning("re-gate could not read %s: %s", proposal_id, exc)
        out = await _gate(
            staging_root=dest,
            editor_outcome=_RebasedEditorOutcome(dest, detail),
            parent=new_parent,
            lane=lane,
            # everything else in `gates` re-runs because it reads the merged tree
            review_gate=_review_gate_for(gates),
            proposal=proposal,
        )
        _record_lane_implementation(
            archive_root, proposal_id, step=step_id, outcome=out
        )
        return _dual_implement.LaneResult(
            passed=out.gates_passed, diff_hash="", detail=out
        )

    result = await _two_lane.run_window(
        archive_root,
        parent_gen=current_gen,
        step=step_id,
        run_lane=run_lane,
        rerun_gates=rerun_gates,
        max_lanes=max_lanes,
        novelty_first=novelty_first,
        on_promote=_archive_sync_hook(archive_root),
        on_lane_settled=_task_accounting_hook(archive_root),
    )

    out = _ReflectionOutcome(triggered=True, promoted_gen=None, discard_reason=None)
    out.candidate_gen_n = candidate_n
    for verdict in result.verdicts:
        if verdict.promoted_gen is None:
            continue
        lane_out = verdict.detail
        out.record_promotion(
            verdict.promoted_gen,
            proposal_id=verdict.proposal_id,
            lane=verdict.lane,
            intent=getattr(lane_out, "intent", "") or "",
            variant_meta_text=getattr(lane_out, "variant_meta_text", "") or "",
            files_changed=verdict.files_changed
            or list(getattr(lane_out, "files_changed", []) or []),
            gates={
                "review": getattr(lane_out, "review_verdict", None),
                "sanity": getattr(lane_out, "sanity_passed", None),
            },
        )
    if not out.promotions:
        out.discard_reason = (
            "; ".join(f"{v.lane}: {v.reason}" for v in result.verdicts)
            or "two-lane: nothing selected"
        )
    return out


class _RebasedEditorOutcome:
    """The first pass's editor session, re-pointed at the rebased tree.

    A rebase produces no new editor session — that is the whole point of reusing
    the change-local verdict — but the gate battery reads an editor outcome for
    the trajectory behind `intent` / `<variant_meta>`. Those still describe this
    change, so they come from the original run; only the tree moved.
    """

    def __init__(self, staging_dir: Path, prior: Any):
        self.staging_dir = staging_dir
        self.success = True
        self.committed = True
        self.error = None
        self.n_edits = getattr(prior, "editor_n_edits", 1) or 1
        # `_trajectory_path` was a name nothing ever wrote, so this read was
        # always None and the re-gate re-derived `intent` / `<variant_meta>`
        # from nothing — a rebased lane reached the archive with no genealogy.
        self.trajectory_path = getattr(prior, "editor_trajectory_path", None)


def _promotion_count(out: _ReflectionOutcome) -> int:
    """How many generations this window produced.

    `meta.json` reports the total as `n_promotions`; counting windows instead of
    generations would under-report a two-lane lineage by half. The `promoted_gen`
    fallback covers a promotion recorded without `record_promotion`.
    """
    if out.promotions:
        return len(out.promotions)
    return 1 if out.promoted_gen is not None else 0


def _remember_promotions(
    archive_root: Path, out: _ReflectionOutcome, *, epoch: int, module: str
) -> None:
    """Log every generation this window produced to the editor memory.

    The consolidator reads this to avoid re-proposing what was already tried, so
    recording one of two generations leaves that memory half-blind — and each
    generation carries its OWN intent, not the window's.
    """
    for promo in out.promotions or (
        [_Promotion(gen=out.promoted_gen, intent=out.intent)]
        if out.promoted_gen is not None
        else []
    ):
        editor_memory.record(
            archive_root,
            epoch=epoch,
            module=module,
            change=(promo.intent or promo.variant_meta_text or "")[:280],
            gen=promo.gen.name,
            verdict="provisional",
        )


def _evolution_log_records(out: _ReflectionOutcome) -> list[dict]:
    """One record per generation this window produced.

    Each generation is still a single change, so merging two lanes into one
    record would destroy the only per-generation attribution there is — and the
    editor reads this log to avoid re-trying what was already tried. A window
    that promoted nothing still gets exactly one ``(discarded)`` record.
    """
    discarded_gen = (
        f"gen_{out.candidate_gen_n}(discarded)"
        if out.candidate_gen_n is not None
        else "gen_?(discarded)"
    )
    promotions = out.promotions or [
        _Promotion(
            gen=out.promoted_gen or Path(discarded_gen),
            intent=out.intent,
            variant_meta_text=out.variant_meta_text,
            files_changed=list(out.files_changed),
        )
    ]
    records = []
    for promo in promotions:
        gen = promo.gen.name if out.promoted_gen is not None else discarded_gen
        record = _evolution_log_record(out, promo, gen)
        # Per-lane verdicts win over the window-level ones: with two lanes each
        # ran its OWN review and sanity, and reporting one lane's verdict
        # against the other's generation logs a judgement nobody made.
        record.update(promo.gates)
        records.append(record)
    return records


async def _maybe_reflect(**kwargs) -> _ReflectionOutcome:
    """Run a reflection, then append one evolution-log record per generation it
    produced (change → hypothesis → verdict) so the next generation's editor can
    see what was already tried and whether it worked — the anti-flip-flop memory.
    """
    archive_root = kwargs["archive_root"]
    out = await _maybe_reflect_inner(**kwargs)
    if out.triggered:
        for record in _evolution_log_records(out):
            append_evolution_log(archive_root, record)
    return out


def _evolution_log_record(out: _ReflectionOutcome, promo: _Promotion, gen: str) -> dict:
    """The window-level half of one generation's record.

    Window-level observations are shared by both lanes. Lane-specific review
    and sanity verdicts are overlaid from ``_Promotion.gates``.
    """
    return {
        "gen": gen,
        "proposal_id": promo.proposal_id or None,
        "lane": promo.lane or None,
        "files": promo.files_changed,
        "intent": promo.intent,
        # parent_mean = batch in-sample reward (reference only). There is
        # no staging re-run any more, so no staging_mean / "did it help"
        # verdict — promotion now means only "passed the crash sanity gate".
        "parent_mean": out.parent_mean_reward,
        "staging_mean": None,
        "review": out.review_verdict,
        # Durable reject taxonomy for retries and resume.
        "review_reject_class": out.review_reject_class
        if out.review_reject_class != "none"
        else None,
        "review_repair_brief": out.review_repair_brief or None,
        "sanity": out.sanity_passed,
        # Soft activation accounting from the sanity battery (which
        # variants the composer picked + kernel trace call counts).
        "activation": out.sanity_activation,
        "promoted": out.promoted_gen is not None,
        "reason": out.discard_reason,
    }


# ---------------------------------------------------------------------------
# Main online loop
# ---------------------------------------------------------------------------


async def run_online_evo(
    *,
    archive_root: Path,
    tasks: list[str],
    model_name: str,
    api_base: str | None,
    api_key: str | None,
    reflect_every: int = 1,
    task_concurrency: int | None = None,
    max_tasks: int | None = None,
    epochs: int = 1,
    shuffle_tasks: bool = True,
    task_seed: int = 0,
    max_editor_turns: int = 30,
    max_solver_turns: int = 150,
    solver_timeout_sec: int | None = None,
    agent_timeout_multiplier: float | None = None,
    editor_timeout_sec: int = 1500,
    editor_call_timeout_sec: int = 240,
    parent_modules_for_gen0: Path | None = None,
    skills_dir: Path | None = None,
    review_max_turns: int = 60,
    sanity_tasks: list[str] | None = None,
    environment: str | None = None,
    max_gate_repairs: int = 1,
    sanity_concurrency: int = 6,
    support_task_dir: Path | None = None,
    e2b_accounts: list[tuple[str, str]] | None = None,
    e2b_per_account_cap: int = 18,
    solver_temperature: float = 0.0,
    locked_module_type: str,
    composer_scope: str = "locked",
    attempts: int = 3,
    max_lanes: int = 2,
) -> OnlineEvoOutcome:
    """Run the online self-evolution loop.

    Args:
        archive_root: dir holding gen_N/ + working dirs.
        tasks: ordered list of task names from terminal-bench@2.0.
        reflect_every: trigger a reflection after every N completed tasks.
        task_concurrency: how many tasks to run in parallel via Harbor (each
            with `-n 1`). If None, defaults to `reflect_every` so that one
            batch = one reflection cycle (recommended).
        max_tasks: cap on tasks to run (None = run all in `tasks`).
        shuffle_tasks: shuffle the task queue before it is consumed so each
            reflection batch is category-mixed (default True). Disable to run in
            the exact order given by `tasks`.
        task_seed: RNG seed for the shuffle — fixed for reproducible, resume-safe
            ordering.
    """
    archive_root = Path(archive_root).resolve()
    archive_root.mkdir(parents=True, exist_ok=True)

    if max_lanes not in (1, 2):
        raise ValueError("max_lanes must be 1 or 2")
    if attempts <= 0:
        raise ValueError("attempts must be positive")

    # 1. Initialize the modules-only baseline generation.
    from harbor.agents.terminus_2_modular import modules as _modules_pkg

    if parent_modules_for_gen0 is None:
        parent_modules_for_gen0 = Path(_modules_pkg.__file__).parent  # type: ignore[arg-type]
    initialize_gen_0(archive_root, parent_modules_for_gen0)
    _seed_archive(archive_root)  # niche/genealogy registry (D3), from gen_0

    starting_gen = _find_latest_gen(archive_root)
    assert starting_gen is not None, "expected at least gen_0 after init"
    current_gen = starting_gen
    _logger.info("Starting from %s", current_gen)
    # Serial lineage: validate the inherited foundation BEFORE any task runs. A
    # broken pin degrades silently at roll level (see the function's docstring).
    _preflight_pin(archive_root, current_gen, locked_module_type)

    if max_tasks is None:
        max_tasks = len(tasks)
    task_subset = tasks[:max_tasks]

    if support_task_dir is not None:
        _logger.info(
            "TRAIN/EVAL SPLIT: support tasks from local dataset %s (%d tasks); "
            "probe/sanity/parent held-out on tb2 (terminal-bench@2.0) — the "
            "editor never sees a tb2 task, so the eval benchmark is uncontaminated",
            support_task_dir,
            len(task_subset),
        )

    # epochs>1: loop the SAME task subset N times within ONE lineage so
    # generations keep accumulating (89 tasks ×N, gen_0 → gen_many).
    # WARNING: re-reflecting on the same tasks every epoch overfits tb2 — only
    # legitimate for dev iteration; real generalization must come from a
    # separate tb1 held-out eval, never fed back into this loop.
    #
    # Shuffle the queue BEFORE it is consumed so each reflection batch (a window
    # of ~task_concurrency consecutive tasks) is category-mixed. Without this,
    # a task list grouped by category makes every batch single-category, and the
    # editor overfits its module edits to one failure mode. The rolling pool
    # only reorders completions within a ~2×concurrency window, so mixing must
    # come from the queue order itself. A fixed seed keeps runs reproducible and
    # resume-safe; each epoch is shuffled independently so batch composition
    # varies across epochs.
    epochs = max(1, epochs)
    if shuffle_tasks:
        import random

        rng = random.Random(task_seed)
        base_subset = task_subset
        task_subset = []
        for _ in range(epochs):
            epoch_tasks = list(base_subset)
            rng.shuffle(epoch_tasks)
            task_subset += epoch_tasks
    elif epochs > 1:
        task_subset = task_subset * epochs

    # Concurrency defaults to reflect_every (one parallel batch == one
    # reflection cycle, the recommended pattern).
    if task_concurrency is None:
        task_concurrency = reflect_every
    task_concurrency = max(1, task_concurrency)

    # e2b account pool: route each sandbox to a free account
    # slot so no single account exceeds its concurrency cap. Peak demand is
    # task_concurrency + sanity_concurrency (solver pool + sanity battery);
    # need capacity >= that or slot() back-pressures (blocks, never 429s).
    e2b_pool = (
        AccountPool(e2b_accounts, e2b_per_account_cap)
        if e2b_accounts
        else NullAccountPool()
    )
    if e2b_accounts:
        peak_demand = task_concurrency + sanity_concurrency
        _logger.info(
            "e2b account pool: %d accounts x cap %d = %d slots (peak demand "
            "task_concurrency+sanity_concurrency = %d+%d = %d)",
            e2b_pool.n_accounts,
            e2b_pool.per_account_cap,
            e2b_pool.capacity,
            task_concurrency,
            sanity_concurrency,
            peak_demand,
        )
        if e2b_pool.capacity < peak_demand:
            _logger.warning(
                "e2b pool capacity %d < peak demand %d — sandboxes will "
                "back-pressure (block on a free slot) during probe batteries; "
                "add accounts or lower task/probe concurrency.",
                e2b_pool.capacity,
                peak_demand,
            )

    records: list[TaskRunRecord] = []
    pending_summaries: list[TrialSummary] = []
    resumed_tasks_since_reflect = 0
    resumed_pending_anchor_idx: int | None = None
    n_promotions = 0

    # This archive_root IS one run (one lineage from gen_0). Solver outputs and
    # progress live directly under it — no extra run_<ts> layer.
    started_at = time.time()
    solver_root = archive_root / "solver"

    # ---- Resume (export ARCHIVE_ROOT at an existing run dir) ----------------
    # progress.json is authoritative: every task_idx already recorded there is
    # skipped because its solver run happened. Entries after the latest record
    # carrying a reflection outcome are restored into pending_summaries so
    # background-solver work is not silently treated as already digested. The
    # prior entries are preserved as a prefix of every subsequent progress.json
    # write. idx↔task alignment holds because
    # the queue order is reproducible (fixed shuffle seed); on any mismatch
    # (changed task list / seed) the old file is moved aside and nothing is
    # skipped — never silently mix two different task orders.
    prior_progress: list[dict] = []
    completed_idx: set[int] = set()
    progress_path = archive_root / "progress.json"
    if progress_path.exists():
        try:
            prior_progress = json.loads(progress_path.read_text())
        except Exception as exc:
            _logger.warning("prior progress.json unreadable (%s) — ignoring", exc)
            prior_progress = []
        aligned = True
        for e in prior_progress:
            try:
                i = int(e["task_idx"])
            except Exception:
                aligned = False
                break
            if not (0 <= i < len(task_subset) and task_subset[i] == e.get("task_name")):
                aligned = False
                break
            completed_idx.add(i)
        if not aligned:
            bak = archive_root / f"progress.json.bak_{int(started_at)}"
            progress_path.rename(bak)
            _logger.warning(
                "prior progress.json does not match the current task order — "
                "moved to %s; resuming with no skips",
                bak.name,
            )
            prior_progress = []
            completed_idx = set()
        elif completed_idx:
            _logger.info(
                "resume: %d/%d task(s) already recorded in progress.json — "
                "skipping them",
                len(completed_idx),
                len(task_subset),
            )
            (
                restored_pending,
                resumed_tasks_since_reflect,
                resumed_pending_anchor_idx,
            ) = _restore_unreflected_progress(prior_progress)
            pending_summaries.extend(restored_pending)
            if resumed_pending_anchor_idx is not None:
                _logger.info(
                    "resume: restored %d unreflected summary roll(s) across "
                    "%d completed task group(s); they will seed the next reflection",
                    len(restored_pending),
                    resumed_tasks_since_reflect,
                )

    # Run metadata: write config now (status=running); merge result at the end.
    run_meta: dict = {
        "run_id": archive_root.name,
        "status": "running",
        "started_at": started_at,
        "model": model_name,
        "model_info": json.loads(DEFAULT_MODEL_INFO),
        "n_tasks_planned": len(task_subset),
        "resumed_completed": len(completed_idx),
        "epochs": epochs,
        "shuffle_tasks": shuffle_tasks,
        "task_seed": task_seed,
        "reflect_every": reflect_every,
        "task_concurrency": task_concurrency,
        "max_tasks": max_tasks,
        "max_solver_turns": max_solver_turns,
        "solver_timeout_sec": solver_timeout_sec,
        "agent_timeout_multiplier": agent_timeout_multiplier,
        "review_gate": True,
        "review_max_turns": review_max_turns,
        "sanity_gate": True,
        "sanity_tasks": sanity_tasks or DEFAULT_SANITY_TASKS,
        "workflow": "phase0-two-lane",
        "max_lanes": max_lanes,
        "locked_module": locked_module_type,
        "composer_scope": composer_scope,
        "attempts": attempts,
        "solver_temperature": solver_temperature,
        "starting_gen": starting_gen.name,
        "sanity_concurrency": sanity_concurrency,
        "support_task_dir": str(support_task_dir) if support_task_dir else None,
    }
    _write_meta(archive_root, run_meta)

    # ---- Streaming pool (rolling reflection) --------------------------------
    # Instead of rigid batches that barrier on the SLOWEST task, keep up to
    # `task_concurrency` solver tasks in flight at once, pulling from the queue
    # as each finishes. Reflect every `reflect_every` COMPLETED tasks. A slow
    # task (e.g. one that hits the solver timeout) only occupies its own slot —
    # the other slots keep churning through the queue instead of idling.
    #
    # A reflection runs as a BACKGROUND task: the pool keeps launching new
    # solver tasks while the reflection chain (lenses → consolidate → review →
    # probe battery) is in flight, so the ~1-2h gate wall-clock no longer
    # stalls solver throughput. One reflection at a time (mutex): while one is
    # running, completions keep accumulating in `pending_summaries`, and the
    # next window fires as soon as the current reflection returns — windows
    # therefore GROW past `reflect_every` when reflections are slower than the
    # solver, which is the intended back-pressure.
    #
    # The gen a task runs on is captured at LAUNCH (the latest promoted gen at
    # that moment). A straggler that finishes after a later promotion folds its
    # trajectory into the current reflection window. This is benign: evolution
    # is additive (the baseline variants it ran still exist in the newer gen),
    # and each trajectory records its own `bundle` so the editor sees exactly
    # which variants produced it. The gen lineage stays linear — only the
    # reflection SIGNAL is occasionally cross-gen, never the parent chain.
    #
    # NOTE on e2b/quota: with reflections overlapping the pool, SUSTAINED
    # concurrency is task_concurrency + sanity_concurrency sandboxes. Keep
    # their sum under the endpoint/e2b
    # ceiling.
    queue: list[tuple[int, str]] = [
        (i, t) for i, t in enumerate(task_subset) if i not in completed_idx
    ]
    qi = 0
    total = len(queue)
    inflight: dict[asyncio.Task, tuple[int, str, Path]] = {}

    # Size the thread pool for the full solver pool + threads the reflection
    # (sanity gate) spins up concurrently, so a saturated pool never starves.
    try:
        import concurrent.futures

        asyncio.get_running_loop().set_default_executor(
            concurrent.futures.ThreadPoolExecutor(
                max_workers=max(32, task_concurrency + sanity_concurrency + 8)
            )
        )
    except Exception:  # pragma: no cover - best effort
        pass

    async def _run_one(idx: int, task: str, gen: Path):
        # Support tasks come from the local training dataset when
        # support_task_dir is set — run via `-p <dir>/<task>`. Probe/sanity/
        # parent tasks stay on tb2 (task_dir=None → `-d terminal-bench@2.0`).
        support_task_path = (
            (support_task_dir / task) if support_task_dir is not None else None
        )
        # K-roll: roll 0 performs the ONE per-task dynamic composition. The
        # shared helper captures that bundle in a private modules snapshot and
        # forces rolls 1..K-1 through StaticComposer. Rolls remain sequential;
        # each holds one e2b slot at a time, so sandbox concurrency stays near
        # task_concurrency rather than multiplying by K.
        group_root = solver_root / f"task_{idx:03d}_{task}"

        async def _run_roll(
            j: int,
            load_root: Path,
            composer_name: str,
            out_dir: Path,
        ) -> HarborTaskResult:
            async with e2b_pool.slot() as s:
                return await asyncio.to_thread(
                    run_harbor_task,
                    task=task,
                    **_gen_run_kwargs(load_root),
                    model_name=model_name,
                    api_base=api_base,
                    api_key=api_key,
                    output_dir=out_dir,
                    max_turns=max_solver_turns,
                    timeout_sec=solver_timeout_sec,
                    agent_timeout_multiplier=agent_timeout_multiplier,
                    environment=environment,
                    task_dir=support_task_path,
                    e2b_key=s.key,
                    e2b_token=s.token,
                    temperature=solver_temperature,
                    locked_module_type=locked_module_type,
                    composer_scope=composer_scope,
                    composer_name=composer_name,
                )

        group = await run_same_bundle_k_rolls(
            attempts=attempts,
            source_root=gen,
            group_root=group_root,
            run_roll=_run_roll,
        )
        return idx, task, gen, group

    async def _do_reflect(
        batch: list[TrialSummary],
        anchor_rec: TaskRunRecord | None = None,
        anchor_prior: dict | None = None,
    ) -> None:
        nonlocal current_gen, n_promotions
        # Real training epoch of the triggering task = global idx // one-epoch
        # length (task_subset = base * epochs; shuffle preserves length). Used by
        # the router cooldown clock, the cross-epoch confirm, and editor-memory.
        # Use the full planned queue, not ``total`` (which is only the number
        # of tasks left after resume).  Otherwise a late resume shrinks the
        # epoch width and corrupts router cooldown / cross-epoch bookkeeping.
        anchor_task_idx = (
            anchor_rec.task_idx
            if anchor_rec is not None
            else int((anchor_prior or {}).get("task_idx", 0))
        )
        epoch_now = anchor_task_idx // max(1, len(task_subset) // max(1, epochs))
        groups_at_start = len(records)
        outcome = await _maybe_reflect(
            archive_root=archive_root,
            current_gen=current_gen,
            pending_summaries=batch,
            max_lanes=max_lanes,
            model_name=model_name,
            api_base=api_base,
            api_key=api_key,
            max_editor_turns=max_editor_turns,
            skills_dir=skills_dir,
            review_max_turns=review_max_turns,
            sanity_tasks=sanity_tasks,
            environment=environment,
            max_solver_turns=max_solver_turns,
            solver_timeout_sec=solver_timeout_sec,
            editor_timeout_sec=editor_timeout_sec,
            editor_call_timeout_sec=editor_call_timeout_sec,
            max_gate_repairs=max_gate_repairs,
            sanity_concurrency=sanity_concurrency,
            e2b_pool=e2b_pool,
            locked_module_type=locked_module_type,
            epoch=epoch_now,
        )
        # Measure how much solver work completed on the old generation while
        # reflection ran.
        outcome.solver_groups_during_reflection = len(records) - groups_at_start
        if reflect_every > 0 and outcome.solver_groups_during_reflection > 0:
            _logger.info(
                "reflection spanned %d solver group(s) (~%.1f batch(es) of %d)",
                outcome.solver_groups_during_reflection,
                outcome.solver_groups_during_reflection / reflect_every,
                reflect_every,
            )
        # Anchor the reflection record to the task whose completion TRIGGERED
        # this window (captured at spawn time — with the pool still launching
        # during the reflection, records[-1] at return time is arbitrary).
        if anchor_rec is not None:
            anchor_rec.reflection = outcome
        elif anchor_prior is not None:
            anchor_prior["reflection"] = dataclasses.asdict(outcome)
        elif records:
            records[-1].reflection = outcome
        if outcome.promoted_gen is not None:
            # the NEWEST gen: with two lanes, staying on the first would keep
            # the solver on a tree the archive has already moved past
            current_gen = outcome.promoted_gen
            n_promotions += _promotion_count(outcome)
        # Cross-epoch confirm / rollback (S8): using the FREE per-variant pass
        # tally accumulated in the records so far, retire any locked-module variant
        # that is clearly worse than the gen_0 baseline on shared tasks. Best-effort
        # — never kills the run. Only meaningful in the contrastive single-module
        # experiment (there is a well-defined baseline to compare against).
        if locked_module_type:
            # Editor-memory: log THIS reflection's change (if it promoted) as
            # provisional, so the next consolidator sees what was tried.
            _remember_promotions(
                archive_root, outcome, epoch=epoch_now, module=locked_module_type
            )
            try:
                from harbor.agents.terminus_2_modular.composer.static import (
                    DEFAULT_BUNDLE,
                )
                from harbor.agents.terminus_2_modular.self_evo import (
                    confirm as _confirm,
                )

                baseline_v = getattr(DEFAULT_BUNDLE, locked_module_type).name
                rep = _confirm.confirm_and_rollback(
                    archive_root, records, locked_module_type, baseline_v
                )
                for v in rep["rolled_back"]:
                    _logger.info(
                        "confirm: rolled back regressing %s variant → superseded: %s",
                        locked_module_type,
                        v,
                    )
                    # Taboo it so the consolidator won't re-propose the same change.
                    editor_memory.record(
                        archive_root,
                        epoch=epoch_now,
                        module=locked_module_type,
                        task=v,
                        change=f"variant '{v}' regressed vs baseline on shared tasks",
                        verdict="rolled_back",
                    )
            except Exception as exc:
                _logger.warning("confirm/rollback failed (non-fatal): %s", exc)

            # Same confirm idea for solver helpers, but keyed on presence rather
            # than on a variant name: a roll records the LIST of helpers it was
            # given, so we compare "this helper was in hand" against "it was not"
            # on the same tasks. A helper that makes things clearly worse is
            # superseded and the Composer stops handing it out.
            try:
                from harbor.agents.terminus_2_modular.self_evo import (
                    confirm as _confirm,
                )

                hrep = _confirm.confirm_and_rollback_helpers(archive_root, records)
                for h in hrep["rolled_back"]:
                    _logger.info(
                        "confirm: rolled back regressing solver helper → "
                        "superseded: %s",
                        h,
                    )
                    editor_memory.record(
                        archive_root,
                        epoch=epoch_now,
                        module=_confirm.HELPER_TYPE,
                        task=h,
                        change=(
                            f"solver helper '{h}' did worse when in hand than "
                            f"when absent, on the same tasks"
                        ),
                        verdict="rolled_back",
                    )
            except Exception as exc:
                _logger.warning("helper confirm/rollback failed (non-fatal): %s", exc)

    reflect_task: asyncio.Task | None = None
    # Reflection cadence counts TASKS, not rolls: with K-roll a completed task
    # appends K summaries to pending_summaries, so triggering on len(summaries)
    # would fire K× too often. Count distinct tasks completed since the last
    # reflection instead.
    tasks_since_reflect = resumed_tasks_since_reflect
    if tasks_since_reflect >= reflect_every:
        batch = pending_summaries
        pending_summaries = []
        tasks_since_reflect = 0
        prior_anchor = (
            prior_progress[resumed_pending_anchor_idx]
            if resumed_pending_anchor_idx is not None
            else None
        )
        reflect_task = asyncio.create_task(
            _do_reflect(batch, anchor_prior=prior_anchor)
        )
        _logger.info(
            "resume: immediately restarted reflection for %d restored summary roll(s)",
            len(batch),
        )

    while qi < total or inflight or reflect_task is not None:
        # Keep filling the pool while reflection runs; the generation used by
        # each task is captured at launch and recorded with its trajectory.
        while len(inflight) < task_concurrency and qi < total:
            idx, task = queue[qi]
            qi += 1
            t = asyncio.create_task(_run_one(idx, task, current_gen))
            inflight[t] = (idx, task, current_gen)
            _logger.info(
                "launch %d/%d %s (gen=%s, pool=%d/%d)",
                idx + 1,
                total,
                task,
                current_gen.name,
                len(inflight),
                task_concurrency,
            )
        wait_set: set[asyncio.Task] = set(inflight)
        if reflect_task is not None:
            wait_set.add(reflect_task)
        if not wait_set:
            break
        done, _ = await asyncio.wait(wait_set, return_when=asyncio.FIRST_COMPLETED)
        if reflect_task is not None and reflect_task in done:
            done.discard(reflect_task)
            await reflect_task  # propagate a reflection crash (same as before)
            reflect_task = None
            _save_progress(archive_root, records, prior=prior_progress)
        for t in done:
            idx, task, gen_used, group = t.result()
            del inflight[t]
            # One completed task = K rolls. Each roll gets its own record +
            # summary (all sharing task_idx/task/gen, distinguished by `roll`),
            # so the same-task pass/fail spread is preserved in the batch.
            roll_rewards: list[float | None] = []
            for j, (hr, out_dir) in enumerate(group.rolls):
                summary = _harbor_results_to_summary(out_dir)
                records.append(
                    TaskRunRecord(
                        task_idx=idx,
                        task_name=task,
                        gen_used=gen_used,
                        reward=hr.reward,
                        error=hr.error,
                        trial_dir=out_dir,
                        summary=summary,
                        roll=j,
                        bundle_id=group.bundle_id,
                        k_group_invalid_reason=group.invalid_reason,
                    )
                )
                if group.valid and summary is not None:
                    pending_summaries.append(summary)
                elif group.valid and hr.error:
                    pending_summaries.append(
                        TrialSummary(
                            task_name=task,
                            trial_name=f"failed_{idx}_r{j}",
                            reward=None,
                            exception_type=None,
                            exception_message=hr.error,
                            n_episodes=0,
                            n_input_tokens=0,
                            n_output_tokens=0,
                            last_step_messages=[],
                            failure_signals=["harbor_error"],
                            trial_dir=str(out_dir),
                        )
                    )
                roll_rewards.append(hr.reward)
            if group.valid:
                tasks_since_reflect += 1
            else:
                _logger.error(
                    "invalid K-roll group for task %s (gen=%s): %s; "
                    "excluding all %d available roll(s) from reflection",
                    task,
                    gen_used.name,
                    group.invalid_reason,
                    len(group.rolls),
                )
            _logger.info(
                "done task %s rewards=%s bundle=%s valid=%s "
                "(pool=%d, pending=%d rolls / "
                "%d tasks since reflect, gen=%s)",
                task,
                roll_rewards,
                group.bundle_id,
                group.valid,
                len(inflight),
                len(pending_summaries),
                tasks_since_reflect,
                gen_used.name,
            )
        _save_progress(archive_root, records, prior=prior_progress)
        # Rolling reflection: every `reflect_every` COMPLETED TASKS (not rolls),
        # run in the BACKGROUND so the pool keeps launching. One at a time
        # (mutex): while a reflection is in flight, completions accumulate and
        # the next window fires as soon as it returns.
        if reflect_task is None and tasks_since_reflect >= reflect_every:
            batch = pending_summaries
            pending_summaries = []
            tasks_since_reflect = 0
            anchor = records[-1] if records else None
            reflect_task = asyncio.create_task(_do_reflect(batch, anchor))

    # Final reflection on the tail (the last partial window).
    if pending_summaries:
        anchor_rec = records[-1] if records else None
        prior_anchor = (
            prior_progress[resumed_pending_anchor_idx]
            if anchor_rec is None and resumed_pending_anchor_idx is not None
            else None
        )
        await _do_reflect(
            pending_summaries,
            anchor_rec=anchor_rec,
            anchor_prior=prior_anchor,
        )
        pending_summaries = []
        _save_progress(archive_root, records, prior=prior_progress)

    run_meta.update(
        {
            "status": "completed",
            "finished_at": time.time(),
            "final_gen": current_gen.name,
            "n_tasks": len(records),
            "n_promotions": n_promotions,
        }
    )
    _write_meta(archive_root, run_meta)

    return OnlineEvoOutcome(
        archive_root=archive_root,
        starting_gen=starting_gen,
        final_gen=current_gen,
        n_tasks=len(records),
        n_promotions=n_promotions,
        records=records,
    )


def _write_meta(archive_root: Path, meta: dict) -> None:
    """Write run-level metadata (config + result summary) to meta.json."""
    try:
        (archive_root / "meta.json").write_text(json.dumps(meta, indent=2))
    except Exception as exc:
        _logger.warning("failed to write meta.json: %s", exc)


def _save_progress(
    archive_root: Path,
    records: list[TaskRunRecord],
    prior: list[dict] | None = None,
) -> None:
    """Dump intermediate state so a crash doesn't lose data.

    `prior` carries the raw entries of a resumed run's earlier progress.json;
    they are preserved verbatim as a prefix of every write.
    """
    out_path = archive_root / "progress.json"

    def _to_jsonable(o):
        if dataclasses.is_dataclass(o) and not isinstance(o, type):
            return dataclasses.asdict(o)
        if isinstance(o, Path):
            return str(o)
        return str(o)

    try:
        out_path.write_text(
            json.dumps(
                list(prior or [])
                + [
                    {
                        "task_idx": r.task_idx,
                        "task_name": r.task_name,
                        "gen_used": str(r.gen_used),
                        "reward": r.reward,
                        "error": r.error,
                        "roll": r.roll,
                        "bundle_id": r.bundle_id,
                        "k_group_invalid_reason": r.k_group_invalid_reason,
                        "trial_dir": str(r.trial_dir) if r.trial_dir else None,
                        "summary": dataclasses.asdict(r.summary) if r.summary else None,
                        "reflection": dataclasses.asdict(r.reflection)
                        if r.reflection
                        else None,
                    }
                    for r in records
                ],
                indent=2,
                default=_to_jsonable,
            )
        )
    except Exception as exc:
        _logger.warning("failed to save progress: %s", exc)
