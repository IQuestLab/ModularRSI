"""Cross-epoch confirm / rollback (S8) — the "free" fitness signal.

We do NOT re-run a dedicated reward battery (that expensive gate is off). Instead
we reuse the rolls the loop already does: every roll records WHICH variant of each
module type the composer picked (agent.extra.bundle → TrialSummary.bundle), plus
its reward. So over epochs a provisional variant of the locked module accumulates
a pass tally FOR FREE, and we compare it — paired on the SAME task — against the
baseline (gen_0) variant. A variant that is clearly worse on the tasks where BOTH
ran gets rolled back (status → superseded), restoring the parent.

Paired-on-task matters: the composer may pick a new variant mainly for hard tasks
where the baseline also fails, so a RAW pass-rate would be unfairly low. Comparing
only the tasks where both variants ran removes most of that selection bias. K is
small, so require a minimum number of shared tasks and a generous margin before
retiring anything — and trust that persistence across epochs, not one noisy point,
is what triggers a rollback.

Pure functions here are unit-testable with no archive/endpoint; `confirm_and_rollback`
is the thin glue that reads the run's records and flips archive status.
"""

from __future__ import annotations

import statistics
from collections import defaultdict


def _passed(reward) -> float:
    return 1.0 if (reward or 0) >= 1.0 else 0.0


def variant_pass_stats(samples, module_type: str) -> dict[str, dict]:
    """samples: iterable of (task, bundle: dict|None, reward). Returns per-variant
    {n_used, n_pass, pass_rate} for the given module type (for logging).

    reward=None rolls (infra/harbor errors) are skipped — an endpoint hiccup must
    not count against whichever variant happened to hit it."""
    stats: dict[str, dict] = {}
    for _task, bundle, reward in samples:
        if not bundle or reward is None:
            continue
        v = bundle.get(module_type)
        if not v:
            continue
        s = stats.setdefault(v, {"n_used": 0, "n_pass": 0})
        s["n_used"] += 1
        s["n_pass"] += int(_passed(reward))
    for s in stats.values():
        s["pass_rate"] = s["n_pass"] / s["n_used"] if s["n_used"] else 0.0
    return stats


def paired_delta(samples, module_type: str, variant: str, baseline: str) -> dict | None:
    """Same-task paired comparison of `variant` vs `baseline` for one module type.

    Returns {shared_tasks, variant_rate, baseline_rate, delta} over the tasks where
    BOTH variants ran (each task's per-variant pass-rate averaged, then averaged
    across shared tasks so every shared task weighs equally). None if no overlap.
    """
    tv: dict[tuple[str, str], list[float]] = defaultdict(list)
    for task, bundle, reward in samples:
        if not bundle or reward is None:  # skip infra-error rolls
            continue
        v = bundle.get(module_type)
        if v in (variant, baseline):
            tv[(task, v)].append(_passed(reward))
    tasks = {t for (t, _v) in tv}
    shared = [t for t in tasks if (t, variant) in tv and (t, baseline) in tv]
    if not shared:
        return None
    v_rate = statistics.mean(statistics.mean(tv[(t, variant)]) for t in shared)
    b_rate = statistics.mean(statistics.mean(tv[(t, baseline)]) for t in shared)
    return {
        "shared_tasks": len(shared),
        "variant_rate": v_rate,
        "baseline_rate": b_rate,
        "delta": v_rate - b_rate,
    }


def find_regressions(
    samples,
    module_type: str,
    baseline: str,
    *,
    candidates: list[str] | None = None,
    min_shared: int = 3,
    margin: float = 0.34,
) -> list[dict]:
    """Non-baseline variants that are worse than baseline on shared tasks by more
    than `margin`, with at least `min_shared` shared tasks. Returns the losing
    variants with their paired-delta detail."""
    stats = variant_pass_stats(samples, module_type)
    cands = candidates if candidates is not None else list(stats)
    out: list[dict] = []
    for v in cands:
        if v == baseline:
            continue
        d = paired_delta(samples, module_type, v, baseline)
        if d and d["shared_tasks"] >= min_shared and d["delta"] < -margin:
            out.append({"variant": v, **d})
    return out


# ---------------------------------------------------------------------------
# Solver helpers. Same idea, different arity: a roll records a LIST of the helpers
# it was given, not one winner, so the contrast is "this helper was in hand" vs
# "it was not" rather than "variant A" vs "variant B" — the equality matching
# above cannot express that.
#
# Where the contrast comes from: the Composer chooses a per-task SUBSET, so within
# one epoch some rolls of a task carry a helper and others do not. That is the
# clean signal. Rolls recorded before a helper existed also read as "absent" and
# are usable, but they are confounded (other modules changed across those
# generations too) — which is why the margin is deliberately generous and a
# minimum number of observations on BOTH sides is required before retiring
# anything. This gate is for catching a clearly harmful tool, not for ranking.
# ---------------------------------------------------------------------------

HELPER_TYPE = "tool_helper"


def _helper_set(bundle) -> set[str] | None:
    """The helpers a roll was given. None = this roll predates helper recording
    (no key at all), so it says nothing either way and must be skipped."""
    if not bundle or HELPER_TYPE not in bundle:
        return None
    raw = bundle.get(HELPER_TYPE)
    if not isinstance(raw, (list, tuple, set)):
        return None
    return {str(x) for x in raw}


def helper_pass_stats(samples) -> dict[str, dict]:
    """Per-helper {n_present, n_pass_present, present_rate} for logging."""
    stats: dict[str, dict] = {}
    for _task, bundle, reward in samples:
        hs = _helper_set(bundle)
        if hs is None or reward is None:
            continue
        for h in hs:
            s = stats.setdefault(h, {"n_present": 0, "n_pass_present": 0})
            s["n_present"] += 1
            s["n_pass_present"] += int(_passed(reward))
    for s in stats.values():
        s["present_rate"] = (
            s["n_pass_present"] / s["n_present"] if s["n_present"] else 0.0
        )
    return stats


def helper_paired_delta(samples, helper: str) -> dict | None:
    """Same-task comparison of rolls WHERE `helper` was in hand vs where it was
    not. Mirrors `paired_delta`: average each task's per-side pass-rate, then
    average across the tasks that have BOTH sides. None if no task has both."""
    present: dict[str, list[float]] = defaultdict(list)
    absent: dict[str, list[float]] = defaultdict(list)
    for task, bundle, reward in samples:
        hs = _helper_set(bundle)
        if hs is None or reward is None:  # skip infra-error / pre-feature rolls
            continue
        (present if helper in hs else absent)[task].append(_passed(reward))
    shared = [t for t in present if t in absent]
    if not shared:
        return None
    p_rate = statistics.mean(statistics.mean(present[t]) for t in shared)
    a_rate = statistics.mean(statistics.mean(absent[t]) for t in shared)
    return {
        "shared_tasks": len(shared),
        "present_rate": p_rate,
        "absent_rate": a_rate,
        "delta": p_rate - a_rate,
    }


def find_helper_regressions(
    samples,
    *,
    candidates: list[str] | None = None,
    min_shared: int = 3,
    margin: float = 0.34,
) -> list[dict]:
    """Helpers that do WORSE when in hand than when absent, on the same tasks, by
    more than `margin`, with at least `min_shared` tasks having both sides."""
    stats = helper_pass_stats(samples)
    cands = candidates if candidates is not None else list(stats)
    out: list[dict] = []
    for h in cands:
        d = helper_paired_delta(samples, h)
        if d and d["shared_tasks"] >= min_shared and d["delta"] < -margin:
            out.append({"helper": h, **d})
    return out


def confirm_and_rollback_helpers(
    archive_root,
    records,
    *,
    min_shared: int = 3,
    margin: float = 0.34,
) -> dict:
    """Glue: find solver helpers that make things worse when in hand and flip them
    to `superseded`, so the Composer stops handing them out. Best-effort — any
    error is swallowed so confirm never kills the run."""
    from harbor.agents.terminus_2_modular import archive as _arch

    samples = _samples_from_records(records)
    stats = helper_pass_stats(samples)
    # Same live-only filter as the variant path — see the note there.
    active = _active_quals(archive_root, HELPER_TYPE)
    cands = [h for h in stats if active is None or h in active]
    regs = find_helper_regressions(
        samples, candidates=cands, min_shared=min_shared, margin=margin
    )
    rolled: list[str] = []
    for reg in regs:
        try:
            if _arch.set_status(
                archive_root, f"{HELPER_TYPE}/{reg['helper']}", "superseded"
            ):
                rolled.append(reg["helper"])
        except Exception:
            continue
    return {"stats": stats, "regressions": regs, "rolled_back": rolled}


def _active_quals(archive_root, type_: str) -> set[str] | None:
    """Names of `type_` that are still live in the archive. None if unreadable —
    callers then skip filtering rather than silently rolling nothing back."""
    try:
        from harbor.agents.terminus_2_modular import archive as _arch

        return {
            e.name
            for e in _arch.load_archive(archive_root)
            if e.type == type_ and e.status == "active"
        }
    except Exception:
        return None


def _samples_from_records(records) -> list[tuple[str, dict | None, float | None]]:
    samples = []
    for r in records:
        summ = getattr(r, "summary", None)
        bundle = getattr(summ, "bundle", None) if summ is not None else None
        samples.append(
            (getattr(r, "task_name", ""), bundle, getattr(r, "reward", None))
        )
    return samples


def confirm_and_rollback(
    archive_root,
    records,
    module_type: str,
    baseline_variant: str,
    *,
    min_shared: int = 3,
    margin: float = 0.34,
) -> dict:
    """Glue: build (task, bundle, reward) samples from the run's records, find
    variants of `module_type` that regress vs `baseline_variant` on shared tasks,
    and flip them to `superseded` (rollback → parent restored). Best-effort: any
    error is swallowed so confirm never kills the run.
    """
    from harbor.agents.terminus_2_modular import archive as _arch

    samples = _samples_from_records(records)
    stats = variant_pass_stats(samples, module_type)
    # Only consider variants that are still LIVE. A retired one stops being
    # selected, but its old rolls stay in `records` forever, so it re-qualifies as
    # a regression on every later reflection — and `set_status` returns True
    # whenever it FINDS the entry (not only when it changes it), so each pass
    # appended another identical taboo to editor memory. Observed 5x for one
    # variant, which would crowd the memory read by future editors.
    active = _active_quals(archive_root, module_type)
    cands = [v for v in stats if active is None or v in active]
    regs = find_regressions(
        samples,
        module_type,
        baseline_variant,
        candidates=cands,
        min_shared=min_shared,
        margin=margin,
    )
    rolled: list[str] = []
    for reg in regs:
        try:
            if _arch.set_status(
                archive_root, f"{module_type}/{reg['variant']}", "superseded"
            ):
                rolled.append(reg["variant"])
        except Exception:
            continue
    return {
        "stats": stats,
        "baseline": baseline_variant,
        "regressions": regs,
        "rolled_back": rolled,
    }
