"""P1-6: would the real composer ever pick this variant?

P0-5 proves a new variant *runs* — but it does that with a private pin, which
bypasses DESCRIPTION-based routing entirely. So "the sanity battery exercised it"
and "a normal solver would ever select it" are different claims, and only the
first one was ever checked. That gap is how `crash_semantics_aware` gets promoted
on the strength of a forced run and then never gets chosen again.

This asks the second question directly, and cheaply: the per-task composer's
choice depends only on the task instruction and the variants' DESCRIPTIONs, so
one LLM call per task answers it — no container, no agent, no benchmark.

The one rule that matters: **`routing_unproven` must never kill a candidate.**
Rewriting a DESCRIPTION moved a variant's selection rate from 2.8% to 49.4%.
"nothing picks it" is at least as likely to mean the description is written badly
as it is to mean the variant is useless, so the remedy is to rewrite and re-probe.
"""

import pytest

from harbor.agents.terminus_2_modular.self_evo import routing_probe

pytestmark = pytest.mark.unit

TASKS = ["fix-git", "posix-tar-r-w"]
INSTRUCTIONS = {"fix-git": "repair the repo", "posix-tar-r-w": "write a tar"}


def _chooser(picks):
    """picks: instruction → variant name the composer would select."""

    async def _choose(instruction):
        got = picks.get(instruction)
        if isinstance(got, Exception):
            raise got
        return got

    return _choose


async def _probe(picks, **over):
    kw = dict(
        target_variant="plan_first",
        tasks=TASKS,
        instructions=INSTRUCTIONS,
        choose=_chooser(picks),
    )
    kw.update(over)
    return await routing_probe.probe_routing(**kw)


# ---- the verdict ----------------------------------------------------------


async def test_one_intended_task_picking_it_is_enough():
    got = await _probe({"repair the repo": "plan_first", "write a tar": "baseline"})
    assert got.verdict == "routing_proven"
    assert got.picked_by == ["fix-git"]


async def test_nothing_picking_it_is_unproven():
    got = await _probe({"repair the repo": "baseline", "write a tar": "baseline"})
    assert got.verdict == "routing_unproven"
    assert got.picked_by == []


async def test_the_per_task_picks_are_kept_for_audit():
    got = await _probe({"repair the repo": "plan_first", "write a tar": "baseline"})
    assert got.picks == {"fix-git": "plan_first", "posix-tar-r-w": "baseline"}


# ---- unproven is a remedy, not a rejection -------------------------------


async def test_unproven_carries_a_rewrite_remedy():
    got = await _probe({"repair the repo": "baseline", "write a tar": "baseline"})
    assert "description" in got.remedy.lower()
    assert got.remedy  # something actionable, not an empty verdict


async def test_proven_needs_no_remedy():
    got = await _probe({"repair the repo": "plan_first", "write a tar": "baseline"})
    assert got.remedy == ""


def test_the_result_exposes_no_pass_fail_gate():
    # a boolean here is an invitation to wire it into a discard. The only way to
    # act on this result is to read the verdict and decide deliberately.
    fields = set(routing_probe.RoutingProbeResult.__dataclass_fields__)
    assert not fields & {"passed", "should_discard", "ok", "rejected"}


def test_the_module_cannot_discard_anything():
    import inspect

    source = inspect.getsource(routing_probe)
    assert "discard_staging" not in source


# ---- absence of evidence is not evidence ---------------------------------


async def test_no_intended_tasks_is_skipped_not_unproven():
    got = await _probe({}, tasks=[])
    assert got.verdict == "routing_skipped"


async def test_tasks_with_no_cached_instruction_are_skipped():
    # the probe needs the task text; not having it says nothing about the variant
    got = await _probe({}, instructions={})
    assert got.verdict == "routing_skipped"
    assert "instruction" in got.reason.lower()


async def test_a_task_that_errors_does_not_sink_the_probe():
    got = await _probe(
        {"repair the repo": RuntimeError("endpoint"), "write a tar": "plan_first"}
    )
    assert got.verdict == "routing_proven"
    assert got.picked_by == ["posix-tar-r-w"]
    assert got.errors == 1


async def test_every_task_erroring_is_skipped_not_unproven():
    got = await _probe(
        {"repair the repo": RuntimeError("x"), "write a tar": RuntimeError("y")}
    )
    assert got.verdict == "routing_skipped"
    assert got.errors == 2


# ---- the candidate set has to match the real run -------------------------


def test_retired_variants_are_excluded(tmp_path):
    from harbor.agents.terminus_2_modular import archive as _archive

    _archive.save_archive(
        tmp_path,
        [
            _archive.ArchiveEntry(name="baseline", type="agent_loop", status="active"),
            _archive.ArchiveEntry(name="old", type="agent_loop", status="superseded"),
            _archive.ArchiveEntry(name="off", type="agent_loop", status="excluded"),
        ],
    )
    assert routing_probe.retired_quals(tmp_path) == frozenset(
        {"agent_loop/old", "agent_loop/off"}
    )


def test_no_archive_means_no_skips(tmp_path):
    # fail open: a probe with a wider candidate set is wrong, but a probe that
    # refuses to run tells us nothing at all
    assert routing_probe.retired_quals(tmp_path) == frozenset()
