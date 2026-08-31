"""P2: the whole two-lane window, end to end.

Selection → two isolated implementations → per-lane gates → promote the first →
rebase the second onto it → re-run only the gates the rebase invalidated →
promote it too. One window, up to two generations, each still a single change.

The parts that are easy to get wrong here are all about **what happens to the
loser**:

* A lane that fails must leave its proposal *retryable*. `rejected_implementation`
  means a sound direction was built badly — the direction itself was never
  tested. Sending it straight to taboo is how a good direction gets exactly one
  chance and then becomes permanently unaskable.
* A lane the other one absorbed (both converged on the identical edit) is not a
  failure at all. It is `superseded`: the change exists, someone else's
  generation carries it, and there is nothing left to promote. Recording that as
  a rejected implementation would blame a direction that actually landed.
* A lane that never reached its implementer keeps its slot: still `open`, still
  holding its evidence, no attempt charged.

And the first lane, once promoted, is untouchable. It is already in the archive,
so nothing the second lane does — conflict, failed re-gate, crash — may undo it.
"""

import pytest

from harbor.agents.terminus_2_modular.self_evo import (
    backlog,
    dual_implement,
    proposals,
    staging as _staging,
    two_lane,
)

pytestmark = pytest.mark.unit


# ---- fixtures -------------------------------------------------------------


def _archive_with_gen0(tmp_path):
    """A modules-only gen_0 assembled the way the Phase0 runner does it."""
    seed = tmp_path / "seed"
    (seed / "modules" / "agent_loop").mkdir(parents=True)
    (seed / "modules" / "observation").mkdir(parents=True)
    (seed / "modules" / "agent_loop" / "confirm_exit.py").write_text(
        "DESCRIPTION = 'baseline'\n"
    )
    (seed / "modules" / "observation" / "plain.py").write_text(
        "DESCRIPTION = 'plain'\n"
    )
    archive = tmp_path / "archive"
    archive.mkdir()
    _staging.prepare_staging(seed / "modules", archive / "gen_0", fresh=True)
    return archive


def _proposal(archive, lane, *, task, text, target=""):
    fid = backlog.ingest_step(
        archive,
        {
            "step": f"w1_{lane}",
            "findings": [{"task": task, "is_culprit": True, "d": text}],
        },
    )[0].finding_id
    return proposals.create_proposal(
        archive,
        step="w1",
        lane=lane,
        action="modify" if lane == "incumbent" else "add",
        target_variant=target
        or ("agent_loop/confirm_exit" if lane == "incumbent" else ""),
        behavioral_delta=f"{lane} delta",
        finding_ids=[fid],
        support_tasks=[task],
    )


def _both_lanes(archive):
    return (
        _proposal(archive, "incumbent", task="t1", text="A happened"),
        _proposal(archive, "novelty", task="t2", text="B happened"),
    )


#: What each lane writes into its staging, keyed by lane.
_DISJOINT = {
    "incumbent": {"modules/agent_loop/confirm_exit.py": "DESCRIPTION = 'lane A'\n"},
    "novelty": {"modules/observation/pager.py": "DESCRIPTION = 'lane B'\n"},
}
_OVERLAPPING = {
    "incumbent": {"modules/agent_loop/confirm_exit.py": "DESCRIPTION = 'lane A'\n"},
    "novelty": {"modules/agent_loop/confirm_exit.py": "DESCRIPTION = 'lane B'\n"},
}


def _implementer(edits=None, *, fails=(), crashes=()):
    edits = _DISJOINT if edits is None else edits

    async def run(brief):
        if brief.lane in crashes:
            raise RuntimeError(f"{brief.lane} editor died")
        for rel, text in (edits.get(brief.lane) or {}).items():
            path = brief.staging_dir / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        return dual_implement.LaneResult(passed=brief.lane not in fails)

    return run


def _regate(fails=(), *, seen=None):
    async def run(*, dest, gates, lane, new_parent, prior, proposal_id=None):
        if seen is not None:
            seen.append(
                {
                    "dest": dest,
                    "gates": gates,
                    "lane": lane,
                    "new_parent": new_parent,
                    "prior": prior,
                    "proposal_id": proposal_id,
                }
            )
        return dual_implement.LaneResult(passed=lane not in fails)

    return run


async def _window(archive, **kwargs):
    kwargs.setdefault("run_lane", _implementer())
    kwargs.setdefault("rerun_gates", _regate())
    return await two_lane.run_window(
        archive, parent_gen=archive / "gen_0", step="w2", **kwargs
    )


def _gens(archive):
    return sorted(p.name for p in archive.iterdir() if p.name.startswith("gen_"))


def _staging_leftovers(archive):
    root = archive / "staging"
    return sorted(p.name for p in root.iterdir()) if root.is_dir() else []


# ---- nothing to do --------------------------------------------------------


async def test_no_open_proposal_means_no_generation(tmp_path):
    archive = _archive_with_gen0(tmp_path)
    outcome = await _window(archive)

    assert outcome.verdicts == []
    assert _gens(archive) == ["gen_0"]


async def test_max_lanes_one_builds_a_single_generation(tmp_path):
    archive = _archive_with_gen0(tmp_path)
    _both_lanes(archive)

    outcome = await _window(archive, max_lanes=1)

    assert len(outcome.promoted) == 1
    assert _gens(archive) == ["gen_0", "gen_1"]


# ---- both lanes land ------------------------------------------------------


async def test_two_passing_lanes_become_two_generations(tmp_path):
    archive = _archive_with_gen0(tmp_path)
    _both_lanes(archive)

    outcome = await _window(archive)

    assert _gens(archive) == ["gen_0", "gen_1", "gen_2"]
    assert [v.promoted_gen.name for v in outcome.promoted] == ["gen_1", "gen_2"]


async def test_the_incumbent_goes_first_by_default(tmp_path):
    archive = _archive_with_gen0(tmp_path)
    _both_lanes(archive)

    outcome = await _window(archive)

    assert [v.lane for v in outcome.promoted] == ["incumbent", "novelty"]


async def test_the_newest_generation_carries_both_changes(tmp_path):
    # the point of rebasing: gen_2 must not be gen_0 + lane B, it must be
    # gen_1 + lane B, or lane A is promoted into a tree nobody ever runs
    archive = _archive_with_gen0(tmp_path)
    _both_lanes(archive)

    await _window(archive)

    newest = archive / "gen_2"
    assert (newest / "modules/agent_loop/confirm_exit.py").read_text() == (
        "DESCRIPTION = 'lane A'\n"
    )
    assert (newest / "modules/observation/pager.py").read_text() == (
        "DESCRIPTION = 'lane B'\n"
    )


async def test_the_rebased_lane_only_re_runs_the_tree_wide_gates(tmp_path):
    archive = _archive_with_gen0(tmp_path)
    _both_lanes(archive)

    outcome = await _window(archive)

    second = outcome.promoted[1]
    assert set(second.gates_rerun) == {"smoke", "activation", "routing"}
    # only the review is reused — the expensive, false-reject-prone one
    assert set(second.gates_reused) == {"review"}


async def test_the_first_lane_re_runs_nothing(tmp_path):
    archive = _archive_with_gen0(tmp_path)
    _both_lanes(archive)

    outcome = await _window(archive)

    assert outcome.promoted[0].gates_rerun == ()


# ---- one lane loses -------------------------------------------------------


async def test_a_lane_that_failed_its_gates_does_not_stop_the_other(tmp_path):
    archive = _archive_with_gen0(tmp_path)
    _both_lanes(archive)

    outcome = await _window(archive, run_lane=_implementer(fails=("incumbent",)))

    assert [v.lane for v in outcome.promoted] == ["novelty"]
    assert _gens(archive) == ["gen_0", "gen_1"]


async def test_a_crashed_lane_does_not_stop_the_other(tmp_path):
    archive = _archive_with_gen0(tmp_path)
    _both_lanes(archive)

    outcome = await _window(archive, run_lane=_implementer(crashes=("novelty",)))

    assert [v.lane for v in outcome.promoted] == ["incumbent"]


async def test_both_lanes_failing_leaves_the_parent_untouched(tmp_path):
    archive = _archive_with_gen0(tmp_path)
    _both_lanes(archive)

    outcome = await _window(
        archive, run_lane=_implementer(fails=("incumbent", "novelty"))
    )

    assert outcome.promoted == []
    assert _gens(archive) == ["gen_0"]


async def test_a_lone_surviving_lane_needs_no_rebase(tmp_path):
    # nothing was promoted before it, so there is nothing to rebase onto
    archive = _archive_with_gen0(tmp_path)
    _both_lanes(archive)

    outcome = await _window(archive, run_lane=_implementer(fails=("incumbent",)))

    assert outcome.promoted[0].gates_rerun == ()


# ---- the rebase goes wrong ------------------------------------------------


async def test_a_rebase_conflict_drops_the_second_lane_only(tmp_path):
    archive = _archive_with_gen0(tmp_path)
    _both_lanes(archive)

    outcome = await _window(archive, run_lane=_implementer(_OVERLAPPING))

    assert [v.lane for v in outcome.promoted] == ["incumbent"]
    assert _gens(archive) == ["gen_0", "gen_1"]


async def test_a_rebase_conflict_never_undoes_the_promoted_lane(tmp_path):
    # gen_1 is already in the archive; lane B losing must not touch it
    archive = _archive_with_gen0(tmp_path)
    _both_lanes(archive)

    await _window(archive, run_lane=_implementer(_OVERLAPPING))

    assert (archive / "gen_1/modules/agent_loop/confirm_exit.py").read_text() == (
        "DESCRIPTION = 'lane A'\n"
    )


async def test_a_conflict_says_so_in_the_reason(tmp_path):
    archive = _archive_with_gen0(tmp_path)
    _both_lanes(archive)

    outcome = await _window(archive, run_lane=_implementer(_OVERLAPPING))

    dropped = next(v for v in outcome.verdicts if v.promoted_gen is None)
    assert "conflict" in dropped.reason
    assert "modules/agent_loop/confirm_exit.py" in dropped.reason


async def test_failing_the_re_run_gates_drops_only_the_second_lane(tmp_path):
    archive = _archive_with_gen0(tmp_path)
    _both_lanes(archive)

    outcome = await _window(archive, rerun_gates=_regate(fails=("novelty",)))

    assert [v.lane for v in outcome.promoted] == ["incumbent"]
    assert _gens(archive) == ["gen_0", "gen_1"]


# ---- the absorbed case ----------------------------------------------------


async def test_a_lane_absorbed_by_the_other_makes_no_empty_generation(tmp_path):
    # both lanes wrote the identical edit; promoting the second would add a
    # generation whose diff is empty
    archive = _archive_with_gen0(tmp_path)
    _both_lanes(archive)
    same = {
        "incumbent": {"modules/agent_loop/confirm_exit.py": "DESCRIPTION = 'same'\n"},
        "novelty": {"modules/agent_loop/confirm_exit.py": "DESCRIPTION = 'same'\n"},
    }

    outcome = await _window(archive, run_lane=_implementer(same))

    assert _gens(archive) == ["gen_0", "gen_1"]
    assert [v.lane for v in outcome.promoted] == ["incumbent"]


async def test_an_absorbed_direction_is_superseded_not_rejected(tmp_path):
    # the change exists — another generation carries it. Blaming the direction
    # for a build that actually landed would be a lie about what happened.
    archive = _archive_with_gen0(tmp_path)
    _, novelty = _both_lanes(archive)
    same = {
        "incumbent": {"modules/agent_loop/confirm_exit.py": "DESCRIPTION = 'same'\n"},
        "novelty": {"modules/agent_loop/confirm_exit.py": "DESCRIPTION = 'same'\n"},
    }

    await _window(archive, run_lane=_implementer(same))

    after = proposals.get_proposal(archive, novelty.proposal_id)
    assert after.state == proposals.SUPERSEDED


# ---- what happens to the proposals ---------------------------------------


async def test_a_promoted_lane_closes_its_proposal_as_promoted(tmp_path):
    archive = _archive_with_gen0(tmp_path)
    incumbent, novelty = _both_lanes(archive)

    await _window(archive)

    for p in (incumbent, novelty):
        assert proposals.get_proposal(archive, p.proposal_id).state == (
            proposals.PROMOTED
        )


async def test_a_failed_lane_leaves_its_direction_retryable(tmp_path):
    # NOT taboo: one botched build is not a refutation of the direction
    archive = _archive_with_gen0(tmp_path)
    incumbent, _ = _both_lanes(archive)

    await _window(archive, run_lane=_implementer(fails=("incumbent",)))

    after = proposals.get_proposal(archive, incumbent.proposal_id)
    assert after.state == proposals.REJECTED_IMPLEMENTATION
    assert after.attempts == 1


async def test_a_lane_that_never_reached_its_implementer_keeps_its_slot(tmp_path):
    archive = _archive_with_gen0(tmp_path)
    incumbent, _ = _both_lanes(archive)

    await two_lane.run_window(
        archive,
        parent_gen=tmp_path / "no_such_gen",
        step="w2",
        run_lane=_implementer(),
        rerun_gates=_regate(),
    )

    after = proposals.get_proposal(archive, incumbent.proposal_id)
    assert after.state == proposals.OPEN
    assert after.attempts == 0


# ---- archive hygiene ------------------------------------------------------


async def test_no_staging_is_left_behind(tmp_path):
    archive = _archive_with_gen0(tmp_path)
    _both_lanes(archive)

    await _window(archive)

    assert _staging_leftovers(archive) == []


async def test_a_dropped_lanes_staging_is_cleaned_up_too(tmp_path):
    archive = _archive_with_gen0(tmp_path)
    _both_lanes(archive)

    await _window(archive, run_lane=_implementer(fails=("novelty",)))

    assert _staging_leftovers(archive) == []


async def test_the_re_gate_is_told_which_tree_it_is_now_measured_against(tmp_path):
    # the rebased tree sits on gen_1, so its diff base is gen_1 — diffing it
    # against gen_0 would attribute the promoted lane's files to this one
    archive = _archive_with_gen0(tmp_path)
    _both_lanes(archive)
    seen = []

    await _window(archive, rerun_gates=_regate(seen=seen))

    (call,) = seen
    assert call["new_parent"] == archive / "gen_1"
    assert call["lane"] == "novelty"


async def test_the_re_gate_can_see_what_the_lane_already_produced(tmp_path):
    # re-gating must not re-run the editor; it needs the original result to
    # reuse the change-local verdict and the trajectory behind it
    archive = _archive_with_gen0(tmp_path)
    _both_lanes(archive)
    seen = []

    await _window(archive, rerun_gates=_regate(seen=seen))

    assert seen[0]["prior"] is not None


async def test_a_conflicting_rebase_leaves_no_rebase_tree_behind(tmp_path):
    # the rebase builds a whole copy of the new parent before it discovers the
    # conflict; dropping the lane must take that copy with it
    archive = _archive_with_gen0(tmp_path)
    _both_lanes(archive)

    await _window(archive, run_lane=_implementer(_OVERLAPPING))

    assert _staging_leftovers(archive) == []


async def test_a_failed_re_gate_leaves_no_rebase_tree_behind(tmp_path):
    archive = _archive_with_gen0(tmp_path)
    _both_lanes(archive)

    await _window(archive, rerun_gates=_regate(fails=("novelty",)))

    assert _staging_leftovers(archive) == []


async def test_a_verdict_carries_what_its_gates_produced(tmp_path):
    # the per-lane review/sanity verdicts live in the lane result; without a
    # channel out of here they cannot reach the evolution log, and both
    # generations would end up reported under one lane's verdicts
    archive = _archive_with_gen0(tmp_path)
    _both_lanes(archive)

    async def tagged(brief):
        for rel, text in _DISJOINT[brief.lane].items():
            path = brief.staging_dir / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        return dual_implement.LaneResult(passed=True, detail=f"gates:{brief.lane}")

    outcome = await _window(archive, run_lane=tagged)

    # the first lane promotes the tree it was gated on, so its own result
    # travels; the rebased lane's comes from the RE-gate instead (next test)
    details = {v.lane: v.detail for v in outcome.promoted}
    assert details["incumbent"] == "gates:incumbent"


async def test_a_rebased_verdict_carries_the_re_gate_result(tmp_path):
    # after a rebase the lane was judged again; the record must show THAT
    # judgement, not the pre-rebase one
    archive = _archive_with_gen0(tmp_path)
    _both_lanes(archive)

    async def regated(*, dest, gates, lane, new_parent, prior, proposal_id=None):
        return dual_implement.LaneResult(passed=True, detail=f"regate:{lane}")

    outcome = await _window(archive, rerun_gates=regated)

    assert outcome.promoted[1].detail == "regate:novelty"


async def test_each_verdict_names_its_proposal(tmp_path):
    archive = _archive_with_gen0(tmp_path)
    incumbent, novelty = _both_lanes(archive)

    outcome = await _window(archive)

    assert {v.proposal_id for v in outcome.verdicts} == {
        incumbent.proposal_id,
        novelty.proposal_id,
    }


# ---- the archive has to learn about every generation ----------------------
#
# `atomic_promote` puts a tree on disk; it does not tell the archive anything.
# Observed 2026-08-27 in run `…__p2_twolane_tools`: five promotions, and
# `archive.json` never changed from its gen_0 seeding — so `tools/raw_file_tools`
# and five `tool_helper/*` variants existed in the tree but not in the archive.
# That is not a cosmetic gap. `evidence_pass.archive_view` builds the router's
# candidate list FROM `archive.json`, so every later window was asked to route
# evidence about variants it could not be told existed: the incumbent lane
# could only ever name `tools/baseline`, and anything about the new variants
# had to come back as novelty. The evidence channel silently narrows to the
# generation the run started from.


def _recorder():
    seen = []

    def on_promote(promoted_gen, parent_gen, result):
        seen.append((promoted_gen.name, parent_gen.name, result))

    return seen, on_promote


async def test_every_promoted_generation_is_reported_for_archive_sync(tmp_path):
    archive = _archive_with_gen0(tmp_path)
    _both_lanes(archive)
    seen, on_promote = _recorder()

    out = await _window(archive, on_promote=on_promote)

    assert [g for g, _, _ in seen] == [v.promoted_gen.name for v in out.promoted]


async def test_the_first_lane_is_reported_against_the_window_parent(tmp_path):
    archive = _archive_with_gen0(tmp_path)
    _both_lanes(archive)
    seen, on_promote = _recorder()

    await _window(archive, on_promote=on_promote)

    assert seen[0][:2] == ("gen_1", "gen_0")


async def test_the_rebased_lane_is_reported_against_what_it_rebased_onto(tmp_path):
    """gen_2 sits on gen_1, so gen_1 is its parent — not the window's parent.

    Reporting gen_0 here would credit the first lane's files to the second
    lane's change, which is how a `modify` gets archived as an `add`.
    """
    archive = _archive_with_gen0(tmp_path)
    _both_lanes(archive)
    seen, on_promote = _recorder()

    await _window(archive, on_promote=on_promote)

    assert seen[1][:2] == ("gen_2", "gen_1")


async def test_a_failing_archive_sync_does_not_undo_a_promotion(tmp_path):
    """The tree is already in the archive; bookkeeping may not take it back."""
    archive = _archive_with_gen0(tmp_path)
    _both_lanes(archive)

    def explode(promoted_gen, parent_gen, result):
        raise RuntimeError("archive.json is unreadable")

    out = await _window(archive, on_promote=explode)

    assert [v.promoted_gen.name for v in out.promoted] == ["gen_1", "gen_2"]
    assert _gens(archive) == ["gen_0", "gen_1", "gen_2"]


# ---- H4: the window cleans up after the one that never came back ---------


async def test_a_window_recovers_proposals_stranded_by_a_dead_predecessor(tmp_path):
    """A previous window took a proposal and died before reporting a verdict.

    Nothing raised — the process was gone — so the in-process "staging failed →
    back to open" path never ran and the proposal is `selected`. The portfolio
    only picks `open`, so without recovery that direction is unaskable for the
    rest of the lineage.
    """
    from harbor.agents.terminus_2_modular.self_evo import lease

    archive = _archive_with_gen0(tmp_path)
    rec = _proposal(archive, "incumbent", task="t1", text="A happened")
    proposals.transition(archive, rec.proposal_id, step="w1", to=proposals.SELECTED)
    lease.record_selected(
        archive,
        rec.proposal_id,
        lane="incumbent",
        step="w1",
        staging_path=str(archive / "staging" / "incumbent"),
    )

    await _window(archive)
    after = proposals.get_proposal(archive, rec.proposal_id)
    # never launched → released untouched, and picked up again by THIS window
    assert after.attempts == 1
    assert after.state != proposals.SELECTED


async def test_a_stranded_proposal_whose_editor_had_launched_is_charged(tmp_path):
    from harbor.agents.terminus_2_modular.self_evo import lease

    archive = _archive_with_gen0(tmp_path)
    rec = _proposal(archive, "incumbent", task="t1", text="A happened")
    proposals.transition(archive, rec.proposal_id, step="w1", to=proposals.SELECTED)
    lease.record_selected(
        archive,
        rec.proposal_id,
        lane="incumbent",
        step="w1",
        staging_path=str(archive / "staging" / "incumbent"),
    )
    lease.record_attempt_started(archive, rec.proposal_id, step="w1")

    await _window(archive)
    after = proposals.get_proposal(archive, rec.proposal_id)
    # the editor ran once and was lost; the attempt is charged and the direction
    # is not selectable again until something reopens it
    assert after.state == proposals.REJECTED_IMPLEMENTATION
    assert after.attempts == 1


async def test_recovery_happens_before_selection_not_after(tmp_path):
    # recovering afterwards is the same as not recovering: the portfolio has
    # already read the backlog and skipped the stranded proposal
    import inspect

    source = inspect.getsource(two_lane.run_window)
    assert source.index("recover(") < source.index("select_portfolio(")


# ---- H1: what KIND of reject it was decides what happens next -------------
#
# Every lane that missed the promotion plan was closed the same way —
# `rejected_implementation` — no matter what the reviewer actually said. Two
# things were lost with it. A reviewer that judged the DIRECTION wrong was
# recorded as a botched build, so a bad direction kept its retry. And a reviewer
# that judged the direction right and the CODE wrong wrote a `repair_brief` that
# nothing ever read: no second implementer, no return to `open`, and the
# portfolio only picks `open` — so the direction was unaskable forever.


class _Reviewed:
    """A lane result carrying the structured review verdict, as the real one does."""

    def __init__(self, reject_class="none", repair_brief=""):
        self.review_reject_class = reject_class
        self.review_repair_brief = repair_brief


def _rejecting(reject_class, *, repair_brief="", passes_on_retry=False):
    """An implementer whose gates reject the way the reviewer says."""
    calls = []

    async def run(brief):
        calls.append(brief)
        first = len(calls) == 1
        path = brief.staging_dir / "modules" / "agent_loop" / "confirm_exit.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"DESCRIPTION = 'attempt {len(calls)}'\n")
        if not first and passes_on_retry:
            return dual_implement.LaneResult(passed=True, detail=_Reviewed())
        return dual_implement.LaneResult(
            passed=False, detail=_Reviewed(reject_class, repair_brief)
        )

    run.calls = calls
    return run


async def test_a_rejected_direction_is_not_recorded_as_a_botched_build(tmp_path):
    archive = _archive_with_gen0(tmp_path)
    rec = _proposal(archive, "incumbent", task="t1", text="A happened")
    await _window(archive, run_lane=_rejecting("proposal"))
    assert proposals.get_proposal(archive, rec.proposal_id).state == (
        proposals.REJECTED_PROPOSAL
    )


async def test_a_rejected_direction_gets_no_second_implementer(tmp_path):
    # a second build of a direction the reviewer called wrong is spend with no
    # hypothesis behind it
    archive = _archive_with_gen0(tmp_path)
    _proposal(archive, "incumbent", task="t1", text="A happened")
    run = _rejecting("proposal")
    await _window(archive, run_lane=run)
    assert len(run.calls) == 1


async def test_flawed_code_on_a_sound_direction_gets_a_second_implementer(tmp_path):
    archive = _archive_with_gen0(tmp_path)
    _proposal(archive, "incumbent", task="t1", text="A happened")
    run = _rejecting("implementation", repair_brief="confirm_exit.py drops the guard")
    await _window(archive, run_lane=run)
    assert len(run.calls) == 2
    assert "confirm_exit.py drops the guard" in run.calls[1].instruction


async def test_the_second_implementer_can_still_win_the_generation(tmp_path):
    archive = _archive_with_gen0(tmp_path)
    rec = _proposal(archive, "incumbent", task="t1", text="A happened")
    out = await _window(
        archive,
        run_lane=_rejecting(
            "implementation", repair_brief="fix it", passes_on_retry=True
        ),
    )
    assert [v.promoted_gen.name for v in out.promoted] == ["gen_1"]
    assert proposals.get_proposal(archive, rec.proposal_id).state == (
        proposals.PROMOTED
    )


async def test_a_second_failed_implementation_ends_the_direction(tmp_path):
    # this is where taboo is finally earned: not one bad build, two
    archive = _archive_with_gen0(tmp_path)
    rec = _proposal(archive, "incumbent", task="t1", text="A happened")
    await _window(archive, run_lane=_rejecting("implementation", repair_brief="fix it"))
    after = proposals.get_proposal(archive, rec.proposal_id)
    assert after.attempts == 2
    assert after.state == proposals.TABOO


async def test_a_failure_the_reviewer_never_classified_keeps_the_old_behaviour(
    tmp_path,
):
    # smoke and sanity failures carry no reject_class — and smoke already has
    # its own repair loop, so a second implementer here would be paying twice
    archive = _archive_with_gen0(tmp_path)
    rec = _proposal(archive, "incumbent", task="t1", text="A happened")
    run = _rejecting("none")
    await _window(archive, run_lane=run)
    assert len(run.calls) == 1
    assert proposals.get_proposal(archive, rec.proposal_id).state == (
        proposals.REJECTED_IMPLEMENTATION
    )


async def test_the_retry_does_not_take_the_other_lanes_slot(tmp_path):
    # the rewrite is triggered directly, not by re-entering the portfolio: it
    # must not cost the second lane its place
    archive = _archive_with_gen0(tmp_path)
    _both_lanes(archive)
    run = _rejecting("implementation", repair_brief="fix it")
    await _window(archive, run_lane=run)
    lanes = [b.lane for b in run.calls]
    assert lanes.count("incumbent") == 2
    assert lanes.count("novelty") == 2


# ---- H1 (P2-3): only the tasks somebody actually acted on are charged -----
#
# The router's exhaustion blacklist is fed by `record_reflection(made_progress)`.
# The blanket call over every diagnosed task was deleted for good reason — it
# punished tasks whose findings were never implemented, so "proposed something
# and lost the selection" counted as "we tried and failed", and 13/120 tasks
# were blacklisted that way. Nothing replaced it, so no task is charged at all
# and the exhaustion signal is dead in the two-lane path.
#
# The replacement is selection-aware: only the SELECTED proposal's support
# tasks, and only once the changed variant actually ran. A change that never
# executed has not tested anything, so it cannot be evidence that the task is
# unfixable.


class _Ran:
    """A lane result whose sanity run has activation evidence for the change."""

    def __init__(self, ran=True, reject_class="none"):
        self.sanity_activation = {"variants": {"agent_loop/x": ["ok"]}} if ran else None
        self.review_reject_class = reject_class
        self.review_repair_brief = ""


def _settling(passed, *, ran=True):
    async def run(brief):
        path = brief.staging_dir / "modules" / "agent_loop" / "confirm_exit.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"DESCRIPTION = '{brief.lane}'\n")
        return dual_implement.LaneResult(passed=passed, detail=_Ran(ran))

    return run


async def test_a_promoted_lane_charges_its_support_tasks_with_progress(tmp_path):
    archive = _archive_with_gen0(tmp_path)
    _proposal(archive, "incumbent", task="fix-git", text="A happened")
    seen = []
    await _window(archive, run_lane=_settling(True), on_lane_settled=seen.append)
    (settled,) = seen
    assert settled.support_tasks == ["fix-git"]
    assert settled.made_progress is True
    assert settled.ran is True


async def test_a_lane_that_ran_and_lost_charges_no_progress(tmp_path):
    archive = _archive_with_gen0(tmp_path)
    _proposal(archive, "incumbent", task="fix-git", text="A happened")
    seen = []
    await _window(archive, run_lane=_settling(False), on_lane_settled=seen.append)
    (settled,) = seen
    assert settled.made_progress is False
    assert settled.ran is True


async def test_a_change_that_never_executed_charges_nothing(tmp_path):
    # a variant that never ran has not tested the direction, so it is not
    # evidence that the task is unfixable — which is exactly what the deleted
    # blanket call got wrong
    archive = _archive_with_gen0(tmp_path)
    _proposal(archive, "incumbent", task="fix-git", text="A happened")
    seen = []
    await _window(
        archive, run_lane=_settling(False, ran=False), on_lane_settled=seen.append
    )
    (settled,) = seen
    assert settled.ran is False


async def test_a_task_nobody_acted_on_is_never_reported(tmp_path):
    # two directions compete for one incumbent slot; the loser was never tried,
    # and charging its tasks is exactly the blacklist bug the blanket call caused
    archive = _archive_with_gen0(tmp_path)
    _proposal(archive, "incumbent", task="direction-a", text="A happened")
    _proposal(archive, "incumbent", task="direction-b", text="B happened")
    seen = []
    await _window(archive, run_lane=_settling(True), on_lane_settled=seen.append)
    charged = {t for s in seen for t in s.support_tasks}
    assert len(charged) == 1, f"both directions were charged: {charged}"


async def test_accounting_never_takes_the_window_down(tmp_path):
    archive = _archive_with_gen0(tmp_path)
    _proposal(archive, "incumbent", task="fix-git", text="A happened")

    def _explode(settlement):
        raise RuntimeError("ledger on fire")

    out = await _window(archive, run_lane=_settling(True), on_lane_settled=_explode)
    assert [v.promoted_gen.name for v in out.promoted] == ["gen_1"]


async def test_the_regate_is_told_which_proposal_it_is_regating(tmp_path):
    # the re-gate runs the same battery, and that battery now holds a change to
    # the action it was routed for — which lives on the proposal. Without the id
    # the rebased lane is the one candidate nobody checks.
    archive = _archive_with_gen0(tmp_path)
    incumbent, novelty = _both_lanes(archive)
    seen = []
    await _window(archive, rerun_gates=_regate(seen=seen))
    assert [call["proposal_id"] for call in seen] == [novelty.proposal_id]
