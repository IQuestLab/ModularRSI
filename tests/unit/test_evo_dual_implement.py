"""P2: two proposals, two isolated implementations, mutually blind.

The portfolio hands back at most one proposal per lane. This module is what
happens next: both are built off the *same* parent tree, in two separate
stagings, by two implementers that cannot see each other.

Three properties are load-bearing, and each of them exists because of a specific
way this has already gone wrong:

* **Separate trees.** One shared staging would let whichever implementer wrote
  last silently decide what the other one built on, and the two gate results
  would then be measuring a tree neither candidate actually proposed.
* **Mutual blindness.** If implementer A can read B's proposal, the two stop
  being independent samples of "what should change" — A talks itself into B's
  framing, and the second lane stops being a second opinion. The whole reason
  novelty gets its own slot is that it must not have to argue against the
  incumbent; showing it the incumbent's proposal reintroduces that argument.
* **Failure isolation.** A crash in one lane must not block, discard, or
  contaminate the other. Two lanes exist to raise the chance that *something*
  lands; coupling their failures throws that away.

And one contract inherited from `f34f2c8`: the implementer is given **every**
finding that supports its proposal, not a sample and not the "best" one. Handing
over a single finding is how a general direction degrades into "make this one
task pass", which is the documented cause of death of that lineage.
"""

import asyncio

import pytest

from harbor.agents.terminus_2_modular.self_evo import (
    backlog,
    dual_implement,
    portfolio,
    proposals,
)

pytestmark = pytest.mark.unit


# ---- fixtures -------------------------------------------------------------


def _parent_tree(tmp_path):
    """A minimal package-layout parent gen."""
    root = tmp_path / "gen_0"
    (root / "modules" / "agent_loop").mkdir(parents=True)
    (root / "modules" / "agent_loop" / "confirm_exit.py").write_text(
        "DESCRIPTION = 'baseline'\n"
    )
    (root / "protocols.py").write_text("# protocols\n")
    return root


def _archive(tmp_path):
    root = tmp_path / "archive"
    root.mkdir()
    return root


def _finding(archive, step, task, text):
    payload = {
        "step": step,
        "findings": [{"task": task, "is_culprit": True, "divergence": text}],
    }
    return backlog.ingest_step(archive, payload)[0].finding_id


def _proposal(
    archive, lane, *, tasks, texts, delta="", why="", target="", pid_step="w1"
):
    fids = [
        _finding(archive, f"{pid_step}_{i}", task, text)
        for i, (task, text) in enumerate(zip(tasks, texts))
    ]
    return proposals.create_proposal(
        archive,
        step=pid_step,
        lane=lane,
        action="modify" if lane == "incumbent" else "add",
        target_variant=target
        or ("agent_loop/confirm_exit" if lane == "incumbent" else ""),
        behavioral_delta=delta or f"{lane} delta",
        causal_hypothesis=why or f"{lane} because",
        finding_ids=fids,
        support_tasks=list(tasks),
    )


def _picks(archive):
    return portfolio.select_portfolio(archive)


async def _noop_lane(brief):
    return dual_implement.LaneResult(passed=True, diff_hash="d")


# ---- separate trees -------------------------------------------------------


async def test_each_lane_gets_its_own_staging_tree(tmp_path):
    archive, parent = _archive(tmp_path), _parent_tree(tmp_path)
    _proposal(archive, "incumbent", tasks=["a"], texts=["A happened"])
    _proposal(archive, "novelty", tasks=["b"], texts=["B happened"])

    outcomes = await dual_implement.implement_lanes(
        archive, _picks(archive), parent_dir=parent, step="w2", run_lane=_noop_lane
    )

    dirs = [o.staging_dir for o in outcomes]
    assert len(dirs) == 2
    assert dirs[0] != dirs[1]
    assert all(d.is_dir() for d in dirs)
    # siblings, never nested: the editor's read_file is scoped to its own
    # staging root, so non-containment is what actually keeps the lanes blind
    assert dirs[0] not in dirs[1].parents
    assert dirs[1] not in dirs[0].parents


async def test_the_staging_name_says_which_lane_and_which_proposal(tmp_path):
    # a run that dies mid-window leaves these dirs behind; they have to be
    # readable without consulting a log to know what was in them
    archive, parent = _archive(tmp_path), _parent_tree(tmp_path)
    p = _proposal(archive, "novelty", tasks=["b"], texts=["B"])

    (outcome,) = await dual_implement.implement_lanes(
        archive, _picks(archive), parent_dir=parent, step="w2", run_lane=_noop_lane
    )

    assert "novelty" in outcome.staging_dir.name
    assert p.proposal_id in outcome.staging_dir.name


async def test_editing_one_lane_tree_leaves_the_other_untouched(tmp_path):
    archive, parent = _archive(tmp_path), _parent_tree(tmp_path)
    _proposal(archive, "incumbent", tasks=["a"], texts=["A"])
    _proposal(archive, "novelty", tasks=["b"], texts=["B"])

    async def edits(brief):
        target = brief.staging_dir / "modules" / "agent_loop" / "confirm_exit.py"
        target.write_text(f"DESCRIPTION = '{brief.lane} was here'\n")
        return dual_implement.LaneResult(passed=True, diff_hash=brief.lane)

    outcomes = await dual_implement.implement_lanes(
        archive, _picks(archive), parent_dir=parent, step="w2", run_lane=edits
    )

    written = {
        o.lane: (
            o.staging_dir / "modules" / "agent_loop" / "confirm_exit.py"
        ).read_text()
        for o in outcomes
    }
    assert "incumbent was here" in written["incumbent"]
    assert "novelty was here" in written["novelty"]
    assert "novelty" not in written["incumbent"]


async def test_both_trees_start_from_the_same_parent(tmp_path):
    archive, parent = _archive(tmp_path), _parent_tree(tmp_path)
    _proposal(archive, "incumbent", tasks=["a"], texts=["A"])
    _proposal(archive, "novelty", tasks=["b"], texts=["B"])

    outcomes = await dual_implement.implement_lanes(
        archive, _picks(archive), parent_dir=parent, step="w2", run_lane=_noop_lane
    )

    for o in outcomes:
        copied = o.staging_dir / "modules" / "agent_loop" / "confirm_exit.py"
        assert copied.read_text() == "DESCRIPTION = 'baseline'\n"


async def test_the_parent_tree_is_never_edited(tmp_path):
    archive, parent = _archive(tmp_path), _parent_tree(tmp_path)
    _proposal(archive, "incumbent", tasks=["a"], texts=["A"])

    async def edits(brief):
        (brief.staging_dir / "modules" / "agent_loop" / "confirm_exit.py").write_text(
            "X"
        )
        return dual_implement.LaneResult(passed=True, diff_hash="d")

    await dual_implement.implement_lanes(
        archive, _picks(archive), parent_dir=parent, step="w2", run_lane=edits
    )

    original = parent / "modules" / "agent_loop" / "confirm_exit.py"
    assert original.read_text() == "DESCRIPTION = 'baseline'\n"


# ---- the brief: everything of mine, nothing of theirs ---------------------


async def test_the_brief_carries_every_supporting_finding(tmp_path):
    # NOT a sample, NOT the strongest one — `f34f2c8` died of being handed one
    # finding and building "make this task pass" out of it
    archive, parent = _archive(tmp_path), _parent_tree(tmp_path)
    texts = ["ate the trailing newline", "hung on the pager", "retried forever"]
    _proposal(archive, "incumbent", tasks=["t1", "t2", "t3"], texts=texts)

    seen = {}

    async def capture(brief):
        seen["brief"] = brief
        return dual_implement.LaneResult(passed=True, diff_hash="d")

    await dual_implement.implement_lanes(
        archive, _picks(archive), parent_dir=parent, step="w2", run_lane=capture
    )

    brief = seen["brief"]
    assert len(brief.findings) == 3
    for text in texts:
        assert text in brief.instruction


async def test_the_brief_never_mentions_the_other_proposal(tmp_path):
    archive, parent = _archive(tmp_path), _parent_tree(tmp_path)
    _proposal(
        archive,
        "incumbent",
        tasks=["a"],
        texts=["A happened"],
        delta="tighten the exit check",
    )
    other = _proposal(
        archive,
        "novelty",
        tasks=["b"],
        texts=["B happened"],
        delta="add a pager escape hatch",
    )

    briefs = {}

    async def capture(brief):
        briefs[brief.lane] = brief.instruction
        return dual_implement.LaneResult(passed=True, diff_hash="d")

    await dual_implement.implement_lanes(
        archive, _picks(archive), parent_dir=parent, step="w2", run_lane=capture
    )

    incumbent_brief = briefs["incumbent"]
    assert other.proposal_id not in incumbent_brief
    assert "pager escape hatch" not in incumbent_brief
    assert "B happened" not in incumbent_brief


async def test_the_brief_states_the_direction_and_the_reasoning(tmp_path):
    archive, parent = _archive(tmp_path), _parent_tree(tmp_path)
    _proposal(
        archive,
        "incumbent",
        tasks=["a"],
        texts=["A"],
        delta="stop exiting on the first empty pane",
        why="the pane is empty while the command is still starting",
    )

    briefs = []

    async def capture(brief):
        briefs.append(brief.instruction)
        return dual_implement.LaneResult(passed=True, diff_hash="d")

    await dual_implement.implement_lanes(
        archive, _picks(archive), parent_dir=parent, step="w2", run_lane=capture
    )

    assert "stop exiting on the first empty pane" in briefs[0]
    assert "the pane is empty while the command is still starting" in briefs[0]


async def test_an_incumbent_brief_names_the_variant_it_must_change(tmp_path):
    archive, parent = _archive(tmp_path), _parent_tree(tmp_path)
    _proposal(
        archive, "incumbent", tasks=["a"], texts=["A"], target="agent_loop/confirm_exit"
    )

    briefs = []

    async def capture(brief):
        briefs.append(brief.instruction)
        return dual_implement.LaneResult(passed=True, diff_hash="d")

    await dual_implement.implement_lanes(
        archive, _picks(archive), parent_dir=parent, step="w2", run_lane=capture
    )

    assert "agent_loop/confirm_exit" in briefs[0]


async def test_a_support_finding_missing_from_the_ledger_is_declared(tmp_path):
    # "all the evidence" is the contract; quietly handing over 2 of 3 would let
    # the implementer believe it saw the whole picture
    archive, parent = _archive(tmp_path), _parent_tree(tmp_path)
    real = _finding(archive, "w1", "a", "A happened")
    proposals.create_proposal(
        archive,
        step="w1",
        lane="incumbent",
        action="modify",
        target_variant="agent_loop/confirm_exit",
        finding_ids=[real, "f_notinledger"],
        support_tasks=["a"],
    )

    briefs = []

    async def capture(brief):
        briefs.append(brief)
        return dual_implement.LaneResult(passed=True, diff_hash="d")

    await dual_implement.implement_lanes(
        archive, _picks(archive), parent_dir=parent, step="w2", run_lane=capture
    )

    assert briefs[0].missing_finding_ids == ["f_notinledger"]
    assert "f_notinledger" in briefs[0].instruction


# ---- failure isolation ----------------------------------------------------


async def test_one_lane_crashing_does_not_stop_the_other(tmp_path):
    archive, parent = _archive(tmp_path), _parent_tree(tmp_path)
    _proposal(archive, "incumbent", tasks=["a"], texts=["A"])
    _proposal(archive, "novelty", tasks=["b"], texts=["B"])

    async def crash_incumbent(brief):
        if brief.lane == "incumbent":
            raise RuntimeError("editor died")
        return dual_implement.LaneResult(passed=True, diff_hash="d")

    outcomes = await dual_implement.implement_lanes(
        archive, _picks(archive), parent_dir=parent, step="w2", run_lane=crash_incumbent
    )

    by_lane = {o.lane: o for o in outcomes}
    assert by_lane["incumbent"].ok is False
    assert by_lane["novelty"].ok is True


async def test_a_crashed_lane_is_reported_not_raised(tmp_path):
    archive, parent = _archive(tmp_path), _parent_tree(tmp_path)
    _proposal(archive, "incumbent", tasks=["a"], texts=["A"])

    async def crash(brief):
        raise RuntimeError("editor died")

    (outcome,) = await dual_implement.implement_lanes(
        archive, _picks(archive), parent_dir=parent, step="w2", run_lane=crash
    )

    assert outcome.ok is False
    assert "editor died" in outcome.error


async def test_a_crash_does_not_discard_the_surviving_lane_staging(tmp_path):
    archive, parent = _archive(tmp_path), _parent_tree(tmp_path)
    _proposal(archive, "incumbent", tasks=["a"], texts=["A"])
    _proposal(archive, "novelty", tasks=["b"], texts=["B"])

    async def crash_incumbent(brief):
        if brief.lane == "incumbent":
            raise RuntimeError("boom")
        return dual_implement.LaneResult(passed=True, diff_hash="d")

    outcomes = await dual_implement.implement_lanes(
        archive, _picks(archive), parent_dir=parent, step="w2", run_lane=crash_incumbent
    )

    survivor = next(o for o in outcomes if o.lane == "novelty")
    assert survivor.staging_dir.is_dir()


async def test_the_two_lanes_really_run_at_the_same_time(tmp_path):
    # serialised lanes would double the wall clock of every window, and this is
    # the only property a later refactor could quietly lose
    archive, parent = _archive(tmp_path), _parent_tree(tmp_path)
    _proposal(archive, "incumbent", tasks=["a"], texts=["A"])
    _proposal(archive, "novelty", tasks=["b"], texts=["B"])

    started = {"incumbent": asyncio.Event(), "novelty": asyncio.Event()}

    async def rendezvous(brief):
        started[brief.lane].set()
        other = "novelty" if brief.lane == "incumbent" else "incumbent"
        await asyncio.wait_for(started[other].wait(), timeout=5)
        return dual_implement.LaneResult(passed=True, diff_hash="d")

    outcomes = await dual_implement.implement_lanes(
        archive, _picks(archive), parent_dir=parent, step="w2", run_lane=rendezvous
    )

    assert all(o.ok for o in outcomes), (
        "a lane timed out waiting: lanes were serialised"
    )


# ---- accounting -----------------------------------------------------------


async def test_an_implemented_proposal_is_recorded_as_attempted(tmp_path):
    archive, parent = _archive(tmp_path), _parent_tree(tmp_path)
    p = _proposal(archive, "incumbent", tasks=["a"], texts=["A"])

    await dual_implement.implement_lanes(
        archive, _picks(archive), parent_dir=parent, step="w2", run_lane=_noop_lane
    )

    after = proposals.get_proposal(archive, p.proposal_id)
    assert after.state == proposals.ATTEMPTED
    assert after.attempts == 1


async def test_a_lane_that_crashed_inside_the_implementer_still_counts_as_attempted(
    tmp_path,
):
    # the intervention happened; it just went badly. Hiding that would let the
    # same proposal be retried forever without ever reaching taboo.
    archive, parent = _archive(tmp_path), _parent_tree(tmp_path)
    p = _proposal(archive, "incumbent", tasks=["a"], texts=["A"])

    async def crash(brief):
        raise RuntimeError("boom")

    await dual_implement.implement_lanes(
        archive, _picks(archive), parent_dir=parent, step="w2", run_lane=crash
    )

    after = proposals.get_proposal(archive, p.proposal_id)
    assert after.state == proposals.ATTEMPTED
    assert after.attempts == 1


async def test_a_lane_that_never_reached_the_implementer_goes_back_to_open(tmp_path):
    # staging could not even be built, so nothing was tried on this proposal's
    # behalf — charging it an attempt would spend one of its two lives for free
    archive = _archive(tmp_path)
    p = _proposal(archive, "incumbent", tasks=["a"], texts=["A"])

    ran = []

    async def never_called(brief):
        ran.append(brief.lane)
        return dual_implement.LaneResult(passed=True, diff_hash="d")

    (outcome,) = await dual_implement.implement_lanes(
        archive,
        _picks(archive),
        parent_dir=tmp_path / "does_not_exist",
        step="w2",
        run_lane=never_called,
    )

    assert ran == []
    assert outcome.ok is False
    after = proposals.get_proposal(archive, p.proposal_id)
    assert after.state == proposals.OPEN
    assert after.attempts == 0


async def test_nothing_selected_means_nothing_happens(tmp_path):
    archive, parent = _archive(tmp_path), _parent_tree(tmp_path)
    outcomes = await dual_implement.implement_lanes(
        archive, [], parent_dir=parent, step="w2", run_lane=_noop_lane
    )
    assert outcomes == []
    assert not (archive / "staging").exists()


# ---- handoff to promotion -------------------------------------------------


async def test_outcomes_become_promotion_candidates(tmp_path):
    archive, parent = _archive(tmp_path), _parent_tree(tmp_path)
    _proposal(archive, "incumbent", tasks=["a"], texts=["A"])
    _proposal(archive, "novelty", tasks=["b"], texts=["B"])

    outcomes = await dual_implement.implement_lanes(
        archive, _picks(archive), parent_dir=parent, step="w2", run_lane=_noop_lane
    )
    candidates = dual_implement.to_candidates(outcomes)

    assert {c.lane for c in candidates} == {"incumbent", "novelty"}
    assert all(c.passed_gates for c in candidates)


async def test_a_crashed_lane_is_not_a_promotion_candidate(tmp_path):
    archive, parent = _archive(tmp_path), _parent_tree(tmp_path)
    _proposal(archive, "incumbent", tasks=["a"], texts=["A"])
    _proposal(archive, "novelty", tasks=["b"], texts=["B"])

    async def crash_incumbent(brief):
        if brief.lane == "incumbent":
            raise RuntimeError("boom")
        return dual_implement.LaneResult(passed=True, diff_hash="d")

    outcomes = await dual_implement.implement_lanes(
        archive, _picks(archive), parent_dir=parent, step="w2", run_lane=crash_incumbent
    )
    candidates = dual_implement.to_candidates(outcomes)

    assert [c.lane for c in candidates] == ["novelty"]


async def test_a_lane_that_failed_its_gates_is_carried_through_as_failing(tmp_path):
    # it must still reach `plan_promotions` — that is where "one lane failing
    # does not block the other" is decided, and it cannot decide about a
    # candidate it was never shown
    archive, parent = _archive(tmp_path), _parent_tree(tmp_path)
    _proposal(archive, "incumbent", tasks=["a"], texts=["A"])

    async def fails_gates(brief):
        return dual_implement.LaneResult(passed=False, diff_hash="d", detail="smoke")

    outcomes = await dual_implement.implement_lanes(
        archive, _picks(archive), parent_dir=parent, step="w2", run_lane=fails_gates
    )
    (candidate,) = dual_implement.to_candidates(outcomes)

    assert candidate.passed_gates is False


# ---- H4: the window writes down what it took ------------------------------
#
# Everything between `open → selected` and `selected → attempted` is tens of
# minutes of editor and gates. A process that dies in there raises nothing, so
# the existing "staging failed → back to open" recovery never fires and the
# proposal is `selected` forever — unpickable, with all its support intact.


async def test_taking_a_proposal_is_written_to_the_window_journal(tmp_path):
    from harbor.agents.terminus_2_modular.self_evo import lease

    archive, parent = _archive(tmp_path), _parent_tree(tmp_path)
    _proposal(archive, "incumbent", tasks=["a"], texts=["A happened"])
    await dual_implement.implement_lanes(
        archive, _picks(archive), parent_dir=parent, step="w2", run_lane=_noop_lane
    )
    # the lane finished, so nothing is left open — but the journal recorded the
    # take and the launch, which is what a crash would have left behind
    assert lease.open_leases(archive) == []
    events = [row["event"] for row in lease._read(archive)]
    assert events == ["selected", "attempt_started", "attempt_finished"]


async def test_a_lane_killed_mid_implementation_leaves_a_recoverable_lease(tmp_path):
    from harbor.agents.terminus_2_modular.self_evo import lease

    archive, parent = _archive(tmp_path), _parent_tree(tmp_path)
    _proposal(archive, "incumbent", tasks=["a"], texts=["A happened"])

    class _Killed(BaseException):
        """Not an Exception: this is what a process death looks like."""

    async def _die(brief):
        raise _Killed

    with pytest.raises(_Killed):
        await dual_implement.implement_lanes(
            archive, _picks(archive), parent_dir=parent, step="w2", run_lane=_die
        )

    (stranded,) = lease.open_leases(archive)
    assert stranded.attempt_started is True
    assert proposals.get_proposal(archive, stranded.proposal_id).state == (
        proposals.SELECTED
    )

    # the next window settles it: the attempt is charged, the direction retried
    (recovered,) = lease.recover(archive, step="w3")
    assert recovered.action == lease.SETTLED
    after = proposals.get_proposal(archive, stranded.proposal_id)
    assert after.state == proposals.REJECTED_IMPLEMENTATION
    assert after.attempts == 1


async def test_a_lane_whose_staging_never_built_leaves_nothing_to_recover(tmp_path):
    from harbor.agents.terminus_2_modular.self_evo import lease

    archive, parent = _archive(tmp_path), _parent_tree(tmp_path)
    _proposal(archive, "incumbent", tasks=["a"], texts=["A happened"])

    def _no_staging(parent_dir, dest):
        raise OSError("no space left on device")

    await dual_implement.implement_lanes(
        archive,
        _picks(archive),
        parent_dir=parent,
        step="w2",
        run_lane=_noop_lane,
        prepare=_no_staging,
    )
    # it was already put back to `open` in-process; a stale lease that reopened
    # it a second time would be harmless but a lease that SETTLED it would
    # charge an attempt for a crash it had no part in
    assert lease.open_leases(archive) == []
    assert lease.recover(archive, step="w3") == []
    assert proposals.get_proposal(archive, "p_0001").attempts == 0
