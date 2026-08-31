"""Tests for deterministic, anti-starvation portfolio selection.

Each lane selects at most one open proposal. Never-attempted and longest-waiting
proposals come first, so every open direction eventually receives a turn.
"""

import pytest

from harbor.agents.terminus_2_modular.self_evo import backlog, portfolio, proposals

pytestmark = pytest.mark.unit


def _finding(tmp_path, step, task):
    payload = {"step": step, "findings": [{"task": task, "is_culprit": True}]}
    return backlog.ingest_step(tmp_path, payload)[0].finding_id


def _mk(tmp_path, lane, *, step="gen_1_c", tasks=("t",), windows=None, action=None):
    """A proposal whose support really exists in the ledger."""
    windows = windows or [step]
    fids = []
    for i, task in enumerate(tasks):
        fids.append(_finding(tmp_path, windows[i % len(windows)], task))
    return proposals.create_proposal(
        tmp_path,
        step=step,
        lane=lane,
        action=action or ("modify" if lane == "incumbent" else "add"),
        target_variant="agent_loop/confirm_exit" if lane == "incumbent" else "",
        finding_ids=fids,
        support_tasks=list(tasks),
    )


# ---- the ceiling ----------------------------------------------------------


def test_at_most_one_per_lane(tmp_path):
    for i in range(3):
        _mk(tmp_path, "incumbent", tasks=(f"a{i}",))
        _mk(tmp_path, "novelty", tasks=(f"b{i}",))
    picks = portfolio.select_portfolio(tmp_path)
    assert len(picks) == 2
    assert {p.lane for p in picks} == {"incumbent", "novelty"}


def test_an_empty_lane_contributes_nothing(tmp_path):
    _mk(tmp_path, "incumbent")
    picks = portfolio.select_portfolio(tmp_path)
    assert [p.lane for p in picks] == ["incumbent"]


def test_no_open_proposals_means_no_change_this_window(tmp_path):
    assert portfolio.select_portfolio(tmp_path) == []


def test_the_ceiling_is_never_a_quota(tmp_path):
    # only one lane has anything: do NOT reach into the other lane to reach K=2
    _mk(tmp_path, "novelty", tasks=("a",))
    _mk(tmp_path, "novelty", tasks=("b",))
    picks = portfolio.select_portfolio(tmp_path)
    assert len(picks) == 1


def test_out_of_scope_is_not_a_lane_here(tmp_path):
    # that work belongs to another module; it must not take a slot
    _mk(tmp_path, "out_of_scope")
    assert portfolio.select_portfolio(tmp_path) == []


def test_only_open_proposals_are_eligible(tmp_path):
    p = _mk(tmp_path, "incumbent")
    proposals.transition(tmp_path, p.proposal_id, step="s", to="selected")
    assert portfolio.select_portfolio(tmp_path) == []


# ---- the lane-internal order ----------------------------------------------


def test_never_attempted_beats_everything_else(tmp_path):
    strong = _mk(
        tmp_path, "incumbent", tasks=("a", "b", "c"), windows=["w1", "w2", "w3"]
    )
    fresh = _mk(tmp_path, "incumbent", step="gen_9_c", tasks=("z",))
    # give the strong one a failed attempt
    for state in ("selected", "attempted", "rejected_implementation", "open"):
        proposals.transition(tmp_path, strong.proposal_id, step="s", to=state)
    (pick,) = portfolio.select_portfolio(tmp_path)
    assert pick.proposal_id == fresh.proposal_id


def test_longest_waiting_wins_among_never_attempted(tmp_path):
    old = _mk(tmp_path, "incumbent", step="gen_2_c", tasks=("a",))
    _mk(tmp_path, "incumbent", step="gen_8_c", tasks=("b", "c"), windows=["w1", "w2"])
    (pick,) = portfolio.select_portfolio(tmp_path)
    assert pick.proposal_id == old.proposal_id, "frequency must not override FIFO"


def test_more_unique_support_tasks_wins_at_equal_age(tmp_path):
    _mk(tmp_path, "incumbent", step="gen_3_c", tasks=("a",))
    broad = _mk(tmp_path, "incumbent", step="gen_3_c", tasks=("x", "y", "z"))
    (pick,) = portfolio.select_portfolio(tmp_path)
    assert pick.proposal_id == broad.proposal_id


def test_the_same_task_seen_twice_does_not_count_twice(tmp_path):
    # a gripe repeating across epochs is one task's worth of evidence, not two
    _mk(tmp_path, "incumbent", step="gen_3_c", tasks=("a", "a", "a"))  # 3 refs, 1 task
    two = _mk(tmp_path, "incumbent", step="gen_3_c", tasks=("m", "n"))
    (pick,) = portfolio.select_portfolio(tmp_path)
    assert pick.proposal_id == two.proposal_id


def test_more_unique_windows_breaks_a_task_tie(tmp_path):
    one_window = _mk(
        tmp_path, "incumbent", step="gen_4_c", tasks=("a", "b"), windows=["w1"]
    )
    spread = _mk(
        tmp_path, "incumbent", step="gen_4_c", tasks=("c", "d"), windows=["w1", "w2"]
    )
    (pick,) = portfolio.select_portfolio(tmp_path)
    assert pick.proposal_id == spread.proposal_id
    assert one_window.proposal_id != spread.proposal_id


def test_ties_are_broken_stably_by_seed(tmp_path):
    _mk(tmp_path, "incumbent", step="gen_5_c", tasks=("a",))
    _mk(tmp_path, "incumbent", step="gen_5_c", tasks=("b",))
    first = portfolio.select_portfolio(tmp_path, seed="run-7")[0].proposal_id
    again = portfolio.select_portfolio(tmp_path, seed="run-7")[0].proposal_id
    assert first == again


def test_a_different_seed_may_pick_the_other_one(tmp_path):
    _mk(tmp_path, "incumbent", step="gen_5_c", tasks=("a",))
    _mk(tmp_path, "incumbent", step="gen_5_c", tasks=("b",))
    picks = {
        portfolio.select_portfolio(tmp_path, seed=f"s{i}")[0].proposal_id
        for i in range(12)
    }
    assert len(picks) == 2, "a stable tie-break must still be a tie-break"


# ---- starvation is bounded ------------------------------------------------


def test_every_open_proposal_is_reached_in_finite_rounds(tmp_path):
    """The property the whole phase exists for: nothing waits forever."""
    made = [
        _mk(tmp_path, "incumbent", step="gen_1_c", tasks=(f"t{i}",)) for i in range(5)
    ]
    seen = set()
    for _ in range(5):
        picks = portfolio.select_portfolio(tmp_path, seed="fifo")
        if not picks:
            break
        pick = picks[0]
        seen.add(pick.proposal_id)
        # simulate a window: selected → attempted → rejected → back to open
        for state in ("selected", "attempted", "rejected_implementation", "open"):
            proposals.transition(tmp_path, pick.proposal_id, step="s", to=state)
    assert seen == {p.proposal_id for p in made}


# ---- the escape hatch -----------------------------------------------------


def test_max_lanes_one_runs_the_portfolio_without_parallelism(tmp_path):
    # lets a defect in *selection* be told apart from a defect in *running two*
    _mk(tmp_path, "incumbent", tasks=("a",))
    _mk(tmp_path, "novelty", tasks=("b",))
    picks = portfolio.select_portfolio(tmp_path, max_lanes=1)
    assert len(picks) == 1


def test_max_lanes_zero_selects_nothing(tmp_path):
    _mk(tmp_path, "incumbent")
    assert portfolio.select_portfolio(tmp_path, max_lanes=0) == []


# ---- the audit ------------------------------------------------------------


def test_each_pick_carries_why_it_won(tmp_path):
    _mk(tmp_path, "incumbent", step="gen_2_c", tasks=("a",))
    (pick,) = portfolio.select_portfolio(tmp_path)
    assert pick.reason
    assert pick.rank_key is not None


# ---- "waiting longest" has to survive the tenth generation ----------------
#
# Step ids look like `gen_7_candidate_1787832340`, and the rank key compared
# them as plain strings. `"gen_10_..." < "gen_2_..."` because "1" sorts before
# "2", so from the tenth generation on the NEWEST proposal reads as the oldest.
# That clause is half of the starvation bound: without it a fresh proposal can
# keep jumping the queue ahead of one that has been waiting since gen_2.


def test_the_older_proposal_wins_across_the_ten_boundary(tmp_path):
    old = _mk(tmp_path, "incumbent", step="gen_2_candidate_1787832340", tasks=("a",))
    _mk(tmp_path, "incumbent", step="gen_10_candidate_1787999999", tasks=("b",))

    picks = portfolio.select_portfolio(tmp_path)

    assert [p.proposal_id for p in picks] == [old.proposal_id]


def test_ordering_within_one_generation_is_still_by_time(tmp_path):
    """The fix must not throw away the timestamp that breaks same-gen ties."""
    early = _mk(tmp_path, "novelty", step="gen_3_candidate_1787839974", tasks=("a",))
    _mk(tmp_path, "novelty", step="gen_3_candidate_1787851454", tasks=("b",))

    picks = portfolio.select_portfolio(tmp_path)

    assert [p.proposal_id for p in picks] == [early.proposal_id]
