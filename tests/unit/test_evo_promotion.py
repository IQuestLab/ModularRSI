"""P2 promotion: two candidates from one window, and what may be reused.

Two lanes are built in parallel off the same parent, so they cannot both be
promoted onto that parent as-is — the second one has to land on a tree that
already contains the first. The subtle part is which gate results survive that
move, and the answer is not "the diff didn't change, so nothing changed":

* **review** judges the change itself. A byte-identical diff is the same code
  and the same reasoning, so that verdict carries over. Throwing it away would
  buy a second review pass for nothing — and review is historically the largest
  source of false rejects, so every extra pass is another chance to kill a good
  change.
* **smoke**, **runtime activation** and **routing evidence** all judge the tree,
  not the diff. Smoke loads the *entire* module library; two lanes that each
  load fine can merge into a tree that does not (the obvious case being two
  variants registering the same name). The merged tree is a combination neither
  lane was ever tested as, so its smoke verdict does not exist yet. Activation
  and routing depend on the composer's candidate set, and the other candidate
  just added a variant — and a DESCRIPTION — to it.

Getting this backwards is expensive in both directions: reusing activation
re-creates the "promoted but never selected" failure this phase exists to close,
and discarding review pays an LLM pass to re-litigate a change nobody touched.
Smoke costs nothing to re-run, so when in doubt it re-runs.
"""

import pytest

from harbor.agents.terminus_2_modular.self_evo import promotion

pytestmark = pytest.mark.unit


def _cand(lane, *, passed=True, pid=None, diff="d1"):
    return promotion.Candidate(
        proposal_id=pid or f"p_{lane}",
        lane=lane,
        passed_gates=passed,
        diff_hash=diff,
    )


# ---- who gets promoted ----------------------------------------------------


def test_both_failing_keeps_the_parent(tmp_path):
    plan = promotion.plan_promotions(
        [_cand("incumbent", passed=False), _cand("novelty", passed=False)]
    )
    assert plan == []


def test_only_the_passing_one_is_promoted(tmp_path):
    plan = promotion.plan_promotions(
        [_cand("incumbent", passed=False), _cand("novelty")]
    )
    assert [s.lane for s in plan] == ["novelty"]
    assert plan[0].rebase_required is False


def test_one_lane_failing_does_not_block_the_other(tmp_path):
    plan = promotion.plan_promotions(
        [_cand("incumbent"), _cand("novelty", passed=False)]
    )
    assert [s.lane for s in plan] == ["incumbent"]


def test_nothing_at_all_is_a_no_op(tmp_path):
    assert promotion.plan_promotions([]) == []


# ---- order when both pass -------------------------------------------------


def test_incumbent_goes_first_by_default(tmp_path):
    plan = promotion.plan_promotions([_cand("novelty"), _cand("incumbent")])
    assert [s.lane for s in plan] == ["incumbent", "novelty"]
    assert [s.order for s in plan] == [0, 1]


def test_the_second_one_must_rebase(tmp_path):
    plan = promotion.plan_promotions([_cand("incumbent"), _cand("novelty")])
    assert plan[0].rebase_required is False
    assert plan[1].rebase_required is True


def test_the_order_can_be_alternated_when_data_says_so(tmp_path):
    # novelty always going second means it always eats the extra review pass —
    # and review is where the false rejects historically came from. If the
    # numbers show novelty dying more often on the rebase pass, flip it.
    plan = promotion.plan_promotions(
        [_cand("incumbent"), _cand("novelty")], novelty_first=True
    )
    assert [s.lane for s in plan] == ["novelty", "incumbent"]
    assert plan[1].rebase_required is True


# ---- which gates survive a rebase ----------------------------------------


def test_an_unchanged_diff_keeps_the_review(tmp_path):
    reuse, _rerun = promotion.gates_after_rebase("abc", "abc")
    assert set(reuse) == {"review"}


def test_an_unchanged_diff_still_re_runs_every_tree_wide_gate(tmp_path):
    # all three read the whole tree, and the tree is not the one they measured
    _reuse, rerun = promotion.gates_after_rebase("abc", "abc")
    assert set(rerun) == {"smoke", "activation", "routing"}


def test_smoke_is_not_reusable_because_it_loads_the_whole_library(tmp_path):
    # two lanes that each load fine can merge into a tree that does not — two
    # variants registering the same name is the obvious case. The merged tree
    # is a combination neither lane was ever tested as, and smoke is free.
    reuse, rerun = promotion.gates_after_rebase("abc", "abc")
    assert "smoke" not in reuse
    assert "smoke" in rerun


def test_any_content_change_invalidates_every_gate(tmp_path):
    reuse, rerun = promotion.gates_after_rebase("abc", "def")
    assert reuse == ()
    assert set(rerun) == {"smoke", "review", "activation", "routing"}


def test_no_tree_wide_gate_is_ever_reusable_whatever_the_diff(tmp_path):
    for before, after in (("x", "x"), ("x", "y")):
        reuse, _ = promotion.gates_after_rebase(before, after)
        for gate in ("smoke", "activation", "routing"):
            assert gate not in reuse


def test_a_first_promotion_needs_no_rebase_decision(tmp_path):
    plan = promotion.plan_promotions([_cand("incumbent")])
    assert plan[0].gates_to_rerun == ()


def test_the_rebased_step_declares_its_gates_up_front(tmp_path):
    plan = promotion.plan_promotions([_cand("incumbent"), _cand("novelty")])
    second = plan[1]
    # the diff is unknown until the rebase actually happens, so the plan asks
    # for the safe superset and narrows it afterwards via gates_after_rebase
    assert set(second.gates_to_rerun) == {"smoke", "review", "activation", "routing"}


# ---- one window, possibly two generations --------------------------------


def test_two_passing_candidates_mean_two_generations(tmp_path):
    plan = promotion.plan_promotions([_cand("incumbent"), _cand("novelty")])
    assert len(plan) == 2, "a window may produce two gens, each a single change"
    assert len({s.proposal_id for s in plan}) == 2


def test_each_step_carries_the_proposal_it_came_from(tmp_path):
    plan = promotion.plan_promotions(
        [_cand("incumbent", pid="p_0007"), _cand("novelty", pid="p_0009")]
    )
    assert [s.proposal_id for s in plan] == ["p_0007", "p_0009"]


def test_duplicate_lanes_are_refused(tmp_path):
    # the selector guarantees one per lane; if two arrive, something upstream
    # broke and silently promoting both would hide it
    with pytest.raises(ValueError):
        promotion.plan_promotions(
            [_cand("novelty", pid="a"), _cand("novelty", pid="b")]
        )
