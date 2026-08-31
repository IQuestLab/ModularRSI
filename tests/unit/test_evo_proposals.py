"""P1-2b: the proposal backlog and its state machine.

A proposal is a *direction* — one intervention several findings agree on. It
exists before any file does, and it has to survive not being chosen: today the
consolidator is told to "pick the SINGLE most common gap" and everything else it
saw is discarded with the prompt, so a direction that is real but not the most
frequent this window can never accumulate.

Two rules here are load-bearing and both come from the plan:

* an unselected proposal stays ``open`` — not deleted, not overwritten, not
  demoted to a failure. It is not evidence of anything that the system chose
  something else.
* one failed implementation must NOT reach ``taboo``. The direction was never
  disproven; only one attempt at it was. Taboo-on-first-failure is how a good
  proposal gets one chance and then becomes permanently unaskable.
"""

import json

import pytest

from harbor.agents.terminus_2_modular.self_evo import proposals

pytestmark = pytest.mark.unit


def _new(tmp_path, **over):
    kw = dict(
        step="gen_3_candidate_100",
        lane="incumbent",
        action="modify",
        target_variant="agent_loop/confirm_exit",
        behavioral_delta="stop accepting a self-report as proof of completion",
        causal_hypothesis="the validation classifier passes on prose alone",
        finding_ids=["f_aaa"],
        support_tasks=["fix-git"],
    )
    kw.update(over)
    return proposals.create_proposal(tmp_path, **kw)


# ---- creation -------------------------------------------------------------


def test_new_proposal_starts_open(tmp_path):
    p = _new(tmp_path)
    assert p.state == "open"
    assert p.proposal_id


def test_proposal_keeps_its_contract_fields(tmp_path):
    p = _new(tmp_path)
    assert p.lane == "incumbent"
    assert p.action == "modify"
    assert p.target_variant == "agent_loop/confirm_exit"
    assert p.behavioral_delta.startswith("stop accepting")
    assert p.causal_hypothesis.startswith("the validation classifier")
    assert p.finding_ids == ["f_aaa"]
    assert p.support_tasks == ["fix-git"]


def test_two_proposals_get_distinct_ids(tmp_path):
    a = _new(tmp_path)
    b = _new(tmp_path, finding_ids=["f_bbb"], behavioral_delta="keep a plan")
    assert a.proposal_id != b.proposal_id


def test_id_is_not_derived_from_the_free_text_contract(tmp_path):
    # behavioral_delta / causal_hypothesis are readable contract text, NOT a
    # classification key — a semantic hash would drift and silently split or
    # merge proposals. Same text twice must still be two distinguishable rows.
    a = _new(tmp_path)
    b = _new(tmp_path, finding_ids=["f_bbb"])
    assert a.proposal_id != b.proposal_id


# ---- the legal path -------------------------------------------------------


def test_full_promotion_path(tmp_path):
    p = _new(tmp_path)
    for state in ("selected", "attempted", "promoted"):
        p = proposals.transition(tmp_path, p.proposal_id, step="s", to=state)
    assert p.state == "promoted"


def test_history_records_every_transition_with_its_reason(tmp_path):
    p = _new(tmp_path)
    proposals.transition(
        tmp_path, p.proposal_id, step="s1", to="selected", reason="top of lane"
    )
    p = proposals.transition(
        tmp_path, p.proposal_id, step="s2", to="attempted", reason="editor ran"
    )
    assert [(h["to"], h["reason"]) for h in p.history] == [
        ("selected", "top of lane"),
        ("attempted", "editor ran"),
    ]


# ---- illegal transitions --------------------------------------------------


def test_cannot_skip_straight_to_promoted(tmp_path):
    p = _new(tmp_path)
    with pytest.raises(proposals.IllegalTransition):
        proposals.transition(tmp_path, p.proposal_id, step="s", to="promoted")


def test_promoted_is_terminal_except_for_supersede(tmp_path):
    p = _new(tmp_path)
    for state in ("selected", "attempted", "promoted"):
        proposals.transition(tmp_path, p.proposal_id, step="s", to=state)
    with pytest.raises(proposals.IllegalTransition):
        proposals.transition(tmp_path, p.proposal_id, step="s", to="open")
    # a promoted variant CAN later be retired by a successor
    p = proposals.transition(tmp_path, p.proposal_id, step="s", to="superseded")
    assert p.state == "superseded"


def test_unknown_state_is_rejected(tmp_path):
    p = _new(tmp_path)
    with pytest.raises(proposals.IllegalTransition):
        proposals.transition(tmp_path, p.proposal_id, step="s", to="parked")


# ---- one failed implementation is not a refutation ------------------------


def test_rejected_implementation_can_be_retried(tmp_path):
    p = _new(tmp_path)
    for state in ("selected", "attempted", "rejected_implementation"):
        p = proposals.transition(tmp_path, p.proposal_id, step="s", to=state)
    p = proposals.transition(tmp_path, p.proposal_id, step="s2", to="open")
    assert p.state == "open"


def test_a_single_failed_implementation_cannot_go_taboo(tmp_path):
    p = _new(tmp_path)
    for state in ("selected", "attempted", "rejected_implementation"):
        proposals.transition(tmp_path, p.proposal_id, step="s", to=state)
    with pytest.raises(proposals.IllegalTransition):
        proposals.transition(tmp_path, p.proposal_id, step="s", to="taboo")


def test_taboo_opens_up_after_a_second_failed_attempt(tmp_path):
    p = _new(tmp_path)
    for state in ("selected", "attempted", "rejected_implementation", "open"):
        proposals.transition(tmp_path, p.proposal_id, step="s1", to=state)
    for state in ("selected", "attempted", "rejected_implementation"):
        proposals.transition(tmp_path, p.proposal_id, step="s2", to=state)
    p = proposals.transition(tmp_path, p.proposal_id, step="s2", to="taboo")
    assert p.state == "taboo"
    assert p.attempts == 2


def test_a_rejected_direction_can_go_taboo_immediately(tmp_path):
    # rejected_proposal means the DIRECTION was judged wrong — that is a
    # refutation, unlike a botched implementation of a sound direction.
    p = _new(tmp_path)
    for state in ("selected", "attempted", "rejected_proposal"):
        proposals.transition(tmp_path, p.proposal_id, step="s", to=state)
    p = proposals.transition(tmp_path, p.proposal_id, step="s", to="taboo")
    assert p.state == "taboo"


# ---- not being chosen costs a proposal nothing ----------------------------


def test_unselected_proposal_is_untouched_when_another_is_selected(tmp_path):
    winner = _new(tmp_path)
    loser = _new(tmp_path, finding_ids=["f_bbb"], behavioral_delta="keep a plan")
    proposals.transition(tmp_path, winner.proposal_id, step="s", to="selected")

    still = proposals.get_proposal(tmp_path, loser.proposal_id)
    assert still.state == "open"
    assert still.support_tasks == ["fix-git"]
    assert still.attempts == 0


def test_open_proposals_survive_across_windows(tmp_path):
    _new(tmp_path)
    later = proposals.open_proposals(tmp_path)
    assert [p.state for p in later] == ["open"]


# ---- accumulating support -------------------------------------------------


def test_linking_a_finding_grows_the_support_set(tmp_path):
    p = _new(tmp_path)
    p = proposals.link_finding(
        tmp_path,
        p.proposal_id,
        step="gen_4_candidate_200",
        finding_id="f_bbb",
        support_tasks=["posix-tar-r-w"],
        reason="same intervention, different task",
    )
    assert p.finding_ids == ["f_aaa", "f_bbb"]
    assert p.support_tasks == ["fix-git", "posix-tar-r-w"]


def test_support_tasks_are_deduped(tmp_path):
    p = _new(tmp_path)
    p = proposals.link_finding(
        tmp_path, p.proposal_id, step="s", finding_id="f_bbb", support_tasks=["fix-git"]
    )
    assert p.support_tasks == ["fix-git"]


def test_relinking_the_same_finding_is_a_no_op(tmp_path):
    # a resumed window re-ingests the same findings; double-counting support
    # would manufacture agreement that nobody observed.
    p = _new(tmp_path)
    proposals.link_finding(tmp_path, p.proposal_id, step="s", finding_id="f_bbb")
    p = proposals.link_finding(tmp_path, p.proposal_id, step="s", finding_id="f_bbb")
    assert p.finding_ids == ["f_aaa", "f_bbb"]


def test_a_finding_cannot_support_two_proposals(tmp_path):
    _new(tmp_path)  # owns f_aaa
    b = _new(tmp_path, finding_ids=["f_bbb"], behavioral_delta="keep a plan")
    with pytest.raises(proposals.AlreadyLinked):
        proposals.link_finding(tmp_path, b.proposal_id, step="s", finding_id="f_aaa")


def test_owner_of_a_finding_is_queryable(tmp_path):
    p = _new(tmp_path)
    assert proposals.proposal_of_finding(tmp_path, "f_aaa").proposal_id == p.proposal_id
    assert proposals.proposal_of_finding(tmp_path, "f_zzz") is None


def test_cannot_add_support_to_a_promoted_proposal(tmp_path):
    # once it is promoted, new same-direction evidence must be compared against
    # the UPDATED archive first (covered / partially covered), not silently
    # appended to a proposal that already shipped.
    p = _new(tmp_path)
    for state in ("selected", "attempted", "promoted"):
        proposals.transition(tmp_path, p.proposal_id, step="s", to=state)
    with pytest.raises(proposals.IllegalTransition):
        proposals.link_finding(tmp_path, p.proposal_id, step="s", finding_id="f_bbb")


# ---- the search view ------------------------------------------------------


def test_open_proposals_filter_by_lane(tmp_path):
    _new(tmp_path)
    _new(tmp_path, lane="novelty", action="add", target_variant="", finding_ids=["f_b"])
    assert [p.lane for p in proposals.open_proposals(tmp_path, lane="novelty")] == [
        "novelty"
    ]


def test_open_proposals_filter_by_target_variant(tmp_path):
    # clustering step 2: an incumbent finding only compares against open
    # proposals on the SAME target variant.
    _new(tmp_path)
    _new(tmp_path, target_variant="agent_loop/budget_aware", finding_ids=["f_b"])
    got = proposals.open_proposals(
        tmp_path, lane="incumbent", target_variant="agent_loop/budget_aware"
    )
    assert [p.target_variant for p in got] == ["agent_loop/budget_aware"]


def test_closed_proposals_are_not_in_the_open_view(tmp_path):
    p = _new(tmp_path)
    proposals.transition(tmp_path, p.proposal_id, step="s", to="selected")
    assert proposals.open_proposals(tmp_path) == []


# ---- durability -----------------------------------------------------------


def test_store_is_an_append_only_event_log(tmp_path):
    p = _new(tmp_path)
    proposals.transition(tmp_path, p.proposal_id, step="s", to="selected")
    rows = [
        json.loads(x)
        for x in proposals.proposals_path(tmp_path).read_text().splitlines()
        if x.strip()
    ]
    assert [r["event"] for r in rows] == ["create", "transition"]


def test_state_survives_a_reload(tmp_path):
    p = _new(tmp_path)
    proposals.transition(tmp_path, p.proposal_id, step="s", to="selected")
    assert proposals.get_proposal(tmp_path, p.proposal_id).state == "selected"


def test_a_corrupt_event_line_does_not_lose_the_backlog(tmp_path):
    p = _new(tmp_path)
    path = proposals.proposals_path(tmp_path)
    path.write_text(path.read_text() + "{ broken\n")
    proposals.transition(tmp_path, p.proposal_id, step="s", to="selected")
    assert proposals.get_proposal(tmp_path, p.proposal_id).state == "selected"


def test_transition_on_an_unknown_proposal_raises(tmp_path):
    with pytest.raises(KeyError):
        proposals.transition(tmp_path, "p_nope", step="s", to="selected")


def test_a_lost_create_line_cannot_make_a_new_proposal_reuse_a_live_id(tmp_path):
    # IDs must not be minted from a count of what parsed: if p_0001's create line
    # is damaged, the transitions still naming p_0001 are live history. Handing
    # that ID to a different direction would silently merge two proposals.
    first = _new(tmp_path)
    proposals.transition(tmp_path, first.proposal_id, step="s", to="selected")
    path = proposals.proposals_path(tmp_path)
    lines = path.read_text().splitlines()
    lines[0] = "{ corrupted create"
    path.write_text("\n".join(lines) + "\n")

    second = _new(tmp_path, finding_ids=["f_bbb"])
    assert second.proposal_id != first.proposal_id
