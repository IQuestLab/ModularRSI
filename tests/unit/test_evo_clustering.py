"""P1-3b: the five-step clustering that turns findings into proposals.

    1. compare the finding against the ACTIVE archive → lane + action
    2. incumbent: compare only against open proposals on the SAME target variant
       novelty:   compare against open novelty proposals
    3. link only when both point at the same intervention
    4. otherwise create a new proposal
    5. every comparison, with its reason, goes to the audit log

Step 1 is the router's job and arrives here as a `Routing`. Everything else is
sequencing, which is what this module owns; the semantic judgements are injected
hooks so the ordering can be tested without an endpoint.

The subtle part is stale evidence. Reflection is non-blocking, so by the time a
finding is routed the variant it accuses may already have been superseded. The
rule is to re-aim at the successor and re-ask against the successor's ACTUAL
code — did the child inherit the gap? — with three outcomes, none of them a
silent drop.
"""

import json

import pytest

from harbor.agents.terminus_2_modular import archive as _archive
from harbor.agents.terminus_2_modular.self_evo import backlog, clustering, proposals

pytestmark = pytest.mark.unit

STEP = "gen_5_candidate_900"


def _entry(name, status="active", parents=(), type_="agent_loop"):
    return _archive.ArchiveEntry(
        name=name, type=type_, status=status, parent_ids=list(parents)
    )


def _finding(tmp_path, task="fix-git", index=0):
    payload = {
        "step": STEP,
        "findings": [
            {"task": task, "is_culprit": True, "lens": "agent_loop", "divergence": "x"}
            for _ in range(index + 1)
        ],
    }
    return backlog.ingest_step(tmp_path, payload)[index]


def _routing(**over):
    kw = dict(
        lane="incumbent",
        action="modify",
        target_variant="agent_loop/confirm_exit",
        behavioral_delta="stop trusting a self-report",
        causal_hypothesis="the classifier passes on prose",
        rationale="confirm_exit._validate accepts any non-empty answer",
    )
    kw.update(over)
    return clustering.Routing(**kw)


class _Hooks:
    """Injected judgement. Records what it was asked, so ordering is testable."""

    def __init__(self, same=False, covers=False):
        self._same = same
        self._covers = covers
        self.compared: list[str] = []
        self.coverage_asked: list[str] = []

    def same_intervention(self, finding, routing, proposal):
        self.compared.append(proposal.proposal_id)
        return (self._same, "same intervention" if self._same else "different fix")

    def successor_covers(self, finding, successor_qual):
        self.coverage_asked.append(successor_qual)
        return (
            self._covers,
            "child already checks artifacts" if self._covers else "gap survived",
        )


def _audit(tmp_path):
    path = clustering.audit_path(tmp_path)
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


# ---- out of scope ---------------------------------------------------------


def test_out_of_scope_finding_creates_no_proposal_here(tmp_path):
    # 92/318 real findings named another module. The investigator's abstention
    # works; the leak is the other way — once is_culprit is true the consolidator
    # is told it may only edit `{locked}/`, so another module's job gets
    # rationalised into this one.
    out = clustering.cluster_finding(
        tmp_path,
        finding=_finding(tmp_path),
        routing=_routing(lane="out_of_scope", action="", target_variant=""),
        hooks=_Hooks(),
        step=STEP,
    )
    assert out.decision == "out_of_scope"
    assert out.proposal_id is None
    assert proposals.load_proposals(tmp_path) == []


def test_out_of_scope_finding_is_still_audited(tmp_path):
    clustering.cluster_finding(
        tmp_path,
        finding=_finding(tmp_path),
        routing=_routing(lane="out_of_scope", action="", target_variant=""),
        hooks=_Hooks(),
        step=STEP,
    )
    assert _audit(tmp_path)[-1]["decision"] == "out_of_scope"


# ---- create / link --------------------------------------------------------


def test_first_finding_of_a_direction_creates_a_proposal(tmp_path):
    out = clustering.cluster_finding(
        tmp_path,
        finding=_finding(tmp_path),
        routing=_routing(),
        hooks=_Hooks(),
        step=STEP,
        archive_entries=[_entry("confirm_exit")],
    )
    assert out.decision == "created"
    (p,) = proposals.load_proposals(tmp_path)
    assert p.lane == "incumbent"
    assert p.action == "modify"
    assert p.target_variant == "agent_loop/confirm_exit"
    assert p.support_tasks == ["fix-git"]


def test_a_matching_finding_joins_the_existing_proposal(tmp_path):
    first = _finding(tmp_path, task="fix-git", index=0)
    clustering.cluster_finding(
        tmp_path,
        finding=first,
        routing=_routing(),
        hooks=_Hooks(),
        step=STEP,
        archive_entries=[_entry("confirm_exit")],
    )
    second = backlog.ingest_step(
        tmp_path,
        {
            "step": "gen_6_c",
            "findings": [{"task": "posix-tar-r-w", "is_culprit": True}],
        },
    )[0]
    out = clustering.cluster_finding(
        tmp_path,
        finding=second,
        routing=_routing(),
        hooks=_Hooks(same=True),
        step="gen_6_c",
        archive_entries=[_entry("confirm_exit")],
    )
    assert out.decision == "linked"
    (p,) = proposals.load_proposals(tmp_path)
    assert len(p.finding_ids) == 2
    assert p.support_tasks == ["fix-git", "posix-tar-r-w"]


def test_a_different_intervention_on_the_same_variant_is_a_second_proposal(tmp_path):
    # "fix the validation classifier" and "add missing-tool recovery" both point
    # at confirm_exit and are still two proposals — target_variant narrows the
    # comparison set, it is not the identity.
    clustering.cluster_finding(
        tmp_path,
        finding=_finding(tmp_path),
        routing=_routing(),
        hooks=_Hooks(),
        step=STEP,
        archive_entries=[_entry("confirm_exit")],
    )
    other = backlog.ingest_step(
        tmp_path, {"step": "gen_6_c", "findings": [{"task": "t2", "is_culprit": True}]}
    )[0]
    out = clustering.cluster_finding(
        tmp_path,
        finding=other,
        routing=_routing(behavioral_delta="recover from a missing tool"),
        hooks=_Hooks(same=False),
        step="gen_6_c",
        archive_entries=[_entry("confirm_exit")],
    )
    assert out.decision == "created"
    assert len(proposals.load_proposals(tmp_path)) == 2


# ---- step 2: the comparison set is narrowed before any judgement ----------


def test_incumbent_only_compares_within_the_same_target_variant(tmp_path):
    proposals.create_proposal(
        tmp_path,
        step="s",
        lane="incumbent",
        action="modify",
        target_variant="agent_loop/budget_aware",
        finding_ids=["f_other"],
    )
    proposals.create_proposal(
        tmp_path,
        step="s",
        lane="incumbent",
        action="modify",
        target_variant="agent_loop/confirm_exit",
        finding_ids=["f_same"],
    )
    hooks = _Hooks()
    clustering.cluster_finding(
        tmp_path,
        finding=_finding(tmp_path),
        routing=_routing(),
        hooks=hooks,
        step=STEP,
        archive_entries=[_entry("confirm_exit"), _entry("budget_aware")],
    )
    assert hooks.compared == ["p_0002"]  # budget_aware never offered


def test_novelty_only_compares_against_open_novelty_proposals(tmp_path):
    proposals.create_proposal(
        tmp_path,
        step="s",
        lane="incumbent",
        action="modify",
        target_variant="agent_loop/confirm_exit",
        finding_ids=["f_inc"],
    )
    proposals.create_proposal(
        tmp_path,
        step="s",
        lane="novelty",
        action="add",
        finding_ids=["f_nov"],
    )
    hooks = _Hooks()
    clustering.cluster_finding(
        tmp_path,
        finding=_finding(tmp_path),
        routing=_routing(lane="novelty", action="add", target_variant=""),
        hooks=hooks,
        step=STEP,
    )
    assert hooks.compared == ["p_0002"]


def test_a_closed_proposal_is_never_offered_for_comparison(tmp_path):
    p = proposals.create_proposal(
        tmp_path,
        step="s",
        lane="incumbent",
        action="modify",
        target_variant="agent_loop/confirm_exit",
        finding_ids=["f_x"],
    )
    proposals.transition(tmp_path, p.proposal_id, step="s", to="selected")
    hooks = _Hooks(same=True)
    out = clustering.cluster_finding(
        tmp_path,
        finding=_finding(tmp_path),
        routing=_routing(),
        hooks=hooks,
        step=STEP,
        archive_entries=[_entry("confirm_exit")],
    )
    assert hooks.compared == []
    assert out.decision == "created"


# ---- stale evidence: the accused variant moved under us -------------------


def test_finding_is_re_aimed_at_the_active_successor(tmp_path):
    entries = [
        _entry("confirm_exit", status="superseded"),
        _entry("deadline_aware", parents=["agent_loop/confirm_exit"]),
    ]
    out = clustering.cluster_finding(
        tmp_path,
        finding=_finding(tmp_path),
        routing=_routing(),
        hooks=_Hooks(covers=False),
        step=STEP,
        archive_entries=entries,
    )
    assert out.decision == "created"
    (p,) = proposals.load_proposals(tmp_path)
    assert p.target_variant == "agent_loop/deadline_aware"


def test_a_successor_that_already_covers_the_gap_absorbs_the_finding(tmp_path):
    entries = [
        _entry("confirm_exit", status="superseded"),
        _entry("deadline_aware", parents=["agent_loop/confirm_exit"]),
    ]
    out = clustering.cluster_finding(
        tmp_path,
        finding=_finding(tmp_path),
        routing=_routing(),
        hooks=_Hooks(covers=True),
        step=STEP,
        archive_entries=entries,
    )
    assert out.decision == "covered_by_successor"
    assert proposals.load_proposals(tmp_path) == []


def test_the_coverage_question_is_asked_about_the_successor_not_the_parent(tmp_path):
    entries = [
        _entry("confirm_exit", status="superseded"),
        _entry("deadline_aware", parents=["agent_loop/confirm_exit"]),
    ]
    hooks = _Hooks(covers=True)
    clustering.cluster_finding(
        tmp_path,
        finding=_finding(tmp_path),
        routing=_routing(),
        hooks=hooks,
        step=STEP,
        archive_entries=entries,
    )
    assert hooks.coverage_asked == ["agent_loop/deadline_aware"]


def test_a_multi_hop_chain_is_followed_to_the_living_variant(tmp_path):
    entries = [
        _entry("confirm_exit", status="superseded"),
        _entry(
            "budget_aware", status="superseded", parents=["agent_loop/confirm_exit"]
        ),
        _entry("deadline_aware", parents=["agent_loop/budget_aware"]),
    ]
    out = clustering.cluster_finding(
        tmp_path,
        finding=_finding(tmp_path),
        routing=_routing(),
        hooks=_Hooks(),
        step=STEP,
        archive_entries=entries,
    )
    assert (
        proposals.load_proposals(tmp_path)[0].target_variant
        == "agent_loop/deadline_aware"
    )
    assert out.decision == "created"


def test_a_broken_chain_is_recorded_never_silently_dropped(tmp_path):
    entries = [_entry("confirm_exit", status="superseded")]  # no successor
    out = clustering.cluster_finding(
        tmp_path,
        finding=_finding(tmp_path),
        routing=_routing(),
        hooks=_Hooks(),
        step=STEP,
        archive_entries=entries,
    )
    assert out.decision == "stale_evidence"
    assert proposals.load_proposals(tmp_path) == []
    assert _audit(tmp_path)[-1]["decision"] == "stale_evidence"


def test_a_variant_missing_from_the_archive_is_stale_evidence(tmp_path):
    out = clustering.cluster_finding(
        tmp_path,
        finding=_finding(tmp_path),
        routing=_routing(),
        hooks=_Hooks(),
        step=STEP,
        archive_entries=[_entry("something_else")],
    )
    assert out.decision == "stale_evidence"


def test_incumbent_routing_without_a_variant_is_refused_not_guessed(tmp_path):
    out = clustering.cluster_finding(
        tmp_path,
        finding=_finding(tmp_path),
        routing=_routing(target_variant=""),
        hooks=_Hooks(),
        step=STEP,
        archive_entries=[_entry("confirm_exit")],
    )
    assert out.decision == "invalid_routing"
    assert proposals.load_proposals(tmp_path) == []


# ---- resume ---------------------------------------------------------------


def test_clustering_the_same_finding_twice_is_a_no_op(tmp_path):
    finding = _finding(tmp_path)
    kw = dict(routing=_routing(), step=STEP, archive_entries=[_entry("confirm_exit")])
    clustering.cluster_finding(tmp_path, finding=finding, hooks=_Hooks(), **kw)
    out = clustering.cluster_finding(tmp_path, finding=finding, hooks=_Hooks(), **kw)
    assert out.decision == "already_clustered"
    assert len(proposals.load_proposals(tmp_path)) == 1


# ---- step 5: everything is auditable --------------------------------------


def test_every_comparison_is_written_with_its_reason(tmp_path):
    proposals.create_proposal(
        tmp_path,
        step="s",
        lane="incumbent",
        action="modify",
        target_variant="agent_loop/confirm_exit",
        finding_ids=["f_x"],
    )
    clustering.cluster_finding(
        tmp_path,
        finding=_finding(tmp_path),
        routing=_routing(),
        hooks=_Hooks(same=False),
        step=STEP,
        archive_entries=[_entry("confirm_exit")],
    )
    row = _audit(tmp_path)[-1]
    assert row["comparisons"] == [
        {"proposal_id": "p_0001", "same": False, "reason": "different fix"}
    ]
    assert row["finding_id"]
    assert row["lane"] == "incumbent"
    assert row["rationale"].startswith("confirm_exit._validate")


# ---- the shape the REAL archive writes ------------------------------------
#
# Every test above builds `parent_ids` qualified — "agent_loop/confirm_exit" —
# because that is what `ArchiveEntry.qual` returns and what the docstring shows.
# The archive on disk is not so consistent: `parse_variant_meta` copies the
# editor's `PARENT:` line through verbatim, and an editor that writes
# `PARENT: confirm_exit` (the name it just read out of a filename) produces a
# bare id. `archive.lineage_lines` already compensates for this; the resolver
# did not, so on real data a superseded variant looked like it had no children
# at all and every finding against it was dropped as `stale_evidence` — the one
# outcome that costs a routing call and yields nothing.


def test_a_successor_is_found_through_an_unqualified_parent_id(tmp_path):
    # `PARENT: confirm_exit` — no type prefix, which is what the archive
    # actually contains for most evolved variants
    entries = [
        _entry("confirm_exit", status="superseded"),
        _entry("deadline_aware", parents=["confirm_exit"]),
    ]
    out = clustering.cluster_finding(
        tmp_path,
        finding=_finding(tmp_path),
        routing=_routing(),
        hooks=_Hooks(covers=False),
        step=STEP,
        archive_entries=entries,
    )
    assert out.decision == "created"
    (p,) = proposals.load_proposals(tmp_path)
    assert p.target_variant == "agent_loop/deadline_aware"


def test_an_unqualified_parent_does_not_cross_module_types(tmp_path):
    # a bare name is resolved WITHIN the child's own type. Two modules may both
    # own a `confirm_exit`; reading a bare parent as "whichever type I am
    # looking for" would graft an observation variant onto an agent_loop line.
    entries = [
        _entry("confirm_exit", status="superseded", type_="agent_loop"),
        _entry("deadline_aware", parents=["confirm_exit"], type_="observation"),
    ]
    out = clustering.cluster_finding(
        tmp_path,
        finding=_finding(tmp_path),
        routing=_routing(),
        hooks=_Hooks(),
        step=STEP,
        archive_entries=entries,
    )
    assert out.decision == "stale_evidence"


def test_a_mixed_chain_resolves_hop_by_hop(tmp_path):
    # real lineages mix both forms — one editor wrote the type, the next did not
    entries = [
        _entry("confirm_exit", status="superseded"),
        _entry("budget_aware", status="superseded", parents=["confirm_exit"]),
        _entry("deadline_aware", parents=["agent_loop/budget_aware"]),
    ]
    clustering.cluster_finding(
        tmp_path,
        finding=_finding(tmp_path),
        routing=_routing(),
        hooks=_Hooks(),
        step=STEP,
        archive_entries=entries,
    )
    assert (
        proposals.load_proposals(tmp_path)[0].target_variant
        == "agent_loop/deadline_aware"
    )


# ---- M3: terminal outcomes are written down, retryable ones are not -------
#
# Without this the resume check ("is this finding attached to a proposal?")
# answers no for every finding that ended any other way, and re-routes it every
# window for the rest of the run — paying a routing call each time to re-reach
# a verdict already reached.


def _dispositions(tmp_path):
    from harbor.agents.terminus_2_modular.self_evo import dispositions

    return {d.finding_id: d for d in dispositions.load(tmp_path)}


def test_out_of_scope_remembers_which_module_owns_it(tmp_path):
    from harbor.agents.terminus_2_modular.self_evo import dispositions

    finding = _finding(tmp_path)
    clustering.cluster_finding(
        tmp_path,
        finding=finding,
        routing=_routing(
            lane="out_of_scope", action="", target_variant="", other_module="tools"
        ),
        hooks=_Hooks(),
        step=STEP,
        archive_entries=[_entry("confirm_exit")],
    )
    row = _dispositions(tmp_path)[finding.finding_id]
    assert row.kind == dispositions.OUT_OF_SCOPE
    assert row.module == "tools"
    # and it is handed to the run that locks that module, on a known path
    assert dispositions.out_of_scope_for(tmp_path, "tools") == [finding.finding_id]


def test_a_successor_that_absorbed_the_finding_is_recorded(tmp_path):
    from harbor.agents.terminus_2_modular.self_evo import dispositions

    finding = _finding(tmp_path)
    clustering.cluster_finding(
        tmp_path,
        finding=finding,
        routing=_routing(),
        hooks=_Hooks(covers=True),
        step=STEP,
        archive_entries=[
            _entry("confirm_exit", status="superseded"),
            _entry("deadline_aware", parents=["agent_loop/confirm_exit"]),
        ],
    )
    row = _dispositions(tmp_path)[finding.finding_id]
    assert row.kind == dispositions.COVERED
    assert row.successor == "agent_loop/deadline_aware"


def test_a_dead_lineage_is_recorded(tmp_path):
    from harbor.agents.terminus_2_modular.self_evo import dispositions

    finding = _finding(tmp_path)
    clustering.cluster_finding(
        tmp_path,
        finding=finding,
        routing=_routing(),
        hooks=_Hooks(),
        step=STEP,
        archive_entries=[_entry("confirm_exit", status="superseded")],
    )
    assert _dispositions(tmp_path)[finding.finding_id].kind == dispositions.STALE


def test_an_incumbent_without_a_target_is_recorded_as_invalid(tmp_path):
    from harbor.agents.terminus_2_modular.self_evo import dispositions

    finding = _finding(tmp_path)
    clustering.cluster_finding(
        tmp_path,
        finding=finding,
        routing=_routing(target_variant=""),
        hooks=_Hooks(),
        step=STEP,
        archive_entries=[_entry("confirm_exit")],
    )
    assert _dispositions(tmp_path)[finding.finding_id].kind == dispositions.INVALID


def test_an_unresolved_comparison_is_recorded_but_stays_routable(tmp_path):
    from harbor.agents.terminus_2_modular.self_evo import dispositions

    proposals.create_proposal(
        tmp_path,
        step="gen_4_candidate_1",
        lane="incumbent",
        action="modify",
        target_variant="agent_loop/confirm_exit",
        behavioral_delta="something else",
        causal_hypothesis="unrelated",
        finding_ids=["f_seed"],
        support_tasks=["seed"],
    )
    finding = _finding(tmp_path)
    out = clustering.cluster_finding(
        tmp_path,
        finding=finding,
        routing=_routing(),
        hooks=_Hooks(same=None),
        step=STEP,
        archive_entries=[_entry("confirm_exit")],
    )
    assert out.decision == "comparison_unresolved"
    assert _dispositions(tmp_path)[finding.finding_id].kind == dispositions.RETRYABLE
    # the whole point: a bad afternoon at the endpoint must not retire evidence
    assert dispositions.terminal_finding_ids(tmp_path) == set()
