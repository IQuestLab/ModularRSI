"""Cluster routed findings into persistent intervention proposals.

A finding links only when pairwise comparison says it describes the same
behavioral intervention; otherwise it creates a new auditable proposal.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from harbor.agents.terminus_2_modular.self_evo import dispositions as _dispositions
from harbor.agents.terminus_2_modular.self_evo import proposals as _proposals
from harbor.agents.terminus_2_modular.self_evo.backlog import (
    FindingRecord,
    backlog_dir,
)

AUDIT_FILENAME = "cluster_audit.jsonl"

LANE_INCUMBENT = "incumbent"
LANE_NOVELTY = "novelty"
LANE_OUT_OF_SCOPE = "out_of_scope"

_MAX_CHAIN_DEPTH = 20


@dataclass
class Routing:
    """Step 1's verdict: where this finding sits relative to the live archive."""

    lane: str
    action: str  # modify | replace | add
    target_variant: str = ""  # "type/name"; required for incumbent
    behavioral_delta: str = ""
    causal_hypothesis: str = ""
    rationale: str = ""  # which variant, cited to a method/line
    #: novelty only: per-variant "why this one is not it". This is the claim that
    #: actually turned out wrong — an independent judge overturned 13 of 15
    #: novelty calls — so it has to survive into the audit log. A judgement whose
    #: justification is discarded cannot be checked afterwards.
    dismissals: dict[str, str] = field(default_factory=dict)
    #: out_of_scope only: the module that DOES own this problem. The router is
    #: required to name it, and it is the one thing this run can hand to a
    #: different lineage — dropping it threw that away.
    other_module: str = ""


@dataclass
class ClusterOutcome:
    decision: str
    proposal_id: str | None = None
    reason: str = ""
    target_variant: str = ""
    comparisons: list[dict[str, Any]] = field(default_factory=list)


class ClusterHooks(Protocol):
    """The two semantic judgements, injected so the sequencing stays pure."""

    def same_intervention(
        self, finding: FindingRecord, routing: Routing, proposal: Any
    ) -> tuple[bool, str]: ...

    def successor_covers(
        self, finding: FindingRecord, successor_qual: str
    ) -> tuple[bool, str]: ...


def audit_path(archive_root: Path | str) -> Path:
    return backlog_dir(archive_root) / AUDIT_FILENAME


def _audit(archive_root: Path | str, row: dict[str, Any]) -> None:
    path = audit_path(archive_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _qual(entry: Any) -> str:
    return f"{entry.type}/{entry.name}"


def _parent_quals(entry: Any) -> list[str]:
    """This entry's parents, every one of them qualified.

    ``parent_ids`` comes from the editor's ``PARENT:`` line verbatim, and
    editors write it both ways: ``agent_loop/confirm_exit`` and, more often,
    just ``confirm_exit`` — the name they read off a filename. Comparing the
    bare form against a qualified frontier never matches, so a superseded
    variant looked childless and every finding aimed at it was dropped as
    `stale_evidence`.

    A bare name resolves within the CHILD's own type. Two module types may each
    own a `confirm_exit`; reading a bare parent as "whichever type is being
    looked up" would splice one type's variant into another type's lineage.
    """
    return [
        p if "/" in p else f"{entry.type}/{p}" for p in (entry.parent_ids or []) if p
    ]


def resolve_live_variant(entries: list[Any], qual: str) -> tuple[str | None, str]:
    """Map a possibly-retired variant to the living descendant that replaced it.

    Returns ``(qual, "active")`` when it is still active, ``(successor, "mapped")``
    after following ``parent_ids`` to an active descendant, or ``(None, reason)``
    when the variant is unknown to the archive or its line died out.
    """
    by_qual = {_qual(e): e for e in entries}
    entry = by_qual.get(qual)
    if entry is None:
        return None, "unknown_variant"
    if entry.status == "active":
        return qual, "active"

    frontier = [qual]
    seen = {qual}
    for _ in range(_MAX_CHAIN_DEPTH):
        children = [
            e
            for e in entries
            if any(parent in frontier for parent in _parent_quals(e))
            and _qual(e) not in seen
        ]
        if not children:
            break
        for child in children:
            if child.status == "active":
                return _qual(child), "mapped"
        frontier = [_qual(c) for c in children]
        seen.update(frontier)
    return None, "chain_broken"


def _candidates(archive_root: Path | str, routing: Routing) -> list[Any]:
    if routing.lane == LANE_NOVELTY:
        return _proposals.open_proposals(archive_root, lane=LANE_NOVELTY)
    return _proposals.open_proposals(
        archive_root, lane=LANE_INCUMBENT, target_variant=routing.target_variant
    )


def cluster_finding(
    archive_root: Path | str,
    *,
    finding: FindingRecord,
    routing: Routing,
    hooks: ClusterHooks,
    step: str,
    archive_entries: list[Any] | None = None,
) -> ClusterOutcome:
    """Run one finding through the five steps. Never raises into the reflection."""

    #: decision → what the ledger should remember. A decision absent here is
    #: either already recorded by the proposal itself (linked / created /
    #: already_clustered) or has no verdict to remember.
    ledger_kinds = {
        "out_of_scope": _dispositions.OUT_OF_SCOPE,
        "covered_by_successor": _dispositions.COVERED,
        "stale_evidence": _dispositions.STALE,
        "invalid_routing": _dispositions.INVALID,
        "comparison_unresolved": _dispositions.RETRYABLE,
    }

    def done(outcome: ClusterOutcome) -> ClusterOutcome:
        kind = ledger_kinds.get(outcome.decision)
        if kind is not None:
            _dispositions.record(
                archive_root,
                finding_id=finding.finding_id,
                kind=kind,
                step=step,
                module=routing.other_module,
                successor=(
                    outcome.target_variant
                    if outcome.decision == "covered_by_successor"
                    else ""
                ),
                reason=outcome.reason,
            )
        _audit(
            archive_root,
            {
                "step": step,
                "finding_id": finding.finding_id,
                "task": finding.task,
                "lane": routing.lane,
                "action": routing.action,
                "target_variant": outcome.target_variant or routing.target_variant,
                "rationale": routing.rationale,
                "dismissals": dict(routing.dismissals),
                "decision": outcome.decision,
                "proposal_id": outcome.proposal_id,
                "reason": outcome.reason,
                "comparisons": outcome.comparisons,
            },
        )
        return outcome

    already = _proposals.proposal_of_finding(archive_root, finding.finding_id)
    if already is not None:
        return done(
            ClusterOutcome(
                decision="already_clustered",
                proposal_id=already.proposal_id,
                reason="this finding is already attached — resumed window",
            )
        )

    # This module is not the place for another module's work: an out_of_scope
    # finding goes to that module's backlog and takes no slot here.
    if routing.lane == LANE_OUT_OF_SCOPE:
        return done(
            ClusterOutcome(
                decision="out_of_scope",
                reason="belongs to another module — cross-module backlog",
            )
        )

    target = routing.target_variant
    if routing.lane == LANE_INCUMBENT:
        if not target:
            return done(
                ClusterOutcome(
                    decision="invalid_routing",
                    reason="incumbent must name the variant that covers it",
                )
            )
        live, how = resolve_live_variant(archive_entries or [], target)
        if live is None:
            return done(
                ClusterOutcome(
                    decision="stale_evidence",
                    reason=f"{target}: {how}",
                    target_variant=target,
                )
            )
        if how == "mapped":
            covered, why = hooks.successor_covers(finding, live)
            if covered:
                return done(
                    ClusterOutcome(
                        decision="covered_by_successor",
                        reason=f"{target} → {live}: {why}",
                        target_variant=live,
                    )
                )
        target = live

    routing = Routing(
        lane=routing.lane,
        action=routing.action,
        target_variant=target,
        behavioral_delta=routing.behavioral_delta,
        causal_hypothesis=routing.causal_hypothesis,
        rationale=routing.rationale,
        dismissals=dict(routing.dismissals),
    )

    comparisons: list[dict[str, Any]] = []
    #: candidates the judge could not answer for. NOT the same fact as "these
    #: are different" — creating on an unknown manufactures exactly the
    #: duplicate the comparison exists to prevent.
    unresolved: list[str] = []
    for candidate in _candidates(archive_root, routing):
        same, why = hooks.same_intervention(finding, routing, candidate)
        comparisons.append(
            {
                "proposal_id": candidate.proposal_id,
                "same": None if same is None else bool(same),
                "reason": why,
            }
        )
        if same is None:
            unresolved.append(candidate.proposal_id)
        if same:
            _proposals.link_finding(
                archive_root,
                candidate.proposal_id,
                step=step,
                finding_id=finding.finding_id,
                support_tasks=finding.support_tasks,
                reason=why,
            )
            return done(
                ClusterOutcome(
                    decision="linked",
                    proposal_id=candidate.proposal_id,
                    reason=why,
                    target_variant=target,
                    comparisons=comparisons,
                )
            )

    if unresolved:
        # Leave the finding unattached. Next window re-routes it — one routing
        # call — instead of the backlog carrying a permanent twin that splits
        # this direction's support in half for the rest of the run.
        return done(
            ClusterOutcome(
                decision="comparison_unresolved",
                reason="could not compare against " + ", ".join(unresolved),
                target_variant=target,
                comparisons=comparisons,
            )
        )

    created = _proposals.create_proposal(
        archive_root,
        step=step,
        lane=routing.lane,
        action=routing.action,
        target_variant=target,
        behavioral_delta=routing.behavioral_delta,
        causal_hypothesis=routing.causal_hypothesis,
        finding_ids=[finding.finding_id],
        support_tasks=finding.support_tasks,
    )
    return done(
        ClusterOutcome(
            decision="created",
            proposal_id=created.proposal_id,
            reason="no open proposal points at the same intervention",
            target_variant=target,
            comparisons=comparisons,
        )
    )
