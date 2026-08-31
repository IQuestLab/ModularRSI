"""Persistent proposal state machine for candidate interventions.

Unselected proposals remain open. Implementation failure is distinct from
proposal rejection so a sound direction can receive a bounded retry.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from harbor.agents.terminus_2_modular.self_evo.backlog import backlog_dir

PROPOSALS_FILENAME = "proposals.jsonl"

OPEN = "open"
SELECTED = "selected"
ATTEMPTED = "attempted"
PROMOTED = "promoted"
REJECTED_PROPOSAL = "rejected_proposal"
REJECTED_IMPLEMENTATION = "rejected_implementation"
SUPERSEDED = "superseded"
TABOO = "taboo"

STATES = frozenset(
    {
        OPEN,
        SELECTED,
        ATTEMPTED,
        PROMOTED,
        REJECTED_PROPOSAL,
        REJECTED_IMPLEMENTATION,
        SUPERSEDED,
        TABOO,
    }
)

#: Legal moves. `rejected_implementation → open` is the retry lane that keeps a
#: sound direction alive after a bad build.
_ALLOWED: dict[str, frozenset[str]] = {
    OPEN: frozenset({SELECTED, REJECTED_PROPOSAL, SUPERSEDED}),
    SELECTED: frozenset({ATTEMPTED, OPEN, REJECTED_PROPOSAL}),
    ATTEMPTED: frozenset(
        {PROMOTED, REJECTED_PROPOSAL, REJECTED_IMPLEMENTATION, SUPERSEDED}
    ),
    REJECTED_IMPLEMENTATION: frozenset({OPEN, SELECTED, TABOO, SUPERSEDED}),
    REJECTED_PROPOSAL: frozenset({TABOO, OPEN}),
    PROMOTED: frozenset({SUPERSEDED}),
    SUPERSEDED: frozenset(),
    TABOO: frozenset(),
}

#: How many failed implementations before a direction may be given up on.
TABOO_AFTER_ATTEMPTS = 2


class IllegalTransition(ValueError):
    """A move the state machine refuses — including taboo-on-first-failure."""


class AlreadyLinked(ValueError):
    """This finding already supports a different proposal."""


@dataclass
class ProposalRecord:
    proposal_id: str
    state: str = OPEN
    lane: str = ""  # incumbent | novelty | out_of_scope
    action: str = ""  # modify | replace | add
    target_variant: str = ""
    behavioral_delta: str = ""
    causal_hypothesis: str = ""
    parent_tree_hash: str = ""
    finding_ids: list[str] = field(default_factory=list)
    support_tasks: list[str] = field(default_factory=list)
    created_step: str = ""
    updated_step: str = ""
    attempts: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)


def proposals_path(archive_root: Path | str) -> Path:
    return backlog_dir(archive_root) / PROPOSALS_FILENAME


def _read_events(archive_root: Path | str) -> list[dict[str, Any]]:
    try:
        text = proposals_path(archive_root).read_text()
    except FileNotFoundError:
        return []
    out: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue  # a corrupt line costs that line, never the backlog
        if isinstance(row, dict) and row.get("proposal_id"):
            out.append(row)
    return out


def _append(archive_root: Path | str, event: dict[str, Any]) -> None:
    path = proposals_path(archive_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")


def _extend_unique(target: list[str], items) -> None:
    for item in items or []:
        if item and item not in target:
            target.append(item)


def _fold(events: list[dict[str, Any]]) -> dict[str, ProposalRecord]:
    by_id: dict[str, ProposalRecord] = {}
    for ev in events:
        pid = ev["proposal_id"]
        kind = ev.get("event")
        if kind == "create":
            rec = ProposalRecord(
                proposal_id=pid,
                lane=str(ev.get("lane") or ""),
                action=str(ev.get("action") or ""),
                target_variant=str(ev.get("target_variant") or ""),
                behavioral_delta=str(ev.get("behavioral_delta") or ""),
                causal_hypothesis=str(ev.get("causal_hypothesis") or ""),
                parent_tree_hash=str(ev.get("parent_tree_hash") or ""),
                created_step=str(ev.get("step") or ""),
                updated_step=str(ev.get("step") or ""),
            )
            _extend_unique(rec.finding_ids, ev.get("finding_ids"))
            _extend_unique(rec.support_tasks, ev.get("support_tasks"))
            by_id[pid] = rec
            continue
        rec = by_id.get(pid)
        if rec is None:
            continue  # an event for a proposal whose create line was lost
        rec.updated_step = str(ev.get("step") or rec.updated_step)
        if kind == "support":
            _extend_unique(rec.finding_ids, ev.get("finding_ids"))
            _extend_unique(rec.support_tasks, ev.get("support_tasks"))
        elif kind == "transition":
            to = str(ev.get("to") or "")
            rec.state = to
            if to == ATTEMPTED:
                rec.attempts += 1
            rec.history.append(
                {
                    "step": str(ev.get("step") or ""),
                    "to": to,
                    "reason": str(ev.get("reason") or ""),
                }
            )
    return by_id


def load_proposals(archive_root: Path | str) -> list[ProposalRecord]:
    return list(_fold(_read_events(archive_root)).values())


def get_proposal(archive_root: Path | str, proposal_id: str) -> ProposalRecord | None:
    return _fold(_read_events(archive_root)).get(proposal_id)


def proposal_of_finding(
    archive_root: Path | str, finding_id: str
) -> ProposalRecord | None:
    for rec in load_proposals(archive_root):
        if finding_id in rec.finding_ids:
            return rec
    return None


def open_proposals(
    archive_root: Path | str,
    *,
    lane: str | None = None,
    target_variant: str | None = None,
) -> list[ProposalRecord]:
    """The search view: what is still available to be chosen this window."""
    out = []
    for rec in load_proposals(archive_root):
        if rec.state != OPEN:
            continue
        if lane is not None and rec.lane != lane:
            continue
        if target_variant is not None and rec.target_variant != target_variant:
            continue
        out.append(rec)
    return out


def _next_id(events: list[dict[str, Any]]) -> str:
    """A running ordinal, minted from the highest ID the log has ever mentioned.

    Not a hash of the contract text: ``behavioral_delta`` / ``causal_hypothesis``
    are readable contract content, and a semantic hash over free text drifts —
    it would split one direction in two or fuse two into one. Sameness is decided
    by explicit pairwise comparison, never by an ID.

    Not a count of what parsed, either: if a ``create`` line is damaged, the
    transitions still naming that ID are live history, and handing the ID to a
    different direction would silently merge two proposals.
    """
    highest = 0
    for ev in events:
        pid = str(ev.get("proposal_id") or "")
        _, _, ordinal = pid.partition("p_")
        if ordinal.isdigit():
            highest = max(highest, int(ordinal))
    return f"p_{highest + 1:04d}"


def create_proposal(
    archive_root: Path | str,
    *,
    step: str,
    lane: str,
    action: str,
    target_variant: str = "",
    behavioral_delta: str = "",
    causal_hypothesis: str = "",
    finding_ids: list[str] | None = None,
    support_tasks: list[str] | None = None,
    parent_tree_hash: str = "",
) -> ProposalRecord:
    pid = _next_id(_read_events(archive_root))
    _append(
        archive_root,
        {
            "event": "create",
            "proposal_id": pid,
            "step": step,
            "lane": lane,
            "action": action,
            "target_variant": target_variant,
            "behavioral_delta": behavioral_delta,
            "causal_hypothesis": causal_hypothesis,
            "parent_tree_hash": parent_tree_hash,
            "finding_ids": list(finding_ids or []),
            "support_tasks": list(support_tasks or []),
        },
    )
    return get_proposal(archive_root, pid)  # type: ignore[return-value]


def link_finding(
    archive_root: Path | str,
    proposal_id: str,
    *,
    step: str,
    finding_id: str,
    support_tasks: list[str] | None = None,
    reason: str = "",
) -> ProposalRecord:
    """Attach one more finding's support to an open proposal.

    Idempotent by ``finding_id`` — a resumed window re-ingests the same findings
    and double-counted support would manufacture agreement nobody observed.
    """
    by_id = _fold(_read_events(archive_root))
    rec = by_id.get(proposal_id)
    if rec is None:
        raise KeyError(proposal_id)
    if rec.state not in (OPEN, SELECTED):
        raise IllegalTransition(
            f"{proposal_id} is {rec.state}; new same-direction evidence must be "
            "compared against the updated archive first, not appended here"
        )
    if finding_id in rec.finding_ids:
        return rec
    owner = next(
        (r for r in by_id.values() if finding_id in r.finding_ids and r is not rec),
        None,
    )
    if owner is not None:
        raise AlreadyLinked(f"{finding_id} already supports {owner.proposal_id}")
    _append(
        archive_root,
        {
            "event": "support",
            "proposal_id": proposal_id,
            "step": step,
            "finding_ids": [finding_id],
            "support_tasks": list(support_tasks or []),
            "reason": reason,
        },
    )
    return get_proposal(archive_root, proposal_id)  # type: ignore[return-value]


def transition(
    archive_root: Path | str,
    proposal_id: str,
    *,
    step: str,
    to: str,
    reason: str = "",
) -> ProposalRecord:
    by_id = _fold(_read_events(archive_root))
    rec = by_id.get(proposal_id)
    if rec is None:
        raise KeyError(proposal_id)
    if to not in STATES:
        raise IllegalTransition(f"unknown state {to!r}")
    if to not in _ALLOWED.get(rec.state, frozenset()):
        raise IllegalTransition(f"{rec.state} → {to} is not a legal move")
    if (
        to == TABOO
        and rec.state == REJECTED_IMPLEMENTATION
        and rec.attempts < TABOO_AFTER_ATTEMPTS
    ):
        raise IllegalTransition(
            f"{proposal_id} has {rec.attempts} attempt(s): a botched build is not "
            "a refutation of the direction — retry it or reject the direction itself"
        )
    _append(
        archive_root,
        {
            "event": "transition",
            "proposal_id": proposal_id,
            "step": step,
            "to": to,
            "reason": reason,
        },
    )
    return get_proposal(archive_root, proposal_id)  # type: ignore[return-value]
