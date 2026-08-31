"""H4 — the window lease: what this window took, so the next one can clean up.

:func:`dual_implement.implement_lanes` moves a proposal ``open → selected``
before it launches the implementers. Everything after that — building the
staging, running the editor, the whole gate battery, the ``selected → attempted``
transition — is tens of minutes against an endpoint with a history of wedging,
on a box with a history of being reaped. Die anywhere in there and the proposal
stays ``selected`` forever, because :func:`portfolio.select_portfolio` only ever
picks ``open``: the direction becomes silently unaskable for the rest of the
lineage, with all its accumulated support intact and useless.

The one recovery that existed covered a staging that failed to build — an
exception inside a process that is still running. A process that is gone raises
nothing, so nothing ran.

The journal is three events, append-only, next to the proposal log:

``selected``
    this proposal was taken by this window; here is the staging it was given
``attempt_started``
    an implementer was actually launched against it
``attempt_finished``
    the lane reached a verdict — there is nothing left to recover

Recovery hinges on whether the implementer was launched, because that decides
who pays. **Launched and lost** settles as ``attempted`` and then
``rejected_implementation``: the direction keeps its retry (that transition is
the retry lane), but the attempt is charged — otherwise a proposal could be
retried forever and never reach taboo, which is the accounting hole the two-lane
lifecycle already refuses elsewhere. **Never launched** goes straight back to
``open``, untouched: nothing was tried on its behalf, and charging it would
spend one of its two lives for a crash it had no part in.

Recovery only ever touches windows OTHER than the current one — the lanes of the
running window are in flight, and handing their proposals back to the portfolio
while they are being built is the failure this module exists to prevent, not
one to cause.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from harbor.agents.terminus_2_modular.self_evo import proposals as _proposals
from harbor.agents.terminus_2_modular.self_evo.backlog import backlog_dir

_logger = logging.getLogger(__name__)

LEASES_FILENAME = "leases.jsonl"

SELECTED = "selected"
ATTEMPT_STARTED = "attempt_started"
ATTEMPT_FINISHED = "attempt_finished"
RECOVERED = "recovered"

#: What recovery did about a stranded proposal.
REOPENED = "reopened"
SETTLED = "settled"


@dataclass
class Lease:
    proposal_id: str
    lane: str = ""
    step: str = ""
    staging_path: str = ""
    attempt_started: bool = False
    attempt_finished: bool = False
    recovered: bool = False


@dataclass
class Recovery:
    proposal_id: str
    action: str
    reason: str = ""


def leases_path(archive_root: Path | str) -> Path:
    return backlog_dir(archive_root) / LEASES_FILENAME


def _read(archive_root: Path | str) -> list[dict[str, Any]]:
    path = leases_path(archive_root)
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:
            _logger.warning("lease journal: skipping unreadable line")
            continue
        if isinstance(row, dict) and row.get("proposal_id"):
            rows.append(row)
    return rows


def _append(archive_root: Path | str, row: dict[str, Any]) -> None:
    try:
        path = leases_path(archive_root)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    except OSError as exc:  # bookkeeping must never cost a generation
        _logger.warning("could not write lease event: %s", exc)


def _fold(archive_root: Path | str) -> dict[str, Lease]:
    by_id: dict[str, Lease] = {}
    for row in _read(archive_root):
        pid = str(row["proposal_id"])
        event = str(row.get("event", ""))
        if event == SELECTED:
            # A fresh take supersedes any earlier one: the same proposal may be
            # retried in a later window, and the lease that matters is the last.
            by_id[pid] = Lease(
                proposal_id=pid,
                lane=str(row.get("lane", "")),
                step=str(row.get("step", "")),
                staging_path=str(row.get("staging_path", "")),
            )
            continue
        rec = by_id.get(pid)
        if rec is None:
            rec = Lease(proposal_id=pid, step=str(row.get("step", "")))
            by_id[pid] = rec
        if event == ATTEMPT_STARTED:
            rec.attempt_started = True
        elif event == ATTEMPT_FINISHED:
            rec.attempt_finished = True
        elif event == RECOVERED:
            rec.recovered = True
    return by_id


def record_selected(
    archive_root: Path | str,
    proposal_id: str,
    *,
    lane: str,
    step: str,
    staging_path: str,
) -> None:
    _append(
        archive_root,
        {
            "event": SELECTED,
            "proposal_id": proposal_id,
            "lane": lane,
            "step": step,
            "staging_path": staging_path,
        },
    )


def record_attempt_started(
    archive_root: Path | str, proposal_id: str, *, step: str
) -> None:
    _append(
        archive_root,
        {"event": ATTEMPT_STARTED, "proposal_id": proposal_id, "step": step},
    )


def record_attempt_finished(
    archive_root: Path | str, proposal_id: str, *, step: str, ok: bool
) -> None:
    _append(
        archive_root,
        {
            "event": ATTEMPT_FINISHED,
            "proposal_id": proposal_id,
            "step": step,
            "ok": bool(ok),
        },
    )


def open_leases(archive_root: Path | str) -> list[Lease]:
    """Takes that never reported a verdict."""
    return [
        rec
        for rec in _fold(archive_root).values()
        if not rec.attempt_finished and not rec.recovered
    ]


def recover(archive_root: Path | str, *, step: str) -> list[Recovery]:
    """Settle or release proposals stranded by a window that never came back.

    Never raises: this runs at the top of a window, and a tidy-up that can take
    down the run it was tidying up for is worse than the mess.
    """
    out: list[Recovery] = []
    for rec in open_leases(archive_root):
        if rec.step == step:
            continue  # this window's own lanes, still in flight
        try:
            proposal = _proposals.get_proposal(archive_root, rec.proposal_id)
        except Exception:  # noqa: BLE001
            proposal = None
        if proposal is None:
            _logger.warning(
                "lease names %s, which the backlog does not have", rec.proposal_id
            )
            _append(
                archive_root,
                {
                    "event": RECOVERED,
                    "proposal_id": rec.proposal_id,
                    "step": step,
                    "action": "unknown_proposal",
                },
            )
            continue
        if proposal.state != _proposals.SELECTED:
            # Something already carried it past `selected`; the lease is stale,
            # the proposal is not stranded.
            _append(
                archive_root,
                {
                    "event": RECOVERED,
                    "proposal_id": rec.proposal_id,
                    "step": step,
                    "action": "not_stranded",
                },
            )
            continue

        reason = f"window {rec.step} did not finish"
        try:
            if rec.attempt_started:
                _proposals.transition(
                    archive_root,
                    rec.proposal_id,
                    step=step,
                    to=_proposals.ATTEMPTED,
                    reason=reason,
                )
                _proposals.transition(
                    archive_root,
                    rec.proposal_id,
                    step=step,
                    to=_proposals.REJECTED_IMPLEMENTATION,
                    reason=reason,
                )
                action = SETTLED
            else:
                _proposals.transition(
                    archive_root,
                    rec.proposal_id,
                    step=step,
                    to=_proposals.OPEN,
                    reason=reason,
                )
                action = REOPENED
        except Exception as exc:  # noqa: BLE001
            _logger.warning("could not recover %s: %s", rec.proposal_id, exc)
            continue
        _append(
            archive_root,
            {
                "event": RECOVERED,
                "proposal_id": rec.proposal_id,
                "step": step,
                "action": action,
            },
        )
        out.append(Recovery(proposal_id=rec.proposal_id, action=action, reason=reason))
    return out
