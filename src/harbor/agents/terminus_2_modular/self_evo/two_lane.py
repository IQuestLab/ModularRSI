"""Implement and promote up to two independently selected proposals.

Each lane owns an isolated staging tree and gate result. If both candidates
pass, the second is rebased onto the first and only invalidated gates rerun.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

from harbor.agents.terminus_2_modular.self_evo import dual_implement as _dual
from harbor.agents.terminus_2_modular.self_evo import lease as _lease
from harbor.agents.terminus_2_modular.self_evo import portfolio as _portfolio
from harbor.agents.terminus_2_modular.self_evo import promotion as _promotion
from harbor.agents.terminus_2_modular.self_evo import proposals as _proposals
from harbor.agents.terminus_2_modular.self_evo import rebase as _rebase
from harbor.agents.terminus_2_modular.self_evo import staging as _staging

_logger = logging.getLogger(__name__)


#: ``(promoted_gen, parent_gen, lane_result)`` — told about each generation the
#: window creates, in the order it creates them. `parent_gen` is what that tree
#: was actually built on: the window's parent for the first lane, the
#: just-promoted generation for a rebased one. Injected rather than imported so
#: this module keeps knowing nothing about archive.json's schema.
OnPromote = Callable[[Path, Path, Any], None]


@dataclass
class LaneVerdict:
    proposal_id: str
    lane: str
    promoted_gen: Path | None = None
    reason: str = ""  # why it did not promote; "" when it did
    files_changed: list[str] = field(default_factory=list)
    gates_reused: tuple[str, ...] = ()
    gates_rerun: tuple[str, ...] = ()
    #: Whatever the lane's gate battery produced. It is opaque here on purpose —
    #: this module orders promotions, it does not interpret verdicts — but it
    #: has to travel, or the two generations get reported under one lane's
    #: review/sanity results. After a rebase this is the RE-gate's result: that
    #: is the judgement actually rendered on the tree being promoted.
    detail: Any = None


@dataclass
class LaneSettlement:
    """What one acted-on direction did to its support tasks.

    The router's exhaustion blacklist is fed by ``record_reflection``. The old
    blanket call over every *diagnosed* task was deleted because it charged
    tasks whose findings were never implemented — "proposed something and lost
    the selection" counted as "we tried and failed", and 13 of 120 tasks were
    blacklisted that way. Nothing replaced it, so nothing is charged at all and
    the exhaustion signal is dead.

    This is the replacement, and it is selection-aware by construction: only a
    proposal that reached an implementer is reported, and only its own support
    tasks. ``ran`` says whether the changed variant actually executed — a change
    that never ran has tested nothing, so it cannot be evidence that a task is
    unfixable, and the caller is expected to charge nothing for it.
    """

    proposal_id: str
    lane: str
    support_tasks: list[str]
    made_progress: bool
    ran: bool


@dataclass
class TwoLaneOutcome:
    verdicts: list[LaneVerdict] = field(default_factory=list)

    @property
    def promoted(self) -> list[LaneVerdict]:
        return [v for v in self.verdicts if v.promoted_gen is not None]


def _close(archive_root: Path, proposal_id: str, *, step: str, to: str, reason: str):
    """Move a proposal to its end state, never letting bookkeeping kill a run."""
    try:
        _proposals.transition(
            archive_root, proposal_id, step=step, to=to, reason=reason
        )
    except Exception as exc:  # IllegalTransition on an odd resume, etc.
        _logger.warning("could not close %s as %s: %s", proposal_id, to, exc)


def _reject_class(outcome: Any) -> tuple[str, str]:
    """What the reviewer actually said about a lane that did not promote.

    ``"proposal"`` — the DIRECTION is wrong. ``"implementation"`` — the
    direction is right and the code is not, plus a brief naming the flaw.
    ``"none"`` — nothing classified it: a crash, or a gate other than review
    (smoke and sanity carry no class, and smoke has its own repair loop).
    """
    detail = getattr(outcome.result, "detail", None)
    return (
        getattr(detail, "review_reject_class", "none") or "none",
        getattr(detail, "review_repair_brief", "") or "",
    )


def _lane_failed(outcome: Any) -> bool:
    return outcome.result is None or not outcome.result.passed


async def _second_implementer(
    archive_root: Path,
    outcomes: list[Any],
    *,
    parent_gen: Path,
    step: str,
    run_lane: Callable[[Any], Awaitable[Any]],
) -> list[Any]:
    """Rebuild the directions the reviewer called sound but badly built.

    The reviewer's `repair_brief` was written and never read: a lane rejected
    for flawed code was closed `rejected_implementation`, which the portfolio
    never picks, so a direction judged RIGHT died of one bad build. That is the
    starvation this whole phase exists to remove.

    The rewrite is triggered here rather than by re-entering the portfolio, so
    it cannot cost the other lane its slot. It runs once: a direction whose
    second build is also rejected has earned taboo, and that is the only way
    taboo is ever earned by an implementation.
    """
    retryable = []
    for outcome in outcomes:
        if not outcome.attempted or not _lane_failed(outcome) or outcome.retry:
            continue
        kind, brief = _reject_class(outcome)
        if kind != "implementation":
            continue
        try:
            proposal = _proposals.get_proposal(archive_root, outcome.proposal_id)
        except Exception:  # noqa: BLE001
            continue
        if proposal is None or proposal.attempts >= _proposals.TABOO_AFTER_ATTEMPTS:
            continue
        retryable.append((outcome, proposal, brief))
    if not retryable:
        return outcomes

    picks = []
    briefs: dict[str, str] = {}
    for outcome, proposal, brief in retryable:
        _close(
            archive_root,
            outcome.proposal_id,
            step=step,
            to=_proposals.REJECTED_IMPLEMENTATION,
            reason=f"[implementation] {_reason_for(outcome)}",
        )
        # The rejected tree goes; the second implementer starts from the parent
        # with the critique in hand. Keeping the bad tree alive across the retry
        # would make it the thing being judged instead of the direction.
        _staging.discard_staging(outcome.staging_dir)
        picks.append(_RetryPick(proposal))
        briefs[outcome.proposal_id] = brief

    retried = await _dual.implement_lanes(
        archive_root,
        picks,
        parent_dir=parent_gen,
        step=step,
        run_lane=run_lane,
        repair_briefs=briefs,
    )
    replaced = {o.proposal_id: o for o in retried}
    return [replaced.get(o.proposal_id, o) for o in outcomes]


class _RetryPick:
    """The shape `implement_lanes` reads out of a portfolio pick."""

    def __init__(self, proposal: Any):
        self.proposal = proposal
        self.proposal_id = proposal.proposal_id
        self.lane = proposal.lane
        self.reason = "second implementer: the reviewer called the direction sound"


def _reason_for(outcome: Any) -> str:
    return outcome.error or "failed its gates"


def _promote(
    archive_root: Path,
    tree: Path,
    *,
    parent_gen: Path,
    result: Any,
    on_promote: OnPromote | None,
) -> Path:
    """Move `tree` in as the next generation and tell the archive about it.

    `atomic_promote` only puts files on disk. Without the second half the
    archive keeps describing the generation the run started from, and since the
    router's candidate list is built from the archive, every later window is
    blind to the variants this run itself produced.

    The announcement runs after the move and may never undo it: the generation
    is already in the archive, and a bookkeeping error is not grounds to take
    it back out.
    """
    target = archive_root / f"gen_{_staging.next_gen_number(archive_root)}"
    promoted = _staging.atomic_promote(tree, target)
    if on_promote is not None:
        try:
            on_promote(promoted, parent_gen, result)
        except Exception as exc:
            _logger.warning(
                "archive sync after promoting %s failed (the generation stands): %s",
                promoted.name,
                exc,
            )
    return promoted


async def run_window(
    archive_root: Path | str,
    *,
    parent_gen: Path,
    step: str,
    run_lane: Callable[[Any], Awaitable[Any]],
    rerun_gates: Callable[..., Awaitable[Any]],
    max_lanes: int = 2,
    seed: str = "",
    novelty_first: bool = False,
    on_promote: OnPromote | None = None,
    on_lane_settled: Callable[[LaneSettlement], None] | None = None,
) -> TwoLaneOutcome:
    """Run one selection → implementation → promotion window.

    ``run_lane(brief)`` builds one lane and reports whether it cleared its own
    gates. ``rerun_gates(dest=, gates=, lane=, new_parent=, prior=, proposal_id=)``
    re-judges
    a rebased tree on just the gates the rebase invalidated — it is given the
    new parent (the rebased tree's diff base is the generation that was just
    promoted, not the one both lanes started from) and the lane's original
    result (so it reuses the change-local verdict instead of re-running the
    editor). Both are injected because they are the two places this loop talks
    to an LLM and to Docker.
    """
    archive_root = Path(archive_root)
    # Before selecting: settle anything a previous window took and never
    # reported on. It has to happen HERE — the portfolio only ever picks `open`,
    # so a proposal still marked `selected` by a dead process is invisible to
    # the very call on the next line, and recovering afterwards would be the
    # same as not recovering at all.
    for recovered in _lease.recover(archive_root, step=step):
        _logger.info(
            "recovered %s from an unfinished window: %s",
            recovered.proposal_id,
            recovered.action,
        )
    picks = _portfolio.select_portfolio(archive_root, max_lanes=max_lanes, seed=seed)
    if not picks:
        return TwoLaneOutcome()

    outcomes = await _dual.implement_lanes(
        archive_root,
        picks,
        parent_dir=parent_gen,
        step=step,
        run_lane=run_lane,
    )
    # A direction the reviewer called sound but badly built gets its one
    # rewrite here, before promotions are planned — the second build is a
    # candidate for this window's generation like any other.
    outcomes = await _second_implementer(
        archive_root,
        outcomes,
        parent_gen=Path(parent_gen),
        step=step,
        run_lane=run_lane,
    )

    by_id = {o.proposal_id: o for o in outcomes}
    plan = _promotion.plan_promotions(
        _dual.to_candidates(outcomes), novelty_first=novelty_first
    )
    planned = {s.proposal_id for s in plan}

    verdicts: list[LaneVerdict] = []

    # Lanes that never made it into the plan: they failed their own gates or
    # crashed. `attempted` decides whether the direction is answerable for it —
    # a lane whose staging never got built was already put back to `open`.
    for outcome in outcomes:
        if outcome.proposal_id in planned:
            continue
        reason = _reason_for(outcome)
        kind, _ = _reject_class(outcome)
        if outcome.attempted:
            # A reviewer that judged the DIRECTION wrong must not be recorded as
            # a botched build: `rejected_implementation` means "sound direction,
            # bad code" and carries a retry this direction has not earned.
            closed_as = (
                _proposals.REJECTED_PROPOSAL
                if kind == "proposal"
                else _proposals.REJECTED_IMPLEMENTATION
            )
            _close(
                archive_root,
                outcome.proposal_id,
                step=step,
                to=closed_as,
                reason=f"[{kind}] {reason}" if kind != "none" else reason,
            )
            if outcome.retry and closed_as == _proposals.REJECTED_IMPLEMENTATION:
                # Two builds of a direction the reviewer kept calling sound.
                # This is the only way an implementation earns taboo.
                _close(
                    archive_root,
                    outcome.proposal_id,
                    step=step,
                    to=_proposals.TABOO,
                    reason="the second implementation was rejected too",
                )
        _staging.discard_staging(outcome.staging_dir)
        verdicts.append(
            LaneVerdict(
                proposal_id=outcome.proposal_id, lane=outcome.lane, reason=reason
            )
        )

    newest: Path | None = None
    for pstep in plan:
        outcome = by_id[pstep.proposal_id]
        verdict = LaneVerdict(proposal_id=pstep.proposal_id, lane=pstep.lane)

        if not pstep.rebase_required:
            newest = _promote(
                archive_root,
                outcome.staging_dir,
                parent_gen=Path(parent_gen),
                result=outcome.result,
                on_promote=on_promote,
            )
            verdict.promoted_gen = newest
            verdict.detail = getattr(outcome.result, "detail", None)
            _close(
                archive_root,
                pstep.proposal_id,
                step=step,
                to=_proposals.PROMOTED,
                reason=newest.name,
            )
            verdicts.append(verdict)
            continue

        # Everything from here on may only ever drop THIS lane: `newest` is
        # already in the archive.
        moved = _rebase.rebase_onto(
            staging=outcome.staging_dir,
            old_parent=Path(parent_gen),
            new_parent=newest,  # type: ignore[arg-type]
            dest=archive_root / _dual.STAGING_DIRNAME / f"rebased_{pstep.proposal_id}",
        )
        verdict.files_changed = list(moved.files_moved)

        if moved.conflicts:
            verdict.reason = (
                "rebase conflict — the promoted lane changed the same file(s): "
                + ", ".join(moved.conflicts)
            )
            _close(
                archive_root,
                pstep.proposal_id,
                step=step,
                to=_proposals.REJECTED_IMPLEMENTATION,
                reason=verdict.reason,
            )
        elif moved.absorbed:
            verdict.reason = (
                f"absorbed: {newest.name} already contains this exact change"  # type: ignore[union-attr]
            )
            _close(
                archive_root,
                pstep.proposal_id,
                step=step,
                to=_proposals.SUPERSEDED,
                reason=verdict.reason,
            )
        else:
            reused, rerun = _promotion.gates_after_rebase(
                moved.diff_hash_before, moved.diff_hash_after
            )
            verdict.gates_reused, verdict.gates_rerun = reused, rerun
            result = await rerun_gates(
                dest=moved.dest,
                gates=rerun,
                lane=pstep.lane,
                # which proposal this tree implements: the battery holds a
                # change to the action it was routed for, and that lives on the
                # proposal. Without it the rebased lane is the one candidate
                # nobody checks.
                proposal_id=pstep.proposal_id,
                # the rebased tree sits on `newest`, so that — not the original
                # parent — is what its change must now be measured against
                new_parent=newest,
                # what this lane already produced, so re-gating never re-runs
                # the editor: the reused verdict and the trajectory behind it
                # both belong to that first pass
                prior=outcome.result,
            )
            verdict.detail = getattr(result, "detail", None)
            if getattr(result, "passed", False):
                newest = _promote(
                    archive_root,
                    moved.dest,
                    # this tree was rebased onto `newest`, so that generation —
                    # not the window's parent — is what its change is measured
                    # against; crediting gen_0 here would file the first lane's
                    # files as this lane's additions
                    parent_gen=newest,  # type: ignore[arg-type]
                    result=result,
                    on_promote=on_promote,
                )
                verdict.promoted_gen = newest
                _close(
                    archive_root,
                    pstep.proposal_id,
                    step=step,
                    to=_proposals.PROMOTED,
                    reason=newest.name,
                )
            else:
                verdict.reason = f"failed re-run gates after rebase: {rerun}"
                _close(
                    archive_root,
                    pstep.proposal_id,
                    step=step,
                    to=_proposals.REJECTED_IMPLEMENTATION,
                    reason=verdict.reason,
                )

        _staging.discard_staging(moved.dest)
        _staging.discard_staging(outcome.staging_dir)
        verdicts.append(verdict)

    _settle_accounting(archive_root, outcomes, verdicts, on_lane_settled)
    return TwoLaneOutcome(verdicts=verdicts)


def _settle_accounting(
    archive_root: Path,
    outcomes: list[Any],
    verdicts: list[LaneVerdict],
    on_lane_settled: Callable[[LaneSettlement], None] | None,
) -> None:
    """Report each acted-on direction to the router's task accounting.

    Only lanes that reached an implementer are reported: a proposal nobody
    selected was never tried, and charging it is precisely the blacklist bug
    this replaces. Never raises — bookkeeping does not get to undo a generation
    that is already in the archive.
    """
    if on_lane_settled is None:
        return
    promoted = {v.proposal_id for v in verdicts if v.promoted_gen is not None}
    for outcome in outcomes:
        if not outcome.attempted:
            continue
        detail = getattr(outcome.result, "detail", None)
        try:
            proposal = _proposals.get_proposal(archive_root, outcome.proposal_id)
            on_lane_settled(
                LaneSettlement(
                    proposal_id=outcome.proposal_id,
                    lane=outcome.lane,
                    support_tasks=list(getattr(proposal, "support_tasks", []) or []),
                    made_progress=outcome.proposal_id in promoted,
                    ran=bool(getattr(detail, "sanity_activation", None)),
                )
            )
        except Exception as exc:  # noqa: BLE001
            _logger.warning(
                "task accounting for %s failed: %s", outcome.proposal_id, exc
            )
