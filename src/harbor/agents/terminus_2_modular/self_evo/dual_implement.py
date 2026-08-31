"""P2 — two proposals, two isolated implementations, mutually blind.

The portfolio returns at most one proposal per lane. This module builds both of
them off the *same* parent tree, in two separate stagings, by two implementers
that cannot see each other.

**Separate trees.** One shared staging would let whichever implementer wrote
last silently decide what the other one built on, and both sets of gate results
would then describe a tree neither candidate actually proposed. The two staging
roots are siblings and never nested, which matters concretely: the editor's
``read_file`` is scoped to its own staging root, so non-containment is what
makes the blindness structural rather than a promise. (``archive_path`` is not a
read root — it only resolves ``archive.json`` for the niche view — and
``trajectory_root`` is the solver output, which holds no proposal text.)

**Mutual blindness.** If implementer A can read B's proposal, the two stop being
independent samples of "what should change": A talks itself into B's framing and
the second lane stops being a second opinion. Novelty gets its own slot
precisely so it does not have to argue against the incumbent — showing it the
incumbent's proposal puts that argument straight back.

**Failure isolation.** A crash in one lane must not block, discard, or
contaminate the other. Two lanes exist to raise the odds that *something* lands;
coupling their failures gives that away for free.

**All of the evidence, always.** The implementer is handed *every* finding
supporting its proposal — not a sample, not the strongest one. Handing over a
single finding is how a general direction degrades into "make this one task
pass", which is the documented cause of death of the ``f34f2c8`` lineage. When a
supporting finding cannot be read back from the ledger, the brief says so out
loud: believing you saw the whole picture when you saw two thirds of it is worse
than knowing a third is missing.

**Accounting.** A lane that reached the implementer is ``attempted`` even if the
implementer crashed — the intervention did happen, it just went badly, and
hiding that would let one proposal be retried forever without ever reaching
taboo. A lane whose staging could not even be built goes back to ``open``: it
was never acted on, so charging it an attempt would spend one of its two lives
for nothing.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

from harbor.agents.terminus_2_modular.self_evo import backlog as _backlog
from harbor.agents.terminus_2_modular.self_evo import lease as _lease
from harbor.agents.terminus_2_modular.self_evo import promotion as _promotion
from harbor.agents.terminus_2_modular.self_evo import proposals as _proposals
from harbor.agents.terminus_2_modular.self_evo import routing as _routing
from harbor.agents.terminus_2_modular.self_evo import staging as _staging

STAGING_DIRNAME = "staging"

#: How much of one finding's verbatim text goes into the brief. The finding is
#: the only record of what was actually observed, so this is generous on
#: purpose; the cap exists for the pathological case, not the normal one.
MAX_FINDING_CHARS = 6000


@dataclass
class LaneBrief:
    """Everything one implementer is given — and nothing from the other lane."""

    proposal_id: str
    lane: str
    staging_dir: Path
    instruction: str
    findings: list[Any] = field(default_factory=list)
    missing_finding_ids: list[str] = field(default_factory=list)
    proposal: Any = None


@dataclass
class LaneResult:
    """What a lane's runner reports back: did the change clear its own gates."""

    passed: bool
    diff_hash: str = ""
    detail: Any = None


@dataclass
class LaneOutcome:
    proposal_id: str
    lane: str
    staging_dir: Path
    ok: bool  # the lane ran to completion without raising
    error: str | None = None
    result: LaneResult | None = None
    attempted: bool = False  # did the implementer actually get to run
    #: this lane is the second implementer on a direction whose first build the
    #: reviewer rejected as flawed code. There is no third.
    retry: bool = False


def staging_dir_for(archive_root: Path | str, lane: str, proposal_id: str) -> Path:
    """``<archive>/staging/<lane>_<proposal_id>``.

    The name carries both lane and proposal so a run that dies mid-window leaves
    dirs that can be identified without consulting a log.
    """
    return Path(archive_root) / STAGING_DIRNAME / f"{lane}_{proposal_id}"


_PREAMBLE = """\
You are implementing ONE agreed direction on the agent's modules.

The direction below was chosen from a backlog of directions, and the evidence
under it is the COMPLETE set of observations that support it — every one of
them, not a sample. Build the change that addresses the direction across all of
that evidence. Do NOT narrow it to whichever single task reads most vividly: a
change that only rescues one named task is worthless here and has sunk a
lineage before.

# The direction — proposal {pid} (lane={lane}, action={action})
{target}
What should change about the agent's behaviour:
{delta}

Why this is believed to be the cause:
{why}

# The evidence — {n} finding(s), all of them
{findings}
{missing}
# How to work
1. Read the module(s) involved before editing.
2. Make the change general: it has to hold for every finding above, not one.
3. {shape}
4. `<validate/>`, then `<commit_patch/>`, then
   `<task_complete>true</task_complete>`.
"""

_SHAPE_MODIFY = (
    "This change is a `modify` of an existing variant: edit "
    "`modules/{path}.py` itself. Creating a sibling variant would be an `add`, "
    "not this change."
)
_SHAPE_REPLACE = (
    "This change is a `replace`: create a NEW variant and retire the target by "
    "putting\n       SUPERSEDES: {target}\n   in its `<variant_meta>` block."
)
_SHAPE_ADD = (
    "This change is an `add`: create a NEW variant for behaviour no active "
    "variant covers. Leave existing variants unchanged."
)

_REPAIR_TEMPLATE = """
# ⚠️ This direction has been built once already, and the build was rejected
The reviewer judged the DIRECTION sound and the CODE wrong. Its brief:

{brief}

You are the second implementer. Start from the parent tree — the rejected tree
is gone — and build the same direction without repeating that flaw. This is the
LAST attempt: another rejected build retires the direction.
"""

_MISSING_TEMPLATE = """
# ⚠️ Incomplete evidence
{n} supporting finding(s) could not be read back from the ledger: {ids}.
What you see above is therefore PART of the support for this direction, not all
of it. Prefer the more general reading of the evidence you do have.
"""


def _render_finding(record: Any) -> str:
    where = f"window `{record.step}`"
    if record.task:
        where = f"task `{record.task}`, " + where
    body = json.dumps(record.raw_finding, indent=2, ensure_ascii=False)
    if len(body) > MAX_FINDING_CHARS:
        body = body[:MAX_FINDING_CHARS] + "\n… [truncated]"
    return f"## finding `{record.finding_id}` — {where}\n\n```json\n{body}\n```"


def _shape_for(action: str, target_variant: str) -> str:
    """Describe the concrete output required by a routed action."""
    if action == _routing.ACTION_REPLACE and target_variant:
        return _SHAPE_REPLACE.format(target=target_variant)
    if action == _routing.ACTION_MODIFY and target_variant:
        return _SHAPE_MODIFY.format(path=target_variant)
    return _SHAPE_ADD


def build_brief(
    archive_root: Path | str,
    proposal: Any,
    staging_dir: Path,
    *,
    findings_by_id: dict[str, Any] | None = None,
    repair_brief: str = "",
) -> LaneBrief:
    """Assemble one lane's instruction from ONLY its own proposal + evidence."""
    if findings_by_id is None:
        findings_by_id = {
            rec.finding_id: rec for rec in _backlog.load_findings(archive_root)
        }

    found = [findings_by_id[f] for f in proposal.finding_ids if f in findings_by_id]
    missing = [f for f in proposal.finding_ids if f not in findings_by_id]

    target = (
        f"Variant to change: `{proposal.target_variant}`\n"
        if proposal.target_variant
        else "This is a NEW variant — no existing one owns this.\n"
    )
    shape = _shape_for(proposal.action or "", proposal.target_variant or "")
    missing_block = (
        _MISSING_TEMPLATE.format(
            n=len(missing), ids=", ".join(f"`{m}`" for m in missing)
        )
        if missing
        else ""
    )
    instruction = _PREAMBLE.format(
        pid=proposal.proposal_id,
        lane=proposal.lane,
        action=proposal.action or "?",
        target=target,
        delta=proposal.behavioral_delta or "(not stated)",
        why=proposal.causal_hypothesis or "(not stated)",
        shape=shape,
        n=len(found),
        findings="\n\n".join(_render_finding(r) for r in found) or "(none)",
        missing=missing_block,
    )
    if repair_brief:
        instruction += _REPAIR_TEMPLATE.format(brief=repair_brief)
    return LaneBrief(
        proposal_id=proposal.proposal_id,
        lane=proposal.lane,
        staging_dir=staging_dir,
        instruction=instruction,
        findings=found,
        missing_finding_ids=missing,
        proposal=proposal,
    )


async def _run_one_lane(
    archive_root: Path | str,
    pick: Any,
    *,
    parent_dir: Path,
    step: str,
    run_lane: Callable[[LaneBrief], Awaitable[LaneResult]],
    prepare: Callable[[Path, Path], Path],
    findings_by_id: dict[str, Any],
    repair_brief: str = "",
) -> LaneOutcome:
    proposal = pick.proposal
    lane, pid = pick.lane, pick.proposal_id
    staging_dir = staging_dir_for(archive_root, lane, pid)

    try:
        prepared = prepare(Path(parent_dir), staging_dir)
    except Exception as exc:
        # Nothing was ever tried on this proposal's behalf — put it back on the
        # shelf with its support intact rather than charging it an attempt.
        _proposals.transition(
            archive_root, pid, step=step, to=_proposals.OPEN, reason="staging failed"
        )
        # The lane is closed out here, in-process. Leaving its lease open would
        # invite the next window to "recover" a proposal that is already back on
        # the shelf.
        _lease.record_attempt_finished(archive_root, pid, step=step, ok=False)
        return LaneOutcome(
            proposal_id=pid,
            lane=lane,
            staging_dir=staging_dir,
            ok=False,
            error=f"{type(exc).__name__}: {exc}",
            attempted=False,
            retry=bool(repair_brief),
        )

    brief = build_brief(
        archive_root,
        proposal,
        Path(prepared),
        findings_by_id=findings_by_id,
        repair_brief=repair_brief,
    )

    result: LaneResult | None = None
    error: str | None = None
    # Recorded BEFORE the await: what distinguishes "the editor was running when
    # we died" from "we never got that far" is who pays for the crash, and a
    # marker written afterwards cannot tell those apart.
    _lease.record_attempt_started(archive_root, pid, step=step)
    try:
        result = await run_lane(brief)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    # The implementer ran, so this counts as an attempt either way.
    _proposals.transition(
        archive_root,
        pid,
        step=step,
        to=_proposals.ATTEMPTED,
        reason=error or "implemented",
    )
    _lease.record_attempt_finished(archive_root, pid, step=step, ok=error is None)
    return LaneOutcome(
        proposal_id=pid,
        lane=lane,
        staging_dir=brief.staging_dir,
        ok=error is None,
        error=error,
        result=result,
        attempted=True,
        retry=bool(repair_brief),
    )


async def implement_lanes(
    archive_root: Path | str,
    picks: list[Any],
    *,
    parent_dir: Path,
    step: str,
    run_lane: Callable[[LaneBrief], Awaitable[LaneResult]],
    prepare: Callable[[Path, Path], Path] | None = None,
    repair_briefs: dict[str, str] | None = None,
) -> list[LaneOutcome]:
    """Build every pick in its own staging, concurrently, without cross-talk.

    ``run_lane`` receives one :class:`LaneBrief` and does whatever a lane means
    to the caller — run the implementer, then its gates — returning a
    :class:`LaneResult`. Anything it raises is recorded against that lane only.
    """
    if not picks:
        return []

    def prepare_modules(parent: Path, dest: Path) -> Path:
        _staging.prepare_staging(parent / "modules", dest, fresh=True)
        return dest

    prep = prepare or prepare_modules
    findings_by_id = {
        rec.finding_id: rec for rec in _backlog.load_findings(archive_root)
    }

    for pick in picks:
        _proposals.transition(
            archive_root,
            pick.proposal_id,
            step=step,
            to=_proposals.SELECTED,
            reason=getattr(pick, "reason", ""),
        )
        # The journal, not the state, is what survives a process that is gone:
        # everything below raises nothing when the box is reaped, so `selected`
        # would be the proposal's last word forever.
        _lease.record_selected(
            archive_root,
            pick.proposal_id,
            lane=pick.lane,
            step=step,
            staging_path=str(
                staging_dir_for(archive_root, pick.lane, pick.proposal_id)
            ),
        )

    # gather, not a loop: serialised lanes would double every window's wall
    # clock, and return_exceptions is unnecessary because each lane already
    # catches its own — anything escaping here is a bug in this module.
    return list(
        await asyncio.gather(
            *(
                _run_one_lane(
                    archive_root,
                    pick,
                    parent_dir=Path(parent_dir),
                    step=step,
                    run_lane=run_lane,
                    prepare=prep,
                    findings_by_id=findings_by_id,
                    repair_brief=(repair_briefs or {}).get(pick.proposal_id, ""),
                )
                for pick in picks
            )
        )
    )


def to_candidates(outcomes: list[LaneOutcome]) -> list[_promotion.Candidate]:
    """Hand the lanes that produced a verdict to :mod:`promotion`.

    A lane that crashed has no verdict to give and is dropped. A lane that ran
    and FAILED its gates is carried through as failing — ``plan_promotions`` is
    where "one lane failing does not block the other" is decided, and it cannot
    decide about a candidate it was never shown.
    """
    return [
        _promotion.Candidate(
            proposal_id=o.proposal_id,
            lane=o.lane,
            passed_gates=bool(o.result.passed),
            diff_hash=o.result.diff_hash,
        )
        for o in outcomes
        if o.ok and o.result is not None
    ]
