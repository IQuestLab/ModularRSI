"""Route contrastive findings and accumulate proposal evidence.

This stage validates each finding against the active archive, clusters it into
an existing or new proposal, and records every comparison. Proposal selection
happens afterwards in the portfolio.
"""

from __future__ import annotations

import asyncio
import logging
import os
import weakref
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable

from harbor.agents.terminus_2_modular.self_evo import backlog as _backlog
from harbor.agents.terminus_2_modular.self_evo import clustering as _clustering
from harbor.agents.terminus_2_modular.self_evo import dispositions as _dispositions
from harbor.agents.terminus_2_modular.self_evo import routing as _routing

_logger = logging.getLogger(__name__)

_DEFAULT_CONCURRENCY = 2
_slots: "weakref.WeakKeyDictionary[Any, asyncio.Semaphore]" = (
    weakref.WeakKeyDictionary()
)


def _concurrency() -> int:
    try:
        return max(1, int(os.environ.get("HARBOR_EVIDENCE_CONCURRENCY", "")))
    except ValueError:
        return _DEFAULT_CONCURRENCY


@asynccontextmanager
async def _slot():
    loop = asyncio.get_running_loop()
    sem = _slots.get(loop)
    if sem is None:
        sem = asyncio.Semaphore(_concurrency())
        _slots[loop] = sem
    async with sem:
        yield


#: One lock per candidate pool, so the compare-and-create step is serialised
#: against the findings it could collide with and nothing else. Routing stays
#: fully parallel — it is the expensive part and it collides with nothing.
_cluster_locks: "weakref.WeakKeyDictionary[Any, dict[tuple, asyncio.Lock]]" = (
    weakref.WeakKeyDictionary()
)


def _pool_key(route) -> tuple:
    """Which findings could race this one into creating the same proposal.

    Mirrors `clustering._candidates` exactly: novelty compares against every
    open novelty proposal, incumbent only against those on its own variant. Two
    findings whose pools are disjoint cannot produce a duplicate of each other,
    so they must not wait for each other either.
    """
    if route.lane == _routing.LANE_NOVELTY:
        return (_routing.LANE_NOVELTY,)
    return (_routing.LANE_INCUMBENT, route.target_variant)


def _cluster_lock(key: tuple) -> asyncio.Lock:
    loop = asyncio.get_running_loop()
    table = _cluster_locks.get(loop)
    if table is None:
        table = {}
        _cluster_locks[loop] = table
    lock = table.get(key)
    if lock is None:
        lock = asyncio.Lock()
        table[key] = lock
    return lock


@dataclass
class EvidencePassReport:
    findings: int = 0
    routed: int = 0
    created: int = 0
    linked: int = 0
    invalid: int = 0
    truncated: int = 0
    errors: int = 0
    out_of_scope: int = 0
    covered_by_successor: int = 0
    stale_evidence: int = 0
    #: the judge could not say whether an open proposal was the same. The
    #: finding stays unattached and is re-routed next window, so this is a
    #: retry counter, not an attrition one.
    comparison_unresolved: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "findings": self.findings,
            "routed": self.routed,
            "created": self.created,
            "linked": self.linked,
            "invalid": self.invalid,
            "truncated": self.truncated,
            "errors": self.errors,
            "out_of_scope": self.out_of_scope,
            "covered_by_successor": self.covered_by_successor,
            "stale_evidence": self.stale_evidence,
            "comparison_unresolved": self.comparison_unresolved,
        }


_SAME_PROMPT = """Two proposed changes to the same module. Do they describe the \
SAME intervention — would implementing one make the other redundant?

## Candidate finding
Would change: {delta}
Because: {hypothesis}

## Existing open proposal
Would change: {p_delta}
Because: {p_hypothesis}

Pointing at the same file is NOT enough: "fix the validation classifier" and
"add missing-tool recovery" both touch the same variant and are different work.

Answer on one line, starting with YES or NO, then the reason."""

_COVERS_PROMPT = """The variant this finding accused has been retired. Its \
successor is `{successor}`.

## The observed gap
{gap}

## The successor's code
{code}

Did the successor inherit this gap, or does it already close it? Answer on one
line, starting with YES (already covered) or NO (the gap survived), then why."""


def _yes(text: str) -> bool:
    return (text or "").strip().upper().startswith("YES")


class _LLMHooks:
    """Clustering's two judgements, backed by the endpoint."""

    def __init__(self, ask: Callable[[str], Awaitable[str]], variant_source):
        self._ask = ask
        self._source = variant_source
        self.errors = 0

    async def same_intervention(self, finding, routing, proposal):
        prompt = _SAME_PROMPT.format(
            delta=routing.behavioral_delta,
            hypothesis=routing.causal_hypothesis,
            p_delta=proposal.behavioral_delta,
            p_hypothesis=proposal.causal_hypothesis,
        )
        reply = await self._ask(prompt)
        return _yes(reply), (reply or "").strip()[:400]

    async def successor_covers(self, finding, successor_qual):
        prompt = _COVERS_PROMPT.format(
            successor=successor_qual,
            gap=str(finding.raw_finding)[:3000],
            code=(self._source(successor_qual) if self._source else "")[:6000],
        )
        reply = await self._ask(prompt)
        return _yes(reply), (reply or "").strip()[:400]


class _SyncHooks:
    """Adapter: clustering is sync, the judgements are awaits.

    Each hook is resolved eagerly before clustering runs, which is only correct
    because clustering asks at most one coverage question and compares against a
    known, finite candidate list.
    """

    def __init__(self, same_answers: dict[str, tuple[bool | None, str]], covers):
        self._same = same_answers
        self._covers = covers

    def same_intervention(self, finding, routing, proposal):
        # `None` = never asked. The clusterer refuses to create against an
        # unanswered candidate; answering `False` here would let a proposal
        # nobody compared against pass for one that was ruled out.
        return self._same.get(proposal.proposal_id, (None, "not compared"))

    def successor_covers(self, finding, successor_qual):
        return self._covers


async def _route_one(
    archive_root: Path,
    finding,
    *,
    ask,
    active_variants,
    locked_type,
    step,
    archive_entries,
    variant_source,
) -> str:
    """Route + cluster a single finding. Returns the decision string."""
    prompt = _routing.build_routing_instruction(
        finding=finding.raw_finding,
        locked_type=locked_type,
        active_variants=active_variants,
    )
    reply = await ask(prompt)
    try:
        decision = _routing.validate_routing(
            _routing.parse_routing(reply),
            active_quals=[q for q, _ in active_variants],
            locked_type=locked_type,
        )
    except _routing.InvalidRouting as exc:
        # A cut-off reply is a token ceiling, not a model that could not decide.
        # Filing it as a refusal would put an endpoint artifact into a statistic
        # about routing quality.
        cut_off = _routing.looks_truncated(reply)
        decision = "routing_truncated" if cut_off else "routing_invalid"
        # A malformed reply is a verdict about this finding: the router looked
        # and could not place it. A CUT-OFF reply is a token ceiling — retiring
        # the finding for that would file an infra artifact as evidence.
        _dispositions.record(
            archive_root,
            finding_id=finding.finding_id,
            kind=(_dispositions.RETRYABLE if cut_off else _dispositions.INVALID),
            step=step,
            reason=("reply was cut off mid-block" if cut_off else str(exc)),
        )
        _clustering._audit(
            archive_root,
            {
                "step": step,
                "finding_id": finding.finding_id,
                "task": finding.task,
                "decision": decision,
                "reason": "reply was cut off mid-block" if cut_off else str(exc),
                "reply": (reply or "")[:2000],
            },
        )
        return decision

    route = _clustering.Routing(
        lane=decision.lane,
        action=decision.action,
        target_variant=decision.target_variant,
        behavioral_delta=decision.behavioral_delta,
        causal_hypothesis=decision.causal_hypothesis,
        rationale=decision.rationale,
        dismissals=dict(decision.dismissals),
        other_module=decision.other_module,
    )

    llm_hooks = _LLMHooks(ask, variant_source)

    # Resolve the semantic questions first so the (sync) clustering sequencer
    # stays a pure function of already-known answers.
    covers = (False, "successor not consulted")
    if route.lane == _routing.LANE_INCUMBENT and route.target_variant:
        live, how = _clustering.resolve_live_variant(
            archive_entries or [], route.target_variant
        )
        if how == "mapped" and live:
            covers = await llm_hooks.successor_covers(finding, live)

    same_answers: dict[str, tuple[bool | None, str]] = {}

    async def _compare_new(candidates) -> None:
        """Answer for every candidate not answered for yet.

        Asks them all at once. Clustering still links to the FIRST one that says
        "same", so the outcome is identical to asking in turn — but the latency
        is one round trip instead of N. Serialising these was measured at
        12.7 min for a step with 3 findings; the cross-finding semaphore could
        not help, because the wait was inside a single finding.
        """
        todo = [c for c in candidates if c.proposal_id not in same_answers]
        if not todo:
            return
        verdicts = await asyncio.gather(
            *(llm_hooks.same_intervention(finding, route, c) for c in todo),
            return_exceptions=True,
        )
        for candidate, verdict in zip(todo, verdicts):
            if isinstance(verdict, BaseException):
                _logger.warning(
                    "comparison against %s failed: %s: %s",
                    candidate.proposal_id,
                    type(verdict).__name__,
                    verdict,
                )
                # `None`, not `False`: the endpoint failing to answer is not the
                # endpoint saying "different". Filing it as "different" is what
                # turns one bad round trip into a permanent duplicate proposal.
                same_answers[candidate.proposal_id] = (None, "comparison failed")
                continue
            same_answers[candidate.proposal_id] = verdict

    if not covers[0]:
        await _compare_new(_clustering._candidates(archive_root, route))

    # Compare-and-create is serialised per candidate pool. Everything above ran
    # concurrently against a SNAPSHOT of the open proposals; a sibling finding
    # may have created one since. The clusterer re-reads the backlog and so does
    # see it — but with no answer for it, an unanswered candidate used to be
    # read as a confident "different" and the twin got created anyway.
    async with _cluster_lock(_pool_key(route)):
        if not covers[0]:
            await _compare_new(_clustering._candidates(archive_root, route))
        outcome = _clustering.cluster_finding(
            archive_root,
            finding=finding,
            routing=route,
            hooks=_SyncHooks(same_answers, covers),
            step=step,
            archive_entries=archive_entries,
        )
    return outcome.decision


async def collect_evidence(
    archive_root: Path | str,
    *,
    snapshot: dict,
    diagnose_items: list | None,
    active_variants: list[tuple[str, str]],
    locked_type: str,
    step: str,
    ask: Callable[[str], Awaitable[str]],
    archive_entries: list[Any] | None = None,
    variant_source: Callable[[str], str] | None = None,
) -> EvidencePassReport:
    """Route and cluster this step's findings without selecting a proposal."""
    report = EvidencePassReport()

    provenance: dict[str, dict] = {}
    for item in diagnose_items or []:
        provenance[item.task] = {
            "bucket": getattr(item, "bucket", ""),
            "source": getattr(item, "source", ""),
            "fail_trial": getattr(
                getattr(item, "contrast_fail", None), "trial_dir", None
            ),
            "pass_trial": getattr(
                getattr(item, "contrast_pass", None), "trial_dir", None
            ),
        }
    try:
        records = _backlog.ingest_step(archive_root, snapshot, provenance=provenance)
    except Exception as exc:
        _logger.warning("evidence pass could not read the findings: %s", exc)
        return report

    report.findings = len(records)
    # Resume: a finding already attached to a proposal is done. Checking here —
    # rather than inside the clusterer — is what makes it free; otherwise the
    # routing call is paid for first and only then thrown away.
    from harbor.agents.terminus_2_modular.self_evo import proposals as _proposals

    placed = {
        fid
        for rec in _proposals.load_proposals(archive_root)
        for fid in rec.finding_ids
    }
    # …and so is a finding whose verdict is already in. Attachment was the only
    # thing checked, so out_of_scope / covered / stale / invalid findings were
    # re-routed every window for the rest of the run, each time paying a routing
    # call to re-reach a decision that was already made. `retryable` is
    # deliberately absent: an endpoint that could not answer is not a verdict.
    placed |= _dispositions.terminal_finding_ids(archive_root)
    culprits = [r for r in records if r.is_culprit and r.finding_id not in placed]

    async def _one(finding):
        async with _slot():
            return await _route_one(
                Path(archive_root),
                finding,
                ask=ask,
                active_variants=active_variants,
                locked_type=locked_type,
                step=step,
                archive_entries=archive_entries,
                variant_source=variant_source,
            )

    results = await asyncio.gather(*(_one(f) for f in culprits), return_exceptions=True)
    counters = {
        "created": "created",
        "linked": "linked",
        "routing_invalid": "invalid",
        "routing_truncated": "truncated",
        "out_of_scope": "out_of_scope",
        "covered_by_successor": "covered_by_successor",
        "stale_evidence": "stale_evidence",
        "comparison_unresolved": "comparison_unresolved",
        "invalid_routing": "invalid",
    }
    for result in results:
        if isinstance(result, BaseException):
            report.errors += 1
            # `str(TimeoutError())` is empty — logging only the message prints
            # "failed: " and hides the single most common failure mode.
            _logger.warning(
                "evidence routing failed for one finding: %s: %s",
                type(result).__name__,
                result,
            )
            continue
        report.routed += 1
        attr = counters.get(result)
        if attr:
            setattr(report, attr, getattr(report, attr) + 1)
    return report


def make_asker(
    *,
    model_name: str,
    api_base: str | None,
    api_key: str | None,
    # Measured on the real prompt: a single routing call is ~150s, because the
    # prompt asks for a per-variant dismissal with citations and the model is a
    # reasoning one. 240s left only 1.6x headroom, so any load tipped calls over
    # the line — those were the "endpoint is overloaded" timeouts, and they were
    # actually this ceiling. Concurrency itself measured clean: 6 parallel calls
    # each still returned in ~139s with zero failures.
    timeout: int = 420,
    max_tokens: int = 16384,
) -> Callable[[str], Awaitable[str]]:
    """A one-shot completion callable — the same LiteLLM path the multi-angle
    analysis already uses. Kept separate from the pass itself so tests inject a
    fake and never touch an endpoint."""

    async def _ask(prompt: str) -> str:
        from harbor.llms.base import OutputLengthExceededError
        from harbor.llms.lite_llm import LiteLLM

        llm = LiteLLM(model_name=model_name, api_base=api_base, api_key=api_key)
        # The prompt asks for citations and a per-variant dismissal, so replies
        # are long by design — and on a reasoning model the hidden reasoning eats
        # the same budget, so a ceiling sized for the visible answer alone is far
        # too small.
        try:
            resp = await asyncio.wait_for(
                llm.call(prompt=prompt, max_tokens=max_tokens), timeout=timeout
            )
        except OutputLengthExceededError as exc:
            # Hand back what did arrive: a cut-off reply is a token ceiling, and
            # the caller classifies it as `routing_truncated`. Letting it surface
            # as a bare exception would file an endpoint limit under "errors",
            # where it looks like flakiness instead of a knob that needs turning.
            return getattr(exc, "truncated_response", "") or ""
        return (resp.content or "") or (getattr(resp, "reasoning_content", "") or "")

    return _ask


def archive_view(
    archive_root: Path | str, modules_root: Path | str, locked_type: str
) -> tuple[list[tuple[str, str]], list[Any]]:
    """(active variants with descriptions, all archive entries) for one type.

    Both halves come from the same archive on purpose: the prompt's candidate
    list and the genealogy used for supersedes-chain mapping must agree, or every
    incumbent routing looks like stale evidence.
    """
    from harbor.agents.terminus_2_modular import archive as _archive
    from harbor.agents.terminus_2_modular.library import build_default_library

    entries = _archive.load_archive(archive_root)
    descriptions: dict[str, str] = {}
    try:
        for info in build_default_library(modules_root=str(modules_root)).list_infos():
            descriptions[f"{info.type}/{info.name}"] = info.description or ""
    except Exception as exc:  # a library that will not load is the smoke gate's job
        _logger.warning("evidence pass could not read module descriptions: %s", exc)
    active = [
        (e.qual, descriptions.get(e.qual, ""))
        for e in entries
        if e.type == locked_type and e.status == "active"
    ]
    return active, entries
