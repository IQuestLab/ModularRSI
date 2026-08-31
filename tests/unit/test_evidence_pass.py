"""Tests for routing and clustering findings into the proposal backlog."""

import json

import pytest

from harbor.agents.terminus_2_modular.self_evo import (
    clustering,
    evidence_pass,
    proposals,
)

pytestmark = pytest.mark.unit

STEP = "gen_7_candidate_42"
ACTIVE = [("agent_loop/confirm_exit", "stops when confident")]

_GOOD_ROUTING = json.dumps(
    {
        "lane": "incumbent",
        "action": "modify",
        "target_variant": "agent_loop/confirm_exit",
        "behavioral_delta": "check the artifact exists",
        "causal_hypothesis": "it trusts the self-report",
        "rationale": "confirm_exit._validate returns True on prose",
    }
)


def _snapshot(findings):
    return {"step": STEP, "findings": findings}


def _culprit(task="fix-git"):
    return {"task": task, "is_culprit": True, "lens": "agent_loop", "divergence": "x"}


class _FakeLLM:
    """Answers by prompt kind, and counts calls so cost can be asserted."""

    def __init__(self, routing_reply=None, same=False, covers=False, fail=False):
        self.routing_reply = (
            routing_reply
            if routing_reply is not None
            else f"<routing>{_GOOD_ROUTING}</routing>"
        )
        self.same = same
        self.covers = covers
        self.fail = fail
        self.prompts: list[str] = []

    async def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if self.fail:
            raise RuntimeError("endpoint said no")
        if "<routing>" in prompt:
            return self.routing_reply
        if "SAME intervention" in prompt:  # the pairwise comparison question
            return "YES same intervention" if self.same else "NO different"
        return "YES covered" if self.covers else "NO gap survived"

    @property
    def n_routing(self):
        return sum(1 for p in self.prompts if "<routing>" in p)


def _entries():
    from harbor.agents.terminus_2_modular import archive as _archive

    # the prompt's variant list and the genealogy must come from one archive —
    # if they disagree, every incumbent routing looks like stale evidence
    return [
        _archive.ArchiveEntry(
            name=q.split("/")[1], type=q.split("/")[0], status="active"
        )
        for q, _ in ACTIVE
    ]


async def _run(tmp_path, findings, llm, **over):
    kw = dict(
        snapshot=_snapshot(findings),
        diagnose_items=None,
        active_variants=ACTIVE,
        locked_type="agent_loop",
        step=STEP,
        archive_entries=_entries(),
        ask=llm,
    )
    kw.update(over)
    return await evidence_pass.collect_evidence(tmp_path, **kw)


def _audit(tmp_path):
    path = clustering.audit_path(tmp_path)
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


# ---- what gets routed -----------------------------------------------------


async def test_a_culprit_finding_becomes_a_proposal(tmp_path):
    report = await _run(tmp_path, [_culprit()], _FakeLLM())
    assert report.created == 1
    (p,) = proposals.load_proposals(tmp_path)
    assert p.target_variant == "agent_loop/confirm_exit"
    assert p.lane == "incumbent"


async def test_non_culprit_findings_are_never_routed(tmp_path):
    # 201/318 real findings are non-culprits. Routing them would be pure cost:
    # they assert there is nothing here to fix.
    llm = _FakeLLM()
    report = await _run(tmp_path, [{"task": "t", "is_culprit": False}], llm)
    assert llm.n_routing == 0
    assert report.routed == 0
    assert proposals.load_proposals(tmp_path) == []


async def test_every_culprit_is_routed_exactly_once(tmp_path):
    llm = _FakeLLM()
    await _run(tmp_path, [_culprit("a"), _culprit("b"), _culprit("c")], llm)
    assert llm.n_routing == 3


# ---- a bad routing is refused, not guessed -------------------------------


async def test_an_unusable_routing_creates_no_proposal(tmp_path):
    llm = _FakeLLM(routing_reply="I am not sure what to say.")
    report = await _run(tmp_path, [_culprit()], llm)
    assert report.invalid == 1
    assert proposals.load_proposals(tmp_path) == []


async def test_an_unusable_routing_is_written_down(tmp_path):
    llm = _FakeLLM(routing_reply="I am not sure what to say.")
    await _run(tmp_path, [_culprit()], llm)
    assert _audit(tmp_path)[-1]["decision"] == "routing_invalid"
    assert _audit(tmp_path)[-1]["reason"]


async def test_a_routing_naming_a_dead_variant_is_refused(tmp_path):
    reply = json.dumps(
        {
            "lane": "incumbent",
            "action": "modify",
            "target_variant": "agent_loop/ghost",
            "behavioral_delta": "d",
            "causal_hypothesis": "c",
            "rationale": "r",
        }
    )
    llm = _FakeLLM(routing_reply=f"<routing>{reply}</routing>")
    report = await _run(tmp_path, [_culprit()], llm)
    assert report.invalid == 1


# ---- failures cannot take the generation down ----------------------------


async def test_an_endpoint_failure_does_not_break_the_pass(tmp_path):
    report = await _run(tmp_path, [_culprit()], _FakeLLM(fail=True))
    assert report.errors == 1
    assert report.created == 0


async def test_one_failed_finding_does_not_stop_the_others(tmp_path):
    class _Flaky(_FakeLLM):
        async def __call__(self, prompt):
            if "fix-git" in prompt and "<routing>" in prompt:
                raise RuntimeError("boom")
            return await super().__call__(prompt)

    report = await _run(tmp_path, [_culprit("fix-git"), _culprit("other")], _Flaky())
    assert report.errors == 1
    assert report.created == 1


# ---- it observes; it does not decide -------------------------------------


async def test_the_pass_reports_what_it_saw(tmp_path):
    report = await _run(tmp_path, [_culprit("a"), _culprit("b")], _FakeLLM())
    assert report.findings == 2
    assert report.routed == 2
    assert report.created + report.linked + report.invalid + report.errors == 2


async def test_support_accumulates_when_the_comparison_says_same(tmp_path):
    await _run(tmp_path, [_culprit("a")], _FakeLLM())
    report = await _run(tmp_path, [_culprit("b")], _FakeLLM(same=True), step="gen_8_c")
    assert report.linked == 1
    (p,) = proposals.load_proposals(tmp_path)
    assert p.support_tasks == ["a", "b"]


async def test_a_truncated_reply_is_not_counted_as_a_model_refusal(tmp_path):
    cut_off = '<routing>\n{"lane": "incumbent", "rationale": "it ran budget_aware and'
    report = await _run(tmp_path, [_culprit()], _FakeLLM(routing_reply=cut_off))
    assert report.truncated == 1
    assert report.invalid == 0
    assert _audit(tmp_path)[-1]["decision"] == "routing_truncated"


async def test_genuine_prose_with_no_block_is_still_a_refusal(tmp_path):
    report = await _run(
        tmp_path, [_culprit()], _FakeLLM(routing_reply="I cannot tell.")
    )
    assert report.invalid == 1
    assert report.truncated == 0


# ---- cost and latency are different problems -----------------------------
# Measured on the real replay: 12.7 min per step with only 3 culprits, while the
# cross-finding semaphore was set to 3. The findings WERE running in parallel;
# what was serial is the inside of one finding — the routing call, then one
# pairwise comparison after another, all while holding a single slot. A finding
# facing 4 open proposals is 5 sequential round trips.


async def test_pairwise_comparisons_do_not_run_one_after_another(tmp_path):
    import asyncio

    for i in range(4):
        proposals.create_proposal(
            tmp_path,
            step="s",
            lane="incumbent",
            action="modify",
            target_variant="agent_loop/confirm_exit",
            finding_ids=[f"f_seed{i}"],
            behavioral_delta=f"unrelated change {i}",
        )

    class _SlowCompare(_FakeLLM):
        def __init__(self):
            super().__init__()
            self.in_flight = 0
            self.peak = 0

        async def __call__(self, prompt):
            if "SAME intervention" in prompt:
                self.in_flight += 1
                self.peak = max(self.peak, self.in_flight)
                await asyncio.sleep(0.02)
                self.in_flight -= 1
                return "NO different"
            return await super().__call__(prompt)

    llm = _SlowCompare()
    await _run(tmp_path, [_culprit()], llm)
    assert llm.peak > 1, "comparisons still serialised — one round trip at a time"


async def test_an_already_clustered_finding_costs_no_endpoint_call(tmp_path):
    # resume: re-running a window must not re-pay for findings already placed.
    # Without this the routing call is spent first and only then discarded.
    await _run(tmp_path, [_culprit()], _FakeLLM())
    llm = _FakeLLM()
    report = await _run(tmp_path, [_culprit()], llm)
    assert llm.prompts == []
    assert report.routed == 0
    assert len(proposals.load_proposals(tmp_path)) == 1


async def test_the_novelty_dismissals_are_kept_for_audit(tmp_path):
    """The per-variant "why this one is not it" is the load-bearing evidence.

    It is the claim that turned out wrong 13 times out of 15, and it was the one
    thing the audit did not store — the disagreement could only be reconstructed
    from the free-text rationale. A judgement whose justification is discarded
    cannot be checked later, which is precisely when you need it.
    """
    reply = json.dumps(
        {
            "lane": "novelty",
            "action": "add",
            "behavioral_delta": "track requirements across episodes",
            "causal_hypothesis": "nothing carries them forward",
            "rationale": "no episode ever re-reads the requirement list",
            "dismissals": {
                "agent_loop/confirm_exit": "only counts completion signals",
            },
        }
    )
    await _run(
        tmp_path, [_culprit()], _FakeLLM(routing_reply=f"<routing>{reply}</routing>")
    )
    row = _audit(tmp_path)[-1]
    assert row["dismissals"] == {
        "agent_loop/confirm_exit": "only counts completion signals"
    }


async def test_an_incumbent_routing_stores_no_dismissals(tmp_path):
    await _run(tmp_path, [_culprit()], _FakeLLM())
    assert _audit(tmp_path)[-1].get("dismissals") in ({}, None)


# ---- H2: two findings clustering at the same time ------------------------
#
# Findings are routed concurrently, which is right — routing is one independent
# LLM call each. But each finding also snapshots the open proposals BEFORE its
# pairwise comparisons, and creates AFTER them. Two findings whose comparison
# calls overlap therefore both decide "no open proposal matches", and both
# create — one intervention, two proposals, forever competing for the same lane
# slot. The clusterer re-reads the backlog before creating, so it does SEE the
# sibling's proposal; it just has no answer for it, and an unanswered candidate
# was being read as a confident "different".


class _YieldingLLM(_FakeLLM):
    """Like `_FakeLLM`, but actually suspends — so findings interleave.

    `_FakeLLM` never awaits anything, so a whole pass runs to completion one
    finding at a time and no concurrency bug can appear. A real endpoint call
    suspends at every prompt; this reproduces that, and nothing else.

    It answers "same intervention" by comparing the two deltas the prompt
    quotes, which is what a competent judge would do and what makes duplicate
    proposals detectable at all.
    """

    def __init__(self, delta="check the artifact exists"):
        super().__init__()
        self.delta = delta

    async def __call__(self, prompt):
        import asyncio

        await asyncio.sleep(0)
        if "SAME intervention" in prompt:
            # the candidate's delta is quoted once; if the proposal's is the
            # same string it appears twice
            return "YES same" if prompt.count(self.delta) >= 2 else "NO different"
        return await super().__call__(prompt)


def _seed_unrelated_proposal(tmp_path):
    """One open proposal on the same variant, about something else.

    Without it neither finding has anything to compare against, so neither
    suspends between snapshot and create and the two never overlap.
    """
    return proposals.create_proposal(
        tmp_path,
        step="gen_6_candidate_1",
        lane="incumbent",
        action="modify",
        target_variant="agent_loop/confirm_exit",
        behavioral_delta="retry the failed download",
        causal_hypothesis="the mirror is flaky",
        finding_ids=["f_seed"],
        support_tasks=["seed-task"],
    )


async def test_two_simultaneous_findings_do_not_create_two_proposals(tmp_path):
    _seed_unrelated_proposal(tmp_path)
    report = await _run(tmp_path, [_culprit("a"), _culprit("b")], _YieldingLLM())
    assert (report.created, report.linked) == (1, 1)
    assert len(proposals.load_proposals(tmp_path)) == 2  # the seed + one new


async def test_the_second_finding_is_compared_against_the_first_ones_proposal(
    tmp_path,
):
    # the fix is not "create fewer": it is that the proposal that appeared while
    # we were comparing gets compared too, before we decide it is new
    _seed_unrelated_proposal(tmp_path)
    await _run(tmp_path, [_culprit("a"), _culprit("b")], _YieldingLLM())
    linked = [row for row in _audit(tmp_path) if row.get("decision") == "linked"]
    assert len(linked) == 1
    compared = {c["proposal_id"] for c in linked[0]["comparisons"]}
    assert "p_0002" in compared, "never asked about the sibling's proposal"


async def test_support_from_both_findings_lands_on_one_proposal(tmp_path):
    # the reason duplicates matter: split support looks like two weak
    # directions instead of one with two tasks behind it
    _seed_unrelated_proposal(tmp_path)
    await _run(tmp_path, [_culprit("a"), _culprit("b")], _YieldingLLM())
    new = [p for p in proposals.load_proposals(tmp_path) if p.proposal_id != "p_0001"]
    assert len(new) == 1
    assert sorted(new[0].support_tasks) == ["a", "b"]


# ---- a comparison that failed is not a "no" ------------------------------


class _ComparisonFailsLLM(_FakeLLM):
    """Routes fine; every pairwise comparison raises."""

    async def __call__(self, prompt):
        import asyncio

        await asyncio.sleep(0)
        if "SAME intervention" in prompt:
            raise RuntimeError("endpoint said no")
        return await super().__call__(prompt)


async def test_an_unanswerable_comparison_does_not_create_a_rival_proposal(tmp_path):
    # "the comparison errored" and "these are different" are not the same fact.
    # Creating on the first one manufactures the duplicate the comparison
    # existed to prevent; leaving the finding unattached costs one routing call
    # next window and nothing else.
    _seed_unrelated_proposal(tmp_path)
    report = await _run(tmp_path, [_culprit("a")], _ComparisonFailsLLM())
    assert report.created == 0
    assert len(proposals.load_proposals(tmp_path)) == 1  # just the seed
    assert _audit(tmp_path)[-1]["decision"] == "comparison_unresolved"


# ---- M3: resume does not re-buy a verdict it already has -----------------


_OUT_OF_SCOPE_ROUTING = json.dumps(
    {
        "lane": "out_of_scope",
        "action": "",
        "other_module": "tools",
        "causal_hypothesis": "the tool never exposed the exit code",
        "rationale": "nothing in agent_loop can see it",
    }
)


async def test_an_out_of_scope_finding_is_not_routed_a_second_time(tmp_path):
    # the resume check only knew about proposals, so every finding that ended
    # any other way was re-routed every window — a routing call each time, for
    # a verdict already reached
    findings = [_culprit("a")]
    first = _FakeLLM(routing_reply=f"<routing>{_OUT_OF_SCOPE_ROUTING}</routing>")
    await _run(tmp_path, findings, first)
    assert first.n_routing == 1

    second = _FakeLLM(routing_reply=f"<routing>{_OUT_OF_SCOPE_ROUTING}</routing>")
    report = await _run(tmp_path, findings, second, step="gen_8_candidate_1")
    assert second.n_routing == 0
    assert report.routed == 0


async def test_the_module_that_owns_it_gets_the_hand_off(tmp_path):
    from harbor.agents.terminus_2_modular.self_evo import dispositions

    llm = _FakeLLM(routing_reply=f"<routing>{_OUT_OF_SCOPE_ROUTING}</routing>")
    report = await _run(tmp_path, [_culprit("a")], llm)
    assert report.out_of_scope == 1
    assert len(dispositions.out_of_scope_for(tmp_path, "tools")) == 1


async def test_a_finding_the_endpoint_could_not_judge_is_routed_again(tmp_path):
    # the negative that matters: an unanswered comparison is weather, not a
    # verdict. Retiring it here would silently drop real evidence whenever the
    # endpoint had a bad afternoon.
    _seed_unrelated_proposal(tmp_path)
    findings = [_culprit("a")]
    await _run(tmp_path, findings, _ComparisonFailsLLM())

    retry = _FakeLLM(same=False)
    report = await _run(tmp_path, findings, retry, step="gen_8_candidate_1")
    assert retry.n_routing == 1
    assert report.created == 1


async def test_a_malformed_routing_reply_is_not_paid_for_twice(tmp_path):
    findings = [_culprit("a")]
    await _run(tmp_path, findings, _FakeLLM(routing_reply="<routing>{}</routing>"))
    again = _FakeLLM(routing_reply="<routing>{}</routing>")
    await _run(tmp_path, findings, again, step="gen_8_candidate_1")
    assert again.n_routing == 0


async def test_a_reply_cut_off_at_the_token_ceiling_is_tried_again(tmp_path):
    # a truncated reply is an endpoint ceiling, not a model that could not
    # decide — retiring the finding for it would be filing an infra artifact as
    # a verdict about the evidence
    findings = [_culprit("a")]
    truncated = '<routing>{"lane": "incumbent", "action": "mod'
    await _run(tmp_path, findings, _FakeLLM(routing_reply=truncated))
    again = _FakeLLM()
    report = await _run(tmp_path, findings, again, step="gen_8_candidate_1")
    assert again.n_routing == 1
    assert report.created == 1
