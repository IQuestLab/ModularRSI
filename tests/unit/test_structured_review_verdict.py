"""P0-1 (rev 2, execution-gated): a review verdict EXISTS only when the
`review_verdict` ACTION was executed by the tool handler, which writes it to a
file in the review staging. Nothing is ever derived from response/message
TEXT.

Why rev 2: the first structured version still scanned the raw response/agent
message for a valid tag — a reviewer merely *planning* in <analysis>
("I will later submit <review_verdict .../> after checking") completed the
session as ACCEPT with zero commands (reproduced). Text containment cannot
distinguish quoting from submitting; only execution can.

Protocol: submit action → handler validates, records on the tools instance,
writes `.review_verdict.json` into staging → next parse ends the session.
No submission → one reminder → then the session is force-ended verdict-less
and the outer driver records `review_skipped_parse_failure` (with the
run-level circuit breaker on top).
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from harbor.agents.terminus_2_modular.modules.tools.editor_file_tools import (
    EditorFileTools,
)
from harbor.agents.terminus_2_modular.protocols import ToolCall
from harbor.agents.terminus_2_modular.self_evo import review_verdict as rv

pytestmark = pytest.mark.unit

FIXTURES = Path(__file__).parent.parent / "fixtures" / "review_trajectories"


# ---- attribute validation --------------------------------------------------


def test_accept_is_valid():
    assert rv.validate_attrs({"decision": "accept", "reason": "clean change"}) is None


def test_reject_requires_a_class():
    err = rv.validate_attrs({"decision": "reject", "reason": "overfit"})
    assert err is not None and "reject_class" in err


def test_reject_with_implementation_class_is_valid():
    attrs = {
        "decision": "reject",
        "reject_class": "implementation",
        "reason": "substring bug",
        "repair_brief": "use exact-match on the score token",
    }
    assert rv.validate_attrs(attrs) is None


def test_accept_with_reject_class_is_invalid():
    err = rv.validate_attrs(
        {"decision": "accept", "reject_class": "proposal", "reason": "x"}
    )
    assert err is not None


def test_accept_partial_is_deferred_to_phase1():
    err = rv.validate_attrs({"decision": "accept_partial", "reason": "x"})
    assert err is not None


def test_reason_is_required():
    err = rv.validate_attrs({"decision": "accept", "reason": "  "})
    assert err is not None


def test_template_with_pipes_is_invalid():
    # the instruction's own format spec must never validate
    err = rv.validate_attrs({"decision": "accept|reject", "reason": "..."})
    assert err is not None


# ---- the file channel ------------------------------------------------------


def test_verdict_file_roundtrip(tmp_path):
    v = rv.write_verdict_file(
        tmp_path, {"decision": "reject", "reject_class": "proposal", "reason": "absent"}
    )
    assert v["decision"] == "reject" and v["reject_class"] == "proposal"
    assert rv.read_verdict_file(tmp_path) == v


def test_read_missing_file_is_none(tmp_path):
    assert rv.read_verdict_file(tmp_path) is None


def test_read_corrupt_or_invalid_content_is_none(tmp_path):
    (tmp_path / rv.VERDICT_FILENAME).write_text("not json")
    assert rv.read_verdict_file(tmp_path) is None
    (tmp_path / rv.VERDICT_FILENAME).write_text(
        json.dumps({"decision": "accept_partial", "reason": "x"})
    )
    assert rv.read_verdict_file(tmp_path) is None


# ---- no text-derived verdicts, ever (the rev-2 contract) -------------------


def test_no_text_scan_api_exists():
    """The message/response scanning APIs are gone — a verdict can only come
    from the executed action's file. This pin guards against re-adding them."""
    assert not hasattr(rv, "parse_from_trajectory")
    assert not hasattr(rv, "find_valid")
    import inspect

    from harbor.agents.terminus_2_modular.self_evo import online_evo as OE

    src = inspect.getsource(OE)
    assert "read_verdict_file" in src
    assert "parse_from_trajectory" not in src


# ---- session mechanics -----------------------------------------------------


def _review_tools() -> EditorFileTools:
    t = EditorFileTools()
    t._review_mode = True
    t._review_verdict_recorded = None
    t._verdict_reminders = 0
    return t


def _ctx(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        state=SimpleNamespace(
            staging_dir=tmp_path,
            trajectory_root=None,
            skills_dir=None,
            archive_path=None,
        ),
        shared=SimpleNamespace(),
    )


def _verdict_call(**attrs) -> ToolCall:
    return ToolCall(
        keystrokes=json.dumps({"action": "review_verdict", **attrs}), duration_sec=0.0
    )


def test_analysis_quote_does_not_complete_or_count():
    """The rev-1 hole, exact repro: a fully-valid tag inside <analysis> with an
    empty <actions> block used to complete the session (is_complete=True,
    commands=0). Now: not complete, nothing recorded."""
    t = _review_tools()
    r = t.parse_llm_response(
        "<response>\n<analysis>I will later submit\n"
        '<review_verdict decision="accept" reason="clean"/>\n'
        "after checking the trajectories.</analysis>\n"
        "<plan>keep reading</plan>\n<actions></actions>\n"
        "<task_complete>false</task_complete>\n</response>"
    )
    assert not r.is_task_complete
    assert t._review_verdict_recorded is None


async def test_submission_records_then_next_parse_completes(tmp_path):
    t = _review_tools()
    # turn N: the action is extracted as a command, session does NOT end yet
    r = t.parse_llm_response(
        "<response><actions>"
        '<review_verdict decision="accept" reason="general and effective"/>'
        "</actions><task_complete>false</task_complete></response>"
    )
    assert not r.is_task_complete and not r.error
    payloads = [json.loads(c.keystrokes) for c in r.commands]
    assert any(p.get("action") == "review_verdict" for p in payloads)
    # execution: handler validates, records, writes the file
    res = await t.execute(_verdict_call(decision="accept", reason="ok"), _ctx(tmp_path))
    assert res.success and "recorded" in res.output
    assert (tmp_path / rv.VERDICT_FILENAME).is_file()
    # turn N+1: whatever the model says, the session ends
    r2 = t.parse_llm_response("done.")
    assert r2.is_task_complete


async def test_task_complete_alongside_submission_lets_it_execute(tmp_path):
    t = _review_tools()
    r = t.parse_llm_response(
        "<response><actions>"
        '<review_verdict decision="reject" reject_class="proposal" reason="absent"/>'
        "</actions><task_complete>true</task_complete></response>"
    )
    # not complete yet (nothing recorded), but NO error either — the pending
    # submission must be allowed to execute this turn
    assert not r.is_task_complete and not r.error


def test_task_complete_without_submission_gets_one_reminder_then_ends():
    t = _review_tools()
    r1 = t.parse_llm_response("I am done.\n<task_complete>true</task_complete>")
    assert not r1.is_task_complete and "review_verdict" in r1.error
    # second attempt to end without a verdict → session force-ends; the outer
    # driver finds no verdict file and records the skip
    r2 = t.parse_llm_response("Really done.\n<task_complete>true</task_complete>")
    assert r2.is_task_complete
    assert t._review_verdict_recorded is None


def test_commandless_prose_stall_gets_reminder():
    # Codex 附属2: prose with no commands and no task_complete used to burn
    # turns silently; now the first stall carries the reminder
    t = _review_tools()
    r = t.parse_llm_response("VERDICT: ACCEPT — all clean.")
    assert not r.is_task_complete
    assert "review_verdict" in r.error


def test_turns_with_commands_are_not_stalls():
    t = _review_tools()
    r = t.parse_llm_response('<actions><read_file path="agent_loop/x.py"/></actions>')
    assert not r.is_task_complete and not r.error
    assert t._verdict_reminders == 0


def test_non_review_session_task_complete_unaffected():
    t = EditorFileTools()
    t._review_mode = False
    r = t.parse_llm_response("<task_complete>true</task_complete>")
    assert r.is_task_complete
    assert not r.error


# ---- the action handler ----------------------------------------------------


async def test_invalid_action_returns_error_to_reviewer(tmp_path):
    t = _review_tools()
    res = await t.execute(
        _verdict_call(decision="reject", reason="bad"), _ctx(tmp_path)
    )
    assert not res.success
    assert "reject_class" in res.error
    assert t._review_verdict_recorded is None
    assert not (tmp_path / rv.VERDICT_FILENAME).exists()


async def test_action_rejected_outside_review_mode(tmp_path):
    t = EditorFileTools()
    t._review_mode = False
    res = await t.execute(_verdict_call(decision="accept", reason="x"), _ctx(tmp_path))
    assert not res.success
    assert "review" in res.error.lower()


async def test_resubmission_last_wins(tmp_path):
    t = _review_tools()
    await t.execute(
        _verdict_call(decision="reject", reject_class="proposal", reason="v1"),
        _ctx(tmp_path),
    )
    await t.execute(_verdict_call(decision="accept", reason="v2 final"), _ctx(tmp_path))
    v = rv.read_verdict_file(tmp_path)
    assert v is not None and v["decision"] == "accept" and v["reason"] == "v2 final"


# ---- the 7 real trajectories under the new protocol ------------------------
# 5 confirmed false rejects + 2 true rejects. None of these sessions executed
# the action (it predates them), so under the execution-gated protocol every
# one becomes a SKIP — no message text can kill (or promote) a candidate again.


@pytest.mark.parametrize(
    "fixture",
    [
        "false_reject_classifier_1",
        "false_reject_spec_quote",
        "false_reject_prose_early_word",
        "false_reject_prose_misread",
        "false_reject_classifier_2",
        "true_reject_substring_bug",
        "true_reject_overfit",
    ],
)
def test_real_trajectories_contain_no_executable_submission(fixture):
    data = json.loads((FIXTURES / f"{fixture}.json").read_text())
    t = _review_tools()
    for step in data["steps"]:
        msg = step.get("message") or ""
        if not msg.strip():
            continue
        for cmd in t._extract_actions(msg):
            payload = json.loads(cmd.keystrokes)
            assert payload.get("action") != "review_verdict", fixture


# ---- run-level circuit breaker (unchanged from rev 1) ----------------------


def test_single_skip_passes_through_without_tripping(tmp_path):
    h = rv.record_review_outcome(tmp_path, skipped=True)
    assert not rv.breaker_tripped(h)


def test_three_consecutive_skips_trip(tmp_path):
    rv.record_review_outcome(tmp_path, skipped=True)
    rv.record_review_outcome(tmp_path, skipped=True)
    h = rv.record_review_outcome(tmp_path, skipped=True)
    assert rv.breaker_tripped(h)


def test_success_resets_the_consecutive_counter(tmp_path):
    rv.record_review_outcome(tmp_path, skipped=True)
    rv.record_review_outcome(tmp_path, skipped=True)
    rv.record_review_outcome(tmp_path, skipped=False)
    h = rv.record_review_outcome(tmp_path, skipped=True)
    assert h["consecutive_skips"] == 1
    assert not rv.breaker_tripped(h)


def test_ratio_over_ten_percent_trips_at_ten_reviews(tmp_path):
    order = [True, False, False, False, False, True, False, False, False, False]
    h = {}
    for skipped in order:
        h = rv.record_review_outcome(tmp_path, skipped=skipped)
    assert h["total_reviews"] == 10 and h["total_skips"] == 2
    assert rv.breaker_tripped(h)


def test_ratio_at_exactly_ten_percent_does_not_trip(tmp_path):
    order = [True] + [False] * 9
    h = {}
    for skipped in order:
        h = rv.record_review_outcome(tmp_path, skipped=skipped)
    assert not rv.breaker_tripped(h)


def test_health_survives_reload_from_disk(tmp_path):
    rv.record_review_outcome(tmp_path, skipped=True)
    rv.record_review_outcome(tmp_path, skipped=True)
    h = rv.record_review_outcome(tmp_path, skipped=True)
    assert h["consecutive_skips"] == 3 and rv.breaker_tripped(h)


# ---- persistence of reject_class / repair_brief (Codex 附属3) ---------------


def test_reflection_outcome_carries_reject_class_fields():
    import dataclasses

    from harbor.agents.terminus_2_modular.self_evo import online_evo as OE

    names = {f.name for f in dataclasses.fields(OE._ReflectionOutcome)}
    assert "review_reject_class" in names
    assert "review_repair_brief" in names
