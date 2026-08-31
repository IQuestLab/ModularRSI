"""P1-2a: every finding is persisted with a stable ID before any selection.

Today `findings/<step>.json` is a per-step SNAPSHOT: 318 findings in the
v3mini_contrastive run, none of them addressable. Nothing can reference a
finding across windows, so an unselected finding evaporates the moment the
reflection ends — which is exactly how a good direction gets starved (the
"C. 好方向被贪心饿死" row of the plan's problem table).

This layer is append-only; selection is handled by the portfolio.
It gives each raw finding an ID that is stable across re-runs (the loop resumes
mid-run — `ARCHIVE_ROOT` 续跑跳已完成题) and appends it to a ledger that never
rewrites history.
"""

import json

import pytest

from harbor.agents.terminus_2_modular.self_evo import backlog

pytestmark = pytest.mark.unit


STEP = "gen_10_candidate_1787443046"


def _payload(findings):
    """Shape of a real findings/<step>.json (top keys: step / router / findings)."""
    return {
        "step": STEP,
        "router": {"fixable_fail": ["draft_dp_58745bc61ce_outa"], "mixed": []},
        "findings": findings,
    }


def _finding(**over):
    base = {
        "task": "draft_dp_58745bc61ce_outa",
        "is_culprit": True,
        "lens": "termination",
        "locked_module": "agent_loop",
        "divergence": "the failing roll never wrote a plan before acting",
        "would_change_outcome": "a plan-first loop would have caught it",
        "fixable_now": True,
        "suggested_change": "require a plan on every turn",
    }
    base.update(over)
    return base


# ---- stable IDs -----------------------------------------------------------


def test_id_is_stable_across_calls():
    a = backlog.finding_id(step=STEP, task="t1", lens="termination", index=0)
    b = backlog.finding_id(step=STEP, task="t1", lens="termination", index=0)
    assert a == b


def test_id_varies_with_each_coordinate():
    base = dict(step=STEP, task="t1", lens="termination", index=0)
    ids = {
        backlog.finding_id(**base),
        backlog.finding_id(**{**base, "step": "gen_11_candidate_1"}),
        backlog.finding_id(**{**base, "task": "t2"}),
        backlog.finding_id(**{**base, "lens": "planning"}),
        backlog.finding_id(**{**base, "index": 1}),
    }
    assert len(ids) == 5


def test_same_task_and_lens_twice_in_one_step_are_distinct():
    # 302/318 real findings carry a lens; duplicates within a step are common,
    # so position has to be part of the identity or two findings collapse.
    first = backlog.finding_id(step=STEP, task="t1", lens="termination", index=0)
    second = backlog.finding_id(step=STEP, task="t1", lens="termination", index=1)
    assert first != second


# ---- ingest ---------------------------------------------------------------


def test_ingest_returns_one_record_per_finding(tmp_path):
    recs = backlog.ingest_step(tmp_path, _payload([_finding(), _finding(task="t2")]))
    assert [r.task for r in recs] == ["draft_dp_58745bc61ce_outa", "t2"]
    assert all(r.finding_id for r in recs)
    assert len({r.finding_id for r in recs}) == 2


def test_ingest_preserves_the_whole_original_finding(tmp_path):
    # "记录 task / window / citations / 原始诊断" — a lossy projection would
    # destroy the only record of what the investigator actually said.
    original = _finding(note="investigator was unsure", other_module="verification")
    (rec,) = backlog.ingest_step(tmp_path, _payload([original]))
    assert rec.raw_finding == original


def test_ingest_records_the_window(tmp_path):
    (rec,) = backlog.ingest_step(tmp_path, _payload([_finding()]))
    assert rec.step == STEP


def test_ingest_is_idempotent(tmp_path):
    # the runner resumes mid-run; re-ingesting a step must not double-count
    # evidence (a duplicated support set would fake consensus).
    backlog.ingest_step(tmp_path, _payload([_finding(), _finding(task="t2")]))
    backlog.ingest_step(tmp_path, _payload([_finding(), _finding(task="t2")]))
    assert len(backlog.load_findings(tmp_path)) == 2


def test_reingesting_a_step_does_not_drop_a_late_finding(tmp_path):
    backlog.ingest_step(tmp_path, _payload([_finding()]))
    backlog.ingest_step(tmp_path, _payload([_finding(), _finding(task="t2")]))
    assert {r.task for r in backlog.load_findings(tmp_path)} == {
        "draft_dp_58745bc61ce_outa",
        "t2",
    }


# ---- the two real finding schemas -----------------------------------------
# contrast findings (the live path) are keyed by `task`; lens findings have no
# `task` at all — they carry `evidence: [{task, what_happened}]`. Both must land
# in the same ledger or half the corpus is invisible to the coverage report.


def test_lens_shaped_finding_without_a_task_key_is_recorded(tmp_path):
    lens_finding = {
        "lens": "termination",
        "is_culprit": True,
        "evidence": [
            {"task": "fix-git", "what_happened": "declared done, tests never run"},
            {"task": "posix-tar-r-w", "what_happened": "stopped at ep2"},
        ],
        "gap": "no evidence-based finishing",
        "fixable_now": True,
    }
    (rec,) = backlog.ingest_step(tmp_path, _payload([lens_finding]))
    assert rec.finding_id
    assert rec.lens == "termination"
    assert rec.raw_finding == lens_finding


def test_support_tasks_come_from_task_or_evidence(tmp_path):
    contrast = _finding(task="fix-git")
    lens = {
        "lens": "termination",
        "is_culprit": True,
        "evidence": [{"task": "a"}, {"task": "b"}, {"task": "a"}],
    }
    got = backlog.ingest_step(tmp_path, _payload([contrast, lens]))
    assert got[0].support_tasks == ["fix-git"]
    assert got[1].support_tasks == ["a", "b"]  # deduped, order preserved


def test_evidence_entries_without_a_task_are_skipped_not_fatal(tmp_path):
    lens = {"lens": "x", "is_culprit": True, "evidence": [{"what_happened": "?"}]}
    (rec,) = backlog.ingest_step(tmp_path, _payload([lens]))
    assert rec.support_tasks == []


# ---- provenance -----------------------------------------------------------
# Everything a citation needs (trial dirs, router bucket, contrast source) is in
# scope at emit time (online_evo.py:2364-2412) but is dropped on the floor today.


def test_provenance_is_attached_when_the_caller_supplies_it(tmp_path):
    prov = {
        "fix-git": {
            "bucket": "fixable_fail",
            "source": "archive_history",
            "fail_trial": "/runs/solver/fix-git.1",
            "pass_trial": None,
        }
    }
    (rec,) = backlog.ingest_step(
        tmp_path, _payload([_finding(task="fix-git")]), provenance=prov
    )
    assert rec.provenance["bucket"] == "fixable_fail"
    assert rec.provenance["source"] == "archive_history"


def test_missing_provenance_is_an_empty_dict_not_a_crash(tmp_path):
    (rec,) = backlog.ingest_step(tmp_path, _payload([_finding()]))
    assert rec.provenance == {}


# ---- lossy edges seen in the real corpus ----------------------------------


def test_finding_with_only_a_raw_blob_is_still_recorded(tmp_path):
    # 16/318 real findings are unparsed `raw` text. Dropping them would silently
    # shrink the evidence base and make the coverage report (P1-7) lie.
    (rec,) = backlog.ingest_step(tmp_path, _payload([{"raw": "model wrote prose"}]))
    assert rec.raw_finding == {"raw": "model wrote prose"}
    assert rec.finding_id


def test_missing_optional_fields_do_not_crash(tmp_path):
    # only 128/318 carry `note`, 202/318 carry `other_module`
    (rec,) = backlog.ingest_step(
        tmp_path, _payload([{"task": "t", "is_culprit": False}])
    )
    assert rec.lens == ""
    assert rec.is_culprit is False


def test_ingest_tolerates_a_step_with_no_findings(tmp_path):
    assert backlog.ingest_step(tmp_path, _payload([])) == []
    assert backlog.load_findings(tmp_path) == []


# ---- the ledger is append-only -------------------------------------------


def test_ledger_is_jsonl_under_the_archive_root(tmp_path):
    backlog.ingest_step(tmp_path, _payload([_finding()]))
    path = tmp_path / "backlog" / "findings.jsonl"
    rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    assert len(rows) == 1
    assert rows[0]["finding_id"]
    assert rows[0]["step"] == STEP


def test_ingest_skips_a_non_dict_finding_without_shifting_the_others(tmp_path):
    got = backlog.ingest_step(tmp_path, _payload(["oops", _finding(task="t2")]))
    assert [r.task for r in got] == ["t2"]
    assert got[0].index == 1  # position preserved: the ID stays honest


def test_a_corrupt_line_does_not_lose_the_rest_of_the_ledger(tmp_path):
    backlog.ingest_step(tmp_path, _payload([_finding()]))
    path = tmp_path / "backlog" / "findings.jsonl"
    path.write_text(path.read_text() + "{not json\n")
    backlog.ingest_step(tmp_path, _payload([_finding(task="t2")]))
    assert {r.task for r in backlog.load_findings(tmp_path)} == {
        "draft_dp_58745bc61ce_outa",
        "t2",
    }


# ---- wiring into the reflection ------------------------------------------
# The two snapshot writes (online_evo.py:2159 lens, :2420 contrast) are the only
# points where a complete findings list exists. The router's DiagnosisItem is in
# scope there and carries the citations the finding dict itself never stores.


def _diag(task, bucket, source, fail_dir, pass_dir=None):
    from harbor.agents.terminus_2_modular.self_evo.router import DiagnosisItem
    from harbor.agents.terminus_2_modular.self_evo.trajectory_analysis import (
        TrialSummary,
    )

    def _ts(name, trial_dir):
        return TrialSummary(
            task_name=task,
            trial_name=name,
            reward=None,
            exception_type=None,
            exception_message=None,
            n_episodes=0,
            n_input_tokens=0,
            n_output_tokens=0,
            trial_dir=trial_dir,
        )

    return DiagnosisItem(
        task=task,
        bucket=bucket,
        contrast_fail=_ts("fail.1", fail_dir),
        contrast_pass=_ts("pass.1", pass_dir) if pass_dir else None,
        source=source,
    )


def test_wiring_attaches_router_citations(tmp_path):
    from harbor.agents.terminus_2_modular.self_evo import online_evo as OE

    items = [_diag("fix-git", "mixed", "same_task", "/s/fix-git.1", "/s/fix-git.2")]
    OE._ingest_findings_to_backlog(
        tmp_path, _payload([_finding(task="fix-git")]), items
    )
    (rec,) = backlog.load_findings(tmp_path)
    assert rec.provenance["bucket"] == "mixed"
    assert rec.provenance["source"] == "same_task"
    assert rec.provenance["fail_trial"] == "/s/fix-git.1"
    assert rec.provenance["pass_trial"] == "/s/fix-git.2"


def test_wiring_records_the_absent_pass_roll(tmp_path):
    # all-fail tasks are exactly the ones the absence channel depends on —
    # "no pass roll" has to be visible in the ledger, not indistinguishable
    # from "nobody recorded it".
    from harbor.agents.terminus_2_modular.self_evo import online_evo as OE

    items = [_diag("t", "fixable_fail", "none", "/s/t.1", None)]
    OE._ingest_findings_to_backlog(tmp_path, _payload([_finding(task="t")]), items)
    (rec,) = backlog.load_findings(tmp_path)
    assert rec.provenance["pass_trial"] is None
    assert rec.provenance["source"] == "none"


def test_wiring_works_without_router_items(tmp_path):
    # the lens flow has no per-task routing context
    from harbor.agents.terminus_2_modular.self_evo import online_evo as OE

    OE._ingest_findings_to_backlog(tmp_path, _payload([_finding()]), None)
    assert len(backlog.load_findings(tmp_path)) == 1


def test_wiring_never_breaks_the_reflection(tmp_path):
    # the ledger is an observer. If it cannot write, the generation still runs.
    from harbor.agents.terminus_2_modular.self_evo import online_evo as OE

    blocked = tmp_path / "blocked"
    blocked.write_text("I am a file, not a directory")
    OE._ingest_findings_to_backlog(blocked, _payload([_finding()]), None)
