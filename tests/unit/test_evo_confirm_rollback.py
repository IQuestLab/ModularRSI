"""Cross-epoch confirm/rollback — the GLUE that actually touches the archive.

`test_evo_confirm.py` covers the pure scoring functions. This file covers
`confirm_and_rollback` / `confirm_and_rollback_helpers`: the part that reads the
run's records, decides, and flips archive status. That layer had zero coverage,
which is how the duplicate-rollback bug below survived a full 9-generation run.

Why this layer matters: promotion uses review and crash gates, so cross-epoch
confirm is the ONLY quality control left in the loop. If it silently does nothing
(or does the same thing forever), a regressing variant stays live indefinitely.
"""

import json

import pytest

from harbor.agents.terminus_2_modular import archive as arch
from harbor.agents.terminus_2_modular.self_evo import confirm as C

MT = "agent_loop"


class _Summary:
    def __init__(self, bundle):
        self.bundle = bundle


class _Rec:
    """Shape `confirm_and_rollback` reads: .task_name / .summary.bundle / .reward"""

    def __init__(self, task, bundle, reward):
        self.task_name = task
        self.summary = _Summary(bundle)
        self.reward = reward


def _seed(run_dir, entries):
    arch.update_archive(
        run_dir,
        [
            arch.ArchiveEntry(
                name=n, type=t, niche={"axis": n}, status=s, born_gen="gen_0"
            )
            for n, t, s in entries
        ],
    )


def _status(run_dir, qual):
    return {e.qual: e.status for e in arch.load_archive(run_dir)}.get(qual)


def _recs(variant, baseline, tasks, v_reward, b_reward):
    """One roll of `variant` and one of `baseline` on each task."""
    out = []
    for t in tasks:
        out.append(_Rec(t, {MT: variant}, v_reward))
        out.append(_Rec(t, {MT: baseline}, b_reward))
    return out


@pytest.mark.unit
class TestVariantRollback:
    def test_clear_regression_is_superseded(self, tmp_path):
        _seed(tmp_path, [("baseline", MT, "active"), ("bad", MT, "active")])
        rep = C.confirm_and_rollback(
            tmp_path,
            _recs("bad", "baseline", ["t1", "t2", "t3", "t4"], 0.0, 1.0),
            MT,
            "baseline",
        )
        assert rep["rolled_back"] == ["bad"]
        assert _status(tmp_path, f"{MT}/bad") == "superseded"
        # never retire the base everything falls back to
        assert _status(tmp_path, f"{MT}/baseline") == "active"

    def test_comparable_variant_survives(self, tmp_path):
        _seed(tmp_path, [("baseline", MT, "active"), ("ok", MT, "active")])
        rep = C.confirm_and_rollback(
            tmp_path,
            _recs("ok", "baseline", ["t1", "t2", "t3", "t4"], 1.0, 1.0),
            MT,
            "baseline",
        )
        assert rep["rolled_back"] == []
        assert _status(tmp_path, f"{MT}/ok") == "active"

    def test_too_few_shared_tasks_is_not_enough(self, tmp_path):
        """K is small and noisy — one or two bad tasks must not retire a variant."""
        _seed(tmp_path, [("baseline", MT, "active"), ("bad", MT, "active")])
        rep = C.confirm_and_rollback(
            tmp_path, _recs("bad", "baseline", ["t1", "t2"], 0.0, 1.0), MT, "baseline"
        )
        assert rep["rolled_back"] == []
        assert _status(tmp_path, f"{MT}/bad") == "active"

    def test_infra_none_rolls_do_not_convict(self, tmp_path):
        """reward=None is a sandbox/endpoint error, not a task failure — it must
        not count against whichever variant happened to hit it."""
        _seed(tmp_path, [("baseline", MT, "active"), ("unlucky", MT, "active")])
        recs = _recs("unlucky", "baseline", ["t1", "t2", "t3", "t4"], None, 1.0)
        rep = C.confirm_and_rollback(tmp_path, recs, MT, "baseline")
        assert rep["rolled_back"] == []
        assert _status(tmp_path, f"{MT}/unlucky") == "active"

    def test_already_retired_variant_is_not_rolled_back_again(self, tmp_path):
        """REGRESSION GUARD. A retired variant stops being selected, but its old
        rolls stay in `records` forever, so it re-qualifies as a regression on
        every later reflection. `set_status` returns True whenever it FINDS the
        entry — not only when it changes it — so each pass appended another
        identical taboo to editor memory. The 2026-07-23 tools run recorded the
        same variant rolled back 5x, crowding out real memory."""
        _seed(tmp_path, [("baseline", MT, "active"), ("bad", MT, "active")])
        recs = _recs("bad", "baseline", ["t1", "t2", "t3", "t4"], 0.0, 1.0)

        first = C.confirm_and_rollback(tmp_path, recs, MT, "baseline")
        assert first["rolled_back"] == ["bad"]

        # same records, next reflection — must be a no-op now
        second = C.confirm_and_rollback(tmp_path, recs, MT, "baseline")
        assert second["rolled_back"] == []
        assert _status(tmp_path, f"{MT}/bad") == "superseded"

    def test_unreadable_archive_does_not_crash(self, tmp_path):
        """Best-effort by contract: confirm must never kill the run."""
        (tmp_path / "archive.json").write_text("{ not json")
        rep = C.confirm_and_rollback(
            tmp_path,
            _recs("bad", "baseline", ["t1", "t2", "t3", "t4"], 0.0, 1.0),
            MT,
            "baseline",
        )
        assert rep["rolled_back"] == []


def _hrecs(tasks, *, with_helper, without_helper):
    """Per task: one roll WITH the helper in hand and one WITHOUT."""
    out = []
    for t in tasks:
        out.append(_Rec(t, {MT: "baseline", C.HELPER_TYPE: ["h"]}, with_helper))
        out.append(_Rec(t, {MT: "baseline", C.HELPER_TYPE: []}, without_helper))
    return out


@pytest.mark.unit
class TestHelperRollback:
    def test_helper_that_hurts_is_superseded(self, tmp_path):
        _seed(tmp_path, [("h", C.HELPER_TYPE, "active")])
        rep = C.confirm_and_rollback_helpers(
            tmp_path,
            _hrecs(["t1", "t2", "t3", "t4"], with_helper=0.0, without_helper=1.0),
        )
        assert rep["rolled_back"] == ["h"]
        assert _status(tmp_path, f"{C.HELPER_TYPE}/h") == "superseded"

    def test_helper_that_helps_survives(self, tmp_path):
        _seed(tmp_path, [("h", C.HELPER_TYPE, "active")])
        rep = C.confirm_and_rollback_helpers(
            tmp_path,
            _hrecs(["t1", "t2", "t3", "t4"], with_helper=1.0, without_helper=0.0),
        )
        assert rep["rolled_back"] == []
        assert _status(tmp_path, f"{C.HELPER_TYPE}/h") == "active"

    def test_pre_feature_rolls_are_not_evidence(self, tmp_path):
        """Rolls recorded before helpers existed carry no `tool_helper` key at all.
        Reading them as 'helper absent' would manufacture a control group out of
        runs that predate the helper — they must be skipped outright."""
        _seed(tmp_path, [("h", C.HELPER_TYPE, "active")])
        recs = [_Rec(f"t{i}", {MT: "baseline"}, 1.0) for i in range(6)]
        recs += [
            _Rec(f"t{i}", {MT: "baseline", C.HELPER_TYPE: ["h"]}, 0.0) for i in range(6)
        ]
        rep = C.confirm_and_rollback_helpers(tmp_path, recs)
        assert rep["rolled_back"] == []

    def test_already_retired_helper_is_not_rolled_back_again(self, tmp_path):
        _seed(tmp_path, [("h", C.HELPER_TYPE, "active")])
        recs = _hrecs(["t1", "t2", "t3", "t4"], with_helper=0.0, without_helper=1.0)
        assert C.confirm_and_rollback_helpers(tmp_path, recs)["rolled_back"] == ["h"]
        assert C.confirm_and_rollback_helpers(tmp_path, recs)["rolled_back"] == []


@pytest.mark.unit
def test_raw_upsert_clobbers_status_so_refreshers_must_guard(tmp_path):
    """`update_archive` is a RAW upsert: it overwrites status with whatever the
    incoming entry carries (default `active`). So any code that refreshes entries
    after a promotion MUST skip superseded ones, or every rollback silently
    un-does itself on the next promote.

    That guard lives in `_sync_archive_after_promote` (online_evo.py):
        if e.status != "superseded":
            e.status = "active" if e.solver_selectable else "excluded"
    This test pins the sharp edge that makes the guard necessary — if a future
    change makes `update_archive` preserve status instead, the guard is redundant
    and this test should be revisited rather than deleted silently."""
    _seed(tmp_path, [("baseline", MT, "active"), ("bad", MT, "active")])
    C.confirm_and_rollback(
        tmp_path,
        _recs("bad", "baseline", ["t1", "t2", "t3", "t4"], 0.0, 1.0),
        MT,
        "baseline",
    )
    assert _status(tmp_path, f"{MT}/bad") == "superseded"

    arch.update_archive(
        tmp_path,
        [
            arch.ArchiveEntry(
                name="bad", type=MT, niche={"axis": "bad"}, born_gen="gen_1"
            )
        ],
    )
    assert _status(tmp_path, f"{MT}/bad") == "active", (
        "raw upsert is expected to clobber — the durability guard belongs to the "
        "caller (_sync_archive_after_promote), not to update_archive"
    )


@pytest.mark.unit
def test_archive_json_is_where_the_composer_looks(tmp_path):
    """The composer resolves the archive at `<modules_root>/../..`; a rollback
    written anywhere else is invisible to it."""
    _seed(tmp_path, [("bad", MT, "active")])
    assert (tmp_path / "archive.json").is_file()
    assert json.loads((tmp_path / "archive.json").read_text())


# --- recorded helper set must match what the agent ACTUALLY had ---------------


@pytest.mark.unit
def test_confirm_is_blind_when_the_recorded_set_lies():
    """Why orchestration re-records the helper set AFTER instantiating tools.

    A tools variant may force-add helpers in __init__ (observed: an
    `always_helpers` variant that injects four file helpers "regardless of what
    the Composer selects"). If the bundle records the Composer's REQUEST instead,
    rolls where the agent really had the helper land in the "absent" group — both
    sides get contaminated and the delta collapses toward zero, hiding a helper
    that is genuinely helping or hurting.
    """
    # Each task: TWO rolls with the helper in hand (both pass) and one without
    # (fails) — honest accounting makes that a clean +1.0.
    # Under the lie, ONE of the two in-hand rolls is recorded as absent. The task
    # still has a present side (so it stays a shared task), but its "absent" side
    # is now polluted with a roll that actually HAD the helper.
    honest, lying = [], []
    for task in ["t1", "t2", "t3", "t4", "t5", "t6"]:
        for recs, mislabel in ((honest, ["h"]), (lying, [])):
            recs.append(_Rec(task, {MT: "baseline", C.HELPER_TYPE: ["h"]}, 1.0))
            recs.append(_Rec(task, {MT: "baseline", C.HELPER_TYPE: mislabel}, 1.0))
            recs.append(_Rec(task, {MT: "baseline", C.HELPER_TYPE: []}, 0.0))

    honest_d = C.helper_paired_delta(
        [(r.task_name, r.summary.bundle, r.reward) for r in honest], "h"
    )
    lying_d = C.helper_paired_delta(
        [(r.task_name, r.summary.bundle, r.reward) for r in lying], "h"
    )
    assert honest_d["delta"] == pytest.approx(1.0), (
        "clean accounting sees the full effect"
    )
    assert lying_d["delta"] < honest_d["delta"], (
        "a lying record must measurably wash the signal out — this is the failure "
        "mode the post-instantiation re-record prevents"
    )
