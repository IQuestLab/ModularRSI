"""Unit tests for the self-evo router (bucketing + ledger). Pure, no endpoint."""

import pytest

from harbor.agents.terminus_2_modular.self_evo.router import (
    Bucket,
    Ledger,
    LedgerEntry,
    RouterConfig,
    TaskOutcome,
    classify,
    route_batch,
)
from harbor.agents.terminus_2_modular.self_evo.trajectory_analysis import TrialSummary

pytestmark = pytest.mark.unit


def roll(task, reward, *, episodes=5, repeat=0.0, trial="t"):
    return TrialSummary(
        task_name=task,
        trial_name=trial,
        reward=reward,
        exception_type=None,
        exception_message=None,
        n_episodes=episodes,
        n_input_tokens=0,
        n_output_tokens=0,
        repeated_command_ratio=repeat,
    )


CFG = RouterConfig(k=3, stuck_n=3, cooldown_m=2, wasteful_episodes=30)


def _classify(rolls, entry=None, epoch=0):
    o = TaskOutcome(task="x", rolls=rolls)
    return classify(o, entry or LedgerEntry(task="x"), epoch, CFG)


def test_all_pass_efficient():
    assert _classify([roll("x", 1.0), roll("x", 1.0), roll("x", 1.0)]) == (
        Bucket.ALL_PASS_EFFICIENT
    )


def test_all_pass_wasteful_by_turns():
    b = _classify([roll("x", 1.0, episodes=40)] * 3)
    assert b == Bucket.ALL_PASS_WASTEFUL


def test_all_pass_wasteful_by_repeat():
    b = _classify([roll("x", 1.0, repeat=0.9)] * 3)
    assert b == Bucket.ALL_PASS_WASTEFUL


def test_mixed():
    assert _classify([roll("x", 1.0), roll("x", 0.0), roll("x", 0.0)]) == Bucket.MIXED


def test_all_fail_never_tried_is_fixable():
    assert _classify([roll("x", 0.0)] * 3) == Bucket.FIXABLE_FAIL


def test_all_fail_ever_passed_is_fixable():
    e = LedgerEntry(task="x", ever_passed=True)
    assert _classify([roll("x", 0.0)] * 3, e) == Bucket.FIXABLE_FAIL


def test_all_fail_tried_out_enters_stuck_not_unreachable():
    # P0-3a: exhaustion is a COOLDOWN (recoverable STUCK), not a permanent
    # blacklist. The old UNREACHABLE_FAIL had no exit: start_cooldown had no
    # caller, and even with one, expiry left the counters over threshold so the
    # task bounced straight back — 13/120 tasks were permanently blacklisted.
    e = LedgerEntry(
        task="x", ever_passed=False, times_reflected=3, reflections_without_progress=3
    )
    assert _classify([roll("x", 0.0)] * 3, e) == Bucket.STUCK_FAIL


def test_all_fail_in_cooldown_is_stuck():
    e = LedgerEntry(task="x", cooldown_until_epoch=5)
    assert _classify([roll("x", 0.0)] * 3, e, epoch=2) == Bucket.STUCK_FAIL


def test_reward_none_is_infra_ignored_not_fail():
    # None = infra error, dropped from the spread: [None,1,1] is all-pass, NOT mixed
    assert _classify([roll("x", None), roll("x", 1.0), roll("x", 1.0)]) == (
        Bucket.ALL_PASS_EFFICIENT
    )
    # a REAL fail (0.0) alongside a pass IS mixed (None still ignored)
    assert _classify([roll("x", None), roll("x", 0.0), roll("x", 1.0)]) == Bucket.MIXED


def test_all_infra_none_is_skipped_by_route_batch():
    summaries = [roll("dead", None), roll("dead", None), roll("dead", None)]
    res = route_batch(summaries, ledger=Ledger(), epoch=0, cfg=CFG)
    assert res.diagnose == [] and res.anchors == []
    assert res.buckets["infra_only"] == ["dead"]


def test_route_batch_pairs_and_anchors():
    summaries = (
        [roll("mix", 1.0, trial="p"), roll("mix", 0.0, trial="f"), roll("mix", 0.0)]
        + [roll("anchor", 1.0)] * 3
        + [roll("fail", 0.0)] * 3
    )
    led = Ledger()
    res = route_batch(summaries, ledger=led, epoch=0, cfg=CFG)
    assert res.buckets[Bucket.MIXED.value] == ["mix"]
    assert res.anchors == ["anchor"]
    assert res.buckets[Bucket.FIXABLE_FAIL.value] == ["fail"]
    # mixed task got a same-task pass/fail pair
    mix_item = next(d for d in res.diagnose if d.task == "mix")
    assert mix_item.source == "same_task"
    assert mix_item.contrast_pass is not None and mix_item.contrast_fail is not None
    assert mix_item.contrast_pass.reward == 1.0 and mix_item.contrast_fail.reward == 0.0
    # fixable-fail with no history → non-contrastive
    fail_item = next(d for d in res.diagnose if d.task == "fail")
    assert fail_item.source == "none" and fail_item.contrast_pass is None


def test_route_batch_history_lookup_gives_contrast():
    summaries = [roll("fail", 0.0)] * 3
    hist = roll("fail", 1.0, trial="historical_pass")
    res = route_batch(
        summaries,
        ledger=Ledger(),
        epoch=0,
        cfg=CFG,
        history_pass_lookup=lambda t: hist if t == "fail" else None,
    )
    item = res.diagnose[0]
    assert item.source == "archive_history"
    assert item.contrast_pass is hist


def test_ledger_roundtrip_and_cooldown(tmp_path):
    led = Ledger()
    led.record_reflection("t", made_progress=False)
    led.record_reflection("t", made_progress=False)
    led.start_cooldown("t", epoch=1, cfg=CFG)
    led.save(tmp_path)
    led2 = Ledger.load(tmp_path)
    e = led2.get("t")
    assert e.times_reflected == 2
    assert e.reflections_without_progress == 2
    assert e.cooldown_until_epoch == 3  # 1 + cooldown_m(2)


def test_ledger_records_ever_passed():
    led = Ledger()
    o = TaskOutcome(task="t", rolls=[roll("t", 0.0), roll("t", 1.0), roll("t", 0.0)])
    led.record_encounter(o, Bucket.MIXED, epoch=0)
    assert led.get("t").ever_passed is True


def test_exhaustion_starts_cooldown_then_recovers_to_fixable():
    """P0-3a acceptance: an exhausted never-passed task cools down for
    cooldown_m epochs, then gets a FRESH attempt window and re-enters
    FIXABLE_FAIL (the old code bounced it straight back to UNREACHABLE)."""
    led = Ledger()
    for _ in range(3):
        led.record_reflection("hard", made_progress=False)

    def _route(epoch):
        return route_batch([roll("hard", 0.0)] * 3, ledger=led, epoch=epoch, cfg=CFG)

    # epoch 5: exhausted → STUCK, and the cooldown clock starts NOW
    res = _route(5)
    assert res.buckets[Bucket.STUCK_FAIL.value] == ["hard"]
    assert res.diagnose == []  # no budget burned while stuck
    assert led.get("hard").cooldown_until_epoch == 7  # 5 + cooldown_m(2)

    # epoch 6: still cooling
    res = _route(6)
    assert res.buckets[Bucket.STUCK_FAIL.value] == ["hard"]

    # epoch 7: cooldown expired → fresh window → FIXABLE_FAIL again
    res = _route(7)
    assert res.buckets[Bucket.FIXABLE_FAIL.value] == ["hard"]
    e = led.get("hard")
    assert e.reflections_without_progress == 0  # window renewed
    assert e.cooldown_cycles == 1
    assert e.times_reflected == 3  # lifetime history is kept


def test_renewed_window_can_exhaust_into_a_second_cycle():
    led = Ledger()
    for _ in range(3):
        led.record_reflection("hard", made_progress=False)
    route_batch([roll("hard", 0.0)] * 3, ledger=led, epoch=0, cfg=CFG)  # → stuck
    route_batch([roll("hard", 0.0)] * 3, ledger=led, epoch=2, cfg=CFG)  # renewed
    for _ in range(3):
        led.record_reflection("hard", made_progress=False)
    res = route_batch([roll("hard", 0.0)] * 3, ledger=led, epoch=3, cfg=CFG)
    assert res.buckets[Bucket.STUCK_FAIL.value] == ["hard"]
    assert led.get("hard").cooldown_cycles == 1  # second cycle's renewal not yet
    assert led.get("hard").cooldown_until_epoch == 5  # 3 + 2: clock restarted


def test_made_progress_resets_the_streak():
    led = Ledger()
    led.record_reflection("t", made_progress=False)
    led.record_reflection("t", made_progress=False)
    led.record_reflection("t", made_progress=True)
    e = led.get("t")
    assert e.reflections_without_progress == 0
    assert e.times_reflected == 3


def test_old_ledger_json_without_new_fields_still_loads(tmp_path):
    # live runs carry router_ledger.json written by the old code — the new
    # field(s) must default, not crash the resume path
    led = Ledger()
    led.record_reflection("t", made_progress=False)
    led.save(tmp_path)
    import json as _json

    p = tmp_path / Ledger.FILENAME
    raw = _json.loads(p.read_text())
    for entry in raw.values():
        entry.pop("cooldown_cycles", None)  # simulate an old-format file
    p.write_text(_json.dumps(raw))
    led2 = Ledger.load(tmp_path)
    assert led2.get("t").cooldown_cycles == 0


def test_route_batch_wasteful_goes_to_efficiency_and_anchors():
    # all-pass but wasteful (40 turns) → both an anchor AND an efficiency item
    summaries = [roll("slow", 1.0, episodes=40)] * 3 + [roll("fast", 1.0)] * 3
    res = route_batch(summaries, ledger=Ledger(), epoch=0, cfg=CFG)
    assert "slow" in res.anchors and "fast" in res.anchors
    assert [d.task for d in res.efficiency] == ["slow"]
    assert res.efficiency[0].source == "efficiency"
    assert res.efficiency[0].contrast_fail.reward == 1.0  # a passing roll to trim
    # efficient all-pass task does NOT enter efficiency
    assert res.diagnose == []


# ---- P0-3a: history-pass recording + lookup (driver-side helpers) ----------


def _passing_trial_dir(tmp_path, task="hist"):
    import json as _json

    trial = tmp_path / f"{task}__abc"
    trial.mkdir()
    (trial / "result.json").write_text(
        _json.dumps(
            {
                "task_name": task,
                "trial_name": f"{task}__abc",
                "verifier_result": {"rewards": {"reward": 1.0}},
            }
        )
    )
    return trial


def test_history_pass_recorded_then_used_as_contrast(tmp_path):
    """End-to-end (P0-3a): epoch 0 passes → recorded; epoch 1 all-fail →
    the router pairs the failure with the RECORDED pass, source=archive_history."""
    from harbor.agents.terminus_2_modular.self_evo import online_evo as OE

    trial = _passing_trial_dir(tmp_path, "hist")
    passing = roll("hist", 1.0)
    passing = type(passing)(**{**passing.__dict__, "trial_dir": str(trial)})
    OE._record_history_passes(tmp_path, [passing, roll("other", 0.0)])

    lookup = OE._make_history_pass_lookup(tmp_path)
    assert lookup("other") is None  # failing roll was NOT recorded
    res = route_batch(
        [roll("hist", 0.0)] * 3,
        ledger=Ledger(),
        epoch=1,
        cfg=CFG,
        history_pass_lookup=lookup,
    )
    item = res.diagnose[0]
    assert item.source == "archive_history"
    assert item.contrast_pass is not None and item.contrast_pass.reward == 1.0


def test_history_pass_lookup_degrades_when_trial_dir_gone(tmp_path):
    from harbor.agents.terminus_2_modular.self_evo import online_evo as OE

    trial = _passing_trial_dir(tmp_path, "hist")
    passing = roll("hist", 1.0)
    passing = type(passing)(**{**passing.__dict__, "trial_dir": str(trial)})
    OE._record_history_passes(tmp_path, [passing])
    import shutil

    shutil.rmtree(trial)  # quarantined/cleaned up later
    assert OE._make_history_pass_lookup(tmp_path)("hist") is None


def test_online_evo_wiring_contract():
    """P0-3a contract pin, as updated by P2-3 — deliberately, not silently.

    The rule was never "never charge a task"; it was "never charge a task
    nobody acted on". The blanket penalty over every *diagnosed* task counted
    "proposed something and lost the selection" as "we tried and failed", and
    that fed the exhaustion blacklist.

    So the reflection path itself still charges nothing — it is the one that
    cannot tell acted-on from diagnosed — and the only `record_reflection` in
    the module lives behind the window's selection-aware settlement, which knows
    which proposal reached an implementer and whether its variant ran.
    """
    import inspect

    from harbor.agents.terminus_2_modular.self_evo import online_evo as OE

    src = inspect.getsource(OE)
    assert "history_pass_lookup=" in src
    # the call form, not the words (comments may explain the deletion)
    reflection_path = inspect.getsource(OE._contrastive_investigate_and_consolidate)
    assert "ledger.record_reflection(" not in reflection_path
    # …and P2-3's replacement is present, so deleting it fails here rather than
    # silently leaving the exhaustion signal dead
    assert "ledger.record_reflection(" in inspect.getsource(OE._task_accounting_hook)
