"""P2: one reflection window may now produce TWO generations, not one.

Everything downstream of a reflection was written when a window could promote at
most once: `_ReflectionOutcome.promoted_gen` is a single Path, the run loop reads
it to decide which tree the solver continues on, and `_maybe_reflect` appends
exactly one evolution-log record. Two lanes break all three assumptions, and the
ways they break are not symmetric:

* The run loop must land on the **newest** generation. Landing on the first
  would keep the solver running a tree that is missing the second lane's change
  — promoted into the archive, never actually used, which is the exact
  "promoted but nothing selects it" failure this phase exists to close.
* The evolution log must get **one record per generation**. Each generation is
  still a single change, so merging two lanes into one record would throw away
  the only per-generation attribution there is — and the editor reads this log
  to avoid re-trying what was already tried.
* A per-lane gate verdict must not be reported against the other lane's
  generation. Two lanes run their own review and their own sanity; copying one
  verdict onto both records would record a judgement that was never made.

The singular path must come through unchanged: one promotion still means one
record, and no promotion still means one `(discarded)` record.
"""

import json

import pytest

from harbor.agents.terminus_2_modular.self_evo import online_evo

pytestmark = pytest.mark.unit


def _outcome(**kwargs):
    base = dict(triggered=True, promoted_gen=None, discard_reason=None)
    base.update(kwargs)
    return online_evo._ReflectionOutcome(**base)


def _log_lines(archive_root):
    path = archive_root / "evolution_log.jsonl"
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


# ---- the outcome carries a list now ---------------------------------------


def test_a_window_that_promoted_nothing_records_nothing(tmp_path):
    out = _outcome()
    assert out.promotions == []
    assert out.promoted_gen is None


def test_recording_a_promotion_updates_the_single_gen_view(tmp_path):
    # the run loop reads promoted_gen; the singular path must not notice a change
    out = _outcome()
    out.record_promotion(tmp_path / "gen_7")

    assert out.promoted_gen == tmp_path / "gen_7"
    assert len(out.promotions) == 1


def test_a_second_promotion_moves_the_single_gen_view_forward(tmp_path):
    # must be the NEWEST: staying on gen_7 would keep the solver on a tree that
    # is missing the second lane's change, which is already in the archive
    out = _outcome()
    out.record_promotion(tmp_path / "gen_7")
    out.record_promotion(tmp_path / "gen_8")

    assert out.promoted_gen == tmp_path / "gen_8"
    assert [p.gen.name for p in out.promotions] == ["gen_7", "gen_8"]


def test_each_promotion_keeps_its_own_lane_and_proposal(tmp_path):
    out = _outcome()
    out.record_promotion(tmp_path / "gen_7", proposal_id="p_0007", lane="incumbent")
    out.record_promotion(tmp_path / "gen_8", proposal_id="p_0009", lane="novelty")

    assert [(p.lane, p.proposal_id) for p in out.promotions] == [
        ("incumbent", "p_0007"),
        ("novelty", "p_0009"),
    ]


def test_a_promotion_inherits_the_windows_intent_when_it_has_none(tmp_path):
    # the singular path sets intent/files on the outcome and calls
    # record_promotion with nothing else; it must keep working untouched
    out = _outcome(intent="tighten the exit check", files_changed=["a.py"])
    out.record_promotion(tmp_path / "gen_7")

    assert out.promotions[0].intent == "tighten the exit check"
    assert out.promotions[0].files_changed == ["a.py"]


def test_a_promotion_can_carry_its_own_intent(tmp_path):
    out = _outcome(intent="window level", files_changed=["a.py"])
    out.record_promotion(
        tmp_path / "gen_8", intent="lane level", files_changed=["b.py"]
    )

    assert out.promotions[0].intent == "lane level"
    assert out.promotions[0].files_changed == ["b.py"]


# ---- the evolution log gets one record per generation ---------------------


async def test_a_discarded_window_still_writes_exactly_one_record(
    tmp_path, monkeypatch
):
    async def fake_inner(**kwargs):
        return _outcome(
            discard_reason="review gate (reject): overfit", candidate_gen_n=3
        )

    monkeypatch.setattr(online_evo, "_maybe_reflect_inner", fake_inner)
    await online_evo._maybe_reflect(archive_root=tmp_path)

    (record,) = _log_lines(tmp_path)
    assert record["gen"] == "gen_3(discarded)"
    assert record["promoted"] is False


async def test_an_untriggered_window_writes_nothing(tmp_path, monkeypatch):
    async def fake_inner(**kwargs):
        return online_evo._ReflectionOutcome(
            triggered=False, promoted_gen=None, discard_reason=None
        )

    monkeypatch.setattr(online_evo, "_maybe_reflect_inner", fake_inner)
    await online_evo._maybe_reflect(archive_root=tmp_path)

    assert _log_lines(tmp_path) == []


async def test_a_single_promotion_still_writes_exactly_one_record(
    tmp_path, monkeypatch
):
    async def fake_inner(**kwargs):
        out = _outcome(intent="tighten exit", files_changed=["agent_loop/x.py"])
        out.record_promotion(tmp_path / "gen_7")
        return out

    monkeypatch.setattr(online_evo, "_maybe_reflect_inner", fake_inner)
    await online_evo._maybe_reflect(archive_root=tmp_path)

    (record,) = _log_lines(tmp_path)
    assert record["gen"] == "gen_7"
    assert record["promoted"] is True
    assert record["intent"] == "tighten exit"


async def test_two_promotions_write_one_record_each(tmp_path, monkeypatch):
    async def fake_inner(**kwargs):
        out = _outcome()
        out.record_promotion(
            tmp_path / "gen_7",
            proposal_id="p_0007",
            lane="incumbent",
            intent="tighten the exit check",
            files_changed=["agent_loop/confirm_exit.py"],
        )
        out.record_promotion(
            tmp_path / "gen_8",
            proposal_id="p_0009",
            lane="novelty",
            intent="add a pager escape hatch",
            files_changed=["observation/pager.py"],
        )
        return out

    monkeypatch.setattr(online_evo, "_maybe_reflect_inner", fake_inner)
    await online_evo._maybe_reflect(archive_root=tmp_path)

    records = _log_lines(tmp_path)
    assert [r["gen"] for r in records] == ["gen_7", "gen_8"]
    assert [r["intent"] for r in records] == [
        "tighten the exit check",
        "add a pager escape hatch",
    ]
    assert [r["files"] for r in records] == [
        ["agent_loop/confirm_exit.py"],
        ["observation/pager.py"],
    ]
    assert all(r["promoted"] for r in records)


async def test_each_record_says_which_proposal_and_lane_it_came_from(
    tmp_path, monkeypatch
):
    # without this the archive cannot be joined back to the backlog, and
    # "did this direction ever get built" becomes unanswerable
    async def fake_inner(**kwargs):
        out = _outcome()
        out.record_promotion(tmp_path / "gen_7", proposal_id="p_0007", lane="incumbent")
        out.record_promotion(tmp_path / "gen_8", proposal_id="p_0009", lane="novelty")
        return out

    monkeypatch.setattr(online_evo, "_maybe_reflect_inner", fake_inner)
    await online_evo._maybe_reflect(archive_root=tmp_path)

    records = _log_lines(tmp_path)
    assert [r["proposal_id"] for r in records] == ["p_0007", "p_0009"]
    assert [r["lane"] for r in records] == ["incumbent", "novelty"]


async def test_a_per_lane_gate_verdict_is_not_copied_onto_the_other_lane(
    tmp_path, monkeypatch
):
    # two lanes run their own review; reporting one verdict against both gens
    # would record a judgement that was never made about that change
    async def fake_inner(**kwargs):
        out = _outcome(review_verdict="accept", sanity_passed=True)
        out.record_promotion(tmp_path / "gen_7", gates={"review": "accept"})
        out.record_promotion(
            tmp_path / "gen_8", gates={"review": "accept_partial", "sanity": False}
        )
        return out

    monkeypatch.setattr(online_evo, "_maybe_reflect_inner", fake_inner)
    await online_evo._maybe_reflect(archive_root=tmp_path)

    first, second = _log_lines(tmp_path)
    assert first["review"] == "accept"
    assert second["review"] == "accept_partial"
    assert second["sanity"] is False


# ---- the run loop's counters and memory ----------------------------------


def test_a_discarded_window_counts_as_no_promotion(tmp_path):
    assert online_evo._promotion_count(_outcome()) == 0


def test_a_two_lane_window_counts_as_two_promotions(tmp_path):
    # meta.json reports n_promotions; counting the window instead of the
    # generations would under-report the lineage by half
    out = _outcome()
    out.record_promotion(tmp_path / "gen_7")
    out.record_promotion(tmp_path / "gen_8")
    assert online_evo._promotion_count(out) == 2


def test_a_promotion_recorded_the_old_way_still_counts_as_one(tmp_path):
    # defensive: promoted_gen set without going through record_promotion
    out = _outcome()
    out.promoted_gen = tmp_path / "gen_7"
    assert online_evo._promotion_count(out) == 1


def test_every_promoted_generation_reaches_the_editor_memory(tmp_path):
    # the consolidator reads this to avoid re-trying what was tried; recording
    # only one of two generations makes that memory half-blind
    from harbor.agents.terminus_2_modular.self_evo import editor_memory

    out = _outcome()
    out.record_promotion(tmp_path / "gen_7", intent="tighten the exit check")
    out.record_promotion(tmp_path / "gen_8", intent="add a pager escape hatch")

    online_evo._remember_promotions(tmp_path, out, epoch=2, module="agent_loop")

    rows = editor_memory.load(tmp_path)
    assert [r["gen"] for r in rows] == ["gen_7", "gen_8"]
    assert [r["change"] for r in rows] == [
        "tighten the exit check",
        "add a pager escape hatch",
    ]
    assert all(r["verdict"] == "provisional" for r in rows)


def test_a_window_that_promoted_nothing_writes_no_memory(tmp_path):
    from harbor.agents.terminus_2_modular.self_evo import editor_memory

    online_evo._remember_promotions(tmp_path, _outcome(), epoch=2, module="agent_loop")
    assert editor_memory.load(tmp_path) == []


def test_progress_json_survives_a_two_generation_window(tmp_path):
    # `_save_progress` swallows its own exceptions and only logs a warning, so
    # a serialisation break here is silent — and progress.json is the per-task
    # journal the reflection decisions are read back from
    out = _outcome()
    out.record_promotion(tmp_path / "gen_7", proposal_id="p_0007", lane="incumbent")
    out.record_promotion(tmp_path / "gen_8", proposal_id="p_0009", lane="novelty")
    record = online_evo.TaskRunRecord(
        task_idx=0,
        task_name="fix-git",
        gen_used=tmp_path / "gen_6",
        reward=0.0,
        error=None,
        trial_dir=None,
        summary=None,
        reflection=out,
    )

    online_evo._save_progress(tmp_path, [record])

    written = json.loads((tmp_path / "progress.json").read_text())
    promotions = written[0]["reflection"]["promotions"]
    assert [p["lane"] for p in promotions] == ["incumbent", "novelty"]
    assert promotions[1]["gen"].endswith("gen_8")


async def test_a_promotion_without_its_own_verdicts_reports_the_windows(
    tmp_path, monkeypatch
):
    async def fake_inner(**kwargs):
        out = _outcome(review_verdict="accept", sanity_passed=True)
        out.record_promotion(tmp_path / "gen_7")
        return out

    monkeypatch.setattr(online_evo, "_maybe_reflect_inner", fake_inner)
    await online_evo._maybe_reflect(archive_root=tmp_path)

    (record,) = _log_lines(tmp_path)
    assert record["review"] == "accept"
    assert record["sanity"] is True
