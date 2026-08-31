"""P2: the assembly that plugs the real editor and gates into the two-lane loop.

`_two_lane_reflect` cannot be exercised in a unit test — it needs an LLM and a
container. What CAN be checked, and is worth checking precisely because nothing
else here can be, is that its wiring is internally consistent: it forwards
exactly what the gate battery accepts, no more and no less. A renamed gate
parameter would otherwise surface as a TypeError an hour into a real run.
"""

import inspect

import pytest

from harbor.agents.terminus_2_modular.self_evo import online_evo

pytestmark = pytest.mark.unit

#: What the assembly supplies itself, per candidate. Everything else has to
#: arrive from the caller through **gate_kwargs.
_FORWARDED_EXPLICITLY = {
    "editor_outcome",
    "staging_root",
    "staging_modules",
    "diff_base",
    "_changed",
    "candidate_n",
    "step_id",
    "archive_root",
    "current_gen",
}


def _gate_params():
    sig = inspect.signature(online_evo._gate_and_promote)
    return set(sig.parameters)


def test_the_assembly_forwards_nothing_the_gate_battery_would_reject():
    assert _FORWARDED_EXPLICITLY <= _gate_params()


def test_the_caller_must_supply_exactly_these_and_nothing_else_is_required():
    # everything else reaching the gate battery has a default, so a caller that
    # omits it still gets the legacy value. These two do NOT — they would be a
    # TypeError an hour into a real run. Pinning the set means that adding a
    # required gate parameter fails here instead of there.
    sig = inspect.signature(online_evo._gate_and_promote)
    required = {
        name
        for name in _gate_params() - _FORWARDED_EXPLICITLY
        if sig.parameters[name].default is inspect.Parameter.empty
    }
    assert required == {
        "api_base",
        "api_key",
        "max_editor_turns",
        "model_name",
        "pending_summaries",
        "skills_dir",
    }


# ---- when the re-gate pays for a second review --------------------------


def test_an_invalidated_review_is_re_run():
    assert online_evo._review_gate_for(("smoke", "review", "activation")) is True


def test_a_carried_over_review_is_not_paid_for_twice():
    # review is the most expensive gate and the biggest source of false
    # rejects; re-running it when the change did not move is buying another
    # chance to kill a good change
    assert online_evo._review_gate_for(("smoke", "activation", "routing")) is False


def test_no_gates_at_all_means_no_review():
    assert online_evo._review_gate_for(()) is False


# ---- the archive has to hear about two-lane promotions --------------------


def test_the_two_lane_window_is_given_an_archive_sync_hook():
    """`atomic_promote` writes files; only this hook writes archive.json.

    Run `…__p2_twolane_tools` promoted five generations and archive.json never
    moved off its gen_0 seeding, because the two-lane path never called the
    sync the legacy path calls. The router's candidate list is built from that
    file, so the run went blind to its own variants.
    """
    source = inspect.getsource(online_evo._two_lane_reflect)
    assert "on_promote=" in source


def test_the_hook_measures_a_promotion_against_the_parent_it_is_given(
    monkeypatch, tmp_path
):
    """Not against `current_gen` — a rebased lane sits on the lane before it."""
    seen = {}

    def fake_sync(archive_root, promoted, parent_modules, editor_text):
        seen.update(
            archive_root=archive_root,
            promoted=promoted,
            parent_modules=parent_modules,
            editor_text=editor_text,
        )

    monkeypatch.setattr(online_evo, "_sync_archive_after_promote", fake_sync)

    class _Detail:
        variant_meta_text = "<variant_meta>name: pager</variant_meta>"

    class _Result:
        detail = _Detail()

    online_evo._archive_sync_hook(tmp_path)(
        tmp_path / "gen_2", tmp_path / "gen_1", _Result()
    )

    assert seen["promoted"] == tmp_path / "gen_2"
    assert seen["parent_modules"] == tmp_path / "gen_1" / "modules"
    assert seen["editor_text"] == "<variant_meta>name: pager</variant_meta>"


def test_a_lane_with_no_variant_meta_still_syncs(monkeypatch, tmp_path):
    """An editor that wrote no <variant_meta> still produced a generation."""
    calls = []
    monkeypatch.setattr(
        online_evo,
        "_sync_archive_after_promote",
        lambda *a: calls.append(a),
    )
    online_evo._archive_sync_hook(tmp_path)(
        tmp_path / "gen_1", tmp_path / "gen_0", None
    )
    assert len(calls) == 1
    assert calls[0][3] == ""


# ---- a rebased lane keeps the editor session behind its change ------------


def test_a_rebased_lane_reuses_the_first_passs_trajectory():
    """`_trajectory_path` was a name nothing wrote — the read was always None.

    A rebase runs no new editor on purpose: `intent` and `<variant_meta>` still
    describe this change, so they come from the original session. Reading a
    nonexistent attribute silently replaced that session with nothing, and the
    re-gate then derived the generation's intent and genealogy from an empty
    trajectory.
    """
    from pathlib import Path as _P

    class _Prior:
        editor_n_edits = 3
        editor_trajectory_path = _P("/tmp/first-pass/trajectory.json")

    rebased = online_evo._RebasedEditorOutcome(_P("/tmp/dest"), _Prior())

    assert rebased.trajectory_path == _P("/tmp/first-pass/trajectory.json")
    assert rebased.n_edits == 3


def test_the_gate_battery_output_carries_the_trajectory_it_read():
    """The producer side: without this field the reuse above has no source."""
    assert (
        "editor_trajectory_path" in online_evo._ReflectionOutcome.__dataclass_fields__
    )


# ---- P2-3: the window's accounting reaches the router's ledger -----------


def test_the_two_lane_window_is_given_the_task_accounting_hook():
    # without it `LaneSettlement` is computed and dropped, and the exhaustion
    # signal stays dead in exactly the path that replaced the deleted one
    source = inspect.getsource(online_evo._two_lane_reflect)
    assert "on_lane_settled=" in source


def test_a_change_that_never_ran_charges_nothing(tmp_path):
    from harbor.agents.terminus_2_modular.self_evo import two_lane
    from harbor.agents.terminus_2_modular.self_evo.router import Ledger

    settle = online_evo._task_accounting_hook(tmp_path)
    settle(
        two_lane.LaneSettlement(
            proposal_id="p_0001",
            lane="incumbent",
            support_tasks=["fix-git"],
            made_progress=False,
            ran=False,
        )
    )
    assert Ledger.load(tmp_path).get("fix-git").times_reflected == 0


def test_a_change_that_ran_and_lost_is_charged_a_no_progress_reflection(tmp_path):
    from harbor.agents.terminus_2_modular.self_evo import two_lane
    from harbor.agents.terminus_2_modular.self_evo.router import Ledger

    settle = online_evo._task_accounting_hook(tmp_path)
    settle(
        two_lane.LaneSettlement(
            proposal_id="p_0001",
            lane="incumbent",
            support_tasks=["fix-git"],
            made_progress=False,
            ran=True,
        )
    )
    entry = Ledger.load(tmp_path).get("fix-git")
    assert entry.times_reflected == 1
    assert entry.reflections_without_progress == 1


def test_a_promoted_change_clears_the_no_progress_streak(tmp_path):
    from harbor.agents.terminus_2_modular.self_evo import two_lane
    from harbor.agents.terminus_2_modular.self_evo.router import Ledger

    settle = online_evo._task_accounting_hook(tmp_path)
    for made_progress in (False, True):
        settle(
            two_lane.LaneSettlement(
                proposal_id="p_0001",
                lane="incumbent",
                support_tasks=["fix-git"],
                made_progress=made_progress,
                ran=True,
            )
        )
    entry = Ledger.load(tmp_path).get("fix-git")
    assert entry.times_reflected == 2
    assert entry.reflections_without_progress == 0


# ---- C3 / C1: the lane tells the gate what it was routed to build --------


def test_the_lane_hands_its_proposals_action_to_the_gate_battery():
    # the action lives on the proposal, and only the lane knows which proposal
    # it is building — the battery cannot look it up
    source = inspect.getsource(online_evo._two_lane_reflect)
    assert "proposal_action=" in source
    assert "proposal_target=" in source


def test_the_gate_battery_accepts_the_declared_action():
    params = inspect.signature(online_evo._gate_and_promote).parameters
    assert params["proposal_action"].default == ""
    assert params["proposal_target"].default == ""


def test_what_a_lane_actually_built_is_recorded_against_its_proposal(tmp_path):
    # C1's other half: the proposal says what was intended, the manifest says
    # what was produced. Without it nothing can check one against the other
    # after the staging is gone.
    from harbor.agents.terminus_2_modular.self_evo import manifest

    class _Outcome:
        files_changed = ["agent_loop/deadline_aware.py"]
        variant_meta_text = "<variant_meta name='deadline_aware' type='agent_loop'>\nSUPERSEDES: confirm_exit\n</variant_meta>"

    online_evo._record_lane_implementation(
        tmp_path, "p_0001", step="w2", outcome=_Outcome()
    )
    rec = manifest.get_implementation(tmp_path, "p_0001")
    assert rec.files == ["agent_loop/deadline_aware.py"]
    assert rec.supersede_targets == ["confirm_exit"]


def test_recording_what_was_built_never_costs_the_lane(tmp_path):
    # bookkeeping does not get to fail a change that passed its gates
    online_evo._record_lane_implementation(tmp_path, "", step="w2", outcome=None)


def test_the_lane_records_what_it_built():
    source = inspect.getsource(online_evo._two_lane_reflect)
    assert "_record_lane_implementation(" in source
