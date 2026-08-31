"""C2b: the gate battery's idea of "the parent" must be the tree it diffs against.

`_gate_and_promote` receives `diff_base` — the tree this candidate's change is
measured against — and separately `current_gen`, the generation the window
started from. For a single-candidate window those are the same tree and nothing
distinguishes them. For the second lane of a two-lane window they are NOT: the
lane was rebased onto the generation the *first* lane just created, so
`diff_base` moved and `current_gen` did not.

The niche dedup gate asks "which variants are NEW in this tree?" by listing the
parent's library and subtracting. Handed the stale `current_gen`, it sees the
other lane's freshly promoted files as this lane's additions — and rejects the
lane for landing in a niche cell that its own sibling, not it, occupies.

So the parent the gates read is derived from `diff_base`, not from `current_gen`.
"""

import inspect
from pathlib import Path

import pytest

from harbor.agents.terminus_2_modular.self_evo import online_evo

pytestmark = pytest.mark.unit


class _PassingSmoke:
    passed = True

    def all_failures(self):
        return []


class _EditorOutcome:
    success = True
    committed = True
    error = None
    n_edits = 1
    trajectory_path = None


def _install_fakes(monkeypatch) -> dict:
    """Let the battery reach the niche gate, and stop it there.

    Everything after the niche gate needs an LLM (review) or a container
    (sanity), so the fake gate returns a rejection: the battery records the
    reason and returns, and the argument it was called with is the whole point
    of the test.
    """
    seen: dict = {}

    def _niche(archive_root, staging, parent, editor_text):
        seen["archive_root"] = archive_root
        seen["staging"] = staging
        seen["parent"] = parent
        return "niche dedup: stop here"

    monkeypatch.setattr(online_evo, "run_fast_smoke", lambda _p: _PassingSmoke())
    monkeypatch.setattr(online_evo, "discard_staging", lambda _root: None)
    monkeypatch.setattr(online_evo, "_niche_dedup_reject", _niche)
    return seen


async def _gate(*, diff_base: Path, current_gen: Path, staging_root: Path, **extra):
    return await online_evo._gate_and_promote(
        editor_outcome=_EditorOutcome(),
        staging_root=staging_root,
        staging_modules=staging_root,
        diff_base=diff_base,
        _changed=lambda _staging: [],
        candidate_n=7,
        step_id="gen_7_candidate_0",
        api_base=None,
        api_key=None,
        archive_root=staging_root.parent,
        current_gen=current_gen,
        max_editor_turns=1,
        model_name="m",
        pending_summaries=[],
        skills_dir=None,
        **extra,
    )


@pytest.mark.asyncio
async def test_a_rebased_lane_is_deduped_against_the_generation_it_sits_on(
    monkeypatch, tmp_path
):
    # the lane was rebased onto gen_8, which the other lane just created;
    # gen_7 is the window's parent and is now one generation stale
    seen = _install_fakes(monkeypatch)
    out = await _gate(
        diff_base=tmp_path / "gen_8" / "modules",
        current_gen=tmp_path / "gen_7",
        staging_root=tmp_path / "staging",
    )
    assert seen["parent"] == tmp_path / "gen_8" / "modules"
    assert out.discard_reason == "niche dedup: stop here"


@pytest.mark.asyncio
async def test_the_unrebased_case_is_unchanged(monkeypatch, tmp_path):
    # a single-candidate window diffs against its own parent; the derived path
    # has to come out byte-identical to the one that was hard-coded before
    seen = _install_fakes(monkeypatch)
    await _gate(
        diff_base=tmp_path / "gen_7" / "modules",
        current_gen=tmp_path / "gen_7",
        staging_root=tmp_path / "staging",
    )
    assert seen["parent"] == tmp_path / "gen_7" / "modules"


def test_the_parent_tree_is_derived_once():
    # a second assignment further down the battery re-derived the same path from
    # `current_gen` and silently shadowed the first for everything after it —
    # which is how a corrected parent goes stale again halfway through. One
    # binding, at the top, or the correction only holds for the gates above it.
    source = inspect.getsource(online_evo._gate_and_promote)
    assert source.count("    parent_modules = ") == 1


# ---- C3: the gate battery holds the change to its declared action --------
#
# The action reached the implementer as prose in a prompt and was never checked
# against what came back, so a `novelty`/`add` proposal could be delivered as an
# edit to the incumbent and every later gate would still pass. What the battery
# owes is narrow: hand the declared action to the check, and treat its verdict
# as a hard gate. The judgement itself is `action_gate`'s, tested on its own.


@pytest.mark.asyncio
async def test_the_declared_action_reaches_the_gate(monkeypatch, tmp_path):
    seen: dict = {}
    monkeypatch.setattr(online_evo, "run_fast_smoke", lambda _p: _PassingSmoke())
    monkeypatch.setattr(online_evo, "discard_staging", lambda _root: None)
    monkeypatch.setattr(online_evo, "_niche_dedup_reject", lambda *a, **k: "stop here")
    monkeypatch.setattr(
        online_evo, "_action_reject", lambda **k: seen.update(k) or None
    )
    await _gate(
        diff_base=tmp_path / "gen_7" / "modules",
        current_gen=tmp_path / "gen_7",
        staging_root=tmp_path / "staging",
        proposal_action="add",
        proposal_target="agent_loop/confirm_exit",
    )
    assert seen["action"] == "add"
    assert seen["target_variant"] == "agent_loop/confirm_exit"
    # and it is judged against the tree the change is diffed against, not the
    # window's parent — same rule as every other gate here
    assert seen["parent_modules"] == tmp_path / "gen_7" / "modules"


@pytest.mark.asyncio
async def test_a_change_that_ignored_its_action_is_discarded(monkeypatch, tmp_path):
    monkeypatch.setattr(online_evo, "run_fast_smoke", lambda _p: _PassingSmoke())
    monkeypatch.setattr(online_evo, "discard_staging", lambda _root: None)
    monkeypatch.setattr(online_evo, "_niche_dedup_reject", lambda *a, **k: None)
    monkeypatch.setattr(
        online_evo,
        "_action_reject",
        lambda **k: "action=add must not edit an incumbent",
    )
    out = await _gate(
        diff_base=tmp_path / "gen_7" / "modules",
        current_gen=tmp_path / "gen_7",
        staging_root=tmp_path / "staging",
        proposal_action="add",
        proposal_target="agent_loop/confirm_exit",
    )
    assert out.discard_reason == "action=add must not edit an incumbent"
    assert out.promoted_gen is None


@pytest.mark.asyncio
async def test_an_unrouted_candidate_never_reaches_the_action_gate(
    monkeypatch, tmp_path
):
    # the legacy single-candidate path has no proposal; paying for two library
    # loads to hold it to an action it was never given is waste
    called = []
    monkeypatch.setattr(online_evo, "run_fast_smoke", lambda _p: _PassingSmoke())
    monkeypatch.setattr(online_evo, "discard_staging", lambda _root: None)
    monkeypatch.setattr(online_evo, "_niche_dedup_reject", lambda *a, **k: "stop here")
    monkeypatch.setattr(
        online_evo, "_action_reject", lambda **k: called.append(k) or None
    )
    await _gate(
        diff_base=tmp_path / "gen_7" / "modules",
        current_gen=tmp_path / "gen_7",
        staging_root=tmp_path / "staging",
    )
    assert called == []


def test_a_gate_that_cannot_run_does_not_cost_a_generation(tmp_path):
    # both library loads are real filesystem work; a gate that rejects on its
    # own infra trouble kills a change for a reason unrelated to the change
    assert (
        online_evo._action_reject(
            action="add",
            target_variant="",
            files_changed=["agent_loop/x.py"],
            variant_meta_text="",
            parent_modules=tmp_path / "nope",
            staged_modules=tmp_path / "also-nope",
        )
        is None
    )
