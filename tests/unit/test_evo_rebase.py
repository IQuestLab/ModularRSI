"""P2: moving the second lane onto the tree the first lane just created.

Both lanes are built off the same parent, so only one of them can be promoted
onto it as-is. The other has to land on a tree that already contains the first —
and how that move goes decides which of its gate results still mean anything
(`promotion.gates_after_rebase`).

The whole design turns on one question: **did the two lanes touch the same
file?**

* **Disjoint files** — the common case, since the two lanes came from different
  proposals. Copying the second lane's changed files onto a fresh copy of the
  new parent reproduces its change exactly: the diff against the new parent is
  byte-for-byte the diff it had against the old one. Smoke and review still
  describe that change, so they carry over; activation and routing do not,
  because the composer's candidate set now contains the first lane's variant.

* **Overlapping files** — copying would silently overwrite the first lane's
  change with the second lane's version of the same file. That must never
  happen quietly: the first lane is already promoted and in the archive, so a
  silent clobber means a generation exists whose change is gone from the tree.
  The conflict is reported and the caller decides (re-implement on the new
  parent, or drop the lane) — this module never resolves it by picking a winner.

A rebase result therefore never claims more than it knows: it reports the files
it moved, the conflicts it refused to resolve, and whether the change came
through unaltered.
"""

import pytest

from harbor.agents.terminus_2_modular.self_evo import promotion, rebase

pytestmark = pytest.mark.unit


def _tree(root, files):
    for rel, text in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return root


def _parent(tmp_path, name="gen_0", **extra):
    return _tree(
        tmp_path / name,
        {
            "modules/agent_loop/confirm_exit.py": "DESCRIPTION = 'baseline'\n",
            "modules/observation/plain.py": "DESCRIPTION = 'plain'\n",
            **extra,
        },
    )


# ---- the disjoint case ----------------------------------------------------


def test_a_disjoint_change_moves_across_untouched(tmp_path):
    old = _parent(tmp_path, "gen_0")
    new = _parent(tmp_path, "gen_1")
    (new / "modules/agent_loop/confirm_exit.py").write_text("DESCRIPTION = 'lane A'\n")
    lane_b = _parent(tmp_path, "staging_b")
    (lane_b / "modules/observation/plain.py").write_text("DESCRIPTION = 'lane B'\n")

    result = rebase.rebase_onto(
        staging=lane_b, old_parent=old, new_parent=new, dest=tmp_path / "rebased"
    )

    assert result.conflicts == []
    assert result.unchanged is True
    text = (result.dest / "modules/observation/plain.py").read_text()
    assert text == "DESCRIPTION = 'lane B'\n"


def test_the_first_lanes_change_is_still_there_afterwards(tmp_path):
    # the point of rebasing rather than re-promoting the old staging
    old = _parent(tmp_path, "gen_0")
    new = _parent(tmp_path, "gen_1")
    (new / "modules/agent_loop/confirm_exit.py").write_text("DESCRIPTION = 'lane A'\n")
    lane_b = _parent(tmp_path, "staging_b")
    (lane_b / "modules/observation/plain.py").write_text("DESCRIPTION = 'lane B'\n")

    result = rebase.rebase_onto(
        staging=lane_b, old_parent=old, new_parent=new, dest=tmp_path / "rebased"
    )

    kept = (result.dest / "modules/agent_loop/confirm_exit.py").read_text()
    assert kept == "DESCRIPTION = 'lane A'\n"


def test_a_brand_new_file_is_carried_over(tmp_path):
    # the novelty lane's normal shape: it ADDS a variant
    old = _parent(tmp_path, "gen_0")
    new = _parent(tmp_path, "gen_1")
    lane_b = _parent(tmp_path, "staging_b")
    (lane_b / "modules/observation/pager.py").write_text("DESCRIPTION = 'pager'\n")

    result = rebase.rebase_onto(
        staging=lane_b, old_parent=old, new_parent=new, dest=tmp_path / "rebased"
    )

    assert (result.dest / "modules/observation/pager.py").is_file()
    assert "NEW:modules/observation/pager.py" in result.files_moved


def test_the_change_is_reported_as_unaltered(tmp_path):
    # this is what lets promotion.gates_after_rebase keep smoke and review
    old = _parent(tmp_path, "gen_0")
    new = _parent(tmp_path, "gen_1")
    (new / "modules/agent_loop/confirm_exit.py").write_text("DESCRIPTION = 'lane A'\n")
    lane_b = _parent(tmp_path, "staging_b")
    (lane_b / "modules/observation/plain.py").write_text("DESCRIPTION = 'lane B'\n")

    result = rebase.rebase_onto(
        staging=lane_b, old_parent=old, new_parent=new, dest=tmp_path / "rebased"
    )

    reuse, rerun = promotion.gates_after_rebase(
        result.diff_hash_before, result.diff_hash_after
    )
    assert set(reuse) == {"review"}
    # smoke is tree-wide, not change-local: the merged tree is a combination
    # neither lane was ever loaded as, and re-running it costs nothing
    assert set(rerun) == {"smoke", "activation", "routing"}


def test_a_lane_that_changed_nothing_is_a_no_op(tmp_path):
    old = _parent(tmp_path, "gen_0")
    new = _parent(tmp_path, "gen_1")
    lane_b = _parent(tmp_path, "staging_b")

    result = rebase.rebase_onto(
        staging=lane_b, old_parent=old, new_parent=new, dest=tmp_path / "rebased"
    )

    assert result.files_moved == []
    assert result.conflicts == []


# ---- the overlapping case -------------------------------------------------


def test_the_same_file_from_both_lanes_is_a_conflict(tmp_path):
    old = _parent(tmp_path, "gen_0")
    new = _parent(tmp_path, "gen_1")
    (new / "modules/agent_loop/confirm_exit.py").write_text("DESCRIPTION = 'lane A'\n")
    lane_b = _parent(tmp_path, "staging_b")
    (lane_b / "modules/agent_loop/confirm_exit.py").write_text(
        "DESCRIPTION = 'lane B'\n"
    )

    result = rebase.rebase_onto(
        staging=lane_b, old_parent=old, new_parent=new, dest=tmp_path / "rebased"
    )

    assert result.conflicts == ["modules/agent_loop/confirm_exit.py"]


def test_a_conflict_never_overwrites_the_promoted_lanes_change(tmp_path):
    # lane A is already promoted and in the archive; clobbering it here would
    # leave a generation on record whose change is not in the tree
    old = _parent(tmp_path, "gen_0")
    new = _parent(tmp_path, "gen_1")
    (new / "modules/agent_loop/confirm_exit.py").write_text("DESCRIPTION = 'lane A'\n")
    lane_b = _parent(tmp_path, "staging_b")
    (lane_b / "modules/agent_loop/confirm_exit.py").write_text(
        "DESCRIPTION = 'lane B'\n"
    )

    result = rebase.rebase_onto(
        staging=lane_b, old_parent=old, new_parent=new, dest=tmp_path / "rebased"
    )

    kept = (result.dest / "modules/agent_loop/confirm_exit.py").read_text()
    assert kept == "DESCRIPTION = 'lane A'\n"


def test_a_conflict_does_not_claim_the_change_survived(tmp_path):
    old = _parent(tmp_path, "gen_0")
    new = _parent(tmp_path, "gen_1")
    (new / "modules/agent_loop/confirm_exit.py").write_text("DESCRIPTION = 'lane A'\n")
    lane_b = _parent(tmp_path, "staging_b")
    (lane_b / "modules/agent_loop/confirm_exit.py").write_text(
        "DESCRIPTION = 'lane B'\n"
    )

    result = rebase.rebase_onto(
        staging=lane_b, old_parent=old, new_parent=new, dest=tmp_path / "rebased"
    )

    assert result.unchanged is False
    reuse, rerun = promotion.gates_after_rebase(
        result.diff_hash_before, result.diff_hash_after
    )
    assert reuse == ()
    assert set(rerun) == set(promotion.ALL_GATES)


def test_disjoint_changes_alongside_a_conflict_still_move(tmp_path):
    # partial progress is reported honestly: what moved, and what did not
    old = _parent(tmp_path, "gen_0")
    new = _parent(tmp_path, "gen_1")
    (new / "modules/agent_loop/confirm_exit.py").write_text("DESCRIPTION = 'lane A'\n")
    lane_b = _parent(tmp_path, "staging_b")
    (lane_b / "modules/agent_loop/confirm_exit.py").write_text(
        "DESCRIPTION = 'lane B'\n"
    )
    (lane_b / "modules/observation/plain.py").write_text("DESCRIPTION = 'lane B obs'\n")

    result = rebase.rebase_onto(
        staging=lane_b, old_parent=old, new_parent=new, dest=tmp_path / "rebased"
    )

    assert result.conflicts == ["modules/agent_loop/confirm_exit.py"]
    assert result.files_moved == ["modules/observation/plain.py"]
    obs = (result.dest / "modules/observation/plain.py").read_text()
    assert obs == "DESCRIPTION = 'lane B obs'\n"


def test_two_lanes_adding_the_same_new_file_is_a_conflict(tmp_path):
    # both lanes invented `modules/observation/pager.py`; taking one silently
    # would report two independent variants where only one exists
    old = _parent(tmp_path, "gen_0")
    new = _parent(tmp_path, "gen_1")
    (new / "modules/observation/pager.py").write_text("DESCRIPTION = 'A pager'\n")
    lane_b = _parent(tmp_path, "staging_b")
    (lane_b / "modules/observation/pager.py").write_text("DESCRIPTION = 'B pager'\n")

    result = rebase.rebase_onto(
        staging=lane_b, old_parent=old, new_parent=new, dest=tmp_path / "rebased"
    )

    assert result.conflicts == ["modules/observation/pager.py"]
    assert (result.dest / "modules/observation/pager.py").read_text() == (
        "DESCRIPTION = 'A pager'\n"
    )


def test_the_same_new_file_with_identical_content_is_not_a_conflict(tmp_path):
    # both lanes wrote the same bytes: there is nothing to lose by taking it
    old = _parent(tmp_path, "gen_0")
    new = _parent(tmp_path, "gen_1")
    (new / "modules/observation/pager.py").write_text("DESCRIPTION = 'pager'\n")
    lane_b = _parent(tmp_path, "staging_b")
    (lane_b / "modules/observation/pager.py").write_text("DESCRIPTION = 'pager'\n")

    result = rebase.rebase_onto(
        staging=lane_b, old_parent=old, new_parent=new, dest=tmp_path / "rebased"
    )

    assert result.conflicts == []


def test_two_different_edits_to_one_file_are_not_the_same_change(tmp_path):
    # `gates_after_rebase` asks "is this literally the same change?", so the
    # fingerprint has to cover content. A path-set fingerprint would call two
    # different rewrites of one file identical and reuse a review of the other.
    assert rebase._diff_hash({"a.py": "hash1"}) != rebase._diff_hash({"a.py": "hash2"})


def test_a_change_the_other_lane_already_made_leaves_nothing_to_promote(tmp_path):
    # both lanes converged on the identical edit: it is not a conflict (nothing
    # is lost by taking it) but there is no longer a change to make a generation
    # out of — promoting it would add an empty gen to the lineage
    old = _parent(tmp_path, "gen_0")
    new = _parent(tmp_path, "gen_1")
    (new / "modules/observation/plain.py").write_text("DESCRIPTION = 'same idea'\n")
    lane_b = _parent(tmp_path, "staging_b")
    (lane_b / "modules/observation/plain.py").write_text("DESCRIPTION = 'same idea'\n")

    result = rebase.rebase_onto(
        staging=lane_b, old_parent=old, new_parent=new, dest=tmp_path / "rebased"
    )

    assert result.conflicts == []
    assert result.absorbed is True


def test_a_normal_rebase_is_not_absorbed(tmp_path):
    old = _parent(tmp_path, "gen_0")
    new = _parent(tmp_path, "gen_1")
    (new / "modules/agent_loop/confirm_exit.py").write_text("DESCRIPTION = 'lane A'\n")
    lane_b = _parent(tmp_path, "staging_b")
    (lane_b / "modules/observation/plain.py").write_text("DESCRIPTION = 'lane B'\n")

    result = rebase.rebase_onto(
        staging=lane_b, old_parent=old, new_parent=new, dest=tmp_path / "rebased"
    )

    assert result.absorbed is False


def test_a_lane_that_changed_nothing_is_not_called_absorbed(tmp_path):
    # it had nothing to contribute in the first place; that is a different fact
    # from "someone else already did it", and the caller reports them differently
    old = _parent(tmp_path, "gen_0")
    new = _parent(tmp_path, "gen_1")
    lane_b = _parent(tmp_path, "staging_b")

    result = rebase.rebase_onto(
        staging=lane_b, old_parent=old, new_parent=new, dest=tmp_path / "rebased"
    )

    assert result.absorbed is False


# ---- the destination ------------------------------------------------------


def test_the_rebase_leaves_the_original_staging_alone(tmp_path):
    # it is still the record of what that lane actually built
    old = _parent(tmp_path, "gen_0")
    new = _parent(tmp_path, "gen_1")
    lane_b = _parent(tmp_path, "staging_b")
    (lane_b / "modules/observation/plain.py").write_text("DESCRIPTION = 'lane B'\n")

    rebase.rebase_onto(
        staging=lane_b, old_parent=old, new_parent=new, dest=tmp_path / "rebased"
    )

    assert (lane_b / "modules/observation/plain.py").read_text() == (
        "DESCRIPTION = 'lane B'\n"
    )


def test_the_rebase_leaves_the_new_parent_alone(tmp_path):
    # gen_1 is already promoted; a rebase must never mutate the archive
    old = _parent(tmp_path, "gen_0")
    new = _parent(tmp_path, "gen_1")
    (new / "modules/agent_loop/confirm_exit.py").write_text("DESCRIPTION = 'lane A'\n")
    lane_b = _parent(tmp_path, "staging_b")
    (lane_b / "modules/observation/plain.py").write_text("DESCRIPTION = 'lane B'\n")

    rebase.rebase_onto(
        staging=lane_b, old_parent=old, new_parent=new, dest=tmp_path / "rebased"
    )

    assert (
        new / "modules/observation/plain.py"
    ).read_text() == "DESCRIPTION = 'plain'\n"
    assert (new / "modules/agent_loop/confirm_exit.py").read_text() == (
        "DESCRIPTION = 'lane A'\n"
    )


def test_an_existing_destination_is_replaced(tmp_path):
    # a resumed window must not merge into a half-built rebase from last time
    old = _parent(tmp_path, "gen_0")
    new = _parent(tmp_path, "gen_1")
    lane_b = _parent(tmp_path, "staging_b")
    dest = _tree(tmp_path / "rebased", {"modules/stale/leftover.py": "junk\n"})

    result = rebase.rebase_onto(
        staging=lane_b, old_parent=old, new_parent=new, dest=dest
    )

    assert not (result.dest / "modules/stale/leftover.py").exists()


# ---- `__init__.py` is inside the writable surface -------------------------
#
# The ignore rule read "any path component starting with `__`", meant for
# `__pycache__`. It also swallows every `modules/*/__init__.py` — and those are
# real, editable registration files (the package tree ships at least seven).
# A file the hash cannot see is a file two lanes can both rewrite without the
# rebase ever calling it a conflict.


def test_a_changed_init_py_is_seen_by_the_hash(tmp_path):
    before = _tree(tmp_path / "a", {"modules/tools/__init__.py": "# nothing\n"})
    after = _tree(tmp_path / "b", {"modules/tools/__init__.py": "from . import x\n"})

    assert rebase._file_hashes(before) != rebase._file_hashes(after)


def test_pycache_is_still_ignored(tmp_path):
    """The rule the `__` prefix was actually there for."""
    root = _tree(
        tmp_path / "c",
        {"modules/tools/__pycache__/x.cpython-313.pyc": "junk\n"},
    )
    assert rebase._file_hashes(root) == {}


def test_compiled_and_hidden_files_are_still_ignored(tmp_path):
    root = _tree(
        tmp_path / "d",
        {"modules/tools/x.pyc": "junk\n", "modules/.DS_Store": "junk\n"},
    )
    assert rebase._file_hashes(root) == {}
