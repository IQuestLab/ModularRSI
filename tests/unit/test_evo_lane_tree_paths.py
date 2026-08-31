"""A rebased lane is diffed against the tree it was rebased onto.

Both lanes of a window start from the same parent. When both clear their gates
the incumbent promotes first and the novelty lane is rebased onto the generation
the incumbent just created — so from that point the novelty lane's parent is
`gen_{N+1}`, not the `current_gen` the window opened with.

The parent must be the generation the lane currently sits on, not necessarily
the generation that opened the window.
"""

import pytest

from harbor.agents.terminus_2_modular.self_evo import online_evo

pytestmark = pytest.mark.unit


def _paths(tmp_path, parent, *, staging="staging/novelty"):
    return online_evo._lane_tree_paths(tmp_path / staging, parent)


def test_the_editor_edits_the_modules_tree_and_diffs_against_the_parent(tmp_path):
    staging, diff_base = _paths(tmp_path, tmp_path / "gen_7")
    assert staging == tmp_path / "staging" / "novelty" / "modules"
    assert diff_base == tmp_path / "gen_7" / "modules"


def test_a_rebased_lane_diffs_against_its_new_parent(tmp_path):
    _, diff_base = _paths(tmp_path, tmp_path / "gen_8")
    assert diff_base == tmp_path / "gen_8" / "modules"
