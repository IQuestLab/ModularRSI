"""Unit tests for the niche-archive layer (niche.py / archive.py).

Covers the pure functions behind MAP-Elites niching + genealogy, and in particular
LOCKS the F1 fix: parse_variant_meta is line-based, so a flattened text (the old
bug: feeding the human-readable out.intent) must be shown to degrade — the pipeline
has to feed the RAW trajectory instead.
"""

import pytest

from harbor.agents.terminus_2_modular import archive as A
from harbor.agents.terminus_2_modular import niche as N


@pytest.mark.unit
def test_niche_key_seed_order_and_edges():
    # seed-axis order regardless of dict insertion order
    k = N.niche_key(
        "observation", {"staleness": "none", "capture": "full", "capacity": "low"}
    )
    assert k == (("capture", "full"), ("capacity", "low"), ("staleness", "none"))
    # missing axis -> coarser cell (only the present axes)
    assert N.niche_key("observation", {"capture": "full"}) == (("capture", "full"),)
    # empty niche -> empty key (all undeclared variants collide = the correct signal)
    assert N.niche_key("observation", {}) == ()
    # extra (non-seed) axes are appended in sorted order
    k2 = N.niche_key("observation", {"capture": "full", "zzz": "1", "aaa": "2"})
    assert k2 == (("capture", "full"), ("aaa", "2"), ("zzz", "1"))


@pytest.mark.unit
def test_parse_variant_meta_multiline():
    raw = (
        '<variant_meta name="cautious" type="observation">\n'
        "  PARENT: a, b\n  CHANGE: merge\n"
        "  ADDRESSES: stale pane\n  SUPERSEDES: old_v\n</variant_meta>"
    )
    (m,) = A.parse_variant_meta(raw)
    assert m["name"] == "cautious" and m["type"] == "observation"
    assert m["parent_ids"] == ["a", "b"]  # comma-separated (merge)
    assert m["change_type"] == "merge"
    assert m["addresses"] == "stale pane"
    assert m["supersedes"] == ["old_v"]


@pytest.mark.unit
def test_parse_variant_meta_flattened_input_degrades():
    """LOCKS F1. Flattening newlines (the old ` `.join(out.intent.split()) path)
    collapses the block onto one line, so only PARENT is hit and it swallows the
    rest — supersedes/change/addresses are lost. This is exactly why the pipeline
    must feed the RAW trajectory (extract_variant_meta_blocks), never out.intent."""
    raw = (
        '<variant_meta name="c" type="observation">\n'
        "  PARENT: a\n  CHANGE: modify\n  SUPERSEDES: old\n</variant_meta>"
    )
    (good,) = A.parse_variant_meta(raw)
    assert good["supersedes"] == ["old"] and good["change_type"] == "modify"

    flat = " ".join(raw.split())  # what extract_editor_intent produces
    (bad,) = A.parse_variant_meta(flat)
    assert bad["supersedes"] == []  # escape hatch dead
    assert bad["change_type"] == "add"  # CHANGE lost
    assert bad["addresses"] == ""


@pytest.mark.unit
def test_parse_variant_meta_missing_and_empty():
    assert A.parse_variant_meta("") == []
    (m,) = A.parse_variant_meta('<variant_meta name="x" type="tools"></variant_meta>')
    assert m["change_type"] == "add"
    assert m["parent_ids"] == [] and m["supersedes"] == [] and m["addresses"] == ""


@pytest.mark.unit
def test_collisions_flags_same_niche_cell():
    entries = [
        A.ArchiveEntry(
            name="a",
            type="observation",
            niche={"capture": "full", "capacity": "low", "staleness": "none"},
        ),
        A.ArchiveEntry(
            name="b",
            type="observation",
            niche={"capture": "full", "capacity": "low", "staleness": "none"},
        ),
        A.ArchiveEntry(
            name="c",
            type="observation",
            niche={"capture": "hybrid", "capacity": "100k", "staleness": "none"},
        ),
    ]
    crowded = [
        names
        for names in A.collisions(entries, "observation").values()
        if len(names) > 1
    ]
    assert crowded == [["a", "b"]]


@pytest.mark.unit
def test_seed_from_library_excludes_editor_only():
    class _Info:
        def __init__(self, name, type_, niche):
            self.name, self.type, self.niche = name, type_, niche

    class _Lib:
        def list_infos(self):
            return [
                _Info("raw", "observation", {"capture": "incremental"}),
                _Info("editor_file_tools", "tools", {"audience": "editor"}),
            ]

    entries = {e.name: e for e in A.seed_from_library(_Lib(), born_gen="gen_0")}
    assert entries["raw"].status == "active"
    assert entries["editor_file_tools"].status == "excluded"


@pytest.mark.unit
def test_render_lineage_is_cycle_safe():
    e1 = A.ArchiveEntry(name="a", type="observation", parent_ids=["observation/b"])
    e2 = A.ArchiveEntry(name="b", type="observation", parent_ids=["observation/a"])
    out = A.render_lineage([e1, e2], "observation/a")  # must NOT RecursionError
    assert "(cycle)" in out


# ---------------------------------------------------------------------------
# supersede_targets — Fix A: a merge retires its parents; seeds are protected.
# ---------------------------------------------------------------------------


def _existing_obs():
    """A little archive: two evolved variants + one baseline seed."""
    return [
        A.ArchiveEntry(
            name="raw_terminal",
            type="observation",
            change_type="baseline",
            born_gen="gen_0",
        ),
        A.ArchiveEntry(
            name="tail_biased", type="observation", change_type="add", born_gen="gen_2"
        ),
        A.ArchiveEntry(
            name="full_pane", type="observation", change_type="add", born_gen="gen_3"
        ),
    ]


@pytest.mark.unit
def test_supersede_targets_merge_retires_parents():
    metas = [
        {
            "name": "hybrid",
            "type": "observation",
            "parent_ids": ["tail_biased", "full_pane"],
            "change_type": "merge",
            "supersedes": [],
        }
    ]
    assert A.supersede_targets(metas, _existing_obs()) == {
        "observation/tail_biased",
        "observation/full_pane",
    }


@pytest.mark.unit
def test_supersede_targets_never_retires_seed():
    # merge whose parent is the baseline seed → seed stays, evolved parent retires
    metas = [
        {
            "name": "hybrid",
            "type": "observation",
            "parent_ids": ["tail_biased", "raw_terminal"],
            "change_type": "merge",
            "supersedes": [],
        }
    ]
    assert A.supersede_targets(metas, _existing_obs()) == {"observation/tail_biased"}
    # even an EXPLICIT supersede of the seed is refused (protects DEFAULT_BUNDLE)
    metas2 = [
        {
            "name": "x",
            "type": "observation",
            "parent_ids": [],
            "change_type": "add",
            "supersedes": ["raw_terminal"],
        }
    ]
    assert A.supersede_targets(metas2, _existing_obs()) == set()


@pytest.mark.unit
def test_supersede_targets_explicit_and_add_semantics():
    # explicit SUPERSEDES on a plain add works
    metas = [
        {
            "name": "x",
            "type": "observation",
            "parent_ids": [],
            "change_type": "add",
            "supersedes": ["full_pane"],
        }
    ]
    assert A.supersede_targets(metas, _existing_obs()) == {"observation/full_pane"}
    # CHANGE: add with a PARENT does NOT retire the parent (keep it distinct)
    metas2 = [
        {
            "name": "x",
            "type": "observation",
            "parent_ids": ["tail_biased"],
            "change_type": "add",
            "supersedes": [],
        }
    ]
    assert A.supersede_targets(metas2, _existing_obs()) == set()


@pytest.mark.unit
def test_supersede_targets_resolves_qual_and_ignores_unknown():
    metas = [
        {
            "name": "x",
            "type": "observation",
            "parent_ids": [],
            "change_type": "add",
            "supersedes": ["observation/full_pane", "does_not_exist"],
        }
    ]
    assert A.supersede_targets(metas, _existing_obs()) == {"observation/full_pane"}


@pytest.mark.unit
def test_render_lineage_finds_children_through_an_unqualified_parent_id():
    # `PARENT: confirm_exit` — the form editors actually write, and the form the
    # archive stores verbatim. The ancestry walk already tolerated it; the
    # descendants line did not, so a variant with children rendered as
    # "children: (none)" — telling the editor a line was dead when it was not.
    parent = A.ArchiveEntry(name="confirm_exit", type="agent_loop")
    child = A.ArchiveEntry(
        name="deadline_aware", type="agent_loop", parent_ids=["confirm_exit"]
    )
    out = A.render_lineage([parent, child], "agent_loop/confirm_exit")
    assert "children: agent_loop/deadline_aware" in out


@pytest.mark.unit
def test_render_lineage_does_not_borrow_children_from_another_type():
    # a bare parent belongs to the CHILD's type; matching it across types would
    # invent a lineage that does not exist
    parent = A.ArchiveEntry(name="confirm_exit", type="agent_loop")
    stranger = A.ArchiveEntry(
        name="deadline_aware", type="observation", parent_ids=["confirm_exit"]
    )
    out = A.render_lineage([parent, stranger], "agent_loop/confirm_exit")
    assert "children: (none)" in out
