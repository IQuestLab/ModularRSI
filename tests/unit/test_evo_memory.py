"""Unit tests for editor memory (append-only log + compact index + taboo)."""

import pytest

from harbor.agents.terminus_2_modular.self_evo import editor_memory as m

pytestmark = pytest.mark.unit


def test_record_and_load(tmp_path):
    m.record(tmp_path, epoch=0, module="agent_loop", change="add retry", gen="gen_1")
    m.record(tmp_path, epoch=1, module="tools", change="new grep tool", gen="gen_2")
    rows = m.load(tmp_path)
    assert len(rows) == 2
    assert rows[0]["module"] == "agent_loop" and rows[0]["change"] == "add retry"


def test_compact_index_filters_by_module_and_lists_taboo(tmp_path):
    m.record(
        tmp_path, epoch=0, module="agent_loop", change="add retry loop", gen="gen_1"
    )
    m.record(tmp_path, epoch=1, module="tools", change="unrelated", gen="gen_2")
    m.record(
        tmp_path,
        epoch=2,
        module="agent_loop",
        change="widen verify window",
        verdict="rolled_back",
    )
    idx = m.compact_index(tmp_path, "agent_loop")
    assert "add retry loop" in idx
    assert "widen verify window" in idx
    assert "unrelated" not in idx  # different module filtered out
    # taboo section lists the rolled-back change
    assert "DO NOT re-propose" in idx
    assert idx.count("widen verify window") >= 2  # in the table AND the taboo list


def test_compact_index_empty_when_no_history(tmp_path):
    assert m.compact_index(tmp_path, "agent_loop") == ""


def test_load_missing_is_empty(tmp_path):
    assert m.load(tmp_path / "nope") == []
