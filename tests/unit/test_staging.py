from pathlib import Path

import pytest

import harbor.agents.terminus_2_modular as modular_package
from harbor.agents.terminus_2_modular.self_evo.staging import (
    atomic_promote,
    discard_staging,
    initialize_gen_0,
    next_gen_number,
    prepare_staging,
)

pytestmark = pytest.mark.unit

MODULES_DIR = Path(modular_package.__file__).parent / "modules"


def test_prepare_staging_copies_only_modules(tmp_path):
    modules = prepare_staging(MODULES_DIR, tmp_path / "staging")
    assert modules == (tmp_path / "staging" / "modules").resolve()
    assert (modules / "agent_loop").is_dir()
    assert not (tmp_path / "staging" / "agent.py").exists()


def test_prepare_staging_fresh_replaces_previous_tree(tmp_path):
    modules = prepare_staging(MODULES_DIR, tmp_path / "staging")
    marker = modules / "marker.py"
    marker.write_text("old")
    prepare_staging(MODULES_DIR, tmp_path / "staging", fresh=True)
    assert not marker.exists()


def test_promote_renames_staging_tree(tmp_path):
    prepare_staging(MODULES_DIR, tmp_path / "staging")
    target = atomic_promote(tmp_path / "staging", tmp_path / "gen_1")
    assert (target / "modules").is_dir()
    assert not (tmp_path / "staging").exists()


def test_initialize_gen_0_is_idempotent(tmp_path):
    gen0 = initialize_gen_0(tmp_path, MODULES_DIR)
    marker = gen0 / "marker"
    marker.write_text("keep")
    assert initialize_gen_0(tmp_path, MODULES_DIR) == gen0
    assert marker.read_text() == "keep"
    assert next_gen_number(tmp_path) == 1


def test_discard_staging_removes_tree(tmp_path):
    prepare_staging(MODULES_DIR, tmp_path / "staging")
    discard_staging(tmp_path / "staging")
    assert not (tmp_path / "staging").exists()
