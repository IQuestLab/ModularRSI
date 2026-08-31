"""P0-2: `archive_path` must accept a run directory, the archive.json file
itself, or None — and the editor's `<archive>` action must see the REAL
status/genealogy through any of them.

Bug being fixed: every caller in the repo either passed nothing (online_evo)
or passed the run DIRECTORY (run_in), while the consumer required
`Path(ap).is_file()` — so every editor session silently fell back to
`seed_from_library`, which marks ALL variants `active`. The editor never saw
a `superseded` status, which is why gen_12+ kept editing retired modules.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

from harbor.agents.terminus_2_modular import archive as arch
from harbor.agents.terminus_2_modular.modules.tools.editor_file_tools import (
    EditorFileTools,
)

pytestmark = pytest.mark.unit


def _write_archive(run_dir: Path) -> None:
    """A minimal real archive: one active + one superseded agent_loop variant."""
    arch.save_archive(
        run_dir,
        [
            arch.ArchiveEntry(
                name="baseline",
                type="agent_loop",
                niche={"grounding": "reactive"},
                parent_ids=[],
                addresses="baseline",
                change_type="baseline",
                born_gen="gen_0",
                status="active",
            ),
            arch.ArchiveEntry(
                name="old_variant",
                type="agent_loop",
                niche={"grounding": "task-anchored"},
                parent_ids=["baseline"],
                addresses="stuck loops",
                change_type="add",
                born_gen="gen_1",
                status="superseded",
            ),
        ],
    )


# ---- resolve_archive_file: the three input shapes -------------------------


def test_resolve_accepts_run_directory(tmp_path):
    _write_archive(tmp_path)
    # str, because run_editor stores archive_path as str
    resolved = arch.resolve_archive_file(str(tmp_path))
    assert resolved == tmp_path / arch.ARCHIVE_FILENAME


def test_resolve_accepts_archive_file(tmp_path):
    _write_archive(tmp_path)
    f = tmp_path / arch.ARCHIVE_FILENAME
    assert arch.resolve_archive_file(f) == f


def test_resolve_none_is_none():
    assert arch.resolve_archive_file(None) is None


def test_resolve_dir_without_archive_falls_back_to_none(tmp_path):
    # no archive.json inside → caller must use its fallback, not crash
    assert arch.resolve_archive_file(tmp_path) is None


# ---- the <archive> action sees real status through a DIRECTORY path -------


def _ctx(archive_path) -> SimpleNamespace:
    return SimpleNamespace(
        state=SimpleNamespace(archive_path=archive_path),
        shared=SimpleNamespace(),
    )


def test_do_archive_shows_superseded_via_directory_path(tmp_path):
    """run_in passes the run DIRECTORY — that must be enough to see the real
    archive (this is exactly the input shape that used to silently fall back)."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_archive(run_dir)
    staging = tmp_path / "staging"  # irrelevant: fallback must NOT be taken
    staging.mkdir()

    res = EditorFileTools._do_archive(staging, "agent_loop", "", _ctx(str(run_dir)))

    assert res.success, res.error
    assert "superseded" in res.output
    assert "old_variant" in res.output


def test_do_archive_supplied_but_unresolvable_path_is_loud(tmp_path):
    """A SUPPLIED archive_path that doesn't resolve must announce the fallback
    — the original bug survived 2+ months precisely because the fallback was
    silent (every session looked like it had a real archive)."""
    staging = tmp_path / "staging"
    staging.mkdir()
    ghost = tmp_path / "no_such_run_dir"

    res = EditorFileTools._do_archive(staging, "agent_loop", "", _ctx(str(ghost)))

    assert res.success, res.error
    assert "archive_path" in res.output  # names the problem…
    assert "no_such_run_dir" in res.output  # …and the path that failed


def test_do_archive_no_path_supplied_stays_quiet(tmp_path):
    # None was never a misconfiguration — the fallback note must not fire
    staging = tmp_path / "staging"
    staging.mkdir()
    res = EditorFileTools._do_archive(staging, "agent_loop", "", _ctx(None))
    assert res.success, res.error
    assert "archive_path" not in res.output


def test_do_archive_shows_superseded_via_file_path(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_archive(run_dir)
    staging = tmp_path / "staging"
    staging.mkdir()
    f = run_dir / arch.ARCHIVE_FILENAME

    res = EditorFileTools._do_archive(staging, "agent_loop", "", _ctx(str(f)))

    assert res.success, res.error
    assert "superseded" in res.output
