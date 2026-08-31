"""Module staging helpers for the Phase0 evolution loop.

parent_gen/modules/   ──cp──>  staging/modules/   ──editor edits──>
staging/modules/      ──smoke pass──>  gen_{N+1}/modules/   (atomic mv)
staging/modules/      ──smoke fail──>  rm -rf, retry or give up
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path


def prepare_staging(
    parent_modules_dir: Path, staging_root: Path, fresh: bool = True
) -> Path:
    """Copy `parent_modules_dir` → `staging_root/modules/`.

    Args:
        parent_modules_dir: e.g., `.../gen_0/modules/` or the installed
            `terminus_2_modular/modules/` directory.
        staging_root: directory to hold the staging copy (will be created).
        fresh: if True, remove any existing `staging_root` first.

    Returns:
        The path to the copied modules dir, i.e. `staging_root / "modules"`.
    """
    parent_modules_dir = Path(parent_modules_dir).resolve()
    if not parent_modules_dir.is_dir():
        raise FileNotFoundError(f"parent modules dir not found: {parent_modules_dir}")

    staging_root = Path(staging_root).resolve()
    if fresh and staging_root.exists():
        shutil.rmtree(staging_root)
    staging_root.mkdir(parents=True, exist_ok=True)

    staging_modules = staging_root / "modules"
    shutil.copytree(
        parent_modules_dir,
        staging_modules,
        # Skip caches / hidden files
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".*"),
    )
    return staging_modules


def atomic_promote(staging_root: Path, target_gen_dir: Path) -> Path:
    """On smoke-test pass: rename `staging_root` → `target_gen_dir`.

    Returns the path of the promoted gen dir. Raises if target already exists.
    """
    staging_root = Path(staging_root).resolve()
    target_gen_dir = Path(target_gen_dir).resolve()
    if target_gen_dir.exists():
        raise FileExistsError(f"target gen dir already exists: {target_gen_dir}")
    target_gen_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_root.rename(target_gen_dir)
    return target_gen_dir


def discard_staging(staging_root: Path) -> None:
    """On smoke-test fail: rm -rf staging."""
    staging_root = Path(staging_root).resolve()
    if staging_root.exists():
        shutil.rmtree(staging_root)


def next_gen_number(archive_root: Path) -> int:
    """Scan `archive_root` for `gen_N/` dirs; return next N."""
    archive_root = Path(archive_root)
    if not archive_root.is_dir():
        return 0
    nums = []
    for p in archive_root.iterdir():
        if not p.is_dir():
            continue
        name = p.name
        if not name.startswith("gen_"):
            continue
        try:
            nums.append(int(name[4:]))
        except ValueError:
            continue
    return (max(nums) + 1) if nums else 0


def initialize_gen_0(archive_root: Path, parent_modules_dir: Path) -> Path:
    """If `archive_root/gen_0/` doesn't exist, snapshot the parent modules dir
    into it. Idempotent. Returns the gen_0 directory."""
    archive_root = Path(archive_root)
    gen0 = archive_root / "gen_0"
    if not gen0.exists():
        archive_root.mkdir(parents=True, exist_ok=True)
        # Build gen_0/modules/ via the staging helper, then rename in place
        tmp = archive_root / f".tmp_init_{int(time.time())}"
        prepare_staging(parent_modules_dir, tmp, fresh=True)
        tmp.rename(gen0)
    return gen0
