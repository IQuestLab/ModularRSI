"""P2 — moving the second lane onto the tree the first lane just created.

Both lanes are built off the same parent, so only one can be promoted onto it
as-is. The other has to land on a tree that already contains the first, and how
that move goes decides which of its gate results still mean anything (see
:func:`promotion.gates_after_rebase`).

Everything turns on one question: **did the two lanes touch the same file?**

*Disjoint* is the common case — the two lanes came from different proposals.
Copying the second lane's changed files onto a fresh copy of the new parent
reproduces its change exactly, so its diff against the new parent is byte-for-
byte the diff it had against the old one. Smoke and review still describe that
change and carry over; activation and routing do not, because the composer's
candidate set now contains the first lane's variant.

*Overlapping* means copying would overwrite the first lane's change with the
second lane's version of the same file. That must never happen quietly: the
first lane is already promoted and in the archive, so a silent clobber leaves a
generation on record whose change is not in the tree. This module reports the
conflict and resolves nothing — picking a winner here would be inventing a
change neither implementer wrote. The caller decides: re-run the second
implementer against the new parent (a real rebase, all gates re-run), or drop
the lane.

Two lanes creating the *same new path* is the same problem in a different
shape — unless they wrote identical bytes, in which case there is nothing to
lose by taking it.
"""

from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass, field
from pathlib import Path

_IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".*")

#: Marker used by the change lists this module produces, matching the convention
#: in `trajectory_analysis._changed_py_files`.
NEW_PREFIX = "NEW:"


@dataclass
class RebaseResult:
    dest: Path
    files_moved: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    #: True when the change came through byte-identical — the only condition
    #: under which the change-local gate verdicts still describe it.
    unchanged: bool = True
    #: The lane had a change, and the promoted lane had already made exactly it.
    #: Not a conflict (nothing is lost by taking it) but nothing is left to
    #: promote either: a generation built from this would have an empty diff.
    #: Distinct from "this lane changed nothing at all", which is a different
    #: fact and is reported differently by the caller.
    absorbed: bool = False
    diff_hash_before: str = ""
    diff_hash_after: str = ""


def _file_hashes(root: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    root = Path(root)
    if not root.is_dir():
        return out
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        # `__pycache__`, not "anything starting with `__`". The broad rule also
        # swallowed every `modules/*/__init__.py`, and those are editable
        # registration files: a file the hash cannot see is one both lanes can
        # rewrite without the rebase ever calling it a conflict.
        if any(part == "__pycache__" or part.startswith(".") for part in rel.parts):
            continue
        if path.suffix in (".pyc", ".pyo"):
            continue
        try:
            out[str(rel)] = hashlib.md5(path.read_bytes()).hexdigest()
        except OSError:
            continue
    return out


def _changed_against(base: dict[str, str], tree: dict[str, str]) -> dict[str, str]:
    """Files in ``tree`` that ``base`` does not have, or has differently."""
    return {rel: h for rel, h in tree.items() if base.get(rel) != h}


def _diff_hash(changed: dict[str, str]) -> str:
    """A stable fingerprint of a change: which files, and their exact content.

    Comparing these before and after the move is what decides whether the
    change-local gates still apply — so it has to cover content, not just the
    set of paths.
    """
    if not changed:
        return ""
    payload = "\n".join(f"{rel}\x1f{h}" for rel, h in sorted(changed.items()))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def rebase_onto(
    *, staging: Path, old_parent: Path, new_parent: Path, dest: Path
) -> RebaseResult:
    """Re-apply ``staging``'s change (vs ``old_parent``) on top of ``new_parent``.

    Neither ``staging`` nor ``new_parent`` is modified: the staging is still the
    record of what that lane built, and the new parent is already in the archive.
    ``dest`` is replaced if it exists — a resumed window must not merge into a
    half-built rebase left over from last time.
    """
    staging, old_parent = Path(staging), Path(old_parent)
    new_parent, dest = Path(new_parent), Path(dest)

    old_h = _file_hashes(old_parent)
    new_h = _file_hashes(new_parent)
    staging_h = _file_hashes(staging)

    mine = _changed_against(old_h, staging_h)  # what this lane changed
    theirs = _changed_against(old_h, new_h)  # what the promoted lane changed

    conflicts = sorted(
        rel
        for rel, h in mine.items()
        # the other lane touched the same file, and not to the same bytes
        if rel in theirs and theirs[rel] != h
    )
    movable = {rel: h for rel, h in mine.items() if rel not in conflicts}

    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(new_parent, dest, ignore=_IGNORE)

    for rel in movable:
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(staging / rel, target)

    after = _changed_against(new_h, _file_hashes(dest))
    files_moved = sorted(
        (NEW_PREFIX + rel if rel not in old_h else rel) for rel in movable
    )
    return RebaseResult(
        dest=dest,
        files_moved=files_moved,
        conflicts=conflicts,
        unchanged=not conflicts,
        # had something to contribute, and the new parent already contains it
        absorbed=bool(mine) and not conflicts and not after,
        diff_hash_before=_diff_hash(mine),
        diff_hash_after=_diff_hash(after),
    )
