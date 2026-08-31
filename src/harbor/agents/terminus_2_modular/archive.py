"""Variant archive: the niche/genealogy registry for one evolution lineage.

This is the source of truth for *how the module library is organized* — replacing
the implicit "gen = full snapshot" chain with an explicit per-variant record:

    - niche      : behavior-descriptor cell (see niche.py) — for dedup + coverage
    - parent_ids : which variant(s) this one descends from  — genealogy DAG (merge → many)
    - addresses  : the symptom it was created to fix          — "has this been tried?"
    - change_type: add | modify | merge | baseline
    - born_gen   : the gen that introduced it
    - status     : active | superseded | excluded

Stored per-run at ``<run_dir>/archive.json`` and updated incrementally at each
promotion (parent/addresses only exist at creation time, so they must be persisted
— they cannot be recovered from the module files alone). NO reward/fitness lives
here: this module only *organizes* the library, it never ranks variants.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from harbor.agents.terminus_2_modular import niche as _niche

ARCHIVE_FILENAME = "archive.json"


# ---------------------------------------------------------------------------
# <variant_meta> declaration parsing (D2 contract) — the editor emits, per
# variant it creates/modifies, a block declaring provenance; the pipeline parses
# it at promotion to populate parent_ids / change_type / addresses (NICHE itself
# lives in the file and is read back from the loaded library).
# ---------------------------------------------------------------------------

_VARIANT_META_RE = re.compile(
    r"<variant_meta\b([^>]*)>(.*?)</variant_meta>", re.DOTALL | re.IGNORECASE
)
_META_ATTR_RE = re.compile(r'([\w-]+)\s*=\s*"([^"]*)"')


def parse_variant_meta(text: str) -> list[dict[str, Any]]:
    """Parse ``<variant_meta name=".." type="..">`` blocks with PARENT / CHANGE /
    ADDRESSES lines. Tolerant of missing fields (change_type defaults to 'add')."""
    out: list[dict[str, Any]] = []
    for m in _VARIANT_META_RE.finditer(text or ""):
        attrs = dict(_META_ATTR_RE.findall(m.group(1)))
        fields: dict[str, str] = {}
        for line in m.group(2).splitlines():
            s = line.strip()
            for key in ("PARENT", "CHANGE", "ADDRESSES", "SUPERSEDES"):
                if s.upper().replace(" ", "").startswith(key + ":"):
                    fields[key] = s.split(":", 1)[1].strip()
        parent = fields.get("PARENT", "").strip()
        parent_ids = [p.strip() for p in parent.split(",") if p.strip()]
        supersedes = [
            p.strip() for p in fields.get("SUPERSEDES", "").split(",") if p.strip()
        ]
        change = (fields.get("CHANGE") or "add").strip().lower()
        if change not in ("add", "modify", "merge"):
            change = "add"
        out.append(
            {
                "name": attrs.get("name", "").strip(),
                "type": attrs.get("type", "").strip(),
                "parent_ids": parent_ids,
                "change_type": change,
                "addresses": fields.get("ADDRESSES", "").strip(),
                "supersedes": supersedes,
            }
        )
    return out


@dataclass
class ArchiveEntry:
    name: str
    type: str
    niche: dict[str, str] = field(default_factory=dict)
    parent_ids: list[str] = field(default_factory=list)  # ["observation/raw_terminal"]
    addresses: str = ""
    change_type: str = "add"  # add | modify | merge | baseline
    born_gen: str = ""
    status: str = "active"  # active | superseded | excluded

    @property
    def qual(self) -> str:
        """Qualified id, unique across types: 'observation/hybrid_clean'."""
        return f"{self.type}/{self.name}"

    @property
    def key(self) -> tuple[tuple[str, str], ...]:
        return _niche.niche_key(self.type, self.niche)

    @property
    def solver_selectable(self) -> bool:
        return _niche.is_solver_selectable(self.niche)


# ---------------------------------------------------------------------------
# Load / save / update
# ---------------------------------------------------------------------------


def archive_path(run_dir: Path | str) -> Path:
    return Path(run_dir) / ARCHIVE_FILENAME


def resolve_archive_file(path_like: Path | str | None) -> Path | None:
    """Normalize an `archive_path` input to the archive.json file, or None.

    Callers hand this either the run directory or the archive.json itself
    (Older callers passed the directory, which `is_file()` silently
    rejected — every editor session then fell back to `seed_from_library`,
    where ALL variants look `active`). None / a path with no archive.json
    resolves to None: the caller keeps its fallback.
    """
    if path_like is None:
        return None
    p = Path(path_like)
    if p.is_dir():
        p = p / ARCHIVE_FILENAME
    return p if p.is_file() else None


def load_archive(run_dir: Path | str) -> list[ArchiveEntry]:
    p = archive_path(run_dir)
    if not p.is_file():
        return []
    raw: list[dict[str, Any]] = json.loads(p.read_text())
    return [ArchiveEntry(**e) for e in raw]


def save_archive(run_dir: Path | str, entries: list[ArchiveEntry]) -> None:
    p = archive_path(run_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps([asdict(e) for e in entries], ensure_ascii=False, indent=1))


def update_archive(
    run_dir: Path | str, new_entries: list[ArchiveEntry]
) -> list[ArchiveEntry]:
    """Upsert ``new_entries`` (keyed by ``qual``) into the run's archive and save.

    A modify/supersede that flips an incumbent's status should be passed as an
    updated entry for that incumbent's qual too. Returns the merged list.
    """
    by_qual = {e.qual: e for e in load_archive(run_dir)}
    for e in new_entries:
        by_qual[e.qual] = e
    merged = list(by_qual.values())
    save_archive(run_dir, merged)
    return merged


def set_status(run_dir: Path | str, qual: str, status: str) -> bool:
    """Flip one variant's status (e.g. roll back a regressing provisional variant).

    Use ``"superseded"`` — NOT ``"excluded"`` — as the durable off-switch: the
    promote sync (_sync_archive_after_promote) re-derives active/excluded from the
    niche each promotion, so a manual ``excluded`` on a solver-selectable variant
    is overwritten; only ``superseded`` survives. Returns True if a matching entry
    was found and saved."""
    entries = load_archive(run_dir)
    hit = False
    for e in entries:
        if e.qual == qual:
            e.status = status
            hit = True
    if hit:
        save_archive(run_dir, entries)
    return hit


def seed_from_library(library: Any, born_gen: str = "gen_0") -> list[ArchiveEntry]:
    """Build the initial archive from a loaded library's ModuleInfos.

    Baseline variants have no parent and change_type='baseline'. Editor-only
    variants (audience=editor, e.g. editor_file_tools) are marked status='excluded'
    so the solver composer skips them.
    """
    out: list[ArchiveEntry] = []
    for info in library.list_infos():
        niche = dict(info.niche or {})
        out.append(
            ArchiveEntry(
                name=info.name,
                type=info.type,
                niche=niche,
                parent_ids=[],
                addresses="baseline",
                change_type="baseline",
                born_gen=born_gen,
                status="active" if _niche.is_solver_selectable(niche) else "excluded",
            )
        )
    return out


# ---------------------------------------------------------------------------
# Queries (dedup / composer)
# ---------------------------------------------------------------------------


def collisions(entries: list[ArchiveEntry], type_: str) -> dict[tuple, list[str]]:
    """Active variants of ``type_`` grouped by niche key; keys with >1 = collisions."""
    cells: dict[tuple, list[str]] = {}
    for e in entries:
        if e.type != type_ or e.status != "active":
            continue
        cells.setdefault(e.key, []).append(e.name)
    return cells


def supersede_targets(
    metas: list[dict[str, Any]], existing: list[ArchiveEntry]
) -> set[str]:
    """Quals of existing variants a promoted batch retires (status → superseded).

    Two semantic sources, both meaning "this variant is now subsumed":
      (a) an explicit ``SUPERSEDES: <name|type/name>`` line on any variant_meta;
      (b) the parents of a ``CHANGE: merge`` variant — merging A+B into C
          consolidates A,B INTO C, so they retire. (Use ``CHANGE: add`` instead
          to keep a parent as a distinct option.)

    Seeds/baseline are NEVER retired: the composer falls back to them
    (DEFAULT_BUNDLE), so superseding one would strip its safety net.
    """
    by_qual = {e.qual: e for e in existing}
    by_name: dict[str, ArchiveEntry] = {}
    for e in existing:
        by_name.setdefault(e.name, e)

    def _resolve(ref: str, default_type: str) -> ArchiveEntry | None:
        ref = ref.strip()
        if not ref:
            return None
        if "/" in ref:
            return by_qual.get(ref)
        # bare name → prefer the same-type qual, else any entry with that name
        return by_qual.get(f"{default_type}/{ref}") or by_name.get(ref)

    def _is_seed(e: ArchiveEntry) -> bool:
        return e.change_type == "baseline" or e.born_gen == "gen_0"

    targets: set[str] = set()
    for m in metas:
        refs = list(m.get("supersedes") or [])
        if (m.get("change_type") or "add") == "merge":
            refs += list(m.get("parent_ids") or [])
        for ref in refs:
            e = _resolve(ref, m.get("type", ""))
            if e is not None and not _is_seed(e):
                targets.add(e.qual)
    return targets


# (Removed dead `active_by_niche`: the composer skips superseded/excluded via
# _archive_skip + its own niche dedup on ModuleInfos, so this ArchiveEntry-based
# helper had no caller — F4.)


# ---------------------------------------------------------------------------
# Rendering (the read-only `archive` editor tool — D4)
# ---------------------------------------------------------------------------


def render_type(entries: list[ArchiveEntry], type_: str) -> str:
    """Module-scoped view: occupied cells (variant · niche · addresses · lineage)
    + which axis values have/haven't been used. This is what the editor sees when
    it decides to touch module type ``type_`` (D4: investigate the module you change).
    """
    rows = [e for e in entries if e.type == type_]
    if not rows:
        return f"(archive: no '{type_}' variants recorded yet)"
    lines = [f"### archive — {type_}  ({len(rows)} variants)"]
    cells = collisions(entries, type_)
    crowded = {k: v for k, v in cells.items() if len(v) > 1}
    for e in sorted(rows, key=lambda x: (x.status, x.name)):
        par = f" ⟵ {', '.join(e.parent_ids)}" if e.parent_ids else ""
        st = "" if e.status == "active" else f" [{e.status}]"
        lines.append(
            f"  {e.name}{st}: niche={e.niche}{par}"
            f"  (born {e.born_gen}; addresses: {e.addresses})"
        )
    if crowded:
        lines.append(
            "  ⚠️ crowded cells (near-duplicates): "
            + "; ".join(f"{list(k)} → {names}" for k, names in crowded.items())
        )
    # per-axis coverage (dimensionality-reduced hint, not a full cartesian dump)
    axes = _niche.SEED_AXES.get(type_, {})
    if axes:
        cov = []
        for axis, seed_vals in axes.items():
            used = sorted({e.niche[axis] for e in rows if axis in e.niche})
            unused = [v for v in seed_vals if v not in used]
            cov.append(
                f"{axis}: used {used}" + (f", not yet {unused}" if unused else "")
            )
        lines.append("  axis coverage — " + " · ".join(cov))
    return "\n".join(lines)


def render_lineage(entries: list[ArchiveEntry], qual: str) -> str:
    """Ancestry + descendants of one variant (accepts 'name' or 'type/name')."""
    by_qual = {e.qual: e for e in entries}
    by_name = {e.name: e for e in entries}
    target = by_qual.get(qual) or by_name.get(qual)
    if target is None:
        return f"(archive: no variant '{qual}')"

    def anc(
        e: ArchiveEntry, depth: int = 0, seen: frozenset[str] = frozenset()
    ) -> list[str]:
        line = f"{'  ' * depth}{e.qual}: niche={e.niche} (addresses: {e.addresses})"
        # Cycle / self-loop guard: a variant appearing in its own ancestor path
        # (or depth blow-up from malformed parent_ids) stops here, not RecursionError.
        if e.qual in seen or depth > 50:
            return [line + "  (cycle)"]
        seen = seen | {e.qual}
        out = [line]
        for pid in e.parent_ids:
            p = by_qual.get(pid) or by_name.get(pid.split("/")[-1])
            if p is not None:
                out += anc(p, depth + 1, seen)
            else:
                out.append(f"{'  ' * (depth + 1)}{pid} (not in archive)")
        return out

    # A bare `PARENT: confirm_exit` names a variant of the CHILD's own type —
    # the ancestry walk above already reads it that way, and the descendants
    # line has to agree or a variant with children renders as "(none)".
    children = [
        e.qual
        for e in entries
        if any(
            (p if "/" in p else f"{e.type}/{p}") == target.qual
            for p in (e.parent_ids or [])
        )
    ]
    lines = [f"### lineage — {target.qual}", "ancestry:"]
    lines += anc(target)
    lines.append("children: " + (", ".join(children) if children else "(none)"))
    return "\n".join(lines)
