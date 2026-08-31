"""Record the files and metadata produced for a proposal implementation.

Proposal intent and implementation evidence are stored separately so review,
promotion, and archive synchronization can refer to what was actually built.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from harbor.agents.terminus_2_modular.self_evo.backlog import backlog_dir

MANIFEST_FILENAME = "implementations.jsonl"


@dataclass
class ImplementationRecord:
    proposal_id: str
    step: str = ""
    files: list[str] = field(default_factory=list)
    variant_meta_text: str = ""
    supersede_targets: list[str] = field(default_factory=list)
    staging_tree_hash: str = ""
    dropped: bool = False

    def to_row(self) -> dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "step": self.step,
            "files": list(self.files),
            "variant_meta_text": self.variant_meta_text,
            "supersede_targets": list(self.supersede_targets),
            "staging_tree_hash": self.staging_tree_hash,
            "dropped": self.dropped,
        }

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "ImplementationRecord":
        return cls(
            proposal_id=str(row.get("proposal_id") or ""),
            step=str(row.get("step") or ""),
            files=list(row.get("files") or []),
            variant_meta_text=str(row.get("variant_meta_text") or ""),
            supersede_targets=list(row.get("supersede_targets") or []),
            staging_tree_hash=str(row.get("staging_tree_hash") or ""),
            dropped=bool(row.get("dropped")),
        )


def manifest_path(archive_root: Path | str) -> Path:
    return backlog_dir(archive_root) / MANIFEST_FILENAME


def _rows(archive_root: Path | str) -> list[dict[str, Any]]:
    try:
        text = manifest_path(archive_root).read_text()
    except FileNotFoundError:
        return []
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict) and row.get("proposal_id"):
            out.append(row)
    return out


def _append(archive_root: Path | str, record: ImplementationRecord) -> None:
    path = manifest_path(archive_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record.to_row(), ensure_ascii=False) + "\n")


def _latest(archive_root: Path | str) -> dict[str, ImplementationRecord]:
    """Last write per proposal wins; the log itself stays append-only."""
    by_id: dict[str, ImplementationRecord] = {}
    for row in _rows(archive_root):
        rec = ImplementationRecord.from_row(row)
        by_id[rec.proposal_id] = rec
    return by_id


def record_implementation(
    archive_root: Path | str,
    *,
    proposal_id: str,
    step: str = "",
    files: list[str] | None = None,
    variant_meta_text: str = "",
    supersede_targets: list[str] | None = None,
    staging_tree_hash: str = "",
) -> ImplementationRecord:
    rec = ImplementationRecord(
        proposal_id=proposal_id,
        step=step,
        files=list(files or []),
        variant_meta_text=variant_meta_text,
        supersede_targets=list(supersede_targets or []),
        staging_tree_hash=staging_tree_hash,
    )
    _append(archive_root, rec)
    return rec


def get_implementation(
    archive_root: Path | str, proposal_id: str
) -> ImplementationRecord | None:
    return _latest(archive_root).get(proposal_id)


def drop_implementation(
    archive_root: Path | str, proposal_id: str, *, step: str = ""
) -> ImplementationRecord | None:
    """Retract everything this proposal contributed, in one write.

    Files, variant_meta, supersede targets and the memory record go together —
    a half-retracted proposal is what retires an incumbent for a variant that no
    longer exists.
    """
    current = _latest(archive_root).get(proposal_id)
    if current is None:
        return None
    dropped = ImplementationRecord(
        proposal_id=proposal_id,
        step=step or current.step,
        files=[],
        variant_meta_text="",
        supersede_targets=[],
        staging_tree_hash="",
        dropped=True,
    )
    _append(archive_root, dropped)
    return dropped


def live_implementations(archive_root: Path | str) -> list[ImplementationRecord]:
    return [rec for rec in _latest(archive_root).values() if not rec.dropped]


def live_supersede_targets(archive_root: Path | str) -> list[str]:
    """Who may be retired — dropped proposals retire nobody."""
    out: list[str] = []
    for rec in live_implementations(archive_root):
        for target in rec.supersede_targets:
            if target not in out:
                out.append(target)
    return out


def live_variant_meta_text(archive_root: Path | str) -> str:
    """The variant_meta the archive sync may act on — dropped ones excluded."""
    return "\n".join(
        rec.variant_meta_text
        for rec in live_implementations(archive_root)
        if rec.variant_meta_text
    )
