"""Append-only ledger of contrastive findings.

Stable finding identifiers make ingestion idempotent across resume and allow
support for an intervention to accumulate across reflection windows.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

BACKLOG_DIRNAME = "backlog"
FINDINGS_FILENAME = "findings.jsonl"

_SEP = "\x1f"


def backlog_dir(archive_root: Path | str) -> Path:
    return Path(archive_root) / BACKLOG_DIRNAME


def findings_path(archive_root: Path | str) -> Path:
    return backlog_dir(archive_root) / FINDINGS_FILENAME


def finding_id(step: str, task: str, lens: str, index: int) -> str:
    """A content-derived ID, stable across re-ingests of the same step.

    Position is part of the identity on purpose: duplicates of (task, lens)
    inside one step are common in the real corpus, and collapsing them would
    silently merge two independent observations into one.
    """
    payload = _SEP.join((step or "", task or "", lens or "", str(index)))
    return "f_" + hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


@dataclass
class FindingRecord:
    finding_id: str
    step: str  # the reflection window this was observed in
    index: int
    task: str  # "" for lens findings — see support_tasks instead
    lens: str
    is_culprit: bool
    #: Whether `is_culprit` is a verdict at all — see trajectory_analysis's
    #: PARSE_* constants. PARSE_FAILED means the investigator's block could not
    #: be read, so `is_culprit=False` here is an absence of evidence and NOT a
    #: finding that the module is innocent. Defaults to "ok" so ledgers written
    #: before this field existed do not all read back as unreadable.
    parse_status: str = "ok"
    support_tasks: list[str] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)
    raw_finding: dict[str, Any] = field(default_factory=dict)

    def to_row(self) -> dict[str, Any]:
        return {
            "finding_id": self.finding_id,
            "step": self.step,
            "index": self.index,
            "task": self.task,
            "lens": self.lens,
            "is_culprit": self.is_culprit,
            "parse_status": self.parse_status,
            "support_tasks": list(self.support_tasks),
            "provenance": dict(self.provenance),
            "raw_finding": dict(self.raw_finding),
        }

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "FindingRecord":
        return cls(
            finding_id=str(row.get("finding_id") or ""),
            step=str(row.get("step") or ""),
            index=int(row.get("index") or 0),
            task=str(row.get("task") or ""),
            lens=str(row.get("lens") or ""),
            is_culprit=bool(row.get("is_culprit")),
            parse_status=str(row.get("parse_status") or "ok"),
            support_tasks=list(row.get("support_tasks") or []),
            provenance=dict(row.get("provenance") or {}),
            raw_finding=dict(row.get("raw_finding") or {}),
        )


def support_tasks_of(finding: dict[str, Any]) -> list[str]:
    """Which tasks this finding speaks for, across both finding schemas."""
    direct = finding.get("task")
    if isinstance(direct, str) and direct.strip():
        return [direct.strip()]
    out: list[str] = []
    for item in finding.get("evidence") or []:
        if not isinstance(item, dict):
            continue
        task = item.get("task")
        if isinstance(task, str) and task.strip() and task.strip() not in out:
            out.append(task.strip())
    return out


def _record_of(
    step: str, index: int, finding: dict[str, Any], provenance: dict[str, Any]
) -> FindingRecord:
    task = finding.get("task")
    task = task.strip() if isinstance(task, str) else ""
    lens = finding.get("lens")
    lens = lens.strip() if isinstance(lens, str) else ""
    return FindingRecord(
        finding_id=finding_id(step, task, lens, index),
        step=step,
        index=index,
        task=task,
        lens=lens,
        is_culprit=bool(finding.get("is_culprit")),
        parse_status=str(finding.get("parse_status") or "ok"),
        support_tasks=support_tasks_of(finding),
        provenance=dict(provenance or {}),
        raw_finding=dict(finding),
    )


def load_findings(archive_root: Path | str) -> list[FindingRecord]:
    """Read the ledger. A corrupt line costs that line, never the rest."""
    path = findings_path(archive_root)
    try:
        text = path.read_text()
    except FileNotFoundError:
        return []
    out: list[FindingRecord] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict) and row.get("finding_id"):
            out.append(FindingRecord.from_row(row))
    return out


def ingest_step(
    archive_root: Path | str,
    payload: dict[str, Any],
    *,
    provenance: dict[str, dict[str, Any]] | None = None,
) -> list[FindingRecord]:
    """Append this step's findings to the ledger; return all of its records.

    Idempotent by ``finding_id``: re-ingesting a step appends only the rows that
    are not already there, so a resumed run neither duplicates nor loses a
    finding that arrived late.

    ``provenance`` maps task name → citation dict (router bucket, contrast
    source, trial dirs). It is optional because the lens flow has no per-task
    routing context.
    """
    step = str(payload.get("step") or "")
    findings = payload.get("findings") or []
    prov_by_task = provenance or {}

    records: list[FindingRecord] = []
    for index, finding in enumerate(findings):
        if not isinstance(finding, dict):
            continue
        rec = _record_of(
            step, index, finding, prov_by_task.get(finding.get("task") or "")
        )
        records.append(rec)

    known = {r.finding_id for r in load_findings(archive_root)}
    fresh = [r for r in records if r.finding_id not in known]
    if fresh:
        path = findings_path(archive_root)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            for rec in fresh:
                handle.write(json.dumps(rec.to_row(), ensure_ascii=False) + "\n")
    return records
