"""Append-only memory of attempted editor changes and later outcomes.

This memory is visible only to the editor and is never injected into evaluated
solver context.
"""

from __future__ import annotations

import json
from pathlib import Path

FILENAME = "editor_memory.jsonl"


def _path(archive_root: Path | str) -> Path:
    return Path(archive_root) / FILENAME


def record(
    archive_root: Path | str,
    *,
    epoch: int,
    module: str,
    change: str,
    task: str = "",
    gen: str = "",
    verdict: str = "provisional",
) -> None:
    """Append one experience row. Best-effort: never raise into the caller."""
    entry = {
        "epoch": epoch,
        "module": module,
        "task": task,
        "change": (change or "").strip()[:280],
        "gen": gen,
        "verdict": verdict,
    }
    try:
        p = _path(archive_root)
        p.parent.mkdir(parents=True, exist_ok=True)
        prior = p.read_text() if p.is_file() else ""
        p.write_text(prior + json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass


def load(archive_root: Path | str) -> list[dict]:
    try:
        text = _path(archive_root).read_text()
    except FileNotFoundError:
        return []
    except Exception:
        return []
    out: list[dict] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out


def compact_index(archive_root: Path | str, module: str, n: int = 12) -> str:
    """A small markdown block for the consolidator: the recent edits to `module`
    and a TABOO list (things tried that were rolled back / didn't help). Empty
    string when there's no history yet (nothing to inject)."""
    rows = [e for e in load(archive_root) if e.get("module") == module]
    if not rows:
        return ""
    recent = rows[-n:]
    lines = [
        "Recent changes to this module (most recent last):",
        "| epoch | gen | change | verdict |",
        "|---|---|---|---|",
    ]
    for e in recent:
        lines.append(
            f"| {e.get('epoch', '?')} | {e.get('gen', '')} | "
            f"{(e.get('change') or '')[:90]} | {e.get('verdict', '')} |"
        )
    taboo = [e for e in rows if e.get("verdict") in ("rolled_back", "no_help")]
    if taboo:
        lines.append("")
        lines.append("DO NOT re-propose these — already tried, did not help:")
        for e in taboo[-n:]:
            lines.append(f"- {(e.get('change') or '')[:110]}")
    return "\n".join(lines)
