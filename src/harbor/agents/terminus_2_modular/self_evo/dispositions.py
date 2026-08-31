"""M3 — the finding disposition ledger: what became of a finding that never
became a proposal.

:mod:`proposals` answers one question about a finding: which proposal is it
attached to. Everything else the backlog forgets. A finding routed
``out_of_scope``, absorbed by a successor, aimed at a lineage that died, or
refused as malformed leaves an audit line and no state — and the resume check
asks only "attached to a proposal?". So on the next window it is routed again,
and the window after that, each time paying a routing call to re-reach a verdict
already reached.

``out_of_scope`` loses more than money. The router is *required* to name the
module that does own the problem (:mod:`routing` refuses the lane otherwise),
and that name was being dropped. A lineage locks one module, so the module named
here belongs to a different run — which makes this file the only way the
observation survives the process that made it.

Two rules keep the ledger honest:

* **Only verdicts are terminal.** ``out_of_scope`` / ``covered`` / ``stale`` /
  ``invalid`` are judgements about the finding. A failed comparison or a reply
  cut off at the token ceiling is endpoint weather; filing those as terminal
  would discard real evidence because of a bad afternoon, which is much worse
  than paying for the retry. They are recorded as ``retryable`` — visible, but
  not blocking.
* **Recording is idempotent.** Resume re-ingests the same step, and a ledger
  that grew a second row per resume would misreport how much evidence this run
  actually spent.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from harbor.agents.terminus_2_modular.self_evo.backlog import backlog_dir

_logger = logging.getLogger(__name__)

DISPOSITIONS_FILENAME = "dispositions.jsonl"
OUT_OF_SCOPE_DIRNAME = "out_of_scope"

#: Verdicts about the finding — never route it again.
OUT_OF_SCOPE = "out_of_scope"
COVERED = "covered"
STALE = "stale"
INVALID = "invalid"
#: Not a verdict: the endpoint could not answer. Route it again next window.
RETRYABLE = "retryable"

TERMINAL_KINDS = frozenset({OUT_OF_SCOPE, COVERED, STALE, INVALID})


@dataclass
class Disposition:
    finding_id: str
    kind: str
    step: str = ""
    module: str = ""  # out_of_scope: who owns it
    successor: str = ""  # covered: which variant absorbed it
    reason: str = ""


def dispositions_path(archive_root: Path | str) -> Path:
    return backlog_dir(archive_root) / DISPOSITIONS_FILENAME


def module_backlog_path(archive_root: Path | str, module: str) -> Path:
    """Where findings this run cannot act on are left for the run that can."""
    return backlog_dir(archive_root) / OUT_OF_SCOPE_DIRNAME / f"{module}.jsonl"


def load(archive_root: Path | str) -> list[Disposition]:
    """Every disposition, in the order recorded. Never raises.

    A malformed line is skipped rather than fatal: this file is read on every
    resume, and one bad append must not cost a run.
    """
    path = dispositions_path(archive_root)
    if not path.exists():
        return []
    out: list[Disposition] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:
            _logger.warning("disposition ledger: skipping unreadable line")
            continue
        if not isinstance(row, dict) or not row.get("finding_id"):
            continue
        out.append(
            Disposition(
                finding_id=str(row.get("finding_id", "")),
                kind=str(row.get("kind", "")),
                step=str(row.get("step", "")),
                module=str(row.get("module", "")),
                successor=str(row.get("successor", "")),
                reason=str(row.get("reason", "")),
            )
        )
    return out


def terminal_finding_ids(archive_root: Path | str) -> set[str]:
    """Findings whose verdict is in; routing them again buys nothing."""
    return {d.finding_id for d in load(archive_root) if d.kind in TERMINAL_KINDS}


def out_of_scope_for(archive_root: Path | str, module: str) -> list[str]:
    """Findings this run observed but another module owns."""
    return [
        d.finding_id
        for d in load(archive_root)
        if d.kind == OUT_OF_SCOPE and d.module == module
    ]


def record(
    archive_root: Path | str,
    *,
    finding_id: str,
    kind: str,
    step: str = "",
    module: str = "",
    successor: str = "",
    reason: str = "",
) -> None:
    """Append one disposition. Idempotent per (finding, kind). Never raises."""
    if not finding_id or not kind:
        return
    try:
        if any(
            d.finding_id == finding_id and d.kind == kind for d in load(archive_root)
        ):
            return
        row = {
            "finding_id": finding_id,
            "kind": kind,
            "step": step,
            "module": module,
            "successor": successor,
            "reason": reason,
        }
        path = dispositions_path(archive_root)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        if kind == OUT_OF_SCOPE and module:
            # Also as its own file, so the run that locks `module` has a
            # predictable path to point at instead of having to know this
            # ledger's schema.
            hand_off = module_backlog_path(archive_root, module)
            hand_off.parent.mkdir(parents=True, exist_ok=True)
            with hand_off.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    except OSError as exc:  # a ledger write must never cost a generation
        _logger.warning("could not record disposition for %s: %s", finding_id, exc)
