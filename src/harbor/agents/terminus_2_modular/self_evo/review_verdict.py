"""Persist execution-gated structured review verdicts.

A verdict exists only after the editor tool executes <review_verdict/> and
writes the validated record. Deliberation text is never parsed as a decision.
"""

from __future__ import annotations

import json
from pathlib import Path

_DECISIONS = ("accept", "reject")
_REJECT_CLASSES = ("proposal", "implementation")


def validate_attrs(attrs: dict[str, str]) -> str | None:
    """Return an error string for the reviewer, or None when valid.

    A format-spec template quoted back verbatim (``decision="accept|reject"``)
    fails here — which is exactly what keeps instruction echoes from ever
    counting as a verdict.
    """
    decision = (attrs.get("decision") or "").strip().lower()
    reject_class = (attrs.get("reject_class") or "").strip().lower()
    reason = (attrs.get("reason") or "").strip()

    if decision == "accept_partial":
        return (
            "accept_partial is not enabled yet (it needs the proposal "
            "manifest); for a partly-bad bundle use reject with "
            'reject_class="implementation" and name the bad file in repair_brief'
        )
    if decision not in _DECISIONS:
        return 'decision must be "accept" or "reject"'
    if not reason:
        return "reason is required (one sentence on the same tag)"
    if decision == "accept" and reject_class not in ("", "none"):
        return 'accept must not carry a reject_class (omit it or use "none")'
    if decision == "reject" and reject_class not in _REJECT_CLASSES:
        return (
            'reject requires reject_class="proposal" (the direction is wrong) '
            'or reject_class="implementation" (right direction, flawed code)'
        )
    return None


def _normalize(attrs: dict[str, str]) -> dict[str, str]:
    return {
        "decision": attrs["decision"].strip().lower(),
        "reject_class": (attrs.get("reject_class") or "none").strip().lower() or "none",
        "reason": (attrs.get("reason") or "").strip(),
        "repair_brief": (attrs.get("repair_brief") or "").strip(),
    }


VERDICT_FILENAME = ".review_verdict.json"


def write_verdict_file(
    staging_dir: Path | str, attrs: dict[str, str]
) -> dict[str, str]:
    """Persist an EXECUTED submission (caller has already validated).

    Written by the tool handler into the review staging; last submission wins
    (the reviewer may change its mind and resubmit). Returns the normalized
    verdict."""
    verdict = _normalize(attrs)
    p = Path(staging_dir) / VERDICT_FILENAME
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(verdict, indent=1, ensure_ascii=False))
    return verdict


def read_verdict_file(staging_dir: Path | str) -> dict[str, str] | None:
    """The executed verdict, or None (→ the caller records a skip).

    Re-validates on read: a corrupt/hand-edited file degrades to None, it
    never fabricates a decision."""
    p = Path(staging_dir) / VERDICT_FILENAME
    try:
        raw = json.loads(p.read_text())
    except Exception:
        return None
    if not isinstance(raw, dict) or validate_attrs(raw) is not None:
        return None
    return _normalize(raw)


# ---------------------------------------------------------------------------
# Run-level review health / circuit breaker
#
# A session with no structured verdict is `review_skipped_parse_failure`: the
# candidate passes through ONCE (lenient, like the existing review-errored
# path) — but a run where reviews systematically fail to submit must not
# silently wave every candidate through. Trip on 3 consecutive skips, or on
# >10% skips once at least 10 reviews exist (the floor keeps 1/1 from
# tripping — a single skip IS the sanctioned pass-through).
# ---------------------------------------------------------------------------

HEALTH_FILENAME = "review_health.json"


def record_review_outcome(run_dir: Path | str, *, skipped: bool) -> dict:
    """Bump the run's review counters on disk; returns the updated health."""
    p = Path(run_dir) / HEALTH_FILENAME
    health = {"total_reviews": 0, "total_skips": 0, "consecutive_skips": 0}
    try:
        health.update(json.loads(p.read_text()))
    except Exception:
        pass  # first review of the run, or a corrupt file — start fresh
    health["total_reviews"] += 1
    if skipped:
        health["total_skips"] += 1
        health["consecutive_skips"] += 1
    else:
        health["consecutive_skips"] = 0
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(health, indent=1))
    return health


def breaker_tripped(health: dict) -> bool:
    if health.get("consecutive_skips", 0) >= 3:
        return True
    total = health.get("total_reviews", 0)
    return total >= 10 and health.get("total_skips", 0) / total > 0.10
