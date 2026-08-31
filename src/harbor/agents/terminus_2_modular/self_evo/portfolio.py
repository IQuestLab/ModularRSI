"""Select at most one open proposal from each search lane.

Within a lane, deterministic FIFO-oriented ordering prevents a repeatedly
observed incumbent issue from starving a rarer capability proposal.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from harbor.agents.terminus_2_modular.self_evo import backlog as _backlog
from harbor.agents.terminus_2_modular.self_evo import proposals as _proposals

#: Lanes that compete for a slot in THIS module. `out_of_scope` is deliberately
#: absent — that work belongs to another module and must not take a slot here.
LANES = ("incumbent", "novelty")


@dataclass
class Pick:
    proposal_id: str
    lane: str
    proposal: Any
    rank_key: tuple
    reason: str


def _window_of(step: str) -> str:
    return step or ""


def _age_key(step: str) -> tuple:
    """Order `gen_<N>_candidate_<ts>` step ids by age, not by spelling.

    Compared as plain strings, `gen_10_…` sorts before `gen_2_…` — "1" beats
    "2" — so from the tenth generation on the NEWEST proposal reads as the one
    that has waited longest. This clause is half of the starvation bound, and
    it inverts exactly when a lineage gets long enough for starvation to
    matter.

    Digit runs compare numerically, text runs compare as text; the trailing
    timestamp still breaks ties inside one generation.
    """
    return tuple(
        (int(part), "") if part.isdigit() else (-1, part)
        for part in re.split(r"(\d+)", step or "")
        if part
    )


def _stable_jitter(seed: str, proposal_id: str) -> str:
    """A deterministic tie-break: same run + same proposal → same order."""
    return hashlib.sha1(f"{seed}\x1f{proposal_id}".encode("utf-8")).hexdigest()


def _rank_key(proposal: Any, windows_of: dict[str, set[str]], seed: str) -> tuple:
    """Lower sorts first. Negatives invert "more is better" into "smaller wins"."""
    unique_tasks = len(set(proposal.support_tasks))
    unique_windows = len(windows_of.get(proposal.proposal_id, set()))
    return (
        proposal.attempts,  # 1. never attempted first
        _age_key(proposal.created_step),  # 2. waiting longest first
        -unique_tasks,  # 3. broader task support first
        -unique_windows,  # 4. seen in more windows first
        _stable_jitter(seed, proposal.proposal_id),  # 5. stable random
    )


def _explain(proposal: Any, key: tuple) -> str:
    return (
        f"lane={proposal.lane} attempts={key[0]} since={key[1]} "
        f"tasks={-key[2]} windows={-key[3]}"
    )


def select_portfolio(
    archive_root: Path | str,
    *,
    max_lanes: int = 2,
    seed: str = "",
) -> list[Pick]:
    """Choose at most one open proposal per lane, at most ``max_lanes`` total.

    ``max_lanes=1`` keeps the portfolio ranking but builds only one change, so a
    defect in *selection* can be told apart from a defect in *running two in
    parallel*. ``max_lanes=0`` selects nothing.
    """
    if max_lanes <= 0:
        return []

    # unique support windows come from the ledger: a proposal stores which
    # findings support it, and each finding knows the window it was seen in.
    window_by_finding = {
        rec.finding_id: _window_of(rec.step)
        for rec in _backlog.load_findings(archive_root)
    }
    open_props = [
        p for p in _proposals.load_proposals(archive_root) if p.state == _proposals.OPEN
    ]
    windows_of: dict[str, set[str]] = {
        p.proposal_id: {
            window_by_finding[f] for f in p.finding_ids if f in window_by_finding
        }
        for p in open_props
    }

    best_per_lane: list[Pick] = []
    for lane in LANES:
        in_lane = [p for p in open_props if p.lane == lane]
        if not in_lane:
            continue  # a ceiling, not a quota — nothing is invented to fill it
        winner = min(in_lane, key=lambda p: _rank_key(p, windows_of, seed))
        key = _rank_key(winner, windows_of, seed)
        best_per_lane.append(
            Pick(
                proposal_id=winner.proposal_id,
                lane=lane,
                proposal=winner,
                rank_key=key,
                reason=_explain(winner, key),
            )
        )

    if len(best_per_lane) <= max_lanes:
        return best_per_lane
    # Trimming to fewer lanes than we have winners: keep the strongest by the
    # same key, so the reduced mode still exercises the real ranking.
    return sorted(best_per_lane, key=lambda pick: pick.rank_key)[:max_lanes]
