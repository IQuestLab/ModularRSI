"""Build contrastive task buckets and maintain per-task attempt history.

K same-code rolls distinguish stable passes, mixed outcomes, repairable failures,
and exhausted tasks. Exhaustion enters a cooldown and later receives a fresh
attempt window rather than a permanent blacklist.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path

from harbor.agents.terminus_2_modular.self_evo.trajectory_analysis import TrialSummary


class Bucket(str, Enum):
    ALL_PASS_EFFICIENT = "all_pass_efficient"
    ALL_PASS_WASTEFUL = "all_pass_wasteful"
    MIXED = "mixed"
    FIXABLE_FAIL = "fixable_fail"
    STUCK_FAIL = "stuck_fail"
    # Compatibility label: classify() no longer produces it; exhaustion goes
    # to STUCK_FAIL with a cooldown instead. Kept so old progress.json /
    # ledger records naming it still load.
    UNREACHABLE_FAIL = "unreachable_fail"


# Buckets whose tasks feed the contrastive diagnosis (they have, or can be given,
# a pass-vs-fail contrast). Anchors and the stuck/unreachable tail do NOT.
DIAGNOSE_BUCKETS = frozenset({Bucket.MIXED, Bucket.FIXABLE_FAIL})
ANCHOR_BUCKETS = frozenset({Bucket.ALL_PASS_EFFICIENT, Bucket.ALL_PASS_WASTEFUL})


@dataclass
class RouterConfig:
    k: int = 3
    # all-pass but "wasteful" if the passing rolls burn too many turns or thrash
    wasteful_episodes: int = 30
    wasteful_repeat_ratio: float = 0.5
    # a 0/K task that has been reflected on this many times WITHOUT progress goes
    # on cooldown for `cooldown_m` epochs (anti flip-flop / stop burning budget).
    stuck_n: int = 3
    cooldown_m: int = 2


def _is_pass(s: TrialSummary) -> bool:
    return s.reward is not None and s.reward >= 1.0


def _is_real_fail(s: TrialSummary) -> bool:
    # reward is None = infra/harbor error (no verifier verdict) — NOT a task
    # failure. Only a real numeric reward < 1.0 is a diagnosable failure. This is
    # the infra-vs-crash distinction: an e2b/endpoint hiccup must not masquerade
    # as a "fail" and send a contrast investigator chasing a non-existent bug.
    return s.reward is not None and s.reward < 1.0


@dataclass
class TaskOutcome:
    """The K rolls of ONE task in one encounter, plus the derived pass spread.

    reward=None rolls (infra errors) are dropped from the spread — they are not
    pass and not fail, so `n_valid` may be < k. Bucketing uses only valid rolls.
    """

    task: str
    rolls: list[TrialSummary]

    @property
    def k(self) -> int:
        return len(self.rolls)

    @property
    def passing(self) -> list[TrialSummary]:
        return [s for s in self.rolls if _is_pass(s)]

    @property
    def failing(self) -> list[TrialSummary]:
        return [s for s in self.rolls if _is_real_fail(s)]

    @property
    def n_pass(self) -> int:
        return len(self.passing)

    @property
    def n_valid(self) -> int:
        # rolls that carry signal (a real pass or a real fail); excludes infra None
        return len(self.passing) + len(self.failing)

    @property
    def all_pass(self) -> bool:
        return self.n_valid > 0 and len(self.failing) == 0

    @property
    def all_fail(self) -> bool:
        return self.n_valid > 0 and len(self.passing) == 0

    def is_wasteful(self, cfg: RouterConfig) -> bool:
        """All-pass but over budget: median turns too high, or heavy repetition.
        Only meaningful on the passing rolls (a fail's length is not 'waste')."""
        good = self.passing or self.rolls
        eps = [s.n_episodes for s in good if s.n_episodes]
        if eps and statistics.median(eps) > cfg.wasteful_episodes:
            return True
        rr = [s.repeated_command_ratio for s in good]
        return bool(rr) and max(rr) > cfg.wasteful_repeat_ratio


@dataclass
class LedgerEntry:
    task: str
    ever_passed: bool = False  # has this task passed in ANY roll/epoch so far
    times_reflected: int = 0  # how many edits have targeted this task's failure
    reflections_without_progress: int = 0
    best_progress: float = 0.0  # best "closeness" seen (0..1); 1.0 once it passes
    cooldown_until_epoch: int = -1  # skip reflection on this task until this epoch
    cooldown_cycles: int = 0  # how many exhaust→cooldown→renew cycles so far
    last_bucket: str = ""
    last_epoch: int = -1


class Ledger:
    """Per-task repair history, persisted to `<run_dir>/router_ledger.json`."""

    FILENAME = "router_ledger.json"

    def __init__(self, entries: dict[str, LedgerEntry] | None = None):
        self._e: dict[str, LedgerEntry] = entries or {}

    @classmethod
    def load(cls, run_dir: Path) -> "Ledger":
        p = Path(run_dir) / cls.FILENAME
        try:
            raw = json.loads(p.read_text())
            return cls({k: LedgerEntry(**v) for k, v in raw.items()})
        except FileNotFoundError:
            return cls()
        except Exception:
            # corrupt ledger must never kill a run — start fresh, keep the file
            return cls()

    def save(self, run_dir: Path) -> None:
        p = Path(run_dir) / self.FILENAME
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({k: asdict(v) for k, v in self._e.items()}, indent=2))

    def get(self, task: str) -> LedgerEntry:
        return self._e.get(task) or LedgerEntry(task=task)

    def _entry(self, task: str) -> LedgerEntry:
        if task not in self._e:
            self._e[task] = LedgerEntry(task=task)
        return self._e[task]

    def record_encounter(self, o: "TaskOutcome", bucket: "Bucket", epoch: int) -> None:
        """After classifying a task this epoch: update ever_passed / bucket."""
        e = self._entry(o.task)
        if o.n_pass > 0:
            e.ever_passed = True
            e.best_progress = 1.0
        e.last_bucket = bucket.value
        e.last_epoch = epoch

    def record_reflection(self, task: str, made_progress: bool) -> None:
        """After an edit targeted this task's failure: bump the try counter and
        the no-progress streak (used to decide cooldown)."""
        e = self._entry(task)
        e.times_reflected += 1
        if made_progress:
            e.reflections_without_progress = 0
        else:
            e.reflections_without_progress += 1

    def start_cooldown(self, task: str, epoch: int, cfg: RouterConfig) -> None:
        self._entry(task).cooldown_until_epoch = epoch + cfg.cooldown_m

    def renew_attempt_window(self, task: str) -> None:
        """Cooldown expiry grants a fresh attempt window.

        The no-progress streak resets so the task re-enters FIXABLE_FAIL —
        without this, the counters stay over threshold and the task bounces
        straight back into exhaustion (the permanent-blacklist bug).
        `times_reflected` is kept as lifetime history; `cooldown_cycles`
        counts how often this task has been through the cycle."""
        e = self._entry(task)
        e.reflections_without_progress = 0
        e.cooldown_until_epoch = -1
        e.cooldown_cycles += 1


def classify(
    o: TaskOutcome, entry: LedgerEntry, epoch: int, cfg: RouterConfig
) -> Bucket:
    """Pure bucket decision for one task's K-roll outcome + its ledger history."""
    if o.all_pass:
        return (
            Bucket.ALL_PASS_WASTEFUL
            if o.is_wasteful(cfg)
            else Bucket.ALL_PASS_EFFICIENT
        )
    if not o.all_fail:
        return Bucket.MIXED
    # ---- all-fail: decide fixable vs stuck (cooldown) ----
    if entry.cooldown_until_epoch > epoch:
        return Bucket.STUCK_FAIL  # still cooling down
    if entry.ever_passed:
        return Bucket.FIXABLE_FAIL  # proven achievable → keep trying
    tried_out = (
        entry.times_reflected >= cfg.stuck_n
        and entry.reflections_without_progress >= cfg.stuck_n
    )
    if tried_out:
        # N tries with zero progress: exhausted THIS window → cool down.
        # route_batch starts the cooldown clock; expiry renews the window
        # rather than a permanent unreachable blacklist.
        return Bucket.STUCK_FAIL
    return Bucket.FIXABLE_FAIL  # never tried enough → still worth a shot


@dataclass
class DiagnosisItem:
    """One task handed to the contrastive reflector, with its contrast source."""

    task: str
    bucket: str
    contrast_fail: TrialSummary  # the failing roll to explain
    contrast_pass: TrialSummary | None = None  # matched success, if any
    source: str = "none"  # same_task | archive_history | reference_solution | none


@dataclass
class RouteResult:
    buckets: dict[str, list[str]] = field(default_factory=dict)  # bucket -> tasks
    diagnose: list[DiagnosisItem] = field(default_factory=list)  # correctness work
    efficiency: list[DiagnosisItem] = field(default_factory=list)  # all_pass_wasteful
    anchors: list[str] = field(default_factory=list)  # all_pass → regression guard

    def summary(self) -> str:
        return " ".join(f"{b}={len(t)}" for b, t in sorted(self.buckets.items()) if t)


def route_batch(
    summaries: list[TrialSummary],
    *,
    ledger: Ledger,
    epoch: int,
    cfg: RouterConfig,
    history_pass_lookup=None,
) -> RouteResult:
    """Group a batch's rolls by task, bucket each, and emit the diagnosis work-list.

    `history_pass_lookup(task) -> TrialSummary | None` (optional): a matched
    passing trajectory from the archive/reference solution, used as the contrast
    partner for a `fixable_fail` task that had no pass THIS epoch.
    """
    by_task: dict[str, list[TrialSummary]] = {}
    for s in summaries:
        by_task.setdefault(s.task_name, []).append(s)

    res = RouteResult(buckets={b.value: [] for b in Bucket})
    res.buckets["infra_only"] = []
    for task, rolls in by_task.items():
        o = TaskOutcome(task=task, rolls=rolls)
        if o.n_valid == 0:
            # every roll was an infra/harbor error (reward=None) — no signal this
            # epoch. Don't diagnose or anchor; it'll be re-rolled next epoch.
            res.buckets["infra_only"].append(task)
            continue
        entry = ledger.get(task)
        # An expired cooldown grants a fresh attempt window before
        # classification — otherwise the over-threshold counters would send
        # the task straight back into exhaustion forever.
        if 0 <= entry.cooldown_until_epoch <= epoch:
            ledger.renew_attempt_window(task)
            entry = ledger.get(task)
        bucket = classify(o, entry, epoch, cfg)
        ledger.record_encounter(o, bucket, epoch)
        if bucket == Bucket.STUCK_FAIL and entry.cooldown_until_epoch < 0:
            # fresh exhaustion (not an already-running cooldown): start the
            # recovery clock now — the old code had no start_cooldown caller
            ledger.start_cooldown(task, epoch, cfg)
        res.buckets[bucket.value].append(task)

        if bucket in ANCHOR_BUCKETS:
            res.anchors.append(task)  # passes → do-no-harm regression guard
            if bucket == Bucket.ALL_PASS_WASTEFUL:
                # Also an efficiency candidate: passed, but the trajectory to
                # trim is one of its passing rolls. contrast_fail carries the
                # wasteful roll (the one to make lean); there is no fail to pair.
                res.efficiency.append(
                    DiagnosisItem(
                        task=task,
                        bucket=bucket.value,
                        contrast_fail=o.passing[0],
                        contrast_pass=None,
                        source="efficiency",
                    )
                )
        elif bucket == Bucket.MIXED:
            res.diagnose.append(
                DiagnosisItem(
                    task=task,
                    bucket=bucket.value,
                    contrast_fail=o.failing[0],
                    contrast_pass=o.passing[0],
                    source="same_task",
                )
            )
        elif bucket == Bucket.FIXABLE_FAIL:
            hist = history_pass_lookup(task) if history_pass_lookup else None
            res.diagnose.append(
                DiagnosisItem(
                    task=task,
                    bucket=bucket.value,
                    contrast_fail=o.failing[0],
                    contrast_pass=hist,
                    source="archive_history" if hist else "none",
                )
            )
        # STUCK_FAIL / UNREACHABLE_FAIL: skipped (no diagnosis, saves budget)
    return res
