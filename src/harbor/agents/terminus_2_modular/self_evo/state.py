"""Persisted and user-visible state for the self-evolution engine.

Field names and defaults in this module are part of the archive compatibility
contract. Execution code may move independently, but these serialized shapes
must remain stable so existing runs can be resumed and inspected.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from harbor.agents.terminus_2_modular.self_evo.trajectory_analysis import TrialSummary


@dataclass
class Promotion:
    """One generation promoted by a reflection window."""

    gen: Path
    proposal_id: str = ""
    lane: str = ""
    intent: str = ""
    variant_meta_text: str = ""
    files_changed: list[str] = field(default_factory=list)
    gates: dict = field(default_factory=dict)


@dataclass
class ReflectionOutcome:
    """Result of one reflection window, including zero or more promotions."""

    triggered: bool
    promoted_gen: Path | None
    discard_reason: str | None
    editor_n_edits: int = 0
    editor_committed: bool = False
    parent_mean_reward: float | None = None
    sanity_passed: bool | None = None
    sanity_break_reason: str | None = None
    sanity_per_task: list[tuple[str, list[float | None]]] = field(default_factory=list)
    sanity_bundles: dict[str, dict] = field(default_factory=dict)
    review_passed: bool = False
    review_verdict: str | None = None
    review_reason: str | None = None
    solver_groups_during_reflection: int | None = None
    review_reject_class: str = "none"
    review_repair_brief: str = ""
    candidate_gen_n: int | None = None
    intent: str = ""
    variant_meta_text: str = ""
    editor_trajectory_path: Path | None = None
    files_changed: list[str] = field(default_factory=list)
    sanity_activation: dict | None = None
    gates_passed: bool = False
    promotions: list[Promotion] = field(default_factory=list)

    def record_promotion(
        self,
        gen: Path,
        *,
        proposal_id: str = "",
        lane: str = "",
        intent: str | None = None,
        variant_meta_text: str | None = None,
        files_changed: list[str] | None = None,
        gates: dict | None = None,
    ) -> Promotion:
        """Append a promotion and advance the compatibility single-gen view."""
        promotion = Promotion(
            gen=gen,
            proposal_id=proposal_id,
            lane=lane,
            intent=self.intent if intent is None else intent,
            variant_meta_text=(
                self.variant_meta_text
                if variant_meta_text is None
                else variant_meta_text
            ),
            files_changed=(
                list(self.files_changed)
                if files_changed is None
                else list(files_changed)
            ),
            gates=dict(gates or {}),
        )
        self.promotions.append(promotion)
        self.promoted_gen = gen
        return promotion


@dataclass
class TaskRunRecord:
    """One K-roll attempt and the generation and frozen bundle it used."""

    task_idx: int
    task_name: str
    gen_used: Path
    reward: float | None
    error: str | None
    trial_dir: Path | None
    summary: TrialSummary | None
    reflection: ReflectionOutcome | None = None
    roll: int = 0
    bundle_id: str | None = None
    k_group_invalid_reason: str | None = None


@dataclass
class OnlineEvoOutcome:
    """Final result returned by the online evolution runner."""

    archive_root: Path
    starting_gen: Path
    final_gen: Path
    n_tasks: int
    n_promotions: int
    records: list[TaskRunRecord] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "=" * 72,
            "  ONLINE EVO OUTCOME",
            f"  archive       : {self.archive_root}",
            f"  starting gen  : {self.starting_gen.name}",
            f"  final gen     : {self.final_gen.name}",
            f"  n_tasks       : {self.n_tasks}",
            f"  n_promotions  : {self.n_promotions}",
            "",
            "  per-task record:",
        ]
        for record in self.records:
            line = (
                f"  #{record.task_idx:>3} {record.task_name:<35s} "
                f"gen={record.gen_used.name:<12s} "
                f"reward={record.reward}"
            )
            if record.error:
                line += f" err={record.error[:60]}"
            lines.append(line)
            if record.reflection and record.reflection.triggered:
                reflection = record.reflection
                gate = ""
                if reflection.parent_mean_reward is not None:
                    gate = (
                        f" batch_in_sample={reflection.parent_mean_reward:.2f}"
                        "(ref-only)"
                    )
                if reflection.promoted_gen:
                    lines.append(
                        "       ↳ reflected → PROMOTED "
                        f"{reflection.promoted_gen.name} "
                        f"(n_edits={reflection.editor_n_edits}, "
                        f"review={reflection.review_verdict}, sanity ok){gate}"
                    )
                else:
                    lines.append(
                        "       ↳ reflected → no promote "
                        f"(committed={reflection.editor_committed}, "
                        f"edits={reflection.editor_n_edits}{gate}, "
                        f"reason={reflection.discard_reason or 'no edits'})"
                    )
        lines.append("=" * 72)
        return "\n".join(lines)
