from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from harbor.agents.terminus_2_modular.self_evo import online_evo, state

pytestmark = pytest.mark.unit


def _field_names(cls) -> tuple[str, ...]:
    return tuple(field.name for field in dataclasses.fields(cls))


def test_reflection_archive_schema_is_stable():
    assert _field_names(state.ReflectionOutcome) == (
        "triggered",
        "promoted_gen",
        "discard_reason",
        "editor_n_edits",
        "editor_committed",
        "parent_mean_reward",
        "sanity_passed",
        "sanity_break_reason",
        "sanity_per_task",
        "sanity_bundles",
        "review_passed",
        "review_verdict",
        "review_reason",
        "solver_groups_during_reflection",
        "review_reject_class",
        "review_repair_brief",
        "candidate_gen_n",
        "intent",
        "variant_meta_text",
        "editor_trajectory_path",
        "files_changed",
        "sanity_activation",
        "gates_passed",
        "promotions",
    )


def test_task_progress_schema_is_stable():
    assert _field_names(state.TaskRunRecord) == (
        "task_idx",
        "task_name",
        "gen_used",
        "reward",
        "error",
        "trial_dir",
        "summary",
        "reflection",
        "roll",
        "bundle_id",
        "k_group_invalid_reason",
    )


def test_record_promotion_preserves_the_single_generation_view():
    outcome = state.ReflectionOutcome(True, None, None, intent="intent")
    promotion = outcome.record_promotion(
        Path("gen_1"),
        proposal_id="p1",
        files_changed=["agent_loop/example.py"],
    )
    assert outcome.promoted_gen == Path("gen_1")
    assert outcome.promotions == [promotion]
    assert dataclasses.asdict(promotion) == {
        "gen": Path("gen_1"),
        "proposal_id": "p1",
        "lane": "",
        "intent": "intent",
        "variant_meta_text": "",
        "files_changed": ["agent_loop/example.py"],
        "gates": {},
    }


def test_online_evo_keeps_compatibility_aliases():
    assert online_evo._Promotion is state.Promotion
    assert online_evo._ReflectionOutcome is state.ReflectionOutcome
    assert online_evo.TaskRunRecord is state.TaskRunRecord
    assert online_evo.OnlineEvoOutcome is state.OnlineEvoOutcome
