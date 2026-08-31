from __future__ import annotations

import pytest

from harbor.agents.terminus_2_modular.self_evo.online_evo import (
    _restore_unreflected_progress,
)


def _summary(task: str, trial: str) -> dict:
    return {
        "task_name": task,
        "trial_name": trial,
        "reward": 0.0,
        "exception_type": None,
        "exception_message": None,
        "n_episodes": 3,
        "n_input_tokens": 100,
        "n_output_tokens": 20,
    }


@pytest.mark.unit
def test_restore_only_suffix_after_latest_reflection():
    progress = [
        {
            "task_idx": 3,
            "task_name": "already-reflected",
            "roll": 0,
            "summary": _summary("already-reflected", "old-r0"),
            "reflection": {"triggered": True},
            "k_group_invalid_reason": None,
        },
        {
            "task_idx": 7,
            "task_name": "pending-valid",
            "roll": 0,
            "summary": _summary("pending-valid", "pending-r0"),
            "reflection": None,
            "k_group_invalid_reason": None,
        },
        {
            "task_idx": 7,
            "task_name": "pending-valid",
            "roll": 1,
            "summary": _summary("pending-valid", "pending-r1"),
            "reflection": None,
            "k_group_invalid_reason": None,
        },
        {
            "task_idx": 8,
            "task_name": "pending-error",
            "roll": 0,
            "summary": None,
            "error": "endpoint failed",
            "trial_dir": "/tmp/pending-error",
            "reflection": None,
            "k_group_invalid_reason": None,
        },
        {
            "task_idx": 9,
            "task_name": "invalid-group",
            "roll": 0,
            "summary": _summary("invalid-group", "invalid-r0"),
            "reflection": None,
            "k_group_invalid_reason": "bundle mismatch",
        },
    ]

    summaries, task_count, anchor_idx = _restore_unreflected_progress(progress)

    assert task_count == 2
    assert anchor_idx == 4
    assert [s.trial_name for s in summaries] == [
        "pending-r0",
        "pending-r1",
        "failed_8_r0",
    ]
    assert summaries[-1].failure_signals == ["harbor_error"]


@pytest.mark.unit
def test_restore_all_progress_when_no_reflection_exists():
    progress = [
        {
            "task_idx": 0,
            "task_name": "first",
            "roll": 0,
            "summary": _summary("first", "first-r0"),
            "reflection": None,
            "k_group_invalid_reason": None,
        }
    ]

    summaries, task_count, anchor_idx = _restore_unreflected_progress(progress)

    assert task_count == 1
    assert anchor_idx == 0
    assert [s.trial_name for s in summaries] == ["first-r0"]


@pytest.mark.unit
def test_restore_nothing_when_latest_entry_is_reflected():
    progress = [
        {
            "task_idx": 0,
            "task_name": "done",
            "roll": 0,
            "summary": _summary("done", "done-r0"),
            "reflection": {"triggered": True},
            "k_group_invalid_reason": None,
        }
    ]

    assert _restore_unreflected_progress(progress) == ([], 0, None)
