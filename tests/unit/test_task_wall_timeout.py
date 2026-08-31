import asyncio
from collections import defaultdict
from pathlib import Path

import pytest
from pydantic import ValidationError

from harbor.job import Job
from harbor.metrics.mean import Mean
from harbor.models.job.config import JobConfig
from harbor.models.job.lock import build_job_lock
from harbor.models.trial.config import (
    AgentConfig,
    EnvironmentConfig,
    TaskConfig,
    TrialConfig,
    VerifierConfig,
)
from harbor.models.trial.result import TrialResult
from harbor.trial.trial import Trial
from tests.unit.test_trial_cleanup import HangingAgent, SlowStopEnvironment


def _create_task_dir(root: Path, name: str) -> Path:
    task_dir = root / name
    task_dir.mkdir()
    (task_dir / "task.toml").write_text(
        "[agent]\ntimeout_sec = 10.0\n[verifier]\ntimeout_sec = 10.0\n[environment]\n"
    )
    (task_dir / "instruction.md").write_text("Do nothing.")
    environment_dir = task_dir / "environment"
    environment_dir.mkdir()
    (environment_dir / "Dockerfile").write_text("FROM ubuntu:24.04\n")
    tests_dir = task_dir / "tests"
    tests_dir.mkdir()
    (tests_dir / "test.sh").write_text(
        "#!/bin/bash\necho 1 > /logs/verifier/reward.txt\n"
    )
    return task_dir


@pytest.mark.unit
def test_job_applies_wall_timeout_only_to_named_task(tmp_path: Path) -> None:
    target = TaskConfig(path=tmp_path / "sam-cell-seg")
    untouched = TaskConfig(path=tmp_path / "fix-git")
    config = JobConfig(
        job_name="task-wall-timeout",
        jobs_dir=tmp_path / "jobs",
        task_wall_timeouts_sec={"sam-cell-seg": 1800},
    )
    job = Job(
        config,
        _task_configs=[target, untouched],
        _metrics=defaultdict(lambda: [Mean()]),
    )
    try:
        by_name = {
            trial.task.get_task_id().get_name(): trial for trial in job._trial_configs
        }
        assert by_name["sam-cell-seg"].wall_timeout_sec == 1800
        assert by_name["fix-git"].wall_timeout_sec is None
        assert by_name["fix-git"].agent.max_timeout_sec is None
        assert by_name["fix-git"].verifier.max_timeout_sec is None
    finally:
        job._close_logger_handlers()


@pytest.mark.unit
def test_task_wall_timeout_must_be_positive() -> None:
    with pytest.raises(ValidationError, match="must be greater than zero"):
        JobConfig(task_wall_timeouts_sec={"sam-cell-seg": 0})


@pytest.mark.unit
def test_task_wall_timeout_is_recorded_in_job_lock() -> None:
    digest = f"sha256:{'a' * 64}"
    task = TaskConfig(name="test-org/sam-cell-seg", ref=digest)
    trial = TrialConfig(task=task, wall_timeout_sec=1800)
    lock = build_job_lock(
        config=JobConfig(tasks=[task]),
        trial_configs=[trial],
        invocation=["harbor", "run"],
    )

    assert lock.trials[0].wall_timeout_sec == 1800


@pytest.mark.unit
async def test_trial_wall_timeout_returns_explicit_result_and_cleans_up(
    tmp_path: Path,
) -> None:
    task_dir = _create_task_dir(tmp_path, "sam-cell-seg")
    trials_dir = tmp_path / "trials"
    trials_dir.mkdir()
    config = TrialConfig(
        task=TaskConfig(path=task_dir),
        trials_dir=trials_dir,
        wall_timeout_sec=0.05,
        agent=AgentConfig(import_path="tests.unit.test_trial_cleanup:HangingAgent"),
        environment=EnvironmentConfig(
            import_path="tests.unit.test_trial_cleanup:SlowStopEnvironment",
            delete=True,
        ),
        verifier=VerifierConfig(disable=True),
    )
    trial = await Trial.create(config)
    agent = trial._agent
    environment = trial._environment
    assert isinstance(agent, HangingAgent)
    assert isinstance(environment, SlowStopEnvironment)

    result = await asyncio.wait_for(trial.run(), timeout=1)

    assert result.exception_info is not None
    assert result.exception_info.exception_type == "TrialWallTimeoutError"
    assert "0.05 seconds" in result.exception_info.exception_message
    assert environment.stop_completed.is_set()
    persisted = TrialResult.model_validate_json(
        (trial.trial_dir / "result.json").read_text()
    )
    assert persisted.exception_info is not None
    assert persisted.exception_info.exception_type == "TrialWallTimeoutError"
