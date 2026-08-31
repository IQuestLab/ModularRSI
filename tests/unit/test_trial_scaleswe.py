from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from harbor.environments.base import ExecResult
from harbor.models.task.config import TaskOS
from harbor.trial.trial import Trial


@pytest.mark.asyncio
async def test_materialize_verifier_workspace_preserves_only_fresh_venvs(
    tmp_path: Path,
) -> None:
    trial = object.__new__(Trial)
    trial._task = SimpleNamespace(
        config=SimpleNamespace(
            verifier_workspace_handoff_path="/workspace/repo",
            verifier_workspace_handoff_excludes=[".venv", "*/.venv"],
        )
    )
    environment = AsyncMock()
    environment.exec = AsyncMock(return_value=ExecResult(return_code=0))
    handoff = tmp_path / "handoff"
    handoff.mkdir()

    await trial._materialize_verifier_workspace(environment, handoff)

    assert environment.exec.await_count == 3
    preserve_command = environment.exec.await_args_list[0].args[0]
    reset_command = environment.exec.await_args_list[1].args[0]
    restore_command = environment.exec.await_args_list[2].args[0]
    assert "find . -type d -name .venv" in preserve_command
    assert "rm -rf -- /workspace/repo" in reset_command
    assert "tar -xf" in restore_command
    environment.upload_dir.assert_awaited_once_with(handoff, "/workspace/repo")


@pytest.mark.asyncio
async def test_required_startup_is_uploaded_and_checked(tmp_path: Path) -> None:
    start_path = tmp_path / "start.sh"
    start_path.write_text("#!/bin/bash\nexit 0\n")
    trial = object.__new__(Trial)
    trial._task = SimpleNamespace(
        config=SimpleNamespace(startup_required=True),
        paths=SimpleNamespace(start_path=start_path),
    )
    environment = AsyncMock()
    environment.task_os = TaskOS.LINUX
    environment.exec = AsyncMock(return_value=ExecResult(return_code=0))

    await trial._run_task_startup(environment)

    environment.upload_file.assert_awaited_once_with(
        start_path, "/tmp/harbor-task-start.sh"
    )
    assert "/tmp/harbor-task-start.sh" in environment.exec.await_args.args[0]
