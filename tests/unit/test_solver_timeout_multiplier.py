"""Tests for the native per-task agent timeout multiplier."""

from __future__ import annotations

import subprocess
import inspect

import pytest

from harbor.agents.terminus_2_modular.self_evo import task_runner
from harbor.agents.terminus_2_modular.self_evo import online_evo

pytestmark = pytest.mark.unit


def _capture(monkeypatch):
    seen: dict = {}

    def fake_run(cmd, **kwargs):
        seen["cmd"] = list(cmd)
        seen["timeout"] = kwargs.get("timeout")
        raise subprocess.TimeoutExpired(cmd, 1)

    monkeypatch.setattr(task_runner.subprocess, "run", fake_run)
    return seen


def _call(tmp_path, **extra):
    return task_runner.run_harbor_task(
        task="hello-world",
        staging_modules_dir=tmp_path / "modules",
        model_name="openai/test-model",
        api_base=None,
        api_key=None,
        output_dir=tmp_path / "out",
        **extra,
    )


def test_multiplier_reaches_harbor_command(monkeypatch, tmp_path):
    seen = _capture(monkeypatch)
    _call(tmp_path, agent_timeout_multiplier=2.0)
    command = seen["cmd"]
    index = command.index("--agent-timeout-multiplier")
    assert command[index + 1] == "2.0"


@pytest.mark.parametrize("multiplier", [None, 1.0])
def test_no_effective_multiplier_omits_flag(monkeypatch, tmp_path, multiplier):
    seen = _capture(monkeypatch)
    _call(tmp_path, agent_timeout_multiplier=multiplier)
    assert "--agent-timeout-multiplier" not in seen["cmd"]


def test_outer_timeout_remains_a_separate_control(monkeypatch, tmp_path):
    seen = _capture(monkeypatch)
    _call(tmp_path, timeout_sec=3000, agent_timeout_multiplier=2.0)
    assert seen["timeout"] == 3000
    assert "3000" not in seen["cmd"]


def test_sanity_battery_does_not_apply_solver_multiplier():
    source = inspect.getsource(online_evo._run_same_bundle_battery)
    assert "agent_timeout_multiplier" not in source
