from __future__ import annotations

from types import SimpleNamespace

import pytest

from harbor.agents.terminus_2_modular.self_evo import phase0

pytestmark = pytest.mark.unit


def test_train_profile_matches_latest_phase0_scale():
    assert phase0.PROFILES["train"] == phase0.WorkloadProfile(
        reflect_every=10,
        task_concurrency=6,
        epochs=3,
        max_tasks=None,
    )


def test_smoke_profile_only_reduces_workload_scale():
    assert phase0.PROFILES["smoke"] == phase0.WorkloadProfile(
        reflect_every=2,
        task_concurrency=2,
        epochs=1,
        max_tasks=2,
    )
    assert phase0.SOLVER_TEMPERATURE == 0
    assert phase0.SOLVER_TIMEOUT_SEC == 3000
    assert phase0.AGENT_TIMEOUT_MULTIPLIER == 2.0


def test_public_parser_does_not_expose_retired_modes():
    help_text = phase0._build_parser().format_help()
    for retired in (
        "--kernel-evolution",
        "--reward-gate-mode",
        "--two-stage",
        "--no-multidim",
        "--reflect-mode",
        "--selection-mode",
        "--portfolio",
    ):
        assert retired not in help_text


def test_public_parser_accepts_one_or_two_lanes_only():
    parser = phase0._build_parser()
    required = [
        "--archive-root",
        "/tmp/archive",
        "--support-dataset-dir",
        "/tmp/support",
        "--locked-module",
        "tools",
        "--model",
        "openai/test-model",
    ]
    assert parser.parse_args(required).max_lanes == 2
    assert parser.parse_args(required + ["--max-lanes", "1"]).max_lanes == 1
    for invalid in ("0", "3"):
        with pytest.raises(SystemExit):
            parser.parse_args(required + ["--max-lanes", invalid])


def test_support_tasks_loads_numbered_manifest_from_pool(tmp_path):
    root = tmp_path / "support"
    for name in ("pool-a", "pool-b", "train-a", "train-b"):
        (root / "tasks" / name).mkdir(parents=True)
    (root / "manifest.json").write_text(
        '{"tasks": ['
        '{"index": 1, "name": "train-b", "split": "train"},'
        '{"index": 2, "name": "train-a", "split": "train"},'
        '{"index": 3, "name": "pool-a", "split": "pool"},'
        '{"index": 4, "name": "pool-b", "split": "pool"}'
        "]}"
    )

    task_dir, tasks = phase0._support_tasks(root)

    assert task_dir == (root / "tasks").resolve()
    assert tasks == ["train-b", "train-a"]
    assert phase0._support_tasks(root, "all")[1] == [
        "pool-a",
        "pool-b",
        "train-a",
        "train-b",
    ]


def test_support_tasks_rejects_invalid_train_list(tmp_path):
    root = tmp_path / "support"
    (root / "tasks" / "available").mkdir(parents=True)
    (root / "manifest.json").write_text(
        '{"tasks": ['
        '{"index": 1, "name": "missing", "split": "train"},'
        '{"index": 2, "name": "missing", "split": "train"}'
        "]}"
    )

    with pytest.raises(ValueError, match="duplicates"):
        phase0._support_tasks(root)


def test_main_pins_algorithm_semantics(monkeypatch, tmp_path):
    (tmp_path / "support" / "tasks" / "a").mkdir(parents=True)
    captured = {}

    async def fake_run(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(summary=lambda: "ok")

    monkeypatch.setattr(phase0, "run_online_evo", fake_run)
    monkeypatch.setenv("HARBOR_EVO_API_KEY", "test-key")

    rc = phase0.main(
        [
            "--archive-root",
            str(tmp_path / "archive"),
            "--support-dataset-dir",
            str(tmp_path / "support"),
            "--locked-module",
            "tools",
            "--model",
            "openai/test-model",
            "--profile",
            "smoke",
        ]
    )

    assert rc == 0
    for retired in (
        "reflect_mode",
        "selection_mode",
        "portfolio",
        "reward_gate_mode",
        "kernel_evolution",
        "multidim_analysis",
        "two_stage",
        "self_test_enabled",
        "reflect_on_pass",
        "block_during_reflection",
    ):
        assert retired not in captured
    assert captured["solver_temperature"] == 0
    assert captured["solver_timeout_sec"] == 3000
    assert captured["agent_timeout_multiplier"] == 2.0
    assert captured["reflect_every"] == 2
    assert captured["task_concurrency"] == 2
    assert captured["epochs"] == 1
    assert captured["max_tasks"] == 2
