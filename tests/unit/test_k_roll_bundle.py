"""One dynamic composition per same-task K-roll group."""

from __future__ import annotations

import json

import pytest

from harbor.agents.terminus_2_modular.self_evo.task_runner import HarborTaskResult
from harbor.agents.terminus_2_modular.self_evo.k_roll import (
    BundleManifest,
    extract_bundle_manifest,
    prepare_frozen_bundle_snapshot,
    run_same_bundle_k_rolls,
)
from harbor.agents.terminus_2_modular.self_evo import online_evo as OE


def _manifest(loop: str = "loop_v1") -> BundleManifest:
    return BundleManifest(
        modules={
            "agent_loop": loop,
            "observation": "obs_v1",
            "context_mgmt": "ctx_v1",
            "tools": "tools_v1",
            "verification": "verify_v1",
        },
        tool_helpers=("grep", "read_file"),
    )


def _write_trajectory(output_dir, manifest: BundleManifest) -> None:
    path = output_dir / "job" / "task__trial" / "agent" / "trajectory.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "agent": {
                    "extra": {
                        "bundle": manifest.as_dict(),
                    }
                }
            }
        )
    )


@pytest.mark.unit
def test_extract_bundle_manifest_includes_helper_subset(tmp_path):
    expected = _manifest()
    _write_trajectory(tmp_path, expected)
    actual = extract_bundle_manifest(tmp_path)
    assert actual == expected
    assert actual is not None
    assert len(actual.bundle_id) == 64


@pytest.mark.unit
def test_private_snapshot_does_not_touch_generation_pin(tmp_path):
    modules = tmp_path / "gen_2" / "modules"
    (modules / "agent_loop").mkdir(parents=True)
    (modules / "agent_loop" / "baseline.py").write_text("ORIGINAL = True\n")
    original_pin = {"agent_loop": {"name": "foundation"}}
    (modules / "active_bundle.json").write_text(json.dumps(original_pin))

    snapshot = prepare_frozen_bundle_snapshot(
        tmp_path / "gen_2", tmp_path / "group", _manifest()
    )

    assert json.loads((modules / "active_bundle.json").read_text()) == original_pin
    frozen = json.loads((snapshot.load_root / "active_bundle.json").read_text())
    assert frozen["agent_loop"]["name"] == "loop_v1"
    assert frozen["tools"]["params"]["helpers"] == ["grep", "read_file"]
    assert (snapshot.load_root / "agent_loop" / "baseline.py").is_file()


@pytest.mark.unit
def test_package_snapshot_preserves_package_layout(tmp_path):
    package = tmp_path / "gen_3"
    (package / "modules" / "agent_loop").mkdir(parents=True)
    (package / "modules" / "agent_loop" / "baseline.py").write_text("X = 1\n")
    (package / "protocols.py").write_text("# package marker\n")

    snapshot = prepare_frozen_bundle_snapshot(package, tmp_path / "group", _manifest())

    assert snapshot.load_root.name == "package"
    assert (snapshot.load_root / "protocols.py").is_file()
    assert (snapshot.load_root / "modules" / "active_bundle.json").is_file()


@pytest.mark.unit
def test_snapshot_pin_is_accepted_by_real_static_bundle_loader(tmp_path):
    from harbor.agents.terminus_2_modular import modules as modules_package
    from harbor.agents.terminus_2_modular.composer.bundle_config import (
        load_bundle_override,
    )
    from harbor.agents.terminus_2_modular.composer.static import DEFAULT_BUNDLE
    from harbor.agents.terminus_2_modular.library import build_default_library

    manifest = BundleManifest(
        modules={
            "agent_loop": "baseline",
            "observation": "baseline",
            "context_mgmt": "baseline",
            "tools": "baseline",
            "verification": "baseline",
        }
    )
    snapshot = prepare_frozen_bundle_snapshot(
        modules_package.__path__[0], tmp_path / "group", manifest
    )
    library = build_default_library(modules_root=snapshot.load_root)

    bundle = load_bundle_override(
        snapshot.load_root, library.list_infos(), DEFAULT_BUNDLE
    )

    assert bundle is not None
    assert bundle.agent_loop.name == "baseline"
    assert bundle.tools.params["helpers"] == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_k3_composes_once_and_reuses_exact_bundle(tmp_path):
    source = tmp_path / "gen_0" / "modules"
    (source / "agent_loop").mkdir(parents=True)
    (source / "agent_loop" / "baseline.py").write_text("X = 1\n")
    chosen = _manifest()
    calls: list[tuple[int, str]] = []

    async def run_roll(roll_idx, load_root, composer_name, output_dir):
        calls.append((roll_idx, composer_name))
        if composer_name == "llm_dynamic":
            observed = chosen
        else:
            active = json.loads((load_root / "active_bundle.json").read_text())
            observed = BundleManifest(
                modules={name: active[name]["name"] for name in chosen.modules},
                tool_helpers=tuple(active["tools"]["params"]["helpers"]),
            )
        _write_trajectory(output_dir, observed)
        return HarborTaskResult(task="t", reward=1.0, error=None)

    result = await run_same_bundle_k_rolls(
        attempts=3,
        source_root=source,
        group_root=tmp_path / "solver" / "task_000_t",
        run_roll=run_roll,
    )

    assert result.valid
    assert result.bundle_id == chosen.bundle_id
    assert len(result.rolls) == 3
    assert calls == [(0, "llm_dynamic"), (1, "static"), (2, "static")]
    assert [extract_bundle_manifest(output_dir) for _, output_dir in result.rolls] == [
        chosen,
        chosen,
        chosen,
    ]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_missing_first_bundle_invalidates_k_group(tmp_path):
    source = tmp_path / "modules"
    source.mkdir()
    calls = []

    async def run_roll(roll_idx, load_root, composer_name, output_dir):
        calls.append(roll_idx)
        return HarborTaskResult(task="t", reward=None, error="agent crashed")

    result = await run_same_bundle_k_rolls(
        attempts=3,
        source_root=source,
        group_root=tmp_path / "group",
        run_roll=run_roll,
    )

    assert not result.valid
    assert "no complete bundle" in (result.invalid_reason or "")
    assert calls == [0]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_k1_preserves_crash_without_requiring_manifest(tmp_path):
    source = tmp_path / "modules"
    source.mkdir()

    async def run_roll(roll_idx, load_root, composer_name, output_dir):
        return HarborTaskResult(task="t", reward=None, error="agent crashed")

    result = await run_same_bundle_k_rolls(
        attempts=1,
        source_root=source,
        group_root=tmp_path / "group",
        run_roll=run_roll,
    )

    assert result.valid
    assert len(result.rolls) == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_online_evo_training_path_uses_same_bundle_helper(tmp_path, monkeypatch):
    from harbor.agents.terminus_2_modular import modules as modules_package

    chosen = BundleManifest(
        modules={
            "agent_loop": "baseline",
            "observation": "baseline",
            "context_mgmt": "baseline",
            "tools": "baseline",
            "verification": "baseline",
        }
    )
    calls: list[tuple[str, float | None]] = []

    def fake_run(*, composer_name, output_dir, staging_modules_dir, task, **kwargs):
        calls.append((composer_name, kwargs.get("agent_timeout_multiplier")))
        if composer_name == "static":
            active = json.loads(
                (staging_modules_dir / "active_bundle.json").read_text()
            )
            observed = BundleManifest(
                modules={name: active[name]["name"] for name in chosen.modules},
                tool_helpers=tuple(active["tools"]["params"]["helpers"]),
            )
        else:
            observed = chosen
        _write_trajectory(output_dir, observed)
        return HarborTaskResult(task=task, reward=1.0, error=None)

    monkeypatch.setattr(OE, "run_harbor_task", fake_run)
    outcome = await OE.run_online_evo(
        archive_root=tmp_path / "archive",
        tasks=["support-task"],
        model_name="m",
        api_base=None,
        api_key=None,
        reflect_every=99,
        task_concurrency=1,
        max_tasks=1,
        shuffle_tasks=False,
        parent_modules_for_gen0=modules_package.__path__[0],
        locked_module_type="tools",
        attempts=3,
        agent_timeout_multiplier=2.0,
    )

    assert calls == [
        ("llm_dynamic", 2.0),
        ("static", 2.0),
        ("static", 2.0),
    ]
    assert len(outcome.records) == 3
    assert {record.bundle_id for record in outcome.records} == {chosen.bundle_id}
    assert all(record.k_group_invalid_reason is None for record in outcome.records)
    meta = json.loads((tmp_path / "archive" / "meta.json").read_text())
    assert meta["agent_timeout_multiplier"] == 2.0
