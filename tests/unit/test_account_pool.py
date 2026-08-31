"""Unit tests for the multi-account e2b sandbox router.

Covers the AccountPool slot allocator (per-account cap, back-pressure,
balance, null fallback) and the task runner's per-subprocess environment
override that routes one harbor run's sandbox to a specific account.
"""

import asyncio

import pytest

from harbor.agents.terminus_2_modular.self_evo import task_runner as task_runner
from harbor.agents.terminus_2_modular.self_evo.online_evo import (
    AccountPool,
    NullAccountPool,
)


@pytest.mark.unit
def test_account_pool_capacity_and_slot_creds():
    pool = AccountPool([("k1", "t1"), ("k2", "t2")], per_account_cap=3)
    assert pool.n_accounts == 2
    assert pool.per_account_cap == 3
    assert pool.capacity == 6

    async def one():
        async with pool.slot() as s:
            return (s.key, s.token)

    key, token = asyncio.run(one())
    assert (key, token) in {("k1", "t1"), ("k2", "t2")}


@pytest.mark.unit
def test_account_pool_caps_per_account_and_backpressures():
    async def run():
        pool = AccountPool([("k1", "t1"), ("k2", "t2")], per_account_cap=2)
        release = asyncio.Event()
        live: dict[str, int] = {}
        peak: dict[str, int] = {}

        async def hold():
            async with pool.slot() as s:
                live[s.key] = live.get(s.key, 0) + 1
                peak[s.key] = max(peak.get(s.key, 0), live[s.key])
                await release.wait()
                live[s.key] -= 1

        # 4 holders fill every slot (2 per account) and then block.
        holders = [asyncio.create_task(hold()) for _ in range(4)]
        await asyncio.sleep(0.05)
        assert sum(live.values()) == 4
        # never more than the per-account cap on any one account
        assert all(v <= 2 for v in peak.values())
        # both accounts were used (balance)
        assert set(peak) == {"k1", "k2"}

        # a 5th acquire must BLOCK (no free slot) — back-pressure, not a 429.
        got = asyncio.Event()

        async def fifth():
            async with pool.slot():
                got.set()

        t5 = asyncio.create_task(fifth())
        await asyncio.sleep(0.05)
        assert not got.is_set()

        # releasing the 4 lets the 5th proceed.
        release.set()
        await asyncio.gather(*holders)
        await asyncio.wait_for(t5, timeout=1.0)
        assert got.is_set()

    asyncio.run(run())


@pytest.mark.unit
def test_account_pool_spreads_below_cap():
    """At concurrency < per_account_cap the pool must still spread across
    accounts (round-robin token order), not pile every slot on account #1.
    Guards the account-blocked-ordering bug where a=all, b/c/d=idle."""

    async def run():
        pool = AccountPool(
            [("k1", "t1"), ("k2", "t2"), ("k3", "t3")], per_account_cap=18
        )
        # Hold 6 slots at once — far below the 18/account cap.
        release = asyncio.Event()
        seen: list[str] = []

        async def hold():
            async with pool.slot() as s:
                seen.append(s.key)
                await release.wait()

        holders = [asyncio.create_task(hold()) for _ in range(6)]
        await asyncio.sleep(0.05)
        # All 3 accounts used, evenly (2 each) — NOT 6 on k1.
        from collections import Counter

        dist = Counter(seen)
        assert set(dist) == {"k1", "k2", "k3"}, f"only used {set(dist)}"
        assert all(v == 2 for v in dist.values()), f"uneven: {dict(dist)}"
        release.set()
        await asyncio.gather(*holders)

    asyncio.run(run())


@pytest.mark.unit
def test_null_account_pool_yields_keyless_slot():
    async def one():
        async with NullAccountPool().slot() as s:
            return s.key, s.token

    assert asyncio.run(one()) == (None, None)
    assert NullAccountPool().capacity == 0


@pytest.mark.unit
def test_run_harbor_task_routes_e2b_key_into_subprocess_env(monkeypatch, tmp_path):
    captured: dict = {}

    class _FakeProc:
        returncode = 1  # non-zero → early return, skip result-parsing path
        stdout = ""
        stderr = "stub"

    def _fake_run(cmd, **kwargs):
        captured["env"] = kwargs.get("env")
        return _FakeProc()

    monkeypatch.setattr(task_runner.subprocess, "run", _fake_run)

    task_runner.run_harbor_task(
        task="fix-git",
        staging_modules_dir=tmp_path,
        model_name="m",
        api_base=None,
        api_key=None,
        output_dir=tmp_path / "out",
        environment="e2b",
        e2b_key="KEY_ABC",
        e2b_token="TOK_XYZ",
    )
    env = captured["env"]
    assert env is not None
    assert env["E2B_API_KEY"] == "KEY_ABC"
    assert env["E2B_ACCESS_TOKEN"] == "TOK_XYZ"
    # inherits the rest of the environment (not a bare 2-key dict)
    assert len(env) > 2


@pytest.mark.unit
def test_run_harbor_task_no_key_inherits_env(monkeypatch, tmp_path):
    captured: dict = {}

    class _FakeProc:
        returncode = 1
        stdout = ""
        stderr = "stub"

    def _fake_run(cmd, **kwargs):
        captured["env"] = kwargs.get("env")
        return _FakeProc()

    monkeypatch.setattr(task_runner.subprocess, "run", _fake_run)

    task_runner.run_harbor_task(
        task="fix-git",
        staging_modules_dir=tmp_path,
        model_name="m",
        api_base=None,
        api_key=None,
        output_dir=tmp_path / "out",
        environment="e2b",
    )
    # env=None → child inherits parent env exactly as before (today's behavior)
    assert captured["env"] is None


@pytest.mark.unit
def test_run_harbor_task_keeps_model_key_out_of_argv(monkeypatch, tmp_path):
    captured: dict = {}

    class _FakeProc:
        returncode = 1
        stdout = ""
        stderr = "stub"

    def _fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["env"] = kwargs.get("env")
        return _FakeProc()

    monkeypatch.setattr(task_runner.subprocess, "run", _fake_run)

    task_runner.run_harbor_task(
        task="fix-git",
        staging_modules_dir=tmp_path,
        model_name="m",
        api_base="http://model.invalid/v1",
        api_key="SECRET_MODEL_KEY",
        output_dir=tmp_path / "out",
    )

    assert "SECRET_MODEL_KEY" not in " ".join(captured["cmd"])
    assert captured["env"]["OPENAI_API_KEY"] == "SECRET_MODEL_KEY"


@pytest.mark.unit
def test_run_harbor_task_can_force_static_composer(monkeypatch, tmp_path):
    captured: dict = {}

    class _FakeProc:
        returncode = 1
        stdout = ""
        stderr = "stub"

    def _fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return _FakeProc()

    monkeypatch.setattr(task_runner.subprocess, "run", _fake_run)

    task_runner.run_harbor_task(
        task="fix-git",
        staging_modules_dir=tmp_path,
        model_name="m",
        api_base=None,
        api_key=None,
        output_dir=tmp_path / "out",
        composer_name="static",
    )

    joined = " ".join(captured["cmd"])
    assert "composer_name=static" in joined
