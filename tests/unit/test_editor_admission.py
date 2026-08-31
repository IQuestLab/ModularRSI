"""P0-4: editor admission partition.

Every editor session (investigator fan-out, consolidator, review, repair) goes
through run_editor — measured worst case was 29 CONCURRENT sessions from one
reflection's bare asyncio.gather, on the same endpoint the solver already
loads with cc10. The partition caps in-flight editor sessions via a global
slot pool (HARBOR_EDITOR_CONCURRENCY, default 2); solver keeps its own budget
at the runner-script level (8), so the endpoint total stays at 10.
"""

import asyncio

import pytest

from harbor.agents.terminus_2_modular.self_evo import run_editor as re_mod

pytestmark = pytest.mark.unit


async def _measure_max_concurrency(n_tasks: int) -> int:
    running = 0
    peak = 0

    async def one():
        nonlocal running, peak
        async with re_mod._editor_slot():
            running += 1
            peak = max(peak, running)
            await asyncio.sleep(0.01)
            running -= 1

    async with asyncio.TaskGroup() as tg:
        for _ in range(n_tasks):
            tg.create_task(one())
    return peak


async def test_default_limit_is_two(monkeypatch):
    monkeypatch.delenv("HARBOR_EDITOR_CONCURRENCY", raising=False)
    assert await _measure_max_concurrency(6) == 2


async def test_env_override(monkeypatch):
    monkeypatch.setenv("HARBOR_EDITOR_CONCURRENCY", "1")
    assert await _measure_max_concurrency(4) == 1


async def test_all_tasks_complete_despite_queueing(monkeypatch):
    # 29 was the measured real-world fan-out — everything must still finish
    monkeypatch.setenv("HARBOR_EDITOR_CONCURRENCY", "2")
    peak = await _measure_max_concurrency(29)
    assert peak == 2


async def test_garbage_env_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("HARBOR_EDITOR_CONCURRENCY", "not-a-number")
    assert await _measure_max_concurrency(5) == 2


# ---- sanity solver budget -------------------------------------------------


def _sanity_cap(monkeypatch, env, requested):
    from harbor.agents.terminus_2_modular.self_evo import online_evo as OE

    if env is None:
        monkeypatch.delenv("HARBOR_AUX_SOLVER_CONCURRENCY", raising=False)
    else:
        monkeypatch.setenv("HARBOR_AUX_SOLVER_CONCURRENCY", env)
    return OE._sanity_solver_concurrency(requested)


def test_aux_cap_defaults_to_two(monkeypatch):
    assert _sanity_cap(monkeypatch, None, 6) == 2


def test_aux_cap_never_raises_the_requested_concurrency(monkeypatch):
    assert _sanity_cap(monkeypatch, "4", 1) == 1


def test_aux_cap_env_override(monkeypatch):
    assert _sanity_cap(monkeypatch, "4", 6) == 4


def test_aux_cap_garbage_env_falls_back(monkeypatch):
    assert _sanity_cap(monkeypatch, "zero", 6) == 2
