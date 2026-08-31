"""Unit tests for cross-epoch confirm/rollback stats (pure, no archive)."""

import pytest

from harbor.agents.terminus_2_modular.self_evo.confirm import (
    find_regressions,
    paired_delta,
    variant_pass_stats,
)

pytestmark = pytest.mark.unit

MT = "agent_loop"


def s(task, variant, reward):
    return (task, {MT: variant, "tools": "tmux_full"}, reward)


def test_variant_pass_stats():
    samples = [s("a", "base", 1.0), s("a", "base", 0.0), s("b", "v2", 1.0)]
    st = variant_pass_stats(samples, MT)
    assert st["base"] == {"n_used": 2, "n_pass": 1, "pass_rate": 0.5}
    assert st["v2"]["pass_rate"] == 1.0


def test_paired_delta_shared_only():
    # task a: base passes, v2 fails ; task b: both run ; task c: only v2
    samples = [
        s("a", "base", 1.0),
        s("a", "v2", 0.0),
        s("b", "base", 1.0),
        s("b", "v2", 1.0),
        s("c", "v2", 0.0),  # not shared → ignored
    ]
    d = paired_delta(samples, MT, "v2", "base")
    assert d["shared_tasks"] == 2  # a, b (c is v2-only)
    assert d["baseline_rate"] == 1.0
    assert d["variant_rate"] == 0.5  # (0 on a, 1 on b)
    assert d["delta"] == -0.5


def test_paired_delta_none_when_no_overlap():
    samples = [s("a", "base", 1.0), s("b", "v2", 0.0)]
    assert paired_delta(samples, MT, "v2", "base") is None


def test_find_regressions_flags_clear_loser():
    samples = []
    # v2 loses badly on 3 shared tasks (baseline passes, v2 fails)
    for t in ["a", "b", "c"]:
        samples += [s(t, "base", 1.0), s(t, "v2", 0.0)]
    regs = find_regressions(samples, MT, "base", min_shared=3, margin=0.34)
    assert [r["variant"] for r in regs] == ["v2"]


def test_find_regressions_respects_min_shared():
    # only 2 shared tasks < min_shared=3 → not flagged even though worse
    samples = [
        s("a", "base", 1.0),
        s("a", "v2", 0.0),
        s("b", "base", 1.0),
        s("b", "v2", 0.0),
    ]
    assert find_regressions(samples, MT, "base", min_shared=3) == []


def test_find_regressions_keeps_comparable_variant():
    # v2 ties baseline → not a regression
    samples = []
    for t in ["a", "b", "c"]:
        samples += [s(t, "base", 1.0), s(t, "v2", 1.0)]
    assert find_regressions(samples, MT, "base", min_shared=3) == []


def test_baseline_never_flags_itself():
    samples = [s("a", "base", 0.0)] * 5
    assert find_regressions(samples, MT, "base", min_shared=1) == []


def test_infra_none_rolls_are_ignored():
    # v2's only "failures" are infra None → must NOT count against it
    samples = [
        s("a", "base", 1.0),
        s("a", "v2", None),  # infra — ignored
        s("a", "v2", 1.0),
        s("b", "base", 1.0),
        s("b", "v2", 1.0),
        s("c", "base", 1.0),
        s("c", "v2", 1.0),
    ]
    st = variant_pass_stats(samples, MT)
    assert st["v2"]["n_used"] == 3  # the None roll was dropped
    assert find_regressions(samples, MT, "base", min_shared=3) == []
