from __future__ import annotations

import pytest

from harbor.agents.terminus_2_modular.self_evo import routing

pytestmark = pytest.mark.unit


def _raw(action: str, target: str, **extra):
    value = {
        "lane": "incumbent",
        "action": action,
        "target_variant": target,
        "behavioral_delta": "require evidence before completion",
        "rationale": "the named variant owns completion checks",
    }
    value.update(extra)
    return value


def _validate(raw):
    return routing.validate_routing(
        raw,
        active_quals=["agent_loop/baseline", "agent_loop/confirm_exit"],
        locked_type="agent_loop",
    )


def test_modify_of_baseline_is_rejected_with_replacement_guidance():
    with pytest.raises(routing.InvalidRouting) as exc:
        _validate(_raw("modify", "agent_loop/baseline"))
    assert "read-only" in str(exc.value).lower()
    assert "replace" in str(exc.value).lower()


def test_replace_of_baseline_is_allowed():
    decision = _validate(
        _raw(
            "replace",
            "agent_loop/baseline",
            supersedes=["agent_loop/baseline"],
        )
    )
    assert decision.action == "replace"


def test_modify_of_non_baseline_variant_is_allowed():
    assert _validate(_raw("modify", "agent_loop/confirm_exit")).action == "modify"


def test_read_only_rule_applies_to_every_module_type():
    with pytest.raises(routing.InvalidRouting):
        routing.validate_routing(
            _raw("modify", "tools/baseline"),
            active_quals=["tools/baseline"],
            locked_type="tools",
        )


def test_prompt_marks_baseline_and_recommends_replace():
    prompt = routing.build_routing_instruction(
        finding={"task": "t", "is_culprit": True},
        locked_type="agent_loop",
        active_variants=[
            ("agent_loop/baseline", "default loop"),
            ("agent_loop/confirm_exit", "two-phase completion"),
        ],
    )
    baseline_line = next(
        line for line in prompt.splitlines() if "agent_loop/baseline`" in line
    )
    writable_line = next(
        line for line in prompt.splitlines() if "agent_loop/confirm_exit`" in line
    )
    assert "read-only" in baseline_line.lower()
    assert "replace" in baseline_line.lower()
    assert "read-only" not in writable_line.lower()
