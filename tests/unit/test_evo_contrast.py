"""Unit tests for the contrastive-diagnosis builders + parser. Pure, no endpoint."""

import json

import pytest

from harbor.agents.terminus_2_modular.self_evo.run_editor import resolve_model_info
from harbor.agents.terminus_2_modular.self_evo.trajectory_analysis import (
    TrialSummary,
    build_contrast_investigation_instruction,
    build_efficiency_investigation_instruction,
    parse_contrast_finding,
)

pytestmark = pytest.mark.unit


def _roll(task, reward, trial):
    return TrialSummary(
        task_name=task,
        trial_name=trial,
        reward=reward,
        exception_type=None,
        exception_message=None,
        n_episodes=3,
        n_input_tokens=0,
        n_output_tokens=0,
    )


def test_investigation_instruction_has_both_rolls_and_locked_module():
    p = _roll("triangular-sum", 1.0, "pass")
    f = _roll("triangular-sum", 0.0, "fail")
    instr = build_contrast_investigation_instruction(
        "agent_loop", "triangular-sum", f, p
    )
    assert "triangular-sum" in instr
    assert "`agent_loop`" in instr
    assert "PASSING roll" in instr and "FAILING roll" in instr
    assert "<contrast_finding>" in instr


def test_each_module_gets_its_own_lens():
    # Every locked module type gets a different tailored
    # yardstick, not one generic prompt
    p, f = _roll("t", 1.0, "p"), _roll("t", 0.0, "f")
    needles = {
        "agent_loop": "ReAct loop",
        "tools": "tmux tool",
        "observation": "truncated terminal",
        "context_mgmt": "summarizes chat",
        "verification": "two-phase self-assessment",
    }
    for module, needle in needles.items():
        instr = build_contrast_investigation_instruction(module, "t", f, p)
        assert needle in instr, f"{module} missing its own yardstick"
        assert "Yardstick" in instr and "probes" in instr.lower()


def test_investigation_instruction_handles_no_pass():
    f = _roll("t", 0.0, "fail")
    instr = build_contrast_investigation_instruction("tools", "t", f, None)
    assert "no passing roll available" in instr
    assert "NONE this epoch" in instr


def test_parse_contrast_finding_valid(tmp_path):
    block = {
        "task": "t",
        "is_culprit": True,
        "locked_module": "agent_loop",
        "divergence": "fail stopped after 1 try",
        "gap": "no retry",
        "fixable_now": True,
        "suggested_change": "add a retry loop",
    }
    traj = tmp_path / "trajectory.json"
    traj.write_text(
        json.dumps(
            {
                "steps": [
                    {
                        "source": "agent",
                        "message": "reasoning...\n<contrast_finding>\n```json\n"
                        + json.dumps(block)
                        + "\n```\n</contrast_finding>",
                    }
                ]
            }
        )
    )
    got = parse_contrast_finding(traj, "t")
    assert got["is_culprit"] is True
    assert got["gap"] == "no retry"
    assert got["lens"] == "agent_loop"  # so consolidator title lookup works


def test_parse_contrast_finding_null_on_missing():
    got = parse_contrast_finding(None, "t")
    assert got["is_culprit"] is False
    got2 = parse_contrast_finding_missing()
    assert got2["is_culprit"] is False


def parse_contrast_finding_missing():
    from pathlib import Path

    return parse_contrast_finding(Path("/tmp/does_not_exist_xyz/trajectory.json"), "t")


def test_parse_contrast_finding_last_block_wins(tmp_path):
    traj = tmp_path / "trajectory.json"
    a = json.dumps({"task": "t", "is_culprit": False})
    b = json.dumps({"task": "t", "is_culprit": True, "gap": "final"})
    traj.write_text(
        json.dumps(
            {
                "steps": [
                    {
                        "source": "agent",
                        "message": f"<contrast_finding>{a}</contrast_finding> "
                        f"then <contrast_finding>{b}</contrast_finding>",
                    }
                ]
            }
        )
    )
    got = parse_contrast_finding(traj, "t")
    assert got["is_culprit"] is True and got["gap"] == "final"


def test_efficiency_instruction_is_correctness_safe():
    r = _roll("slow", 1.0, "wasteful")
    instr = build_efficiency_investigation_instruction("agent_loop", "slow", r)
    assert "wasteful" in instr.lower()
    assert "`agent_loop`" in instr
    # must warn against trading correctness for speed
    assert "Correctness first" in instr or "correctness" in instr.lower()
    assert "<contrast_finding>" in instr  # reuses the same parser


def test_editor_uses_lineage_model_info(monkeypatch):
    profile = {
        "max_input_tokens": 512000,
        "max_output_tokens": 32000,
        "input_cost_per_token": 0.0,
        "output_cost_per_token": 0.0,
    }
    monkeypatch.setenv("HARBOR_MODEL_INFO", json.dumps(profile))
    assert resolve_model_info(None) == profile

    explicit = {"max_input_tokens": 10, "max_output_tokens": 2}
    assert resolve_model_info(explicit) is explicit
