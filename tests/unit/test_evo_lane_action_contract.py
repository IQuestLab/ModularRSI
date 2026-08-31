from __future__ import annotations

from pathlib import Path

import pytest

from harbor.agents.terminus_2_modular.self_evo import backlog, dual_implement, proposals

pytestmark = pytest.mark.unit


def _brief_for(tmp_path: Path, action: str, target: str = "") -> str:
    finding_id = backlog.ingest_step(
        tmp_path,
        {"step": "w1", "findings": [{"task": "t", "is_culprit": True}]},
    )[0].finding_id
    proposal = proposals.create_proposal(
        tmp_path,
        step="w1",
        lane="novelty" if action == "add" else "incumbent",
        action=action,
        target_variant=target,
        behavioral_delta="require evidence before completion",
        causal_hypothesis="the loop accepts an unsupported completion signal",
        finding_ids=[finding_id],
        support_tasks=["t"],
    )
    return dual_implement.build_brief(
        tmp_path, proposal, tmp_path / "staging"
    ).instruction


def test_modify_names_the_file_to_edit(tmp_path):
    brief = _brief_for(tmp_path, "modify", "agent_loop/confirm_exit")
    assert "agent_loop/confirm_exit.py" in brief
    assert "# How to work" in brief


def test_replace_requires_supersedes_and_a_new_variant(tmp_path):
    brief = _brief_for(tmp_path, "replace", "tools/baseline")
    assert "SUPERSEDES: tools/baseline" in brief
    assert "new variant" in brief.lower()


def test_add_creates_a_variant_without_supersedes(tmp_path):
    brief = _brief_for(tmp_path, "add")
    assert "new variant" in brief.lower()
    assert "SUPERSEDES" not in brief


def test_action_obligation_is_in_work_steps(tmp_path):
    brief = _brief_for(tmp_path, "modify", "agent_loop/confirm_exit")
    assert brief.index("agent_loop/confirm_exit.py") > brief.index("# How to work")
