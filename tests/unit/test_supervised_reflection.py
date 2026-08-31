"""Tests for the support-signal and review-blinding boundary."""

import json

import pytest

from harbor.agents.terminus_2_modular.self_evo import online_evo as OE
from harbor.agents.terminus_2_modular.self_evo import trajectory_analysis as TA
from harbor.agents.terminus_2_modular.self_evo.task_runner import HarborTaskResult


def _summary(**over) -> TA.TrialSummary:
    base = dict(
        task_name="fix-git",
        trial_name="fix-git__abc",
        reward=0.0,
        exception_type=None,
        exception_message=None,
        n_episodes=12,
        n_input_tokens=150_000,
        n_output_tokens=4_200,
        duration_sec=549.0,
        trial_dir="/tmp/trial",
    )
    base.update(over)
    return TA.TrialSummary(**base)


@pytest.mark.unit
def test_index_shows_ground_truth_and_efficiency():
    md = _summary().to_markdown()
    assert "**FAILED** (reward 0)" in md
    assert "agent time 549s" in md
    assert "tokens 150k in/4.2k out" in md
    md_pass = _summary(reward=1.0).to_markdown()
    assert "**PASSED** (reward 1)" in md_pass
    md_err = _summary(reward=None, exception_type="AgentTimeoutError").to_markdown()
    assert "**NO-SCORE**" in md_err and "AgentTimeoutError" in md_err


@pytest.mark.unit
def test_review_gate_index_is_reward_blind():
    s = _summary()
    md = s.to_markdown(include_reward=False)
    assert "FAILED" not in md and "PASSED" not in md and "reward" not in md
    instr = TA.build_review_instruction("diff", "intent", [s])
    assert "reward 0" not in instr
    # sanity: right template — the structured action, with an unparseable
    # (pipe-bearing) template so quoting it back can never count as a verdict
    assert "<review_verdict" in instr
    assert 'decision="accept|reject"' in instr


@pytest.mark.unit
@pytest.mark.unit
@pytest.mark.unit
def test_duration_extracted_from_result_json(tmp_path):
    trial = tmp_path / "fix-git__abc"
    trial.mkdir()
    (trial / "result.json").write_text(
        json.dumps(
            {
                "task_name": "fix-git",
                "trial_name": "fix-git__abc",
                "verifier_result": {"rewards": {"reward": 0.0}},
                "agent_execution": {
                    "started_at": "2026-07-07T11:18:58.000000Z",
                    "finished_at": "2026-07-07T11:28:07.000000Z",
                },
            }
        )
    )
    s = TA.summarize_trial(trial)
    assert s is not None
    assert s.duration_sec == pytest.approx(549.0)
    assert s.reward == 0.0


@pytest.mark.unit
def test_code_break_buildexception_is_infra_not_crash():
    hr = HarborTaskResult(task="compile-compcert", reward=None, error="harbor run rc=1")
    s = _summary(
        task_name="compile-compcert",
        reward=None,
        exception_type="BuildException",
        n_episodes=0,
    )
    is_break, reason = OE._is_code_break(hr, s)
    assert is_break is False
    assert "infra" in reason.lower()


@pytest.mark.unit
def test_code_break_ratelimit_and_sandbox_are_infra():
    for exc in ("RateLimitException", "SandboxException"):
        hr = HarborTaskResult(task="fix-git", reward=None, error="rc=1")
        s = _summary(task_name="fix-git", reward=None, exception_type=exc, n_episodes=0)
        assert OE._is_code_break(hr, s)[0] is False


@pytest.mark.unit
def test_code_break_real_module_crash_still_caught():
    # a genuine module-code crash: 0 episodes, NO infra exception → still a break
    hr = HarborTaskResult(task="fix-git", reward=None, error="rc=1")
    s = _summary(task_name="fix-git", reward=None, exception_type=None, n_episodes=0)
    is_break, reason = OE._is_code_break(hr, s)
    assert is_break is True
    assert "0 episodes" in reason


@pytest.mark.unit
def test_code_break_no_summary_but_build_error_is_infra():
    # build failed so early no result.json was written → summary None, but the
    # harbor error names the infra failure → still not a crash.
    hr = HarborTaskResult(
        task="compile-compcert",
        reward=None,
        error="harbor run rc=1; stderr: BuildException: build was cancelled",
    )
    is_break, reason = OE._is_code_break(hr, None)
    assert is_break is False and "infra" in reason.lower()


@pytest.mark.unit
def test_code_break_no_summary_unknown_error_still_break():
    # no summary + an error that is NOT a known infra marker → treat as a crash
    hr = HarborTaskResult(
        task="fix-git", reward=None, error="harbor run rc=1; stderr: ImportError foo"
    )
    assert OE._is_code_break(hr, None)[0] is True


# --- sanity gate: the changed module must be selected and traced ---------------


def _registering_file(path, names=("registered_loop",), type_="agent_loop"):
    """A REAL module file that registers `names` under `type_` — the pin target
    must be derived from what the changed code actually registers, so these
    tests use real register() calls, not a stubbed library."""
    lines = ["def register(library):"]
    for n in names:
        lines.append(
            f"    library.register({type_!r}, {n!r}, "
            "factory=lambda **kw: None, description='test variant')"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def _staging_with(tmp_path, filename, names=("registered_loop",)):
    staging = tmp_path / "staging"
    _registering_file(staging / "agent_loop" / filename, names=names)
    return staging


@pytest.mark.unit
def test_required_sanity_variant_name_comes_from_the_code(tmp_path):
    # file name ≠ registered name; NO variant_meta → the target is what the
    # changed file itself registers (the old stem-fallback couldn't do this)
    staging = _staging_with(tmp_path, "implementation_file.py")
    target = OE._required_sanity_variant(
        archive_root=tmp_path / "archive",
        staging_root=staging,
        files_changed=["NEW:agent_loop/implementation_file.py"],
        variant_meta_text="",
        locked_module_type="agent_loop",
    )
    assert target == ("agent_loop", "registered_loop")


@pytest.mark.unit
def test_required_sanity_variant_matching_meta_passes(tmp_path):
    staging = _staging_with(tmp_path, "implementation_file.py")
    target = OE._required_sanity_variant(
        archive_root=tmp_path / "archive",
        staging_root=staging,
        files_changed=["NEW:agent_loop/implementation_file.py"],
        variant_meta_text=(
            '<variant_meta name="registered_loop" type="agent_loop">\n'
            "CHANGE: add\n</variant_meta>"
        ),
        locked_module_type="agent_loop",
    )
    assert target == ("agent_loop", "registered_loop")


@pytest.mark.unit
def test_required_sanity_variant_rejects_lying_meta(tmp_path):
    """The evil repro: changed file registers `evil` but variant_meta claims
    `baseline` (which IS registered, by another file). The old code pinned
    baseline → gate passed while the changed code never ran. Now: refuse."""
    staging = _staging_with(tmp_path, "evilfile.py", names=("evil",))
    _registering_file(staging / "agent_loop" / "baseline.py", names=("baseline",))

    with pytest.raises(ValueError, match="actually registers"):
        OE._required_sanity_variant(
            archive_root=tmp_path / "archive",
            staging_root=staging,
            files_changed=["NEW:agent_loop/evilfile.py"],
            variant_meta_text=(
                '<variant_meta name="baseline" type="agent_loop">\n'
                "CHANGE: modify\n</variant_meta>"
            ),
            locked_module_type="agent_loop",
        )


@pytest.mark.unit
def test_required_sanity_variant_rejects_multi_registration_file(tmp_path):
    staging = _staging_with(tmp_path, "double.py", names=("one", "two"))
    with pytest.raises(ValueError, match="exactly one"):
        OE._required_sanity_variant(
            archive_root=tmp_path / "archive",
            staging_root=staging,
            files_changed=["NEW:agent_loop/double.py"],
            variant_meta_text="",
            locked_module_type="agent_loop",
        )


@pytest.mark.unit
def test_required_sanity_variant_rejects_retired_target(tmp_path):
    staging = _staging_with(tmp_path, "registered_loop.py")
    archive_root = tmp_path / "archive"
    archive_root.mkdir()
    OE._archive.save_archive(
        archive_root,
        [
            OE._archive.ArchiveEntry(
                name="registered_loop",
                type="agent_loop",
                born_gen="gen_2",
                status="superseded",
            )
        ],
    )

    with pytest.raises(ValueError, match="superseded"):
        OE._required_sanity_variant(
            archive_root=archive_root,
            staging_root=staging,
            files_changed=["agent_loop/registered_loop.py"],
            variant_meta_text="",
            locked_module_type="agent_loop",
        )


@pytest.mark.unit
def test_forced_sanity_source_pins_privately_and_preserves_params(tmp_path):
    source = tmp_path / "source"
    (source / "agent_loop").mkdir(parents=True)
    (source / "agent_loop" / "registered_loop.py").write_text("# implementation\n")
    original = {
        "agent_loop": {"name": "registered_loop", "params": {"limit": 7}},
        "observation": {"name": "baseline"},
    }
    (source / "active_bundle.json").write_text(json.dumps(original))

    forced = OE._prepare_forced_sanity_source(
        source,
        tmp_path / "forced",
        ("agent_loop", "registered_loop"),
    )

    forced_bundle = json.loads((forced / "active_bundle.json").read_text())
    assert forced_bundle["agent_loop"] == {
        "name": "registered_loop",
        "params": {"limit": 7},
    }
    assert forced_bundle["observation"] == {"name": "baseline"}
    assert json.loads((source / "active_bundle.json").read_text()) == original


@pytest.mark.unit
def test_required_activation_needs_selection_and_trace():
    target = ("agent_loop", "registered_loop")
    key = "agent_loop:registered_loop"

    assert "no sanity trajectory" in OE._required_activation_failure(None, target)
    selected_only = {
        "n_trials": 1,
        "variants": {key: {"selected": 1, "trace_calls": 0}},
    }
    assert "zero calls" in OE._required_activation_failure(selected_only, target)
    selected_and_called = {
        "n_trials": 1,
        "variants": {key: {"selected": 1, "trace_calls": 2}},
    }
    assert OE._required_activation_failure(selected_and_called, target) is None


@pytest.mark.unit
def test_collect_sanity_activation_expands_tool_helper_list(tmp_path):
    trajectory = tmp_path / "00_task" / "r0" / "trial" / "agent" / "trajectory.json"
    trajectory.parent.mkdir(parents=True)
    trajectory.write_text(
        json.dumps(
            {
                "agent": {
                    "extra": {
                        "bundle": {
                            "agent_loop": "registered_loop",
                            "tool_helper": ["grep", "patch"],
                        },
                        "module_trace": [
                            {"module": "agent_loop:registered_loop"},
                            {"module": "tool_helper:grep"},
                        ],
                    }
                }
            }
        )
    )

    activation = OE._collect_sanity_activation(tmp_path)
    assert activation is not None
    variants = activation["variants"]
    assert variants["tool_helper:grep"] == {"selected": 1, "trace_calls": 1}
    assert variants["tool_helper:patch"] == {"selected": 1, "trace_calls": 0}
    assert not any(key.startswith("tool_helper:[") for key in variants)


# --- review partial-accept: drop only the overfit file, keep the clean rest ---
# Observed 2026-07-15: a candidate bundled clean new file-editing tools with one
# overfit file (`if task_name == "fix-git"`); the all-or-nothing review gate
# discarded the whole bundle. Partial-accept reverts only the flagged file.


def _traj(steps: list[dict], tmp_path):
    p = tmp_path / "trajectory.json"
    p.write_text(json.dumps({"steps": steps}))
    return p


@pytest.mark.unit
def test_parse_review_drop_extracts_files(tmp_path):
    p = _traj(
        [
            {
                "source": "agent",
                "message": (
                    "VERDICT: ACCEPT — the new tools are a clean improvement.\n"
                    "DROP: agent_loop/verifier_feedback.py, tools/helpers/bad.py"
                ),
            }
        ],
        tmp_path,
    )
    assert TA.parse_review_drop(p) == [
        "agent_loop/verifier_feedback.py",
        "tools/helpers/bad.py",
    ]


@pytest.mark.unit
def test_parse_review_drop_empty_when_absent(tmp_path):
    p = _traj(
        [{"source": "agent", "message": "VERDICT: ACCEPT — all clean."}], tmp_path
    )
    assert TA.parse_review_drop(p) == []


@pytest.mark.unit
def test_parse_review_drop_normalizes_prefix_and_backticks(tmp_path):
    p = _traj(
        [
            {
                "source": "agent",
                "message": "VERDICT: ACCEPT\nDROP: `NEW:tools/helpers/grep.py`",
            }
        ],
        tmp_path,
    )
    assert TA.parse_review_drop(p) == ["tools/helpers/grep.py"]


@pytest.mark.unit
def test_parse_review_drop_ignores_format_placeholder(tmp_path):
    # a quote of the DROP format spec (placeholder, no real .py) → no drop
    p = _traj(
        [
            {
                "source": "agent",
                "message": "the harness reads a `DROP: <path>, <path>` line",
                "reasoning_content": "DROP: none",
            }
        ],
        tmp_path,
    )
    assert TA.parse_review_drop(p) == []


@pytest.mark.unit
def test_parse_review_drop_last_wins(tmp_path):
    p = _traj(
        [
            {"source": "agent", "message": "DROP: agent_loop/a.py"},
            {"source": "agent", "message": "DROP: tools/helpers/b.py"},
        ],
        tmp_path,
    )
    assert TA.parse_review_drop(p) == ["tools/helpers/b.py"]


@pytest.mark.unit
def test_apply_review_drops_deletes_new_restores_modified(tmp_path):
    parent = tmp_path / "parent"
    staging = tmp_path / "staging"
    (parent / "agent_loop").mkdir(parents=True)
    (staging / "agent_loop").mkdir(parents=True)
    (staging / "tools" / "helpers").mkdir(parents=True)
    # MODIFIED: in both, differs → restore parent
    (parent / "agent_loop" / "baseline.py").write_text("PARENT")
    (staging / "agent_loop" / "baseline.py").write_text("EDITED")
    # NEW: only in staging → delete
    (staging / "tools" / "helpers" / "grep.py").write_text("new tool")
    applied, skipped = OE._apply_review_drops(
        staging, parent, ["agent_loop/baseline.py", "tools/helpers/grep.py"]
    )
    assert set(applied) == {"agent_loop/baseline.py", "tools/helpers/grep.py"}
    assert skipped == []
    assert (staging / "agent_loop" / "baseline.py").read_text() == "PARENT"
    assert not (staging / "tools" / "helpers" / "grep.py").exists()


@pytest.mark.unit
def test_apply_review_drops_skips_traversal_and_missing(tmp_path):
    parent = tmp_path / "parent"
    staging = tmp_path / "staging"
    staging.mkdir()
    parent.mkdir()
    (tmp_path / "secret.py").write_text("SECRET")
    applied, skipped = OE._apply_review_drops(
        staging, parent, ["../secret.py", "tools/helpers/nope.py"]
    )
    assert applied == []
    assert (tmp_path / "secret.py").read_text() == "SECRET"  # traversal untouched
    assert set(skipped) == {"../secret.py", "tools/helpers/nope.py"}
