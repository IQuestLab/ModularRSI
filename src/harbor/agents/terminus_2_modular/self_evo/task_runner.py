"""Run one Harbor task against a frozen modules directory."""

from __future__ import annotations

import json
import logging
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path


_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step results
# ---------------------------------------------------------------------------


@dataclass
class HarborTaskResult:
    task: str
    reward: float | None
    error: str | None


# ---------------------------------------------------------------------------
# Harbor invocation
# ---------------------------------------------------------------------------


# A4: terminal-bench solvers via litellm need the *served* model's context
# window / output cap, else litellm falls back to wrong defaults and Terminus's
# context manager mis-fires (early/late summarization, wrong max_output).
# Keep the historical conservative fallback, but let each launcher pin the
# actual served model out-of-band. This is read at process import time, so one
# online-evo lineage has one immutable model profile without putting it in every
# subprocess call site.
_FALLBACK_MODEL_INFO = (
    '{"max_input_tokens": 176000, "max_output_tokens": 16000, '
    '"input_cost_per_token": 0.0, "output_cost_per_token": 0.0}'
)
DEFAULT_MODEL_INFO = os.environ.get("HARBOR_MODEL_INFO", _FALLBACK_MODEL_INFO)


def run_harbor_task(
    task: str,
    staging_modules_dir: Path,
    model_name: str,
    api_base: str | None,
    api_key: str | None,
    output_dir: Path,
    max_turns: int = 50,
    timeout_sec: int | None = None,
    agent_timeout_multiplier: float | None = None,
    model_info: str | None = DEFAULT_MODEL_INFO,
    llm_call_timeout_sec: int | None = 600,
    environment: str | None = None,
    task_dir: Path | None = None,
    e2b_key: str | None = None,
    e2b_token: str | None = None,
    temperature: float | None = None,
    locked_module_type: str | None = None,
    composer_scope: str | None = None,
    composer_name: str | None = None,
) -> HarborTaskResult:
    """Invoke `harbor run` with --ak modules_root pointing at the candidate
    gen. Read TRIAL-level reward from the per-trial result.json (more
    granular and consistent with trajectory_analysis.summarize_trial which
    also reads trial-level).

    timeout_sec is an *outer* wall-clock cap on the whole `harbor run`
    subprocess (build + agent + verify). Default None = no outer cap → let
    Harbor's native per-task timeout (task.toml `agent.timeout_sec`, which is
    900–12000s depending on the task) decide. A blanket cap here (e.g. 1800s)
    strangles heavy tasks whose native budget is much larger and, worse, kills
    the process before the verifier runs → reward=None instead of a real 0/1.

    """
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd: list[str] = [
        "harbor",
        "run",
        "-a",
        "terminus-2-modular",
        "-m",
        model_name,
    ]
    # Task source: a local task directory (-p, e.g. a v3 training-set task) or a
    # named task from the tb2 registry (-i … -d terminal-bench@2.0). This is the
    # train/eval split — support tasks come from the local dataset, held-out
    # probe/sanity/parent tasks come from tb2.
    if task_dir is not None:
        cmd += ["-p", str(Path(task_dir).resolve())]
    else:
        cmd += ["-i", task, "-d", "terminal-bench@2.0"]
    cmd += [
        "-n",
        "1",
        "-o",
        str(output_dir),
    ]
    cmd += ["--ak", f"modules_root={staging_modules_dir.resolve()}"]
    cmd += ["--ak", f"max_turns={max_turns}"]
    # Execution environment for the task container. Default (None) = harbor's
    # docker env (honors REMOTE_DOCKER_HOST). Pass e.g. "e2b" to run on E2B.
    if environment:
        cmd += ["-e", environment]
    if api_base:
        cmd += ["--ak", f"api_base={api_base}"]
    # Never put credentials in argv: Harbor persists argv-derived agent kwargs
    # in config/lock/result files and the evolution logger records the command.
    # LiteLLM natively reads OPENAI_API_KEY from the child environment.
    # A4: pass the served model's context window / output cap to litellm.
    if model_info:
        cmd += ["--ak", f"model_info={model_info}"]
    # A3: per-LLM-call timeout (forwarded to litellm) so one wedged request
    # fails fast and the agent retries, instead of a single hung call eating
    # the whole (now-native, up to 12000s) task budget once the outer cap is off.
    if llm_call_timeout_sec:
        cmd += ["--ak", f'llm_call_kwargs={{"timeout": {llm_call_timeout_sec}}}']
    # This scales Harbor's native per-task agent wall. The outer timeout above
    # caps the whole subprocess and is intentionally a separate control.
    if agent_timeout_multiplier and agent_timeout_multiplier != 1.0:
        cmd += ["--agent-timeout-multiplier", str(agent_timeout_multiplier)]
    # Rollout sampling temperature. The solver's LiteLLM ctor accepts a
    # first-class `temperature` agent kwarg (kernel/orchestration.py). K-roll
    # contrast needs temperature>0 so the K rolls of one task can diverge into
    # pass/fail — at temperature 0 every roll is identical and there is no
    # same-task contrast to learn from.
    if temperature is not None:
        cmd += ["--ak", f"temperature={temperature}"]
    # Module lock: name the one module type this experiment may WRITE to.
    if locked_module_type is not None:
        cmd += ["--ak", f"locked_module_type={locked_module_type}"]
    # How far the per-task composer may range. "locked" (default) = only the
    # locked type is picked among its variants, everything else sits on its
    # library default. "all" = every type is picked normally, lock or no lock —
    # SERIAL evolution needs this so a lineage inherited from a previous one can
    # actually USE the variants it inherited (see composer/llm_dynamic.py).
    if composer_scope is not None:
        cmd += ["--ak", f"composer_scope={composer_scope}"]
    # K-roll bundle freezing: roll 0 uses the normal dynamic composer; the
    # remaining rolls load a private active_bundle.json through StaticComposer.
    # None preserves every existing caller's default.
    if composer_name is not None:
        cmd += ["--ak", f"composer_name={composer_name}"]

    _logger.info("Harbor: %s", " ".join(cmd))
    # Per-subprocess E2B account override: when the caller hands us a specific
    # account key (from the AccountPool in online_evo), route THIS harbor run's
    # sandbox to that account instead of the single inherited E2B_API_KEY. env
    # stays None (= inherit parent env, today's behavior) when no key is given.
    run_env = None
    if api_key:
        run_env = {**os.environ, "OPENAI_API_KEY": api_key}
    if e2b_key:
        run_env = {
            **(run_env or os.environ),
            "E2B_API_KEY": e2b_key,
        }
        if e2b_token:
            run_env["E2B_ACCESS_TOKEN"] = e2b_token
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout_sec, env=run_env
        )
    except subprocess.TimeoutExpired:
        return HarborTaskResult(task=task, reward=None, error="timeout")
    if proc.returncode != 0:
        return HarborTaskResult(
            task=task,
            reward=None,
            error=f"harbor run rc={proc.returncode}; stderr: {proc.stderr[-500:]}",
        )

    # Find latest job dir, then the single trial inside it.
    # TODO(harbor-jobid): parse --job-id from harbor stdout instead of mtime.
    job_dirs = sorted(
        [d for d in output_dir.iterdir() if d.is_dir()],
        key=lambda p: p.stat().st_mtime,
    )
    if not job_dirs:
        return HarborTaskResult(task=task, reward=None, error="no job dir")
    job_dir = job_dirs[-1]

    # Use trial-level result.json (path consistent with trajectory_analysis)
    trial_dirs = sorted([d for d in job_dir.iterdir() if d.is_dir() and "__" in d.name])
    if not trial_dirs:
        return HarborTaskResult(task=task, reward=None, error="no trial dir")
    trial_result_json = trial_dirs[0] / "result.json"
    if not trial_result_json.exists():
        return HarborTaskResult(
            task=task, reward=None, error=f"missing {trial_result_json}"
        )
    try:
        data = json.loads(trial_result_json.read_text())
        # Same field path used by trajectory_analysis.summarize_trial.
        # Defensive `or {}` because verifier_result is sometimes explicitly
        # null when the trial crashed before the verifier ran.
        verifier = data.get("verifier_result") or {}
        rewards = verifier.get("rewards") or {}
        reward = rewards.get("reward")
        exc_info = data.get("exception_info") or {}
        if reward is None:
            err = (
                exc_info.get("exception_type")
                or exc_info.get("exception_message")
                or "no reward in trial result"
            )
            return HarborTaskResult(task=task, reward=None, error=str(err))
        return HarborTaskResult(task=task, reward=float(reward), error=None)
    except Exception as exc:
        return HarborTaskResult(task=task, reward=None, error=f"parse error: {exc}")
