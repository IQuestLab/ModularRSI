"""Margin-gated termination: requires a numeric benchmark margin before allowing
termination, on top of evidence-gating logic. After the standard evidence
gates pass (consecutive signals, last tool result, observation failure
indicators, optional ever-success), this variant scans the latest observation
for a benchmark speedup value (e.g. 'Speedup: 11.86x'). If the speedup is
below a configurable `min_measured_speedup` (default 15.0), termination is
rejected — the agent must keep optimizing until its self-test shows a healthy
margin above the requirement.

Niche:
  - evidence: benchmark-margin  — checks numeric benchmark margin in observations
  - error_gate: checked          — gates on failure indicators / zero success
"""

from __future__ import annotations

import re

from harbor.agents.terminus_2_modular.modules.verification.baseline import (
    BaselineVerification,
)
from harbor.agents.terminus_2_modular.protocols import AgentLoopState, ModuleCtx


class MarginGatedVerification(BaselineVerification):
    NAME = "margin_gated"
    NICHE = {"evidence": "benchmark-margin", "error_gate": "checked"}
    DESCRIPTION = (
        "Two consecutive task_complete signals AND real evidence the work succeeded "
        "AND a numeric benchmark speedup with sufficient margin (default 15x minimum). "
        "Rejects termination when the latest observation contains a speedup measurement "
        "below the configured threshold, even if the baseline evidence gates pass — "
        "forcing the agent to continue optimizing until the self-test shows a healthy "
        "margin above the requirement. Optionally requires the speedup to appear in "
        "two separate executed commands (require_repeat_pass). "
        "Prefer for optimization/benchmark tasks where the model may declare 'done' "
        "on a barely-passing self-test that is likely to drop below the requirement "
        "under noise."
    )
    PARAMS_SCHEMA = {
        "required_consecutive": "int (default 2)",
        "check_last_tool": "bool (default True)",
        "check_obs_failure": "bool (default True)",
        "require_ever_success": "bool (default False)",
        "min_measured_speedup": "float (default 15.0) — benchmark speedup below this threshold is rejected",
        "require_repeat_pass": "bool (default False) — speedup measurement must appear in two separate executed commands",
    }

    # Regex patterns for benchmark speedup, ordered by specificity (most specific first).
    _SPEEDUP_PATTERNS = [
        re.compile(r"Speedup:\s*([0-9]+(?:\.[0-9]+)?)x", re.IGNORECASE),
        re.compile(r"([0-9]+(?:\.[0-9]+)?)x\s*speedup", re.IGNORECASE),
        re.compile(r"([0-9]+(?:\.[0-9]+)?)x\s*faster", re.IGNORECASE),
        re.compile(r"([0-9]+(?:\.[0-9]+)?)x", re.IGNORECASE),
    ]

    def __init__(
        self,
        required_consecutive: int = 2,
        check_last_tool: bool = True,
        check_obs_failure: bool = True,
        require_ever_success: bool = False,
        min_measured_speedup: float = 15.0,
        require_repeat_pass: bool = False,
    ):
        super().__init__(required_consecutive=required_consecutive)
        self.check_last_tool = check_last_tool
        self.check_obs_failure = check_obs_failure
        self.require_ever_success = require_ever_success
        self.min_measured_speedup = float(min_measured_speedup)
        self.require_repeat_pass = bool(require_repeat_pass)
        # Cross-call memory: the last executed command's outcome, carried across
        # pure confirmation turns (where state.last_tool_result is None).
        self._last_tool_result = None
        self._ever_succeeded = False
        # Count of separate executed commands that passed the margin check.
        self._margin_passes = 0

    async def should_terminate(
        self,
        state: AgentLoopState,
        ctx: ModuleCtx,
    ) -> tuple[bool, str]:
        # Learn from this iteration's execution outcome regardless of whether the
        # LLM declared completion (state.last_tool_result is set before this call).
        tr = getattr(state, "last_tool_result", None)
        if tr is not None:
            self._last_tool_result = tr
            if getattr(tr, "success", False):
                self._ever_succeeded = True

        # Baseline gate: two consecutive task_complete signals.
        consec = getattr(state, "consecutive_complete_signals", 0)
        if consec < self.required:
            return False, ""

        # 1. The most recent executed command must not have failed.
        if self.check_last_tool:
            last = self._last_tool_result
            if last is not None and not getattr(last, "success", True):
                reason = (
                    f"last executed command failed: "
                    f"{getattr(last, 'error', 'unknown error')}. "
                    "Cannot terminate while the most recent command failed."
                )
                return False, reason

        # 2. At least one command must have succeeded during the episode.
        if self.require_ever_success and not self._ever_succeeded:
            return (
                False,
                "no tool command succeeded in this episode; refusing to terminate "
                "with zero successful actions.",
            )

        # 3. The latest observation must be free of failure indicators.
        if self.check_obs_failure:
            last_obs = getattr(state, "last_obs", None)
            if last_obs is not None:
                obs_text = str(last_obs)
                failure_indicators = [
                    "FAILED",
                    "Traceback",
                    "exit code 1",
                    "exit code 2",
                ]
                lowered = obs_text.lower()
                for indicator in failure_indicators:
                    if indicator.lower() in lowered:
                        return (
                            False,
                            f"last observation contains failure indicator '{indicator}'",
                        )

        # ---- Standard evidence gates passed — now check speedup margin. ----
        last_obs = getattr(state, "last_obs", None)
        if last_obs is None:
            # No observation to check; fall back to base decision.
            return True, (
                f"task_complete confirmed {consec} consecutive times with "
                "successful evidence (no observation to check speedup margin)"
            )

        obs_text = str(last_obs)
        speedup_value = None
        for pattern in self._SPEEDUP_PATTERNS:
            match = pattern.search(obs_text)
            if match:
                speedup_value = float(match.group(1))
                break

        if speedup_value is None:
            # No benchmark speedup found in observation — cannot gate on margin.
            return True, (
                f"task_complete confirmed {consec} consecutive times with "
                "successful evidence (no benchmark speedup value found in observation)"
            )

        if speedup_value < self.min_measured_speedup:
            return False, (
                f"benchmark speedup {speedup_value}x is below margin threshold "
                f"{self.min_measured_speedup}x; keep optimizing until the self-test "
                f"shows a healthy margin above the requirement"
            )

        # Speedup passes the margin check.
        if not self.require_repeat_pass:
            return True, (
                f"task_complete confirmed {consec} consecutive times with "
                f"speedup {speedup_value}x (above {self.min_measured_speedup}x margin)"
            )

        # require_repeat_pass: increment counter on each call where the
        # executed command changed and succeeded.
        tr = getattr(state, "last_tool_result", None)
        if tr is not None and getattr(tr, "success", False):
            self._margin_passes += 1
            if self._margin_passes >= 2:
                return True, (
                    f"task_complete confirmed {consec} consecutive times with "
                    f"speedup {speedup_value}x (confirmed twice, "
                    f"above {self.min_measured_speedup}x margin)"
                )
            else:
                # First pass recorded but need a second distinct command execution.
                return False, (
                    f"benchmark speedup {speedup_value}x passes margin but has been "
                    f"confirmed only once (need 2 separate successful command executions); "
                    f"keep optimizing"
                )
        else:
            # No successful command this turn — wait for one.
            return False, "no successful command execution to measure benchmark margin"


def register(library):
    library.register(
        type_="verification",
        name=MarginGatedVerification.NAME,
        factory=lambda params: MarginGatedVerification(
            required_consecutive=int(params.get("required_consecutive", 2)),
            check_last_tool=bool(params.get("check_last_tool", True)),
            check_obs_failure=bool(params.get("check_obs_failure", True)),
            require_ever_success=bool(params.get("require_ever_success", False)),
            min_measured_speedup=float(params.get("min_measured_speedup", 15.0)),
            require_repeat_pass=bool(params.get("require_repeat_pass", False)),
        ),
        description=MarginGatedVerification.DESCRIPTION,
        params_schema=MarginGatedVerification.PARAMS_SCHEMA,
        niche=MarginGatedVerification.NICHE,
    )
