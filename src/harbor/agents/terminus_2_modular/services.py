"""Default implementations of the KernelServices interfaces.

Phase 4: TrajectoryRecorder is now a real ATIF writer (no longer a no-op).
Phase 6 will append per-step Step objects from the agent_loop; here we just
provide the writer + storage.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from harbor.agents.terminus_2_modular.protocols import KernelServices
from harbor.models.trajectories import (
    Agent,
    FinalMetrics,
    Step,
    Trajectory,
)
from harbor.utils.trajectory_utils import format_trajectory_json


@dataclass
class _StdLoggerAdapter:
    inner: logging.Logger

    def debug(self, msg, *args):
        self.inner.debug(msg, *args)

    def info(self, msg, *args):
        self.inner.info(msg, *args)

    def warning(self, msg, *args):
        self.inner.warning(msg, *args)

    def error(self, msg, *args):
        self.inner.error(msg, *args)


@dataclass
class _InMemoryFailureReporter:
    tags: list[tuple[str, str, str]] = field(default_factory=list)

    def raise_tag(self, tag: str, module_name: str, detail: str = "") -> None:
        self.tags.append((tag, module_name, detail))


@dataclass
class _InMemoryStatsWriter:
    data: dict[str, dict[str, float]] = field(default_factory=dict)

    def record(self, module_name: str, metric: str, value: float) -> None:
        self.data.setdefault(module_name, {}).setdefault(metric, 0.0)
        self.data[module_name][metric] += value


@dataclass
class AtifTrajectoryRecorder:
    """ATIF trajectory writer. Faithful port of
    `Terminus2._dump_trajectory_with_continuation_index` (1951-2017)."""

    logs_dir: Path
    session_id: str
    agent_name: str
    agent_version: str
    model_name: str
    # Goes into Trajectory.agent.extra
    agent_extra: dict[str, Any] = field(default_factory=dict)
    # TrajectoryConfig fields
    linear_history: bool = False
    raw_content: bool = False

    # Mutable per-task state
    steps: list[Step] = field(default_factory=list)
    final_metrics: FinalMetrics | None = None
    summarization_count: int = 0
    logger: logging.Logger | None = None
    # Kernel module-execution trace (fed by tracing.TracingProxy). Dumped as
    # `agent.extra.module_trace` so the editor can see runtime behavior of
    # every active variant. Capped to keep trajectories bounded.
    module_events: list[dict] = field(default_factory=list)
    module_events_dropped: int = 0
    MAX_MODULE_EVENTS = 4000

    def append_step(self, step: Step) -> None:
        self.steps.append(step)

    def trace(self, module: str, call: str, summary: str) -> None:
        """Dual-sink module trace: one line to the agent log (→ trial.log)
        and one compact event into the trajectory."""
        if self.logger:
            self.logger.info("[%s] %s %s", module, call, summary)
        if len(self.module_events) < self.MAX_MODULE_EVENTS:
            self.module_events.append(
                {"module": module, "call": call, "summary": summary}
            )
        else:
            self.module_events_dropped += 1

    def record_asciinema_marker(self, marker_text: str) -> None:
        # Matches Terminus2._record_asciinema_marker (currently a no-op in
        # the original; left as TODO there too). Keep no-op for parity.
        return

    def set_final_metrics(self, fm: FinalMetrics) -> None:
        self.final_metrics = fm

    def dump(self) -> None:
        """Default dump: writes the base trajectory (continuation_index=0).

        When linear_history mode + multiple summarization splits are wired up
        in Phase 5/6, the agent will additionally call
        `dump_with_continuation_index(N)` at split boundaries.
        """
        self._dump_with_index(self.summarization_count)

    def dump_with_continuation_index(self, continuation_index: int) -> None:
        self._dump_with_index(continuation_index)

    def _dump_with_index(self, continuation_index: int) -> None:
        if not self.steps:
            if self.logger:
                self.logger.warning("trajectory has no steps, skipping dump")
            return

        agent_extra = dict(self.agent_extra)
        if self.linear_history and continuation_index > 0:
            agent_extra["continuation_index"] = continuation_index
        if self.module_events:
            agent_extra["module_trace"] = list(self.module_events)
            if self.module_events_dropped:
                agent_extra["module_trace_dropped"] = self.module_events_dropped

        continued_trajectory_ref = None
        if self.linear_history and continuation_index < self.summarization_count:
            next_idx = continuation_index + 1
            continued_trajectory_ref = f"trajectory.cont-{next_idx}.json"

        trajectory = Trajectory(
            session_id=self.session_id,
            agent=Agent(
                name=self.agent_name,
                version=self.agent_version or "unknown",
                model_name=self.model_name,
                extra=agent_extra,
            ),
            steps=self.steps,
            final_metrics=self.final_metrics,
            continued_trajectory_ref=continued_trajectory_ref,
        )

        if self.linear_history and continuation_index > 0:
            traj_path = self.logs_dir / f"trajectory.cont-{continuation_index}.json"
        else:
            traj_path = self.logs_dir / "trajectory.json"

        try:
            json_str = format_trajectory_json(trajectory.to_json_dict())
            traj_path.write_text(json_str)
            if self.logger:
                self.logger.debug("Trajectory dumped to %s", traj_path)
        except Exception as exc:
            if self.logger:
                self.logger.error("Failed to dump trajectory: %s", exc)


def build_default_services(
    logger: logging.Logger,
    trajectory_recorder: "AtifTrajectoryRecorder | None" = None,
) -> KernelServices:
    """Build the default KernelServices.

    If `trajectory_recorder` is None, falls back to a no-op recorder. The agent
    typically builds the real recorder itself (it knows logs_dir / session_id
    / agent metadata) and passes it in.
    """
    if trajectory_recorder is None:
        trajectory_recorder = _NoopTrajectoryRecorder()
    return KernelServices(
        failures=_InMemoryFailureReporter(),
        stats=_InMemoryStatsWriter(),
        logger=_StdLoggerAdapter(logger),
        trajectory=trajectory_recorder,
    )


# Kept for tests / scenarios that don't need ATIF dump (e.g. unit tests of
# individual modules).
@dataclass
class _NoopTrajectoryRecorder:
    steps: list[Any] = field(default_factory=list)
    module_events: list[dict] = field(default_factory=list)

    def append_step(self, step: Any) -> None:
        self.steps.append(step)

    def trace(self, module: str, call: str, summary: str) -> None:
        self.module_events.append({"module": module, "call": call, "summary": summary})

    def dump(self) -> None:
        pass

    def record_asciinema_marker(self, marker_text: str) -> None:
        pass
