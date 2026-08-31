"""Consolidated full-scrollback terminal observation — the single full-history variant.

MERGE of the four full-capture siblings in the observation archive:

- `full_capture`        — founding idea: read the ENTIRE pane log
                          (`terminus_2_modular.pane`) instead of only incremental
                          output since the last poll.
- `bulk_capture`        — 200k default capacity so very large outputs (build
                          logs, generated files) are not truncated.
- `idle_aware_capture`  — staleness breadcrumb + shell idle/busy status.
- `fresh_full_scrollback` — cross-call `self._accumulated` history so the
                          observation grows instead of shrinking to a stub when
                          the pane log is temporarily unavailable, plus
                          staleness detection/suppression.

Why this variant exists (the failure class it fixes):

The baseline observation calls `get_incremental_output()`, which returns only
the bytes written since the previous poll. Across a multi-command task the
agent then sees ~70-char stubs and never the results of its `ls`, `cat`,
`build`, `edit`, or verification commands — it operates blind. The fix is a
full-scrollback observation: return the COMPLETE terminal state at every
decision point, keep the whole history across calls on the instance, and never
degrade back to an incremental-only stub.

Behaviour:

1. Every capture reads the current pane-log scrollback (tail up to 2*max_bytes),
   so the agent sees the whole terminal state — previous command results, error
   messages, file contents, build logs — at each decision point.
2. It additionally drains `get_incremental_output()` and merges any bytes not
   yet flushed to the pane log, closing the log-flush-lag race so output is
   never stuck invisible in the tmux buffer.
3. When the pane log is unavailable/empty, incremental output is accumulated in
   `self._accumulated` across calls, so successive observations still grow into
   the full history instead of returning near-empty stubs.
4. Staleness (no new bytes since the last observation) returns a short note plus
   a generous tail of the scrollback, instead of replaying the same stale block.
5. Every observation is prefixed with an explicit
   '[terminal scrollback: N bytes — FULL history]' header (so the model knows
   it is seeing the whole scrollback, not a delta) and suffixed with shell
   idle/busy status ('[shell status: idle — prompt visible]' vs '[shell status:
   busy — command still running, no prompt visible]', plus an interrupt hint
   when a busy command has produced no new output for two observations).

SUPERSEDES: full_capture, bulk_capture, idle_aware_capture, fresh_full_scrollback.

Niche: full capture, 200k capacity, staleness detect-and-suppress — the
canonical occupant of the full-scrollback cell.
"""

from __future__ import annotations

import re
from pathlib import Path

from harbor.agents.terminus_2_modular.modules.observation.baseline import (
    BaselineObservation,
    _smart_truncate,
)
from harbor.agents.terminus_2_modular.protocols import ObsResult, ObsState
from harbor.models.trial.paths import EnvironmentPaths

# Shell prompt patterns: common shell prompt endings.
PROMPT_PATTERN = re.compile(r"[\$#%]\s*$")

DEFAULT_MAX_BYTES = 200_000
STALE_TAIL_BYTES = 8_000


class TerminalScrollbackObservation(BaselineObservation):
    """Full terminal scrollback, never an incremental stub.

    Reads the entire pane log each capture (plus unflushed incremental bytes),
    keeps the full history on the instance as a fallback, suppresses stale
    repeats, and annotates the observation with a full-history header and shell
    idle/busy status.
    """

    NAME = "terminal_scrollback"
    NICHE = {"capture": "full", "capacity": "200k", "staleness": "detect-suppress"}
    DESCRIPTION = (
        "Full terminal scrollback (200k default capacity) that never degrades to a "
        "tiny incremental-only stub: reads the ENTIRE pane log every observation "
        "(plus any unflushed incremental bytes, so output is never lost), keeps the "
        "complete history across calls on the instance, and returns the whole "
        "scrollback each time — with a '[terminal scrollback: N bytes — FULL "
        "history]' header, a short 'no new output' note + tail instead of replaying "
        "stale content when nothing changed, and a shell idle/busy status "
        "annotation. Consolidates the full-scrollback family (full_capture, "
        "bulk_capture, idle_aware_capture, fresh_full_scrollback). Prefer for ANY "
        "task where the agent must see command results, error messages, build logs, "
        "file listings, or generated-content output — use instead of baseline when "
        "incremental capture would leave the agent blind to prior terminal output."
    )
    PARAMS_SCHEMA = {"max_bytes": "int (default 200000)"}

    def __init__(self, max_bytes: int = DEFAULT_MAX_BYTES):
        super().__init__(max_bytes=max_bytes)
        self._last_log_size: int = -1  # -1 = never captured yet
        self._accumulated: str = ""  # cross-call full-history fallback
        self._prev_busy: bool = False
        self._busy_no_new_count: int = 0

    async def capture(self, prev, ctx) -> tuple:
        """Return the full terminal scrollback; fall back to self-accumulated
        incremental history when the pane log is unavailable."""
        session = ctx.shared.tmux_session
        if session is None:
            return ObsResult(text=""), prev

        # Strategy 1: read the pane log directly (full scrollback).
        pane_log = await self._probe_pane_log_path(ctx)
        if pane_log is not None:
            size, raw = await self._read_pane_log_size_and_tail(pane_log, ctx)

            # Drain unflushed incremental bytes and merge them into the pane
            # snapshot so output is never lost between the two capture sources.
            delta = ""
            try:
                delta = await session.get_incremental_output()
            except Exception as exc:
                ctx.services.logger.warning(
                    "terminal_scrollback: get_incremental_output failed: %s", exc
                )
            if delta:
                raw = self._merge_delta(raw, delta)

            if raw and size > 0:
                # Keep our own full-history copy so later fallbacks are never empty.
                self._accumulated = raw

                fresh = size != self._last_log_size or bool(delta)
                if size < self._last_log_size:
                    # Log rotated (shrank): treat as fresh content next read.
                    self._last_log_size = -1
                    fresh = True
                self._last_log_size = size

                if not fresh:
                    # No new bytes since the last observation: suppress the
                    # stale full-scrollback repeat; keep a generous tail so the
                    # agent still sees recent results.
                    note = (
                        "\n[no new terminal output since the last observation; "
                        "scrollback unchanged. Tail of current scrollback:]\n"
                        f"{raw[-STALE_TAIL_BYTES:]}"
                    )
                    return ObsResult(text=self._annotate_shell_status(note, raw)), prev

                text = _smart_truncate(raw, self.max_bytes)
                text = self._prefix_history_header(text, raw)
                annotated = self._annotate_shell_status(text, raw)
                return ObsResult(text=annotated), ObsState(prev_terminal=raw)

        # Strategy 2 (fallback): pane log unavailable/empty — accumulate the
        # incremental output on SELF, so successive observations grow instead
        # of returning a near-empty stub that hides earlier results.
        try:
            incremental = await session.get_incremental_output()
        except Exception as exc:
            ctx.services.logger.warning(
                "terminal_scrollback: get_incremental_output failed: %s", exc
            )
            incremental = ""

        if incremental:
            if self._accumulated:
                self._accumulated = f"{self._accumulated}\n{incremental}"
            else:
                self._accumulated = incremental
            # Bound the stored history (keep the last max_bytes*2 bytes) so
            # memory stays flat on very long tasks.
            encoded = self._accumulated.encode("utf-8")
            if len(encoded) > self.max_bytes * 2:
                self._accumulated = encoded[-(self.max_bytes * 2) :].decode(
                    "utf-8", errors="ignore"
                )
        elif not self._accumulated:
            # Nothing since we started: adopt whatever the loop threaded, if any.
            prev_terminal = (
                (prev.prev_terminal or "") if hasattr(prev, "prev_terminal") else ""
            )
            if prev_terminal:
                self._accumulated = prev_terminal

        combined = self._accumulated or ""
        text = _smart_truncate(combined, self.max_bytes)
        text = self._prefix_history_header(text, combined)
        annotated = self._annotate_shell_status(text, combined)
        return ObsResult(text=annotated), ObsState(prev_terminal=combined)

    # -- helpers ------------------------------------------------------------

    def _merge_delta(self, raw: str, delta: str) -> str:
        """Append incremental bytes to the pane-log snapshot unless they are
        already present at the end of the snapshot (already flushed)."""
        if not raw:
            return delta
        if not delta:
            return raw
        lines = [ln for ln in delta.strip().splitlines() if ln.strip()]
        last = lines[-1].strip() if lines else ""
        if last and raw.endswith(last):
            return raw
        return f"{raw}\n{delta}"

    def _prefix_history_header(self, text: str, raw: str) -> str:
        """Prepend an explicit marker so the model knows it is seeing the FULL
        terminal scrollback rather than an incremental delta."""
        if not text:
            return text
        try:
            n = len(raw.encode("utf-8"))
        except Exception:
            n = len(raw)
        return f"[terminal scrollback: {n} bytes — FULL history]\n{text}"

    def _annotate_shell_status(self, text: str, raw: str) -> str:
        """Check if the shell prompt is visible and append status annotation.

        Uses the raw (un-truncated) terminal output to check for the prompt,
        since the truncated text might cut off the last line.
        """
        if not raw:
            return text

        lines = [line for line in raw.split("\n") if line.strip()]
        last_line = lines[-1] if lines else ""

        is_idle = bool(PROMPT_PATTERN.search(last_line))

        if is_idle:
            annotation = "\n[shell status: idle — prompt visible]"
            self._prev_busy = False
            self._busy_no_new_count = 0
            return text + annotation

        # Busy: no shell prompt visible
        self._prev_busy = True
        self._busy_no_new_count += 1

        if self._busy_no_new_count >= 2:
            annotation = (
                "\n[shell status: busy — command still running, no prompt visible. "
                "The previous command may be hanging; consider interrupting with Ctrl+C "
                "if needed.]"
            )
        else:
            annotation = (
                "\n[shell status: busy — command still running, no prompt visible]"
            )
        return text + annotation

    async def _probe_pane_log_path(self, ctx) -> Path | None:
        """Return the pane log path, or None if unavailable."""
        try:
            return EnvironmentPaths.agent_dir / "terminus_2_modular.pane"
        except Exception:
            return None

    async def _read_pane_log_size_and_tail(self, path: Path, ctx) -> tuple[int, str]:
        """Return (size_in_bytes, tail of the pane log)."""
        try:
            size_result = await ctx.state.env.exec(
                command=f"wc -c < '{path}'",
                timeout_sec=3,
            )
            if not (
                size_result and size_result.return_code == 0 and size_result.stdout
            ):
                return 0, ""
            size = int(size_result.stdout.strip())
            if size <= 0:
                return size, ""

            tail_result = await ctx.state.env.exec(
                command=f"tail -c {self.max_bytes * 2} '{path}'",
                timeout_sec=5,
            )
            if tail_result and tail_result.return_code == 0 and tail_result.stdout:
                return size, tail_result.stdout
        except Exception as exc:
            ctx.services.logger.debug(
                "terminal_scrollback: pane log read failed: %s", exc
            )
        return 0, ""


def register(library):
    library.register(
        type_="observation",
        name=TerminalScrollbackObservation.NAME,
        factory=lambda params: TerminalScrollbackObservation(
            max_bytes=int(params.get("max_bytes", DEFAULT_MAX_BYTES))
        ),
        description=TerminalScrollbackObservation.DESCRIPTION,
        params_schema=TerminalScrollbackObservation.PARAMS_SCHEMA,
        niche=TerminalScrollbackObservation.NICHE,
    )
