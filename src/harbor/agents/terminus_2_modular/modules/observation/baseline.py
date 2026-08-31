"""Raw terminal observation — faithful port of Terminus2's terminal pane
capture + `_limit_output_length`.

- `capture()` reads incremental output via `ctx.shared.tmux_session.get_incremental_output()`
  (TmuxSession tracks `_previous_buffer` internally so we always get only the
  new content since the last read).
- `_smart_truncate` matches `Terminus2._limit_output_length` byte-for-byte:
  head + tail bytes preserved, middle replaced with explicit
  `"[... output limited to N bytes; M interior bytes omitted ...]"` marker.
"""

from __future__ import annotations

from harbor.agents.terminus_2_modular.protocols import (
    ModuleCtx,
    ObsResult,
    ObsState,
)

DEFAULT_MAX_BYTES = 10_000


def _smart_truncate(output: str, max_bytes: int) -> str:
    """Byte-accurate head+tail truncation; same marker format as
    `Terminus2._limit_output_length` (terminus_2.py:529-565)."""
    output_bytes = output.encode("utf-8")
    if len(output_bytes) <= max_bytes:
        return output

    portion = max_bytes // 2
    first = output_bytes[:portion].decode("utf-8", errors="ignore")
    last = output_bytes[-portion:].decode("utf-8", errors="ignore")

    omitted = len(output_bytes) - len(first.encode("utf-8")) - len(last.encode("utf-8"))
    return (
        f"{first}\n[... output limited to {max_bytes} bytes; "
        f"{omitted} interior bytes omitted ...]\n{last}"
    )


class BaselineObservation:
    NAME = "baseline"
    NICHE = {"capture": "incremental", "capacity": "low", "staleness": "none"}
    DESCRIPTION = (
        "Capture incremental terminal output via tmux session "
        "(`get_incremental_output`) and apply byte-accurate head+tail "
        "truncation matching Terminus2._limit_output_length."
    )
    PARAMS_SCHEMA = {"max_bytes": "int (default 10000)"}

    def __init__(self, max_bytes: int = DEFAULT_MAX_BYTES):
        self.max_bytes = max_bytes

    async def capture(
        self,
        prev: ObsState,
        ctx: ModuleCtx,
    ) -> tuple[ObsResult, ObsState]:
        session = ctx.shared.tmux_session
        if session is None:
            return ObsResult(text=""), prev

        try:
            raw = await session.get_incremental_output()
        except Exception as exc:
            ctx.services.logger.warning(
                "raw_terminal: get_incremental_output failed: %s", exc
            )
            raw = ""

        text = _smart_truncate(raw or "", self.max_bytes)
        return ObsResult(text=text), ObsState(prev_terminal=raw or "")


def register(library):
    library.register(
        type_="observation",
        name=BaselineObservation.NAME,
        factory=lambda params: BaselineObservation(
            max_bytes=int(params.get("max_bytes", DEFAULT_MAX_BYTES))
        ),
        description=BaselineObservation.DESCRIPTION,
        params_schema=BaselineObservation.PARAMS_SCHEMA,
        niche=BaselineObservation.NICHE,
    )
