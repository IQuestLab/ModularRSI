"""Evaluator-side transparent module execution tracing.

Evolved variants can ship
plausible-looking mechanisms that never activate at runtime (a plan checklist
whose step list is never populated, a guard counter that is never read). The
load/import/protocol gates can't see that, and the review gate only reads the
variant's self-description. The fix is to make runtime behavior *observable*:
the kernel wraps every module instance in a transparent proxy that records one
compact line per Protocol-method call, so both a human (trial.log) and the
editor (trajectory.json → `agent.extra.module_trace`) can see what each
variant actually did — including variants a future generation writes, which
cannot be trusted to instrument themselves.

Dual sink (one `trace()` call feeds both):
1. `services.logger.info("[type:variant] method → summary")` → trial.log
2. `AtifTrajectoryRecorder.module_events` → dumped as
   `agent.extra.module_trace` (rendered for the editor by
   `trajectory_analysis._render_module_trace`).

The proxy is attribute-transparent: non-traced attributes/methods pass through
untouched, so `hasattr(...)` feature probes and duck-typed extras
(`get_metrics`, `build_skills_section`, …) keep working. A summarization
failure never breaks the underlying call; an exception raised by the wrapped
method is traced (`⚠ raise …`) and re-raised unchanged.
"""

from __future__ import annotations

import inspect
from typing import Any

# Protocol methods traced per module type. Anything not listed passes through
# untraced (setup/teardown/format_* are structural, not per-step behavior).
TRACED_METHODS: dict[str, tuple[str, ...]] = {
    "observation": ("capture",),
    "context_mgmt": ("maybe_compress", "force_summarize"),
    "tools": ("parse_llm_response", "execute"),
    "verification": ("should_terminate",),
    "agent_loop": ("run",),
}

_SUMMARY_CAP = 200


def _clip(text: str, cap: int = _SUMMARY_CAP) -> str:
    text = " ".join(str(text).split())  # collapse newlines/runs of whitespace
    return text if len(text) <= cap else text[: cap - 1] + "…"


# ---------------------------------------------------------------------------
# Per-method result summarizers. Defensive by design: any attribute may be
# missing on an evolved variant's return value — fall back to type name.
# ---------------------------------------------------------------------------


def _summarize_capture(result: Any) -> str:
    obs = result[0] if isinstance(result, tuple) and result else result
    text = getattr(obs, "text", None)
    if not isinstance(text, str):
        return f"→ {type(obs).__name__}"
    truncated = " (truncated)" if "output limited to" in text else ""
    return f"→ {len(text)} chars{truncated}"


def _summarize_compress(result: Any) -> str:
    occurred = bool(getattr(result, "summarization_occurred", False))
    handoff = getattr(result, "handoff_prompt", None)
    if not occurred and handoff is None:
        return "→ no-op"
    parts = ["→ summarized" if occurred else "→ modified"]
    if handoff is not None:
        parts.append(f"handoff {len(str(handoff))} chars")
    return ", ".join(parts)


def _summarize_parse(result: Any) -> str:
    commands = getattr(result, "commands", None)
    n_cmds = len(commands) if isinstance(commands, list) else "?"
    flags = [
        f"{n_cmds} cmd(s)",
        f"task_complete={bool(getattr(result, 'is_task_complete', False))}",
        f"plan={'yes' if getattr(result, 'plan', '') else 'no'}",
    ]
    if getattr(result, "error", ""):
        flags.append("PARSE-ERROR")
    if getattr(result, "warning", ""):
        flags.append("warning")
    return "→ " + ", ".join(flags)


def _summarize_execute(result: Any, call_args: tuple, call_kwargs: dict) -> str:
    tool_call = call_kwargs.get("call") or (call_args[0] if call_args else None)
    keys = _clip(getattr(tool_call, "keystrokes", "") or "", 80)
    ok = bool(getattr(result, "success", False))
    out_len = len(getattr(result, "output", "") or "")
    tail = (
        f"ok, {out_len} chars out"
        if ok
        else f"FAIL: {_clip(getattr(result, 'error', '') or '', 80)}"
    )
    return f"`{keys}` → {tail}"


def _summarize_should_terminate(result: Any) -> str:
    if isinstance(result, tuple) and len(result) == 2:
        return f"→ ({bool(result[0])}, {_clip(str(result[1]), 100)})"
    return f"→ {type(result).__name__}"


def _summarize_run(result: Any) -> str:
    return (
        f"→ finished success={bool(getattr(result, 'success', False))}"
        f" failure_tag={getattr(result, 'failure_tag', None)}"
    )


def _summarize(method: str, result: Any, args: tuple, kwargs: dict) -> str:
    if method == "capture":
        return _summarize_capture(result)
    if method in ("maybe_compress", "force_summarize"):
        return _summarize_compress(result)
    if method == "parse_llm_response":
        return _summarize_parse(result)
    if method == "execute":
        return _summarize_execute(result, args, kwargs)
    if method == "should_terminate":
        return _summarize_should_terminate(result)
    if method == "run":
        return _summarize_run(result)
    return f"→ {type(result).__name__}"


# ---------------------------------------------------------------------------
# The proxy
# ---------------------------------------------------------------------------


class TracingProxy:
    """Wraps one module instance; traces the Protocol methods of its type."""

    def __init__(self, inner: Any, module_type: str, variant: str, services: Any):
        # Names prefixed _tp_ to keep the passthrough namespace clean.
        self._tp_inner = inner
        self._tp_label = f"{module_type}:{variant}"
        self._tp_traced = frozenset(TRACED_METHODS.get(module_type, ()))
        self._tp_services = services

    def _tp_emit(self, method: str, summary: str) -> None:
        """Dual-sink emit; never raises."""
        try:
            recorder = getattr(self._tp_services, "trajectory", None)
            trace = getattr(recorder, "trace", None)
            if callable(trace):
                trace(self._tp_label, method, summary)
            else:  # recorder without trace support → log-only sink
                logger = getattr(self._tp_services, "logger", None)
                if logger is not None:
                    logger.info("[%s] %s %s", self._tp_label, method, summary)
        except Exception:
            pass

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._tp_inner, name)
        if name not in self._tp_traced or not callable(attr):
            return attr

        if inspect.iscoroutinefunction(attr):

            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                if name == "run":
                    self._tp_emit(name, "→ start")
                try:
                    result = await attr(*args, **kwargs)
                except Exception as exc:
                    self._tp_emit(
                        name, f"⚠ raise {type(exc).__name__}: {_clip(str(exc), 120)}"
                    )
                    raise
                try:
                    summary = _summarize(name, result, args, kwargs)
                except Exception:
                    summary = f"→ {type(result).__name__}"
                self._tp_emit(name, summary)
                return result

            return async_wrapper

        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                result = attr(*args, **kwargs)
            except Exception as exc:
                self._tp_emit(
                    name, f"⚠ raise {type(exc).__name__}: {_clip(str(exc), 120)}"
                )
                raise
            try:
                summary = _summarize(name, result, args, kwargs)
            except Exception:
                summary = f"→ {type(result).__name__}"
            self._tp_emit(name, summary)
            return result

        return sync_wrapper


def wrap_module(inner: Any, module_type: str, variant: str, services: Any) -> Any:
    """Wrap a module instance for tracing; unknown types pass through."""
    if module_type not in TRACED_METHODS:
        return inner
    return TracingProxy(inner, module_type, variant, services)
