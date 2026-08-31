"""Baseline agent_loop — Phase 6: full Terminus2-parity main loop.

Per-iteration flow:
  1. Session alive check; bail if dead
  2. context_mgmt.maybe_compress (proactive summarization)
  3. _setup_episode_logging (per-iter prompt/response files)
  4. Token snapshot
  5. _query_llm: chat.chat wrapped in tenacity retry + ContextLengthExceededError
     fallback (calls context_mgmt.force_summarize) + OutputLengthExceededError handling
  6. Record system + user handoff steps (if summarization occurred this iter)
  7. tools.parse_llm_response
  8. asciinema marker
  9. Parse error → record agent step + retry prompt → continue
  10. Execute commands
  11. observation.capture (+ two-phase completion gate + warnings prefix)
  12. Append agent step (message / tool_calls / observation / metrics)
  13. recorder.dump
  14. verification.should_terminate
  15. Build next prompt
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tenacity import (
    retry,
    retry_if_exception_type,
    retry_if_not_exception_type,
    stop_after_attempt,
)

from harbor.agents.terminus_2_modular.protocols import (
    AgentLoopResult,
    AgentLoopState,
    ContextMgmt,
    ModuleCtx,
    ObsState,
    Observation,
    ToolResult,
    ToolSet,
    VerificationLoop,
)
from harbor.llms.base import (
    ContextLengthExceededError,
    LLMResponse,
    OutputLengthExceededError,
)
from harbor.llms.chat import Chat
from harbor.models.trajectories import (
    Metrics,
    Observation as AtifObservation,
    ObservationResult,
    Step,
    ToolCall as AtifToolCall,
)


_DEFAULT_MAX_ITERATIONS = 1_000_000
_MAX_HANDOFF_TERMINAL_CHARS = 20_000


@dataclass
class _LLMCallResult:
    response: LLMResponse
    pending_handoff_prompt: str | None = None
    pending_subagent_refs: list[Any] | None = None
    summarization_occurred: bool = False


class BaselineAgentLoop:
    NAME = "baseline"
    NICHE = {"grounding": "reactive", "drift": "none", "parse": "baseline"}
    DESCRIPTION = (
        "Full-parity Terminus2 main loop: tenacity retry + "
        "ContextLengthExceededError fallback (3-tier summarization) + "
        "OutputLengthExceededError handling + per-step ATIF recording + "
        "per-episode log files + session-alive guard."
    )
    PARAMS_SCHEMA = {
        "max_iterations": "int (default 1_000_000)",
        "llm_call_kwargs": "dict (forwarded to chat.chat each call)",
        "raw_content": "bool (if True, dump raw LLM response; default False)",
    }

    def __init__(
        self,
        max_iterations: int = _DEFAULT_MAX_ITERATIONS,
        llm_call_kwargs: dict | None = None,
        raw_content: bool = False,
    ):
        self.max_iterations = max_iterations
        self.llm_call_kwargs = dict(llm_call_kwargs or {})
        self.raw_content = raw_content
        # accumulator for per-call request times (ms), exposed to agent.py
        self.api_request_times: list[float] = []

    def get_metrics(self) -> dict:
        """Exposed for agent.py to merge into context.metadata."""
        return {
            "api_request_times_msec": list(self.api_request_times),
            "n_episodes": getattr(self, "_n_episodes", 0),
        }

    # ---------- Main loop ----------

    async def run(
        self,
        initial_prompt: str,
        original_instruction: str,
        observation: Observation,
        context_mgmt: ContextMgmt,
        tools: ToolSet,
        verification: VerificationLoop,
        chat: Chat,
        ctx: ModuleCtx,
    ) -> AgentLoopResult:
        logger = ctx.services.logger
        recorder = ctx.services.trajectory
        logging_dir = Path(ctx.state.logs_dir) if ctx.state.logs_dir else None

        state = AgentLoopState(
            step_idx=0,
            last_obs=None,
            last_tool_result=None,
            chat=chat,
        )
        prompt = initial_prompt
        obs_state = ObsState()
        pending_completion = False

        # Carry-over of summarization side-effects across iterations:
        # When _query_llm summarizes reactively, it stashes (refs, handoff_prompt)
        # which we record as system + user steps right before the next agent step.
        carry_refs: list[Any] | None = None
        carry_handoff: str | None = None
        consecutive_parse_errors = 0

        self._n_episodes = 0
        for step_idx in range(self.max_iterations):
            self._n_episodes = step_idx + 1
            state.step_idx = step_idx

            # 1. session alive check (matches terminus_2.py:1291-1293)
            dead = await self._check_session_alive(ctx)
            if dead is not None:
                return dead

            # 2. proactive compress
            comp = await context_mgmt.maybe_compress(chat, original_instruction, ctx)
            chat = comp.chat
            if comp.handoff_prompt is not None:
                prompt = self._protocol_safe_handoff(
                    comp.handoff_prompt,
                    original_instruction,
                    prompt,
                    tools,
                    ctx,
                )
                carry_refs = comp.subagent_refs
                carry_handoff = prompt
                if comp.summarization_occurred and hasattr(
                    recorder, "summarization_count"
                ):
                    recorder.summarization_count += 1

            # 3. per-episode log paths
            logging_paths = self._setup_episode_logging(logging_dir, step_idx)

            # 4. token snapshot
            tokens_before_input = chat.total_input_tokens
            tokens_before_output = chat.total_output_tokens
            tokens_before_cache = chat.total_cache_tokens
            cost_before = chat.total_cost

            # HOOK: last chance for a variant to shape the prompt before the LLM
            # call (planning injection / reminder / re-plan trigger). Default: no-op.
            prompt = self._pre_llm_prompt(
                prompt, iteration=step_idx, state=state, ctx=ctx
            )

            # 5. LLM call (with retry + overflow fallback)
            try:
                call_result = await self._query_llm(
                    chat=chat,
                    prompt=prompt,
                    ctx=ctx,
                    context_mgmt=context_mgmt,
                    tools=tools,
                    original_instruction=original_instruction,
                    logging_paths=logging_paths,
                )
            except Exception as exc:
                import traceback

                logger.error(
                    "step %d: LLM call failed (post-retry): %s\n%s",
                    step_idx,
                    exc,
                    traceback.format_exc(),
                )
                ctx.services.failures.raise_tag(
                    "llm_call_failed", "agent_loop.baseline", str(exc)
                )
                return AgentLoopResult(
                    success=False,
                    final_text=str(exc),
                    failure_tag="llm_call_failed",
                )
            llm_response = call_result.response

            # 6. Reactive summarization (from inside _query_llm) carry-over
            if call_result.summarization_occurred:
                if call_result.pending_subagent_refs:
                    carry_refs = call_result.pending_subagent_refs
                if call_result.pending_handoff_prompt:
                    carry_handoff = call_result.pending_handoff_prompt
                if hasattr(recorder, "summarization_count"):
                    recorder.summarization_count += 1

            # Append summarization handoff steps (system + user) BEFORE agent step
            self._record_summarization_handoff(recorder, carry_refs, carry_handoff)
            carry_refs = None
            carry_handoff = None

            # 7. asciinema marker (no-op-by-default, parity with original)
            try:
                recorder.record_asciinema_marker(f"Episode {step_idx}")
            except Exception:
                pass

            # 8. parse
            parse = tools.parse_llm_response(llm_response.content)
            logger.debug(
                "step %d: parsed %d cmds, complete=%s, error=%s",
                step_idx,
                len(parse.commands),
                parse.is_task_complete,
                bool(parse.error),
            )

            # 9. parse error → record step + retry next iter
            if parse.error:
                ctx.services.failures.raise_tag("parse_error", "tools", parse.error)
                consecutive_parse_errors += 1
                retry_prompt = self._build_parse_error_prompt(parse, tools)
                self._append_parse_error_step(
                    recorder=recorder,
                    chat=chat,
                    llm_response=llm_response,
                    next_prompt=retry_prompt,
                    snapshot=(
                        tokens_before_input,
                        tokens_before_output,
                        tokens_before_cache,
                        cost_before,
                    ),
                    ctx=ctx,
                )
                if consecutive_parse_errors >= 2:
                    logger.error(
                        "step %d: response still malformed after one parse retry; "
                        "ending explicitly",
                        step_idx,
                    )
                    # Unlike an ordinary parse retry there is no later regular
                    # agent step whose dump would persist these failure steps.
                    recorder.dump()
                    return AgentLoopResult(
                        success=False,
                        final_text=parse.error,
                        failure_tag="parse_error",
                    )
                # Don't dump here — matches Terminus2.py:1402-1451 which only
                # dumps after the regular agent step (line 1603).
                prompt = retry_prompt
                continue
            consecutive_parse_errors = 0

            # 10. execute — keep ALL tool results (not just the last). For
            # tools modules that produce meaningful per-call output (e.g.,
            # editor_file_tools.read_file returning file contents), we want
            # to feed every result back to the LLM, not just the final one.
            tool_results, last_tool_result = await self._execute_commands(
                parse, tools, ctx, iteration=step_idx
            )

            # 11. observe + two-phase completion + warning prefix
            obs_result, obs_state = await observation.capture(obs_state, ctx)
            state.last_obs = obs_result
            state.last_tool_result = last_tool_result

            # Build the raw observation that gets fed back to the LLM next
            # round. For solver/tmux mode tool outputs are empty (the tmux
            # pane is the source of truth via observation.capture). For
            # editor mode tool outputs ARE the source of truth (read_file
            # output, edit confirmations) and observation.capture returns "".
            # Combining both lets the same baseline serve both roles.
            observation_text, pending_completion = self._build_observation_text(
                parse=parse,
                obs_result=obs_result,
                tool_results=tool_results,
                state=state,
                pending_completion=pending_completion,
            )

            # HOOK: let a variant reshape what the model sees/records this step
            # (e.g. surface an error / verification reason). Default: no-op.
            observation_text = self._shape_observation(
                observation_text,
                parse=parse,
                tool_results=tool_results,
                state=state,
                ctx=ctx,
            )

            # 12. append agent step
            self._append_agent_step(
                recorder=recorder,
                chat=chat,
                llm_response=llm_response,
                parse=parse,
                observation_text=observation_text,
                snapshot=(
                    tokens_before_input,
                    tokens_before_output,
                    tokens_before_cache,
                    cost_before,
                ),
                step_episode=step_idx,
                ctx=ctx,
            )
            recorder.dump()

            # 13. verify termination
            terminate, reason = await verification.should_terminate(state, ctx)
            # HOOK: a variant may override the stop/continue cadence (e.g. force a
            # re-plan instead of stopping, or stop early). Default: stop iff
            # verification asked to.
            if not self._should_continue(
                terminate=terminate,
                reason=reason,
                parse=parse,
                iteration=step_idx,
                state=state,
                ctx=ctx,
            ):
                logger.debug("step %d: terminating (%s)", step_idx, reason)
                return AgentLoopResult(
                    success=True,
                    final_text=reason,
                    failure_tag=None,
                )

            # 14. next prompt
            prompt = self._next_prompt(observation_text, parse, tools, ctx)

        ctx.services.failures.raise_tag(
            "max_iterations",
            "agent_loop.baseline",
            f"hit cap={self.max_iterations}",
        )
        return AgentLoopResult(
            success=False,
            final_text=f"hit max_iterations={self.max_iterations}",
            failure_tag="max_iterations",
        )

    # ---------- Extracted loop phases ----------
    # run() is a thin skeleton that calls these named phases. Each is a clean,
    # overridable seam with explicit inputs/outputs and NO hidden shared state,
    # so a variant can replace ONE phase (~15-30 lines) instead of the whole loop.
    # Every default here is the exact code that used to be inline in run() —
    # behavior is byte-for-byte identical to stock terminus-2.

    async def _check_session_alive(self, ctx: ModuleCtx) -> AgentLoopResult | None:
        """Guard: bail out if the tmux session has died. Returns a terminal result
        to return from run(), or None to continue (terminus_2.py:1291-1293)."""
        logger = ctx.services.logger
        session = ctx.shared.tmux_session
        if session is not None:
            try:
                alive = await session.is_session_alive()
            except Exception as exc:
                logger.debug("session.is_session_alive errored: %s", exc)
                alive = True  # err on side of continuing
            if not alive:
                logger.debug("session has ended; breaking out of agent loop")
                return AgentLoopResult(
                    success=False,
                    final_text="tmux session ended",
                    failure_tag="session_dead",
                )
        return None

    def _record_summarization_handoff(
        self, recorder, carry_refs: list[Any] | None, carry_handoff: str | None
    ) -> None:
        """Record the system + user handoff steps left over from a reactive
        summarization, right before the agent step. No-op when nothing is pending;
        the caller clears carry_refs/carry_handoff after."""
        if carry_refs:
            recorder.append_step(
                Step(
                    step_id=len(recorder.steps) + 1,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    source="system",
                    message=(
                        "Performed context summarization and handoff to continue task."
                    ),
                    observation=AtifObservation(
                        results=[
                            ObservationResult(
                                subagent_trajectory_ref=carry_refs,
                            )
                        ]
                    ),
                )
            )
        if carry_handoff:
            recorder.append_step(
                Step(
                    step_id=len(recorder.steps) + 1,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    source="user",
                    message=carry_handoff,
                )
            )

    def _build_parse_error_prompt(self, parse: Any, tools: ToolSet) -> str:
        """Build the retry prompt shown after a parse failure. Mirrors
        Terminus2._handle_llm_interaction (terminus_2.py:1191-1197) + the
        response-type suffix from _get_error_response_type."""
        feedback = f"ERROR: {parse.error}"
        if parse.warning:
            feedback += f"\nWARNINGS: {parse.warning}"
        parser_name = getattr(tools, "_parser_name", "JSON")
        if isinstance(parser_name, str) and parser_name.lower() == "xml":
            response_type = "XML response with all required fields"
        else:
            response_type = "JSON response with all required fields"
        return (
            f"Previous response had parsing errors:\n{feedback}\n\n"
            f"Please fix these issues and provide a proper {response_type}."
        )

    async def _execute_commands(
        self, parse: Any, tools: ToolSet, ctx: ModuleCtx, *, iteration: int
    ) -> tuple[list[ToolResult], ToolResult | None]:
        """Execute every parsed command in order, keeping ALL tool results (not
        just the last) so per-call output (e.g. editor read_file) is fed back."""
        logger = ctx.services.logger
        tool_results: list[ToolResult] = []
        last_tool_result: ToolResult | None = None
        for call in parse.commands:
            tr = await tools.execute(call, ctx)
            tool_results.append(tr)
            last_tool_result = tr
            if not tr.success:
                logger.debug("step %d: tool exec failed: %s", iteration, tr.error)
        return tool_results, last_tool_result

    def _build_observation_text(
        self,
        *,
        parse: Any,
        obs_result: Any,
        tool_results: list[ToolResult],
        state: AgentLoopState,
        pending_completion: bool,
    ) -> tuple[str, bool]:
        """Compose the observation the model sees next round from tool output +
        captured terminal, then apply the two-phase completion gate and warning
        prefix. Returns (observation_text, pending_completion); mutates
        state.consecutive_complete_signals exactly as the inline code did."""
        tool_output_parts: list[str] = []
        for tr in tool_results:
            if tr.output:
                tool_output_parts.append(tr.output)
            elif tr.error:
                tool_output_parts.append(f"[tool error] {tr.error}")
        obs_pieces: list[str] = []
        if tool_output_parts:
            obs_pieces.append("\n\n".join(tool_output_parts))
        if obs_result.text:
            obs_pieces.append(obs_result.text)
        combined_obs_text = "\n\n".join(obs_pieces)

        if parse.is_task_complete:
            state.consecutive_complete_signals += 1
            if pending_completion:
                observation_text = combined_obs_text
            else:
                pending_completion = True
                observation_text = self._completion_confirmation(combined_obs_text)
        else:
            state.consecutive_complete_signals = 0
            pending_completion = False
            if parse.warning:
                # Match Terminus2's feedback format: "WARNINGS: <text>"
                # (terminus_2.py:1196-1197, 1481-1487)
                observation_text = (
                    f"Previous response had warnings:\n"
                    f"WARNINGS: {parse.warning}\n\n"
                    f"{combined_obs_text}"
                )
            else:
                observation_text = combined_obs_text
        return observation_text, pending_completion

    def _next_prompt(
        self, observation_text: str, parse: Any, tools: ToolSet, ctx: ModuleCtx
    ) -> str:
        """The prompt fed to the next iteration: raw observation on a completion
        turn, else the tools' continuation-formatted prompt."""
        return (
            observation_text
            if parse.is_task_complete
            else tools.format_continuation_prompt(observation_text, ctx)
        )

    # ---------- Control-flow hooks ----------
    # Override ONE of these in a variant to make a FOCUSED structural change to
    # the control loop — instead of rewriting run() or injecting a static prompt.
    # Each default preserves baseline behavior exactly. A variant subclasses
    # BaselineAgentLoop and overrides one hook; per-iteration state that a hook
    # needs to remember can be stashed on `self` (the instance lives for the whole
    # run). This is the seam that lets the editor evolve real loop structure
    # (a plan phase, error-driven re-planning, adaptive termination) as ~30 lines.

    def _pre_llm_prompt(
        self, prompt: str, *, iteration: int, state: AgentLoopState, ctx: ModuleCtx
    ) -> str:
        """Shape the prompt just before each LLM call — planning injection,
        reminders, a re-plan trigger driven by `self`-stashed state. Default: no-op."""
        return prompt

    def _shape_observation(
        self,
        observation_text: str,
        *,
        parse: Any,
        tool_results: list[ToolResult],
        state: AgentLoopState,
        ctx: ModuleCtx,
    ) -> str:
        """Reshape what the model sees/records after executing this step — e.g.
        surface a verification/error reason so it isn't silently dropped.
        Default: unchanged."""
        return observation_text

    def _should_continue(
        self,
        *,
        terminate: bool,
        reason: str,
        parse: Any,
        iteration: int,
        state: AgentLoopState,
        ctx: ModuleCtx,
    ) -> bool:
        """Whether to keep looping after verification. Default: stop iff
        verification asked to terminate. A variant may force a re-plan instead of
        stopping, or stop early on its own evidence."""
        return not terminate

    # ---------- LLM call with retry + overflow fallback ----------

    def _sanitize_summary_handoff(self, handoff_prompt: str) -> str:
        """Model-neutral hook for cleaning a summarizer handoff.

        The shared baseline deliberately preserves the summary verbatim. A
        model-specific evaluation snapshot may override this hook when that
        model emits a known non-executable wire format during summarization.
        """
        return handoff_prompt.strip()

    def _protocol_safe_handoff(
        self,
        handoff_prompt: str,
        original_instruction: str,
        latest_terminal_prompt: str,
        tools: ToolSet,
        ctx: ModuleCtx,
    ) -> str:
        """Restore the selected tools module's contract after summarization.

        Summaries carry task progress, but they are not guaranteed to retain
        the active JSON/XML/editor response grammar. Rebuilding the initial
        prompt through the selected ToolSet restores that exact contract while
        retaining only a bounded tail of the latest terminal prompt.
        """
        clean = self._sanitize_summary_handoff(handoff_prompt)
        terminal_tail = latest_terminal_prompt[-_MAX_HANDOFF_TERMINAL_CHARS:]
        protocol_prompt = tools.format_initial_prompt(
            original_instruction,
            terminal_tail,
            ctx,
        )
        if not clean:
            return protocol_prompt
        return (
            f"Compacted progress summary:\n{clean}\n\n"
            f"Active task and response protocol:\n{protocol_prompt}"
        )

    @retry(
        stop=stop_after_attempt(3),
        retry=(
            retry_if_not_exception_type(ContextLengthExceededError)
            & retry_if_exception_type(Exception)
        ),
        reraise=True,
    )
    async def _query_llm(
        self,
        chat: Chat,
        prompt: str,
        ctx: ModuleCtx,
        context_mgmt: ContextMgmt,
        tools: ToolSet,
        original_instruction: str,
        logging_paths: tuple[Path | None, Path | None, Path | None],
    ) -> _LLMCallResult:
        """Wraps chat.chat with retry + ContextLengthExceededError fallback +
        OutputLengthExceededError handling. Mirrors `Terminus2._query_llm`.
        """
        _logging_path, prompt_path, response_path = logging_paths

        if prompt_path is not None:
            try:
                prompt_path.write_text(prompt)
            except Exception:
                pass

        try:
            start = time.time()
            llm_response = await chat.chat(prompt=prompt, **self.llm_call_kwargs)
            self.api_request_times.append((time.time() - start) * 1000)

            if response_path is not None:
                try:
                    response_path.write_text(llm_response.content)
                except Exception:
                    pass
            return _LLMCallResult(response=llm_response)

        except ContextLengthExceededError:
            ctx.services.logger.debug(
                "ContextLengthExceededError; falling back to summarization"
            )
            comp = await context_mgmt.force_summarize(chat, original_instruction, ctx)
            # Match stock Terminus-2 when summarization is disabled: it re-raises
            # the context error immediately.  Retrying the unchanged chat turned
            # one terminal overflow into a synthetic response and a 200-step spin.
            if not comp.summarization_occurred and comp.handoff_prompt is None:
                ctx.services.logger.debug(
                    "Context length exceeded and summarization is OFF."
                )
                raise
            summary_prompt = self._protocol_safe_handoff(
                comp.handoff_prompt or "Continue from the compacted state.",
                original_instruction,
                prompt,
                tools,
                ctx,
            )
            if prompt_path is not None:
                try:
                    prompt_path.write_text(summary_prompt)
                except Exception:
                    pass

            try:
                start = time.time()
                llm_response = await chat.chat(
                    prompt=summary_prompt, **self.llm_call_kwargs
                )
                self.api_request_times.append((time.time() - start) * 1000)
            except Exception as exc:
                ctx.services.logger.error(
                    "fallback chat failed after summarization: %s", exc
                )
                raise

            if response_path is not None:
                try:
                    response_path.write_text(llm_response.content)
                except Exception:
                    pass

            return _LLMCallResult(
                response=llm_response,
                pending_handoff_prompt=summary_prompt,
                pending_subagent_refs=comp.subagent_refs,
                summarization_occurred=comp.summarization_occurred,
            )

        except OutputLengthExceededError as exc:
            ctx.services.logger.debug("OutputLengthExceededError: %s", exc)

            truncated_response = getattr(
                exc, "truncated_response", "[TRUNCATED RESPONSE NOT AVAILABLE]"
            )

            # Try parser-specific salvage
            salvaged = None
            if hasattr(tools, "_parser") and hasattr(
                tools._parser, "salvage_truncated_response"
            ):
                try:
                    salvaged, _ = tools._parser.salvage_truncated_response(
                        truncated_response
                    )
                except Exception:
                    pass
            if salvaged:
                if response_path is not None:
                    try:
                        response_path.write_text(salvaged)
                    except Exception:
                        pass
                return _LLMCallResult(
                    response=LLMResponse(content=salvaged),
                )

            # Otherwise send retry prompt
            error_msg = (
                "ERROR!! NONE of the actions you just requested were performed "
                "because you exceeded the maximum output length. "
                "Re-issue this request, breaking it into smaller chunks."
            )
            chat.messages.append({"role": "user", "content": prompt})
            chat.messages.append({"role": "assistant", "content": truncated_response})
            chat.reset_response_chain()
            if response_path is not None:
                try:
                    response_path.write_text(error_msg)
                except Exception:
                    pass
            # Recursive retry with the error message (tenacity counts this as a
            # separate call; one nested level is fine).
            return await self._query_llm(
                chat=chat,
                prompt=error_msg,
                ctx=ctx,
                context_mgmt=context_mgmt,
                tools=tools,
                original_instruction=original_instruction,
                logging_paths=logging_paths,
            )

    # ---------- helpers ----------

    @staticmethod
    def _setup_episode_logging(
        logging_dir: Path | None, episode: int
    ) -> tuple[Path | None, Path | None, Path | None]:
        if logging_dir is None:
            return None, None, None
        try:
            ep_dir = logging_dir / f"episode-{episode}"
            ep_dir.mkdir(parents=True, exist_ok=True)
            return ep_dir / "debug.json", ep_dir / "prompt.txt", ep_dir / "response.txt"
        except Exception:
            return None, None, None

    @staticmethod
    def _completion_confirmation(terminal_output: str) -> str:
        return (
            "You declared the task complete. Please verify with a final check, "
            "then respond again. If still complete, emit another response with "
            "task_complete=true.\n\n"
            f"Latest terminal output:\n{terminal_output}"
        )

    def _build_message_content(self, parse, llm_response) -> str:
        if self.raw_content:
            return llm_response.content
        parts = []
        if parse.analysis:
            parts.append(f"Analysis: {parse.analysis}")
        if parse.plan:
            parts.append(f"Plan: {parse.plan}")
        return "\n".join(parts) if parts else ""

    def _build_metrics(self, chat: Chat, snapshot, llm_response) -> Metrics:
        ti, to, tc, cost = snapshot
        cache_used = chat.total_cache_tokens - tc
        step_cost = chat.total_cost - cost
        return Metrics(
            prompt_tokens=chat.total_input_tokens - ti,
            completion_tokens=chat.total_output_tokens - to,
            cached_tokens=cache_used if cache_used > 0 else None,
            cost_usd=step_cost if step_cost > 0 else None,
            prompt_token_ids=llm_response.prompt_token_ids,
            completion_token_ids=llm_response.completion_token_ids,
            logprobs=llm_response.logprobs,
        )

    def _append_agent_step(
        self,
        recorder,
        chat,
        llm_response,
        parse,
        observation_text: str,
        snapshot,
        step_episode: int,
        ctx: ModuleCtx,
    ) -> None:
        tool_calls = None
        obs_results = []
        if not self.raw_content:
            tool_calls_list = []
            if parse.commands:
                for i, cmd in enumerate(parse.commands):
                    tool_calls_list.append(
                        AtifToolCall(
                            tool_call_id=f"call_{step_episode}_{i + 1}",
                            function_name="bash_command",
                            arguments={
                                "keystrokes": cmd.keystrokes,
                                "duration": cmd.duration_sec,
                            },
                        )
                    )
                obs_results.append(ObservationResult(content=observation_text))
            if parse.is_task_complete:
                tool_calls_list.append(
                    AtifToolCall(
                        tool_call_id=f"call_{step_episode}_task_complete",
                        function_name="mark_task_complete",
                        arguments={},
                    )
                )
                if not parse.commands:
                    obs_results.append(ObservationResult(content=observation_text))
            elif not parse.commands:
                obs_results.append(ObservationResult(content=observation_text))
            tool_calls = tool_calls_list or None
        else:
            obs_results.append(ObservationResult(content=observation_text))

        step_id = len(recorder.steps) + 1
        recorder.append_step(
            Step(
                step_id=step_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
                source="agent",
                model_name=llm_response.model_name or ctx.state.model_name,
                message=(
                    llm_response.content
                    if self.raw_content
                    else self._build_message_content(parse, llm_response)
                ),
                reasoning_content=llm_response.reasoning_content,
                tool_calls=tool_calls,
                observation=AtifObservation(results=obs_results),
                metrics=self._build_metrics(chat, snapshot, llm_response),
            )
        )

    def _append_parse_error_step(
        self,
        recorder,
        chat,
        llm_response,
        next_prompt: str,
        snapshot,
        ctx: ModuleCtx,
    ) -> None:
        step_id = len(recorder.steps) + 1
        recorder.append_step(
            Step(
                step_id=step_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
                source="agent",
                model_name=llm_response.model_name or ctx.state.model_name,
                message=llm_response.content,
                reasoning_content=llm_response.reasoning_content,
                observation=AtifObservation(
                    results=[ObservationResult(content=next_prompt)]
                ),
                metrics=self._build_metrics(chat, snapshot, llm_response),
            )
        )


def register(library):
    library.register(
        type_="agent_loop",
        name=BaselineAgentLoop.NAME,
        factory=lambda params: BaselineAgentLoop(
            max_iterations=int(params.get("max_iterations", _DEFAULT_MAX_ITERATIONS)),
            llm_call_kwargs=params.get("llm_call_kwargs") or {},
            raw_content=bool(params.get("raw_content", False)),
        ),
        description=BaselineAgentLoop.DESCRIPTION,
        params_schema=BaselineAgentLoop.PARAMS_SCHEMA,
        niche=BaselineAgentLoop.NICHE,
    )
