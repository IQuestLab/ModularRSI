"""3-step QA context summarization (faithful port of Terminus2._summarize).

When chat history grows past `proactive_summarization_threshold` tokens below
the model's context limit, run a three-subagent QA pipeline to compress the
history while preserving critical info:

  1. **Summary subagent**:    given full history → comprehensive summary
  2. **Questions subagent**:  given (original task + summary + terminal screen)
                              with fresh chat → list of clarifying questions
  3. **Answers subagent**:    given (history + summary + questions) → detailed
                              answers

Final state:
- `chat._messages` is replaced with [system, question_prompt, model_questions]
- Returned `handoff_prompt` contains the answers + continuation instruction
  (which agent_loop will use as the next user prompt)

This module also handles:
- `_unwind_messages_to_free_tokens`: drop tail message-pairs when context is full
- subagent metrics accumulation (queried by agent.py at end of run)
- subagent rollout details accumulation
- per-subagent trajectory JSON files in `<logs_dir>/trajectory.summarization-N-*.json`
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from harbor.agents.terminus_2_modular.protocols import CompressResult, ModuleCtx
from harbor.llms.base import LLMResponse
from harbor.llms.chat import Chat
from harbor.models.trajectories import (
    Agent,
    FinalMetrics,
    Metrics,
    Step,
    SubagentTrajectoryRef,
    Trajectory,
)
from harbor.utils.trajectory_utils import format_trajectory_json


@dataclass
class _SubagentMetrics:
    """Mirrors Terminus2.SubagentMetrics."""

    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_cached_tokens: int = 0
    total_cost_usd: float = 0.0


class BaselineContextMgmt:
    NAME = "baseline"
    NICHE = {"strategy": "qa-summarize"}
    DESCRIPTION = (
        "3-step QA summarization (summary → questions → answers). Triggers "
        "when free_tokens < proactive_summarization_threshold. Faithful port "
        "of Terminus2._summarize."
    )
    PARAMS_SCHEMA = {
        "enable_summarize": "bool (default True)",
        "proactive_summarization_threshold": "int (default 8000)",
        "unwind_target_free_tokens": "int (default 4000)",
        "linear_history": "bool (default False; future use for trajectory split)",
    }

    def __init__(
        self,
        enable_summarize: bool = True,
        proactive_summarization_threshold: int = 8000,
        unwind_target_free_tokens: int = 4000,
        linear_history: bool = False,
        **_ignored,
    ):
        self.enable_summarize = enable_summarize
        self.proactive_threshold = proactive_summarization_threshold
        self.unwind_target = unwind_target_free_tokens
        self.linear_history = linear_history
        # Cross-summarization state
        self.summarization_count: int = 0
        self._metrics = _SubagentMetrics()
        self._rollout_details: list[dict] = []
        # Track which model produced the most recent LLM response (for step
        # serialization of copied messages)
        self._last_response_model_name: str | None = None

    # ---------- Public Protocol ----------

    async def maybe_compress(
        self,
        chat: Chat,
        original_instruction: str,
        ctx: ModuleCtx,
    ) -> CompressResult:
        if not self.enable_summarize:
            return CompressResult(chat=chat)
        if not original_instruction:
            return CompressResult(chat=chat)

        try:
            free_tokens = self._estimate_free_tokens(chat)
        except Exception as exc:
            ctx.services.logger.warning(
                "three_step_qa: estimating free tokens failed: %s; skipping",
                exc,
            )
            return CompressResult(chat=chat)

        if free_tokens >= self.proactive_threshold:
            return CompressResult(chat=chat)

        ctx.services.logger.debug(
            "three_step_qa: proactive summarize (free=%s, threshold=%s)",
            free_tokens,
            self.proactive_threshold,
        )
        try:
            handoff_prompt, refs = await self._summarize(
                chat, original_instruction, ctx
            )
        except Exception as exc:
            ctx.services.logger.error("three_step_qa: summarization failed: %s", exc)
            return CompressResult(chat=chat)

        return CompressResult(
            chat=chat,
            handoff_prompt=handoff_prompt,
            subagent_refs=refs,
            summarization_occurred=True,
        )

    async def force_summarize(
        self,
        chat: Chat,
        original_instruction: str,
        ctx: ModuleCtx,
    ) -> CompressResult:
        """Reactive summarization on ContextLengthExceededError.

        Mirrors `Terminus2._query_llm` overflow path (terminus_2.py:1021-1075):
        1. Unwind messages to free 4000 tokens
        2. Try full 3-step QA summary
        3. Fallback: short 1-LLM-call summary
        4. Ultimate fallback: no LLM, just use last terminal screen
        """
        if not self.enable_summarize:
            return CompressResult(chat=chat)

        session = ctx.shared.tmux_session
        self._unwind_messages_to_free_tokens(chat, target_free_tokens=4000)

        # Tier 1: full 3-step QA
        try:
            ctx.services.logger.debug("three_step_qa: force_summarize tier 1 (full)")
            handoff_prompt, refs = await self._summarize(
                chat, original_instruction, ctx
            )
            return CompressResult(
                chat=chat,
                handoff_prompt=handoff_prompt,
                subagent_refs=refs,
                summarization_occurred=True,
            )
        except Exception as exc:
            ctx.services.logger.debug(
                "three_step_qa: tier 1 failed: %s; trying tier 2", exc
            )

        # Tier 2: short summary (single LLM call)
        try:
            ctx.services.logger.debug("three_step_qa: force_summarize tier 2 (short)")
            current_screen = ""
            if session is not None:
                try:
                    current_screen = await session.capture_pane(capture_entire=False)
                except Exception:
                    pass
            limited_screen = current_screen[-1000:] if current_screen else ""
            short_prompt = (
                f"Briefly continue this task: {original_instruction}\n\n"
                f"Current state: {limited_screen}\n\n"
                "Next steps (2-3 sentences):"
            )
            short_response = await chat._model.call(prompt=short_prompt)
            self._update_subagent_metrics(short_response.usage)
            self._collect_subagent_rollout_detail(short_response)
            handoff_prompt = (
                f"{original_instruction}\n\nSummary: {short_response.content}"
            )
            return CompressResult(
                chat=chat,
                handoff_prompt=handoff_prompt,
                subagent_refs=None,
                summarization_occurred=True,
            )
        except Exception as exc:
            ctx.services.logger.error(
                "three_step_qa: tier 2 failed: %s; using tier 3", exc
            )

        # Tier 3: ultimate fallback — no LLM, just terminal screen
        current_screen = ""
        if session is not None:
            try:
                current_screen = await session.capture_pane(capture_entire=False)
            except Exception:
                pass
        limited_screen = current_screen[-1000:] if current_screen else ""
        handoff_prompt = f"{original_instruction}\n\nCurrent state: {limited_screen}"
        return CompressResult(
            chat=chat,
            handoff_prompt=handoff_prompt,
            subagent_refs=None,
            summarization_occurred=True,
        )

    def get_subagent_metrics(self) -> dict[str, Any]:
        return {
            "total_prompt_tokens": self._metrics.total_prompt_tokens,
            "total_completion_tokens": self._metrics.total_completion_tokens,
            "total_cached_tokens": self._metrics.total_cached_tokens,
            "total_cost_usd": self._metrics.total_cost_usd,
            "rollout_details": list(self._rollout_details),
            "summarization_count": self.summarization_count,
        }

    # ---------- Internals ----------

    @staticmethod
    def _count_total_tokens(chat: Chat) -> int:
        """Rough token count using the underlying LLM if it exposes it."""
        model = chat._model
        if hasattr(model, "count_message_tokens"):
            return model.count_message_tokens(chat.messages)
        # Crude fallback: 4 chars per token
        total_chars = sum(len(m.get("content", "") or "") for m in chat.messages)
        return total_chars // 4

    @staticmethod
    def _get_context_limit(chat: Chat) -> int:
        model = chat._model
        if hasattr(model, "get_model_context_limit"):
            return model.get_model_context_limit()
        return 200_000  # safe-ish default

    def _estimate_free_tokens(self, chat: Chat) -> int:
        return self._get_context_limit(chat) - self._count_total_tokens(chat)

    def _unwind_messages_to_free_tokens(
        self, chat: Chat, target_free_tokens: int | None = None
    ) -> None:
        target = (
            target_free_tokens if target_free_tokens is not None else self.unwind_target
        )
        context_limit = self._get_context_limit(chat)
        while len(chat.messages) > 1:
            current_tokens = self._count_total_tokens(chat)
            free_tokens = context_limit - current_tokens
            if free_tokens >= target:
                break
            if len(chat.messages) >= 2:
                chat._messages = chat.messages[:-2]
            else:
                break
        chat.reset_response_chain()

    # ---- subagent helpers ----

    def _update_subagent_metrics(self, usage_info) -> None:
        if not usage_info:
            return
        self._metrics.total_prompt_tokens += usage_info.prompt_tokens
        self._metrics.total_completion_tokens += usage_info.completion_tokens
        self._metrics.total_cached_tokens += usage_info.cache_tokens
        self._metrics.total_cost_usd += usage_info.cost_usd

    @staticmethod
    def _extract_usage_metrics(usage_info):
        if not usage_info:
            return 0, 0, 0, 0
        return (
            usage_info.prompt_tokens,
            usage_info.completion_tokens,
            usage_info.cache_tokens,
            usage_info.cost_usd if usage_info.cost_usd > 0 else 0,
        )

    def _collect_subagent_rollout_detail(
        self, response: LLMResponse, collect: bool = True
    ) -> None:
        if not collect:
            return
        detail: dict = {}
        if response.prompt_token_ids is not None:
            detail["prompt_token_ids"] = [response.prompt_token_ids]
        if response.completion_token_ids is not None:
            detail["completion_token_ids"] = [response.completion_token_ids]
        if response.logprobs is not None:
            detail["logprobs"] = [response.logprobs]
        if response.extra is not None:
            detail["extra"] = {k: [v] for k, v in response.extra.items()}
        if detail:
            self._rollout_details.append(detail)

    @staticmethod
    def _remove_metrics_from_copied_steps(steps: list[Step]) -> None:
        for step in steps:
            step.is_copied_context = True
            if step.metrics:
                step.metrics = None
                if step.extra is None:
                    step.extra = {}
                step.extra["note"] = (
                    "Metrics omitted to avoid duplication - already recorded "
                    "in parent trajectory"
                )

    def _prepare_copied_trajectory_steps(
        self, parent_steps: list[Step], steps_to_include: int
    ) -> tuple[list[Step], int]:
        copied = copy.deepcopy(parent_steps[:steps_to_include])
        self._remove_metrics_from_copied_steps(copied)
        return copied, len(copied) + 1

    def _append_subagent_response_step(
        self,
        steps: list[Step],
        step_id: int,
        response: LLMResponse,
        usage_info,
        model_name_fallback: str,
        logger,
    ) -> None:
        if usage_info:
            steps.append(
                Step(
                    step_id=step_id,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    source="agent",
                    model_name=response.model_name or model_name_fallback,
                    message=response.content,
                    reasoning_content=response.reasoning_content,
                    metrics=Metrics(
                        prompt_tokens=usage_info.prompt_tokens,
                        completion_tokens=usage_info.completion_tokens,
                        cached_tokens=usage_info.cache_tokens,
                        cost_usd=usage_info.cost_usd
                        if usage_info.cost_usd > 0
                        else None,
                        prompt_token_ids=response.prompt_token_ids,
                        completion_token_ids=response.completion_token_ids,
                        logprobs=response.logprobs,
                    ),
                )
            )
        else:
            logger.warning("subagent: no usage info for response")
            steps.append(
                Step(
                    step_id=step_id,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    source="agent",
                    model_name=response.model_name or model_name_fallback,
                    message=response.content,
                    reasoning_content=response.reasoning_content,
                )
            )

    def _save_subagent_trajectory(
        self,
        ctx: ModuleCtx,
        session_id: str,
        agent_name: str,
        steps: list[Step],
        usage_info,
        filename_suffix: str,
        summary_text: str,
    ) -> SubagentTrajectoryRef:
        total_prompt, total_completion, total_cached, total_cost = (
            self._extract_usage_metrics(usage_info)
        )
        trajectory = Trajectory(
            session_id=session_id,
            agent=Agent(
                name=agent_name,
                version="0.3.0",  # match Terminus2Modular.version()
                model_name=ctx.state.model_name,
                extra={
                    "parent_session_id": ctx.state.task_id,
                    "summarization_index": self.summarization_count,
                },
            ),
            steps=steps,
            final_metrics=FinalMetrics(
                total_prompt_tokens=total_prompt,
                total_completion_tokens=total_completion,
                total_cached_tokens=total_cached,
                total_cost_usd=total_cost if total_cost > 0 else None,
            ),
        )
        logs_dir = Path(ctx.state.logs_dir)
        traj_path = (
            logs_dir / f"trajectory.summarization-{self.summarization_count}"
            f"-{filename_suffix}.json"
        )
        try:
            traj_path.write_text(format_trajectory_json(trajectory.to_json_dict()))
        except Exception as exc:
            ctx.services.logger.error(
                "Failed to save %s subagent trajectory: %s",
                filename_suffix,
                exc,
            )
        return SubagentTrajectoryRef(
            session_id=session_id,
            trajectory_path=traj_path.name,
            extra={"summary": summary_text},
        )

    async def _run_subagent(
        self,
        chat: Chat,
        ctx: ModuleCtx,
        prompt: str,
        message_history: list[dict],
        steps: list[Step],
        session_id: str,
        agent_name: str,
        filename_suffix: str,
        summary_text: str,
        llm_call_kwargs: dict,
    ) -> tuple[LLMResponse, SubagentTrajectoryRef]:
        # User-prompt step
        prompt_step_id = len(steps) + 1
        steps.append(
            Step(
                step_id=prompt_step_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
                source="user",
                message=prompt,
            )
        )
        response_step_id = prompt_step_id + 1

        # Call LLM directly (not chat — this is a subagent, no history mutation)
        llm = chat._model
        start = time.time()
        response: LLMResponse = await llm.call(
            prompt=prompt,
            message_history=message_history,
            **(llm_call_kwargs or {}),
        )
        elapsed_ms = (time.time() - start) * 1000
        ctx.services.logger.debug(
            "three_step_qa: subagent %s LLM call %.0fms",
            filename_suffix,
            elapsed_ms,
        )

        usage_info = response.usage
        self._update_subagent_metrics(usage_info)
        self._append_subagent_response_step(
            steps,
            response_step_id,
            response,
            usage_info,
            model_name_fallback=ctx.state.model_name,
            logger=ctx.services.logger,
        )
        # rollout details collection only matters if parent agent asked for it;
        # we always accumulate so agent.py can decide whether to surface
        self._collect_subagent_rollout_detail(response)

        traj_ref = self._save_subagent_trajectory(
            ctx=ctx,
            session_id=session_id,
            agent_name=agent_name,
            steps=steps,
            usage_info=usage_info,
            filename_suffix=filename_suffix,
            summary_text=summary_text,
        )
        return response, traj_ref

    # ---- main 3-step QA ----

    async def _summarize(
        self,
        chat: Chat,
        original_instruction: str,
        ctx: ModuleCtx,
    ) -> tuple[str, list[SubagentTrajectoryRef] | None]:
        if not chat.messages:
            return original_instruction, None

        # Free up tokens so subagent LLM calls fit
        self._unwind_messages_to_free_tokens(chat)

        self.summarization_count += 1
        refs: list[SubagentTrajectoryRef] = []
        session = ctx.shared.tmux_session

        # Snapshot parent trajectory steps (for "copied context" in subagent traj)
        parent_steps_snapshot: list[Step] = list(
            getattr(ctx.services.trajectory, "steps", [])
        )

        # ----- SUBAGENT 1: Summary Generation -----
        steps_to_include = 1 + (len(chat.messages) - 1) // 2
        summary_steps, _ = self._prepare_copied_trajectory_steps(
            parent_steps_snapshot, steps_to_include
        )

        summary_prompt = f"""You are about to hand off your work to another AI agent.
            Please provide a comprehensive summary of what you have
            accomplished so far on this task:

Original Task: {original_instruction}

Based on the conversation history, please provide a detailed summary covering:
1. **Major Actions Completed** - List each significant command you executed
            and what you learned from it.
2. **Important Information Learned** - A summary of crucial findings, file
            locations, configurations, error messages, or system state discovered.
3. **Challenging Problems Addressed** - Any significant issues you
            encountered and how you resolved them.
4. **Current Status** - Exactly where you are in the task completion process.


Be comprehensive and detailed. The next agent needs to understand everything
            that has happened so far in order to continue."""

        summary_session_id = (
            f"{ctx.state.task_id}-summarization-{self.summarization_count}-summary"
        )
        summary_response, summary_ref = await self._run_subagent(
            chat=chat,
            ctx=ctx,
            prompt=summary_prompt,
            message_history=chat.messages,
            steps=summary_steps,
            session_id=summary_session_id,
            agent_name="terminus-2-summarization-summary",
            filename_suffix="summary",
            summary_text=(
                f"Context summarization {self.summarization_count}: "
                "Step 1 - Summary generation"
            ),
            llm_call_kwargs={},
        )
        refs.append(summary_ref)

        # ----- SUBAGENT 2: Question Asking -----
        if session is not None:
            try:
                current_screen = await session.capture_pane(capture_entire=False)
            except Exception as exc:
                ctx.services.logger.warning(
                    "three_step_qa: capture_pane failed: %s; using empty screen",
                    exc,
                )
                current_screen = ""
        else:
            current_screen = ""

        question_prompt = f"""You are picking up work from a previous AI agent on this task:

**Original Task:** {original_instruction}

**Summary from Previous Agent:**
{summary_response.content}

**Current Terminal Screen:**
{current_screen}

Please begin by asking several questions (at least five, more if necessary)
about the current state of the solution that are not answered in the summary
from the prior agent. After you ask these questions you will be on your own,
so ask everything you need to know."""

        questions_session_id = (
            f"{ctx.state.task_id}-summarization-{self.summarization_count}-questions"
        )
        questions_steps: list[Step] = []
        questions_response, questions_ref = await self._run_subagent(
            chat=chat,
            ctx=ctx,
            prompt=question_prompt,
            message_history=[],
            steps=questions_steps,
            session_id=questions_session_id,
            agent_name="terminus-2-summarization-questions",
            filename_suffix="questions",
            summary_text=(
                f"Context summarization {self.summarization_count}: "
                "Step 2 - Question asking"
            ),
            llm_call_kwargs={},
        )
        model_questions = questions_response.content
        refs.append(questions_ref)

        # ----- SUBAGENT 3: Answer Providing -----
        answers_steps, step_id_counter = self._prepare_copied_trajectory_steps(
            parent_steps_snapshot, steps_to_include
        )
        # Inject summary prompt + summary response as copied context
        answers_steps.append(
            Step(
                step_id=step_id_counter,
                timestamp=datetime.now(timezone.utc).isoformat(),
                source="user",
                message=summary_prompt,
                is_copied_context=True,
            )
        )
        step_id_counter += 1
        answers_steps.append(
            Step(
                step_id=step_id_counter,
                timestamp=datetime.now(timezone.utc).isoformat(),
                source="agent",
                model_name=summary_response.model_name or ctx.state.model_name,
                message=summary_response.content,
                reasoning_content=summary_response.reasoning_content,
                is_copied_context=True,
                extra={
                    "note": (
                        "Copied from summary subagent - metrics already recorded there"
                    )
                },
            )
        )
        step_id_counter += 1

        answer_request_prompt = (
            "The next agent has a few questions for you, please answer each "
            "of them one by one in detail:\n\n" + model_questions
        )
        answers_message_history = chat.messages + [
            {"role": "user", "content": summary_prompt},
            {"role": "assistant", "content": summary_response.content},
        ]

        answers_session_id = (
            f"{ctx.state.task_id}-summarization-{self.summarization_count}-answers"
        )
        answers_response, answers_ref = await self._run_subagent(
            chat=chat,
            ctx=ctx,
            prompt=answer_request_prompt,
            message_history=answers_message_history,
            steps=answers_steps,
            session_id=answers_session_id,
            agent_name="terminus-2-summarization-answers",
            filename_suffix="answers",
            summary_text=(
                f"Context summarization {self.summarization_count}: "
                "Step 3 - Answer providing"
            ),
            llm_call_kwargs={},
        )
        refs.append(answers_ref)

        # Replace chat history with [system, question_prompt, model_questions]
        # Mirrors the original behavior exactly.
        chat._messages = [
            chat.messages[0],
            {"role": "user", "content": question_prompt},
            {"role": "assistant", "content": model_questions},
        ]
        chat.reset_response_chain()

        handoff_prompt = (
            "Here are the answers the other agent provided.\n\n"
            + answers_response.content
            + "\n\n"
            + "Continue working on this task from where the previous agent "
            "left off. You can no longer ask questions. Please follow the "
            "spec to interact with the terminal."
        )

        return handoff_prompt, refs


def register(library):
    library.register(
        type_="context_mgmt",
        name=BaselineContextMgmt.NAME,
        factory=lambda params: BaselineContextMgmt(**params),
        description=BaselineContextMgmt.DESCRIPTION,
        params_schema=BaselineContextMgmt.PARAMS_SCHEMA,
        niche=BaselineContextMgmt.NICHE,
    )
