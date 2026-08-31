"""Parse-error recovery loop — breaks format-death spirals without extra guards.

Baseline agent_loop with enhanced parse-error recovery: stashes the raw LLM
response so the retry prompt can show the model what it actually wrote wrong,
and escalates to a strict, unambiguous format template after N consecutive parse
failures (default 3). This prevents the infinite parse-error death spiral seen
in tasks where the model regenerates the same malformed output because the retry
prompt never changes shape.

Does NOT include:
- Evidence-gated termination (empty completion rejection)
- Repetition detection
- Re-orientation injection
- Any other behavior change beyond parse-error recovery.

Addresses failure modes where the agent:
  - enters an infinite parse-error loop (the same malformed output repeated
    because the retry prompt doesn't show what it wrote wrong)
  - repeatedly produces malformed JSON/XML without being able to see its own
    mistake
  - gets stuck in a format deadlock that a mere "Please fix these issues"
    reminder cannot break
"""

from typing import Any

from harbor.agents.terminus_2_modular.modules.agent_loop.baseline import (
    BaselineAgentLoop,
)
from harbor.agents.terminus_2_modular.protocols import (
    AgentLoopState,
    ModuleCtx,
    ToolSet,
)


class ParseErrorRecoveryLoop(BaselineAgentLoop):
    NAME = "parse_error_recovery"
    NICHE = {"grounding": "reactive", "drift": "none", "parse": "robust-json"}
    DESCRIPTION = (
        "Baseline + parse-error recovery with raw-output feedback and strict-format "
        "escalation after consecutive failures. Stashes the raw LLM response in "
        "_query_llm and shows it in the retry prompt (truncated to "
        "max_raw_output_chars) so the model can see what it wrote wrong. After "
        "max_parse_error_escalation (default 3) consecutive parse errors, the retry "
        "prompt switches to a strict, minimal-format template that breaks format "
        "deadlocks. Does NOT include empty-completion guards, repetition detection, "
        "or re-orientation injection — it is a focused parse-error fix only."
    )
    PARAMS_SCHEMA = {
        "max_raw_output_chars": (
            "int (default 2000) — truncate raw LLM output to this many characters "
            "when showing in the parse-error retry prompt, to avoid context flooding"
        ),
        "max_parse_error_escalation": (
            "int (default 3) — after this many CONSECUTIVE parse errors, the retry "
            "prompt switches to a strict, minimal response-format instruction with "
            "an explicit template, to break malformed-output deadlocks"
        ),
    }

    def __init__(
        self,
        max_iterations: int = 1_000_000,
        llm_call_kwargs: dict | None = None,
        raw_content: bool = False,
        max_raw_output_chars: int = 2000,
        max_parse_error_escalation: int = 3,
    ):
        super().__init__(
            max_iterations=max_iterations,
            llm_call_kwargs=llm_call_kwargs,
            raw_content=raw_content,
        )
        self._last_raw_response: str | None = None
        self.max_raw_output_chars = max_raw_output_chars
        self._consecutive_parse_errors: int = 0
        self.max_parse_error_escalation = max_parse_error_escalation

    # ---------- Parse-error recovery ----------

    async def _query_llm(
        self,
        chat,
        prompt,
        ctx,
        context_mgmt,
        tools,
        original_instruction,
        logging_paths,
    ) -> Any:
        """Override: stash the final LLM response so _build_parse_error_prompt can
        include the raw output in the retry message."""
        result = await super()._query_llm(
            chat=chat,
            prompt=prompt,
            ctx=ctx,
            context_mgmt=context_mgmt,
            tools=tools,
            original_instruction=original_instruction,
            logging_paths=logging_paths,
        )
        self._last_raw_response = result.response.content
        return result

    def _build_parse_error_prompt(self, parse: Any, tools: ToolSet) -> str:
        """Build the retry prompt after a parse failure, including the raw LLM
        output (truncated to max_raw_output_chars) so the model can see what it
        wrote wrong without flooding the context window.

        After max_parse_error_escalation CONSECUTIVE parse failures the retry
        escalates to a strict, minimal-format instruction: the model must output
        exactly one response object with no prose or markdown. This breaks the
        deadlock where the same malformed generation pattern repeats forever
        because the retry prompt never changes shape."""
        # Count consecutive failures: this hook is only reached on the
        # parse-error path, and _shape_observation resets the counter after the
        # first successful parse.
        self._consecutive_parse_errors += 1

        feedback = f"ERROR: {parse.error}"
        if parse.warning:
            feedback += f"\nWARNINGS: {parse.warning}"
        if self._last_raw_response:
            raw = self._last_raw_response
            if len(raw) > self.max_raw_output_chars:
                raw = "...[TRUNCATED]...\n" + raw[-self.max_raw_output_chars :]
            feedback += f"\n\nYour actual response was:\n{raw}"

        parser_name = getattr(tools, "_parser_name", "JSON")
        is_xml = isinstance(parser_name, str) and parser_name.lower() == "xml"
        if is_xml:
            response_type = "XML response with all required fields"
        else:
            response_type = "JSON response with all required fields"

        if self._consecutive_parse_errors >= self.max_parse_error_escalation:
            return (
                f"Previous response had parsing errors:\n{feedback}\n\n"
                f"{self._strict_format_instruction(is_xml=is_xml)}"
            )
        return (
            f"Previous response had parsing errors:\n{feedback}\n\n"
            f"Please fix these issues and provide a proper {response_type}."
        )

    @staticmethod
    def _strict_format_instruction(*, is_xml: bool) -> str:
        """A maximally literal formatting instruction used after repeated parse
        errors. Matches the grammar taught by the tools' system prompt so a model
        that has fallen into a malformed-output rut gets an unambiguous template."""
        left_angle, right_angle = (
            chr(60),
            chr(62),
        )  # angle brackets, kept out of this source text
        if is_xml:
            template = (
                f"{left_angle}response{right_angle}\n"
                f"  {left_angle}analysis{right_angle}"
                f"What you see and what you've learned."
                f"{left_angle}/analysis{right_angle}\n"
                f"  {left_angle}plan{right_angle}"
                f"What you'll do next, in one or two sentences."
                f"{left_angle}/plan{right_angle}\n"
                f"  {left_angle}commands{right_angle}\n"
                f"    {left_angle}keystrokes{right_angle}command 1 here"
                f"{left_angle}/keystrokes{right_angle}\n"
                f"  {left_angle}/commands{right_angle}\n"
                f"  {left_angle}task_complete{right_angle}false"
                f"{left_angle}/task_complete{right_angle}\n"
                f"{left_angle}/response{right_angle}"
            )
        else:
            template = (
                "{\n"
                '  "analysis": "What you see and what you\'ve learned.",\n'
                '  "plan": "What you\'ll do next, in one or two sentences.",\n'
                '  "commands": [\n'
                '    {"command": "command 1 here"}\n'
                "  ],\n"
                '  "task_complete": false\n'
                "}"
            )
        return (
            "SYSTEM — FORMAT ESCALATION: Your previous responses could not be "
            "parsed at all. From now on your ENTIRE response must be EXACTLY ONE "
            f"{'XML' if is_xml else 'JSON'} object. No explanation, no markdown "
            "fences, no text outside the object. Copy this template and fill in "
            "the values:\n\n"
            f"{template}\n\n"
            "Do not add anything before or after it."
        )

    # ---------- Termination (unchanged from baseline) ----------

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
        return not terminate

    # ---------- Observe hook: reset parse-error counter on success ----------

    def _shape_observation(
        self,
        observation_text: str,
        *,
        parse: Any,
        tool_results: list,
        state: AgentLoopState,
        ctx: ModuleCtx,
    ) -> str:
        """Reset consecutive parse error counter on the first successful parse
        after any error streak, so an isolated parse error later starts from the
        normal retry prompt."""
        # A successful non-empty step means the model is not in a parse-error
        # rut anymore — reset the escalation streak.
        self._consecutive_parse_errors = 0
        return observation_text


def register(library):
    library.register(
        type_="agent_loop",
        name=ParseErrorRecoveryLoop.NAME,
        factory=lambda params: ParseErrorRecoveryLoop(
            max_iterations=int(params.get("max_iterations", 1_000_000)),
            llm_call_kwargs=params.get("llm_call_kwargs") or {},
            raw_content=bool(params.get("raw_content", False)),
            max_raw_output_chars=int(params.get("max_raw_output_chars", 2000)),
            max_parse_error_escalation=int(params.get("max_parse_error_escalation", 3)),
        ),
        description=ParseErrorRecoveryLoop.DESCRIPTION,
        params_schema=ParseErrorRecoveryLoop.PARAMS_SCHEMA,
        niche=ParseErrorRecoveryLoop.NICHE,
    )
