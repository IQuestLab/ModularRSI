"""Minimal tools module: env.exec + simple regex XML parser.

This was the v1 MVP scaffolding. It's strictly weaker than `tmux_full` (no
real tmux, no auto-fix parser, no asciinema). Kept in the library as an
ALTERNATIVE the Composer can pick (e.g., for unit-test scenarios where we
don't want a real tmux session).

LLM response grammar (simplified):
    <response>
      <analysis>...</analysis>
      <plan>...</plan>
      <commands>
        <keystrokes>bash command 1</keystrokes>
        <keystrokes>bash command 2</keystrokes>
      </commands>
      <task_complete>true|false</task_complete>
    </response>
"""

from __future__ import annotations

import re

from harbor.agents.terminus_2_modular.protocols import (
    LLMResponseParseResult,
    ModuleCtx,
    ToolCall,
    ToolResult,
)

_SYSTEM_PROMPT = """\
You can execute shell commands in a Linux environment. Respond ONLY in this XML format:

<response>
  <analysis>What you see in the terminal so far. What you've learned.</analysis>
  <plan>What you'll do next, in one or two sentences.</plan>
  <commands>
    <keystrokes>command 1 here</keystrokes>
    <keystrokes>command 2 here</keystrokes>
  </commands>
  <task_complete>false</task_complete>
</response>

Rules:
- Each <keystrokes> block is one shell command, executed sequentially.
- Set <task_complete>true</task_complete> ONLY when the entire task is done and verified.
- Do not output anything outside the <response>...</response> block.

# Task
{instruction}

Current terminal state:
{terminal_state}
"""

_RESP = re.compile(r"<response>(.*?)</response>", re.DOTALL | re.IGNORECASE)
_KEYS = re.compile(r"<keystrokes>(.*?)</keystrokes>", re.DOTALL | re.IGNORECASE)
_DONE = re.compile(
    r"<task_complete>\s*(true|false)\s*</task_complete>",
    re.IGNORECASE,
)
_ANALYSIS = re.compile(r"<analysis>(.*?)</analysis>", re.DOTALL | re.IGNORECASE)
_PLAN = re.compile(r"<plan>(.*?)</plan>", re.DOTALL | re.IGNORECASE)


class TmuxXMLTools:
    NAME = "tmux_xml"
    NICHE = {"transport": "env-exec", "encoding": "xml"}
    DESCRIPTION = (
        "Minimal tools: env.exec + simplistic regex XML parser. No real tmux, "
        "no asciinema. Weaker than tmux_full; useful as a unit-test default."
    )
    PARAMS_SCHEMA = {"timeout_sec": "int (default 30)"}

    def __init__(self, timeout_sec: int = 30):
        self.timeout_sec = timeout_sec

    # ---------- Lifecycle (no-op for this module) ----------

    async def setup(self, ctx: ModuleCtx) -> None:
        return

    async def teardown(self, ctx: ModuleCtx) -> None:
        return

    # ---------- Prompt formatting ----------

    def format_initial_prompt(
        self, instruction: str, terminal_state: str, ctx: ModuleCtx
    ) -> str:
        return _SYSTEM_PROMPT.format(
            instruction=instruction, terminal_state=terminal_state
        )

    def format_continuation_prompt(self, terminal_state: str, ctx: ModuleCtx) -> str:
        return (
            f"Terminal output from previous commands:\n{terminal_state}\n\n"
            f"Continue. Emit one <response>...</response> block."
        )

    # ---------- LLM response parsing ----------

    def parse_llm_response(self, response: str) -> LLMResponseParseResult:
        m = _RESP.search(response)
        body = m.group(1) if m else response
        commands = [k.group(1).strip() for k in _KEYS.finditer(body)]
        done_m = _DONE.search(body)
        is_complete = done_m is not None and done_m.group(1).lower() == "true"
        a = _ANALYSIS.search(body)
        p = _PLAN.search(body)
        return LLMResponseParseResult(
            commands=[ToolCall(keystrokes=c) for c in commands if c],
            is_task_complete=is_complete,
            analysis=a.group(1).strip() if a else "",
            plan=p.group(1).strip() if p else "",
        )

    # ---------- Execution ----------

    async def execute(self, call: ToolCall, ctx: ModuleCtx) -> ToolResult:
        cmd = call.keystrokes.strip()
        if not cmd:
            return ToolResult(success=False, output="", error="Empty command")
        env = ctx.state.env
        try:
            result = await env.exec(command=cmd, timeout_sec=self.timeout_sec)
        except Exception as exc:
            ctx.services.failures.raise_tag(
                "tool_exec_exception", "tools.tmux_xml", str(exc)
            )
            return ToolResult(success=False, output="", error=str(exc))

        rc = result.return_code
        stdout = result.stdout or ""
        stderr = result.stderr or ""
        success = rc == 0
        combined = stdout
        if stderr:
            combined = f"{stdout}\n[stderr]\n{stderr}" if stdout else stderr
        return ToolResult(
            success=success,
            output=combined,
            error="" if success else f"return_code={rc}",
        )


def register(library):
    library.register(
        type_="tools",
        name=TmuxXMLTools.NAME,
        factory=lambda params: TmuxXMLTools(
            timeout_sec=int(params.get("timeout_sec", 30))
        ),
        description=TmuxXMLTools.DESCRIPTION,
        params_schema=TmuxXMLTools.PARAMS_SCHEMA,
        niche=TmuxXMLTools.NICHE,
    )
