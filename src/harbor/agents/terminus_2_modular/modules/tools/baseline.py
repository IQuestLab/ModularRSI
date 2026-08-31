"""Full-fidelity tmux tools module: faithful port of terminus-2's tmux + parser
+ prompt-template behavior.

Components reused as-is from terminus_2/:
- TmuxSession (real tmux + asciinema + incremental output)
- TerminusXMLPlainParser / TerminusJSONPlainParser (with auto-fix logic)
- prompt template files (templates/terminus-xml-plain.txt / -json-plain.txt)
- skills frontmatter parser

Lifecycle:
- setup(ctx)    → builds TmuxSession from ctx.state, stashes in ctx.shared.tmux_session,
                  starts the session.
- teardown(ctx) → stops the session.
- execute(call) → sends keystrokes via session.send_keys(keystrokes, duration).
- parse_llm_response(text) → delegates to real parser, returns LLMResponseParseResult.
- format_initial_prompt(instr, term, ctx) → reads template, injects MCP info + skills XML.
- format_continuation_prompt(term, ctx)  → same template with new terminal_state.
"""

from __future__ import annotations

import inspect
import shlex
from pathlib import Path
from xml.etree.ElementTree import Element, SubElement, tostring

from harbor.agents.terminus_2.terminus_2 import (
    Terminus2 as _T2,
)  # for _parse_skill_frontmatter
from harbor.agents.terminus_2.terminus_json_plain_parser import (
    TerminusJSONPlainParser,
)
from harbor.agents.terminus_2.terminus_xml_plain_parser import (
    TerminusXMLPlainParser,
)
from harbor.agents.terminus_2.tmux_session import TmuxSession
from harbor.agents.terminus_2_modular.protocols import (
    LLMResponseParseResult,
    ModuleCtx,
    ToolCall,
    ToolResult,
)
from harbor.models.trial.paths import EnvironmentPaths


# Templates live alongside the original terminus_2/templates/ — reuse them.
# CRITICAL: resolve via the installed `harbor.agents.terminus_2` package,
# NOT via `__file__`. When this module is loaded from a candidate gen at e.g.
# `self_evo_runs/gen_5/modules/tools/baseline.py`, `__file__` points there
# and `__file__.parent.parent.parent.parent` resolves to `self_evo_runs/`,
# making the template path `self_evo_runs/terminus_2/templates/` which doesn't
# exist. Going through the harbor.agents.terminus_2 package keeps the
# resolution stable across gen swaps.
def _resolve_templates_dir() -> Path:
    import importlib

    pkg = importlib.import_module("harbor.agents.terminus_2")
    return Path(pkg.__file__).parent / "templates"


_TEMPLATES_DIR = _resolve_templates_dir()


def _read_template(parser_name: str) -> str:
    fname = (
        "terminus-json-plain.txt" if parser_name == "json" else "terminus-xml-plain.txt"
    )
    return (_TEMPLATES_DIR / fname).read_text()


# ---------------------------------------------------------------------------
# Solver helper tools — the EDITOR's extension point for giving the agent a
# NEW tool without touching the grammar template or the parser.
#
# A helper lives in `modules/tool_helper/<name>.py` — its OWN top-level module
# type, not a subdirectory of this one — and looks like any other module file:
#   NAME : str        — the command the agent types (first token)
#   USAGE: str        — the one line shown to the agent in its prompt
#   DESCRIPTION, NICHE = {...}
#   async def run(args: list[str], ctx) -> str   — does the work, returns text
#   def register(library)  →  library.register(type_="tool_helper", ...)
#
# At task time `execute()` intercepts a command whose first token is a helper
# NAME, runs the helper, and returns its text as the tool output — which the
# agent_loop feeds straight back to the agent (it already surfaces
# ToolResult.output). No parser/template/agent_loop change is needed. A helper
# typically reads the sandbox via `ctx.state.env.exec(...)`.
#
# Helpers used to live in a sibling `helpers/` dir, auto-loaded by scanning it.
# That made them invisible to the library — discovery is a SHALLOW glob over
# `modules/<type>/*.py`, so a subdirectory is never seen — and therefore invisible
# to the archive, the niche grid, the lineage DAG and cross-epoch confirm: a
# helper could only ever accumulate, never be benched, superseded or rolled back.
# They are now a first-class archived type: the Composer picks the SUBSET for a
# task, the Kernel instantiates them from the library and injects them here.
#
# Robust by design: an empty set is normal (gen_0 ships none) and a helper that
# raises is reported as a failed ToolResult — a bad tool must never break the
# solver.
# ---------------------------------------------------------------------------


class BaselineTools:
    """Real tmux + real parser + real templates. Default tools module for
    terminus-2-modular's gen_0."""

    NAME = "baseline"
    NICHE = {"transport": "tmux", "encoding": "both"}
    DESCRIPTION = (
        "Faithful tmux + parser + template stack from terminus-2. Owns a "
        "TmuxSession across the task. Supports XML or JSON response grammar."
    )
    PARAMS_SCHEMA = {
        "parser_name": "str: 'xml' or 'json' (default 'json')",
        "tmux_pane_width": "int (default 160)",
        "tmux_pane_height": "int (default 40)",
        "session_name": "str (default 'terminus-2-modular')",
        "helper_tools": "list[SolverHelper]: injected by the Kernel (see below)",
    }

    def __init__(
        self,
        parser_name: str = "json",
        tmux_pane_width: int = 160,
        tmux_pane_height: int = 40,
        session_name: str = "terminus-2-modular",
        helper_tools: list | None = None,
    ):
        if parser_name not in ("xml", "json"):
            raise ValueError(
                f"parser_name must be 'xml' or 'json'; got {parser_name!r}"
            )
        self._parser_name = parser_name
        self._pane_width = tmux_pane_width
        self._pane_height = tmux_pane_height
        self._session_name = session_name

        if parser_name == "json":
            self._parser = TerminusJSONPlainParser()
        else:
            self._parser = TerminusXMLPlainParser()

        self._template = _read_template(parser_name)

        # Editor-authored helper tools, CHOSEN per task by the Composer and
        # instantiated from the library by the Kernel. Empty by default (gen_0
        # ships none) → behavior unchanged until evolution adds one.
        self._helpers = {
            h.name: h for h in (helper_tools or []) if getattr(h, "name", "")
        }

    # ---------- Helper tools (editor extension point) ----------

    def _helpers_prompt_section(self) -> str:
        """One block listing the available helper commands, appended to the
        agent's initial prompt. Empty string when there are no helpers."""
        if not self._helpers:
            return ""
        lines = [
            "\n\n## Helper commands available in this environment",
            "Run any of these as a STANDALONE command (its own command, no pipes "
            "or redirection). Each returns its result directly to you:",
        ]
        for h in self._helpers.values():
            lines.append(f"- {h.usage}")
        return "\n".join(lines)

    async def _maybe_run_helper(
        self, call: ToolCall, ctx: ModuleCtx
    ) -> ToolResult | None:
        """If the command is a single registered helper invocation, run it and
        return its output (fed straight back to the agent). Otherwise return
        None so execute() falls through to the normal tmux send_keys path."""
        if not self._helpers:
            return None
        ks = (call.keystrokes or "").strip()
        if not ks or any(c in ks for c in "|&;><`$\n"):
            # Anything with shell metacharacters is a real shell command — let
            # tmux handle it (helper output can't be piped anyway).
            return None
        try:
            tokens = shlex.split(ks)
        except Exception:
            return None
        if not tokens or tokens[0] not in self._helpers:
            return None
        helper = self._helpers[tokens[0]]
        try:
            res = helper.run(tokens[1:], ctx)
            if inspect.isawaitable(res):  # accept both `async def` and `def` run
                res = await res
            out = "" if res is None else str(res)
        except Exception as exc:
            ctx.services.failures.raise_tag(
                "solver_helper_failed", "tools.baseline", f"{tokens[0]}: {exc}"
            )
            return ToolResult(
                success=False, output="", error=f"[helper {tokens[0]}] {exc}"
            )
        ctx.services.logger.debug("tools.baseline: ran helper %s", tokens[0])
        return ToolResult(success=True, output=out, error="")

    # ---------- Lifecycle ----------

    async def setup(self, ctx: ModuleCtx) -> None:
        """Create and start the tmux session. Stash into ctx.shared."""
        state = ctx.state
        env = state.env

        if state.record_terminal_session:
            local_recording_path = env.trial_paths.agent_dir / "recording.cast"
            remote_recording_path = EnvironmentPaths.agent_dir / "recording.cast"
        else:
            local_recording_path = None
            remote_recording_path = None

        session = TmuxSession(
            session_name=self._session_name,
            environment=env,
            logging_path=EnvironmentPaths.agent_dir / "terminus_2_modular.pane",
            local_asciinema_recording_path=local_recording_path,
            remote_asciinema_recording_path=remote_recording_path,
            pane_width=self._pane_width,
            pane_height=self._pane_height,
            extra_env=state.extra_env,
            user=env.default_user,
        )
        await session.start()
        ctx.shared.tmux_session = session
        ctx.services.logger.debug("tmux_full: session started (%s)", self._session_name)

    async def teardown(self, ctx: ModuleCtx) -> None:
        session = ctx.shared.tmux_session
        if session is None:
            return
        try:
            await session.stop()
            ctx.services.logger.debug("tmux_full: session stopped")
        except Exception as exc:
            ctx.services.logger.warning("tmux_full: session.stop failed: %s", exc)
        finally:
            ctx.shared.tmux_session = None

    # ---------- Prompt formatting ----------

    def format_initial_prompt(
        self,
        instruction: str,
        terminal_state: str,
        ctx: ModuleCtx,
    ) -> str:
        """Build the first user prompt: instruction + MCP info + skills XML
        + template-formatted terminal_state."""
        augmented = instruction
        # MCP info (terminus_2.py:1638-1647)
        mcp_servers = ctx.state.mcp_servers or []
        if mcp_servers:
            mcp_info = (
                "\n\nMCP Servers:\n"
                "The following MCP servers are available for this task.\n"
            )
            for s in mcp_servers:
                if s.transport == "stdio":
                    args_str = " ".join(s.args)
                    mcp_info += f"- {s.name}: stdio transport, command: {s.command} {args_str}\n"
                else:
                    mcp_info += f"- {s.name}: {s.transport} transport, url: {s.url}\n"
            augmented = augmented + mcp_info

        # Tell the agent about any editor-authored helper commands.
        augmented = augmented + self._helpers_prompt_section()

        # Note: skills section requires async env.exec — done in async path
        # below by tools_format_initial_prompt_with_skills(). This sync method
        # is called from agent_loop with already-augmented instruction.
        return self._template.format(
            instruction=augmented,
            terminal_state=terminal_state,
        )

    async def build_skills_section(self, ctx: ModuleCtx) -> str | None:
        """Discover Agent Skills in skills_dir and return an <available_skills>
        XML block. Mirrors Terminus2._build_skills_section (418-466).
        """
        skills_dir = ctx.state.skills_dir
        if not skills_dir:
            return None
        env = ctx.state.env

        if not await env.is_dir(skills_dir):
            ctx.services.logger.debug(
                "tmux_full: skills_dir %s does not exist; skipping", skills_dir
            )
            return None

        result = await env.exec(
            f"find {shlex.quote(skills_dir)} -mindepth 2 -maxdepth 2 -name SKILL.md",
            timeout_sec=10,
        )
        if result.return_code != 0:
            return None
        skill_md_paths = (result.stdout or "").strip().splitlines()
        if not skill_md_paths:
            return None

        entries: list[tuple[str, str, str]] = []
        for skill_md_path in skill_md_paths:
            cat_result = await env.exec(
                f"cat {shlex.quote(skill_md_path)}", timeout_sec=10
            )
            if cat_result.return_code != 0:
                continue
            fm = _T2._parse_skill_frontmatter(cat_result.stdout or "")
            if fm is None:
                continue
            entries.append((fm["name"], fm["description"], skill_md_path))

        if not entries:
            return None

        root = Element("available_skills")
        for name, description, location in entries:
            skill = SubElement(root, "skill")
            SubElement(skill, "name").text = name
            SubElement(skill, "description").text = description
            SubElement(skill, "location").text = location
        return "\n\n" + tostring(root, encoding="unicode")

    def format_continuation_prompt(
        self,
        terminal_state: str,
        ctx: ModuleCtx,
    ) -> str:
        """Per-step continuation prompt = just the new terminal state.

        terminus-2's main loop uses bare terminal_state as the next prompt
        (the Chat object keeps the history; we don't re-send the template).
        """
        return terminal_state

    # ---------- LLM response parsing ----------

    def parse_llm_response(self, response: str) -> LLMResponseParseResult:
        result = self._parser.parse_response(response)
        # Cap duration at 60s to match Terminus2._handle_llm_interaction
        # (terminus_2.py:1207): some LLMs occasionally request 120s+ which
        # ties up the tmux loop.
        commands = [
            ToolCall(keystrokes=c.keystrokes, duration_sec=min(c.duration, 60))
            for c in result.commands
        ]
        return LLMResponseParseResult(
            commands=commands,
            is_task_complete=result.is_task_complete,
            analysis=result.analysis,
            plan=result.plan,
            error=result.error,
            warning=result.warning,
        )

    # ---------- Execution ----------

    async def execute(self, call: ToolCall, ctx: ModuleCtx) -> ToolResult:
        # In-band helper dispatch: a single registered helper invocation runs
        # the helper and feeds its output straight back to the agent. Anything
        # else falls through to the normal tmux send_keys path below.
        helper_result = await self._maybe_run_helper(call, ctx)
        if helper_result is not None:
            return helper_result

        session = ctx.shared.tmux_session
        if session is None:
            ctx.services.failures.raise_tag(
                "no_session", "tools.baseline", "session not initialized"
            )
            return ToolResult(
                success=False, output="", error="tmux session not initialized"
            )
        try:
            # Match terminus_2._execute_commands: non-blocking send, sleep
            # `duration_sec` after to give the command time to start producing
            # output. NEVER use positional 2nd arg — that's `block: bool`.
            await session.send_keys(
                call.keystrokes,
                block=False,
                min_timeout_sec=call.duration_sec,
            )
        except TimeoutError:
            # Original terminus_2 surfaces this back to the LLM via a
            # timeout template; for now just tag + return so the loop continues.
            ctx.services.failures.raise_tag(
                "tmux_command_timeout",
                "tools.baseline",
                f"keystrokes={call.keystrokes!r} duration={call.duration_sec}",
            )
            return ToolResult(
                success=False,
                output="",
                error=f"timeout after {call.duration_sec}s",
            )
        except Exception as exc:
            ctx.services.failures.raise_tag(
                "tmux_send_failed", "tools.baseline", str(exc)
            )
            return ToolResult(success=False, output="", error=str(exc))

        # We don't capture output here — observation module owns capture.
        return ToolResult(success=True, output="", error="")


def register(library):
    library.register(
        type_="tools",
        name=BaselineTools.NAME,
        factory=lambda params: BaselineTools(
            parser_name=str(params.get("parser_name", "json")),
            tmux_pane_width=int(params.get("tmux_pane_width", 160)),
            tmux_pane_height=int(params.get("tmux_pane_height", 40)),
            session_name=str(params.get("session_name", "terminus-2-modular")),
            helper_tools=params.get("helper_tools") or [],
        ),
        description=BaselineTools.DESCRIPTION,
        params_schema=BaselineTools.PARAMS_SCHEMA,
        niche=BaselineTools.NICHE,
    )
