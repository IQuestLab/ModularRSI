"""Baseline tools + bypass for edit_block / write_file metacharacter check.

What this closes:
- The baseline `_maybe_run_helper` refuses commands containing shell metacharacters
  (| & ; > < ` $ \\n). For content-bearing helpers (write_file, edit_block) the
  content legitimately contains newlines and special characters. This variant
  routes write_file and edit_block by first token, bypassing the metacharacter
  check — exactly as always_helpers does — but does NOT auto-inject helpers.
  The Composer still controls which helpers are available; this variant just
  ensures that when edit_block or write_file is selected, it actually works.

Unlike always_helpers (which always injects six helpers) and full_helpers
(which also adds fallback parsing), this variant is a minimal surgical change:
only the bypass logic, no forced injection, no fallback parser.
"""

from __future__ import annotations

import inspect

from harbor.agents.terminus_2_modular.modules.tools.baseline import BaselineTools
from harbor.agents.terminus_2_modular.protocols import (
    ModuleCtx,
    ToolCall,
    ToolResult,
)


class BypassHelpersTools(BaselineTools):
    """BaselineTools + metacharacter bypass for write_file and edit_block.

    These two helpers carry arbitrary content (source code with ;, $, <, >, etc.)
    which the baseline dispatch refuses. We route by first token: if the command
    is write_file or edit_block, bypass the metacharacter check and split the
    raw remainder into path (first token) and content (everything after).
    For all other helpers, delegate to the baseline's dispatch.
    """

    NAME = "bypass_helpers"
    NICHE = {"transport": "tmux", "encoding": "both", "helpers": "bypass-only"}
    DESCRIPTION = (
        "BaselineTools + metacharacter bypass for write_file and edit_block. "
        "Does not inject helpers; the Composer controls which helpers are "
        "available. For write_file/edit_block, the content argument may contain "
        "any characters including newlines, $, ;, <, >, |, etc. All other "
        "helpers use the standard baseline dispatch."
    )

    async def _maybe_run_helper(
        self, call: ToolCall, ctx: ModuleCtx
    ) -> ToolResult | None:
        """Override baseline's dispatch to route write_file and edit_block through
        despite metacharacters in the content argument.

        write_file and edit_block commands carry arbitrary content (source code
        with ;, $, <, >, etc.) which the baseline dispatcher refuses. We route
        by first token: if the command is write_file or edit_block, bypass the
        metacharacter check and split the raw remainder into path (first token)
        and content (everything after). For all other helpers, delegate to the
        baseline's dispatch.
        """
        if not self._helpers:
            return None
        ks = (call.keystrokes or "").strip()
        if not ks:
            return None

        # Extract the first token (command name) before any space
        parts = ks.split(maxsplit=1)
        cmd_name = parts[0] if parts else ""
        if cmd_name not in self._helpers:
            return None

        # write_file and edit_block bypass the metacharacter check
        if cmd_name in ("write_file", "edit_block"):
            if len(parts) < 2:
                return ToolResult(
                    success=True,
                    output=self._helpers[cmd_name].usage,
                    error="",
                )
            remainder = parts[1]
            # Split remainder: first token is path, rest is content
            path_parts = remainder.split(maxsplit=1)
            if len(path_parts) < 2:
                return ToolResult(
                    success=True,
                    output=self._helpers[cmd_name].usage,
                    error="",
                )
            path = path_parts[0]
            content = path_parts[1]
            helper = self._helpers[cmd_name]
            try:
                res = helper.run([path, content], ctx)
                if inspect.isawaitable(res):
                    res = await res
                out = "" if res is None else str(res)
            except Exception as exc:
                ctx.services.failures.raise_tag(
                    "solver_helper_failed",
                    "tools.bypass_helpers",
                    f"{cmd_name}: {exc}",
                )
                return ToolResult(
                    success=False, output="", error=f"[helper {cmd_name}] {exc}"
                )
            ctx.services.logger.debug(
                "tools.bypass_helpers: ran helper %s %s", cmd_name, path
            )
            return ToolResult(success=True, output=out, error="")

        # For all other helpers, use the baseline metacharacter + shlex dispatch
        return await super()._maybe_run_helper(call, ctx)


def register(library):
    library.register(
        type_="tools",
        name=BypassHelpersTools.NAME,
        factory=lambda params: BypassHelpersTools(
            parser_name=str(params.get("parser_name", "json")),
            tmux_pane_width=int(params.get("tmux_pane_width", 160)),
            tmux_pane_height=int(params.get("tmux_pane_height", 40)),
            session_name=str(params.get("session_name", "terminus-2-modular")),
            helper_tools=params.get("helper_tools") or [],
        ),
        description=BypassHelpersTools.DESCRIPTION,
        params_schema=BypassHelpersTools.PARAMS_SCHEMA,
        niche=BypassHelpersTools.NICHE,
    )
