"""Combined robust tools: resilient parser fallback + always-on helpers w/ multi-line support.

MERGE of resilient_parser and always_helpers into a single variant.

What this closes:
1. Parse-error waste: When the LLM outputs malformed responses (markdown fences,
   unclosed quotes, free-form preamble text before JSON), the baseline parser
   returns zero commands and an error. The agent_loop retries, often producing
   similarly malformed output, wasting episodes. This variant first tries the
   baseline parser; if it fails with zero commands, falls back through four regex
   strategies (strip fences, JSON block, XML tags, raw command heuristic) before
   reporting failure.

2. Multi-line helper blocking: The baseline's _maybe_run_helper rejects commands
   containing newlines and other shell metacharacters. For content-bearing helpers
   (write_file, edit_block) the content legitimately contains newlines (multi-line
   source code). This variant routes write_file and edit_block by first token,
   bypassing the metacharacter check — as always_helpers already does.

3. Always-on file helpers: The six file helpers (read_file, edit_file, edit_block,
   glob_files, grep_search, write_file) are always injected, preventing the agent
   from resorting to fragile raw shell commands (cat, heredocs, python3 -c) for
   file operations.

4. Raw `cat <path>` interception: agents habitually type `cat` even when a
   read_file helper is available, and tmux pane truncation can then swallow the
   output entirely. A single-file `cat <path>` is routed through read_file's
   line-numbered, untruncated output instead of the shell, so the content always
   comes back to the agent.

  SUPERSEDES: resilient_parser, always_helpers, full_helpers

  5. Heredoc file-write blocking: When the agent types `cat > <file> << 'EOF'...EOF`
     (or similar heredoc patterns) to write a file, the command is blocked with a
     message directing the agent to use `write_file` instead, avoiding shell
     escaping issues and ensuring reliable file writes.
"""

from __future__ import annotations

import ast
import base64
import inspect
import json
import re
import shlex

from harbor.agents.terminus_2_modular.modules.tools.baseline import BaselineTools
from harbor.agents.terminus_2_modular.protocols import (
    LLMResponseParseResult,
    ModuleCtx,
    SolverHelper,
    ToolCall,
    ToolResult,
)

# ---------------------------------------------------------------------------
# Heredoc file-write blocker pattern
# ---------------------------------------------------------------------------

# Heredoc file-write pattern: "cat > <file> << ['"]?<delim>?['"]?"
# We match any command that starts with "cat" and contains both ">" and "<<"
# (heredoc marker). If found, we reject and redirect to write_file.
_HEREDOC_CAT_RE = re.compile(
    r"^\s*cat\s+.*>.*<<",
    re.DOTALL | re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Regex patterns for fallback parsing (same as resilient_parser.py)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Regex patterns for fallback parsing (same as resilient_parser.py)
# ---------------------------------------------------------------------------

_JSON_COMMANDS_RE = re.compile(
    r"\{[^{}]*\"commands\"[^{}]*\}",
    re.DOTALL,
)
_JSON_BLOCK_RE = re.compile(r"(\{.*?\})", re.DOTALL)
_KEYS_RE = re.compile(
    "\x3ckeystrokes\x3e(.*?)\x3c/keystrokes\x3e", re.DOTALL | re.IGNORECASE
)
# Tolerant variant: allows attributes on the keystrokes tag, extra whitespace,
# and UNCLOSED keystrokes blocks (LLM output truncated before the closing tag).
# A block ends at the next tag, at a closing commands/response tag, or at the
# end of the response text.
_TOLERANT_KEYS_RE = re.compile(
    "\x3ckeystrokes[^\x3e]*\x3e\\s*(.*?)(?=\x3c/keystrokes\x3e|\x3c/commands\x3e|\x3c/response\x3e|\x3ckeystrokes|$)",
    re.DOTALL | re.IGNORECASE,
)
_DONE_RE = re.compile(
    "\x3ctask_complete\x3e\\s*(true|false)\\s*\x3c/task_complete\x3e",
    re.IGNORECASE,
)
_ANALYSIS_RE = re.compile(r"<analysis>(.*?)</analysis>", re.DOTALL | re.IGNORECASE)
_PLAN_RE = re.compile(r"<plan>(.*?)</plan>", re.DOTALL | re.IGNORECASE)

# ---------------------------------------------------------------------------
# Inline helper implementations — same as always_helpers.py
# ---------------------------------------------------------------------------


async def _read_file(args, ctx):
    if not args:
        return "usage: read_file <path> [start [end]]"
    path = args[0]
    try:
        start = int(args[1]) if len(args) > 1 else 1
    except ValueError:
        return f"error: start must be an integer, got {args[1]!r}"
    try:
        end = int(args[2]) if len(args) > 2 else start + 199
    except ValueError:
        return f"error: end must be an integer, got {args[2]!r}"
    if start < 1:
        return "error: start must be >= 1"
    if end < start:
        return "error: end must be >= start"
    if end - start + 1 > 400:
        end = start + 399

    quoted = shlex.quote(path)
    probe = await ctx.state.env.exec(
        f"test -f {quoted} && wc -l < {quoted}", timeout_sec=10
    )
    if probe.return_code != 0:
        return f"error: not a readable file: {path}"
    try:
        total = int((probe.stdout or "0").strip() or "0")
    except ValueError:
        total = 0
    if total == 0:
        return "(file is empty)"
    if start > total:
        return f"error: start {start} beyond file length {total}"

    end = min(end, total)
    awk_prog = (
        "awk -v s=%d -v e=%d 'NR>=s && NR<=e {printf \"%%6d\\t%%s\\n\", NR, $0}' %s"
        % (start, end, quoted)
    )
    res = await ctx.state.env.exec(awk_prog, timeout_sec=15)
    if res.return_code != 0:
        return f"error reading {path}: {res.stderr or res.return_code}"
    out = (res.stdout or "").rstrip("\n")
    if end < total:
        out += (
            f"\n... ({total - end} more lines; "
            f"read_file {shlex.quote(path)} {end + 1} {total})"
        )
    return out


async def _edit_file(args, ctx):
    if len(args) != 3:
        return "usage: edit_file <path> <old> <new>"
    path, old, new = args[0], args[1], args[2]
    script = (
        "import sys\n"
        "path, old, new = sys.argv[1], sys.argv[2], sys.argv[3]\n"
        "try:\n"
        "    s = open(path, encoding='utf-8', errors='replace').read()\n"
        "except Exception as e:\n"
        "    print(f'error reading {path}: {e}')\n"
        "    sys.exit(1)\n"
        "if old not in s:\n"
        "    print(f'error: old string not found in {path}')\n"
        "    sys.exit(1)\n"
        "s2 = s.replace(old, new, 1)\n"
        "try:\n"
        "    open(path, 'w', encoding='utf-8').write(s2)\n"
        "except Exception as e:\n"
        "    print(f'error writing {path}: {e}')\n"
        "    sys.exit(1)\n"
        "lines = s2.split('\\n')\n"
        "hits = [str(i + 1) for i, ln in enumerate(lines) if new in ln]\n"
        "suffix = (' on line(s) ' + ','.join(hits[:8])) if hits else ''\n"
        "print(f'edited {path}: replaced 1 occurrence{suffix}')\n"
    )
    cmd = (
        "python3 -c "
        + shlex.quote(script)
        + " "
        + shlex.quote(path)
        + " "
        + shlex.quote(old)
        + " "
        + shlex.quote(new)
    )
    res = await ctx.state.env.exec(cmd, timeout_sec=15)
    if res.return_code != 0:
        return f"error: {(res.stderr or res.stdout or str(res.return_code)).strip()}"
    return (res.stdout or "").strip() or "done"


async def _glob_files(args, ctx):
    if not args:
        return "usage: glob_files <pattern>"
    pattern = args[0]
    script = (
        "import glob, os, sys\n"
        "rows = []\n"
        "seen = set()\n"
        "for p in sys.argv[1:]:\n"
        "    for f in sorted(glob.glob(p, recursive=True)):\n"
        "        if os.path.isfile(f) and f not in seen:\n"
        "            seen.add(f)\n"
        "            rows.append(f)\n"
        "            if len(rows) >= 100:\n"
        "                break\n"
        "print('\\n'.join(rows[:100]))\n"
        "if len(rows) == 100:\n"
        "    print('... (first 100 shown)')\n"
    )
    cmd = f"python3 -c {shlex.quote(script)} {shlex.quote(pattern)}"
    res = await ctx.state.env.exec(cmd, timeout_sec=15)
    if res.return_code != 0:
        return f"error: {(res.stderr or '').strip() or res.return_code}"
    out = (res.stdout or "").strip()
    return out or f"no files match {pattern}"


async def _grep_search(args, ctx):
    if not args:
        return "usage: grep_search <pattern> [path]"
    pattern = args[0]
    path = args[1] if len(args) > 1 else "."
    cmd = f"grep -rn -I -m 50 -E {shlex.quote(pattern)} {shlex.quote(path)} 2>/dev/null"
    res = await ctx.state.env.exec(cmd, timeout_sec=15)
    if res.return_code not in (0, 1):
        return (
            f"error searching {path}: {(res.stderr or '').strip() or res.return_code}"
        )
    lines = [ln for ln in (res.stdout or "").splitlines() if ln]
    if not lines:
        return "no matches"
    out = "\n".join(lines[:50])
    if len(lines) > 50:
        out += f"\n... ({len(lines) - 50} more matches)"
    return out


async def _write_file(args, ctx):
    if len(args) < 2:
        return "usage: write_file <path> <content>"
    path = args[0]
    content = args[1]
    script = (
        "import sys\n"
        "path, content = sys.argv[1], sys.argv[2]\n"
        "with open(path, 'w', encoding='utf-8') as f:\n"
        "    f.write(content)\n"
        "print(f'wrote {len(content)} bytes to {path}')\n"
    )
    cmd = (
        "python3 -c "
        + shlex.quote(script)
        + " "
        + shlex.quote(path)
        + " "
        + shlex.quote(content)
    )
    res = await ctx.state.env.exec(cmd, timeout_sec=15)
    if res.return_code != 0:
        return f"error: {(res.stderr or res.stdout or str(res.return_code)).strip()}"
    return (res.stdout or "").strip() or "done"


async def _edit_block(args, ctx):
    if len(args) < 2:
        return (
            "usage: edit_block <path> <old> <<<<<<<< <new> — replace the FIRST "
            "exact multi-line occurrence of <old> with <new> in <path>. Put the "
            "marker `<<<<<<<<` on its own line between old and new."
        )
    path = args[0]
    content = args[1]
    marker_nl = "\n<<<<<<<<\n"
    marker_sp = " <<<<<<<< "
    if marker_nl in content:
        old, new = content.split(marker_nl, 1)
        old = old.strip("\n")
        new = new.strip("\n")
    elif marker_sp in content:
        old, new = content.split(marker_sp, 1)
        old = old.strip()
        new = new.strip()
    else:
        return (
            "error: edit_block needs the marker `<<<<<<<<` on its own line "
            "between <old> and <new>"
        )
    if not old:
        return "error: edit_block <old> must not be empty"
    old_b64 = base64.b64encode(old.encode("utf-8")).decode("ascii")
    new_b64 = base64.b64encode(new.encode("utf-8")).decode("ascii")
    script = (
        "import base64, sys\n"
        "path, old_b, new_b = sys.argv[1], sys.argv[2], sys.argv[3]\n"
        "old = base64.b64decode(old_b).decode('utf-8')\n"
        "new = base64.b64decode(new_b).decode('utf-8')\n"
        "try:\n"
        "    s = open(path, encoding='utf-8', errors='replace').read()\n"
        "except Exception as e:\n"
        "    print(f'error reading {path}: {e}')\n"
        "    sys.exit(1)\n"
        "if old not in s:\n"
        "    print(f'error: old string not found in {path}')\n"
        "    sys.exit(1)\n"
        "s2 = s.replace(old, new, 1)\n"
        "try:\n"
        "    open(path, 'w', encoding='utf-8').write(s2)\n"
        "except Exception as e:\n"
        "    print(f'error writing {path}: {e}')\n"
        "    sys.exit(1)\n"
        "print(f'edited {path}: replaced 1 multi-line occurrence')\n"
    )
    cmd = (
        "python3 -c "
        + shlex.quote(script)
        + " "
        + shlex.quote(path)
        + " "
        + old_b64
        + " "
        + new_b64
    )
    res = await ctx.state.env.exec(cmd, timeout_sec=15)
    if res.return_code != 0:
        return f"error: {(res.stderr or res.stdout or str(res.return_code)).strip()}"
    return (res.stdout or "").strip() or "done"


_ALWAYS_HELPERS: dict[str, SolverHelper] = {
    "read_file": SolverHelper(
        name="read_file",
        usage=(
            "read_file <path> [start [end]] — print a line-numbered slice of a file "
            "(default: lines 1-200, max 400 lines per call)"
        ),
        run=_read_file,
    ),
    "edit_file": SolverHelper(
        name="edit_file",
        usage=(
            "edit_file <path> <old> <new> — replace the FIRST exact occurrence of "
            "<old> with <new> in a file and report the edited line(s). Avoid "
            "| & ; > < ` $ and newlines in <old>/<new>; quote multi-word arguments."
        ),
        run=_edit_file,
    ),
    "glob_files": SolverHelper(
        name="glob_files",
        usage=(
            "glob_files <pattern> — list files matching a glob, e.g. "
            "glob_files '**/*.py' or glob_files '*.json' (up to 100 entries)"
        ),
        run=_glob_files,
    ),
    "grep_search": SolverHelper(
        name="grep_search",
        usage=(
            "grep_search <pattern> [path] — regex-search files under path "
            "(default .), return up to 50 matches as file:line: text. Pattern "
            "must avoid | & ; > < ` $ and newlines (helper dispatch restriction)."
        ),
        run=_grep_search,
    ),
    "write_file": SolverHelper(
        name="write_file",
        usage=(
            "write_file <path> <content> — write <content> to <path>, "
            "overwriting if it exists. Content is the raw remainder after the "
            "path, so it CAN contain any characters (;, $, <, >, |, etc.). "
            "Returns 'wrote N bytes to <path>' on success."
        ),
        run=_write_file,
    ),
    "edit_block": SolverHelper(
        name="edit_block",
        usage=(
            "edit_block <path> <old> <<<<<<<< <new> — replace the FIRST exact "
            "multi-line occurrence of <old> with <new> in <path>. Put the marker "
            "`<<<<<<<<` on its own line between old and new; both are passed "
            "VERBATIM (do not quote them) and may span lines and contain any "
            "characters. Returns 'edited <path>: replaced 1 multi-line occurrence' "
            "on success."
        ),
        run=_edit_block,
    ),
}


class CombinedRobustTools(BaselineTools):
    """BaselineTools + resilient parser fallback + always-on helpers w/ multi-line support.

    MERGE of resilient_parser and always_helpers into a single variant.
    SUPERSEDES: resilient_parser, always_helpers, full_helpers
    """

    NAME = "combined_robust"
    NICHE = {"transport": "tmux", "encoding": "both", "robustness": "combined"}
    DESCRIPTION = (
        "BaselineTools + resilient parser fallback + always-on file helpers. "
        "When the standard JSON/XML parser returns an error+zero-commands, "
        "attempts four regex strategies (strip fences, JSON block, XML tags, "
        "raw command heuristic) before reporting failure. The XML fallback "
        "tolerates tag attributes, extra whitespace, and unclosed keystrokes "
        "tags (truncated responses); the JSON fallback tolerates single-quoted "
        "keys and trailing commas. The six file helpers "
        "(read_file, edit_file, edit_block, glob_files, grep_search, write_file) "
        "are always available. write_file and edit_block bypass the metacharacter "
        "check to support multi-line content. A bare `cat <path>` is also "
        "intercepted and served through read_file's full untruncated output, so "
        "habitual shell file reads never get swallowed by the tmux pane. "
        "Heredoc-style file writes (cat > file << ...) are blocked and redirected "
        "to the write_file helper. "
        "Prevents both PARSE-ERROR stalls and fragile raw-shell file operations."
    )

    def __init__(self, *args, **kwargs):
        # Let the baseline handle all normal parameters, including helper_tools
        # from the Composer. This populates self._helpers.
        super().__init__(*args, **kwargs)

        # Always ensure our six helpers are present — they supplement any
        # Composer-selected helpers. If a helper with the same name was already
        # injected by the Composer, we skip it so the Composer's version wins.
        for name, helper in _ALWAYS_HELPERS.items():
            if name not in self._helpers:
                self._helpers[name] = helper

    # ---------- Parse resilience (from resilient_parser.py) ----------

    def parse_llm_response(self, response: str) -> LLMResponseParseResult:
        # 1. Try the baseline parser first
        result = super().parse_llm_response(response)

        # If the baseline succeeded (commands > 0) or had no error, return as-is
        has_error = bool(result.error)
        has_no_commands = len(result.commands) == 0

        if not has_error or not has_no_commands:
            return result

        # 2. Baseline failed — try fallback strategies
        fallback_result = self._fallback_parse(response)
        if fallback_result is not None:
            # Guard: never return is_task_complete=True with zero commands.
            # This prevents the fallback from prematurely terminating the episode
            # when it matches "task_complete" in free-form analysis text but
            # produces no actual commands.
            if fallback_result.is_task_complete and len(fallback_result.commands) == 0:
                return LLMResponseParseResult(
                    commands=fallback_result.commands,
                    is_task_complete=False,
                    analysis=fallback_result.analysis,
                    plan=fallback_result.plan,
                    error=fallback_result.error,
                    warning=fallback_result.warning
                    + " (overrode false task_complete — zero commands)",
                )
            return fallback_result

        # 3. Fallback also failed — return the original error
        return result

    def _fallback_parse(self, response: str) -> LLMResponseParseResult | None:
        """Attempt four fallback strategies in order. Return the first that
        produces at least one command, or None if all fail."""
        # Strategy A: Strip markdown code fences and re-try baseline parser.
        fallback = self._try_strip_fences(response)
        if fallback is not None and fallback.commands:
            return fallback

        # Strategy B: JSON block with "commands"
        fallback = self._try_json_commands(response)
        if fallback is not None and fallback.commands:
            return fallback

        # Strategy C: XML-style tags
        fallback = self._try_xml_tags(response)
        if fallback is not None and fallback.commands:
            return fallback

        # Strategy D: Raw lines that look like commands (heuristic)
        fallback = self._try_raw_commands(response)
        if fallback is not None and fallback.commands:
            return fallback

        return None

    def _try_strip_fences(self, response: str) -> LLMResponseParseResult | None:
        """Detect markdown code fences (```xml, ```json, or plain ```) and
        extract the inner content, then re-try the baseline parser on that
        content."""
        # Also match a fence with no closing ``` (truncated response): the
        # fenced content then runs to the end of the response text.
        m = re.search(
            r"```(?:\w*)[ \t]*\r?\n(.+?)(?:\r?\n```|\Z)",
            response,
            re.DOTALL,
        )
        if not m:
            return None
        inner = m.group(1).strip()
        if not inner:
            return None
        result = super().parse_llm_response(inner)
        if len(result.commands) > 0 or not result.error:
            if len(result.commands) > 0:
                return LLMResponseParseResult(
                    commands=result.commands,
                    is_task_complete=result.is_task_complete,
                    analysis=result.analysis,
                    plan=result.plan,
                    error="",
                    warning="recovered via code fence stripping fallback",
                )
        return None

    @staticmethod
    def _json_loads_tolerant(text: str):
        """Try progressively more tolerant JSON parsers:
        1. strict json.loads
        2. json.loads after stripping trailing commas
        3. ast.literal_eval (single-quoted keys/values, Python True/False/None)
        Returns the parsed dict, or None if all attempts fail.
        """
        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                return obj
        except ValueError:
            pass
        try:
            obj = json.loads(re.sub(r",\s*([}\]])", r"\1", text))
            if isinstance(obj, dict):
                return obj
        except ValueError:
            pass
        try:
            obj = ast.literal_eval(text)
            if isinstance(obj, dict):
                return obj
        except (ValueError, SyntaxError, TypeError):
            pass
        return None

    def _try_json_commands(self, response: str) -> LLMResponseParseResult | None:
        """Find the first JSON object containing a 'commands' key and parse it."""
        m = _JSON_COMMANDS_RE.search(response)
        if not m:
            stack = []
            start = None
            for i, ch in enumerate(response):
                if ch == "{":
                    if start is None:
                        start = i
                    stack.append(i)
                elif ch == "}":
                    if stack:
                        stack.pop()
                        if not stack and start is not None:
                            candidate = response[start : i + 1]
                            if ('"commands"' in candidate) or (
                                "'commands'" in candidate
                            ):
                                m = type("", (), {})()
                                m.group = lambda: candidate
                                break
                            start = None
        if not m:
            return None
        obj = self._json_loads_tolerant(m.group())
        if obj is None:
            return None
        commands_raw = obj.get("commands")
        if commands_raw is None:
            return None
        if isinstance(commands_raw, str):
            commands_list = [commands_raw]
        elif isinstance(commands_raw, list):
            commands_list = []
            for c in commands_raw:
                if isinstance(c, (str, int, float)):
                    commands_list.append(str(c))
                elif isinstance(c, dict):
                    # LLMs sometimes emit objects like
                    # {"keystrokes": "ls", "duration": 5}
                    for key in ("keystrokes", "command", "cmd", "shell"):
                        val = c.get(key)
                        if isinstance(val, (str, int, float)):
                            commands_list.append(str(val))
                            break
                # other shapes (nested lists, booleans) are skipped
            if not commands_list:
                return None
        else:
            return None
        commands = [
            ToolCall(keystrokes=c.strip()) for c in commands_list if c and c.strip()
        ]
        if not commands:
            return None

        is_complete = bool(obj.get("task_complete", False))
        analysis = obj.get("analysis", "")
        plan = obj.get("plan", "")
        return LLMResponseParseResult(
            commands=commands,
            is_task_complete=is_complete,
            analysis=str(analysis),
            plan=str(plan),
            error="",
            warning="recovered via JSON fallback parser",
        )

    def _try_xml_tags(self, response: str) -> LLMResponseParseResult | None:
        """Extract <keystrokes> blocks from the raw response (even outside a
        well-formed <response> wrapper)."""
        # Tolerant matcher: handles tag attributes, extra whitespace, and
        # unclosed trailing keystrokes blocks (truncated responses).
        keystrokes_matches = list(_TOLERANT_KEYS_RE.finditer(response))
        if not keystrokes_matches:
            return None
        commands = []
        for m in keystrokes_matches:
            ks = m.group(1).strip()
            if ks:
                commands.append(ToolCall(keystrokes=ks))
        if not commands:
            return None

        done_m = _DONE_RE.search(response)
        is_complete = done_m is not None and done_m.group(1).lower() == "true"
        a = _ANALYSIS_RE.search(response)
        p = _PLAN_RE.search(response)
        return LLMResponseParseResult(
            commands=commands,
            is_task_complete=is_complete,
            analysis=a.group(1).strip() if a else "",
            plan=p.group(1).strip() if p else "",
            error="",
            warning="recovered via XML tag fallback parser",
        )

    def _try_raw_commands(self, response: str) -> LLMResponseParseResult | None:
        """Heuristic: find lines that look like shell commands in the raw text
        and join them into a SINGLE multi-line command (preserving heredocs,
        multi-line scripts, etc.)."""
        lines = response.splitlines()
        candidate_commands = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("```"):
                continue
            first_word = stripped.split(maxsplit=1)[0] if " " in stripped else stripped
            shell_commands = {
                "cd",
                "ls",
                "cat",
                "echo",
                "python3",
                "python",
                "mkdir",
                "rm",
                "cp",
                "mv",
                "touch",
                "grep",
                "find",
                "sed",
                "awk",
                "chmod",
                "git",
                "make",
                "source",
                "export",
                ".",
                "./",
                "sh",
                "bash",
                "head",
                "tail",
                "wc",
                "sort",
                "uniq",
                "tee",
                "exec",
                "nohup",
                "ps",
                "kill",
                "pkill",
                "timeout",
                "which",
                "type",
                "test",
                "[",
                "read_file",
                "edit_file",
                "glob_files",
                "grep_search",
                "write_file",
                "edit_block",
            }
            if first_word in shell_commands or first_word.startswith("./"):
                candidate_commands.append(stripped)

        if not candidate_commands:
            return None

        # Join all candidate lines into a single multi-line command.
        # This preserves heredoc structure (cat > file <<'EOF' … EOF) and
        # multi-line scripts.  In bash, newline-separated commands execute
        # sequentially, so this is safe for independent commands too.
        joined = "\n".join(candidate_commands)
        commands = [ToolCall(keystrokes=joined)]

        # Also guard: never set is_complete=True when no commands were found,
        # even if the raw text contains "task_complete" in analysis/plan context.
        if re.search(
            r"<task_complete>\s*true\s*</task_complete>", response, re.IGNORECASE
        ):
            is_complete = True
        elif re.search(r"task_complete.*true", response, re.IGNORECASE):
            is_complete = True
        elif re.search(r'"task_complete"\s*:\s*true', response, re.IGNORECASE):
            is_complete = True
        else:
            is_complete = False

        return LLMResponseParseResult(
            commands=commands,
            is_task_complete=is_complete,
            analysis="",
            plan="",
            error="",
            warning="recovered via raw command line heuristic fallback parser (multi-line)",
        )

    # ---------- Helper dispatch (from always_helpers.py) ----------

    async def _maybe_run_helper(
        self, call: ToolCall, ctx: ModuleCtx
    ) -> ToolResult | None:
        """Override baseline's dispatch to route write_file and edit_block through
        despite metacharacters in the content argument.

        write_file and edit_block commands carry arbitrary content (source code
        with ;, $, <, >, etc.) which the baseline dispatcher refuses. We route
        by first token: if the command is write_file or edit_block, bypass the
        metacharacter check and split the raw remainder into path (first token)
        and content (everything after). A bare `cat <path>` is also routed
        through read_file so the full file content comes back untruncated. For
        all other helpers, delegate to the baseline's dispatch.
        """
        if not self._helpers:
            return None
        ks = (call.keystrokes or "").strip()
        if not ks:
            return None

        # Extract the first token (command name) before any space
        parts = ks.split(maxsplit=1)
        cmd_name = parts[0] if parts else ""

        # Intercept raw `cat <path>` (single file argument, no flags, pipes or
        # redirection). The agent often reaches for `cat` out of habit even
        # when read_file is available; through tmux the pane truncation can
        # swallow the output entirely, wasting steps on a working-but-invisible
        # command. Routing it through _read_file returns the full line-numbered
        # content via ToolResult.output — same result as read_file, so there is
        # no penalty for choosing `cat`.
        if cmd_name == "cat" and len(parts) == 2 and "cat" not in self._helpers:
            path_parts = parts[1].split(maxsplit=1)
            if (
                len(path_parts) == 1
                and path_parts[0]
                and not any(c in ks for c in "|&;><`$\n")
            ):
                try:
                    res = await _read_file(path_parts, ctx)
                    out = "" if res is None else str(res)
                except Exception as exc:
                    ctx.services.failures.raise_tag(
                        "solver_helper_failed",
                        "tools.combined_robust",
                        f"cat: {exc}",
                    )
                    return ToolResult(
                        success=False, output="", error=f"[helper cat] {exc}"
                    )
                return ToolResult(success=True, output=out, error="")

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
                        "tools.combined_robust",
                        f"{cmd_name}: {exc}",
                    )
                    return ToolResult(
                        success=False, output="", error=f"[helper {cmd_name}] {exc}"
                    )
                ctx.services.logger.debug(
                    "tools.combined_robust: ran helper %s %s", cmd_name, path
                )
                return ToolResult(success=True, output=out, error="")

            # For all other helpers, use the baseline metacharacter + shlex dispatch
            return await super()._maybe_run_helper(call, ctx)

    # ---------- Execute override: block heredoc file writes ----------

    async def execute(self, call: ToolCall, ctx: ModuleCtx) -> ToolResult:
        """Override baseline execute to block heredoc file-write patterns and
        redirect the agent to use write_file instead."""
        ks = (call.keystrokes or "").strip()
        if ks and _HEREDOC_CAT_RE.match(ks):
            # Extract a hint from the command for the error message
            hint = ks[:80].replace("\n", " ")
            return ToolResult(
                success=False,
                output="",
                error=(
                    f"Use write_file <path> <content> instead of cat heredoc. "
                    f"The write_file helper handles arbitrary content reliably. "
                    f"Blocked command: {hint}"
                ),
            )
        return await super().execute(call, ctx)


def register(library):
    library.register(
        type_="tools",
        name=CombinedRobustTools.NAME,
        factory=lambda params: CombinedRobustTools(
            parser_name=str(params.get("parser_name", "json")),
            tmux_pane_width=int(params.get("tmux_pane_width", 160)),
            tmux_pane_height=int(params.get("tmux_pane_height", 40)),
            session_name=str(params.get("session_name", "terminus-2-modular")),
            helper_tools=params.get("helper_tools") or [],
        ),
        description=CombinedRobustTools.DESCRIPTION,
        params_schema=CombinedRobustTools.PARAMS_SCHEMA,
        niche=CombinedRobustTools.NICHE,
    )
