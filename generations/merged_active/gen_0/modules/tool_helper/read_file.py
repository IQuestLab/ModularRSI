"""Structured file reading for the solver.

The agent's only built-in action is raw keystrokes into tmux, where `cat` output
is clipped by the pane capture truncation. This helper returns its output
directly through ToolResult.output (bypassing the pane), so the agent can
inspect a file in controlled, line-numbered slices without the truncation cap.
"""

import shlex

NAME = "read_file"
USAGE = (
    "read_file <path> [start [end]] — print a line-numbered slice of a file "
    "(default: lines 1-200, max 400 lines per call)"
)
DESCRIPTION = (
    "Untruncated, line-numbered file reading for draft/edit tasks; replaces "
    "cat+wc when long output is clipped by the pane."
)
NICHE = {"op": "read", "target": "file"}


async def run(args, ctx):
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


def register(library):
    from harbor.agents.terminus_2_modular.protocols import SolverHelper

    library.register(
        type_="tool_helper",
        name=NAME,
        factory=lambda params: SolverHelper(name=NAME, usage=USAGE, run=run),
        description=DESCRIPTION,
        niche=NICHE,
    )
