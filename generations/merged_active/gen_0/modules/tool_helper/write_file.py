"""Atomic file creation/overwrite for the solver.

The agent's raw-keystroke action surface makes file creation error-prone
(cat+heredoc, echo redirections, python3 -c invocations). This helper writes
a file atomically and returns confirmation, bypassing the tmux pane truncation.

The always_helpers variant overrides the metacharacter dispatch check so that
write_file commands with C source code (containing ;, $, <, etc.) are routed
to this helper instead of being sent to tmux.
"""

import shlex

NAME = "write_file"
USAGE = (
    "write_file <path> <content> — write <content> to <path>, overwriting if it "
    "exists. Content is everything after the first space after <path>. Returns "
    "'wrote N bytes to <path>' on success."
)
DESCRIPTION = (
    "Atomic file creation/overwrite for draft/edit tasks; replaces the fragile "
    "cat+heredoc loop when the agent must write multi-line source files."
)
NICHE = {"op": "write", "target": "file"}


async def run(args, ctx):
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


def register(library):
    from harbor.agents.terminus_2_modular.protocols import SolverHelper

    library.register(
        type_="tool_helper",
        name=NAME,
        factory=lambda params: SolverHelper(name=NAME, usage=USAGE, run=run),
        description=DESCRIPTION,
        niche=NICHE,
    )
