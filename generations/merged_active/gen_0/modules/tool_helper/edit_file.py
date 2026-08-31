"""Atomic exact-match in-place file edits.

The agent's raw-keystroke action surface makes in-place edits error-prone
(heredoc surgery, repeated sed, then cat to verify). This helper replaces the
FIRST exact occurrence of a string with another in one atomic step and reports
what changed — the agent can re-read the file afterward to confirm.

The baseline tools module only dispatches a helper when the typed command
contains none of `|&;><`$\\n`, so `old`/`new` must avoid those characters.
Spaces are fine (quote multi-word arguments).
"""

import shlex

NAME = "edit_file"
USAGE = (
    "edit_file <path> <old> <new> — replace the FIRST exact occurrence of <old> "
    "with <new> in a file and report the edited line(s). Avoid | & ; > < ` $ and "
    "newlines in <old>/<new>; quote multi-word arguments."
)
DESCRIPTION = (
    "Atomic exact-match in-place file edit with confirmation; the safe "
    "replacement for sed/heredoc churn when editing draft files."
)
NICHE = {"op": "edit", "target": "file"}


async def run(args, ctx):
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


def register(library):
    from harbor.agents.terminus_2_modular.protocols import SolverHelper

    library.register(
        type_="tool_helper",
        name=NAME,
        factory=lambda params: SolverHelper(name=NAME, usage=USAGE, run=run),
        description=DESCRIPTION,
        niche=NICHE,
    )
