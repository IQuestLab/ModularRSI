"""One-shot file discovery by glob pattern.

Replaces repeated ls/find invocations when the agent must locate the draft file
before reading or editing it.
"""

import shlex

NAME = "glob_files"
USAGE = (
    "glob_files <pattern> — list files matching a glob, e.g. glob_files '**/*.py' "
    "or glob_files '*.json' (up to 100 entries)"
)
DESCRIPTION = (
    "Fast file discovery by name pattern; replaces repeated ls/find when the "
    "agent must locate the draft file before reading or editing it."
)
NICHE = {"op": "list", "target": "files"}


async def run(args, ctx):
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


def register(library):
    from harbor.agents.terminus_2_modular.protocols import SolverHelper

    library.register(
        type_="tool_helper",
        name=NAME,
        factory=lambda params: SolverHelper(name=NAME, usage=USAGE, run=run),
        description=DESCRIPTION,
        niche=NICHE,
    )
