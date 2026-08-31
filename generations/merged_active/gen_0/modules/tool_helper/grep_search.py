"""Content search across the sandbox.

Replaces the locate-by-ls-then-cat-each-candidate loop when the agent needs to
find where something is defined or referenced in a draft project. The NAME is
deliberately `grep_search`, not `grep`, so it does not shadow the real shell
command.
"""

import shlex

NAME = "grep_search"
USAGE = (
    "grep_search <pattern> [path] — regex-search files under path (default .), "
    "return up to 50 matches as file:line: text. Pattern must avoid | & ; > < ` $ "
    "and newlines (helper dispatch restriction)."
)
DESCRIPTION = (
    "Recursive content search by regex; closes the repeated ls/find/cat churn "
    "when locating definitions or usages in draft/edit tasks."
)
NICHE = {"op": "search", "target": "file-content"}


async def run(args, ctx):
    if not args:
        return "usage: grep_search <pattern> [path]"
    pattern = args[0]
    path = args[1] if len(args) > 1 else "."
    # -m 50 bounds matches per file; we additionally cap the total lines below
    # so a huge tree can't flood the observation. No pipe: truncate in Python.
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


def register(library):
    from harbor.agents.terminus_2_modular.protocols import SolverHelper

    library.register(
        type_="tool_helper",
        name=NAME,
        factory=lambda params: SolverHelper(name=NAME, usage=USAGE, run=run),
        description=DESCRIPTION,
        niche=NICHE,
    )
