"""Multi-line exact in-place file replacement.

The baseline tools module only dispatches a helper when the typed command
contains none of |&;><`$\n, so single-line edit_file cannot handle multi-line
content. This helper uses a `<<<<<<<<` marker between old and new text, and
bypasses the metacharacter check via the tools variant's dispatch logic.

The old and new text are base64-encoded to safely pass through the shell.
Replace the FIRST exact occurrence of <old> with <new> in <path>.
"""

import base64

NAME = "edit_block"
USAGE = (
    "edit_block <path> <old> <<<<<<<< <new> — replace the FIRST exact "
    "multi-line occurrence of <old> with <new> in <path>. Put the marker "
    "`<<<<<<<<` on its own line between old and new. Both may span lines "
    "and contain any characters. Returns 'edited <path>: replaced 1 "
    "multi-line occurrence' on success."
)
DESCRIPTION = (
    "Multi-line exact in-place file replacement via base64-encoded "
    "old/new strings; the safe alternative to fragile sed/heredoc "
    "surgery when editing multi-line code blocks."
)
NICHE = {"op": "edit", "target": "file", "scope": "multi-line"}


async def run(args, ctx):
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
        + __import__("shlex").quote(script)
        + " "
        + __import__("shlex").quote(path)
        + " "
        + old_b64
        + " "
        + new_b64
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
