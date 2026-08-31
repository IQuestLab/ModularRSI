---
name: adding-a-solver-tool
description: How to give the SOLVER a brand-new tool (a command it can call) cheaply, without touching the response grammar or parser. Read this when your diagnosis shows the agent lacks a capability — e.g. it does a routine the hard way across many tasks.
---

# Adding a new tool to the solver

The solver's only built-in action is "type keystrokes into tmux." This skill
covers HOW to add a new tool (a command the agent can call) cheaply. WHETHER a
tool is the right response — versus tuning an existing variant, changing a
different module, or making no change — follows from your diagnosis (see
`failure-mode-checklist`); a tool is the answer only when the evidence shows the
agent lacks a capability it needs repeatedly, not by default.

You do NOT need to touch the grammar template or the parser (those live outside
your editable tree). You add a **helper command**: a module file under
`modules/tool_helper/` that the Composer selects per task and the live `tools`
module advertises to the agent and dispatches to.

## When to add one (and when not to)

- ADD one when the BEHAVIOR/TOOL evidence shows a recurring, GENERAL capability
  gap — something a single command would close for *many* tasks (e.g. the agent
  repeatedly navigates/searches by hand, or loses output to truncation).
- Do NOT add one for a one-task quirk, or for something the agent already does
  fine. A tool nobody reaches for is dead weight. Match the tool to the cause.
- This is one option among the three in `failure-mode-checklist` (tune / new
  module variant / new tool) — pick it only when the cause is a missing
  capability.

## The contract

Create `modules/tool_helper/<name>.py`. A helper is its OWN module type — an
ordinary module file, not a loose script in a subdirectory — so it looks like any
variant you would write, plus the two action-specific names `USAGE` and `run`:

```python
NAME = "your_tool"          # the command the agent types (the FIRST token)
USAGE = "your_tool <arg> [opt]  — one line telling the agent what it does + syntax"
DESCRIPTION = "what capability this closes, and for which kind of task"
NICHE = {"op": "search", "target": "file-content"}   # its cell in the archive grid

async def run(args: list[str], ctx) -> str:
    """args = the tokens AFTER the command name. Return TEXT shown to the agent.
    Act on the task sandbox through ctx.state.env.exec(...)."""
    ...
    return "result text the agent will read"


def register(library):
    from harbor.agents.terminus_2_modular.protocols import SolverHelper
    library.register(
        type_="tool_helper",
        name=NAME,
        factory=lambda params: SolverHelper(name=NAME, usage=USAGE, run=run),
        description=DESCRIPTION,
        niche=NICHE,
    )
```

What happens at task time (all automatic — you only write the file):

1. The Composer picks the SUBSET of active helpers for the task and the Kernel
   instantiates them from the library. Each chosen helper's `USAGE` line is
   appended to the agent's initial prompt, so the agent knows the command exists.
2. When the agent issues a command whose first token is your `NAME` (as a
   standalone command — no pipes/redirection), the `tools` module calls your
   `run()` and feeds the returned string straight back to the agent as its
   observation. Any other command goes to tmux as normal.

Helpers are **subset-selected, not pick-one**: a task can be given several at
once (you want `grep` AND `read_file`), unlike the five module types where the
Composer picks exactly one variant each.

## Your helper is archived like a variant — so place it and supersede it

Because it registers under `type_="tool_helper"`, your helper gets an archive
entry, a niche cell, DAG lineage and a status, exactly like a module variant:

- `NICHE` is REQUIRED. Helpers used to be loose files with no niche, which meant
  they could only ever pile up — never deduped, never benched, never rolled back.
  Survey `<archive type="tool_helper"/>` first.
- If an existing helper already occupies your cell, MERGE into it (broaden that
  file) or declare `SUPERSEDES: <name>` — do not ship a near-duplicate. Two
  helpers that do almost the same thing split the evidence and neither can be
  judged.
- Avoid a `NAME` that shadows a real shell command (a helper named `grep`
  intercepts the agent's `grep`, which is rarely what you want). Prefer a
  distinct token.

## How to act on the sandbox

The agent's files live in the task sandbox, reachable via the environment:

```python
res = await ctx.state.env.exec("sed -n '1,50p' /app/foo.py", timeout_sec=10)
# res.stdout / res.stderr / res.return_code
return res.stdout
```

Because your output is returned DIRECTLY (not through the tmux pane), it is NOT
subject to the pane's truncation — a read/search tool can hand the agent exactly
the slice it asked for. That is the main reason a tool can beat raw `cat`.

## Rules & gotchas

- `run(args, ctx) -> str` should be `async def` (you'll usually `await
  ctx.state.env.exec(...)`); a plain `def` also works. Keep it total: catch your
  own errors and return a readable message rather than raising.
- Keep it GENERAL and bounded in output (don't dump megabytes — page it).
- No sibling imports inside a helper file (the gen tree isn't a package); use
  absolute `harbor...` imports or the standard library only. `ctx` gives you
  everything you need (`ctx.state.env`, `ctx.services.logger`).
- Invocation is single-command only (a line with `|`, `&&`, `;`, `>` falls
  through to tmux). Tell the agent, via `USAGE`, to call it on its own.
- **You do NOT enable it** — the Composer does, per task, from the archive. No
  `active_bundle.json` entry is needed. A helper whose archive status is
  `superseded` may still be discovered in the additive library, but the Composer
  filters it out of solver selection.
- Do NOT put helpers in `modules/tools/helpers/` (the OLD location). Module
  discovery only globs `<type>/*.py` and never descends into a subdirectory, so a
  file there registers nothing and the solver never sees it. Smoke fails on it.
- `<validate/>` then `<commit_patch/>` as usual. A helper that raises at run time
  is reported as a failed tool result rather than crashing the solver — so a
  broken one won't take the run down, but it also won't help. Check your logic.

## Decide it from the evidence

This skill is purely mechanical — it tells you HOW to add a tool, not WHICH tool
to add or whether to. That comes from your diagnosis of the actual trajectories:
name the recurring routine the agent does the hard way, design the one command
that collapses it, and confirm it generalizes beyond a single task.
