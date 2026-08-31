---
name: kernel-boundaries
description: Which files are Kernel (NEVER edit, off-limits) versus which are evolvable modules. Read this BEFORE making any edits.
---

# Kernel boundaries

The codebase you're editing is split into **Kernel** (fixed plumbing) and
**Modules** (evolvable strategy). You MAY only modify modules.

## Kernel — DO NOT EDIT

Anything at the top level of `staging/` that is NOT a `modules/<type>/` file:

| File / Dir | What it does | Why off-limits |
|---|---|---|
| `protocols.py` | Defines the Protocol contracts every module must satisfy | Changing it would invalidate every existing module's interface |
| `services.py` | KernelServices: logger / failure reporter / stats writer / ATIF trajectory recorder | Wired up at startup; modules consume but don't define these |
| `agent.py` | `Terminus2Modular` / `Terminus2ModularEditor` classes; agent lifecycle | This file is what loads YOU |
| `library.py` | Auto-discovers modules (so you do NOT need to update imports here when adding a new module) | Auto-discovery means you don't need to touch it |
| `composer/` | The Composer that picks which module variant to use PER TASK (reads the task + each variant's DESCRIPTION) | `llm_dynamic` per-task selection; off-limits to you — you grow the library, it selects |
| `__init__.py` | Package exports | — |

The editor tooling will REJECT any `<edit_file>` or `<create_file>` action
that targets these files with `path is off-limits (Kernel)`.

## Modules — SAFE to edit

Everything under `modules/`. Five module types, each with sub-directories:

```
modules/
├── agent_loop/        ← main control loop incl. planning strategy (ReAct / plan-execute / ToT / ...)
├── observation/       ← env raw output → agent-readable obs
├── context_mgmt/      ← summarize / unwind / compress chat history
├── tools/             ← what actions the agent can take (tmux send-keys etc.)
└── verification/      ← decide when agent has finished the task
```

You can:
- **Modify** a routed, non-baseline variant when the action explicitly permits
  it. Every `baseline.py` is read-only; replacing baseline behavior requires a
  new variant.
- **Create** new `.py` files under any `modules/<type>/` directory.
- New files are auto-discovered by `library.py` on next startup — you do NOT
  need to register them anywhere.

You CANNOT:
- Modify Kernel files (above).
- Delete files (tooling has no delete action).
- Rename files (no rename action).
- Add Python dependencies (no way to touch `pyproject.toml`).

## Quick reference

If you ever wonder "is this file editable?":

- Path starts with `modules/<type>/`, ends in `.py`, and is not a baseline or
  other protected file? → **YES**, subject to the routed action.
- Otherwise → **NO**, off-limits.
