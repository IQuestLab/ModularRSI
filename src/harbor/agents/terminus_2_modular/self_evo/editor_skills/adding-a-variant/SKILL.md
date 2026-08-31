---
name: adding-a-variant
description: How evolution works here — the library is a niche archive; ADD a new variant for an empty cell, or MERGE/SUPERSEDE (never overwrite the live module) when your idea overlaps an existing one. Read this whenever your diagnosis says a module's behavior should change. Covers create-vs-edit-vs-merge, the register contract, and why the Composer (not you) decides what goes live.
---

# Adding a module variant (additive evolution)

Evolution here is **additive**. You improve the agent by ADDING a new variant
file under `modules/<type>/`, not by rewriting the live module. The old variant
stays in the library untouched, so:

- The old variant remains available for fallback and comparison. A bad new
  variant can still regress tasks when the Composer selects it, so review,
  activation and sanity gates remain mandatory.
- The library grows in capability instead of churning one file back and forth
  (which is how earlier runs slowly drifted worse). But growth ≠ hoarding: when a
  new idea OVERLAPS an existing variant you CONSOLIDATE (merge/supersede), you
  don't pile up near-duplicates — see "When to MERGE / SUPERSEDE vs. add" below.

## If the brief names an `action`

The routed action decides the output shape:

| `action` | Output | `CHANGE:` |
|---|---|---|
| `add` | Create a new variant for an uncovered capability. | `add` |
| `modify` | Edit the named, non-baseline variant in place. | `modify` |
| `replace` | Create a new variant and declare `SUPERSEDES: <target>`. | `merge` |

`baseline.py` is read-only. A baseline incumbent must therefore be routed as
`replace`, never `modify`. On `replace`, `SUPERSEDES` is required so the archive
retires the target rather than treating the result as an unrelated addition.

## Create vs. edit — the default rule

- **Changing behavior → `<create_file>` a NEW variant.** When no routed action
  was provided, any change to what a module DOES — in any of the five module
  types — is a NEW file (e.g. `observation/<name>.py`,
  `agent_loop/<name>.py`), never an overwrite of the live implementation.
- **`<edit_file>` is only for** fixing a bug or tuning a param INSIDE a variant
  whose behavior you are not otherwise changing (e.g. a typo'd constant, a crash).
  If you're changing what the module *does*, that's a new variant, not an edit.

## When to MERGE / SUPERSEDE vs. add — the library is a niche archive

The library is not a flat pile: it is a **niche archive** that keeps ONE active
variant per behavioral cell. Every variant declares `NICHE = {axis: value}` (a
class attr next to `DESCRIPTION`) naming its cell. **Investigate before you add** —
`<archive type="<type>"/>` lists each sibling's niche, what it ADDRESSES, its
status, and flags crowded (near-duplicate) cells. Then:

- **Empty cell** (no sibling does this) → `CHANGE: add` a new variant. Declare a
  non-empty `NICHE`; a new variant with no niche is REJECTED by the gate.
- **Your idea OVERLAPS a sibling** (same niche cell, or its `ADDRESSES` already
  covers your symptom) → do NOT spawn a look-alike (the gate rejects it).
  CONSOLIDATE instead:
    - `CHANGE: merge` — build ONE variant combining the ideas and list the
      overlapping siblings under `PARENT` (comma-separated). **A merge automatically
      retires its parents** (they become `superseded`, the Composer stops selecting
      them), so the library shrinks instead of accumulating near-duplicates. Use
      `CHANGE: add` only if you truly want to keep a parent as a distinct option.
    - `SUPERSEDES: <incumbent>` — a straight replacement of an incumbent that is
      not your parent.

A new axis/value is for a genuinely NEW behavioral dimension only — never to
relabel a near-duplicate into a fresh cell and dodge dedup.

Declare provenance in a `<variant_meta>` block in your final message:

```
<variant_meta name="settled_hybrid" type="observation">
  PARENT: hybrid_fallback, fresh_capture      # comma-sep ⇒ a merge of both
  CHANGE: merge                               # add | modify | merge
  ADDRESSES: stale pane repeats old output    # the symptom it fixes
  SUPERSEDES: hybrid_fallback, fresh_capture  # incumbents retired (a merge retires its PARENTs)
</variant_meta>
```

Why this matters: earlier runs bloated the library with ~10 near-identical "gate"
agent-loops and ~13 near-identical "capture" observations because every change
defaulted to a fresh file. Consolidating overlaps keeps the archive small and the
per-task Composer's choice sharp.

## The contract

Create `modules/<type>/<your_name>.py` where `<type>` is one of the five types
(`agent_loop` / `observation` / `context_mgmt` / `tools` / `verification`). It
must implement that type's Protocol EXACTLY (see the `module-architecture` skill
for each signature) and expose a top-level `register()`:

```python
# modules/<type>/<your_name>.py
from harbor.agents.terminus_2_modular.protocols import ModuleCtx  # + types your Protocol uses


class MyVariant:
    NAME = "your_name"           # how the Composer refers to this variant
    DESCRIPTION = "..."          # ONE accurate line — see "Why DESCRIPTION matters"
    PARAMS_SCHEMA = {}           # tunable params, or {}

    def __init__(self, **params): ...

    # implement the Protocol method(s) for <type>, matching protocols.py exactly.


def register(library):
    library.register(
        type_="<type>",                       # MUST match the directory / Protocol
        name=MyVariant.NAME,
        factory=lambda params: MyVariant(**params),
        description=MyVariant.DESCRIPTION,
        params_schema=MyVariant.PARAMS_SCHEMA,
    )
```

All of this is auto-discovered: the library scans `modules/<type>/*.py` (files
starting with `_` are skipped) and calls `register()`. No central registry edit.

## Subclassing the baseline — the standard way to write a variant

EVERY module type's default implementation lives in `<type>/baseline.py` as
`Baseline<Type>`. It is READ-ONLY, and it is what your variant should subclass:
build ON it rather than rewriting the module, and override exactly ONE method
instead of copying the whole file. Import the parent by its **absolute package
path** (never `spec_from_file_location` / `Path(__file__)`).

| type | parent class (in `<type>/baseline.py`) | override ONE of |
|---|---|---|
| `agent_loop` | `BaselineAgentLoop` | `_pre_llm_prompt` / `_shape_observation` / `_should_continue`, or one named phase method of `run()` |
| `observation` | `BaselineObservation` | `capture` |
| `context_mgmt` | `BaselineContextMgmt` | `maybe_compress` / `force_summarize` |
| `tools` | `BaselineTools` | `parse_llm_response` / `execute` / `format_initial_prompt` / `format_continuation_prompt` |
| `verification` | `BaselineVerification` | `should_terminate` |

Worked example (`agent_loop` — same shape for every type):

```python
from harbor.agents.terminus_2_modular.modules.agent_loop.baseline import (
    BaselineAgentLoop,
)


class MyLoop(BaselineAgentLoop):
    NAME = "my_loop"
    NICHE = {"grounding": "reactive", "drift": "none", "parse": "baseline"}
    DESCRIPTION = "baseline + <the one behavior you change, and for which tasks>"

    # Override exactly ONE hook (defaults are byte-for-byte baseline; see the
    # "Control-flow hooks" section of baseline.py for the full list + signatures):
    #   _pre_llm_prompt(prompt, *, iteration, state, ctx)      -> str
    #   _shape_observation(observation_text, *, parse, tool_results, state, ctx) -> str
    #   _should_continue(*, terminate, reason, parse, iteration, state, ctx) -> bool
    def _shape_observation(self, observation_text, *, parse, tool_results, state, ctx):
        # stash cross-iteration state on `self` and REACT to it — that's structural
        return observation_text


def register(library):
    library.register(
        type_="agent_loop",
        name=MyLoop.NAME,
        factory=lambda params: MyLoop(**params),
        description=MyLoop.DESCRIPTION,
        params_schema={},
        niche=MyLoop.NICHE,
    )
```

The absolute `harbor...baseline import` **always works**, even when your variant
is loaded from a `gen_N/` snapshot in path-mode — it resolves to the installed,
stable baseline. Do **NOT** hand-roll `importlib.util.spec_from_file_location`
or `Path(__file__).parent / "baseline.py"` to load the sibling: `spec.loader`
is `None` there, so it fails with `AttributeError: 'NoneType' object has no
attribute '__dict__'` and you will burn edits fighting it. That anti-pattern is
exactly what the "no sibling imports" gotcha below means — the absolute import
is the sanctioned way.

## You do NOT enable it — the Composer does, PER TASK

Registering a variant does NOT make the agent run it. At the start of EACH task a
**Composer** reads that task + every variant's `DESCRIPTION` and picks the variant
of each module type best suited TO THAT TASK. So the same variant may be chosen on
some tasks and skipped on others — there is no single "live" choice you set, and
nothing to hand-edit (no `active_bundle.json` to write). Selection is the
Composer's job, decided per task. Your job ends at: a well-described variant that
loads cleanly.

### Why DESCRIPTION matters

The `DESCRIPTION` is the ONLY thing the Composer sees about your variant when it
picks per task. Write it to say plainly *what the variant does* and *for which
kind of task it is the right choice* — e.g. "tail-biased terminal truncation:
keeps the last 75% of output where command results/errors live; prefer for tasks
whose output is long and the part that matters is at the end." A vague description
("improved observation") gives the Composer nothing to match against a task.

## Gotchas

- Don't change Protocol signatures — that breaks Composer dispatch for every gen.
- No RELATIVE sibling imports (`from .baseline import ...`) — the gen tree isn't a
  package on `sys.path`. To reuse a sibling (e.g. subclass `BaselineAgentLoop`),
  import it by its ABSOLUTE `harbor...` path (see "Subclassing an existing variant"
  above); that always resolves. Resolve any external resource (templates, etc.) via
  `importlib.import_module("harbor...")`, never `Path(__file__).parent...` — the
  latter breaks when the variant loads from a `gen_N/` snapshot at a different path.
- `<validate/>` then `<commit_patch/>`. A variant that fails to import is caught
  by smoke before promotion — but a variant that imports yet is wrong just sits
  unused, so check your logic.

## Related

- `module-architecture` — the five module types and their exact Protocol signatures.
- `adding-a-solver-tool` — a *tool* (a command the agent can call) is a special
  additive case: a `modules/tool_helper/<name>.py` that the library discovers
  and the Composer subset-selects per task. The live `tools` module advertises
  and dispatches the selected helpers.
- `failure-mode-checklist` — how to attribute a failure to a module from the trace.

## Runtime visibility: your mechanism must ACTUALLY fire

Every Protocol-method call of every active variant is kernel-traced: one line
per call lands in the trial's `trajectory.json` under `agent.extra.module_trace`
(and in `trial.log`). When you reflect on later batches you WILL see whether
your variant's methods fired and what they returned — and so will the review
gate. A mechanism that never activates is dead weight and evidence against the
change that added it.

Before committing, trace the DATA PATH of your feature end to end: where does
its input come from, which call populates the state it reads, what observable
output proves it ran? A checklist renderer whose step list is never populated,
a counter that is never read, a regex that can't match the real marker format —
all of these import cleanly, pass smoke, and do nothing. If you cannot point at
the line that FEEDS your mechanism, the mechanism does not exist.

For internal details the generic trace can't see (e.g. WHICH lines you
truncated), call `ctx.services.trajectory.trace("type:variant", "note", "...")`
yourself at the key decision points.

## Discovery contract (smoke-enforced)

Smoke now HARD-FAILS on: empty `.py` files anywhere under a type dir; a
non-underscore file directly under a type dir whose `register()` is missing,
raises, or registers nothing; a `tool_helper/*.py` without `NAME: str` + callable
`run`. Shared code belongs in an underscore-prefixed file (`_shared.py`).
