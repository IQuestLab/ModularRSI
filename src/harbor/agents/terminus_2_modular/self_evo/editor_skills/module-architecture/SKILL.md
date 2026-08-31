---
name: module-architecture
description: The five module types and their Protocol contracts. Read this BEFORE proposing edits that change module behavior.
---

# Module architecture

The agent is composed of 5 modules at task time. Each implements a Python
Protocol defined in `protocols.py` (off-limits to edit). When proposing an
edit, you MUST preserve the Protocol signature — changing it would break
every other gen.

## The module library + per-task selection

The library starts with these implementations; the editor ADDS more variants over
time (this list grows):

| type | implementations (start) |
|---|---|
| `agent_loop` | `baseline` |
| `observation` | `baseline` |
| `context_mgmt` | `baseline` (`passthrough` is editor-only) |
| `tools` | `baseline`, `tmux_xml` (`editor_file_tools` is editor-only) |
| `verification` | `baseline` |

**There is no single "live" bundle.** At the start of EACH task the Composer
(`llm_dynamic`) reads that task + every variant's `DESCRIPTION` and picks the
variant of each type best suited to THAT task — so which variant runs depends on
the task. A variant you create does NOT run on its own, but once it is in the
library with a clear `DESCRIPTION`, the Composer will pick it for the tasks it
fits. You never hand-pick and never write `active_bundle.json`; selection is
per-task and the Composer's job. So: your job is to grow the library with
well-described variants.

NOTE: you (the editor) run a *different* bundle than the solver (your
`context_mgmt` is `passthrough`, your `tools` is `editor_file_tools`) — never edit
a module just because it is the one YOU run on.

## The 5 module types

### 1. `agent_loop`

The main coordinator. Receives instruction + the other 4 modules, runs the
"while not done: think → act → observe" loop.

```python
class AgentLoop(Protocol):
    async def run(
        self,
        initial_prompt: str,
        original_instruction: str,
        observation: Observation,
        context_mgmt: ContextMgmt,
        tools: ToolSet,
        verification: VerificationLoop,
        chat: Chat,
        ctx: ModuleCtx,
    ) -> AgentLoopResult: ...
```

`baseline.py` is a read-only faithful port of the original terminus-2 loop: it sequences
think→act→observe, drives the other 4 modules, handles parse-error fallbacks,
and records ATIF steps. Evolve it by adding a variant rather than editing it — but the agent's own
narration over-points here (it describes its own thinking), so a loop change is
rarely the true cause. See the failure-mode-checklist's reverse diagnostic.

There is no separate `planner` module — task-level planning lives INSIDE the
agent_loop (the baseline does it implicitly in each LLM turn's `<plan>`). A
"plan-and-execute" or "plan-then-replan" strategy is therefore just a different
agent_loop VARIANT, not a separate slot.

**How to change the loop — override a SEAM, don't rewrite it.** `baseline.py`
exposes named, overridable methods so a variant SUBCLASSES `BaselineAgentLoop`
and replaces ONE method (~15-30 lines) instead of copying the ~700-line `run()`.
Two kinds (every default is byte-for-byte baseline):

- **Insertion hooks — INJECT at a decision point:**
  - `_pre_llm_prompt(prompt, *, iteration, state, ctx) -> str` — shape the prompt
    before each LLM call (planning injection / re-plan trigger from `self` state).
  - `_shape_observation(observation_text, *, parse, tool_results, state, ctx) -> str`
    — change what the model sees next round (surface an error / verification reason).
  - `_should_continue(*, terminate, reason, parse, iteration, state, ctx) -> bool`
    — override the stop/continue (re-plan) cadence.
- **Phase methods — REPLACE a whole step's behavior:** `_check_session_alive`,
  `_execute_commands`, `_build_observation_text`, `_build_parse_error_prompt`,
  `_record_summarization_handoff`, `_next_prompt`.

Stash cross-iteration state on `self` and REACT to it — that is a real structural
change (a static prompt string is not). See the `adding-a-variant` skill for the
exact subclass + import recipe (`from harbor.agents.terminus_2_modular.modules.
agent_loop.baseline import BaselineAgentLoop`; NEVER load the sibling via
`spec_from_file_location` / `Path(__file__)`).

### 2. `observation`

Env raw output → agent-readable text. Reads from `ctx.shared.tmux_session`
when present.

```python
class Observation(Protocol):
    async def capture(
        self,
        prev: ObsState,
        ctx: ModuleCtx,
    ) -> tuple[ObsResult, ObsState]: ...
```

`baseline.py` does byte-accurate head+tail truncation matching the
original `Terminus2._limit_output_length`.

### 3. `context_mgmt`

Decide when chat history needs compression / summarization.

```python
class ContextMgmt(Protocol):
    async def maybe_compress(
        self, chat: Chat, original_instruction: str, ctx: ModuleCtx,
    ) -> CompressResult: ...
    async def force_summarize(
        self, chat: Chat, original_instruction: str, ctx: ModuleCtx,
    ) -> CompressResult: ...
```

`baseline.py` ports the original 3-step QA subagent flow. `passthrough.py`
does nothing and is reserved for the editor bundle; the solver Composer filters
it out.

### 4. `tools`

Action surface — what the agent can do. Defines a system-prompt section that
describes the tool grammar, parses LLM responses into ToolCalls, and executes
those calls.

```python
class ToolSet(Protocol):
    async def setup(self, ctx: ModuleCtx) -> None: ...
    async def teardown(self, ctx: ModuleCtx) -> None: ...
    def format_initial_prompt(...) -> str: ...
    def format_continuation_prompt(...) -> str: ...
    def parse_llm_response(self, response: str) -> LLMResponseParseResult: ...
    async def execute(self, call: ToolCall, ctx: ModuleCtx) -> ToolResult: ...
```

`baseline.py` is the solver default — real tmux session + asciinema +
XML/JSON parsers. Two `tools/` modules can coexist; Composer picks one.

To give the agent a NEW tool (a command it can call) without rewriting the
grammar/parser, add a **helper command**: a small `modules/tool_helper/<name>.py`
that the `tools` baseline auto-discovers, advertises in the agent's prompt, and dispatches
to. See the `adding-a-solver-tool` skill for the contract. This is the low-friction
path for a "missing capability" fix.

### 5. `verification`

Decide when the agent should terminate.

```python
class VerificationLoop(Protocol):
    async def should_terminate(
        self,
        state: AgentLoopState,
        ctx: ModuleCtx,
    ) -> tuple[bool, str]: ...
```

`baseline.py` requires `task_complete=True` in TWO consecutive LLM
responses before terminating (mirrors original terminus-2's pending_completion).

## Shared types you'll see

These are imported from `protocols.py`:

| Type | Where used |
|---|---|
| `ModuleCtx` | First arg of nearly every Protocol method. Has `state` (read-only RuntimeState), `services` (logger / trajectory / failures / stats), `shared` (mutable, e.g., `tmux_session`). |
| `Chat` | `harbor.llms.chat.Chat` — message history + LLM client wrapper. |
| `ObsResult`, `ObsState`, `ToolCall`, `ToolResult`, `LLMResponseParseResult`, `AgentLoopState`, `AgentLoopResult`, `CompressResult` | Plain dataclasses for inter-module IO. |
| `Step`, `Metrics`, `Observation`, `ObservationResult`, `Agent`, `Trajectory` | ATIF trajectory dataclasses, imported from `harbor.models.trajectories`. |

## How you evolve a module — add a new VARIANT (additive, the default)

Evolution is **additive**: you improve the agent by ADDING a new variant file
under `modules/<type>/`, NOT by overwriting the live one. The old variant stays
available for fallback and comparison, but a bad new variant can regress tasks
when selected; the gates are still required. Overwrite (`<edit_file>`) is
ONLY for fixing a bug / tuning a param inside a variant whose behavior you are not
otherwise changing. Full contract: the `adding-a-variant` skill.

This section is mechanical (*how* to add one), not *whether* you should or *what*
it should do — decide that from your diagnosis. Placeholder names below are not a
suggested design.

### Step 1: Write the module file

Create `modules/<type>/<your_name>.py`, where `<type>` is one of the five types
and `<your_name>` is a fresh filename. It must implement that type's Protocol
exactly (see the signatures above), and expose a top-level `register()`:

```python
# modules/<type>/my_variant.py
from harbor.agents.terminus_2_modular.protocols import ModuleCtx  # + the types your Protocol uses


class MyImpl:
    NAME = "my_variant"          # the name the Composer references to select it
    DESCRIPTION = "..."          # one line — the Composer READS this to decide
                                 # when this variant should be live; make it accurate
    PARAMS_SCHEMA = {}           # tunable params, or {}

    def __init__(self, **params): ...

    # ... implement the Protocol method(s) for <type>, matching the exact
    #     signature from protocols.py. Bodies are up to you.


def register(library):
    library.register(
        type_="<type>",                 # MUST match the directory / Protocol
        name=MyImpl.NAME,
        factory=lambda params: MyImpl(**params),
        description=MyImpl.DESCRIPTION,
        params_schema=MyImpl.PARAMS_SCHEMA,
    )
```

### Step 2: You do NOT enable it — the Composer does, per task

Auto-discovery REGISTERS your variant; it does NOT make the agent use it. But you
do **not** hand-pick or write `active_bundle.json` — at the start of each task the
**Composer** reads that task + every variant's `DESCRIPTION` and picks the variant
of each type best suited to that task. So your job ends at: create a well-described
variant that loads cleanly. Which task it runs on is the Composer's call, not
yours.

This is why the `DESCRIPTION` matters: it is the ONLY thing the Composer sees
about your variant. Say plainly what it does and when it should be preferred over
the alternatives. A created variant + a clear `DESCRIPTION` is ONE complete logical
change for the reflection round.
