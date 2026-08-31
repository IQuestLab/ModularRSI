"""Protocol definitions for the modular self-evo terminus-2 agent.

The Kernel owns these contracts. Modules implement them. The Editor (when we
build it) is allowed to spawn / mutate modules, but never to change these
contracts.

Note: this file is going to grow as we port terminus-2 features in phases.
Right now it covers what Phase 2 (real tmux tools) needs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from harbor.environments.base import BaseEnvironment
from harbor.llms.chat import Chat
from harbor.models.task.config import MCPServerConfig


# ---------- Runtime context exposed to modules ----------


@dataclass(frozen=True)
class RuntimeState:
    """Read-only snapshot of the current task's config. Modules see this; they
    cannot mutate it. Mutable shared resources (tmux session) live in
    SharedResources. Mutable shared state (chat, observation buffer) is threaded
    explicitly through Protocol method args/returns."""

    task_id: str
    gen_id: str
    # env is required for SOLVER mode (talks to Docker). Editor mode runs
    # locally and passes None.
    env: BaseEnvironment | None
    instruction: str
    started_at: float
    model_name: str
    logs_dir: Path
    skills_dir: str | None
    mcp_servers: list[MCPServerConfig]
    extra_env: dict[str, str] | None
    record_terminal_session: bool
    # New (Editor agent support):
    # - staging_dir: when running in editor mode, all edit_file / create_file
    #   actions operate on this local filesystem path (copy of a parent gen's
    #   modules/). None for solver mode.
    # - modules_root: where to load the ModuleLibrary from. None means use the
    #   package default (`terminus_2_modular/modules/`). Set to e.g.
    #   `gen_archive/gen_5/modules/` to load a specific generation.
    staging_dir: Path | None = None
    modules_root: Path | None = None
    # - trajectory_root: when set (editor/self-evo mode), the editor's file
    #   tools may additionally READ (never write) files under this path — the
    #   batch's solver output dir (trajectories, per-episode prompt/response,
    #   result.json, terminal panes). Lets the editor investigate the raw run
    #   itself instead of only consuming a pre-digested summary. None = no
    #   extra read access; writes are always confined to staging_dir.
    trajectory_root: Path | None = None
    # - archive_path: the run's archive.json (niche/genealogy registry). Lets the
    #   read-only `archive` editor tool answer "what cells/lineage already exist".
    #   None → the tool falls back to computing niches on-the-fly from staging_dir.
    archive_path: Path | None = None
    # - locked_module_type: single-module ablation lock. When set (e.g.
    #   "agent_loop"), this experiment may only evolve that one module type: the
    #   editor's file tools reject writes outside modules/<locked>/. None = no
    #   lock.
    # - composer_scope: how far the per-task composer may range, INDEPENDENTLY of
    #   the write lock. "locked" (default) = only the locked type is picked among
    #   its variants, every other type sits on its library default — the K rolls
    #   of a task then differ in one variable, which is what makes a parallel
    #   single-lock ablation attributable. "all" = every type is picked normally
    #   under the usual rules (archive skip + niche dedup), lock or no lock.
    #   SERIAL evolution needs "all": a lineage that inherits a previous
    #   lineage's tree must be able to USE the variants it inherited, and
    #   freezing them to one pick would throw away the search space that lineage
    #   produced (the more variants it produced, the more is thrown away).
    locked_module_type: str | None = None
    composer_scope: str = "locked"


@dataclass
class SharedResources:
    """Mutable resources created at task setup time and shared across modules.

    Tools owns the lifecycle: tools.setup() populates these; tools.teardown()
    clears. Other modules read-only.
    """

    # Tmux session is the canonical example: created by tools.setup, used by
    # tools.execute (send keys), observation.capture (read pane), and
    # services.trajectory (asciinema markers).
    tmux_session: Any | None = None  # avoid hard import on TmuxSession


class FailureReporter(Protocol):
    def raise_tag(self, tag: str, module_name: str, detail: str = "") -> None: ...


class StatsWriter(Protocol):
    def record(self, module_name: str, metric: str, value: float) -> None: ...


class Logger(Protocol):
    def debug(self, msg: str, *args: Any) -> None: ...
    def info(self, msg: str, *args: Any) -> None: ...
    def warning(self, msg: str, *args: Any) -> None: ...
    def error(self, msg: str, *args: Any) -> None: ...


class TrajectoryRecorder(Protocol):
    """ATIF trajectory append/dump. Phase 4 implements; Phase 2 can no-op."""

    def append_step(self, step: Any) -> None: ...
    def dump(self) -> None: ...
    def record_asciinema_marker(self, marker_text: str) -> None: ...


@dataclass
class KernelServices:
    failures: FailureReporter
    stats: StatsWriter
    logger: Logger
    trajectory: TrajectoryRecorder


@dataclass
class ModuleCtx:
    """Single argument every Protocol method takes.

    - state: read-only task state
    - services: mutating Kernel-provided services
    - shared: mutable per-task resources (tmux session etc.)
    """

    state: RuntimeState
    services: KernelServices
    shared: SharedResources


# ---------- Shared module-IO datatypes ----------


@dataclass
class SolverHelper:
    """One editor-authored solver tool, as handed to the `tools` module.

    A helper is a whole extra ACTION for the agent (grep, read_file, …) obtained
    without touching the grammar template or the parser: the tools module
    intercepts a command whose first token is `name` and returns `run`'s text.

    Helpers are a first-class archived unit (`type="tool_helper"`), so each one
    carries a niche, lineage and status like any module variant, and is selected
    per task rather than auto-loaded off the filesystem. They differ from module
    variants in ARITY: a task gets a SUBSET of helpers all at once, not one
    winner — you want grep AND read_file, not either/or.
    """

    name: str  # the command token the agent types
    usage: str  # the one line describing it in the agent's prompt
    run: Any  # (args: list[str], ctx) -> str | Awaitable[str]


@dataclass
class ObsResult:
    """What `observation.capture()` returns."""

    text: str  # human-readable text fed to LLM (latest terminal pane)


@dataclass
class ObsState:
    """Carried across calls so observation can do incremental diffs etc."""

    prev_terminal: str = ""


@dataclass
class ToolCall:
    """A single command the LLM wants to run. For tmux tools, this represents
    one <keystrokes> block."""

    keystrokes: str
    duration_sec: float = 1.0


@dataclass
class ToolResult:
    success: bool
    output: str
    error: str = ""


@dataclass
class ToolSpec:
    name: str
    description: str
    params_schema: dict[str, Any]


@dataclass
class LLMResponseParseResult:
    """Result of parsing the LLM response. Mirrors terminus_2's ParseResult so
    we can reuse the existing parsers."""

    commands: list[ToolCall]
    is_task_complete: bool
    analysis: str = ""
    plan: str = ""
    error: str = ""
    warning: str = ""


@dataclass
class AgentLoopState:
    """Snapshot passed to verification to decide termination."""

    step_idx: int
    last_obs: ObsResult | None
    last_tool_result: ToolResult | None
    chat: Chat
    consecutive_complete_signals: int = 0


@dataclass
class AgentLoopResult:
    success: bool
    final_text: str = ""
    failure_tag: str | None = None  # e.g. "context_overflow", "parse_error"


# ---------- The 5 module Protocols ----------


@runtime_checkable
class Observation(Protocol):
    """env raw output → agent-readable observation. Reads from
    ctx.shared.tmux_session in the tmux-based variant."""

    async def capture(
        self,
        prev: ObsState,
        ctx: ModuleCtx,
    ) -> tuple[ObsResult, ObsState]: ...


@dataclass
class CompressResult:
    """Returned by ContextMgmt.maybe_compress.

    - `chat`: the (possibly modified) Chat object to use going forward
    - `handoff_prompt`: when not None, agent_loop should use this as the NEXT
      prompt for the LLM call (replacing whatever it was about to send),
      AND record it as a user-source Step
    - `subagent_refs`: SubagentTrajectoryRef list to record as a system-source
      Step (before the agent step) when summarization occurred
    - `summarization_occurred`: bumps recorder.summarization_count so
      linear_history splits trajectory files properly
    """

    chat: Chat
    handoff_prompt: str | None = None
    subagent_refs: list[Any] | None = None
    summarization_occurred: bool = False


@runtime_checkable
class ContextMgmt(Protocol):
    """Decide if/when to compress chat history."""

    async def maybe_compress(
        self, chat: Chat, original_instruction: str, ctx: ModuleCtx
    ) -> CompressResult: ...

    async def force_summarize(
        self, chat: Chat, original_instruction: str, ctx: ModuleCtx
    ) -> CompressResult:
        """Called by agent_loop when an LLM call raises
        ContextLengthExceededError. The implementation should ALWAYS try to
        summarize (no threshold check). Passthrough impls return the chat
        unchanged. Full impls implement a 3-tier fallback: full QA summary →
        short summary (1 LLM call) → ultimate fallback (no LLM, just terminal).
        """
        ...


@runtime_checkable
class ToolSet(Protocol):
    """Action surface: tool grammar + prompt templates + execution.

    Lifecycle: Kernel calls setup() once per task after Composer chooses;
    teardown() at the end. setup() typically creates the tmux session and
    stashes it into ctx.shared.
    """

    async def setup(self, ctx: ModuleCtx) -> None: ...

    async def teardown(self, ctx: ModuleCtx) -> None: ...

    def format_initial_prompt(
        self,
        instruction: str,
        terminal_state: str,
        ctx: ModuleCtx,
    ) -> str: ...

    def format_continuation_prompt(
        self,
        terminal_state: str,
        ctx: ModuleCtx,
    ) -> str: ...

    def parse_llm_response(self, response: str) -> LLMResponseParseResult: ...

    async def execute(
        self,
        call: ToolCall,
        ctx: ModuleCtx,
    ) -> ToolResult: ...


@runtime_checkable
class VerificationLoop(Protocol):
    """Decide if the agent is done (or should keep going)."""

    async def should_terminate(
        self,
        state: AgentLoopState,
        ctx: ModuleCtx,
    ) -> tuple[bool, str]: ...  # (terminate, reason)


@runtime_checkable
class AgentLoop(Protocol):
    """Main coordinator. Orchestrates other 4 modules to solve the task."""

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


# ---------- Composer + Bundle ----------


@dataclass
class ModuleSpec:
    name: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModuleBundle:
    agent_loop: ModuleSpec
    observation: ModuleSpec
    context_mgmt: ModuleSpec
    tools: ModuleSpec
    verification: ModuleSpec


@dataclass
class ModuleInfo:
    name: str
    type: str
    description: str
    params_schema: dict[str, Any]
    # Behavior descriptor for MAP-Elites-style niching (see niche.py). Free-form
    # {axis: value}; empty for legacy variants that predate the niche system.
    niche: dict[str, str] = field(default_factory=dict)


@dataclass
class ModuleStatsView:
    per_module: dict[str, dict[str, float]] = field(default_factory=dict)


@runtime_checkable
class Composer(Protocol):
    async def choose(
        self,
        instruction: str,
        library: list[ModuleInfo],
        stats: ModuleStatsView,
        ctx: ModuleCtx,
    ) -> ModuleBundle: ...
