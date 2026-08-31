"""EditorStaticComposer: default bundle for editor mode.

Same general-purpose modules as the solver, EXCEPT:
- `tools` = `editor_file_tools` (edit/read/create on staging dir)
- `context_mgmt` = `passthrough` (editor tasks are short; no summarization)

Everything else stays the same — agent_loop, observation, verification.
That way the editor goes through the same baseline loop machinery as the
solver, and any improvements to those modules apply to both roles.
"""

from __future__ import annotations

from harbor.agents.terminus_2_modular.protocols import (
    ModuleBundle,
    ModuleCtx,
    ModuleInfo,
    ModuleSpec,
    ModuleStatsView,
)


EDITOR_DEFAULT_BUNDLE = ModuleBundle(
    agent_loop=ModuleSpec(name="baseline"),
    observation=ModuleSpec(name="baseline"),
    # Editor sessions are short; no summarization needed.
    context_mgmt=ModuleSpec(name="passthrough", params={"enable_summarize": False}),
    # The crucial difference: editor uses file-edit tools instead of tmux.
    tools=ModuleSpec(name="editor_file_tools"),
    verification=ModuleSpec(name="baseline"),
)


class EditorStaticComposer:
    """Always returns EDITOR_DEFAULT_BUNDLE."""

    async def choose(
        self,
        instruction: str,
        library: list[ModuleInfo],
        stats: ModuleStatsView,
        ctx: ModuleCtx,
    ) -> ModuleBundle:
        return EDITOR_DEFAULT_BUNDLE
