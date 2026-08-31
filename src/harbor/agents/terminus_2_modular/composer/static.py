"""Static Composer: returns the gen_0 default bundle unless an editor-written
`active_bundle.json` (under modules_root) overrides specific module types.

The composer logic itself stays fixed Kernel code; the only dynamic input is
the editor-writable bundle-override file (validated + fallback in
`bundle_config.load_bundle_override`). This is what lets the editor ENABLE a
module it just created — without it, a newly spawned module would be discovered
by the library but never selected. A later LLM-driven Composer can replace this.
"""

from __future__ import annotations

from harbor.agents.terminus_2_modular.composer.bundle_config import (
    load_bundle_override,
)
from harbor.agents.terminus_2_modular.protocols import (
    ModuleBundle,
    ModuleCtx,
    ModuleInfo,
    ModuleSpec,
    ModuleStatsView,
)


# The gen_0 default bundle. Treat this as the "always-works" fallback that
# the LLM composer's retries also fall back to.
DEFAULT_BUNDLE = ModuleBundle(
    agent_loop=ModuleSpec(name="baseline"),
    observation=ModuleSpec(name="baseline"),
    # The context_mgmt baseline (3-step QA summarize) with enable_summarize=False
    # acts as passthrough; the agent forwards `enable_summarize` from ctor, so
    # this is the safe default that also enables summarization when configured.
    context_mgmt=ModuleSpec(name="baseline"),
    tools=ModuleSpec(name="baseline"),
    verification=ModuleSpec(name="baseline"),
)


class StaticComposer:
    """Returns DEFAULT_BUNDLE, unless an editor-written active_bundle.json
    under modules_root overrides specific module types (validated; bad/missing
    config falls back to DEFAULT_BUNDLE)."""

    async def choose(
        self,
        instruction: str,
        library: list[ModuleInfo],
        stats: ModuleStatsView,
        ctx: ModuleCtx,
    ) -> ModuleBundle:
        override = load_bundle_override(
            modules_root=ctx.state.modules_root,
            library=library,
            default_bundle=DEFAULT_BUNDLE,
            logger=ctx.services.logger,
        )
        return override if override is not None else DEFAULT_BUNDLE
