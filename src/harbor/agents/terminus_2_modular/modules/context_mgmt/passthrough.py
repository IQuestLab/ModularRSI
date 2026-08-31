"""Editor-only pass-through context adapter.

This is not a second Terminus-2 solver strategy.  The solver's native
``context_mgmt/baseline.py`` already implements both runtime branches: summarize
when enabled and triggered, otherwise pass the history through unchanged.  The
editor uses this small adapter because its local sessions deliberately disable
summarization; the dynamic solver composer excludes it.
"""

from __future__ import annotations

from harbor.agents.terminus_2_modular.protocols import CompressResult, ModuleCtx
from harbor.llms.chat import Chat


class PassthroughContextMgmt:
    NAME = "passthrough"
    NICHE = {"strategy": "passthrough", "audience": "editor"}
    DESCRIPTION = (
        "Editor-only adapter that never compresses or unwinds chat history; "
        "not a Terminus-2 solver alternative."
    )
    PARAMS_SCHEMA = {
        "enable_summarize": "bool (ignored here)",
        "proactive_summarization_threshold": "int (ignored here)",
    }

    def __init__(self, **_ignored):
        pass

    async def maybe_compress(
        self, chat: Chat, original_instruction: str, ctx: ModuleCtx
    ) -> CompressResult:
        return CompressResult(chat=chat)

    async def force_summarize(
        self, chat: Chat, original_instruction: str, ctx: ModuleCtx
    ) -> CompressResult:
        return CompressResult(chat=chat)

    def get_subagent_metrics(self) -> dict:
        return {
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_cached_tokens": 0,
            "total_cost_usd": 0.0,
            "rollout_details": [],
        }


def register(library):
    library.register(
        type_="context_mgmt",
        name=PassthroughContextMgmt.NAME,
        factory=lambda params: PassthroughContextMgmt(**params),
        description=PassthroughContextMgmt.DESCRIPTION,
        params_schema=PassthroughContextMgmt.PARAMS_SCHEMA,
        niche=PassthroughContextMgmt.NICHE,
    )
