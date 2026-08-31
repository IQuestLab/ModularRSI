"""Pass-through context management: never compress, never unwind.

Fine for short tasks (hello-world). For long tasks this will eventually
hit the model context limit; future modules should implement real
summarization (3-step QA / sliding window / hierarchical etc.)."""

from __future__ import annotations

from harbor.agents.terminus_2_modular.protocols import CompressResult, ModuleCtx
from harbor.llms.chat import Chat


class PassthroughContextMgmt:
    NAME = "passthrough"
    NICHE = {"strategy": "passthrough"}
    DESCRIPTION = (
        "Does not compress or unwind chat history. Suitable only for short "
        "tasks that fit in the model context window."
    )
    PARAMS_SCHEMA = {
        # accepted for forward-compat with Terminus2 ctor surface; passthrough
        # ignores them, real summarization module (three_step_qa) will honor.
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
        # Passthrough has nothing to summarize with; just return unchanged.
        return CompressResult(chat=chat)

    def get_subagent_metrics(self) -> dict:
        """Aggregate stats produced by subagent calls (this module has none)."""
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
