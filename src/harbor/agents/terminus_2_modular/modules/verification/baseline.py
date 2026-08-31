"""Two-phase verification: only terminate after the LLM has declared task
complete in two consecutive steps. This is a simplified version of the
verification logic in `terminus_2.py`.

In the modular system, the agent_loop is responsible for tracking
`consecutive_complete_signals` on the AgentLoopState; this module merely
reads that count.
"""

from __future__ import annotations

from harbor.agents.terminus_2_modular.protocols import AgentLoopState, ModuleCtx


class BaselineVerification:
    NAME = "baseline"
    NICHE = {"evidence": "executed-ever", "error_gate": "none"}
    DESCRIPTION = (
        "Terminate after the LLM has signaled 'task_complete=True' in two "
        "consecutive steps. Reduces false-positive task completion."
    )
    PARAMS_SCHEMA = {"required_consecutive": "int (default 2)"}

    def __init__(self, required_consecutive: int = 2):
        self.required = max(1, required_consecutive)

    async def should_terminate(
        self,
        state: AgentLoopState,
        ctx: ModuleCtx,
    ) -> tuple[bool, str]:
        consec = getattr(state, "consecutive_complete_signals", 0)
        if consec >= self.required:
            return True, f"task_complete confirmed {consec} consecutive times"
        return False, ""


def register(library):
    library.register(
        type_="verification",
        name=BaselineVerification.NAME,
        factory=lambda params: BaselineVerification(
            required_consecutive=int(params.get("required_consecutive", 2))
        ),
        description=BaselineVerification.DESCRIPTION,
        params_schema=BaselineVerification.PARAMS_SCHEMA,
        niche=BaselineVerification.NICHE,
    )
