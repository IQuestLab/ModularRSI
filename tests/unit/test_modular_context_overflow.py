import logging
from types import SimpleNamespace

import pytest

from harbor.agents.terminus_2_modular.modules.agent_loop.baseline import (
    BaselineAgentLoop,
)
from harbor.agents.terminus_2_modular.modules.context_mgmt.baseline import (
    BaselineContextMgmt,
)
from harbor.agents.terminus_2_modular.protocols import CompressResult
from harbor.agents.terminus_2_modular.protocols import LLMResponseParseResult
from harbor.llms.base import ContextLengthExceededError, LLMResponse


class _OverflowChat:
    def __init__(self):
        self.calls = 0

    async def chat(self, **_kwargs):
        self.calls += 1
        raise ContextLengthExceededError("full")


class _DisabledContext:
    async def force_summarize(self, chat, _instruction, _ctx):
        return CompressResult(chat=chat)


class _RecoveredOverflowChat:
    def __init__(self):
        self.calls = 0
        self._messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "stale"},
            {"role": "assistant", "content": "stale"},
        ]

    @property
    def messages(self):
        return self._messages

    def reset_response_chain(self):
        pass

    async def chat(self, **_kwargs):
        self.calls += 1
        raise ContextLengthExceededError("still full")


class _RecoveredContext:
    async def force_summarize(self, chat, _instruction, _ctx):
        chat._messages = chat.messages[:1]
        chat.reset_response_chain()
        return CompressResult(
            chat=chat,
            handoff_prompt="minimal handoff",
            summarization_occurred=True,
        )


class _ProtocolTools:
    def format_initial_prompt(self, instruction, terminal_state, _ctx):
        return f"TASK={instruction}\nTERMINAL={terminal_state}\nJSON CONTRACT"


class _FailingSummaryModel:
    def get_model_context_limit(self):
        return 10_000

    def count_message_tokens(self, messages):
        return len(messages) * 1_000

    async def call(self, **_kwargs):
        raise RuntimeError("summary unavailable")


class _HistoryChat:
    def __init__(self):
        self._model = _FailingSummaryModel()
        self._messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "old-1"},
            {"role": "assistant", "content": "old-1"},
            {"role": "user", "content": "old-2"},
            {"role": "assistant", "content": "old-2"},
        ]
        self.reset_calls = 0

    @property
    def messages(self):
        return self._messages

    def reset_response_chain(self):
        self.reset_calls += 1


class _MalformedChat:
    total_input_tokens = 0
    total_output_tokens = 0
    total_cache_tokens = 0
    total_cost = 0.0

    def __init__(self):
        self.calls = 0

    async def chat(self, **_kwargs):
        self.calls += 1
        return LLMResponse(content=f"malformed response {self.calls}")


class _NoCompression:
    async def maybe_compress(self, chat, _instruction, _ctx):
        return CompressResult(chat=chat)


class _AlwaysMalformedTools:
    _parser_name = "JSON"

    def parse_llm_response(self, _response):
        return LLMResponseParseResult(
            commands=[],
            is_task_complete=False,
            error="invalid JSON",
        )


class _Recorder:
    def __init__(self):
        self.steps = []
        self.dumps = 0
        self.summarization_count = 0

    def append_step(self, step):
        self.steps.append(step)

    def record_asciinema_marker(self, _marker):
        pass

    def dump(self):
        self.dumps += 1


class _Failures:
    def __init__(self):
        self.tags = []

    def raise_tag(self, *tag):
        self.tags.append(tag)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_disabled_summarization_reraises_overflow_without_retrying():
    """Stock Terminus-2 raises immediately when enable_summarize=False."""
    chat = _OverflowChat()
    ctx = SimpleNamespace(services=SimpleNamespace(logger=logging.getLogger(__name__)))

    with pytest.raises(ContextLengthExceededError, match="full"):
        await BaselineAgentLoop()._query_llm(
            chat=chat,
            prompt="continue",
            ctx=ctx,
            context_mgmt=_DisabledContext(),
            tools=object(),
            original_instruction="do the task",
            logging_paths=(None, None, None),
        )

    assert chat.calls == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_failed_fallback_chat_reraises_instead_of_returning_synthetic_text():
    chat = _RecoveredOverflowChat()
    ctx = SimpleNamespace(services=SimpleNamespace(logger=logging.getLogger(__name__)))

    with pytest.raises(ContextLengthExceededError, match="still full"):
        await BaselineAgentLoop()._query_llm(
            chat=chat,
            prompt="continue",
            ctx=ctx,
            context_mgmt=_RecoveredContext(),
            tools=_ProtocolTools(),
            original_instruction="do the task",
            logging_paths=(None, None, None),
        )

    assert chat.calls == 2
    assert chat.messages == [{"role": "system", "content": "system"}]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_tier3_context_fallback_discards_over_limit_history():
    chat = _HistoryChat()
    ctx = SimpleNamespace(
        shared=SimpleNamespace(tmux_session=None),
        services=SimpleNamespace(logger=logging.getLogger(__name__)),
    )

    result = await BaselineContextMgmt().force_summarize(chat, "do the task", ctx)

    assert result.summarization_occurred is True
    assert result.handoff_prompt == "do the task\n\nCurrent state: "
    assert chat.messages == [{"role": "system", "content": "system"}]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_second_consecutive_parse_error_ends_and_records_both_attempts():
    chat = _MalformedChat()
    recorder = _Recorder()
    failures = _Failures()
    ctx = SimpleNamespace(
        shared=SimpleNamespace(tmux_session=None),
        state=SimpleNamespace(logs_dir=None, model_name="test-model"),
        services=SimpleNamespace(
            logger=logging.getLogger(__name__),
            trajectory=recorder,
            failures=failures,
        ),
    )

    result = await BaselineAgentLoop(max_iterations=10).run(
        initial_prompt="start",
        original_instruction="do the task",
        observation=object(),
        context_mgmt=_NoCompression(),
        tools=_AlwaysMalformedTools(),
        verification=object(),
        chat=chat,
        ctx=ctx,
    )

    assert result.success is False
    assert result.failure_tag == "parse_error"
    assert chat.calls == 2
    assert len(recorder.steps) == 2
    assert recorder.dumps == 1
    assert [tag[0] for tag in failures.tags] == ["parse_error", "parse_error"]
