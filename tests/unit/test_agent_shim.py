"""Agent shim to installed orchestration boundary."""

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from harbor.agents.terminus_2_modular.agent import (
    Terminus2Modular,
    Terminus2ModularEditor,
)
from harbor.agents.terminus_2_modular.kernel import orchestration
from harbor.agents.terminus_2_modular.protocols import ModuleBundle, ModuleSpec


def _agent(tmp_path, **over):
    kw = dict(
        logs_dir=tmp_path / "logs",
        model_name="openai/test-model",
        max_turns=7,
        suppress_max_turns_warning=True,
        parser_name="xml",
        temperature=0.3,
        api_key="sk-test",
        staging_dir=str(tmp_path / "stg"),
        composer_name="static",
        composer_timeout=600,
        composer_llm_kwargs={"chat_template_kwargs": {"enable_thinking": False}},
        composer_cache_dir=str(tmp_path / "composer-cache"),
        composer_cache_mode="write",
        composer_only=True,
    )
    kw.update(over)
    return Terminus2Modular(**kw)


@pytest.mark.unit
class TestShimBoundary:
    async def test_run_delegates_to_installed_orchestration(self, tmp_path):
        agent = _agent(tmp_path)
        with patch.object(orchestration, "run_task", new=AsyncMock()) as rt:
            await agent.run("do things", environment=None, context=object())
        assert rt.await_count == 1
        kw = rt.await_args.kwargs
        assert kw["instruction"] == "do things"
        p = kw["params"]
        # the primitives the orchestration consumes, packed faithfully
        assert p["model_name"] == "openai/test-model"
        assert p["max_turns"] == 7
        assert p["parser_name"] == "xml"
        assert p["temperature"] == 0.3
        assert p["api_key"] == "sk-test"
        assert p["composer_name"] == "static"
        assert p["composer_timeout"] == 600
        assert p["composer_llm_kwargs"] == {
            "chat_template_kwargs": {"enable_thinking": False}
        }
        assert p["composer_cache_dir"] == tmp_path / "composer-cache"
        assert p["composer_cache_mode"] == "write"
        assert p["composer_only"] is True
        assert p["staging_dir"] == Path(str(tmp_path / "stg"))
        assert p["agent_name"] == "terminus-2-modular"
        assert p["enable_summarize"] is True
        assert p["tmux_pane_width"] == 160 and p["tmux_pane_height"] == 40

    def test_composer_cache_validation(self, tmp_path):
        with pytest.raises(ValueError, match="composer_cache_mode"):
            _agent(tmp_path, composer_cache_mode="invalid", composer_only=False)
        with pytest.raises(ValueError, match="composer_cache_dir"):
            _agent(
                tmp_path,
                composer_cache_mode="read",
                composer_cache_dir=None,
                composer_only=False,
            )

    def test_editor_defaults_preserved(self, tmp_path):
        ed = Terminus2ModularEditor(
            logs_dir=tmp_path / "logs", model_name="openai/test-model"
        )
        assert ed._composer_name == "editor_static"
        assert ed._enable_summarize is False


@pytest.mark.unit
def test_composer_cache_round_trip(tmp_path):
    bundle = ModuleBundle(
        agent_loop=ModuleSpec(name="planning"),
        observation=ModuleSpec(name="scrollback"),
        context_mgmt=ModuleSpec(name="passthrough"),
        tools=ModuleSpec(name="robust", params={"helpers": ["read_file"]}),
        verification=ModuleSpec(name="margin_gated"),
    )
    cache_dir = tmp_path / "cache"
    written = orchestration._write_composer_cache(
        cache_dir, "do the task", bundle, "session-1"
    )
    restored, read_path = orchestration._read_composer_cache(cache_dir, "do the task")
    assert read_path == written
    assert restored == bundle
    assert list(cache_dir.glob("*.json")) == [written]
