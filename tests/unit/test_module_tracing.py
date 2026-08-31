"""Unit tests for kernel module-execution tracing (tracing.py + recorder sink
+ trajectory_analysis rendering) and the smoke discovery-contract check."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from harbor.agents.terminus_2_modular.protocols import (
    LLMResponseParseResult,
    ObsResult,
    ObsState,
    ToolResult,
)
from harbor.agents.terminus_2_modular.services import (
    AtifTrajectoryRecorder,
    build_default_services,
)
from harbor.agents.terminus_2_modular.tracing import TracingProxy, wrap_module
from harbor.agents.terminus_2_modular.self_evo.smoke_tests import (
    check_discovery_contract,
)
from harbor.agents.terminus_2_modular.self_evo.trajectory_analysis import (
    _render_module_trace,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeObservation:
    async def capture(self, prev, ctx):
        return ObsResult(text="x" * 120), ObsState()


class FakeTools:
    def parse_llm_response(self, response):
        return LLMResponseParseResult(
            commands=[], is_task_complete=True, plan="1. do it"
        )

    async def execute(self, call, ctx):
        return ToolResult(success=True, output="ok")

    def extra_duck_typed(self):
        return "quack"


class ExplodingVerification:
    async def should_terminate(self, state, ctx):
        raise AttributeError("no raw_content")


def _services():
    return build_default_services(logging.getLogger("test-tracing"))


# ---------------------------------------------------------------------------
# Proxy behavior
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_proxy_traces_async_capture():
    services = _services()
    obs = wrap_module(FakeObservation(), "observation", "fake", services)
    result, _state = await obs.capture(ObsState(), None)
    assert result.text == "x" * 120
    events = services.trajectory.module_events
    assert len(events) == 1
    assert events[0]["module"] == "observation:fake"
    assert events[0]["call"] == "capture"
    assert "120 chars" in events[0]["summary"]


@pytest.mark.asyncio
async def test_proxy_traces_sync_parse_and_passthrough():
    services = _services()
    tools = wrap_module(FakeTools(), "tools", "fake", services)
    parse = tools.parse_llm_response("<response/>")
    assert parse.is_task_complete
    # duck-typed extras pass through untraced
    assert tools.extra_duck_typed() == "quack"
    assert hasattr(tools, "extra_duck_typed")
    assert not hasattr(tools, "does_not_exist")
    events = services.trajectory.module_events
    assert len(events) == 1
    assert "task_complete=True" in events[0]["summary"]
    assert "plan=yes" in events[0]["summary"]


@pytest.mark.asyncio
async def test_proxy_traces_exception_and_reraises():
    services = _services()
    ver = wrap_module(ExplodingVerification(), "verification", "boom", services)
    with pytest.raises(AttributeError):
        await ver.should_terminate(None, None)
    events = services.trajectory.module_events
    assert len(events) == 1
    assert "⚠ raise AttributeError" in events[0]["summary"]


def test_wrap_module_unknown_type_passthrough():
    inner = FakeTools()
    assert wrap_module(inner, "not_a_type", "x", _services()) is inner


def test_proxy_survives_recorder_without_trace():
    class Bare:
        pass

    class Svc:
        trajectory = Bare()
        logger = None

    proxy = TracingProxy(FakeTools(), "tools", "fake", Svc())
    # emit must not raise even with no trace() and no logger
    parse = proxy.parse_llm_response("x")
    assert parse.is_task_complete


# ---------------------------------------------------------------------------
# Recorder sink + dump
# ---------------------------------------------------------------------------


def test_recorder_trace_dump_and_cap(tmp_path: Path):
    rec = AtifTrajectoryRecorder(
        logs_dir=tmp_path,
        session_id="s",
        agent_name="a",
        agent_version="v",
        model_name="m",
    )
    rec.MAX_MODULE_EVENTS  # exists
    rec.trace("tools:fake", "execute", "→ ok")
    from harbor.models.trajectories import Step

    rec.append_step(
        Step(
            step_id=1,
            timestamp="2026-07-15T00:00:00+00:00",
            source="user",
            message="hi",
        )
    )
    rec.dump()
    import json

    traj = json.loads((tmp_path / "trajectory.json").read_text())
    trace = traj["agent"]["extra"]["module_trace"]
    assert trace == [{"module": "tools:fake", "call": "execute", "summary": "→ ok"}]

    # cap: events past MAX are counted as dropped, not stored
    rec2 = AtifTrajectoryRecorder(
        logs_dir=tmp_path,
        session_id="s2",
        agent_name="a",
        agent_version="v",
        model_name="m",
    )
    rec2.MAX_MODULE_EVENTS = 3  # type: ignore[misc]
    for i in range(5):
        rec2.trace("m", "c", str(i))
    assert len(rec2.module_events) == 3
    assert rec2.module_events_dropped == 2


# ---------------------------------------------------------------------------
# Rendering for the editor
# ---------------------------------------------------------------------------


def test_render_module_trace_aggregates_and_degrades():
    traj = {
        "agent": {
            "extra": {
                "module_trace": [
                    {"module": "tools:t", "call": "execute", "summary": "`ls` → ok"},
                    {"module": "tools:t", "call": "execute", "summary": "`pwd` → ok"},
                    {
                        "module": "verification:v",
                        "call": "should_terminate",
                        "summary": "⚠ raise AttributeError: x",
                    },
                ]
            }
        }
    }
    block = _render_module_trace(traj)
    assert "tools:t.execute ×2" in block
    assert "verification:v.should_terminate ×1" in block
    assert "⚠ raise AttributeError" in block  # notable
    # old trajectories without the field → graceful ""
    assert _render_module_trace({"agent": {"extra": {}}}) == ""
    assert _render_module_trace({}) == ""


# ---------------------------------------------------------------------------
# Smoke: discovery contract
# ---------------------------------------------------------------------------

_GOOD_MODULE = """
DESCRIPTION = "d"


class Impl:
    NAME = "good"


def register(library):
    library.register("{type}", "good", lambda p: Impl(), DESCRIPTION)
"""

_GOOD_HELPER = """
NAME = "helper_cmd"
USAGE = "helper_cmd <arg>"
DESCRIPTION = "does a helpful thing"
NICHE = {"op": "help"}


async def run(args, ctx):
    return "ok"


def register(library):
    from harbor.agents.terminus_2_modular.protocols import SolverHelper

    library.register(
        type_="tool_helper",
        name=NAME,
        factory=lambda params: SolverHelper(name=NAME, usage=USAGE, run=run),
        description=DESCRIPTION,
        niche=NICHE,
    )
"""


def _tree(tmp_path: Path) -> Path:
    root = tmp_path / "modules"
    for t in ("agent_loop", "observation", "context_mgmt", "tools", "verification"):
        d = root / t
        d.mkdir(parents=True)
        (d / "main.py").write_text(_GOOD_MODULE.replace("{type}", t))
    return root


def test_discovery_contract_passes_on_valid_tree(tmp_path: Path):
    root = _tree(tmp_path)
    helpers = root / "tool_helper"
    helpers.mkdir()
    (helpers / "good_helper.py").write_text(_GOOD_HELPER)
    rep = check_discovery_contract(root)
    assert rep.passed, rep.failures


def test_discovery_contract_fails_on_empty_file(tmp_path: Path):
    root = _tree(tmp_path)
    # the 0715_0040 case: a promoted 0-byte verification module
    (root / "verification" / "requirements_gate.py").write_text("")
    rep = check_discovery_contract(root)
    assert not rep.passed
    assert any("EMPTY" in f for f in rep.failures)


def test_discovery_contract_fails_on_missing_register(tmp_path: Path):
    root = _tree(tmp_path)
    (root / "agent_loop" / "orphan.py").write_text("X = 1\n")
    rep = check_discovery_contract(root)
    assert not rep.passed
    assert any("no register()" in f for f in rep.failures)


def test_discovery_contract_fails_on_register_noop(tmp_path: Path):
    root = _tree(tmp_path)
    (root / "observation" / "noop_reg.py").write_text(
        "def register(library):\n    pass\n"
    )
    rep = check_discovery_contract(root)
    assert not rep.passed
    assert any("registered 0 implementations" in f for f in rep.failures)


def test_discovery_contract_fails_on_bad_helper(tmp_path: Path):
    root = _tree(tmp_path)
    helpers = root / "tool_helper"
    helpers.mkdir()
    # registers fine but has no NAME → the tools module can never dispatch to it
    (helpers / "broken.py").write_text(
        _GOOD_HELPER.replace('NAME = "helper_cmd"', "NAME = None")
    )
    rep = check_discovery_contract(root)
    assert not rep.passed
    assert any("helper contract" in f for f in rep.failures)


def test_discovery_contract_fails_on_helper_without_niche(tmp_path: Path):
    """A helper with no niche cannot be deduped, superseded or rolled back — the
    accumulate-forever failure mode the archive exists to prevent."""
    root = _tree(tmp_path)
    helpers = root / "tool_helper"
    helpers.mkdir()
    (helpers / "nicheless.py").write_text(
        _GOOD_HELPER.replace('NICHE = {"op": "help"}', "NICHE = {}")
    )
    rep = check_discovery_contract(root)
    assert not rep.passed
    assert any("no NICHE" in f for f in rep.failures)


def test_discovery_contract_fails_on_helper_in_old_location(tmp_path: Path):
    """`<type>/helpers/` is never discovered (discovery is a shallow glob), so a
    helper left there registers nothing and the solver never sees it."""
    root = _tree(tmp_path)
    legacy = root / "tools" / "helpers"
    legacy.mkdir()
    (legacy / "stray.py").write_text(_GOOD_HELPER)
    rep = check_discovery_contract(root)
    assert not rep.passed
    assert any("OLD helper location" in f for f in rep.failures)


def test_discovery_contract_allows_underscore_shared_code(tmp_path: Path):
    root = _tree(tmp_path)
    (root / "agent_loop" / "_shared.py").write_text("HELPER = 1\n")
    rep = check_discovery_contract(root)
    assert rep.passed, rep.failures


# --- unguarded model-text parsers -------------------------------------------
# A variant that drops the baseline's try/except around a parser of MODEL output
# crashes the run on the first malformed command. Invisible to every other gate:
# it imports cleanly, the design is sound, and whether the 4 fixed sanity tasks
# trip it is luck. Observed in a promoted `tools` variant (ValueError: No closing
# quotation out of the agent loop).

_BARE = """
import shlex


def parse(cmd):
    return shlex.split(cmd)
"""

_GUARDED = """
import shlex


def parse(cmd):
    try:
        return shlex.split(cmd)
    except ValueError:
        return None
"""


def test_bare_text_parser_is_flagged(tmp_path: Path):
    from harbor.agents.terminus_2_modular.self_evo.smoke_tests import (
        _unguarded_text_parsers,
    )

    f = tmp_path / "v.py"
    f.write_text(_BARE)
    assert [c for _ln, c in _unguarded_text_parsers(f)] == ["shlex.split"]


def test_guarded_text_parser_is_accepted(tmp_path: Path):
    from harbor.agents.terminus_2_modular.self_evo.smoke_tests import (
        _unguarded_text_parsers,
    )

    f = tmp_path / "v.py"
    f.write_text(_GUARDED)
    assert _unguarded_text_parsers(f) == []


def test_shlex_quote_is_not_a_parser(tmp_path: Path):
    """`quote` escapes, it does not parse — flagging it would be noise."""
    from harbor.agents.terminus_2_modular.self_evo.smoke_tests import (
        _unguarded_text_parsers,
    )

    f = tmp_path / "v.py"
    f.write_text("import shlex\n\n\ndef q(p):\n    return shlex.quote(p)\n")
    assert _unguarded_text_parsers(f) == []


def test_static_contract_fails_the_staging_tree(tmp_path: Path):
    from harbor.agents.terminus_2_modular.self_evo.smoke_tests import (
        check_static_contract,
    )

    root = _tree(tmp_path)
    (root / "tools" / "risky.py").write_text(_BARE)
    rep = check_static_contract(root)
    assert not rep.passed
    assert any("bare `shlex.split`" in f for f in rep.failures)
