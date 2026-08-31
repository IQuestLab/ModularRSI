"""The SERIAL pin: `<modules_root>/active_bundle.json` freezes a module type.

Why this needs tests at all — the failure mode is silent by construction. The
pin's reader is fail-open (a bad file logs one warning and degrades to
`DEFAULT_BUNDLE`), and a de-pinned run does not crash: it just quietly runs
`baseline` for the inherited type on every roll, so a "serial" lineage becomes an
ordinary one and nothing in the results says so. The two properties below are the
whole contract:

1. a pinned type comes back in the bundle as the PINNED variant, and
2. a pinned type is never offered to the per-task picker — otherwise the LLM
   could hand the foundation back to `baseline` task by task.
"""

import json

import pytest

from harbor.agents.terminus_2_modular.composer.llm_dynamic import LLMComposer
from harbor.agents.terminus_2_modular.protocols import ModuleInfo

_TYPES = ("agent_loop", "observation", "context_mgmt", "tools", "verification")


def _lib(**extra: list[str]) -> list[ModuleInfo]:
    """A library with `baseline` for all five types, plus named extra variants.

    Niches are left empty on purpose: the composer does not dedup undeclared
    niches, so every variant here stays visible and the test isolates the pin.
    """
    infos = [
        ModuleInfo(
            type=t,
            name="baseline",
            description=f"{t} baseline",
            params_schema={},
            niche={},
        )
        for t in _TYPES
    ]
    for t, names in extra.items():
        infos += [
            ModuleInfo(
                type=t,
                name=n,
                description=f"{t} variant {n}",
                params_schema={},
                niche={},
            )
            for n in names
        ]
    return infos


class _FakeCtx:
    """Minimal stand-in for ModuleCtx — the composer only touches
    `ctx.state.{modules_root,locked_module_type,composer_scope}`."""

    class _State:
        def __init__(self, modules_root, locked, scope):
            self.modules_root = modules_root
            self.locked_module_type = locked
            self.composer_scope = scope

    def __init__(self, modules_root=None, locked=None, scope="locked"):
        self.state = self._State(modules_root, locked, scope)


def _write_pin(root, pin: dict) -> None:
    (root / "active_bundle.json").write_text(json.dumps(pin))


@pytest.fixture
def composer():
    return LLMComposer(model_name="x/y", api_base=None, api_key=None)


@pytest.mark.unit
@pytest.mark.asyncio
class TestPinApplies:
    async def test_pinned_type_wins_over_library_default(self, tmp_path, composer):
        _write_pin(tmp_path, {"agent_loop": {"name": "planning_with_guard"}})
        lib = _lib(agent_loop=["planning_with_guard"])
        bundle = await composer.choose("do a thing", lib, None, _FakeCtx(tmp_path))
        assert bundle.agent_loop.name == "planning_with_guard"
        # Everything else still falls back to the library default.
        assert bundle.observation.name == "baseline"

    async def test_pinned_type_is_not_offered_to_the_picker(self, tmp_path, composer):
        """The pinned type must not reach `_llm_pick`. Here the ONLY multi-impl
        type is the pinned one, so a correct implementation makes no LLM call at
        all — and `_llm_pick` would blow up if it did (no endpoint)."""
        _write_pin(tmp_path, {"agent_loop": {"name": "planning_with_guard"}})
        lib = _lib(agent_loop=["planning_with_guard", "parse_error_recovery"])

        called = False

        async def _boom(*a, **k):
            nonlocal called
            called = True
            raise AssertionError("pinned type was sent to the per-task picker")

        composer._llm_pick = _boom
        bundle = await composer.choose("do a thing", lib, None, _FakeCtx(tmp_path))
        assert not called
        assert bundle.agent_loop.name == "planning_with_guard"

    async def test_no_pin_file_leaves_behavior_unchanged(self, tmp_path, composer):
        bundle = await composer.choose("do a thing", _lib(), None, _FakeCtx(tmp_path))
        assert all(getattr(bundle, t).name == "baseline" for t in _TYPES)


@pytest.mark.unit
@pytest.mark.asyncio
class TestPinRefusesToFreezeTheExperiment:
    async def test_pin_on_the_locked_type_is_dropped(self, tmp_path, composer):
        """Pinning the type the ablation is supposed to VARY would freeze the run
        into a no-op. The lock wins and the pin is discarded (loudly). `online_evo`
        rejects this config up front; this is the last-line behavior."""
        _write_pin(tmp_path, {"observation": {"name": "scrollback"}})
        lib = _lib(observation=["scrollback"])
        ctx = _FakeCtx(tmp_path, locked="observation")

        picked = {}

        async def _pick(instruction, multi, base, helper_infos=None):
            picked.update(multi)
            return None, set()

        composer._llm_pick = _pick
        await composer.choose("do a thing", lib, None, ctx)
        # observation stayed pickable despite being named in the pin file.
        assert "observation" in picked


@pytest.mark.unit
@pytest.mark.asyncio
class TestPinSurvivesFailure:
    async def test_pin_held_when_the_composer_falls_over(self, tmp_path, composer):
        """`choose` swallows everything and returns a usable bundle. That fallback
        must NOT drop the pin — the foundation is a property of the tree, not a
        per-task choice, so degrading to `DEFAULT_BUNDLE` here would swap the whole
        agent out from under the lineage."""
        _write_pin(tmp_path, {"agent_loop": {"name": "planning_with_guard"}})
        lib = _lib(agent_loop=["planning_with_guard"])

        async def _explode(*a, **k):
            raise RuntimeError("niche module blew up")

        composer._choose = _explode
        bundle = await composer.choose("do a thing", lib, None, _FakeCtx(tmp_path))
        assert bundle.agent_loop.name == "planning_with_guard"


@pytest.mark.unit
@pytest.mark.asyncio
class TestInvalidPinDoesNotPartiallyApply:
    async def test_unknown_impl_name_voids_the_whole_file(self, tmp_path, composer):
        """`load_bundle_overrides` validates against the library and returns None on
        any problem — never a half-applied config. At run level `online_evo`'s
        preflight turns this into a hard abort rather than a silent de-pin."""
        _write_pin(tmp_path, {"agent_loop": {"name": "does_not_exist"}})
        bundle = await composer.choose("do a thing", _lib(), None, _FakeCtx(tmp_path))
        assert bundle.agent_loop.name == "baseline"


@pytest.mark.unit
@pytest.mark.asyncio
class TestComposerScope:
    """`composer_scope` separates WHAT MAY BE WRITTEN from WHAT MAY BE PICKED.

    A SERIAL lineage starts from another lineage's tree. Under the default
    "locked" scope the composer puts every non-locked type back on its `baseline`
    default, so everything the donor lineage produced sits in the library
    unreachable — the inheritance is decorative and the run is indistinguishable
    from an ordinary parallel single-lock one, with nothing in the logs saying so.
    """

    @staticmethod
    def _spy(composer):
        seen: dict[str, list[str]] = {}

        async def _pick(instruction, multi, base, helper_infos=None):
            seen.clear()
            seen.update({t: sorted(i.name for i in lst) for t, lst in multi.items()})
            return None, set()

        composer._llm_pick = _pick
        return seen

    async def test_locked_scope_hides_other_types(self, tmp_path, composer):
        """Today's behavior, unchanged: parallel single-lock runs must not move."""
        seen = self._spy(composer)
        lib = _lib(agent_loop=["planning_with_guard"], observation=["scrollback"])
        await composer.choose(
            "t", lib, None, _FakeCtx(tmp_path, locked="observation", scope="locked")
        )
        assert set(seen) == {"observation"}

    async def test_all_scope_offers_every_multi_type(self, tmp_path, composer):
        seen = self._spy(composer)
        lib = _lib(agent_loop=["planning_with_guard"], observation=["scrollback"])
        await composer.choose(
            "t", lib, None, _FakeCtx(tmp_path, locked="observation", scope="all")
        )
        assert set(seen) == {"agent_loop", "observation"}
        assert seen["agent_loop"] == ["baseline", "planning_with_guard"]

    async def test_future_context_variant_remains_composable(self, tmp_path, composer):
        """Gen0 has one native context implementation, but future evolved
        implementations remain a normal composer dimension."""
        seen = self._spy(composer)
        lib = _lib(context_mgmt=["hierarchical_summary"])
        await composer.choose(
            "t", lib, None, _FakeCtx(tmp_path, locked="agent_loop", scope="all")
        )
        assert seen["context_mgmt"] == ["baseline", "hierarchical_summary"]

    async def test_native_passthrough_adapter_is_not_a_solver_choice(
        self, tmp_path, composer
    ):
        seen = self._spy(composer)
        bundle = await composer.choose(
            "t",
            _lib(context_mgmt=["passthrough"]),
            None,
            _FakeCtx(tmp_path, locked="agent_loop", scope="all"),
        )
        assert "context_mgmt" not in seen
        assert bundle.context_mgmt.name == "baseline"

    async def test_quarantined_tool_is_hidden_but_future_tool_remains_composable(
        self, tmp_path, composer
    ):
        seen = self._spy(composer)
        await composer.choose(
            "t",
            _lib(tools=["combined_robust", "future_tool"]),
            None,
            _FakeCtx(tmp_path, locked="agent_loop", scope="all"),
        )
        assert seen["tools"] == ["baseline", "future_tool"]

    async def test_all_scope_still_honors_the_usual_filters(self, tmp_path, composer):
        """ "Free selection" means the LOCK stops narrowing it — not that the
        archive stops applying. A retired variant must stay unselectable."""
        import json as _json

        (tmp_path / "archive.json").write_text(
            _json.dumps(
                [
                    {
                        "name": "planning_with_guard",
                        "type": "agent_loop",
                        "status": "superseded",
                        "born_gen": "gen_0",
                    }
                ]
            )
        )
        mods = tmp_path / "gen_0" / "modules"
        mods.mkdir(parents=True)
        seen = self._spy(composer)
        lib = _lib(agent_loop=["planning_with_guard", "parse_error_recovery"])
        await composer.choose(
            "t", lib, None, _FakeCtx(mods, locked="observation", scope="all")
        )
        assert seen["agent_loop"] == ["baseline", "parse_error_recovery"]

    async def test_unset_scope_defaults_to_locked(self, tmp_path, composer):
        """A ctx with no `composer_scope` at all (older callers) keeps the old
        behavior — the flag fails open to today's semantics, never to 'all'."""
        seen = self._spy(composer)

        class _Old:
            class state:
                modules_root = tmp_path
                locked_module_type = "observation"

        lib = _lib(agent_loop=["planning_with_guard"], observation=["scrollback"])
        await composer.choose("t", lib, None, _Old())
        assert set(seen) == {"observation"}
