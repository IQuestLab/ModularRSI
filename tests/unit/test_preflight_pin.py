"""`online_evo._preflight_pin` — the serial pin's only loud failure path.

The pin's reader is fail-open by design: one warning, then degrade to
`DEFAULT_BUNDLE`. That is right for a single roll (a composer must always return
a usable bundle) and is the worst possible behavior for a RUN — every roll would
quietly use `baseline`, the lineage would stop being serial, and the results
would look completely normal. Preflight is what converts that into an abort, so
each rejection below is a bug that would otherwise cost a multi-day run.
"""

import json

import pytest

from harbor.agents.terminus_2_modular import archive as arch
from harbor.agents.terminus_2_modular.library import build_default_library
from harbor.agents.terminus_2_modular.self_evo.online_evo import _preflight_pin


@pytest.fixture
def run_dir(tmp_path):
    """A minimal lineage: gen_0 = the installed package's modules, seeded archive."""
    import shutil
    from pathlib import Path

    from harbor.agents.terminus_2_modular import modules as pkg

    mods = tmp_path / "gen_0" / "modules"
    mods.parent.mkdir(parents=True)
    shutil.copytree(
        Path(pkg.__file__).parent,
        mods,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".*"),
    )
    arch.save_archive(
        tmp_path,
        arch.seed_from_library(
            build_default_library(modules_root=mods), born_gen="gen_0"
        ),
    )
    return tmp_path


def _pin(run_dir, obj):
    (run_dir / "gen_0" / "modules" / "active_bundle.json").write_text(json.dumps(obj))


@pytest.mark.unit
class TestPreflightPin:
    def test_no_pin_file_is_fine(self, run_dir):
        """An ordinary (non-serial) lineage must not be disturbed."""
        _preflight_pin(run_dir, run_dir / "gen_0", "observation")

    def test_valid_pin_passes(self, run_dir):
        _pin(run_dir, {"tools": {"name": "tmux_xml"}})
        _preflight_pin(run_dir, run_dir / "gen_0", "observation")

    def test_unknown_impl_aborts(self, run_dir):
        """Typo in the variant name. Without preflight this is the silent killer:
        the reader voids the whole file and every roll runs `baseline`."""
        _pin(run_dir, {"agent_loop": {"name": "no_such_variant"}})
        with pytest.raises(SystemExit, match="yielded no valid pin"):
            _preflight_pin(run_dir, run_dir / "gen_0", "observation")

    def test_unparseable_aborts(self, run_dir):
        (run_dir / "gen_0" / "modules" / "active_bundle.json").write_text("{not json")
        with pytest.raises(SystemExit, match="yielded no valid pin"):
            _preflight_pin(run_dir, run_dir / "gen_0", "observation")

    def test_pinning_the_locked_type_aborts(self, run_dir):
        """Freezing the one type the ablation exists to vary — the run would
        produce generations that cannot differ from their parent."""
        _pin(run_dir, {"observation": {"name": "baseline"}})
        with pytest.raises(SystemExit, match="locked-module"):
            _preflight_pin(run_dir, run_dir / "gen_0", "observation")

    def test_pinning_a_retired_variant_aborts(self, run_dir):
        """`_archive_skip` would drop it from the catalog while `base` still names
        it — a foundation that contradicts itself."""
        _pin(run_dir, {"tools": {"name": "tmux_xml"}})
        arch.set_status(run_dir, "tools/tmux_xml", "superseded")
        with pytest.raises(SystemExit, match="retired"):
            _preflight_pin(run_dir, run_dir / "gen_0", "observation")
