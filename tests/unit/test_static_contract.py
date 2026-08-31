"""check_static_contract — the L3.7 static landmine audit.

Reproduces the two 2026-07-16 stuck_resistant landmines that import-level
smoke and a non-selecting sanity battery both missed:

  a) function-local `from X import Y` placed AFTER the first use of Y
     (UnboundLocalError on every call; imports cleanly) — ruff F821/F823
  b) call to a self.<method> that exists nowhere on the instance
     (hallucinated-by-analogy helper) — AST self-attr audit
"""

from pathlib import Path

import pytest

from harbor.agents.terminus_2_modular.self_evo.smoke_tests import (
    check_static_contract,
)

pytestmark = pytest.mark.unit

_GOOD_MODULE = """
DESCRIPTION = "d"


class Impl:
    NAME = "good"

    def __init__(self):
        self.calls = 0

    def use(self):
        self.calls += 1
        return self.calls


def register(library):
    library.register("{type}", "good", lambda p: Impl(), DESCRIPTION)
"""


def _tree(tmp_path: Path) -> Path:
    root = tmp_path / "modules"
    for t in ("agent_loop", "observation", "context_mgmt", "tools", "verification"):
        d = root / t
        d.mkdir(parents=True)
        (d / "main.py").write_text(_GOOD_MODULE.replace("{type}", t))
    return root


def test_static_contract_passes_on_clean_tree(tmp_path: Path):
    rep = check_static_contract(_tree(tmp_path))
    assert rep.passed, rep.failures


def test_static_contract_catches_use_before_local_import(tmp_path: Path):
    root = _tree(tmp_path)
    # landmine (a): Path used at the top of run(), imported mid-function —
    # the local import makes Path function-scoped ⇒ UnboundLocalError.
    (root / "agent_loop" / "boom.py").write_text(
        'DESCRIPTION = "d"\n\n'
        "class Boom:\n"
        '    NAME = "boom"\n\n'
        "    def run(self, logs_dir):\n"
        "        p = Path(logs_dir) if logs_dir else None\n"
        "        from pathlib import Path\n"
        "        return p\n\n"
        "def register(library):\n"
        '    library.register("agent_loop", "boom", lambda p: Boom(), DESCRIPTION)\n'
    )
    rep = check_static_contract(root)
    assert not rep.passed
    assert any("F82" in f for f in rep.failures), rep.failures


def test_static_contract_catches_missing_self_method(tmp_path: Path):
    root = _tree(tmp_path)
    # landmine (b): hallucinated helper — exists nowhere, never assigned.
    (root / "observation" / "ghost.py").write_text(
        'DESCRIPTION = "d"\n\n'
        "class Ghost:\n"
        '    NAME = "ghost"\n\n'
        "    def capture(self, prev, ctx):\n"
        "        return self._append_summarization_step(prev)\n\n"
        "def register(library):\n"
        '    library.register("observation", "ghost", lambda p: Ghost(), DESCRIPTION)\n'
    )
    rep = check_static_contract(root)
    assert not rep.passed
    assert any("_append_summarization_step" in f for f in rep.failures), rep.failures


def test_static_contract_allows_runtime_assigned_attrs(tmp_path: Path):
    root = _tree(tmp_path)
    # attrs assigned anywhere in the file (init/run) must NOT be flagged,
    # nor attrs inherited from a parent instance.
    (root / "tools" / "ok_attrs.py").write_text(
        'DESCRIPTION = "d"\n\n'
        "class OkAttrs:\n"
        '    NAME = "ok_attrs"\n\n'
        "    def run(self):\n"
        "        self._n = 0\n"
        "        self._n += 1\n"
        "        return self._n\n\n"
        "def register(library):\n"
        '    library.register("tools", "ok_attrs", lambda p: OkAttrs(), DESCRIPTION)\n'
    )
    rep = check_static_contract(root)
    assert rep.passed, rep.failures
