"""Off-limits path checks for the editor's file tools.

The gatekeeper (`editor_file_tools.py`) enforces the Kernel off-limits list and
the answer-key blocks, yet it lives inside the editable modules/ tree — it must
itself be off-limits, or evolution could rewrite its own guardrails (design
doc §7.2 / §10 Step 0).
"""

import pytest

from harbor.agents.terminus_2_modular.modules.tools.editor_file_tools import (
    _is_off_limits,
)


@pytest.mark.unit
class TestIsOffLimits:
    def test_kernel_files_blocked(self):
        for rel in [
            "protocols.py",
            "services.py",
            "agent.py",
            "library.py",
            "__init__.py",
            "composer",
            "composer/static.py",
        ]:
            assert _is_off_limits(rel), rel

    def test_gatekeeper_itself_blocked(self):
        assert _is_off_limits("tools/editor_file_tools.py")
        # Path normalization must not open a bypass.
        assert _is_off_limits("./tools/editor_file_tools.py")
        assert _is_off_limits("tools//editor_file_tools.py")

    def test_ordinary_module_files_editable(self):
        for rel in [
            "tools/tmux_full.py",
            "agent_loop/react.py",
            "observation/truncate.py",
            "verification/two_phase.py",
            "tools/helpers/some_helper.py",
        ]:
            assert not _is_off_limits(rel), rel

    def test_empty_path_blocked(self):
        assert _is_off_limits("")
