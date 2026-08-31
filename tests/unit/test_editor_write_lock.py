"""Write-boundary checks for the module editor."""

import pytest

from harbor.agents.terminus_2_modular.modules.tools.editor_file_tools import (
    _write_allowed,
)

pytestmark = pytest.mark.unit

MODULE_TYPES = (
    "agent_loop",
    "observation",
    "context_mgmt",
    "tools",
    "verification",
)


@pytest.mark.parametrize("module_type", MODULE_TYPES)
@pytest.mark.parametrize("lock", [None, "", "agent_loop", "tools"])
def test_baseline_is_read_only_regardless_of_lock(module_type, lock):
    assert not _write_allowed(f"{module_type}/baseline.py", lock)


@pytest.mark.parametrize("lock", [None, "", "observation", "tools"])
def test_active_bundle_is_read_only(lock):
    assert not _write_allowed("active_bundle.json", lock)


def test_locked_type_accepts_only_its_variants():
    assert _write_allowed("tools/my_variant.py", "tools")
    assert not _write_allowed("observation/my_variant.py", "tools")


@pytest.mark.parametrize("lock", [None, ""])
def test_no_lock_allows_any_module_type(lock):
    for module_type in MODULE_TYPES:
        assert _write_allowed(f"{module_type}/my_variant.py", lock)


def test_tools_lock_also_covers_tool_helpers():
    assert _write_allowed("tool_helper/grep_x.py", "tools")
    assert not _write_allowed("tool_helper/grep_x.py", "observation")


def test_tool_helpers_are_writable_without_a_lock():
    assert _write_allowed("tool_helper/grep_x.py", None)
