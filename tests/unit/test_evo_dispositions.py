"""M3: what happened to a finding that never became a proposal.

The backlog answers "which findings are attached to which proposal". Everything
else it forgets. A finding routed `out_of_scope`, absorbed by a successor, aimed
at a dead lineage, or refused as malformed leaves one audit line and no state —
so the resume check, which asks only "is this finding attached to a proposal?",
says no and re-routes it. Every window. Forever. That is a routing call per
dead finding per window, buying a verdict that was already reached.

`out_of_scope` loses more than cost: the router is required to name the module
that DOES own the problem, and that name was dropped on the floor. The one
finding that identifies work for another module was the one piece of evidence
this run could hand to a different lineage, and it went nowhere.

So terminal outcomes get written down, and the retryable ones deliberately do
not. `comparison_unresolved` and a truncated reply are endpoint weather, not
verdicts: recording them as terminal would silently discard real evidence on a
bad afternoon, which is a far worse failure than paying for a retry.
"""

import pytest

from harbor.agents.terminus_2_modular.self_evo import dispositions

pytestmark = pytest.mark.unit


def test_a_finding_with_no_disposition_is_not_terminal(tmp_path):
    assert dispositions.terminal_finding_ids(tmp_path) == set()


def test_a_recorded_disposition_is_read_back(tmp_path):
    dispositions.record(
        tmp_path, finding_id="f_a", kind=dispositions.OUT_OF_SCOPE, module="tools"
    )
    (row,) = dispositions.load(tmp_path)
    assert row.finding_id == "f_a"
    assert row.kind == dispositions.OUT_OF_SCOPE
    assert row.module == "tools"


def test_terminal_kinds_stop_a_finding_from_being_routed_again(tmp_path):
    for i, kind in enumerate(
        (
            dispositions.OUT_OF_SCOPE,
            dispositions.COVERED,
            dispositions.STALE,
            dispositions.INVALID,
        )
    ):
        dispositions.record(tmp_path, finding_id=f"f_{i}", kind=kind)
    assert dispositions.terminal_finding_ids(tmp_path) == {"f_0", "f_1", "f_2", "f_3"}


def test_a_retryable_disposition_is_recorded_but_not_terminal(tmp_path):
    # the endpoint failing to answer is not a verdict about the finding
    dispositions.record(
        tmp_path,
        finding_id="f_a",
        kind=dispositions.RETRYABLE,
        reason="comparison failed",
    )
    assert dispositions.load(tmp_path)[0].kind == dispositions.RETRYABLE
    assert dispositions.terminal_finding_ids(tmp_path) == set()


def test_the_ledger_is_append_only_and_survives_a_reopen(tmp_path):
    dispositions.record(tmp_path, finding_id="f_a", kind=dispositions.STALE)
    dispositions.record(tmp_path, finding_id="f_b", kind=dispositions.INVALID)
    assert [r.finding_id for r in dispositions.load(tmp_path)] == ["f_a", "f_b"]


def test_a_corrupt_line_does_not_take_the_ledger_down(tmp_path):
    # this file is read on every resume; one bad append must not cost a run
    dispositions.record(tmp_path, finding_id="f_a", kind=dispositions.STALE)
    path = dispositions.dispositions_path(tmp_path)
    path.write_text(path.read_text() + "{not json\n")
    assert [r.finding_id for r in dispositions.load(tmp_path)] == ["f_a"]


# ---- the cross-module backlog --------------------------------------------


def test_out_of_scope_findings_are_listed_under_the_module_that_owns_them(tmp_path):
    dispositions.record(
        tmp_path, finding_id="f_a", kind=dispositions.OUT_OF_SCOPE, module="tools"
    )
    dispositions.record(
        tmp_path,
        finding_id="f_b",
        kind=dispositions.OUT_OF_SCOPE,
        module="observation",
    )
    assert dispositions.out_of_scope_for(tmp_path, "tools") == ["f_a"]
    assert dispositions.out_of_scope_for(tmp_path, "observation") == ["f_b"]


def test_the_cross_module_backlog_is_a_file_another_run_can_point_at(tmp_path):
    # a lineage locks ONE module, so the module that owns this problem is a
    # different run. Handing it over means leaving something on disk with a
    # predictable path, not a dict that dies with the process.
    dispositions.record(
        tmp_path, finding_id="f_a", kind=dispositions.OUT_OF_SCOPE, module="tools"
    )
    assert dispositions.module_backlog_path(tmp_path, "tools").exists()


def test_recording_the_same_finding_twice_does_not_double_count(tmp_path):
    # resume re-ingests the same step; the ledger must not grow a duplicate
    for _ in range(2):
        dispositions.record(
            tmp_path, finding_id="f_a", kind=dispositions.OUT_OF_SCOPE, module="tools"
        )
    assert dispositions.out_of_scope_for(tmp_path, "tools") == ["f_a"]
