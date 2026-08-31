"""P1-1: two-level manifest — the proposal, then what was actually built for it.

A proposal exists before any file does, so it needs its own record (that is the
ProposalRecord). What was built for it is a separate fact, recorded after the
editor runs: which files, which variant_meta, which incumbents it retires.

The reason these are two levels and not one is the drop bug. When a change is
partially rejected, the files can be removed while `variant_meta_text` survives
in the reflection outcome — and `_sync_archive_after_promote` then reads a
SUPERSEDES line belonging to a dropped proposal and retires its parent anyway.
The variant is gone; its predecessor is retired regardless; the archive now has a
hole where a working incumbent used to be. Making the drop atomic over all four
(files, variant_meta, supersede targets, memory record) is what closes it.
"""

import pytest

from harbor.agents.terminus_2_modular.self_evo import manifest, proposals

pytestmark = pytest.mark.unit


def _proposal(tmp_path):
    return proposals.create_proposal(
        tmp_path,
        step="gen_2_c",
        lane="incumbent",
        action="modify",
        target_variant="agent_loop/confirm_exit",
        finding_ids=["f_a"],
    )


def _record(tmp_path, pid, **over):
    kw = dict(
        proposal_id=pid,
        step="gen_2_c",
        files=["agent_loop/confirm_exit.py"],
        variant_meta_text='<variant_meta name="x" type="agent_loop">\nSUPERSEDES: confirm_exit\n</variant_meta>',
        supersede_targets=["agent_loop/confirm_exit"],
        staging_tree_hash="abc123",
    )
    kw.update(over)
    return manifest.record_implementation(tmp_path, **kw)


def test_implementation_is_recorded_against_its_proposal(tmp_path):
    p = _proposal(tmp_path)
    got = _record(tmp_path, p.proposal_id)
    assert got.proposal_id == p.proposal_id
    assert got.files == ["agent_loop/confirm_exit.py"]
    assert got.supersede_targets == ["agent_loop/confirm_exit"]


def test_manifest_is_readable_back(tmp_path):
    p = _proposal(tmp_path)
    _record(tmp_path, p.proposal_id)
    assert (
        manifest.get_implementation(tmp_path, p.proposal_id).staging_tree_hash
        == "abc123"
    )


def test_recording_twice_keeps_the_latest(tmp_path):
    p = _proposal(tmp_path)
    _record(tmp_path, p.proposal_id)
    _record(tmp_path, p.proposal_id, staging_tree_hash="def456", files=["b.py"])
    got = manifest.get_implementation(tmp_path, p.proposal_id)
    assert got.staging_tree_hash == "def456"
    assert got.files == ["b.py"]


def test_no_implementation_yet_reads_as_none(tmp_path):
    p = _proposal(tmp_path)
    assert manifest.get_implementation(tmp_path, p.proposal_id) is None


# ---- the drop has to take everything with it -----------------------------


def test_dropping_a_proposal_clears_all_four_records(tmp_path):
    p = _proposal(tmp_path)
    _record(tmp_path, p.proposal_id)
    dropped = manifest.drop_implementation(tmp_path, p.proposal_id, step="gen_2_c")
    assert dropped.files == []
    assert dropped.variant_meta_text == ""
    assert dropped.supersede_targets == []
    assert dropped.dropped is True


def test_a_dropped_proposal_retires_nobody(tmp_path):
    # the exact bug: files removed, SUPERSEDES survives, the parent gets retired
    # for a variant that no longer exists.
    p = _proposal(tmp_path)
    _record(tmp_path, p.proposal_id)
    manifest.drop_implementation(tmp_path, p.proposal_id, step="gen_2_c")
    assert manifest.live_supersede_targets(tmp_path) == []


def test_a_live_proposal_still_retires_its_incumbent(tmp_path):
    p = _proposal(tmp_path)
    _record(tmp_path, p.proposal_id)
    assert manifest.live_supersede_targets(tmp_path) == ["agent_loop/confirm_exit"]


def test_dropping_one_proposal_leaves_the_others_alone(tmp_path):
    a = _proposal(tmp_path)
    b = proposals.create_proposal(
        tmp_path, step="gen_2_c", lane="novelty", action="add", finding_ids=["f_b"]
    )
    _record(tmp_path, a.proposal_id)
    _record(tmp_path, b.proposal_id, supersede_targets=["agent_loop/budget_aware"])
    manifest.drop_implementation(tmp_path, a.proposal_id, step="gen_2_c")
    assert manifest.live_supersede_targets(tmp_path) == ["agent_loop/budget_aware"]


def test_dropping_something_never_implemented_is_not_an_error(tmp_path):
    p = _proposal(tmp_path)
    assert manifest.drop_implementation(tmp_path, p.proposal_id, step="s") is None


def test_variant_meta_of_a_dropped_proposal_is_not_returned(tmp_path):
    p = _proposal(tmp_path)
    _record(tmp_path, p.proposal_id)
    manifest.drop_implementation(tmp_path, p.proposal_id, step="s")
    assert manifest.live_variant_meta_text(tmp_path) == ""


def test_live_variant_meta_joins_the_surviving_proposals(tmp_path):
    p = _proposal(tmp_path)
    _record(tmp_path, p.proposal_id, variant_meta_text="<variant_meta name='a'/>")
    assert "name='a'" in manifest.live_variant_meta_text(tmp_path)
