"""H4: a window that dies mid-flight must not strand its proposals.

`implement_lanes` moves a proposal `open → selected` before it starts the two
implementers. Everything after that — building the staging, running the editor,
the gate battery, the `selected → attempted` transition — takes tens of minutes
against an endpoint with a history of wedging, on a box with a history of being
reaped. Die anywhere in there and the proposal is `selected` forever: the
portfolio only ever picks `open`, so the direction is silently unaskable for the
rest of the lineage, with its accumulated support intact and useless.

The only recovery that existed was for a staging that failed to build — an
exception inside a still-running process. A process that is gone raises nothing.

So the window writes down what it is doing, and the next window reads it:

    selected        — this proposal was taken, here is its staging
    attempt_started — an implementer was actually launched
    attempt_finished— the lane reached a verdict; nothing to recover

Recovery turns on whether the implementer was launched, because that is what
decides who pays. Launched-and-lost is `attempted` and settles through
`rejected_implementation` — the direction keeps its retry, but the attempt is
charged, or a proposal could be retried forever and never reach taboo. Never
launched is put back to `open` untouched: nothing was tried on its behalf, and
charging it would spend one of its two lives for a crash it had no part in.
"""

import pytest

from harbor.agents.terminus_2_modular.self_evo import lease, proposals

pytestmark = pytest.mark.unit

WINDOW = "gen_5_candidate_100"
LATER = "gen_6_candidate_200"


def _proposal(tmp_path, pid_step=WINDOW):
    return proposals.create_proposal(
        tmp_path,
        step=pid_step,
        lane="incumbent",
        action="modify",
        target_variant="agent_loop/confirm_exit",
        behavioral_delta="check the artifact",
        causal_hypothesis="it trusts the self-report",
        finding_ids=["f_a"],
        support_tasks=["fix-git"],
    )


def _take(tmp_path, rec, *, step=WINDOW, started=False, finished=False):
    proposals.transition(tmp_path, rec.proposal_id, step=step, to=proposals.SELECTED)
    lease.record_selected(
        tmp_path,
        rec.proposal_id,
        lane="incumbent",
        step=step,
        staging_path=str(tmp_path / "staging" / "incumbent"),
    )
    if started:
        lease.record_attempt_started(tmp_path, rec.proposal_id, step=step)
    if finished:
        lease.record_attempt_finished(tmp_path, rec.proposal_id, step=step, ok=True)


# ---- the journal ---------------------------------------------------------


def test_a_taken_proposal_is_written_down_with_its_staging(tmp_path):
    rec = _proposal(tmp_path)
    _take(tmp_path, rec)
    (open_,) = lease.open_leases(tmp_path)
    assert open_.proposal_id == rec.proposal_id
    assert open_.step == WINDOW
    assert open_.staging_path.endswith("incumbent")
    assert open_.attempt_started is False


def test_a_finished_lane_leaves_no_open_lease(tmp_path):
    rec = _proposal(tmp_path)
    _take(tmp_path, rec, started=True, finished=True)
    assert lease.open_leases(tmp_path) == []


def test_launching_the_implementer_is_recorded_separately(tmp_path):
    rec = _proposal(tmp_path)
    _take(tmp_path, rec, started=True)
    (open_,) = lease.open_leases(tmp_path)
    assert open_.attempt_started is True


# ---- recovery ------------------------------------------------------------


def test_a_proposal_stranded_before_the_implementer_goes_back_to_open(tmp_path):
    rec = _proposal(tmp_path)
    _take(tmp_path, rec)  # crash here: selected, never launched

    (recovered,) = lease.recover(tmp_path, step=LATER)
    assert recovered.proposal_id == rec.proposal_id
    assert recovered.action == lease.REOPENED
    after = proposals.get_proposal(tmp_path, rec.proposal_id)
    assert after.state == proposals.OPEN
    # nothing was tried on its behalf, so nothing is charged
    assert after.attempts == 0


def test_a_proposal_stranded_after_the_implementer_launched_is_settled(tmp_path):
    rec = _proposal(tmp_path)
    _take(tmp_path, rec, started=True)  # crash here: the editor was running

    (recovered,) = lease.recover(tmp_path, step=LATER)
    assert recovered.action == lease.SETTLED
    after = proposals.get_proposal(tmp_path, rec.proposal_id)
    # rejected_implementation, not taboo: a build that never finished says
    # nothing about whether the direction was right
    assert after.state == proposals.REJECTED_IMPLEMENTATION
    assert after.attempts == 1


def test_a_settled_proposal_can_still_be_retried(tmp_path):
    rec = _proposal(tmp_path)
    _take(tmp_path, rec, started=True)
    lease.recover(tmp_path, step=LATER)
    proposals.transition(
        tmp_path, rec.proposal_id, step=LATER, to=proposals.OPEN, reason="retry"
    )
    assert [p.proposal_id for p in proposals.open_proposals(tmp_path)] == [
        rec.proposal_id
    ]


def test_the_current_window_is_not_recovered_out_from_under_itself(tmp_path):
    # the lanes of THIS window are in flight; treating them as stranded would
    # hand their proposals back to the portfolio while they are being built
    rec = _proposal(tmp_path)
    _take(tmp_path, rec, started=True)
    assert lease.recover(tmp_path, step=WINDOW) == []
    assert proposals.get_proposal(tmp_path, rec.proposal_id).state == (
        proposals.SELECTED
    )


def test_recovery_is_idempotent(tmp_path):
    rec = _proposal(tmp_path)
    _take(tmp_path, rec)
    assert len(lease.recover(tmp_path, step=LATER)) == 1
    assert lease.recover(tmp_path, step=LATER) == []
    assert proposals.get_proposal(tmp_path, rec.proposal_id).attempts == 0


def test_a_proposal_that_moved_on_without_us_is_left_alone(tmp_path):
    # the lease is stale, not the proposal: something already carried it past
    # `selected`, so there is nothing stranded to recover
    rec = _proposal(tmp_path)
    _take(tmp_path, rec, started=True)
    proposals.transition(tmp_path, rec.proposal_id, step=WINDOW, to=proposals.ATTEMPTED)
    assert lease.recover(tmp_path, step=LATER) == []
    assert proposals.get_proposal(tmp_path, rec.proposal_id).state == (
        proposals.ATTEMPTED
    )


def test_recovery_never_raises_into_the_window(tmp_path):
    # a proposal named in the journal but missing from the backlog must not
    # take down the window that was only trying to tidy up
    lease.record_selected(
        tmp_path, "p_9999", lane="novelty", step=WINDOW, staging_path="/nope"
    )
    assert lease.recover(tmp_path, step=LATER) == []
