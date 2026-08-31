"""C3: the two-axis routing has to be an invariant, not a suggestion.

The router decides `lane` and `action` — modify this incumbent, replace it, or
add something new — and the implementer is *told* which one it is. Nothing then
checked that it did it. So a novelty proposal could come back as an edit to the
incumbent, a `replace` could come back as more logic piled into the 1659-line
variant it was supposed to retire, and every downstream gate would still pass.

That makes the whole two-axis routing decorative: the archive records an action
the tree does not implement, the niche cell the change was supposed to open
stays empty, and the "did adding variants help?" question can never be answered
because half the adds were modifies.

The checks are mechanical on purpose — they compare the declared action against
the library before and after plus the files touched, and nothing here asks a
model anything. Each one is stated so it can only fire on unambiguous evidence:
a check that guesses would reject good changes, and a rejected change costs a
whole generation.
"""

import pytest

from harbor.agents.terminus_2_modular.self_evo import action_gate

pytestmark = pytest.mark.unit

PARENT = {"agent_loop/confirm_exit", "observation/plain"}


def _check(action, **over):
    kw = dict(
        action=action,
        target_variant="agent_loop/confirm_exit",
        files_changed=[],
        variant_meta_text="",
        parent_quals=PARENT,
        staged_quals=set(PARENT),
    )
    kw.update(over)
    return action_gate.check_action(**kw)


# ---- MODIFY --------------------------------------------------------------


def test_a_modify_that_changes_its_target_passes(_=None):
    assert _check("modify", files_changed=["agent_loop/confirm_exit.py"]) is None


def test_a_modify_that_never_touches_its_target_is_refused():
    reason = _check("modify", files_changed=["agent_loop/something_else.py"])
    assert reason is not None
    assert "agent_loop/confirm_exit" in reason


def test_a_modify_that_edits_another_live_variant_is_refused():
    # one generation, one change: an edit that also rewrites a second active
    # variant makes the generation's result unattributable to either
    reason = _check(
        "modify",
        files_changed=["agent_loop/confirm_exit.py", "observation/plain.py"],
    )
    assert reason is not None
    assert "observation/plain" in reason


def test_a_modify_may_add_helper_files():
    # a helper is not a variant — nothing registers it, so nothing selects it
    assert (
        _check(
            "modify",
            files_changed=[
                "agent_loop/confirm_exit.py",
                "agent_loop/exit_helper/probe.py",
            ],
        )
        is None
    )


# ---- ADD -----------------------------------------------------------------


def test_an_add_that_registers_a_new_variant_passes():
    assert (
        _check(
            "add",
            target_variant="",
            files_changed=["agent_loop/deadline_aware.py"],
            staged_quals=PARENT | {"agent_loop/deadline_aware"},
        )
        is None
    )


def test_an_add_that_registers_nothing_new_is_refused():
    # the library is the only thing that decides what exists; a file that
    # registers nothing is not an addition, whatever it is called
    reason = _check("add", target_variant="", files_changed=["agent_loop/notes.py"])
    assert reason is not None
    assert "new variant" in reason


def test_an_add_that_is_really_an_edit_to_the_incumbent_is_refused():
    reason = _check(
        "add",
        target_variant="",
        files_changed=["agent_loop/deadline_aware.py", "agent_loop/confirm_exit.py"],
        staged_quals=PARENT | {"agent_loop/deadline_aware"},
    )
    assert reason is not None
    assert "agent_loop/confirm_exit" in reason


# ---- REPLACE -------------------------------------------------------------


_META = """<variant_meta name="deadline_aware" type="agent_loop">
PARENT: agent_loop/confirm_exit
SUPERSEDES: {supersedes}
</variant_meta>"""


def test_a_replace_that_registers_a_successor_and_declares_it_passes():
    assert (
        _check(
            "replace",
            files_changed=["agent_loop/deadline_aware.py"],
            staged_quals=PARENT | {"agent_loop/deadline_aware"},
            variant_meta_text=_META.format(supersedes="agent_loop/confirm_exit"),
        )
        is None
    )


def test_a_replace_may_name_its_target_without_the_type_prefix():
    # editors write `SUPERSEDES: confirm_exit` at least as often as the
    # qualified form; refusing that would reject correct work over punctuation
    assert (
        _check(
            "replace",
            files_changed=["agent_loop/deadline_aware.py"],
            staged_quals=PARENT | {"agent_loop/deadline_aware"},
            variant_meta_text=_META.format(supersedes="confirm_exit"),
        )
        is None
    )


def test_a_replace_that_does_not_declare_what_it_supersedes_is_refused():
    # without the declaration the archive keeps the retired variant active and
    # the composer goes on selecting the thing this was meant to replace
    reason = _check(
        "replace",
        files_changed=["agent_loop/deadline_aware.py"],
        staged_quals=PARENT | {"agent_loop/deadline_aware"},
        variant_meta_text=_META.format(supersedes="observation/plain"),
    )
    assert reason is not None
    assert "SUPERSEDES" in reason


def test_a_replace_that_keeps_piling_logic_into_the_incumbent_is_refused():
    reason = _check(
        "replace",
        files_changed=["agent_loop/deadline_aware.py", "agent_loop/confirm_exit.py"],
        staged_quals=PARENT | {"agent_loop/deadline_aware"},
        variant_meta_text=_META.format(supersedes="agent_loop/confirm_exit"),
    )
    assert reason is not None
    assert "in place" in reason


def test_a_replace_that_registers_no_successor_is_refused():
    reason = _check(
        "replace",
        files_changed=["agent_loop/confirm_exit.py"],
        variant_meta_text=_META.format(supersedes="agent_loop/confirm_exit"),
    )
    assert reason is not None
    assert "new variant" in reason


# ---- staying out of the way ----------------------------------------------


def test_an_unrouted_change_is_not_second_guessed():
    # the legacy single-candidate path has no proposal and therefore no
    # declared action; there is nothing here to hold it to
    assert _check("", target_variant="") is None
    assert _check("none", target_variant="") is None


# ---- what "retired" actually means ---------------------------------------
#
# A variant is retired when the library stops offering it — that is what the
# composer reads and what the archive records. Whether its FILE was edited on
# the way out says nothing: an editor that empties the incumbent so it no longer
# `register()`s has retired it more thoroughly than one that leaves the file
# untouched and only declares SUPERSEDES.


def test_a_replace_may_retire_its_target_by_unregistering_it():
    # the successor exists, SUPERSEDES is declared, and the incumbent is GONE
    # from the staged library — that is a completed replace, and rejecting it
    # for having touched the file costs a generation for doing the job properly
    assert (
        _check(
            "replace",
            files_changed=[
                "NEW:agent_loop/deadline_aware.py",
                "agent_loop/confirm_exit.py",
            ],
            staged_quals={"agent_loop/deadline_aware", "observation/plain"},
            variant_meta_text=_META.format(supersedes="agent_loop/confirm_exit"),
        )
        is None
    )


# ---- the shape `files_changed` actually arrives in ------------------------
#
# `_changed_py_files` prefixes files absent from the parent tree with `NEW:`.
# Every test above hand-writes bare paths, so the gate has only ever been
# exercised on a shape production never produces for an ADDED file — and an
# added file is the one thing every `add` and `replace` turns on.


def test_a_newly_added_file_is_recognised_through_its_marker():
    from harbor.agents.terminus_2_modular.self_evo.action_gate import qual_for_path

    assert qual_for_path("NEW:agent_loop/deadline_aware.py") == (
        "agent_loop/deadline_aware"
    )
    # kernel mode stacks the marker in front of the package root
    assert qual_for_path("NEW:modules/agent_loop/deadline_aware.py") == (
        "agent_loop/deadline_aware"
    )


def test_an_add_that_edits_an_incumbent_is_caught_in_the_real_shape():
    # the add registers its new variant (marked NEW:, as production would) and
    # also rewrites the incumbent — the incumbent edit is what makes it not an
    # add, and the marker must not hide either half
    reason = _check(
        "add",
        target_variant="",
        files_changed=[
            "NEW:agent_loop/deadline_aware.py",
            "agent_loop/confirm_exit.py",
        ],
        staged_quals=PARENT | {"agent_loop/deadline_aware"},
    )
    assert reason is not None
    assert "agent_loop/confirm_exit" in reason
    assert "NEW:" not in reason  # the marker is a transport detail, not a name
