"""P1-3: two-axis routing — Lane × Action, decided against the real archive.

    Lane   incumbent | novelty | out_of_scope   — does this capability exist yet?
    Action modify | replace | add               — can the existing file carry it?

There is deliberately no taxonomy here: no `yardstick_ref`, no
`mechanism_signature`. The decision is a *closed comparison* against the
variants that are actually active right now, and it is only accepted when it can
point at them — the variant that covers it, or a reason for each one that does
not. An unfalsifiable routing is worse than none, because it looks like evidence.

`replace` exists so that "the implementation route itself is wrong" has an
outlet other than piling more onto the incumbent file. Without it the only legal
move on a 1659-line variant is to make it longer — which is how it got that long.
"""

import pytest

from harbor.agents.terminus_2_modular.self_evo import routing

pytestmark = pytest.mark.unit

ACTIVE = ["agent_loop/baseline", "agent_loop/confirm_exit", "agent_loop/budget_aware"]


def _raw(**over):
    base = {
        "lane": "incumbent",
        "action": "modify",
        "target_variant": "agent_loop/confirm_exit",
        "behavioral_delta": "require an artifact check before declaring done",
        "causal_hypothesis": "the validation classifier accepts prose",
        "rationale": "confirm_exit.py `_validate_completion` returns True on any non-empty answer",
    }
    base.update(over)
    return base


def _novelty(**over):
    base = _raw(
        lane="novelty",
        action="add",
        target_variant="",
        dismissals={
            q: f"{q} does not track requirements across episodes" for q in ACTIVE
        },
    )
    base.update(over)
    return base


def _validate(raw):
    return routing.validate_routing(raw, active_quals=ACTIVE, locked_type="agent_loop")


# ---- the happy paths ------------------------------------------------------


def test_incumbent_modify_is_accepted(tmp_path):
    got = _validate(_raw())
    assert got.lane == "incumbent"
    assert got.action == "modify"
    assert got.target_variant == "agent_loop/confirm_exit"


def test_novelty_add_is_accepted_when_every_incumbent_is_dismissed():
    got = _validate(_novelty())
    assert got.lane == "novelty"
    assert got.action == "add"


def test_out_of_scope_names_the_module_that_owns_it():
    got = _validate(
        _raw(
            lane="out_of_scope",
            action="",
            target_variant="",
            other_module="verification",
        )
    )
    assert got.lane == "out_of_scope"


# ---- axis 1 × axis 2 have to agree ---------------------------------------


def test_novelty_cannot_modify_an_incumbent():
    with pytest.raises(routing.InvalidRouting):
        _validate(_novelty(action="modify", target_variant="agent_loop/confirm_exit"))


def test_incumbent_cannot_add():
    with pytest.raises(routing.InvalidRouting):
        _validate(_raw(action="add"))


def test_out_of_scope_takes_no_action():
    with pytest.raises(routing.InvalidRouting):
        _validate(_raw(lane="out_of_scope", action="modify", other_module="tools"))


def test_unknown_lane_is_refused():
    with pytest.raises(routing.InvalidRouting):
        _validate(_raw(lane="maybe"))


def test_unknown_action_is_refused():
    with pytest.raises(routing.InvalidRouting):
        _validate(_raw(action="tweak"))


# ---- incumbent has to point at something that exists ---------------------


def test_incumbent_must_name_a_variant():
    with pytest.raises(routing.InvalidRouting):
        _validate(_raw(target_variant=""))


def test_incumbent_must_name_a_variant_that_is_actually_active():
    # naming a retired or invented variant is how a change lands on a module
    # nothing selects — 9/16 promotions had zero runtime activation.
    with pytest.raises(routing.InvalidRouting) as excinfo:
        _validate(_raw(target_variant="agent_loop/ghost"))
    assert "ghost" in str(excinfo.value)


def test_a_bare_variant_name_is_qualified_against_the_locked_type():
    got = _validate(_raw(target_variant="confirm_exit"))
    assert got.target_variant == "agent_loop/confirm_exit"


# ---- replace has to say who it retires -----------------------------------


def test_replace_must_name_what_it_supersedes():
    with pytest.raises(routing.InvalidRouting):
        _validate(_raw(action="replace", supersedes=[]))


def test_replace_with_a_named_incumbent_is_accepted():
    got = _validate(_raw(action="replace", supersedes=["agent_loop/confirm_exit"]))
    assert got.action == "replace"
    assert got.supersedes == ["agent_loop/confirm_exit"]


def test_replace_cannot_supersede_a_variant_that_is_not_active():
    with pytest.raises(routing.InvalidRouting):
        _validate(_raw(action="replace", supersedes=["agent_loop/ghost"]))


# ---- novelty has to dismiss every incumbent, one by one ------------------


def test_novelty_missing_one_dismissal_is_refused():
    partial = {q: "no" for q in ACTIVE[:-1]}
    with pytest.raises(routing.InvalidRouting) as excinfo:
        _validate(_novelty(dismissals=partial))
    assert "budget_aware" in str(excinfo.value)


def test_novelty_with_no_dismissals_at_all_is_refused():
    with pytest.raises(routing.InvalidRouting):
        _validate(_novelty(dismissals={}))


def test_an_empty_dismissal_reason_does_not_count():
    hollow = {q: "   " for q in ACTIVE}
    with pytest.raises(routing.InvalidRouting):
        _validate(_novelty(dismissals=hollow))


# ---- nothing unfalsifiable gets through ----------------------------------


def test_routing_without_a_citation_is_refused():
    with pytest.raises(routing.InvalidRouting):
        _validate(_raw(rationale=""))


def test_routing_without_a_behavioral_delta_is_refused():
    with pytest.raises(routing.InvalidRouting):
        _validate(_raw(behavioral_delta=""))


def test_out_of_scope_must_say_which_module_owns_it():
    with pytest.raises(routing.InvalidRouting):
        _validate(_raw(lane="out_of_scope", action="", target_variant=""))


# ---- parsing --------------------------------------------------------------


def test_parses_a_routing_block():
    text = '<routing>{"lane": "novelty", "action": "add"}</routing>'
    assert routing.parse_routing(text)["lane"] == "novelty"


def test_the_last_block_wins():
    text = (
        '<routing>{"lane": "incumbent", "action": "modify"}</routing>\n'
        'on reflection:\n<routing>{"lane": "novelty", "action": "add"}</routing>'
    )
    assert routing.parse_routing(text)["lane"] == "novelty"


def test_a_fenced_block_is_still_read():
    text = '<routing>\n```json\n{"lane": "novelty", "action": "add"}\n```\n</routing>'
    assert routing.parse_routing(text)["action"] == "add"


def test_no_block_parses_to_none():
    assert routing.parse_routing("I could not decide.") is None


def test_invalid_json_parses_to_none():
    assert routing.parse_routing("<routing>{lane: novelty}</routing>") is None


# ---- the prompt -----------------------------------------------------------


def _prompt():
    return routing.build_routing_instruction(
        finding={"task": "fix-git", "divergence": "declared done without checking"},
        locked_type="agent_loop",
        active_variants=[(q, f"description of {q}") for q in ACTIVE],
    )


def test_prompt_lists_every_active_variant():
    text = _prompt()
    for qual in ACTIVE:
        assert qual in text


def test_prompt_says_what_the_module_does_not_own():
    # the abstention channel already works (92/318 findings named another
    # module); the leak is the other way — once is_culprit is true, another
    # module's job gets rationalised into the locked one. Stating the boundary
    # is what gives out_of_scope something to bite on.
    text = _prompt().lower()
    assert "does not own" in text
    assert "verification" in text  # termination belongs there, not to agent_loop


def test_echoing_the_prompt_template_can_never_be_a_decision():
    # P0-1's lesson: a fully-valid example block in a prompt is an invitation for
    # the model to echo it back as if it were a verdict. There is no execution
    # gate on a one-shot call, so the defence has to be in the template itself —
    # every slot holds a `|` menu or a placeholder, so the template parses but
    # cannot validate. Worst case is a refusal, never a fabricated routing.
    echoed = routing.parse_routing(_prompt())
    with pytest.raises(routing.InvalidRouting):
        routing.validate_routing(echoed, active_quals=ACTIVE, locked_type="agent_loop")


def test_every_lane_slot_in_the_template_is_a_menu_not_a_choice():
    echoed = routing.parse_routing(_prompt())
    assert echoed["lane"] not in routing.LANES


def test_a_missing_routing_block_is_refused_not_defaulted():
    with pytest.raises(routing.InvalidRouting):
        routing.validate_routing(None, active_quals=ACTIVE, locked_type="agent_loop")


# ---- a truncated reply is an infra fact, not a model judgement -----------
# Observed live: the model writes a long, well-formed rationale and the reply is
# cut off mid-string, so there is no closing tag and nothing parses. Counting
# that as "the model could not route this" would put an endpoint artifact into a
# statistic about routing quality — the exact confusion Phase 0 existed to fix.


def test_an_unclosed_routing_block_is_detected_as_truncated():
    text = '<routing>\n{"lane": "incumbent", "rationale": "it ran budget_aware and'
    assert routing.looks_truncated(text) is True


def test_a_complete_block_is_not_truncated():
    assert routing.looks_truncated('<routing>{"lane": "novelty"}</routing>') is False


def test_prose_with_no_block_at_all_is_not_truncated():
    # nothing was started, so nothing was cut off — that IS a model failure
    assert routing.looks_truncated("I could not decide.") is False


def test_empty_reply_is_not_truncated():
    assert routing.looks_truncated("") is False


# ---- out_of_scope is declining to change anything ------------------------
# Found by running it: the validator demanded `behavioral_delta` from every
# lane, so a correct out_of_scope call — "this belongs to verification" — was
# thrown away as unusable. That does not just lose one judgement; it
# systematically suppresses the module-boundary signal this phase exists to
# measure, and inflates the refusal rate with correct answers.


def test_out_of_scope_does_not_owe_a_behavioral_delta():
    got = _validate(
        _raw(
            lane="out_of_scope",
            action="",
            target_variant="",
            behavioral_delta="",
            other_module="verification",
        )
    )
    assert got.lane == "out_of_scope"
    assert got.other_module == "verification"


def test_out_of_scope_still_owes_a_rationale():
    # it must still say WHY it belongs elsewhere — that part is auditable
    with pytest.raises(routing.InvalidRouting):
        _validate(
            _raw(
                lane="out_of_scope",
                action="",
                target_variant="",
                behavioral_delta="",
                rationale="",
                other_module="verification",
            )
        )


def test_lanes_that_propose_a_change_still_owe_one():
    for lane_kw in ({}, {"lane": "novelty", "action": "add", "target_variant": ""}):
        with pytest.raises(routing.InvalidRouting):
            base = _novelty() if lane_kw else _raw()
            base.update(lane_kw)
            base["behavioral_delta"] = ""
            _validate(base)


# ---- "covers it" was ambiguous, and both readings were self-consistent ----
# Measured: the router called `novelty` 15 times; an independent judge said 13 of
# those were already someone's job. Zero errors the other way. The cause was not
# accuracy — it was this prompt. "An active variant already covers it" reads
# equally as "owns the responsibility" and as "actually achieves the behaviour",
# and the two judges each picked one and held it consistently.
#
# For the decision this drives — modify an existing file, or write a new one —
# "owns the responsibility" is the reading that serves the goal. A variant that
# gates on evidence but defines evidence too loosely should be tightened, not
# duplicated. Duplicating is how the archive ended up with confirm_exit,
# budget_aware and deadline_aware all doing termination.


def test_prompt_defines_incumbent_as_owning_the_responsibility():
    text = _prompt().lower()
    assert "responsib" in text


def test_prompt_says_a_broken_implementation_is_still_an_incumbent():
    # the exact case that was mis-routed 13 times
    text = _prompt().lower()
    assert "too loose" in text or "buggy" in text or "wrong" in text
    assert "do not" in text or "not a reason" in text


def test_prompt_still_requires_novelty_to_dismiss_each_variant():
    assert "every" in _prompt().lower()
