"""P2 promotion — two candidates from one window, and what survives a rebase.

Both lanes are implemented in parallel off the *same* parent, so they cannot
both be promoted onto that parent as-is: whichever goes second has to land on a
tree that already contains the first. Everything here is about that second move.

**Which gate results survive the rebase** is the part that is easy to get wrong
in both directions:

* ``review`` judges *the change itself*. A byte-identical diff is the same code
  and the same reasoning, so that verdict carries over. Throwing it away would
  buy a second review pass for nothing — and review is historically the biggest
  source of false rejects, so every extra pass is another chance to kill a good
  change.
* ``smoke``, ``activation`` and ``routing`` judge *the tree*, not the diff, so
  none of them survives a rebase — the tree is not the one they measured.

  Smoke is the non-obvious one. It is not a syntax check on the changed files:
  ``check_library_load`` builds the **entire** module library, so two lanes that
  each load cleanly can merge into a tree that does not (two variants
  registering the same name being the obvious case). The merged tree is a
  combination neither lane was ever tested as, so its smoke verdict does not
  exist yet — and smoke costs no LLM call and no container, so re-running it is
  free insurance.

  Activation and routing depend on the composer's candidate set, and the other
  candidate has just added a variant — and a DESCRIPTION — to it. Reusing them
  would re-create exactly the "promoted but nothing ever selects it" failure
  that this phase exists to close.

Order: incumbent first by default. That is not neutral — it means novelty always
takes the extra gate exposure, and review is where false rejects live. The
``novelty_first`` switch exists so that if the data shows novelty dying more
often on its rebase pass than on its first pass, the order can be alternated
rather than argued about.

A window may therefore produce **two** generations. Each is still a single
change, so per-generation attribution is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: Gates that judge the change on its own — a byte-identical diff means the same
#: verdict. Only review qualifies, and only because it reasons about the diff.
_CHANGE_LOCAL_GATES = ("review",)

#: Gates that judge the whole TREE. Never reusable after a rebase, because the
#: tree they measured is not the tree being promoted. Smoke belongs here despite
#: feeling change-local: it loads the entire library, and two individually
#: loadable trees can merge into one that is not.
_TREE_WIDE_GATES = ("smoke", "activation", "routing")

ALL_GATES = _CHANGE_LOCAL_GATES + _TREE_WIDE_GATES

LANE_ORDER_DEFAULT = ("incumbent", "novelty")


@dataclass(frozen=True)
class Candidate:
    proposal_id: str
    lane: str
    passed_gates: bool
    diff_hash: str = ""


@dataclass
class Step:
    proposal_id: str
    lane: str
    order: int
    rebase_required: bool
    gates_to_rerun: tuple[str, ...] = field(default_factory=tuple)


def gates_after_rebase(
    diff_before: str, diff_after: str
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """``(reusable, must_rerun)`` once the rebase has actually happened.

    Identical diff → the change-local verdicts stand; the tree-wide ones do not,
    because the tree is not the one they were measured on.
    """
    if diff_before and diff_before == diff_after:
        return _CHANGE_LOCAL_GATES, _TREE_WIDE_GATES
    return (), ALL_GATES


def plan_promotions(
    candidates: list[Candidate], *, novelty_first: bool = False
) -> list[Step]:
    """Order the passing candidates and say which needs a rebase.

    Only candidates that cleared their own gates are considered; a failure in
    one lane never blocks the other.
    """
    lanes = [c.lane for c in candidates]
    if len(lanes) != len(set(lanes)):
        raise ValueError(
            f"one candidate per lane is the contract, got {lanes} — "
            "two in the same lane means the selector broke upstream"
        )

    order = tuple(reversed(LANE_ORDER_DEFAULT)) if novelty_first else LANE_ORDER_DEFAULT
    passing = [c for c in candidates if c.passed_gates]
    ranked = sorted(
        passing,
        key=lambda c: order.index(c.lane) if c.lane in order else len(order),
    )

    steps: list[Step] = []
    for index, cand in enumerate(ranked):
        rebase = index > 0
        steps.append(
            Step(
                proposal_id=cand.proposal_id,
                lane=cand.lane,
                order=index,
                rebase_required=rebase,
                # Before the rebase runs we do not know whether the diff will
                # come out identical, so the plan asks for the safe superset and
                # narrows it afterwards with `gates_after_rebase`.
                gates_to_rerun=ALL_GATES if rebase else (),
            )
        )
    return steps
