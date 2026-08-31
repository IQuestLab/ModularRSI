"""Route findings on two axes: search lane and implementation action.

Routing is accepted only when its claims are grounded against every active
variant in the target module library.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

LANE_INCUMBENT = "incumbent"
LANE_NOVELTY = "novelty"
LANE_OUT_OF_SCOPE = "out_of_scope"
LANES = (LANE_INCUMBENT, LANE_NOVELTY, LANE_OUT_OF_SCOPE)

ACTION_MODIFY = "modify"
ACTION_REPLACE = "replace"
ACTION_ADD = "add"

#: Which actions each lane may take. out_of_scope takes none — this module is
#: not where that work belongs, so it does not get a slot here.
_LANE_ACTIONS: dict[str, frozenset[str]] = {
    LANE_INCUMBENT: frozenset({ACTION_MODIFY, ACTION_REPLACE}),
    LANE_NOVELTY: frozenset({ACTION_ADD}),
    LANE_OUT_OF_SCOPE: frozenset({""}),
}

#: What each module does NOT own, and who does. The investigator's abstention
#: channel already works — 92 of 318 real findings named another module — but the
#: opposite leak has nothing holding it back: once a finding is a culprit, the
#: consolidator is told it may only edit `{locked}/`, so another module's job
#: gets rationalised into this one. Stating the boundary is what gives
#: `out_of_scope` something to bite on. Observed instance: all four evolved
#: agent_loop variants were doing termination work, which belongs to
#: `verification`.
MODULE_BOUNDARIES: dict[str, str] = {
    "agent_loop": (
        "agent_loop does NOT own: deciding whether the work is finished "
        "(verification), how terminal output is captured or truncated "
        "(observation), what stays in the prompt (context_mgmt), or how a "
        "command is executed (tools)."
    ),
    "observation": (
        "observation does NOT own: what to do about what it saw (agent_loop), "
        "judging completion (verification), or command execution (tools)."
    ),
    "context_mgmt": (
        "context_mgmt does NOT own: what to capture (observation), the control "
        "flow that consumes the context (agent_loop), or completion (verification)."
    ),
    "tools": (
        "tools does NOT own: when to run a command (agent_loop), how the result "
        "is summarised (observation), or completion (verification)."
    ),
    "verification": (
        "verification does NOT own: the control loop itself (agent_loop), "
        "capture (observation), or execution (tools)."
    ),
}

_ROUTING_RE = re.compile(r"<routing>(.*?)</routing>", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"^```[a-zA-Z]*\s*|\s*```$", re.MULTILINE)


class InvalidRouting(ValueError):
    """The routing cannot be audited — refused rather than half-trusted."""


@dataclass
class RoutingDecision:
    lane: str
    action: str
    target_variant: str = ""
    supersedes: list[str] = field(default_factory=list)
    dismissals: dict[str, str] = field(default_factory=dict)
    behavioral_delta: str = ""
    causal_hypothesis: str = ""
    rationale: str = ""
    other_module: str = ""


def parse_routing(text: str | None) -> dict | None:
    """Read the last ``<routing>`` block. Unparseable → ``None``, never a guess."""
    if not text:
        return None
    blocks = _ROUTING_RE.findall(text)
    if not blocks:
        return None
    raw = _FENCE_RE.sub("", blocks[-1]).strip()
    try:
        parsed = json.loads(raw)
    except ValueError:
        return None
    return parsed if isinstance(parsed, dict) else None


def looks_truncated(text: str | None) -> bool:
    """Was a routing block started and then cut off?

    Observed live: the model writes a long, well-formed rationale and the reply
    ends mid-string, so there is no closing tag and nothing parses. That is a
    fact about the endpoint (a token ceiling), not about whether the finding
    could be routed — and counting it as a model refusal would put an infra
    artifact straight into a statistic about routing quality. Prose with no block
    at all is NOT truncation: nothing was started, so nothing was cut off.
    """
    if not text:
        return False
    return "<routing>" in text.lower() and "</routing>" not in text.lower()


_READ_ONLY_BASENAMES = frozenset({"baseline"})


def is_read_only(qual: str) -> bool:
    """Return whether an editor may modify this qualified variant in place."""
    return qual.rsplit("/", 1)[-1] in _READ_ONLY_BASENAMES


def _qualify(name: str, locked_type: str) -> str:
    name = (name or "").strip()
    if not name or "/" in name:
        return name
    return f"{locked_type}/{name}"


def _text(raw: dict, key: str) -> str:
    value = raw.get(key)
    return value.strip() if isinstance(value, str) else ""


def validate_routing(
    raw: dict, *, active_quals: list[str], locked_type: str
) -> RoutingDecision:
    """Turn a parsed routing block into a decision, or refuse it with a reason."""
    if not isinstance(raw, dict):
        raise InvalidRouting("no routing block was emitted")
    lane = _text(raw, "lane")
    action = _text(raw, "action")
    if lane not in LANES:
        raise InvalidRouting(f"unknown lane {lane!r}")
    if action not in _LANE_ACTIONS[lane]:
        allowed = " | ".join(sorted(x for x in _LANE_ACTIONS[lane] if x)) or "(none)"
        raise InvalidRouting(f"lane {lane} allows {allowed}, got {action!r}")

    behavioral_delta = _text(raw, "behavioral_delta")
    rationale = _text(raw, "rationale")
    # out_of_scope is a refusal to change anything HERE, so demanding "what would
    # this module do differently" contradicts the verdict. Requiring it anyway
    # threw away correct boundary calls and suppressed the very signal this phase
    # is measuring. The other two lanes propose a change and still owe one.
    if lane != LANE_OUT_OF_SCOPE and not behavioral_delta:
        raise InvalidRouting("behavioral_delta is required — what would change?")
    if not rationale:
        raise InvalidRouting(
            "rationale is required — cite the variant or the trajectory"
        )

    active = set(active_quals)
    target = _qualify(_text(raw, "target_variant"), locked_type)
    supersedes = [
        _qualify(x, locked_type)
        for x in raw.get("supersedes") or []
        if isinstance(x, str) and x.strip()
    ]
    dismissals = {
        _qualify(k, locked_type): v
        for k, v in (raw.get("dismissals") or {}).items()
        if isinstance(k, str) and isinstance(v, str) and v.strip()
    }
    other_module = _text(raw, "other_module")

    if lane == LANE_INCUMBENT:
        if not target:
            raise InvalidRouting("incumbent must name the variant that covers it")
        if target not in active:
            raise InvalidRouting(
                f"{target} is not an active variant — a change aimed at it would "
                "land on code nothing selects"
            )
        if action == ACTION_MODIFY and is_read_only(target):
            raise InvalidRouting(
                f"{target} is read-only — the editor's tools refuse every write "
                "to a baseline.py, so `modify` cannot be implemented. Route "
                "`replace` instead: create a new variant and name the target "
                "under SUPERSEDES."
            )
        if action == ACTION_REPLACE:
            if not supersedes:
                raise InvalidRouting("replace must name the incumbent it retires")
            unknown = [s for s in supersedes if s not in active]
            if unknown:
                raise InvalidRouting(f"cannot supersede inactive variant(s): {unknown}")

    elif lane == LANE_NOVELTY:
        missing = [q for q in active_quals if q not in dismissals]
        if missing:
            raise InvalidRouting(
                "novelty must say why EACH active variant does not cover it; "
                f"missing: {missing}"
            )

    else:  # out_of_scope
        if not other_module:
            raise InvalidRouting("out_of_scope must name the module that owns it")

    return RoutingDecision(
        lane=lane,
        action=action,
        target_variant=target,
        supersedes=supersedes,
        dismissals=dismissals,
        behavioral_delta=behavioral_delta,
        causal_hypothesis=_text(raw, "causal_hypothesis"),
        rationale=rationale,
        other_module=other_module,
    )


#: The response template. Every field carries a `|`-separated menu or a
#: description rather than a committed value, so the template itself can never
#: parse as a decision — echoing the example back is not a verdict. (Same defence
#: as the review verdict: a fully-valid example in a prompt is an invitation.)
_RESPONSE_TEMPLATE = """<routing>
{
  "lane": "incumbent | novelty | out_of_scope",
  "action": "modify | replace | add | (empty for out_of_scope)",
  "target_variant": "<type/name of the ACTIVE variant that covers it — incumbent only>",
  "supersedes": ["<type/name you are retiring — replace only>"],
  "dismissals": {"<type/name>": "<why this active variant does not cover it — novelty only, one entry per active variant>"},
  "behavioral_delta": "<what the agent would DO differently>",
  "causal_hypothesis": "<why that would move this failure>",
  "rationale": "<cite it: the file+method that covers it, or where in the trajectory the agent repeatedly could not>",
  "other_module": "<which module owns it — out_of_scope only>"
}
</routing>"""

_PREAMBLE = """You are routing ONE finding against the modules that are live right now.

## The finding

{finding}

## Active `{locked}` variants — this is the whole comparison set

{variants}

## Scope boundary

{boundaries}

If the finding is really about work this module does not own, say so
(`lane: out_of_scope`) and name the module that does. Do not fold another
module's job into `{locked}` just because `{locked}` is the one you may edit.

## Two independent decisions

**Lane — is one of the variants above already RESPONSIBLE for this?**

Responsibility, not quality. Ask "whose job is this?", never "is it done well?".

- `incumbent` — one active variant above already owns this responsibility. Name
  which one, and cite the method or line in its file.

  **A variant that owns it but does it badly is STILL the incumbent.** If it has
  the check but the check is too loose, or the rule is buggy, or it covers three
  cases out of four — that is an incumbent to tighten, NOT a missing capability.
  "Its definition of X is wrong" is not a reason to call this novelty; it is the
  reason to modify that variant.

- `novelty` — NO variant above owns this responsibility at all. Nobody is even
  trying. You must give a separate reason for EVERY variant listed above saying
  why this is not its job, and cite where in the trajectory the agent repeatedly
  could not do this. A blanket "nothing covers this" is not acceptable.

- `out_of_scope` — another module's job.

Why the line is drawn there: this decision picks between editing an existing
file and writing a new one. Calling a badly-implemented responsibility "novelty"
produces a second variant that does the same job slightly differently, and then
both compete for selection. That is exactly how this module type ended up with
three separate variants all deciding when to stop.

**Action — can the existing file carry the change?**

- `modify` — incumbent, and editing that file is the honest place for it.
- `replace` — incumbent, but the implementation route itself is the problem.
  A new file that must name the incumbent it retires. Prefer this over piling
  another special case onto a file that is already large.
- `add` — novelty only.

Answer with exactly one block in this shape, filled in with your decision:

{template}"""


def build_routing_instruction(
    *,
    finding: dict,
    locked_type: str,
    active_variants: list[tuple[str, str]],
) -> str:
    """Build the one-shot routing prompt for a single finding."""
    variants = (
        "\n".join(
            f"- `{qual}` — {desc}"
            + (
                " [read-only: `modify` is unavailable; use `replace` to "
                "retire it with a new variant]"
                if is_read_only(qual)
                else ""
            )
            for qual, desc in active_variants
        )
        or "- (none active)"
    )
    return _PREAMBLE.format(
        finding=json.dumps(finding, indent=2, ensure_ascii=False)[:6000],
        locked=locked_type,
        variants=variants,
        boundaries=MODULE_BOUNDARIES.get(
            locked_type, f"`{locked_type}` owns only its own responsibility."
        ),
        template=_RESPONSE_TEMPLATE,
    )
