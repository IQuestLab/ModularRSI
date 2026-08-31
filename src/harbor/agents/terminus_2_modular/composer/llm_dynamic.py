"""LLMComposer — PER-TASK dynamic module selection.

Unlike ``StaticComposer`` (which returns the same bundle for every task), this
composer is called once per task with the task ``instruction`` and asks an LLM to
assemble the bundle BEST SUITED TO THAT TASK from whatever variants the library
currently holds — e.g. a task with no Python need not pull in a Python-heavy tool
variant.

It still honors ``active_bundle.json``, but as a PIN rather than a full bundle: a
type named there is frozen to that variant for this modules tree and never
reaches the per-task prompt. That is what lets a SERIAL lineage stand on a
previous lineage's winner (see ``bundle_config``).

It selects by JUDGMENT over each variant's ``DESCRIPTION`` (there is no held-out
fitness signal yet; that is a later layer). It ALWAYS returns a valid
``ModuleBundle``: on any LLM/parse failure, or for a module type with a single
implementation, it falls back to the library's default for that type.

Cost note: this adds ONE LLM call at the start of every task (the per-task tax of
per-task composition). Types with only one implementation never trigger a call.
"""

from __future__ import annotations

import asyncio
import json
import logging

from harbor.agents.terminus_2_modular.composer.static import DEFAULT_BUNDLE
from harbor.agents.terminus_2_modular.protocols import (
    ModuleBundle,
    ModuleCtx,
    ModuleInfo,
    ModuleSpec,
    ModuleStatsView,
)

_logger = logging.getLogger(__name__)

_TYPES = ("agent_loop", "observation", "context_mgmt", "tools", "verification")

# Solver helper tools: an archived type like the five above (own niche cell,
# lineage, status) but with SUBSET arity — a task is given all the active helpers
# at once, not one winner — so it is composed onto the tools spec's params
# instead of being a sixth pick-one slot in ModuleBundle.
_HELPER_TYPE = "tool_helper"

# Qualified implementations that must never reach the solver picker.
#
# The first two are editor-only infrastructure. ``combined_robust`` is retained
# in its lineage for diagnosis but quarantined after a held-out TB2 regression:
# it blocks valid heredoc writes, while its advertised write_file/edit_block
# bypass is unreachable due to a bad branch nesting. A repaired tool should be
# introduced under a new, independently validated variant name.
_NON_SOLVER_IMPLS = frozenset(
    {
        ("tools", "editor_file_tools"),
        ("context_mgmt", "passthrough"),
        ("tools", "combined_robust"),
    }
)

_MAX_INSTRUCTION_CHARS = 2000


def _archive_skip(ctx: ModuleCtx) -> frozenset[str]:
    """Quals (``type/name``) the run's archive.json marks superseded/excluded, so
    the composer never selects a retired variant even though its file is still in
    the additive library (F2). archive.json lives at the run root =
    ``<modules_root>/../..``. Fail-open: no modules_root / no archive / any error →
    empty set (composer degrades to pure niche dedup)."""
    try:
        from pathlib import Path

        mr = getattr(getattr(ctx, "state", None), "modules_root", None)
        if not mr:
            return frozenset()
        from harbor.agents.terminus_2_modular import archive as _arch

        run_dir = Path(mr).resolve().parent.parent
        return frozenset(
            f"{e.type}/{e.name}"
            for e in _arch.load_archive(run_dir)
            if e.status in ("superseded", "excluded")
        )
    except Exception:
        return frozenset()


def _pinned(ctx: ModuleCtx, library: list[ModuleInfo]) -> dict[str, str]:
    """Module types FROZEN for this modules tree by
    ``<modules_root>/active_bundle.json`` → ``{type: impl_name}``.

    Serial evolution: a lineage inherits a previous lineage's winner as its
    foundation (e.g. ``agent_loop=planning_with_guard``) and evolves a different
    type on top. A pinned type keeps its named variant and never reaches the
    per-task picker, so the inherited winner runs on EVERY roll instead of
    silently losing to `DEFAULT_BUNDLE`'s `baseline`.

    The pin lives in the tree rather than in a CLI flag on purpose: it then
    travels through staging → promotion → held-out eval automatically, so a
    generation is always evaluated on the same foundation it was evolved on. A
    flag would have to be remembered at every one of those call sites, and
    forgetting it fails silently.

    Fail-open to {} (same contract as `_archive_skip`); `online_evo` runs a hard
    preflight so a broken pin aborts the RUN rather than degrading a roll.
    """
    try:
        mr = getattr(getattr(ctx, "state", None), "modules_root", None)
        if not mr:
            return {}
        from harbor.agents.terminus_2_modular.composer.bundle_config import (
            load_bundle_overrides,
        )

        overrides = load_bundle_overrides(mr, library, logger=_logger)
        return {t: spec.name for t, spec in (overrides or {}).items() if t in _TYPES}
    except Exception:
        return {}


def _locked_type(ctx: ModuleCtx) -> str | None:
    """The single module type this experiment may WRITE to (self-evo ablation
    lock), read from RuntimeState. None = no lock. Fail-open to None."""
    try:
        return getattr(getattr(ctx, "state", None), "locked_module_type", None) or None
    except Exception:
        return None


def _scope_is_all(ctx: ModuleCtx) -> bool:
    """True when the lock narrows WRITES only and the composer should pick every
    type normally (RuntimeState.composer_scope == "all").

    SERIAL lineages need this. A serial lineage starts from a previous lineage's
    tree, so its other types already hold that lineage's variants; under the
    default "locked" scope those variants would sit in the library unreachable —
    the composer would put every non-locked type back on its `baseline` default
    and the inheritance would be decorative. Pinning them to ONE variant instead
    (`active_bundle.json`) is not the answer either: it discards the rest of the
    search space the donor lineage produced, and the more variants it produced,
    the more gets discarded.

    Fail-open to False = today's behavior, so parallel single-lock runs are
    unchanged.
    """
    try:
        return (
            str(
                getattr(getattr(ctx, "state", None), "composer_scope", "") or ""
            ).lower()
            == "all"
        )
    except Exception:
        return False


class LLMComposer:
    """Per-task LLM-driven Composer. Construct with the same model/api config the
    agent uses (the Composer Protocol's ctx does not carry api credentials)."""

    def __init__(
        self,
        *,
        model_name: str,
        api_base: str | None,
        api_key: str | None,
        timeout: int = 120,
        call_kwargs: dict | None = None,
    ):
        self._model_name = model_name
        self._api_base = api_base
        self._api_key = api_key
        self._timeout = timeout
        # Extra kwargs forwarded to the pick call. The pick is a *classification*
        # (read the catalog, emit one line of JSON) — it does not need a reasoning
        # budget, and on a reasoning model it silently becomes the dominant cost:
        # On a measured large-context model, the real 9.4k-char prompt took >900s
        # with thinking on (blowing the 120s timeout on EVERY task, so every trial
        # fell back to the all-`baseline` bundle and the evolved variants were never
        # reachable) versus 1.8s with `chat_template_kwargs={"enable_thinking": False}`.
        # Left empty by default: the flag is chat-template specific, so it is the
        # caller's job to pass what its endpoint understands.
        self._call_kwargs = call_kwargs or {}

    async def choose(
        self,
        instruction: str,
        library: list[ModuleInfo],
        stats: ModuleStatsView,
        ctx: ModuleCtx,
    ) -> ModuleBundle:
        try:
            return await self._choose(
                instruction,
                library,
                _archive_skip(ctx),
                None if _scope_is_all(ctx) else _locked_type(ctx),
                _pinned(ctx, library),
            )
        except Exception as exc:  # a composer must always return a usable bundle
            # The pin survives the fallback: it is the tree's foundation, not a
            # per-task choice, so degrading to DEFAULT_BUNDLE here would silently
            # swap the whole agent out from under the lineage.
            import dataclasses

            fallback = DEFAULT_BUNDLE
            try:
                pins = _pinned(ctx, library)
                if pins:
                    fallback = dataclasses.replace(
                        DEFAULT_BUNDLE,
                        **{t: ModuleSpec(name=n) for t, n in pins.items()},
                    )
            except Exception:
                pass
            _logger.warning("LLMComposer failed (%s); using default bundle", exc)
            return fallback

    async def _choose(
        self,
        instruction: str,
        library: list[ModuleInfo],
        skip: frozenset[str] = frozenset(),
        locked_type: str | None = None,
        pinned: dict[str, str] | None = None,
    ) -> ModuleBundle:
        from harbor.agents.terminus_2_modular import niche as _niche

        # One active representative per niche cell (S6): caps the LLM's choices to
        # behaviorally-distinct variants and drops near-duplicates, so the per-task
        # composer context stays bounded no matter how big the library grows.
        # Editor-only variants (audience=editor) are excluded. Variants with an
        # undeclared (empty) niche are NOT deduped — each is kept.
        by_type: dict[str, list[ModuleInfo]] = {t: [] for t in (*_TYPES, _HELPER_TYPE)}
        seen_cells: dict[str, set] = {t: set() for t in (*_TYPES, _HELPER_TYPE)}
        for info in library:
            if info.type not in by_type:
                continue
            if (info.type, info.name) in _NON_SOLVER_IMPLS or not (
                _niche.is_solver_selectable(info.niche)
            ):
                continue
            if f"{info.type}/{info.name}" in skip:
                continue  # superseded / excluded per archive.json (F2)
            key = _niche.niche_key(info.type, info.niche)
            if key and key in seen_cells[info.type]:
                continue
            if key:
                seen_cells[info.type].add(key)
            by_type[info.type].append(info)

        # Per-type baseline = the library default (DEFAULT_BUNDLE name if present,
        # else the sole/first registered impl). Used as-is for single-impl types
        # and as the fallback if the LLM doesn't pick (or picks invalidly).
        base: dict[str, str] = {}
        for t in _TYPES:
            names = [i.name for i in by_type[t]]
            default_name = getattr(DEFAULT_BUNDLE, t).name
            if default_name in names:
                base[t] = default_name
            elif names:
                # The declared default is missing from the library. Falling back to
                # an arbitrary variant here USED to be silent, which made a
                # half-finished rename (or an old gen tree whose files still use
                # pre-`baseline.py` names) look healthy while the solver quietly ran
                # a different module — e.g. context_mgmt degrading to `passthrough`
                # because it sorts first. Still degrade rather than abort a run, but
                # make it impossible to miss in the log.
                base[t] = names[0]
                _logger.error(
                    "composer: DEFAULT_BUNDLE wants %s=%r but the library only has "
                    "%r — falling back to %r. The modules tree is out of sync with "
                    "the package defaults (old gen tree, or an incomplete rename).",
                    t,
                    default_name,
                    names,
                    base[t],
                )
            else:
                base[t] = default_name  # 0 impls (shouldn't happen post-smoke)

        # Pinned types (active_bundle.json): the tree's fixed foundation. They
        # override the library default AND drop out of the pick-one prompt below.
        pinned = dict(pinned or {})
        if locked_type and locked_type in pinned:
            # Pinning the type the experiment is supposed to VARY would freeze the
            # experiment into a no-op. Loud, and the lock wins — but `online_evo`
            # preflight rejects this config before a run ever gets here.
            _logger.error(
                "active_bundle.json pins %r, which is also the LOCKED type — the "
                "experiment would vary nothing. Ignoring that pin.",
                locked_type,
            )
            pinned.pop(locked_type)
        base.update(pinned)

        # Ablation lock, scope="locked" (the caller passes locked_type=None when
        # scope="all"): only the locked type may be LLM-picked among its variants;
        # every other type stays on its `base` default even if it happens to have
        # >1 variant in the library (e.g. tools). This keeps the non-locked modules
        # deterministic across the K rolls of a task so the same-task pass/fail
        # contrast isolates the locked module. A SERIAL lineage turns this OFF
        # (scope="all") — it inherits another lineage's tree, and freezing those
        # types to `baseline` would make the inheritance decorative.
        # `_TYPES` only — tool_helper shares this dict but is subset-selected, not
        # picked one-of, so it must never reach the pick-one prompt.
        multi = {
            t: lst
            for t, lst in by_type.items()
            if t in _TYPES
            and len(lst) > 1
            and t not in pinned
            and (locked_type is None or t == locked_type)
        }
        # Solver helpers are NOT pick-one: a task gets a SUBSET, all at once (you
        # want grep AND read_file, not either/or). They ride on the tools spec's
        # params rather than becoming a 6th ModuleBundle field — that field list is
        # what `bundle_config.VALID_MODULE_TYPES` derives from, so adding one there
        # would make `active_bundle.json` validation reject the new key.
        #
        # Selection is biased to INCLUSION: the task starts with every active
        # helper in hand and the LLM may only take some AWAY. So a missing answer,
        # a malformed one, or a failed call all keep the full set — the only
        # failure direction that cannot silently bench a good tool. (Benching good
        # variants is a measured failure mode here: across 20 generations the
        # per-task picker fell back to baseline on ~85% of held-out tasks and the
        # variants that did get picked won 4-0.)
        helper_infos = by_type[_HELPER_TYPE]
        helpers = [i.name for i in helper_infos]

        chosen = dict(base)
        if multi or helper_infos:
            picked, dropped = await self._llm_pick(
                instruction, multi, base, helper_infos
            )
            for t, name in (picked or {}).items():
                if t in multi and any(i.name == name for i in multi[t]):
                    chosen[t] = name
            if dropped:
                helpers = [n for n in helpers if n not in dropped]

        return ModuleBundle(
            agent_loop=ModuleSpec(name=chosen["agent_loop"]),
            observation=ModuleSpec(name=chosen["observation"]),
            context_mgmt=ModuleSpec(name=chosen["context_mgmt"]),
            tools=ModuleSpec(name=chosen["tools"], params={"helpers": helpers}),
            verification=ModuleSpec(name=chosen["verification"]),
        )

    async def _llm_pick(
        self,
        instruction: str,
        multi: dict[str, list[ModuleInfo]],
        base: dict[str, str],
        helper_infos: list[ModuleInfo] | None = None,
    ) -> tuple[dict[str, str] | None, set[str]]:
        """One call decides both halves of the composition: which variant for each
        multi-impl module type (pick-one), and which solver helpers to LEAVE OUT
        (subset). Returns ``(picks, dropped_helper_names)``; on any failure the
        picks are None and nothing is dropped, i.e. defaults + all helpers."""
        from harbor.llms.lite_llm import LiteLLM

        helper_infos = helper_infos or []

        lines: list[str] = []
        for t, lst in multi.items():
            lines.append(f"\n## {t}  (default: {base.get(t)})")
            for i in lst:
                lines.append(f"- {i.name}: {i.description}")
        catalog = "\n".join(lines) or "(nothing to choose — module types are fixed)"

        task = instruction.strip()
        if len(task) > _MAX_INSTRUCTION_CHARS:
            task = task[:_MAX_INSTRUCTION_CHARS] + " …[truncated]"

        # The helper half is omitted entirely when there are none, so a run with an
        # empty tool_helper library sees the exact prompt it saw before.
        helper_block = ""
        if helper_infos:
            hl = "\n".join(f"- {i.name}: {i.description}" for i in helper_infos)
            helper_block = (
                "\n# Extra commands the agent will have\n"
                "These are ALL given to the agent by default — they are extra "
                "actions, not alternatives, so several can be useful at once.\n"
                f"{hl}\n\n"
                'Optionally add "exclude_helpers": ["<name>", …] to leave some OUT. '
                "Leave it empty unless a command is clearly irrelevant to THIS "
                "task; a command the agent never types costs it nothing, while "
                "removing a useful one costs it the capability.\n"
            )

        prompt = (
            "You assemble the module bundle for a terminal coding agent about to "
            "attempt the task below. For EACH module type listed, pick the SINGLE "
            "implementation whose description best fits THIS task. Match the variant "
            "to what the task actually needs; when in doubt prefer the default. "
            "There is no reward signal — judge from the descriptions and the task.\n\n"
            f"# Task\n{task}\n\n"
            f"# Choices per module type\n{catalog}\n"
            f"{helper_block}\n"
            "Answer with ONLY a JSON object mapping each type to the chosen impl "
            'name, e.g. {"observation": "scrollback_terminal"}. Include only the '
            "types listed."
        )
        try:
            llm = LiteLLM(
                model_name=self._model_name,
                api_base=self._api_base,
                api_key=self._api_key,
            )
            resp = await asyncio.wait_for(
                llm.call(prompt=prompt, **self._call_kwargs), timeout=self._timeout
            )
        except Exception as exc:
            # Log the exception *type*: the most common failure here is
            # `asyncio.TimeoutError`, whose `str()` is the empty string — logging
            # only `exc` printed a bare "LLM call failed: " that read like a
            # transport blip while it was in fact the timeout, silently benching
            # every variant behind the default bundle.
            _logger.warning(
                "LLMComposer LLM call failed (%s, timeout=%ss): %s",
                type(exc).__name__,
                self._timeout,
                exc,
            )
            return None, set()

        known = {i.name for i in helper_infos}
        for blob in (resp.content or "", getattr(resp, "reasoning_content", "") or ""):
            obj = _extract_json(blob)
            if obj is None:
                continue
            raw_drop = obj.pop("exclude_helpers", None)
            dropped = set()
            if isinstance(raw_drop, list):
                # Only names we actually offered — a hallucinated one must not
                # silently remove nothing while looking like it did something.
                dropped = {_coerce_name(v) for v in raw_drop} & known
            return {str(k): _coerce_name(v) for k, v in obj.items()}, dropped
        return None, set()


def _coerce_name(value) -> str:
    if isinstance(value, dict) and "name" in value:
        return str(value["name"])
    return str(value)


def _extract_json(text: str) -> dict | None:
    """First balanced ``{...}`` JSON object in free text, ignoring braces inside
    string literals."""
    if not text:
        return None
    start = text.find("{")
    while start != -1:
        depth = 0
        in_str = False
        esc = False
        for j in range(start, len(text)):
            c = text[j]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
                continue
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start : j + 1])
                    except Exception:
                        break
                    return obj if isinstance(obj, dict) else None
        start = text.find("{", start + 1)
    return None
