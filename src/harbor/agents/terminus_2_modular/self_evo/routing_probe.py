"""Measure whether normal composer routing would select a candidate.

The result is diagnostic rather than a promotion gate: an unselected candidate
may have a weak description rather than a weak implementation.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable

_logger = logging.getLogger(__name__)

VERDICT_PROVEN = "routing_proven"
VERDICT_UNPROVEN = "routing_unproven"
VERDICT_SKIPPED = "routing_skipped"

#: Where Harbor caches each task's `instruction.md`. The composer sees this text
#: and nothing else, so it is the only input the probe needs.
TASK_CACHE = Path("/root/.cache/harbor/tasks")

_REMEDY = (
    "No intended task selected this variant. Before concluding the variant is "
    "useless, rewrite its DESCRIPTION and re-probe: a description that names a "
    "task family, or states a runtime property instead of a selection criterion, "
    "gets benched regardless of what the code does."
)


@dataclass
class RoutingProbeResult:
    verdict: str
    target_variant: str = ""
    picked_by: list[str] = field(default_factory=list)
    picks: dict[str, str] = field(default_factory=dict)
    errors: int = 0
    reason: str = ""
    remedy: str = ""


def retired_quals(archive_root: Path | str) -> frozenset[str]:
    """``type/name`` the archive marks superseded/excluded.

    A real run filters these before the catalog reaches the composer prompt, so a
    probe that skips this step offers retired variants that soak up picks and its
    distribution stops matching the run it is supposed to model. Fail-open: a
    probe with a slightly wide candidate set is wrong, but a probe that refuses to
    run tells us nothing at all.
    """
    try:
        from harbor.agents.terminus_2_modular import archive as _archive

        return frozenset(
            f"{e.type}/{e.name}"
            for e in _archive.load_archive(archive_root)
            if e.status in ("superseded", "excluded")
        )
    except Exception as exc:
        _logger.warning("routing probe could not read the archive: %s", exc)
        return frozenset()


def load_instructions(tasks: list[str]) -> dict[str, str]:
    """Task name → instruction text, from Harbor's task cache."""
    found: dict[str, str] = {}
    wanted = set(tasks)
    try:
        for path in TASK_CACHE.glob("*/*/instruction.md"):
            name = path.parent.name
            if name in wanted and name not in found:
                found[name] = path.read_text(errors="ignore")
    except Exception as exc:
        _logger.warning("routing probe could not read the task cache: %s", exc)
    return found


async def probe_routing(
    *,
    target_variant: str,
    tasks: list[str],
    instructions: dict[str, str],
    choose: Callable[[str], Awaitable[str | None]],
    concurrency: int = 2,
) -> RoutingProbeResult:
    """Ask the composer, per intended task, which variant it would select."""
    usable = [(t, instructions[t]) for t in tasks if instructions.get(t)]
    if not usable:
        return RoutingProbeResult(
            verdict=VERDICT_SKIPPED,
            target_variant=target_variant,
            reason=(
                "no intended tasks"
                if not tasks
                else "no cached instruction for any intended task"
            ),
        )

    sem = asyncio.Semaphore(max(1, concurrency))
    picks: dict[str, str] = {}
    errors = 0

    async def _one(task: str, instruction: str) -> None:
        nonlocal errors
        async with sem:
            try:
                picked = await choose(instruction)
            except Exception as exc:
                errors += 1
                _logger.warning("routing probe failed for %s: %s", task, exc)
                return
        if picked:
            picks[task] = picked

    await asyncio.gather(*(_one(t, i) for t, i in usable))

    if not picks:
        # every call failed — that is a fact about the endpoint, not the variant
        return RoutingProbeResult(
            verdict=VERDICT_SKIPPED,
            target_variant=target_variant,
            errors=errors,
            reason="no task produced a composer choice",
        )

    picked_by = [t for t, _ in usable if picks.get(t) == target_variant]
    if picked_by:
        return RoutingProbeResult(
            verdict=VERDICT_PROVEN,
            target_variant=target_variant,
            picked_by=picked_by,
            picks=picks,
            errors=errors,
            reason=f"{len(picked_by)}/{len(usable)} intended task(s) selected it",
        )
    return RoutingProbeResult(
        verdict=VERDICT_UNPROVEN,
        target_variant=target_variant,
        picks=picks,
        errors=errors,
        reason=f"0/{len(usable)} intended task(s) selected it",
        remedy=_REMEDY,
    )


def make_composer_chooser(
    *,
    modules_root: Path | str,
    archive_root: Path | str,
    locked_type: str,
    model_name: str,
    api_base: str | None,
    api_key: str | None,
) -> Callable[[str], Awaitable[str | None]]:
    """A chooser backed by the same `LLMComposer._choose` a real roll uses.

    Kept out of :func:`probe_routing` so the verdict logic can be tested without
    an endpoint, and so the candidate-set construction (archive skips, locked
    type) lives in exactly one place.
    """
    from harbor.agents.terminus_2_modular.composer.llm_dynamic import LLMComposer
    from harbor.agents.terminus_2_modular.library import build_default_library

    library = build_default_library(modules_root=str(modules_root))
    infos = library.list_infos()
    skip = retired_quals(archive_root)
    composer = LLMComposer(model_name=model_name, api_base=api_base, api_key=api_key)

    async def _choose(instruction: str) -> str | None:
        bundle = await composer._choose(instruction, infos, skip, locked_type, {})
        chosen = getattr(bundle, locked_type, None)
        return getattr(chosen, "name", None)

    return _choose
