"""Niche (behavior descriptor) schema for MAP-Elites-style variant management.

Each module variant declares a ``NICHE = {axis: value}`` class attribute. Variants
that land in the SAME cell (same niche key) are near-duplicates — the archive keeps
one active representative per cell so the editor is forced to explore empty cells or
explicitly supersede an incumbent, instead of spawning look-alike variants.

Design choice **C (hybrid / open-vocabulary)**: the axes below are a *seed* schema,
not a closed enum. A variant MAY use a new axis or a new value not listed here —
``niche_key`` just uses whatever keys are present. ``SEED_AXES`` exists so the
reflection view and the dedup gate can show "here are the usual axes + empty cells",
without forbidding the editor from carving a genuinely new dimension.

No fitness is involved here (that lives on a separate branch): this module only
*organizes* the library by declared behavior — it never ranks variants.
"""

from __future__ import annotations

# Seed axes per module type. Order matters only for stable display / key building.
# Values listed are the ones observed so far; the editor may introduce more.
SEED_AXES: dict[str, dict[str, list[str]]] = {
    "observation": {
        "capture": ["incremental", "full", "hybrid"],
        "capacity": ["low", "100k", "500k"],
        "staleness": ["none", "detect-retry", "detect-suppress"],
    },
    "agent_loop": {
        "grounding": ["reactive", "diagnostic-first", "deployment-aware"],
        "drift": ["none", "periodic-reinject", "stuck-detection"],
        "parse": ["baseline", "robust-json"],
    },
    "verification": {
        "evidence": ["executed-ever", "executed-last-iter", "output-pattern"],
        "error_gate": ["none", "checked"],
    },
    "tools": {
        "transport": ["tmux", "env-exec"],
        "encoding": ["xml", "json", "both"],
    },
    "context_mgmt": {
        "strategy": [
            "passthrough",
            "qa-summarize",
            "sliding-window",
            "verbatim-truncate",
        ],
    },
}

# Variants tagged with this axis are NOT part of the solver's selectable grid
# (e.g. editor_file_tools is the editor's own toolset). The per-task solver
# composer should skip them; only the editor bundle uses them.
AUDIENCE_AXIS = "audience"


def niche_key(type_: str, niche: dict[str, str]) -> tuple[tuple[str, str], ...]:
    """Canonical, hashable key for a niche cell.

    Uses seed-axis order first (for readability), then any extra axes the variant
    declared, sorted. Tolerant of missing axes — a variant that omits an axis
    simply occupies a coarser cell. Empty niche -> empty key (all such variants
    collide, which is the correct signal: they are undescribed).
    """
    if not niche:
        return ()
    seed_order = list(SEED_AXES.get(type_, {}).keys())
    ordered = [a for a in seed_order if a in niche]
    extra = sorted(a for a in niche if a not in seed_order)
    return tuple((a, str(niche[a])) for a in (*ordered, *extra))


def is_solver_selectable(niche: dict[str, str]) -> bool:
    """False for editor-only tool variants (audience=editor)."""
    return niche.get(AUDIENCE_AXIS) != "editor"
