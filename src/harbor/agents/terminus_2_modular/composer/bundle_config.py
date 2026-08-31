"""Bundle-override loader — `<modules_root>/active_bundle.json`.

The Composer is fixed Kernel logic and stays off-limits to evolution. But the
*decision data* it uses — "which implementation of each module type to run" —
is dropped down to a file that travels WITH the modules tree.

Two readers, two meanings — both "this type is fixed for this tree":

- `StaticComposer`: the whole bundle, overridden per the file.
- `LLMComposer` (the solver default): a named type is **PINNED** — it keeps the
  named variant and is never offered to the per-task picker. This is what makes
  SERIAL evolution work: a lineage can stand on a previous lineage's winner
  (e.g. `agent_loop`) while evolving a different type on top. Without a pin the
  per-task picker falls back to `DEFAULT_BUNDLE`'s `baseline` for every
  non-locked type, so the inherited winner would silently never run.

NOT editor-writable: `editor_file_tools._write_allowed` refuses it with the lock
on OR off, same as `baseline.py`. It is the lineage's foundation, not something
a generation gets to redefine — an editor that could rewrite it could un-pin the
very ground its diagnoses were collected on.

Design:
- **Partial override**: start from the default bundle and override only the
  module types named in the JSON. The bundle can name just
  `{"agent_loop": {"name": "cautious"}}` instead of repeating all 5 types.
- **Validated**: every overridden type must be a real module type, and its
  `name` must exist in the auto-discovered library for that type. Any problem
  (missing file / bad JSON / unknown type / unknown name) → return None so the
  caller falls back to the default bundle. We never half-apply a bad config.
- **JSON shape**: `{ "<type>": {"name": "<impl>", "params": {...}}, ... }`
  (`params` optional).
"""

from __future__ import annotations

import dataclasses
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from harbor.agents.terminus_2_modular.protocols import (
    ModuleBundle,
    ModuleInfo,
    ModuleSpec,
)

BUNDLE_CONFIG_FILENAME = "active_bundle.json"

# The valid module-type names are exactly the fields of ModuleBundle.
VALID_MODULE_TYPES: frozenset[str] = frozenset(
    f.name for f in dataclasses.fields(ModuleBundle)
)


def _log(logger: Any | None, msg: str, *args: Any) -> None:
    if logger is not None:
        logger.warning(msg, *args)


def load_bundle_overrides(
    modules_root: Path | str | None,
    library: list[ModuleInfo],
    logger: Any | None = None,
) -> dict[str, ModuleSpec] | None:
    """The validated per-type overrides declared by `active_bundle.json`, or None
    if there is no config file / the config is invalid.

    Split out from `load_bundle_override` because callers need to know WHICH
    types the file names, not just the merged result: `LLMComposer` treats a
    named type as PINNED (frozen for this modules tree — never offered to the
    per-task picker), and "merged bundle equals default" cannot distinguish
    "not named" from "named, and happens to equal the default".
    """
    if modules_root is None:
        return None
    config_path = Path(modules_root) / BUNDLE_CONFIG_FILENAME
    if not config_path.exists():
        return None

    try:
        raw = json.loads(config_path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        _log(logger, "active_bundle.json unreadable (%s); using default bundle", exc)
        return None

    if not isinstance(raw, dict):
        _log(logger, "active_bundle.json must be a JSON object; using default bundle")
        return None

    # Index available implementations by type for existence checks.
    available: dict[str, set[str]] = defaultdict(set)
    for info in library:
        available[info.type].add(info.name)

    overrides: dict[str, ModuleSpec] = {}
    for type_name, spec_dict in raw.items():
        if type_name not in VALID_MODULE_TYPES:
            _log(
                logger,
                "active_bundle.json: unknown module type %r; using default bundle",
                type_name,
            )
            return None
        if not isinstance(spec_dict, dict) or "name" not in spec_dict:
            _log(
                logger,
                "active_bundle.json: entry for %r must be an object with a 'name'; "
                "using default bundle",
                type_name,
            )
            return None
        impl_name = spec_dict["name"]
        if impl_name not in available.get(type_name, set()):
            _log(
                logger,
                "active_bundle.json: %r has no implementation %r in the library; "
                "using default bundle",
                type_name,
                impl_name,
            )
            return None
        params = spec_dict.get("params", {})
        if not isinstance(params, dict):
            _log(
                logger,
                "active_bundle.json: params for %r must be an object; "
                "using default bundle",
                type_name,
            )
            return None
        overrides[type_name] = ModuleSpec(name=impl_name, params=params)

    if not overrides:
        # Empty config: nothing to override.
        return None

    return overrides


def load_bundle_override(
    modules_root: Path | str | None,
    library: list[ModuleInfo],
    default_bundle: ModuleBundle,
    logger: Any | None = None,
) -> ModuleBundle | None:
    """Load an editor-written bundle override, validated against `library`.

    Returns a ModuleBundle (default bundle with the JSON's types overridden) on
    success, or None if there is no config file or the config is invalid (the
    caller should then use `default_bundle`).
    """
    overrides = load_bundle_overrides(modules_root, library, logger)
    if not overrides:
        return None
    return dataclasses.replace(default_bundle, **overrides)
