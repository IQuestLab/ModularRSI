"""Module library — auto-discovers modules from `modules/<type>/*.py`.

Discovery rules:
- For each subdirectory of `modules/` (other than `__pycache__` and dunder
  names), recursively import every `.py` file (excluding `__init__.py`).
- If the imported module defines a `register(library)` function, call it.
- Errors importing one module are logged but do not abort the whole scan;
  the bad module is simply unavailable in the library.

This means **editor-spawned modules are picked up automatically** — no need
to edit `library.py` to register a new file. The Kernel boundary stays small.

Two loading modes:

- **Package mode (default)**: scan the installed `harbor.agents.terminus_2_modular.modules`
  package via `importlib.import_module`. Used when running the agent normally.

- **Path mode**: when `modules_root: Path` is given (e.g.,
  `self_evo_runs/gen_3/modules/`), each module file is loaded via
  `importlib.util.spec_from_file_location` so a *different* gen's modules can
  be loaded without affecting the installed package.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from harbor.agents.terminus_2_modular.protocols import (
    ModuleInfo,
    ModuleSpec,
)

_logger = logging.getLogger(__name__)


# A factory takes params dict, returns module instance.
_Factory = Callable[[dict[str, Any]], Any]


@dataclass
class ModuleLibrary:
    # type ("agent_loop" / "observation" / ...) -> {name -> (factory, info)}
    _registry: dict[str, dict[str, tuple[_Factory, ModuleInfo]]] = field(
        default_factory=dict
    )

    def register(
        self,
        type_: str,
        name: str,
        factory: _Factory,
        description: str,
        params_schema: dict[str, Any] | None = None,
        niche: dict[str, str] | None = None,
    ) -> None:
        info = ModuleInfo(
            name=name,
            type=type_,
            description=description,
            params_schema=params_schema or {},
            niche=niche or {},
        )
        self._registry.setdefault(type_, {})[name] = (factory, info)

    def instantiate(self, type_: str, spec: ModuleSpec) -> Any:
        if type_ not in self._registry:
            raise KeyError(f"Unknown module type: {type_}")
        if spec.name not in self._registry[type_]:
            raise KeyError(
                f"Unknown {type_} module: '{spec.name}'. "
                f"Available: {sorted(self._registry[type_])}"
            )
        factory, _ = self._registry[type_][spec.name]
        return factory(spec.params)

    def list_infos(self) -> list[ModuleInfo]:
        out: list[ModuleInfo] = []
        for type_modules in self._registry.values():
            for _, info in type_modules.values():
                out.append(info)
        return out

    def has(self, type_: str, name: str) -> bool:
        return type_ in self._registry and name in self._registry[type_]


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def _should_skip_dir(name: str) -> bool:
    return name.startswith("_") or name.startswith(".") or name == "__pycache__"


def _should_skip_file(name: str) -> bool:
    return not name.endswith(".py") or name.startswith("_")


def _discover_package_mode(lib: ModuleLibrary, pkg_name: str, pkg_path: Path) -> None:
    """Discover modules under an installed Python package."""
    for type_dir in sorted(pkg_path.iterdir()):
        if not type_dir.is_dir() or _should_skip_dir(type_dir.name):
            continue
        for mod_file in sorted(type_dir.glob("*.py")):
            if _should_skip_file(mod_file.name):
                continue
            full_name = f"{pkg_name}.{type_dir.name}.{mod_file.stem}"
            try:
                mod = importlib.import_module(full_name)
            except Exception as exc:
                _logger.warning("Failed to import %s: %s", full_name, exc)
                continue
            register_fn = getattr(mod, "register", None)
            if callable(register_fn):
                try:
                    register_fn(lib)
                except Exception as exc:
                    _logger.warning("register() failed for %s: %s", full_name, exc)


def _discover_path_mode(lib: ModuleLibrary, modules_root: Path) -> None:
    """Discover modules from a filesystem path (e.g., a gen_N/modules/ dir).

    Each file is loaded via importlib.util with a unique synthetic module name
    so loading the same file twice (e.g., from two gens) doesn't collide.
    """
    modules_root = modules_root.resolve()
    if not modules_root.is_dir():
        raise FileNotFoundError(f"modules_root does not exist: {modules_root}")

    # Make sure protocols / utils etc. are findable even when loading from path.
    # We rely on the installed harbor package for those.

    counter = 0
    for type_dir in sorted(modules_root.iterdir()):
        if not type_dir.is_dir() or _should_skip_dir(type_dir.name):
            continue
        for mod_file in sorted(type_dir.glob("*.py")):
            if _should_skip_file(mod_file.name):
                continue
            counter += 1
            synthetic_name = (
                f"_evo_mod_{modules_root.name}_{type_dir.name}_"
                f"{mod_file.stem}_{counter}"
            )
            spec = importlib.util.spec_from_file_location(synthetic_name, mod_file)
            if spec is None or spec.loader is None:
                _logger.warning("Could not build spec for %s", mod_file)
                continue
            mod = importlib.util.module_from_spec(spec)
            sys.modules[synthetic_name] = mod
            try:
                spec.loader.exec_module(mod)
            except Exception as exc:
                _logger.warning("Failed to load %s: %s", mod_file, exc)
                sys.modules.pop(synthetic_name, None)
                continue
            register_fn = getattr(mod, "register", None)
            if callable(register_fn):
                try:
                    register_fn(lib)
                except Exception as exc:
                    _logger.warning("register() failed for %s: %s", mod_file, exc)


def build_default_library(modules_root: Path | str | None = None) -> ModuleLibrary:
    """Build a ModuleLibrary by auto-discovering modules.

    Args:
        modules_root: If None, load from the installed
            `harbor.agents.terminus_2_modular.modules` package.
            If a Path/str, load from that filesystem directory instead
            (used by evo loop to test a candidate gen).
    """
    lib = ModuleLibrary()

    if modules_root is None:
        from harbor.agents.terminus_2_modular import modules as _modules_pkg

        pkg_name = _modules_pkg.__name__
        pkg_path = Path(_modules_pkg.__file__).parent  # type: ignore[arg-type]
        _discover_package_mode(lib, pkg_name, pkg_path)
    else:
        _discover_path_mode(lib, Path(modules_root))

    return lib
