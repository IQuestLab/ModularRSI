"""Smoke tests for evo loop — run after editor commits, before promotion.

Three tiers, fast → slow:

1. **AST kernel check** — parse the staging modules and confirm that no
   Kernel-protected paths were introduced (defense in depth — editor tooling
   already rejects these, but verify at the staging level).

2. **Import test** — `importlib.import_module` each module file, fail on
   ImportError / SyntaxError. Catches obviously-broken Python.

3. **Library load** — build a `ModuleLibrary` from the staging path; confirm
   that all five module types still have at least one registered module.

Runtime crash checks are handled separately by the Phase0 sanity battery.
"""

from __future__ import annotations

import ast
import importlib.util
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

_logger = logging.getLogger(__name__)


REQUIRED_MODULE_TYPES = (
    "agent_loop",
    "observation",
    "context_mgmt",
    "tools",
    "verification",
)

# Deliberately NOT in REQUIRED_MODULE_TYPES: gen_0 ships zero solver helpers (the
# agent starts with exactly terminus-2's action surface), so an empty
# `modules/tool_helper/` is the healthy state, not a missing module type.
_HELPER_TYPE = "tool_helper"


@dataclass
class SmokeReport:
    passed: bool
    failures: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def fail(self, msg: str) -> None:
        self.failures.append(msg)
        self.passed = False

    def note(self, msg: str) -> None:
        self.notes.append(msg)


# ---------------------------------------------------------------------------
# Test 1: Kernel AST check (no Kernel files in modules/ dir, all module
# files have a `register` function)
# ---------------------------------------------------------------------------


def check_ast(staging_modules_dir: Path) -> SmokeReport:
    """AST sanity checks on the staging modules tree.

    - The staging modules dir should NOT contain any of the Kernel files
      (protocols.py / services.py / agent.py / library.py / composer/).
      Editor tooling already blocks these; this is defense in depth.
    - Every non-`__init__.py` module file should parse with `ast.parse`.
    - Every module file should define a top-level `register` function.
    """
    report = SmokeReport(passed=True)
    staging_modules_dir = Path(staging_modules_dir).resolve()
    if not staging_modules_dir.is_dir():
        report.fail(f"staging modules dir missing: {staging_modules_dir}")
        return report

    off_limits_filenames = {
        "protocols.py",
        "services.py",
        "agent.py",
        "library.py",
    }

    for type_dir in sorted(staging_modules_dir.iterdir()):
        if not type_dir.is_dir() or type_dir.name.startswith("_"):
            continue
        # No Kernel-like dirs at module-type level
        if type_dir.name == "composer":
            report.fail("`composer/` directory must not exist under modules/")
            continue
        for mod_file in sorted(type_dir.glob("*.py")):
            if mod_file.name in off_limits_filenames:
                report.fail(
                    f"Kernel filename leaked into modules/: "
                    f"{mod_file.relative_to(staging_modules_dir)}"
                )
                continue
            if mod_file.name.startswith("_") or mod_file.name == "__init__.py":
                continue
            try:
                tree = ast.parse(mod_file.read_text(), filename=str(mod_file))
            except SyntaxError as exc:
                report.fail(
                    f"SyntaxError in {mod_file.relative_to(staging_modules_dir)}: {exc}"
                )
                continue
            has_register = any(
                isinstance(node, ast.FunctionDef) and node.name == "register"
                for node in tree.body
            )
            if not has_register:
                report.note(
                    f"no top-level `register` function in "
                    f"{mod_file.relative_to(staging_modules_dir)} "
                    f"(harmless if file is a helper, but will not be picked "
                    f"up by library auto-discovery)"
                )

    return report


# ---------------------------------------------------------------------------
# Test 2: Import test (load each module file via spec)
# ---------------------------------------------------------------------------


def check_imports(staging_modules_dir: Path) -> SmokeReport:
    """Try to import every module file in staging. Reports first failure.

    LIMITATION (TODO): each file is imported in isolation under a synthetic
    name. If the editor changes module A AND module B where A imports B
    via `from harbor.agents.terminus_2_modular.modules.X.B import ...`,
    A's import will resolve to the INSTALLED B (not the staged B). This
    catches SyntaxError / per-file ImportError but NOT cross-module API
    breakage. To fully test, would need to run the whole library in a
    subprocess with PYTHONPATH pointing at staging — defer for now.
    """
    report = SmokeReport(passed=True)
    staging_modules_dir = Path(staging_modules_dir).resolve()
    if not staging_modules_dir.is_dir():
        report.fail(f"staging modules dir missing: {staging_modules_dir}")
        return report

    counter = 0
    for type_dir in sorted(staging_modules_dir.iterdir()):
        if not type_dir.is_dir() or type_dir.name.startswith("_"):
            continue
        for mod_file in sorted(type_dir.glob("*.py")):
            if mod_file.name.startswith("_") or mod_file.name == "__init__.py":
                continue
            counter += 1
            synthetic_name = f"_evo_smoke_{type_dir.name}_{mod_file.stem}_{counter}"
            spec = importlib.util.spec_from_file_location(synthetic_name, mod_file)
            if spec is None or spec.loader is None:
                report.fail(f"could not build spec for {mod_file}")
                continue
            mod = importlib.util.module_from_spec(spec)
            sys.modules[synthetic_name] = mod
            try:
                spec.loader.exec_module(mod)
            except Exception as exc:
                report.fail(
                    f"import failed for "
                    f"{mod_file.relative_to(staging_modules_dir)}: "
                    f"{type(exc).__name__}: {exc}"
                )
                sys.modules.pop(synthetic_name, None)
                continue
            # Clean up so we don't leak synthetic modules
            sys.modules.pop(synthetic_name, None)

    return report


# ---------------------------------------------------------------------------
# Test 3: Library build (auto-discovery must yield at least one of each type)
# ---------------------------------------------------------------------------


def check_library_load(staging_modules_dir: Path) -> SmokeReport:
    """Use the production library loader on the staging modules dir.

    Verifies that all 5 module types have at least one module each — if
    editor accidentally deleted all `tools/` modules, this catches it.
    """
    report = SmokeReport(passed=True)

    # Local import to avoid loading harbor at top-of-file
    from harbor.agents.terminus_2_modular.library import build_default_library

    try:
        lib = build_default_library(modules_root=staging_modules_dir)
    except Exception as exc:
        report.fail(f"build_default_library failed: {type(exc).__name__}: {exc}")
        return report

    by_type: dict[str, list[str]] = {}
    for info in lib.list_infos():
        by_type.setdefault(info.type, []).append(info.name)

    for required in REQUIRED_MODULE_TYPES:
        if not by_type.get(required):
            report.fail(
                f"no modules of type '{required}' registered after auto-discovery"
            )

    summary = ", ".join(f"{t}={len(by_type.get(t, []))}" for t in REQUIRED_MODULE_TYPES)
    report.note(f"module counts per type: {summary}")
    return report


# ---------------------------------------------------------------------------
# Test 3.2: Discovery contract — every file the editor can create must
# actually LAND somewhere (0715_0040 postmortem: a promoted gen carried a
# 0-byte verification/requirements_gate.py that auto-discovery silently
# skipped — the editor "created" a gate that never existed).
#
# Rules (all cheap, all deterministic):
# - No empty (whitespace-only) non-underscore .py anywhere under a type dir.
# - Every non-underscore .py DIRECTLY under a type dir must define register()
#   and, when probed against a fresh library, register >= 1 implementation
#   (shared code belongs in an underscore-prefixed file — discovery skips it).
# - Every non-underscore .py under <type>/helpers/ must satisfy the helper
#   contract (`NAME: str` + callable `run`) or `_discover_helper_tools` will
#   silently drop it.
# ---------------------------------------------------------------------------


def _probe_register(mod_file: Path, synthetic_name: str):
    """Import one module file in isolation and return (module, error)."""
    spec = importlib.util.spec_from_file_location(synthetic_name, mod_file)
    if spec is None or spec.loader is None:
        return None, "could not build import spec"
    mod = importlib.util.module_from_spec(spec)
    sys.modules[synthetic_name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"
    finally:
        sys.modules.pop(synthetic_name, None)
    return mod, None


def check_discovery_contract(staging_modules_dir: Path) -> SmokeReport:
    """Verify every editor-reachable file actually lands in discovery."""
    from harbor.agents.terminus_2_modular.library import ModuleLibrary

    report = SmokeReport(passed=True)
    staging_modules_dir = Path(staging_modules_dir).resolve()
    if not staging_modules_dir.is_dir():
        report.fail(f"staging modules dir missing: {staging_modules_dir}")
        return report

    counter = 0
    for type_dir in sorted(staging_modules_dir.iterdir()):
        if not type_dir.is_dir() or type_dir.name.startswith("_"):
            continue

        # Empty-file sweep (recursive; the silent-skip trap applies at any depth)
        empty_files: set[Path] = set()
        for f in sorted(type_dir.rglob("*.py")):
            if f.name == "__init__.py" or f.name.startswith("_"):
                continue
            try:
                if not f.read_text().strip():
                    empty_files.add(f)
                    report.fail(
                        f"{f.relative_to(staging_modules_dir)} is EMPTY — "
                        f"auto-discovery will silently skip it; the module/tool "
                        f"you intended does not exist"
                    )
            except Exception as exc:
                report.fail(f"cannot read {f.relative_to(staging_modules_dir)}: {exc}")

        # Depth-1 module files: must register >= 1 implementation
        for mod_file in sorted(type_dir.glob("*.py")):
            if (
                mod_file.name == "__init__.py"
                or mod_file.name.startswith("_")
                or mod_file in empty_files  # already reported as EMPTY
            ):
                continue
            rel = mod_file.relative_to(staging_modules_dir)
            counter += 1
            mod, err = _probe_register(mod_file, f"_evo_disc_{counter}_{mod_file.stem}")
            if err:
                report.fail(f"discovery probe: import failed for {rel}: {err}")
                continue
            register_fn = getattr(mod, "register", None)
            if not callable(register_fn):
                report.fail(
                    f"{rel} has no register() — auto-discovery will import it "
                    f"and register NOTHING. Module variants must register(); "
                    f"shared code belongs in an underscore-prefixed file."
                )
                continue
            probe_lib = ModuleLibrary()
            try:
                register_fn(probe_lib)
            except Exception as exc:
                report.fail(f"register() raised for {rel}: {type(exc).__name__}: {exc}")
                continue
            infos = probe_lib.list_infos()
            if not infos:
                report.fail(
                    f"{rel}: register() ran but registered 0 implementations — "
                    f"the variant this file declares does not exist in the library"
                )
                continue
            wrong_type = [i.name for i in infos if i.type != type_dir.name]
            if wrong_type:
                report.note(
                    f"{rel} registers under a DIFFERENT type than its directory "
                    f"({wrong_type}) — composer/bundle lookups by directory "
                    f"convention will not find it"
                )

        # A helper placed in the OLD location is invisible: module discovery is a
        # shallow glob over `<type>/*.py` and never descends into a subdirectory,
        # so such a file registers nothing and the solver never sees the tool.
        # This used to "work" only because the tools module scanned the dir itself.
        legacy_helpers_dir = type_dir / "helpers"
        if legacy_helpers_dir.is_dir() and any(
            f.name != "__init__.py" and not f.name.startswith("_")
            for f in legacy_helpers_dir.glob("*.py")
        ):
            report.fail(
                f"{legacy_helpers_dir.relative_to(staging_modules_dir)}/ is the OLD "
                f"helper location and is NOT discovered — module discovery only "
                f"globs `<type>/*.py`. Move solver helpers to "
                f"`modules/{_HELPER_TYPE}/<name>.py` and give each a NICHE + "
                f"register(library) like any other module file."
            )

        # Helper contract: modules/tool_helper/*.py. These also go through the
        # generic per-file checks above (register present, declared type matches
        # the directory); this adds the action-specific part of the contract.
        if type_dir.name == _HELPER_TYPE:
            for f in sorted(type_dir.glob("*.py")):
                if (
                    f.name == "__init__.py"
                    or f.name.startswith("_")
                    or f in empty_files  # already reported as EMPTY
                ):
                    continue
                rel = f.relative_to(staging_modules_dir)
                counter += 1
                mod, err = _probe_register(f, f"_evo_helper_{counter}_{f.stem}")
                if err:
                    report.fail(f"helper probe: import failed for {rel}: {err}")
                    continue
                name = getattr(mod, "NAME", None)
                run = getattr(mod, "run", None)
                if not (isinstance(name, str) and name and callable(run)):
                    report.fail(
                        f"{rel} violates the helper contract (needs NAME: str "
                        f"and callable run) — the tools module would never "
                        f"dispatch to it and the solver would never see it"
                    )
                    continue
                if not isinstance(getattr(mod, "NICHE", None), dict) or not getattr(
                    mod, "NICHE", None
                ):
                    report.fail(
                        f"{rel} has no NICHE — an unplaced helper cannot be "
                        f"deduped, superseded or rolled back, which is exactly "
                        f"the accumulate-forever failure mode the archive fixes"
                    )
                    continue
                if not getattr(mod, "USAGE", ""):
                    report.note(
                        f"{rel}: no USAGE string — the solver prompt will only "
                        f"show the bare command name"
                    )

    return report


# ---------------------------------------------------------------------------
# Static landmine audit catches failures that import-only checks cannot see:
#
#   a) undefined / use-before-assignment names inside function bodies
#      (ruff F821/F823). Import-level smoke can NOT see these: a function-local
#      `from pathlib import Path` placed AFTER the first use imports fine but
#      raises UnboundLocalError on every call; a missing `Step` import only
#      explodes when the branch runs.
#   b) calls to self.<method> that exists NOWHERE on the instance
#      (the editor "remembered" a helper by analogy — _append_summarization_step
#      — that baseline never had). Runtime-only AttributeError otherwise.
#
# Both passed L3 import checks and L5 sanity (composer never selected the
# variant during the battery), so gen_2 promoted a module that crashed 100% of
# the trials that DID select it.
# ---------------------------------------------------------------------------


def _self_attr_audit(py_file: Path, instance: object) -> list[str]:
    """Return self.<attr> names referenced in `py_file` that do not exist on
    `instance` and are never assigned in the file (so not created at runtime
    by __init__/run either)."""
    tree = ast.parse(py_file.read_text())
    refs: set[str] = set()
    assigned: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "self"
        ):
            refs.add(node.attr)
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if (
                    isinstance(t, ast.Attribute)
                    and isinstance(t.value, ast.Name)
                    and t.value.id == "self"
                ):
                    assigned.add(t.attr)
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            t = node.target
            if (
                isinstance(t, ast.Attribute)
                and isinstance(t.value, ast.Name)
                and t.value.id == "self"
            ):
                assigned.add(t.attr)
    return sorted(r for r in refs if r not in assigned and not hasattr(instance, r))


# Calls that parse MODEL-GENERATED text and raise on anything malformed. In this
# codebase their input is always the agent's own output (keystrokes, a JSON tool
# call), which is malformed regularly — an unbalanced quote in a heredoc is enough.
# `shlex.quote` is deliberately absent: it escapes, it does not parse.
_TEXT_PARSERS = {("shlex", "split"), ("json", "loads"), ("json", "load")}


def _unguarded_text_parsers(py_file: Path) -> list[tuple[int, str]]:
    """(lineno, "mod.fn") for each `_TEXT_PARSERS` call NOT inside a try.

    Why this is a hard fail rather than a note: it is invisible to every other
    gate. The file imports cleanly (so AST/import pass), the design is sound (so
    review passes), and whether it crashes depends on whether the sampled model
    output happens to contain an unbalanced quote — so the 4 fixed sanity tasks
    catch it only by luck. Observed: a promoted `tools` variant dropped the
    baseline's try around `shlex.split` and raised `ValueError: No closing
    quotation` out of the agent loop; the identical bug in the NEXT candidate was
    caught by sanity purely because that batch happened to trigger it.
    """
    try:
        tree = ast.parse(py_file.read_text())
    except Exception:
        return []  # check_ast owns syntax errors

    out: list[tuple[int, str]] = []

    class _V(ast.NodeVisitor):
        def __init__(self) -> None:
            self.depth = 0

        def visit_Try(self, node: ast.Try) -> None:
            self.depth += 1
            self.generic_visit(node)
            self.depth -= 1

        def visit_Call(self, node: ast.Call) -> None:
            fn = node.func
            if isinstance(fn, ast.Attribute):
                key = (getattr(fn.value, "id", None), fn.attr)
                if key in _TEXT_PARSERS and self.depth == 0:
                    out.append((node.lineno, f"{key[0]}.{key[1]}"))
            self.generic_visit(node)

    _V().visit(tree)
    return out


def check_static_contract(staging_modules_dir: Path) -> SmokeReport:
    """Static name/attribute audit over the whole staging tree."""
    import shutil
    import subprocess

    import os

    report = SmokeReport(passed=True)
    staging_modules_dir = Path(staging_modules_dir).resolve()

    # --- a) ruff F821 (undefined name) + F823 (local used before assignment)
    ruff = shutil.which("ruff")
    cmd = (
        [ruff] if ruff else [sys.executable, "-m", "ruff"]  # pip-installed fallback
    )
    try:
        proc = subprocess.run(
            cmd
            + [
                "check",
                "--select",
                "F821,F823",
                "--no-cache",
                "--output-format",
                "concise",
                str(staging_modules_dir),
            ],
            capture_output=True,
            text=True,
            timeout=60,
            env={**os.environ, "NO_COLOR": "1"},
        )
        if proc.returncode not in (0, 1):
            report.note(f"ruff audit unavailable (rc={proc.returncode}) — skipped")
        elif proc.returncode == 1:
            lines = [ln for ln in proc.stdout.splitlines() if ln.strip() and "F8" in ln]
            for ln in lines[:10]:
                # strip the absolute prefix for readable editor feedback
                report.fail(
                    "static name audit: "
                    + ln.replace(str(staging_modules_dir) + "/", "")
                    + " — this raises NameError/UnboundLocalError at RUNTIME "
                    "even though the file imports cleanly"
                )
    except Exception as exc:  # ruff missing entirely → degrade gracefully
        report.note(f"ruff audit skipped ({type(exc).__name__}: {exc})")

    # --- b) self.<attr> contract per registered variant
    try:
        from harbor.agents.terminus_2_modular.library import build_default_library
        from harbor.agents.terminus_2_modular.protocols import ModuleSpec

        lib = build_default_library(modules_root=str(staging_modules_dir))
        import inspect

        for info in lib.list_infos():
            try:
                inst = lib.instantiate(info.type, ModuleSpec(name=info.name))
            except Exception as exc:
                report.note(
                    f"{info.type}:{info.name} not instantiable with default "
                    f"params ({type(exc).__name__}: {exc}) — self-attr audit "
                    f"skipped for it"
                )
                continue
            try:
                src_file = Path(inspect.getfile(type(inst))).resolve()
            except Exception:
                continue
            if staging_modules_dir not in src_file.parents:
                continue  # re-exported installed class — not editor-owned
            missing = _self_attr_audit(src_file, inst)
            if missing:
                rel = src_file.relative_to(staging_modules_dir)
                report.fail(
                    f"{rel} ({info.type}:{info.name}) references "
                    f"self.{', self.'.join(missing)} which exist(s) NOWHERE on "
                    f"the instance and are never assigned in the file — "
                    f"AttributeError the first time that line runs. If you "
                    f"meant a helper from the parent class, check its REAL "
                    f"name in the source before calling it."
                )
    except Exception as exc:
        # library build failure is check_library_load's job to report
        report.note(f"self-attr audit skipped ({type(exc).__name__}: {exc})")

    # --- c) parsers of model-generated text must be guarded
    for f in sorted(staging_modules_dir.rglob("*.py")):
        if f.name.startswith("_"):
            continue
        for line, call in _unguarded_text_parsers(f):
            report.fail(
                f"{f.relative_to(staging_modules_dir)}:{line}: bare `{call}` — "
                f"it parses MODEL-GENERATED text, so malformed input raises at "
                f"RUNTIME and takes down the whole run. The baseline wraps this "
                f"call in try/except for exactly that reason; an override must "
                f"keep the guard. Wrap it and fall back to the unparsed path."
            )

    return report


# ---------------------------------------------------------------------------
# Test 3.5: Bundle-config validation (if editor wrote active_bundle.json,
# every overridden module name must exist in the auto-discovered library —
# otherwise the composer would silently fall back and the editor's "enable"
# would be a no-op, or worse, fail at runtime)
# ---------------------------------------------------------------------------


def check_bundle_config(staging_modules_dir: Path) -> SmokeReport:
    """Validate an editor-written active_bundle.json against the library.

    No file → pass (the composer just uses the default bundle). If the file
    exists, every overridden type must be a real module type and name an
    implementation that auto-discovery actually registered. This catches the
    case where the editor enables a module it never successfully created
    (typo, wrong type dir, missing register()), which would otherwise make the
    enable a silent no-op or crash the solver at instantiation time.
    """
    import json

    from harbor.agents.terminus_2_modular.composer.bundle_config import (
        BUNDLE_CONFIG_FILENAME,
        VALID_MODULE_TYPES,
    )
    from harbor.agents.terminus_2_modular.library import build_default_library

    report = SmokeReport(passed=True)
    staging_modules_dir = Path(staging_modules_dir).resolve()
    config_path = staging_modules_dir / BUNDLE_CONFIG_FILENAME
    if not config_path.exists():
        report.note(f"no {BUNDLE_CONFIG_FILENAME} (composer uses default bundle)")
        return report

    try:
        raw = json.loads(config_path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        report.fail(f"{BUNDLE_CONFIG_FILENAME} unreadable: {exc}")
        return report
    if not isinstance(raw, dict):
        report.fail(f"{BUNDLE_CONFIG_FILENAME} must be a JSON object")
        return report

    try:
        lib = build_default_library(modules_root=staging_modules_dir)
    except Exception as exc:
        report.fail(
            f"build_default_library failed (cannot validate bundle config): "
            f"{type(exc).__name__}: {exc}"
        )
        return report
    available: dict[str, set[str]] = {}
    for info in lib.list_infos():
        available.setdefault(info.type, set()).add(info.name)

    for type_name, spec_dict in raw.items():
        if type_name not in VALID_MODULE_TYPES:
            report.fail(f"{BUNDLE_CONFIG_FILENAME}: unknown module type '{type_name}'")
            continue
        if not isinstance(spec_dict, dict) or "name" not in spec_dict:
            report.fail(
                f"{BUNDLE_CONFIG_FILENAME}: entry for '{type_name}' must be an "
                f"object with a 'name'"
            )
            continue
        impl_name = spec_dict["name"]
        if impl_name not in available.get(type_name, set()):
            report.fail(
                f"{BUNDLE_CONFIG_FILENAME}: '{type_name}' names implementation "
                f"'{impl_name}', which is not registered (did the module file "
                f"register() under type '{type_name}'?)"
            )
            continue
        params = spec_dict.get("params", {})
        if not isinstance(params, dict):
            report.fail(
                f"{BUNDLE_CONFIG_FILENAME}: params for '{type_name}' must be an object"
            )

    if report.passed:
        report.note(f"{BUNDLE_CONFIG_FILENAME} valid: overrides {sorted(raw.keys())}")
    return report


# ---------------------------------------------------------------------------
# Combined runner
# ---------------------------------------------------------------------------


@dataclass
class FullSmokeReport:
    ast: SmokeReport
    imports: SmokeReport
    library: SmokeReport
    bundle_config: SmokeReport
    # Discovery contract: no empty files, module files must register,
    # helpers must satisfy NAME/run. None only for old pickled reports.
    discovery: SmokeReport | None = None
    # Static landmine audit: ruff F821/F823 + self-attribute contract.
    # Catches runtime-only NameError/UnboundLocalError/AttributeError that
    # import-level checks and an unlucky (non-selecting) sanity battery miss.
    static: SmokeReport | None = None

    @property
    def passed(self) -> bool:
        return (
            self.ast.passed
            and self.imports.passed
            and self.library.passed
            and self.bundle_config.passed
            and (self.discovery is None or self.discovery.passed)
            and (self.static is None or self.static.passed)
        )

    def all_failures(self) -> list[str]:
        out = []
        for r, label in (
            (self.ast, "ast"),
            (self.imports, "imports"),
            (self.library, "library"),
            (self.bundle_config, "bundle_config"),
            (self.discovery, "discovery"),
            (self.static, "static"),
        ):
            if r is not None:
                out.extend(f"[{label}] {f}" for f in r.failures)
        return out


def run_fast_smoke(staging_dir: Path) -> FullSmokeReport:
    """Run all fast smoke tests against a staged modules directory."""
    modules_dir = Path(staging_dir)
    return FullSmokeReport(
        ast=check_ast(modules_dir),
        imports=check_imports(modules_dir),
        library=check_library_load(modules_dir),
        bundle_config=check_bundle_config(modules_dir),
        discovery=check_discovery_contract(modules_dir),
        static=check_static_contract(modules_dir),
    )
