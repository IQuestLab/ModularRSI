"""K-roll execution with one frozen module bundle per task encounter.

The contrastive contract is same-task *and* same-code.  A normal Harbor trial
constructs its own agent process, so letting every trial use ``LLMComposer``
silently violates that contract: each roll may receive a different bundle.

This module keeps the first roll's per-task choice, writes it into a private
modules snapshot, and runs the remaining rolls through ``StaticComposer``.
The generation's own ``active_bundle.json`` is never mutated; it is a
lineage-level pin and is shared by concurrently running tasks.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path

from harbor.agents.terminus_2_modular.self_evo.task_runner import HarborTaskResult


_MODULE_TYPES = (
    "agent_loop",
    "observation",
    "context_mgmt",
    "tools",
    "verification",
)
_HELPER_TYPE = "tool_helper"


@dataclass(frozen=True)
class BundleManifest:
    """The complete dynamic choice made by today's ``LLMComposer``.

    The composer currently emits names for the five pick-one module types and
    one dynamic param: ``tools.params.helpers``.  The trajectory records that
    helper subset as ``tool_helper``.  If the composer gains more dynamic
    params, this schema must be extended before K-roll remains same-code.
    """

    modules: dict[str, str]
    tool_helpers: tuple[str, ...] = ()

    def as_dict(self) -> dict:
        return {
            **{name: self.modules[name] for name in _MODULE_TYPES},
            _HELPER_TYPE: list(self.tool_helpers),
        }

    @property
    def bundle_id(self) -> str:
        raw = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(raw).hexdigest()

    def as_active_bundle(self) -> dict:
        active = {name: {"name": self.modules[name]} for name in _MODULE_TYPES}
        active["tools"]["params"] = {"helpers": list(self.tool_helpers)}
        return active


@dataclass(frozen=True)
class FrozenBundleSnapshot:
    """Private load root containing the first roll's frozen bundle."""

    load_root: Path
    snapshot_root: Path
    manifest_path: Path


@dataclass
class KRollGroupResult:
    """Results and audit state for one task encounter's K rolls."""

    rolls: list[tuple[HarborTaskResult, Path]] = field(default_factory=list)
    manifest: BundleManifest | None = None
    snapshot: FrozenBundleSnapshot | None = None
    invalid_reason: str | None = None

    @property
    def valid(self) -> bool:
        return self.invalid_reason is None

    @property
    def bundle_id(self) -> str | None:
        return self.manifest.bundle_id if self.manifest is not None else None


# roll index, modules/package load root, composer name, output directory
RollRunner = Callable[[int, Path, str, Path], Awaitable[HarborTaskResult]]


def extract_bundle_manifest(output_dir: Path) -> BundleManifest | None:
    """Read the newest ATIF trajectory under one Harbor run directory."""

    trajectories = list(Path(output_dir).glob("*/**/agent/trajectory.json"))
    if not trajectories:
        return None
    trajectory_path = max(trajectories, key=lambda path: path.stat().st_mtime_ns)
    try:
        trajectory = json.loads(trajectory_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    raw = ((trajectory.get("agent") or {}).get("extra") or {}).get("bundle")
    if not isinstance(raw, dict):
        return None

    modules: dict[str, str] = {}
    for module_type in _MODULE_TYPES:
        value = raw.get(module_type)
        if not isinstance(value, str) or not value:
            return None
        modules[module_type] = value

    helpers = raw.get(_HELPER_TYPE, [])
    if not isinstance(helpers, list) or not all(
        isinstance(name, str) and name for name in helpers
    ):
        return None
    return BundleManifest(modules=modules, tool_helpers=tuple(sorted(helpers)))


def _source_layout(source_root: Path) -> tuple[str, Path]:
    """Return (layout, copy source) for a generation/staging/load root."""

    source_root = Path(source_root).resolve()
    if (source_root / "protocols.py").is_file() and (source_root / "modules").is_dir():
        return "package", source_root
    if (source_root / "modules").is_dir():
        return "modules", source_root / "modules"
    if source_root.is_dir():
        return "modules", source_root
    raise ValueError(f"K-roll source root does not exist: {source_root}")


def prepare_frozen_bundle_snapshot(
    source_root: Path,
    group_root: Path,
    manifest: BundleManifest,
) -> FrozenBundleSnapshot:
    """Copy the captured generation and add a private full bundle pin.

    ``mkdtemp`` deliberately creates a new snapshot on a resumed/partial task;
    no possibly stale code is overwritten or reused.  Generation trees are
    small, and retaining the snapshot gives the evolution log an auditable copy
    of the exact code run by rolls 1..K-1.
    """

    group_root = Path(group_root)
    group_root.mkdir(parents=True, exist_ok=True)
    snapshot_root = Path(
        tempfile.mkdtemp(
            prefix=f"bundle_snapshot_{manifest.bundle_id[:12]}_",
            dir=group_root,
        )
    )
    layout, copy_source = _source_layout(source_root)
    if layout == "package":
        load_root = snapshot_root / "package"
        shutil.copytree(copy_source, load_root)
        modules_root = load_root / "modules"
    else:
        modules_root = snapshot_root / "modules"
        shutil.copytree(copy_source, modules_root)
        load_root = modules_root

    active_path = modules_root / "active_bundle.json"
    active_path.write_text(
        json.dumps(manifest.as_active_bundle(), indent=2, sort_keys=True) + "\n"
    )
    manifest_path = snapshot_root / "bundle_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "bundle_id": manifest.bundle_id,
                "bundle": manifest.as_dict(),
                "source_root": str(Path(source_root).resolve()),
                "layout": layout,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return FrozenBundleSnapshot(
        load_root=load_root,
        snapshot_root=snapshot_root,
        manifest_path=manifest_path,
    )


async def run_same_bundle_k_rolls(
    *,
    attempts: int,
    source_root: Path,
    group_root: Path,
    run_roll: RollRunner,
) -> KRollGroupResult:
    """Run one task K times while invoking the dynamic composer at most once."""

    attempts = max(1, attempts)
    group_root = Path(group_root)
    first_output = group_root / "r0"
    first = await run_roll(0, Path(source_root), "llm_dynamic", first_output)
    result = KRollGroupResult(rolls=[(first, first_output)])

    manifest = extract_bundle_manifest(first_output)
    result.manifest = manifest
    if attempts == 1:
        # K=1 has no cross-roll attribution requirement. Preserve legacy
        # behavior even when the trial crashed before writing a trajectory.
        return result
    if manifest is None:
        result.invalid_reason = "roll 0 produced no complete bundle manifest"
        return result

    snapshot = prepare_frozen_bundle_snapshot(source_root, group_root, manifest)
    result.snapshot = snapshot
    for roll_idx in range(1, attempts):
        output_dir = group_root / f"r{roll_idx}"
        harbor_result = await run_roll(
            roll_idx, snapshot.load_root, "static", output_dir
        )
        result.rolls.append((harbor_result, output_dir))
        observed = extract_bundle_manifest(output_dir)
        if observed != manifest:
            result.invalid_reason = (
                f"roll {roll_idx} bundle mismatch: expected {manifest.bundle_id}, "
                f"observed {observed.bundle_id if observed else 'missing'}"
            )
            # Finish no more rolls: once the invariant is broken, additional
            # trajectories cannot make this contrastive group usable.
            break
    return result
