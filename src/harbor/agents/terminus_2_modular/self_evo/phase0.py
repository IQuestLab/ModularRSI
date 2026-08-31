"""Public entry point for the Phase0 contrastive two-lane workflow.

This module intentionally exposes one evolution algorithm. ``train`` matches
the latest Phase0 training configuration. ``smoke`` changes only workload
scale; all algorithmic choices and safety gates remain identical.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from harbor.agents.terminus_2_modular.self_evo.online_evo import run_online_evo

LOCKED_MODULES = (
    "agent_loop",
    "observation",
    "context_mgmt",
    "tools",
    "verification",
)


@dataclass(frozen=True)
class WorkloadProfile:
    reflect_every: int
    task_concurrency: int
    epochs: int
    max_tasks: int | None


PROFILES = {
    "train": WorkloadProfile(
        reflect_every=10,
        task_concurrency=6,
        epochs=3,
        max_tasks=None,
    ),
    "smoke": WorkloadProfile(
        reflect_every=2,
        task_concurrency=2,
        epochs=1,
        max_tasks=2,
    ),
}

# Algorithm and safety semantics from the latest Phase0 runner. These values
# are not CLI choices in the public release.
SOLVER_TEMPERATURE = 0.0
SOLVER_TIMEOUT_SEC = 3000
AGENT_TIMEOUT_MULTIPLIER = 2.0
MAX_EDITOR_TURNS = 200
MAX_SOLVER_TURNS = 200
EDITOR_TIMEOUT_SEC = 3600
EDITOR_CALL_TIMEOUT_SEC = 3600
REVIEW_MAX_TURNS = 60
MAX_GATE_REPAIRS = 2
E2B_PER_ACCOUNT_CAP = 18


def _support_tasks(
    dataset_root: Path, support_split: str = "train"
) -> tuple[Path, list[str]]:
    root = dataset_root.resolve()
    task_dir = root / "tasks"
    if not task_dir.is_dir():
        raise ValueError(f"support dataset has no tasks/ directory: {root}")

    excluded: set[str] = set()
    contamination = root / "PROBE_CONTAMINATION.json"
    if contamination.exists():
        try:
            payload = json.loads(contamination.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot read {contamination}: {exc}") from exc
        excluded = set(payload.get("exclude_task_names") or [])

    available = sorted(
        path.name
        for path in task_dir.iterdir()
        if path.is_dir() and path.name not in excluded
    )
    tasks = available
    manifest = root / "manifest.json"
    train_list = root / "train_120.txt"
    if support_split == "train" and manifest.exists():
        try:
            payload = json.loads(manifest.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot read {manifest}: {exc}") from exc
        rows = payload.get("tasks") or []
        tasks = [row["name"] for row in rows if row.get("split") == "train"]
    elif support_split == "train" and train_list.exists():
        tasks = [line.strip() for line in train_list.read_text().splitlines() if line.strip()]
    if support_split == "train" and (manifest.exists() or train_list.exists()):
        if len(tasks) != len(set(tasks)):
            raise ValueError(f"support task list contains duplicates: {root}")
        unavailable = sorted(set(tasks) - set(available))
        if unavailable:
            raise ValueError(
                f"support task list references unavailable tasks: {', '.join(unavailable)}"
            )
    if not tasks:
        raise ValueError(f"support dataset contains no usable tasks: {root}")
    return task_dir, tasks


def _e2b_accounts() -> list[tuple[str, str]] | None:
    raw = os.environ.get("EVO_E2B_ACCOUNTS", "").strip()
    if not raw:
        return None
    accounts: list[tuple[str, str]] = []
    for row in raw.split(";"):
        key, _, token = row.strip().partition("|")
        if key:
            accounts.append((key, token))
    return accounts or None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="phase0-evo",
        description="Run the contrastive evidence/portfolio two-lane workflow.",
    )
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--support-dataset-dir", type=Path, required=True)
    parser.add_argument("--support-split", choices=("train", "all"), default="train")
    parser.add_argument("--locked-module", choices=LOCKED_MODULES, required=True)
    parser.add_argument("--model", dest="model_name", required=True)
    parser.add_argument("--api-base", default=None)
    parser.add_argument(
        "--api-key",
        default=os.environ.get("HARBOR_EVO_API_KEY") or None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--environment", choices=("docker", "e2b"), default="docker")
    parser.add_argument("--profile", choices=tuple(PROFILES), default="train")
    parser.add_argument("--max-lanes", type=int, choices=(1, 2), default=2)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument(
        "--agent-timeout-multiplier",
        type=float,
        default=AGENT_TIMEOUT_MULTIPLIER,
        help="Scale each support task's native Harbor agent timeout.",
    )
    parser.add_argument("--reflect-every", type=int, default=None)
    parser.add_argument("--task-concurrency", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--task-seed", type=int, default=0)
    parser.add_argument(
        "--skills-dir",
        type=Path,
        default=Path(__file__).parent / "editor_skills",
    )
    return parser


def _positive(name: str, value: int) -> int:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    args = _build_parser().parse_args(argv)
    if not args.api_key:
        print("HARBOR_EVO_API_KEY is required", file=sys.stderr)
        return 2
    if args.environment == "e2b" and not (
        os.environ.get("E2B_API_KEY") or os.environ.get("EVO_E2B_ACCOUNTS")
    ):
        print(
            "ENVIRONMENT=e2b requires E2B_API_KEY or EVO_E2B_ACCOUNTS",
            file=sys.stderr,
        )
        return 2

    try:
        support_task_dir, tasks = _support_tasks(
            args.support_dataset_dir, args.support_split
        )
        profile = PROFILES[args.profile]
        reflect_every = _positive(
            "reflect_every", args.reflect_every or profile.reflect_every
        )
        task_concurrency = _positive(
            "task_concurrency", args.task_concurrency or profile.task_concurrency
        )
        epochs = _positive("epochs", args.epochs or profile.epochs)
        attempts = _positive("attempts", args.attempts)
        if args.agent_timeout_multiplier <= 0:
            raise ValueError("agent_timeout_multiplier must be positive")
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    max_tasks = args.max_tasks
    if max_tasks is None:
        max_tasks = profile.max_tasks
    if max_tasks is not None and max_tasks <= 0:
        print("max_tasks must be positive when provided", file=sys.stderr)
        return 2

    outcome = asyncio.run(
        run_online_evo(
            archive_root=args.archive_root,
            tasks=tasks,
            model_name=args.model_name,
            api_base=args.api_base,
            api_key=args.api_key,
            reflect_every=reflect_every,
            task_concurrency=task_concurrency,
            max_tasks=max_tasks,
            epochs=epochs,
            shuffle_tasks=True,
            task_seed=args.task_seed,
            max_editor_turns=MAX_EDITOR_TURNS,
            max_solver_turns=MAX_SOLVER_TURNS,
            solver_timeout_sec=SOLVER_TIMEOUT_SEC,
            agent_timeout_multiplier=args.agent_timeout_multiplier,
            editor_timeout_sec=EDITOR_TIMEOUT_SEC,
            editor_call_timeout_sec=EDITOR_CALL_TIMEOUT_SEC,
            skills_dir=args.skills_dir,
            review_max_turns=REVIEW_MAX_TURNS,
            sanity_tasks=None,
            environment=args.environment,
            max_gate_repairs=MAX_GATE_REPAIRS,
            support_task_dir=support_task_dir,
            e2b_accounts=_e2b_accounts(),
            e2b_per_account_cap=E2B_PER_ACCOUNT_CAP,
            solver_temperature=SOLVER_TEMPERATURE,
            locked_module_type=args.locked_module,
            composer_scope="locked",
            attempts=attempts,
            max_lanes=args.max_lanes,
            sanity_concurrency=6,
        )
    )
    print(outcome.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
