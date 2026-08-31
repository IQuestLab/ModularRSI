#!/usr/bin/env bash
# Public entry point for modular self-evolution and fixed-generation evaluation.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HARBOR_ROOT="$(cd "$HERE/.." && pwd)"

usage() {
    cat <<'EOF'
Usage:
  scripts/self_evolve.sh evolve <module>
  scripts/self_evolve.sh evaluate [modules-root] [task-name]

Modules: observation | tools | context_mgmt | agent_loop | verification

Configuration is read from environment variables and, when present, the
repository-root .env file. Set DRY_RUN=true to validate and print the command
without launching work.
EOF
}

die() {
    echo "error: $*" >&2
    exit 2
}

ACTION="${1:-}"
case "$ACTION" in
    evolve|evaluate) shift ;;
    -h|--help|help|"") usage; exit 0 ;;
    *) usage >&2; die "unknown action '$ACTION'" ;;
esac

if [[ -f "$HARBOR_ROOT/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "$HARBOR_ROOT/.env"
    set +a
fi

MODEL="${MODEL:-}"
[[ -n "$MODEL" ]] || die "MODEL is required"

API_KEY="${HARBOR_EVO_API_KEY:-${OPENAI_API_KEY:-}}"
[[ -n "$API_KEY" && "$API_KEY" != "change-me" ]] || \
    die "set HARBOR_EVO_API_KEY or OPENAI_API_KEY"
export HARBOR_EVO_API_KEY="$API_KEY"
export OPENAI_API_KEY="$API_KEY"

DEFAULT_MODEL_INFO='{"max_input_tokens":176000,"max_output_tokens":16000,"input_cost_per_token":0.0,"output_cost_per_token":0.0}'
export HARBOR_MODEL_INFO="${HARBOR_MODEL_INFO:-$DEFAULT_MODEL_INFO}"
export PYTHONDONTWRITEBYTECODE=1

if [[ -z "${PYTHON_BIN:-}" ]]; then
    if [[ -x "$HARBOR_ROOT/.venv/bin/python" ]]; then
        PYTHON_BIN="$HARBOR_ROOT/.venv/bin/python"
    else
        PYTHON_BIN="$(command -v python3 || true)"
    fi
fi
[[ -n "${PYTHON_BIN:-}" && -x "$PYTHON_BIN" ]] || \
    die "Python is unavailable; install this release first"
PYTHON_BIN="$(cd "$(dirname "$PYTHON_BIN")" && pwd)/$(basename "$PYTHON_BIN")"
PYTHON_DIR="$(dirname "$PYTHON_BIN")"
HARBOR_BIN="$PYTHON_DIR/harbor"
export PATH="$PYTHON_DIR:$PATH"

EXPECTED_IMPORT="$HARBOR_ROOT/src/harbor/__init__.py"
ACTUAL_IMPORT="$($PYTHON_BIN -c 'from pathlib import Path; import harbor; print(Path(harbor.__file__).resolve())')"
[[ "$ACTUAL_IMPORT" == "$EXPECTED_IMPORT" && -x "$HARBOR_BIN" ]] || \
    die "selected Python is not installed from this release tree"

$PYTHON_BIN - <<'PY'
import json
import os

value = json.loads(os.environ["HARBOR_MODEL_INFO"])
for key in ("max_input_tokens", "max_output_tokens"):
    if not isinstance(value.get(key), int) or value[key] <= 0:
        raise SystemExit(f"HARBOR_MODEL_INFO.{key} must be a positive integer")
PY

ENVIRONMENT="${ENVIRONMENT:-docker}"
case "$ENVIRONMENT" in docker|e2b) ;; *) die "ENVIRONMENT must be docker or e2b" ;; esac
if [[ "$ENVIRONMENT" == "e2b" && -z "${E2B_API_KEY:-${EVO_E2B_ACCOUNTS:-}}" ]]; then
    die "E2B_API_KEY or EVO_E2B_ACCOUNTS is required for ENVIRONMENT=e2b"
fi

API_BASE="${API_BASE:-${OPENAI_API_BASE:-}}"
if [[ -n "${REMOTE_DOCKER_HOST:-}" ]]; then
    export DOCKER_HOST="${DOCKER_HOST:-$REMOTE_DOCKER_HOST}"
fi

print_command() {
    printf 'command:'
    printf ' %q' "$@"
    printf '\n'
}

run_evolve() {
    local locked_module="${1:-${LOCKED_MODULE:-tools}}"
    case "$locked_module" in
        observation|tools|context_mgmt|agent_loop|verification) ;;
        *) die "invalid module '$locked_module'" ;;
    esac

    local dataset_root="${SUPPORT_DATASET_DIR:-}"
    [[ -d "$dataset_root/tasks" ]] || \
        die "SUPPORT_DATASET_DIR must contain tasks/"

    local profile="${PROFILE:-smoke}"
    case "$profile" in train|smoke) ;; *) die "PROFILE must be train or smoke" ;; esac

    local run_id="${RUN_ID:-$(date -u +%Y%m%d_%H%M%S)__${locked_module}}"
    local archive_root="${ARCHIVE_ROOT:-$HARBOR_ROOT/self_evo_runs/runs/$run_id}"
    local skills_dir="${SKILLS_DIR:-$HARBOR_ROOT/src/harbor/agents/terminus_2_modular/self_evo/editor_skills}"

    local args=(
        --archive-root "$archive_root"
        --support-dataset-dir "$dataset_root"
        --support-split "${SUPPORT_SPLIT:-train}"
        --locked-module "$locked_module"
        --model "$MODEL"
        --environment "$ENVIRONMENT"
        --profile "$profile"
        --max-lanes "${MAX_LANES:-2}"
        --attempts "${ATTEMPTS:-3}"
        --agent-timeout-multiplier "${AGENT_TIMEOUT_MULTIPLIER:-2}"
        --skills-dir "$skills_dir"
    )
    [[ -z "$API_BASE" ]] || args+=(--api-base "$API_BASE")
    [[ -z "${REFLECT_EVERY:-}" ]] || args+=(--reflect-every "$REFLECT_EVERY")
    [[ -z "${TASK_CONCURRENCY:-}" ]] || args+=(--task-concurrency "$TASK_CONCURRENCY")
    [[ -z "${EPOCHS:-}" ]] || args+=(--epochs "$EPOCHS")
    [[ -z "${MAX_TASKS:-}" ]] || args+=(--max-tasks "$MAX_TASKS")
    [[ -z "${TASK_SEED:-}" ]] || args+=(--task-seed "$TASK_SEED")

    echo "action       : evolve"
    echo "module       : $locked_module"
    echo "profile      : $profile"
    echo "dataset      : $dataset_root"
    echo "support split: ${SUPPORT_SPLIT:-train}"
    echo "archive      : $archive_root"
    echo "environment  : $ENVIRONMENT"
    echo "model        : $MODEL"

    if [[ "${DRY_RUN:-false}" == "true" ]]; then
        print_command "$PYTHON_BIN" -m \
            harbor.agents.terminus_2_modular.self_evo.phase0 "${args[@]}"
        return
    fi
    exec "$PYTHON_BIN" -m \
        harbor.agents.terminus_2_modular.self_evo.phase0 "${args[@]}"
}

run_evaluate() {
    local modules_root="${1:-${MODULES_ROOT:-$HARBOR_ROOT/generations/merged_active/gen_0/modules}}"
    local task_name="${2:-${TASK_NAME:-fix-git}}"
    [[ -d "$modules_root" ]] || die "modules root does not exist: $modules_root"
    modules_root="$(cd "$modules_root" && pwd)"

    local dataset="${DATASET:-terminal-bench@2.0}"
    local stamp
    stamp="$(date -u +%Y%m%d_%H%M%S)"
    local output_dir="${OUTPUT_DIR:-$HARBOR_ROOT/results/evaluation/$stamp}"
    local args=(
        run
        --yes
        -d "$dataset"
        -a terminus-2-modular
        -m "$MODEL"
        -e "$ENVIRONMENT"
        -n "${N_CONCURRENT:-1}"
        -k "${N_ATTEMPTS:-1}"
        -o "$output_dir"
        --ak modules_root="$modules_root"
        --ak max_turns="${MAX_TURNS:-200}"
        --ak model_info="$HARBOR_MODEL_INFO"
        --ak temperature="${SOLVER_TEMPERATURE:-0}"
        --ak locked_module_type="${LOCKED_MODULE:-tools}"
        --ak composer_scope="${COMPOSER_SCOPE:-all}"
        --ak composer_name="${COMPOSER_NAME:-llm_dynamic}"
    )
    [[ -z "$API_BASE" ]] || args+=(--ak api_base="$API_BASE")
    [[ -z "$task_name" ]] || args+=(-i "$task_name")

    echo "action       : evaluate"
    echo "modules      : $modules_root"
    echo "dataset/task : $dataset / $task_name"
    echo "output       : $output_dir"
    echo "environment  : $ENVIRONMENT"
    echo "model        : $MODEL"

    if [[ "${DRY_RUN:-false}" == "true" ]]; then
        print_command "$HARBOR_BIN" "${args[@]}"
        return
    fi
    exec "$HARBOR_BIN" "${args[@]}"
}

case "$ACTION" in
    evolve) run_evolve "${1:-}" ;;
    evaluate) run_evaluate "${1:-}" "${2:-}" ;;
esac
