#!/usr/bin/env bash
# Run the first controlled CARLA smoke experiment on WSL.
# The CARLA server must already be running.
#
#     source scripts/wsl/env.sh
#     bash scripts/wsl/run_controlled_smoke.sh sudden_stop 0
#
# Arguments:
#   1. scenario: sudden_stop | cut_in | route_stop | random_traffic
#   2. seed: first seed for the fixed one-episode smoke run
#
# Environment overrides:
#   TOWN=Town10HD TIMEOUT=90 NO_RENDER=1 CONFIGS="agent/configs/k4_workspace.json ..."

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$SCRIPT_DIR/env.sh"
BENCH_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

: "${CARLA_ROOT:?source scripts/wsl/env.sh first}"

LOCAL_CARLA_ROOT="$BENCH_ROOT/CARLA_Latest"
if [ ! -d "$CARLA_ROOT/PythonAPI/carla/agents" ] && [ -d "$LOCAL_CARLA_ROOT/PythonAPI/carla/agents" ]; then
    echo "[info] using repo-local CARLA PythonAPI at $LOCAL_CARLA_ROOT"
    export CARLA_ROOT="$LOCAL_CARLA_ROOT"
    export PYTHONPATH="$CARLA_ROOT/PythonAPI/carla:$SCENARIO_RUNNER_ROOT:$LEADERBOARD_ROOT:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
fi

SCENARIO="${1:-sudden_stop}"
SEED="${2:-0}"
TOWN="${TOWN:-Town10HD}"
TIMEOUT="${TIMEOUT:-90}"
NO_RENDER="${NO_RENDER:-1}"
CONFIGS="${CONFIGS:-agent/configs/k4_workspace.json agent/configs/k4_random.json agent/configs/unbounded.json agent/configs/blind.json}"
PYTHON_BIN="${PYTHON_BIN:-}"
if [ -n "$PYTHON_BIN" ]; then
    :
elif [ -x "$VENV_ROOT/bin/python" ]; then
    PYTHON_BIN="$VENV_ROOT/bin/python"
else
    PYTHON_BIN="python"
fi

case "$SCENARIO" in
    sudden_stop|cut_in|route_stop|random_traffic) ;;
    *) echo "scenario must be one of: sudden_stop, cut_in, route_stop, random_traffic" >&2; exit 2 ;;
esac

if ! "$PYTHON_BIN" -c "import carla; from agents.navigation.basic_agent import BasicAgent" >/dev/null 2>&1; then
    echo "active Python cannot import the CARLA client stack." >&2
    echo "Tried: $PYTHON_BIN" >&2
    echo "VENV_ROOT: ${VENV_ROOT:-unset}" >&2
    if [ -n "${VENV_ROOT:-}" ] && [ ! -x "$VENV_ROOT/bin/python" ]; then
        echo "No executable Python at: $VENV_ROOT/bin/python" >&2
    fi
    if [ -n "${CARLA_ROOT:-}" ] && [ ! -d "$CARLA_ROOT/PythonAPI/carla/agents" ]; then
        echo "No BasicAgent package at: $CARLA_ROOT/PythonAPI/carla/agents" >&2
        echo "CARLA_ROOT must point at the extracted CARLA package, not only a pip-installed carla client." >&2
    fi
    echo "Run: source scripts/wsl/env.sh" >&2
    echo "If env.sh reports a missing venv, run: bash scripts/wsl/01_setup.sh --carla=$CARLA_SOURCE" >&2
    echo "If your CARLA venv is elsewhere, run with: PYTHON_BIN=/path/to/venv/bin/python bash scripts/wsl/run_controlled_smoke.sh $SCENARIO $SEED" >&2
    exit 1
fi

RESULTS="$BENCH_ROOT/results"
mkdir -p "$RESULTS"

cd "$BENCH_ROOT"
for config in $CONFIGS; do
    case "$config" in /*) config_path="$config" ;; *) config_path="$BENCH_ROOT/$config" ;; esac
    [ -f "$config_path" ] || { echo "no such config: $config_path" >&2; exit 1; }

    echo
    echo "================ $SCENARIO / $config ================"
    args=(
        "$BENCH_ROOT/runner/carla_runner.py"
        --config "$config_path"
        --port "$CARLA_PORT"
        --tm-port "$TRAFFIC_MANAGER_PORT"
        --town "$TOWN"
        --scenario "$SCENARIO"
        --seeds 1
        --first-seed "$SEED"
        --vehicles 0
        --walkers 0
        --timeout "$TIMEOUT"
    )
    if [ "$NO_RENDER" != "0" ]; then
        args+=(--no-render)
    fi
    "$PYTHON_BIN" "${args[@]}"
done

echo
"$PYTHON_BIN" "$BENCH_ROOT/verify/analyze_ablation.py" --results-dir "$RESULTS"
