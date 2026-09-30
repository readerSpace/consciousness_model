#!/usr/bin/env bash
# Run one leaderboard evaluation with the consciousness agent.
#     source scripts/wsl/env.sh
#     bash scripts/wsl/run_leaderboard.sh agent/configs/k4_workspace.json [routes-subset]
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$SCRIPT_DIR/config.sh"
BENCH_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

: "${LEADERBOARD_ROOT:?source scripts/wsl/env.sh first}"

AGENT_CONFIG="${1:-agent/configs/k4_workspace.json}"
ROUTES_SUBSET="${2:-0}"
case "$AGENT_CONFIG" in /*) ;; *) AGENT_CONFIG="$BENCH_ROOT/$AGENT_CONFIG" ;; esac
[ -f "$AGENT_CONFIG" ] || { echo "no such config: $AGENT_CONFIG" >&2; exit 1; }

ROUTES="${ROUTES:-$LEADERBOARD_ROOT/data/routes_devtest.xml}"
TAG="$(basename "${AGENT_CONFIG%.json}")"
RESULTS="$BENCH_ROOT/results"
mkdir -p "$RESULTS"

cd "$BENCH_ROOT"
python "$LEADERBOARD_ROOT/leaderboard/leaderboard_evaluator.py" \
    --host=127.0.0.1 \
    --port="$CARLA_PORT" \
    --traffic-manager-port="$TRAFFIC_MANAGER_PORT" \
    --routes="$ROUTES" \
    --routes-subset="$ROUTES_SUBSET" \
    --repetitions="${REPETITIONS:-1}" \
    --track="${TRACK:-SENSORS}" \
    --checkpoint="$RESULTS/$TAG.leaderboard.json" \
    --agent="$BENCH_ROOT/agent/consciousness_agent.py" \
    --agent-config="$AGENT_CONFIG" \
    --debug="${DEBUG_CHALLENGE:-0}"

echo "leaderboard result -> $RESULTS/$TAG.leaderboard.json"
