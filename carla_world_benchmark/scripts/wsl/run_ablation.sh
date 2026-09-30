#!/usr/bin/env bash
# Run every workspace configuration over the same route, then print the table.
#     source scripts/wsl/env.sh
#     bash scripts/wsl/run_ablation.sh [routes-subset]
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SUBSET="${1:-0}"

CONFIGS="${CONFIGS:-agent/configs/k4_workspace.json agent/configs/k1_workspace.json agent/configs/k4_random.json agent/configs/unbounded.json agent/configs/blind.json}"

for config in $CONFIGS; do
    echo
    echo "================ $config ================"
    bash "$SCRIPT_DIR/run_leaderboard.sh" "$config" "$SUBSET"
done

echo
python "$BENCH_ROOT/verify/analyze_ablation.py" --results-dir "$BENCH_ROOT/results"
