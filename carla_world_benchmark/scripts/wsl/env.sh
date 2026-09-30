# Source this in every shell:  source scripts/wsl/env.sh
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$SCRIPT_DIR/config.sh"

export BENCH_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export PROJECT_ROOT="$(cd "$BENCH_ROOT/.." && pwd)"
export CARLA_ROOT="$CARLA_INSTALL_ROOT"
export LEADERBOARD_ROOT="$BENCH_ROOT/external/leaderboard"
export SCENARIO_RUNNER_ROOT="$BENCH_ROOT/external/scenario_runner"
export CARLA_PORT

# PythonAPI/carla carries the agents package (BasicAgent, LocalPlanner). The
# carla client itself comes from the venv wheel, not from the .egg.
export PYTHONPATH="$CARLA_ROOT/PythonAPI/carla:$SCENARIO_RUNNER_ROOT:$LEADERBOARD_ROOT:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

if [ -f "$VENV_ROOT/bin/activate" ]; then
    . "$VENV_ROOT/bin/activate"
else
    echo "[warn] no venv at $VENV_ROOT (run scripts/wsl/01_setup.sh)"
fi

echo "CARLA_ROOT = $CARLA_ROOT"
echo "PYTHONPATH = $PYTHONPATH"

# An active anaconda base environment can win the PATH race and silently give
# every later command the wrong interpreter, which then cannot import carla.
resolved="$(command -v python || true)"
case "$resolved" in
    "$VENV_ROOT"/*) echo "python     = $resolved" ;;
    *) echo "[warn] python resolves to $resolved, not $VENV_ROOT/bin/python."
       echo "[warn] Another environment (anaconda?) is ahead on PATH. Try 'conda deactivate' first," 
       echo "[warn] or call $VENV_ROOT/bin/python explicitly." ;;
esac
