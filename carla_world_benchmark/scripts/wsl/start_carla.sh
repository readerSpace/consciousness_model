#!/usr/bin/env bash
# Start the CARLA server. Leave this shell running.
#     bash scripts/wsl/start_carla.sh              off-screen (recommended in WSL)
#     bash scripts/wsl/start_carla.sh --windowed   needs WSLg; often flaky for UE4
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$SCRIPT_DIR/config.sh"

MODE="-RenderOffScreen"
for arg in "$@"; do
    case "$arg" in
        --windowed) MODE="-windowed -ResX=800 -ResY=600" ;;
        --port=*) CARLA_PORT="${arg#*=}" ;;
        --quality=*) CARLA_QUALITY="${arg#*=}" ;;
        *) echo "unknown option: $arg" >&2; exit 2 ;;
    esac
done

EXE="$CARLA_INSTALL_ROOT/CarlaUE4.sh"
[ -x "$EXE" ] || { echo "not found or not executable: $EXE" >&2; exit 1; }

# SDL has no display to open when rendering off screen; saying so up front avoids
# a confusing startup failure.
[ "$MODE" = "-RenderOffScreen" ] && export SDL_VIDEODRIVER=offscreen

echo "[carla] $EXE $MODE -carla-server -world-port=$CARLA_PORT -quality-level=$CARLA_QUALITY"
exec "$EXE" $MODE -carla-server -world-port="$CARLA_PORT" -quality-level="$CARLA_QUALITY" -nosound
