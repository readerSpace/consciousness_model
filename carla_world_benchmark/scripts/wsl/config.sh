# Shared settings for the WSL2 side. Edit, then source env.sh.
# Everything lives on the WSL ext4 filesystem, not /mnt/c: CARLA and the venv
# are unusably slow over the Windows drive mount.

CARLA_INSTALL_ROOT="${CARLA_INSTALL_ROOT:-$HOME/carla_leaderboard}"
DOWNLOAD_DIR="${DOWNLOAD_DIR:-$HOME/carla_downloads}"

# --- which CARLA to install --------------------------------------------------
# nightly     : Linux nightly build (0.9.16 dev) + AdditionalMaps, from the only
#               host CARLA currently documents. AdditionalMaps carries Town12 and
#               Town13, so the leaderboard routes run. Needs Python 3.10 and the
#               relaxed dependency set, because the 0.9.16 client has no wheel
#               below cp310.
# leaderboard : the official Leaderboard 2.0 package (0.9.14). As of 2026-09-05
#               its S3 object answers AllAccessDisabled, so this will fail; kept
#               because it is the combination leaderboard-2.0 was written for and
#               may come back.
# release0915 : CARLA 0.9.15 release + AdditionalMaps. Its CDN answers 403.
CARLA_SOURCE="${CARLA_SOURCE:-nightly}"

# The only package host carla-simulator/carla/Docs/download.md still points at.
B2_HOST="https://carla-releases.s3.us-east-005.backblazeb2.com"
NIGHTLY_URL="$B2_HOST/Linux/Dev/CARLA_Latest.tar.gz"
NIGHTLY_NAME="CARLA_Latest.tar.gz"
NIGHTLY_MAPS_URL="$B2_HOST/Linux/Dev/AdditionalMaps_Latest.tar.gz"
NIGHTLY_MAPS_NAME="AdditionalMaps_Latest.tar.gz"

LEADERBOARD_PACKAGE_URL="https://leaderboard-public-contents.s3.us-west-2.amazonaws.com/CARLA_Leaderboard_2.0.tar.xz"
LEADERBOARD_PACKAGE_NAME="CARLA_Leaderboard_2.0.tar.xz"

RELEASE0915_URL="https://tiny.carla.org/carla-0-9-15-linux"
RELEASE0915_NAME="CARLA_0.9.15.tar.gz"
RELEASE0915_MAPS_URL="https://tiny.carla.org/additional-maps-0-9-15-linux"
RELEASE0915_MAPS_NAME="AdditionalMaps_0.9.15.tar.gz"

# Client version and interpreter follow the server build.
#   0.9.14 -> cp37/cp38 wheels -> Python 3.8 with the pinned leaderboard stack.
#   0.9.16 -> cp310/311/312    -> Python 3.10 with the relaxed set.
# Both were installed and imported end to end before being written down here.
case "$CARLA_SOURCE" in
    nightly)
        PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
        CARLA_CLIENT_SPEC="${CARLA_CLIENT_SPEC:-carla==0.9.16}"
        REQUIREMENTS_MODE="${REQUIREMENTS_MODE:-relaxed}"
        ;;
    release0915)
        PYTHON_VERSION="${PYTHON_VERSION:-3.8}"
        CARLA_CLIENT_SPEC="${CARLA_CLIENT_SPEC:-carla==0.9.15}"
        REQUIREMENTS_MODE="${REQUIREMENTS_MODE:-pinned}"
        ;;
    *)
        PYTHON_VERSION="${PYTHON_VERSION:-3.8}"
        CARLA_CLIENT_SPEC="${CARLA_CLIENT_SPEC:-carla==0.9.14}"
        REQUIREMENTS_MODE="${REQUIREMENTS_MODE:-pinned}"
        ;;
esac
VENV_ROOT="${VENV_ROOT:-$HOME/.venvs/carla_py${PYTHON_VERSION/./}}"

CARLA_PORT="${CARLA_PORT:-2000}"
TRAFFIC_MANAGER_PORT="${TRAFFIC_MANAGER_PORT:-8000}"
CARLA_QUALITY="${CARLA_QUALITY:-Epic}"
