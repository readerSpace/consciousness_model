#!/usr/bin/env bash
# Build the Leaderboard 2.0 environment inside WSL2.
#     bash scripts/wsl/01_setup.sh
#
#   --skip-apt              do not touch apt (no sudo prompt)
#   --skip-download         the package is already at $CARLA_INSTALL_ROOT
#   --write-vulkan-icd      install the NVIDIA ICD json that WSL usually lacks
#   --carla=nightly         Linux nightly (0.9.16 dev) + AdditionalMaps, from the
#                           only host CARLA still documents. Default, because the
#                           other two are currently refused by their servers.
#   --carla=leaderboard     official Leaderboard 2.0 package (0.9.14). Its bucket
#                           answers AllAccessDisabled as of 2026-09-05.
#   --carla=release0915     CARLA 0.9.15 release + AdditionalMaps. CDN answers 403.
#
# Safe to re-run: the download resumes, the venv and extraction are reused.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

SKIP_APT=0; SKIP_DOWNLOAD=0; WRITE_ICD=0
for arg in "$@"; do
    case "$arg" in
        --skip-apt) SKIP_APT=1 ;;
        --skip-download) SKIP_DOWNLOAD=1 ;;
        --write-vulkan-icd) WRITE_ICD=1 ;;
        --carla=*) export CARLA_SOURCE="${arg#*=}" ;;
        *) echo "unknown option: $arg" >&2; exit 2 ;;
    esac
done

. "$SCRIPT_DIR/config.sh"
case "$CARLA_SOURCE" in
    nightly|leaderboard|release0915) ;;
    *) echo "CARLA_SOURCE must be nightly, leaderboard or release0915; got '$CARLA_SOURCE'" >&2; exit 2 ;;
esac

say() { printf '\n[%s] %s\n' "$1" "$2"; }

# An anaconda install ahead of /usr/bin on PATH would otherwise supply curl, xz
# and tar for a multi-gigabyte resumable download and an archive of tens of GB.
# Prefer the distro's own binaries for those.
pick() { for candidate in "$@"; do [ -x "$candidate" ] && { echo "$candidate"; return; }; done; command -v "$(basename "$1")"; }
# Pre-set CURL or TAR in the environment to override (used by the stub tests).
CURL="${CURL:-$(pick /usr/bin/curl "$(command -v curl || true)")}"
TAR="${TAR:-$(pick /usr/bin/tar "$(command -v tar || true)")}"

# --- system libraries --------------------------------------------------------
# Package names differ between Ubuntu releases: 24.04's 64-bit time_t transition
# renamed libpng16-16 to libpng16-16t64, and libtiff5 became libtiff6. Install
# each requirement by trying its known names in turn, and never let one missing
# name abort the run.
apt_any() {
    local label="$1"; shift
    for candidate in "$@"; do
        if sudo apt-get install -y --no-install-recommends "$candidate" >/dev/null 2>&1; then
            echo "  + $label -> $candidate"
            return 0
        fi
    done
    echo "  - $label (none of: $* )  -- continuing; 02_check_binary.sh will say if CARLA misses it"
    return 0
}

if [ "$SKIP_APT" -eq 0 ]; then
    say apt "installing the shared libraries the CARLA binary links against"
    sudo apt-get update -qq
    apt_any "curl"        curl
    apt_any "xz"          xz-utils
    apt_any "ca-certs"    ca-certificates
    apt_any "openmp"      libomp5 libomp5-18 libomp5-17 libomp5-14
    apt_any "sdl2"        libsdl2-2.0-0
    apt_any "xdg"         xdg-user-dirs
    apt_any "vulkan"      libvulkan1
    apt_any "vulkan-tools" vulkan-tools
    apt_any "libpng"      libpng16-16t64 libpng16-16
    apt_any "libjpeg"     libjpeg-turbo8t64 libjpeg-turbo8
    apt_any "libtiff"     libtiff6 libtiff5
    apt_any "libgomp"     libgomp1
    apt_any "libglu"      libglu1-mesa
    CURL="${CURL_OVERRIDE:-$(pick /usr/bin/curl "$CURL")}"
    TAR="${TAR_OVERRIDE:-$(pick /usr/bin/tar "$TAR")}"
fi
echo "  using curl: $CURL"
echo "  using tar:  $TAR"

. "$SCRIPT_DIR/lib_download.sh"

if [ "$WRITE_ICD" -eq 1 ]; then
    say vulkan "writing /usr/share/vulkan/icd.d/nvidia_icd.json"
    sudo mkdir -p /usr/share/vulkan/icd.d
    printf '%s\n' \
      '{"file_format_version":"1.0.0","ICD":{"library_path":"libGLX_nvidia.so.0","api_version":"1.3.194"}}' \
      | sudo tee /usr/share/vulkan/icd.d/nvidia_icd.json >/dev/null
    vulkaninfo --summary >/dev/null 2>&1 && echo "  vulkan now works" || echo "  vulkaninfo still fails"
fi

# --- python ------------------------------------------------------------------
# uv provides CPython 3.8 without conda, deadsnakes or a compiler. The official
# leaderboard instructions use conda and Python 3.7; 3.8 is equivalent here
# because the carla client and every pinned dependency ship cp38 linux wheels.
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
    say uv "installing to \$HOME/.local/bin"
    "$CURL" -LsSf https://astral.sh/uv/install.sh | sh
fi

say python "creating $VENV_ROOT on CPython $PYTHON_VERSION"
uv python install "$PYTHON_VERSION"
mkdir -p "$(dirname "$VENV_ROOT")"
[ -x "$VENV_ROOT/bin/python" ] || uv venv --python "$PYTHON_VERSION" "$VENV_ROOT"

say pip "CARLA client ($CARLA_CLIENT_SPEC), $REQUIREMENTS_MODE dependency set"
uv pip install --python "$VENV_ROOT/bin/python" "$CARLA_CLIENT_SPEC"
if [ "$REQUIREMENTS_MODE" = relaxed ]; then
    # The upstream pins have no wheels for this interpreter; see the header of
    # requirements-leaderboard-py310.txt for what was verified.
    uv pip install --python "$VENV_ROOT/bin/python" \
        -r "$BENCH_ROOT/scripts/requirements-leaderboard-py310.txt"
else
    uv pip install --python "$VENV_ROOT/bin/python" -r "$BENCH_ROOT/external/scenario_runner/requirements.txt"
    uv pip install --python "$VENV_ROOT/bin/python" -r "$BENCH_ROOT/external/leaderboard/requirements.txt"
fi

# ElementTree.Element.getchildren() was removed in Python 3.9 and these branches
# still call it; without this no routes file parses.
say patches "adapting the vendored leaderboard checkouts to modern Python"
bash "$BENCH_ROOT/scripts/apply_patches.sh"

# --- CARLA package -----------------------------------------------------------
mkdir -p "$CARLA_INSTALL_ROOT" "$DOWNLOAD_DIR"

extract_into_root() {
    # extract_into_root <archive> <kind>
    local archive="$1" kind="$2" flag inner
    [ "$kind" = xz ] && flag=-J || flag=-z
    PATH="/usr/bin:/bin:$PATH" "$TAR" -x $flag -f "$archive" -C "$CARLA_INSTALL_ROOT"
    if [ ! -f "$CARLA_INSTALL_ROOT/CarlaUE4.sh" ]; then
        # Some packages carry a single top-level directory; flatten it.
        inner="$(find "$CARLA_INSTALL_ROOT" -maxdepth 2 -name CarlaUE4.sh -printf '%h\n' 2>/dev/null | head -1)"
        if [ -n "$inner" ] && [ "$inner" != "$CARLA_INSTALL_ROOT" ]; then
            echo "  found CarlaUE4.sh in $inner, moving its contents up"
            (shopt -s dotglob; mv "$inner"/* "$CARLA_INSTALL_ROOT"/)
        fi
    fi
}

import_maps() {
    # A packaged release takes extra content through its Import folder; if that
    # script is absent the archive is laid straight over the tree.
    local maps="$1"
    if [ -f "$CARLA_INSTALL_ROOT/ImportAssets.sh" ]; then
        mkdir -p "$CARLA_INSTALL_ROOT/Import"
        cp -n "$maps" "$CARLA_INSTALL_ROOT/Import/" || true
        (cd "$CARLA_INSTALL_ROOT" && bash ./ImportAssets.sh)
    else
        echo "  no ImportAssets.sh; extracting the maps into the root instead"
        PATH="/usr/bin:/bin:$PATH" "$TAR" -xzf "$maps" -C "$CARLA_INSTALL_ROOT"
    fi
}

if [ -f "$CARLA_INSTALL_ROOT/CarlaUE4.sh" ]; then
    say package "skipped, CarlaUE4.sh already present at $CARLA_INSTALL_ROOT"
elif [ "$CARLA_SOURCE" = nightly ]; then
    ARCHIVE="$DOWNLOAD_DIR/$NIGHTLY_NAME"
    MAPS="$DOWNLOAD_DIR/$NIGHTLY_MAPS_NAME"
    say download "CARLA Linux nightly build"
    if [ "$SKIP_DOWNLOAD" -eq 0 ]; then
        fetch_archive "$NIGHTLY_URL" "$ARCHIVE" gzip || exit 1
        say download "AdditionalMaps nightly (this is what carries Town12 / Town13)"
        fetch_archive "$NIGHTLY_MAPS_URL" "$MAPS" gzip || exit 1
    fi
    say extract "$ARCHIVE -> $CARLA_INSTALL_ROOT  (this takes a while)"
    extract_into_root "$ARCHIVE" gzip
    say maps "importing AdditionalMaps"
    import_maps "$MAPS"
elif [ "$CARLA_SOURCE" = leaderboard ]; then
    ARCHIVE="$DOWNLOAD_DIR/$LEADERBOARD_PACKAGE_NAME"
    say download "Leaderboard 2.0 package"
    if [ "$SKIP_DOWNLOAD" -eq 0 ]; then
        if ! fetch_archive "$LEADERBOARD_PACKAGE_URL" "$ARCHIVE" xz; then
            cat <<'MSG'

The leaderboard bucket did not serve the package. AllAccessDisabled is set on
the object itself, so retrying will not help. The Linux nightly build plus its
AdditionalMaps carry Town12 and Town13 as well, and it is the only package host
CARLA still documents, so the leaderboard routes still run:

    bash scripts/wsl/03_check_download.sh       # what each host actually serves
    bash scripts/wsl/01_setup.sh --carla=nightly

MSG
            exit 1
        fi
    fi
    say extract "$ARCHIVE -> $CARLA_INSTALL_ROOT  (this takes a while)"
    extract_into_root "$ARCHIVE" xz
else
    ARCHIVE="$DOWNLOAD_DIR/$RELEASE0915_NAME"
    MAPS="$DOWNLOAD_DIR/$RELEASE0915_MAPS_NAME"
    say download "CARLA 0.9.15 Linux release"
    if [ "$SKIP_DOWNLOAD" -eq 0 ]; then
        fetch_archive "$RELEASE0915_URL" "$ARCHIVE" gzip || exit 1
        say download "AdditionalMaps 0.9.15 (Town12 / Town13)"
        fetch_archive "$RELEASE0915_MAPS_URL" "$MAPS" gzip || exit 1
    fi
    say extract "$ARCHIVE -> $CARLA_INSTALL_ROOT  (this takes a while)"
    extract_into_root "$ARCHIVE" gzip
    say maps "importing AdditionalMaps"
    import_maps "$MAPS"
fi

[ -f "$CARLA_INSTALL_ROOT/CarlaUE4.sh" ] || { echo "CarlaUE4.sh not found under $CARLA_INSTALL_ROOT" >&2; exit 1; }
chmod +x "$CARLA_INSTALL_ROOT/CarlaUE4.sh" || true

say libraries "checking the extracted binary for missing shared objects"
bash "$SCRIPT_DIR/02_check_binary.sh" || true

echo
echo "Setup finished."
echo "  CARLA_ROOT   = $CARLA_INSTALL_ROOT"
echo "  CARLA_SOURCE = $CARLA_SOURCE"
echo "  venv         = $VENV_ROOT"
echo
echo "Next:"
echo "  source scripts/wsl/env.sh"
echo "  bash scripts/wsl/start_carla.sh          # leave running"
echo "  python verify/verify_setup.py            # in another shell"
echo "  bash scripts/wsl/run_ablation.sh"
