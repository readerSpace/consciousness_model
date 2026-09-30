#!/usr/bin/env bash
# Preflight BEFORE downloading anything. Run this first:
#     bash scripts/wsl/00_check.sh
# It answers the only questions that can sink the WSL2 route: is this really
# WSL2, does the GPU reach it, does Vulkan work, and is there room for the
# package. Nothing here writes outside /tmp.

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$SCRIPT_DIR/config.sh"

fails=0
warns=0
pass() { printf 'PASS  %-26s %s\n' "$1" "${2:-}"; }
warn() { printf 'WARN  %-26s %s\n' "$1" "${2:-}"; warns=$((warns+1)); }
fail() { printf 'FAIL  %-26s %s\n' "$1" "${2:-}"; fails=$((fails+1)); }

# --- WSL ---------------------------------------------------------------------
if grep -qi microsoft /proc/version 2>/dev/null; then
    if [ -d /run/WSL ] || [ -e /dev/dxg ]; then
        pass "WSL2" "$(grep -o 'WSL[0-9]*' /proc/version | head -1) $(uname -r)"
    else
        fail "WSL2" "looks like WSL1; run 'wsl --set-version <distro> 2' in Windows"
    fi
else
    warn "WSL2" "not running under WSL (that is fine on a native Linux box)"
fi
pass "distro" "$( (. /etc/os-release && echo "$PRETTY_NAME") 2>/dev/null || uname -a)"

# --- GPU ---------------------------------------------------------------------
# In WSL the driver comes from Windows via /usr/lib/wsl/lib. Never install
# NVIDIA drivers inside the distro; that breaks the passthrough.
if command -v nvidia-smi >/dev/null 2>&1; then
    gpu="$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null | head -1)"
    if [ -n "$gpu" ]; then pass "nvidia-smi" "$gpu"; else fail "nvidia-smi" "present but returned nothing"; fi
else
    fail "nvidia-smi" "no GPU visible. Update the NVIDIA driver on the WINDOWS side, not in WSL."
fi

# --- Vulkan ------------------------------------------------------------------
# CARLA's Unreal build renders through Vulkan; without a working ICD the server
# starts and then dies with no useful message.
if command -v vulkaninfo >/dev/null 2>&1; then
    if vulkaninfo --summary >/tmp/vulkaninfo.txt 2>&1; then
        pass "vulkan" "$(grep -m1 'deviceName' /tmp/vulkaninfo.txt | sed 's/^ *//')"
    else
        fail "vulkan" "vulkaninfo failed; see /tmp/vulkaninfo.txt and the ICD note below"
    fi
else
    warn "vulkan" "vulkan-tools not installed yet (01_setup.sh installs it)"
fi
if [ ! -f /usr/share/vulkan/icd.d/nvidia_icd.json ] && [ ! -f /etc/vulkan/icd.d/nvidia_icd.json ]; then
    warn "vulkan ICD" "no nvidia_icd.json; if vulkaninfo fails, run 01_setup.sh --write-vulkan-icd"
fi

# --- disk --------------------------------------------------------------------
# The package downloads compressed and extracts to several times its size.
avail_kb="$(df -Pk "$HOME" | awk 'NR==2 {print $4}')"
avail_gb=$((avail_kb / 1024 / 1024))
if [ "$avail_gb" -ge 60 ]; then
    pass "disk under \$HOME" "${avail_gb} GB free"
else
    fail "disk under \$HOME" "${avail_gb} GB free; allow at least 60 GB for download plus extraction"
fi

# --- toolchain ---------------------------------------------------------------
for tool in curl tar xz; do
    if command -v "$tool" >/dev/null 2>&1; then pass "$tool" "$(command -v "$tool")";
    else warn "$tool" "missing (01_setup.sh installs it)"; fi
done
if command -v uv >/dev/null 2>&1 || [ -x "$HOME/.local/bin/uv" ]; then
    pass "uv" "present"
else
    warn "uv" "not installed; 01_setup.sh fetches it to provide Python $PYTHON_VERSION"
fi

echo
echo "$fails failed, $warns warnings"
if [ "$fails" -gt 0 ]; then
    cat <<'NOTE'

Vulkan ICD note (the usual WSL failure): if vulkaninfo fails but nvidia-smi
works, the loader has no NVIDIA ICD entry. Create one with

  sudo mkdir -p /usr/share/vulkan/icd.d
  sudo tee /usr/share/vulkan/icd.d/nvidia_icd.json >/dev/null <<'JSON'
  {"file_format_version":"1.0.0","ICD":{"library_path":"libGLX_nvidia.so.0","api_version":"1.3.194"}}
JSON

or let 01_setup.sh do it with --write-vulkan-icd.
NOTE
    exit 1
fi
exit 0
