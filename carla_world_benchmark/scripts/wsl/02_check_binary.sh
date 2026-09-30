#!/usr/bin/env bash
# Report shared objects the extracted CARLA binary cannot resolve.
#     bash scripts/wsl/02_check_binary.sh
#
# The Leaderboard package was built for an older Ubuntu than 24.04, and the
# usual failure mode is a silent death at startup because one .so is missing.
# ldd answers that in a second instead of after a confusing crash.

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$SCRIPT_DIR/config.sh"

BIN="$(find "$CARLA_INSTALL_ROOT" -maxdepth 4 -type f -name 'CarlaUE4-Linux-Shipping' 2>/dev/null | head -1)"
if [ -z "$BIN" ]; then
    echo "no CarlaUE4-Linux-Shipping under $CARLA_INSTALL_ROOT (extract the package first)"
    exit 1
fi
echo "binary: $BIN"

missing="$(ldd "$BIN" 2>/dev/null | awk '/not found/ {print $1}' | sort -u)"
if [ -z "$missing" ]; then
    echo "PASS  every shared object resolves"
    exit 0
fi

echo "FAIL  missing shared objects:"
echo "$missing" | sed 's/^/        /'
echo
echo "Try the distro package first, for example:"
echo "$missing" | while read -r so; do
    base="${so%%.so*}"
    echo "        sudo apt-get install -y \$(apt-cache search --names-only \"^${base#lib}\" | awk '{print \$1}' | head -3 | tr '\\n' ' ')"
done
cat <<'NOTE'

If a name genuinely no longer exists on this Ubuntu (libtiff.so.5 is the common
one on 24.04, where only libtiff6 ships), point the old soname at the new
library rather than downgrading the system:

    cd /usr/lib/x86_64-linux-gnu
    sudo ln -s libtiff.so.6 libtiff.so.5

That works when the ABI is compatible, which it is for the libraries CARLA
loads. If the server still dies, run it once in the foreground and read the
first error line; it names the real cause more often than not.
NOTE
exit 1
