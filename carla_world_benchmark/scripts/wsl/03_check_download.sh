#!/usr/bin/env bash
# Ask every candidate host what it will actually serve, without downloading.
#     bash scripts/wsl/03_check_download.sh
#
# Uses a 1 KB ranged GET rather than HEAD: some CDNs refuse HEAD but serve GET.

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$SCRIPT_DIR/config.sh"
CURL="${CURL:-$(command -v /usr/bin/curl || command -v curl)}"
. "$SCRIPT_DIR/lib_download.sh"

echo "using curl: $CURL"
echo
printf '%-14s %-8s %s\n' "SOURCE" "STATUS" "URL"
printf '%-14s %-8s %s\n' "------" "------" "---"

check() {
    local label="$1" url="$2"
    local status; status="$(probe_url "$url")"
    printf '%-14s %-8s %s\n' "$label" "$status" "$url"
    case "$status" in 200|206) return 0 ;; *) return 1 ;; esac
}

nightly_ok=0
check "nightly"     "$NIGHTLY_URL"            && nightly_ok=$((nightly_ok+1))
check "nightly-maps" "$NIGHTLY_MAPS_URL"      && nightly_ok=$((nightly_ok+1))
check "leaderboard" "$LEADERBOARD_PACKAGE_URL" || true
check "0915"        "$RELEASE0915_URL"        || true
check "0915-maps"   "$RELEASE0915_MAPS_URL"   || true

echo
echo "== bodies of any refused response =="
for url in "$NIGHTLY_URL" "$LEADERBOARD_PACKAGE_URL" "$RELEASE0915_URL"; do
    status="$(probe_url "$url")"
    case "$status" in
        200|206) ;;
        *) echo "--- $url ($status)"; show_error_body "$url" ;;
    esac
done

echo
echo "== already downloaded files =="
shopt -s nullglob
found=0
for f in "$DOWNLOAD_DIR"/*; do
    found=1
    printf '  %-42s %12s bytes  %s\n' "$(basename "$f")" "$(file_size "$f")" "$(archive_kind "$f")"
done
[ "$found" -eq 1 ] || echo "  (none in $DOWNLOAD_DIR)"

echo
if [ "$nightly_ok" -eq 2 ]; then
    echo "The nightly build and its AdditionalMaps are both being served."
    echo "AdditionalMaps is what carries Town12 and Town13, so the leaderboard routes will run."
    echo "  bash scripts/wsl/01_setup.sh --carla=nightly"
else
    echo "The nightly build is not reachable from here either. Nothing left to try"
    echo "automatically -- report the statuses above rather than retrying."
fi
