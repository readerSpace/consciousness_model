#!/usr/bin/env bash
# Make the vendored leaderboard-2.0 / scenario_runner-2.0 checkouts run on
# modern Python. Idempotent: re-run after re-cloning external/.
#
#     bash scripts/apply_patches.sh
#
# Why this is needed: those branches were written for Python 3.7, and
# ElementTree.Element.getchildren() was REMOVED in Python 3.9. Parsing any
# routes file raises AttributeError on 3.9+. `list(elem)` is the documented
# replacement and behaves identically on 3.7 too, so the patch is safe on every
# interpreter and does not fork behaviour between setups.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

TARGETS="
external/leaderboard/leaderboard/utils/route_parser.py
external/scenario_runner/srunner/tools/scenario_parser.py
external/scenario_runner/srunner/tools/route_parser.py
"

changed=0
for rel in $TARGETS; do
    path="$BENCH_ROOT/$rel"
    if [ ! -f "$path" ]; then
        echo "SKIP  $rel (not present)"
        continue
    fi
    if grep -q '\.getchildren()' "$path"; then
        sed -i 's/\([A-Za-z_][A-Za-z0-9_]*\)\.getchildren()/list(\1)/g' "$path"
        echo "PATCH $rel  (getchildren -> list)"
        changed=$((changed+1))
    else
        echo "OK    $rel  (already patched)"
    fi
done

remaining="$(grep -rn '\.getchildren()' "$BENCH_ROOT/external" --include=*.py 2>/dev/null || true)"
if [ -n "$remaining" ]; then
    echo
    echo "still using the removed API:"
    echo "$remaining"
    exit 1
fi
echo
echo "$changed file(s) patched; no getchildren() left under external/"
