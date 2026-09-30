# Download helpers, kept separate so 03_check_download.sh and 01_setup.sh share
# exactly one implementation. Source this after config.sh.
#
# The hard lesson behind every guard here: an S3 error response is a 299-byte
# XML document that curl happily saves under the archive's name, and tar then
# fails with "File format not recognized" -- a message that says nothing about
# the real cause. Never let a non-200 response reach the disk as an archive.

MIN_ARCHIVE_BYTES="${MIN_ARCHIVE_BYTES:-104857600}"   # 100 MB; any real package is far larger

file_size() { [ -f "$1" ] && stat -c%s "$1" 2>/dev/null || echo 0; }

magic_of() { od -An -tx1 -N6 "$1" 2>/dev/null | tr -d ' \n'; }

archive_kind() {
    # Returns xz / gzip / unknown for the file named by $1.
    case "$(magic_of "$1")" in
        fd377a585a00*) echo xz ;;
        1f8b*)         echo gzip ;;
        *)             echo unknown ;;
    esac
}

describe_url() {
    # Print what the server actually says, following redirects.
    local url="$1"
    echo "  url: $url"
    "$CURL" -sIL -o /dev/null \
        -w '  final url: %{url_effective}\n  status: %{http_code}\n  content-type: %{content_type}\n  content-length: %{size_download} (HEAD)\n' \
        "$url" 2>&1 || echo "  (HEAD request failed)"
}

show_error_body() {
    local url="$1"
    echo "  --- first 800 bytes the server returned ---"
    "$CURL" -sL "$url" | head -c 800
    echo
    echo "  --- end ---"
}

probe_url() {
    # Status of a 1 KB ranged GET: 206 (range honoured) or 200 both mean the
    # object is being served. Costs a kilobyte, not a gigabyte.
    # curl already prints 000 when it never got a response, so a `|| echo 000`
    # here would concatenate into "000000"; capture first, default after.
    local code=""
    code="$("$CURL" -sL -o /dev/null -r 0-1023 -w '%{http_code}' --max-time 60 "$1" 2>/dev/null)" || true
    [ -n "$code" ] || code="000"
    printf '%s' "$code"
}

fetch_archive() {
    # fetch_archive <url> <output> <expected-kind>
    local url="$1" out="$2" want="$3"
    local size

    # A tiny leftover from a failed attempt would make -C - resume into garbage,
    # so throw it away and show what it was.
    size="$(file_size "$out")"
    if [ "$size" -gt 0 ] && [ "$size" -lt "$MIN_ARCHIVE_BYTES" ] && [ "$(archive_kind "$out")" = unknown ]; then
        echo "  discarding $size-byte file that is not an archive:"
        head -c 400 "$out" | sed 's/^/    /'
        echo
        rm -f "$out"
    fi

    # Probe with a ranged GET, not HEAD. A CDN in front of these packages can
    # answer 403 to HEAD while serving GET perfectly well, and gating on HEAD
    # would then refuse a download that would have worked.
    local status
    status="$(probe_url "$url")"
    echo "  probe status (GET, first 1 KB): $status"
    if [ "$status" != "200" ] && [ "$status" != "206" ]; then
        echo "  the server is not serving this object."
        describe_url "$url"
        show_error_body "$url"
        return 1
    fi

    # -f makes curl fail on an HTTP error instead of saving the error body.
    # -C - resumes, so an interrupted multi-gigabyte download costs nothing.
    "$CURL" -fL -C - --retry 5 --retry-delay 10 -o "$out" "$url" || {
        echo "  curl failed (exit $?)"
        return 1
    }

    size="$(file_size "$out")"
    echo "  downloaded: $((size / 1024 / 1024)) MB"
    if [ "$size" -lt "$MIN_ARCHIVE_BYTES" ]; then
        echo "  that is far too small for this package. Contents:"
        head -c 400 "$out" | sed 's/^/    /'
        echo
        return 1
    fi
    local kind
    kind="$(archive_kind "$out")"
    if [ "$kind" != "$want" ]; then
        echo "  expected a $want archive but the magic bytes say '$kind'"
        return 1
    fi
    echo "  verified: $kind archive, magic bytes OK"
    return 0
}
