"""Does the agent the launcher starts still have the functions it is meant to have?

``start_coding_agent.ps1`` builds the UI and starts Electron; Electron spawns the
Python bridge and loads the built bundle.  Three things can silently come apart
in that chain, and each is checked here rather than assumed:

1. **The UI calls an endpoint the bridge does not serve.**  Every ``API_BASE``
   call is parsed out of ``web/src/main.jsx`` and actually requested, against a
   bridge started exactly the way Electron starts it.
2. **The bundle is older than the source.**  The launcher rebuilds on every run,
   so a stale ``web/dist`` means the agent that is running is not the agent in
   the repository.  Timestamps and content are both checked, because a rebuild
   that silently dropped the panel would pass a timestamp check.
3. **A command is advertised and not handled.**  ``/api/features`` is what the UI
   offers the person, so every command in it is run through the handler that is
   supposed to implement it.

Run it against a live agent with ``--port 8787`` -- on Windows, with the agent
started by the launcher, that verifies the process the person is actually using.
With no port it starts its own bridge on a free port with the same command line
Electron uses, which verifies the repository.
"""
from __future__ import annotations

import json
import re
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
MAIN_JSX = ROOT / "web" / "src" / "main.jsx"
STYLES = ROOT / "web" / "src" / "styles.css"
DIST = ROOT / "web" / "dist"

#: What a state snapshot has to carry for the panel to render.
STATE_KEYS = ("status", "diagnosis", "episode", "episodes", "blocked", "surface",
              "candidates", "question", "structural_question", "suspended",
              "lexicon", "senses", "invariants", "turns", "store")
INVARIANTS = ("executed_without_authority", "authority_while_disputed",
              "committed_without_commitment", "stale_episode_leak")

UNKNOWN_WORD_REQUEST = "64x64のケースで処理時間ではなくフガ率をならして"


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _get(port: int, path: str, timeout: float = 30.0) -> dict:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=timeout) as response:
        return json.load(response)


def _post(port: int, path: str, body: dict, timeout: float = 60.0) -> dict:
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def _wait(port: int, seconds: float = 30.0) -> bool:
    deadline = time.time() + seconds
    while time.time() < deadline:
        try:
            _get(port, "/api/health", timeout=2.0)
            return True
        except (urllib.error.URLError, OSError, ValueError):
            time.sleep(0.2)
    return False


# --------------------------------------------------------------------------
# what the UI asks for
# --------------------------------------------------------------------------

def ui_api_calls() -> tuple[str, ...]:
    """Every endpoint the bundle's source calls, read out of the source.

    Template paths (``semantic/${path}``) are expanded from the literal action
    names the panel passes, so an action added without an endpoint is caught.
    """
    source = MAIN_JSX.read_text(encoding="utf-8")
    found = set(re.findall(r"\$\{API_BASE\}/([a-z/]+)", source))
    # The panel receives ``semanticAction`` as ``onAction``, so both spellings
    # are read: an action wired into a button with no endpoint behind it is
    # exactly the drift this check exists to catch.
    dynamic = set(re.findall(r"(?:semanticAction|onAction)\('([a-z]+)'", source))
    # The theorem panel has its own dispatcher; the two are named apart so an
    # action cannot be attributed to the wrong endpoint family.
    dynamic |= {f"theorem/{name}" for name in
                re.findall(r"(?:theoremAction|onTheorem)\('([a-z]+)'", source)}
    found = {path for path in found if not path.endswith("/")}
    return tuple(sorted(found | {name if "/" in name else f"semantic/{name}"
                                 for name in dynamic}))


def verify_routes(port: int) -> list[Check]:
    """Call every endpoint the UI calls, and check the shape of what comes back."""
    checks: list[Check] = []

    health = _get(port, "/api/health")
    checks.append(Check("GET /api/health", health.get("ok") is True,
                        f"workspace={health.get('workspace')}"))

    features = _get(port, "/api/features")
    commands = [item["command"] for item in features.get("features", [])]
    checks.append(Check("GET /api/features", len(commands) >= 12,
                        f"{len(commands)} commands"))

    state = _get(port, "/api/semantic/state").get("state", {})
    missing = [key for key in STATE_KEYS if key not in state]
    checks.append(Check("GET /api/semantic/state", not missing,
                        "all panel keys present" if not missing else f"missing {missing}"))
    checks.append(Check("invariants reported",
                        tuple(state.get("invariants", {})) == INVARIANTS,
                        ", ".join(state.get("invariants", {}))))

    asked = _post(port, "/api/semantic/ask",
                  {"message": UNKNOWN_WORD_REQUEST}).get("state", {})
    question = asked.get("question") or {}
    checks.append(Check("POST /api/semantic/ask",
                        asked.get("status") == "ASKED" and bool(question.get("text")),
                        question.get("text", asked.get("status", "?"))))
    checks.append(Check("goal suspended, not restarted",
                        asked.get("suspended") == UNKNOWN_WORD_REQUEST,
                        str(asked.get("suspended"))))

    unknown = _post(port, "/api/semantic/answer", {"text": "分からない"}).get("state", {})
    committed = unknown.get("turns", [{}])[-1].get("committed")
    checks.append(Check("「分からない」 commits nothing",
                        unknown.get("status") == "NO_COMMIT" and committed is False,
                        f"status={unknown.get('status')} committed={committed}"))

    answered = _post(port, "/api/semantic/answer", {"text": "はい"}).get("state", {})
    checks.append(Check("POST /api/semantic/answer resumes the goal",
                        answered.get("status") == "ANSWER"
                        and answered.get("resumed") == UNKNOWN_WORD_REQUEST,
                        f"{answered.get('status')} / {answered.get('diagnosis')}"))

    rerun = _post(port, "/api/semantic/rerun",
                  {"request": UNKNOWN_WORD_REQUEST}).get("state", {})
    programs = rerun.get("rerun", {})
    checks.append(Check("POST /api/semantic/rerun", any(programs.values()),
                        " / ".join(f"{k}: {v}" for k, v in programs.items())))

    episode = _post(port, "/api/semantic/episode", {"episode": "lab"}).get("state", {})
    blocked_after_switch = episode.get("episode") == "lab"
    checks.append(Check("POST /api/semantic/episode", blocked_after_switch,
                        f"episode={episode.get('episode')}"))

    opened = _post(port, "/api/semantic/open", {"episode": "lab"}).get("state", {})
    recallable = {row["name"]: row["recallable"] for row in opened.get("episodes", [])}
    checks.append(Check("POST /api/semantic/open", recallable.get("lab") is True,
                        str(recallable)))

    # No structural question is pending in this short scenario -- L8.35 needs
    # two uses per episode before time and context compete -- so what is checked
    # here is that the endpoint answers with a state rather than an error, and
    # the detail says so instead of implying more.
    model = _post(port, "/api/semantic/model", {"label": "場面"}).get("state", {})
    checks.append(Check("POST /api/semantic/model answers",
                        all(key in model for key in STATE_KEYS),
                        model.get("reply", "")[:48] or "state returned"))

    reset = _post(port, "/api/semantic/reset", {}).get("state", {})
    checks.append(Check("POST /api/semantic/reset",
                        reset.get("lexicon") == [] and reset.get("turns") == [],
                        "state cleared"))

    chat = _post(port, "/api/chat", {"message": "/semantics"})
    checks.append(Check("POST /api/chat routes semantic commands",
                        chat.get("provider") == "semantics",
                        f"provider={chat.get('provider')}"))

    ask = _post(port, "/api/chat", {"message": f"/ask {UNKNOWN_WORD_REQUEST}"})
    checks.append(Check("POST /api/chat handles /ask",
                        chat.get("ok") and "ASKED" in ask.get("text", ""),
                        "a question came back"))

    follow_up = _post(port, "/api/chat", {"message": "はい"})
    checks.append(Check("a pending question owns the next utterance",
                        follow_up.get("provider") == "semantics",
                        f"provider={follow_up.get('provider')}"))
    _post(port, "/api/semantic/reset", {})

    return checks


def coverage_check(checks: list[Check]) -> Check:
    """Was every endpoint the UI calls actually exercised above?

    Computed over the whole run rather than inside one group, so an endpoint
    covered by a later group still counts -- the question is whether the UI can
    call something nothing touched, not which function touched it.
    """
    served = " ".join(check.name for check in checks if check.ok)
    wanted = [f"/api/{path}" for path in ui_api_calls()
              if path not in ("workspace", "workspace/pick")]
    missing = [endpoint for endpoint in wanted if endpoint not in served]
    return Check("every endpoint the UI calls was exercised", not missing,
                 ", ".join(sorted(wanted)) if not missing
                 else f"not exercised: {missing}")


def verify_templates(port: int) -> list[Check]:
    """Replay every template the UI offers and check it still does what it says.

    This is what makes "検証済み" in the panel mean something: the catalogue
    records what the frozen dialogue hold-out established each form does, and
    this replays all of them through the running bridge rather than trusting the
    label.
    """
    templates = _get(port, "/api/semantic/templates").get("templates", [])
    checks = [Check("GET /api/semantic/templates", bool(templates),
                    f"{len(templates)} templates")]
    request = next((item["preview"] for item in templates
                    if item["id"] == "unknown_argument"), UNKNOWN_WORD_REQUEST)
    for item in templates:
        expect = item.get("expect") or {}
        _post(port, "/api/semantic/reset", {})
        if not expect:
            spoken = _post(port, "/api/chat", {"message": item["preview"]})
            checks.append(Check(f"template {item['id']}",
                                spoken.get("provider") == "semantics",
                                f"{item['preview']} -> {spoken.get('provider')}"))
            continue
        if item["kind"] == "REQUEST":
            state = _post(port, "/api/semantic/ask",
                          {"message": item["preview"]}).get("state", {})
        else:
            _post(port, "/api/semantic/ask", {"message": request})
            state = _post(port, "/api/semantic/answer",
                          {"text": item["preview"]}).get("state", {})
        turns = state.get("turns") or [{}]
        committed = bool(turns[-1].get("committed"))
        ok = (state.get("status") == expect.get("decision")
              and committed == expect.get("commits"))
        checks.append(Check(
            f"template {item['id']}", ok,
            f"{item['preview']} -> {state.get('status')} / committed={committed}"
            + ("" if ok else f" (期待 {expect})")))
    _post(port, "/api/semantic/reset", {})
    return checks


def verify_script(port: int) -> list[Check]:
    """The code surface: it runs, it refuses as a whole, and it agrees with prose.

    The last one is the claim worth checking. ``yes()`` and 「はい」 are two ways
    of writing the same move, so if they ever stopped landing in the same state
    the code surface would be a second, unverified dialogue rather than a way of
    writing down the verified one.
    """
    templates = _get(port, "/api/semantic/templates").get("templates", [])
    request = next((item["preview"] for item in templates
                    if item["id"] == "unknown_argument"), UNKNOWN_WORD_REQUEST)

    _post(port, "/api/semantic/reset", {})
    source = ('open_episode("lab")\n'
              f'ask("{request}")\n'
              "no()\n"
              "rerun()")
    state = _post(port, "/api/semantic/script", {"source": source}).get("state", {})
    script = state.get("script", {})
    checks = [Check("POST /api/semantic/script", script.get("ok") is True
                    and len(script.get("results", [])) == 4,
                    " -> ".join(item["call"] for item in script.get("results", [])))]

    # A script with one unreadable line must not run its readable ones.
    _post(port, "/api/semantic/reset", {})
    broken = _post(port, "/api/semantic/script",
                   {"source": f'ask("{request}")\nimport os'}).get("state", {})
    checks.append(Check("a script is refused as a whole",
                        broken.get("script", {}).get("ok") is False
                        and not broken.get("turns")
                        and broken.get("question") is None,
                        broken.get("script", {}).get("error", "")))

    # And the code form of every verified template has to land where the prose
    # form lands.
    for item in templates:
        expect = item.get("expect") or {}
        if not expect or not item.get("code"):
            continue
        _post(port, "/api/semantic/reset", {})
        if item["kind"] != "REQUEST":
            _post(port, "/api/semantic/ask", {"message": request})
        state = _post(port, "/api/semantic/script",
                      {"source": item["code"]}).get("state", {})
        turns = state.get("turns") or [{}]
        committed = bool(turns[-1].get("committed"))
        ok = (state.get("status") == expect.get("decision")
              and committed == expect.get("commits"))
        checks.append(Check(f"code form of {item['id']}", ok,
                            f"{item['code']} -> {state.get('status')} / "
                            f"committed={committed}"))
    _post(port, "/api/semantic/reset", {})
    return checks


PHYSICS = ("Bz = 1.0\nm = 1.0\n"
           "vector: B = (0, 0, Bz)\nlorentz: F exists\n"
           "calc(m d^2 x / dt^2 = lorentz(B))\ndraw(x)")


def verify_calculation(port: int) -> list[Check]:
    """Arithmetic, and the physics language's refusal to guess a symbol."""
    checks: list[Check] = []
    for source, wanted in (("1+1=?", "2"), ("2^10=?", "1024"),
                           ("sqrt(2)=?", "1.414213562")):
        spoken = _post(port, "/api/chat", {"message": source}).get("text", "")
        checks.append(Check(f"chat answers {source}", wanted in spoken,
                            spoken.strip()[:60]))
    unsafe = _post(port, "/api/chat", {"message": '__import__("os")=?'}).get("text", "")
    checks.append(Check("the calculator refuses what is not arithmetic",
                        "使えません" in unsafe, unsafe.strip()[:60]))
    prose = _post(port, "/api/chat", {"message": "これは1+1みたいな話です"})
    checks.append(Check("prose with a number in it is not a calculation",
                        prose.get("provider") != "semantics",
                        f"provider={prose.get('provider')}"))

    # The symbol nobody bound is the whole point: it must stop the run and be
    # named, not be quietly assumed.
    unbound = PHYSICS.split("\n", 2)[2]
    state = _post(port, "/api/semantic/script", {"source": unbound}).get("state", {})
    script = state.get("script", {})
    checks.append(Check("an unbound symbol stops the physics script",
                        script.get("ok") is False
                        and set(script.get("needs", [])) == {"Bz", "m"},
                        f"needs={script.get('needs')}"))

    state = _post(port, "/api/semantic/script", {"source": PHYSICS}).get("state", {})
    script = state.get("script", {})
    checks.append(Check("POST /api/semantic/script solves and draws",
                        script.get("ok") is True
                        and script.get("language") == "physics"
                        and "data:image/png;base64," in script.get("output", ""),
                        " / ".join(item["detail"] for item in
                                   script.get("results", []))[:90]))
    checks.append(Check("the solver is compared with the closed form",
                        "解析解" in script.get("output", ""),
                        next((line for line in script.get("output", "").splitlines()
                              if "解析解" in line), "")[:80]))

    typo = _post(port, "/api/semantic/script",
                 {"source": "lorentz: F exitst"}).get("state", {})
    checks.append(Check("a misspelled keyword is refused with the nearest one",
                        typo.get("script", {}).get("ok") is False
                        and "exists" in typo.get("script", {}).get("error", ""),
                        typo.get("script", {}).get("error", "")[:70]))
    return checks


AMBIGUOUS_QUESTION = "(A in B) and (B in A)ならば？"


def verify_logic(port: int) -> list[Check]:
    """The ambiguous operator is carried, not chosen -- and can then be fixed."""
    _post(port, "/api/semantic/reset", {})
    both = _post(port, "/api/chat", {"message": AMBIGUOUS_QUESTION}).get("text", "")
    checks = [
        Check("both readings of an ambiguous operator are given",
              "A = B" in both and "矛盾" in both and "正則性" in both,
              "⊆ なら A = B / ∈ なら矛盾"),
        Check("the reader is told how to fix the reading",
              "in := subset" in both and "in := member" in both,
              "両方の読みが提示された"),
    ]
    transitive = _post(port, "/api/chat",
                       {"message": "A ⊆ B かつ B ⊆ C ならば？"}).get("text", "")
    checks.append(Check("transitivity is derived with its rule named",
                        "A ⊆ C" in transitive and "推移律" in transitive,
                        "A ⊆ C"))
    denied = _post(port, "/api/chat",
                   {"message": "A ∈ B and B ∈ C => A ∈ C ?"}).get("text", "")
    checks.append(Check("a non-consequence is denied rather than assumed",
                        "出てきません" in denied, denied.strip()[:60]))

    fixed = _post(port, "/api/chat", {"message": "in := subset"}).get("text", "")
    after = _post(port, "/api/chat", {"message": AMBIGUOUS_QUESTION}).get("text", "")
    checks.append(Check("fixing the reading leaves one answer",
                        "固定" in fixed and "矛盾" not in after and "A = B" in after,
                        "in := subset のあとは ⊆ の読みだけ"))
    _post(port, "/api/semantic/reset", {})

    prose = _post(port, "/api/chat", {"message": "これは in の話ではありません"})
    checks.append(Check("prose containing a relation word is not a question",
                        prose.get("provider") != "semantics",
                        f"provider={prose.get('provider')}"))
    return checks


def verify_theorems(port: int) -> list[Check]:
    """The discovery bundle: reachable, searchable, runnable -- and not oversold."""
    found = _get(port, "/api/theorem/status").get("status", {})
    checks = [Check("GET /api/theorem/status", found.get("available") is True,
                    f"corpus={found.get('corpus')} at {found.get('path')}")]
    if not found.get("available"):
        return checks

    hits = _post(port, "/api/theorem/search", {"query": "assoc"})
    names = [item["name"] for item in hits.get("results", [])]
    checks.append(Check("POST /api/theorem/search", "dA_add_assoc" in names,
                        "、".join(names[:4])))

    run = _post(port, "/api/theorem/run", {"cycles": 12})
    summary = run.get("summary", {})
    checks.append(Check("POST /api/theorem/run", run.get("ok") is True
                        and summary.get("cycles") == 12,
                        f"asserted={summary.get('asserted')} "
                        f"valid={summary.get('valid')} "
                        f"lean_calls={summary.get('lean_calls')}"))
    # The claim that must not drift: a sieved candidate is not a proved theorem,
    # and the report has to say so in the run where Lean was never called.
    checks.append(Check("a sieved candidate is not reported as proved",
                        "証明された" not in run.get("output", "").split("asserted")[0]
                        and "「証明された」ではありません" in run.get("output", ""),
                        "asserted と valid を分けて出している"))
    checks.append(Check("a run without Lean says nothing was proved",
                        summary.get("lean_calls") != 0
                        or "1 回も呼ばれていません" in run.get("output", ""),
                        f"lean_calls={summary.get('lean_calls')}"))

    duties = _post(port, "/api/theorem/obligations", {"cycles": 12})
    checks.append(Check("POST /api/theorem/obligations",
                        "import Mathlib" in duties.get("obligations", ""),
                        "Lean ファイルとして出た"))

    # The bundle's own suite, run through the agent. It is also what says the
    # transplanted corpus path is right: with the old one, five of its tests
    # looked for a file outside the bundle and failed.
    tests = _post(port, "/api/theorem/verify", {}, timeout=420.0)
    counts = tests.get("counts", {})
    checks.append(Check("POST /api/theorem/verify", tests.get("ok") is True,
                        f"passed={counts.get('passed')} "
                        f"failed={counts.get('failed', 0)}"))

    spoken = _post(port, "/api/chat", {"message": "/theorem search inter"})
    checks.append(Check("chat reaches the bundle",
                        spoken.get("provider") == "semantics"
                        and "dB_inter" in spoken.get("text", ""),
                        f"provider={spoken.get('provider')}"))
    return checks


def verify_workspace(port: int, folder: Path) -> list[Check]:
    result = _post(port, "/api/workspace", {"path": str(folder)})
    return [Check("POST /api/workspace", result.get("ok") is True,
                  str(result.get("workspace")))]


# --------------------------------------------------------------------------
# the bundle the launcher builds
# --------------------------------------------------------------------------

def verify_bundle() -> list[Check]:
    checks: list[Check] = []
    if not DIST.exists():
        return [Check("web/dist exists", False,
                      "未ビルド。start_coding_agent.ps1 がビルドするので、"
                      "一度起動すれば作られる")]
    assets = list((DIST / "assets").glob("*"))
    built = max((path.stat().st_mtime for path in assets), default=0.0)
    source = max(MAIN_JSX.stat().st_mtime, STYLES.stat().st_mtime)
    checks.append(Check("bundle is newer than its source", built >= source,
                        time.strftime("built %Y-%m-%d %H:%M", time.localtime(built))))

    script = "".join(path.read_text(encoding="utf-8", errors="ignore")
                     for path in assets if path.suffix == ".js")
    style = "".join(path.read_text(encoding="utf-8", errors="ignore")
                    for path in assets if path.suffix == ".css")
    checks.append(Check("bundle contains the Semantics panel",
                        "SESSION INVARIANTS" in script, "panel markup found"))
    checks.append(Check("bundle calls the semantic endpoints",
                        "/semantic/state" in script, "endpoint found"))
    checks.append(Check("bundle carries the panel styles",
                        "semantic-status" in style and "episode-row" in style,
                        "styles found"))
    return checks


# --------------------------------------------------------------------------
# the commands the UI advertises
# --------------------------------------------------------------------------

def verify_commands(port: int) -> list[Check]:
    from .semantic_service import SemanticSession, handle_command

    advertised = [item["command"]
                  for item in _get(port, "/api/features").get("features", [])]
    semantic = [command for command in advertised
                if command.startswith(("/semantics", "/answer", "/ask"))]
    session = SemanticSession()
    unhandled = []
    for command in semantic:
        stem = command.split(" ")[0]
        sample = {"/answer": "/answer はい",
                  "/ask": f"/ask {UNKNOWN_WORD_REQUEST}"}.get(
            stem, command.replace("<episode>", "lab").replace("<name>", "lab")
                         .replace("<label>", "場面").replace("<reply>", "はい"))
        if handle_command(session, sample) is None:
            unhandled.append(command)
    return [Check("every advertised semantic command is handled", not unhandled,
                  f"{len(semantic)} commands" if not unhandled else str(unhandled)),
            Check("the workspace commands are still advertised",
                  all(command in advertised
                      for command in ("/inspect", "/tests", "/behavior-audit")),
                  f"{len(advertised)} in total")]


# --------------------------------------------------------------------------
# running it
# --------------------------------------------------------------------------

def spawn_bridge(port: int) -> subprocess.Popen:
    """The same command line ``electron/main.cjs`` uses, from the same directory."""
    return subprocess.Popen(
        [sys.executable, "-m", "coding_world_benchmark.coding_agent_bridge", str(port)],
        cwd=str(PROJECT_ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def run(port: int | None = None) -> dict:
    import tempfile

    started = None
    if port is None:
        port = _free_port()
        started = spawn_bridge(port)
    live = _wait(port)
    checks = [Check("bridge is listening", live, f"127.0.0.1:{port}")]
    try:
        if live:
            with tempfile.TemporaryDirectory() as folder:
                checks += verify_workspace(port, Path(folder))
            checks += verify_routes(port)
            checks += verify_templates(port)
            checks += verify_script(port)
            checks += verify_calculation(port)
            checks += verify_logic(port)
            checks += verify_theorems(port)
            checks += verify_commands(port)
            checks.append(coverage_check(checks))
    finally:
        if started is not None:
            started.terminate()
            started.wait(timeout=10)
    checks += verify_bundle()
    return {"port": port, "spawned": started is not None,
            "checks": [vars(check) for check in checks]}


def format_report(report: dict) -> str:
    checks = report["checks"]
    failed = [check for check in checks if not check["ok"]]
    lines = ["## Coding agent contract verification", "",
             f"- bridge: 127.0.0.1:{report['port']} "
             f"({'spawned for this run' if report['spawned'] else 'already running'})",
             f"- **{len(checks) - len(failed)}/{len(checks)} checks passed**", "",
             "| check | result | detail |", "| --- | --- | --- |"]
    for check in checks:
        lines.append(f"| {check['name']} | {'OK' if check['ok'] else '**NG**'} "
                     f"| {check['detail']} |")
    if failed:
        lines += ["", "### 失敗した項目", ""]
        lines += [f"- **{check['name']}** — {check['detail']}" for check in failed]
    return "\n".join(lines)


def main() -> None:  # pragma: no cover
    port = None
    if "--port" in sys.argv:
        port = int(sys.argv[sys.argv.index("--port") + 1])
    report = run(port)
    print(format_report(report))
    sys.exit(0 if all(check["ok"] for check in report["checks"]) else 1)


if __name__ == "__main__":  # pragma: no cover
    main()
