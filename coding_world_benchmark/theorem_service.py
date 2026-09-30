"""The theorem-discovery bundle, reachable from the agent.

``theorem_discovery_system/`` is a research scaffold that sits beside this
package: a Lean-extracted corpus, a discovery core that proposes algebraic laws
and sieves them against finite models, and its own test suite.  This module is
the seam that lets the agent search it, run it and check it, from the chat, the
HTTP API and the panel.

**It is driven as a subprocess, not imported.**  Three reasons, all of them
about not letting a research scaffold into the agent's process:

* its modules are plain top-level names -- ``corpus``, ``codec``, ``compressor``,
  ``diagram`` -- and putting that directory on ``sys.path`` would let any of them
  shadow something the agent or a library imports;
* its data paths are relative to the bundle root, so its own README says to run
  from inside the folder, which is exactly what a subprocess with ``cwd`` set
  does;
* the bundle's own notes record an experiment that hangs, and a hang inside the
  bridge process would take the agent with it.  A subprocess takes a timeout.

**What the numbers mean, kept separate.**  The core's own docstring says it:
``asserted`` counts what passed the finite-model sieve, *not* what was proved.
Proof is ``valid``, and it only moves when Lean is actually called.  So this
module never reports a discovery as a theorem: it reports ``asserted`` and
``valid`` side by side and, when ``lean_calls`` is zero, says in words that
nothing was proved.  That is L8.22's rule in another costume -- the sieve has
proposal rights, the prover has authority -- and collapsing the two would be the
confidently-wrong failure this project keeps measuring.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

BUNDLE = Path(__file__).resolve().parent.parent / "theorem_discovery_system"
CORPUS = BUNDLE / "theorems.json"
DEFAULT_CYCLES = 12
MAX_CYCLES = 400
RUN_TIMEOUT = 120
VERIFY_TIMEOUT = 300

THEOREM_FEATURE_COMMANDS: tuple[tuple[str, str], ...] = (
    ("/theorem", "定理発見システムの状態と使い方"),
    ("/theorem search <語>", "Lean 抽出コーパスを名前・領域・前提・命題から検索"),
    ("/theorem run [周期]", "発見核を回し、篩を通った候補を出す（証明ではない）"),
    ("/theorem obligations [周期]", "主張された辺を Lean ファイルにまとめる"),
    ("/theorem verify", "バンドル自身のテストを走らせる"),
)

#: The runner. Written here rather than added to the bundle so that integrating
#: the agent changes nothing inside the research scaffold.
_RUNNER = """
import json, sys
sys.path.insert(0, "python")
import corpus
from discovery_core import TheoremDiscoveryCore

pool, labels = corpus.build()
core = TheoremDiscoveryCore(capacity={capacity})
core.run(pool, cycles={cycles})
asserted = []
for record in core.history:
    candidate = record.candidate
    if not record.asserted or candidate is None:
        continue
    asserted.append({{
        "cycle": record.cycle,
        "name": candidate.name,
        "mode": candidate.mode,
        "confidence": round(float(candidate.confidence), 4),
        "known": bool(candidate.known_conclusion),
        "source": candidate.source_kind,
        "lean": candidate.lean,
    }})
print(json.dumps({{
    "summary": core.summary(),
    "asserted": asserted,
    "obligations": core.obligations(),
    "pool": len(pool),
}}, ensure_ascii=False, default=str))
"""


@dataclass(frozen=True)
class Outcome:
    ok: bool
    detail: str
    payload: dict


def available() -> bool:
    return BUNDLE.is_dir() and (BUNDLE / "python" / "discovery_core.py").is_file()


def _missing() -> Outcome:
    return Outcome(False,
                   f"定理発見システムが見つかりません（{BUNDLE.name}/ を "
                   f"{BUNDLE.parent} に置いてください）", {})


def _python() -> str:
    return sys.executable or shutil.which("python3") or "python3"


def _subprocess(script: str, timeout: int) -> Outcome:
    """Run one script inside the bundle, with a timeout and no shell."""
    if not available():
        return _missing()
    try:
        done = subprocess.run([_python(), "-c", script], cwd=str(BUNDLE),
                              capture_output=True, text=True, encoding="utf-8",
                              timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        return Outcome(False, f"{timeout} 秒で打ち切りました"
                              "（バンドルには停止しない実験が記録されています）", {})
    except OSError as error:
        return Outcome(False, f"起動できませんでした: {error}", {})
    if done.returncode != 0:
        tail = (done.stderr or done.stdout or "").strip().splitlines()
        return Outcome(False, "\n".join(tail[-4:]) or "失敗しました", {})
    try:
        return Outcome(True, "", json.loads(done.stdout))
    except json.JSONDecodeError:
        return Outcome(False, "出力を読めませんでした", {"raw": done.stdout[-400:]})


# --------------------------------------------------------------------------
# the corpus
# --------------------------------------------------------------------------

def corpus_records() -> list[dict]:
    if not CORPUS.is_file():
        return []
    try:
        return json.loads(CORPUS.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []


def search(query: str, limit: int = 12) -> list[dict]:
    """Name, domain, premises and statement, matched as plain substrings.

    No ranking model: the corpus is 42 records and a substring match is honest
    about what it did. Anything cleverer would need saying what it scored.
    """
    wanted = query.strip().lower()
    found = []
    for record in corpus_records():
        haystack = " ".join([
            str(record.get("name", "")), str(record.get("domain", "")),
            " ".join(record.get("premises", []) or []),
            str(record.get("statement", "")),
        ]).lower()
        if wanted and wanted not in haystack:
            continue
        found.append({
            "name": record.get("name", "?"),
            "domain": record.get("domain", "?"),
            "statement": str(record.get("statement", ""))[:160],
            "premises": list(record.get("premisesThm", []) or []),
            "proof_size": record.get("proofSize"),
            "statement_size": record.get("statementSize"),
        })
        if len(found) >= limit:
            break
    return found


# --------------------------------------------------------------------------
# running the core
# --------------------------------------------------------------------------

def run_discovery(cycles: int = DEFAULT_CYCLES, capacity: int = 12,
                  timeout: int = RUN_TIMEOUT) -> Outcome:
    cycles = max(1, min(int(cycles), MAX_CYCLES))
    capacity = max(1, min(int(capacity), 64))
    return _subprocess(_RUNNER.format(cycles=cycles, capacity=capacity), timeout)


def verify(timeout: int = VERIFY_TIMEOUT) -> Outcome:
    """The bundle's own tests, run the way its README runs them."""
    if not available():
        return _missing()
    # No ``-q`` here: the bundle's pyproject already sets it, and a second one
    # makes pytest ``-qq``, which drops the very summary line this parses.
    command = [_python(), "-m", "pytest", "python/test_qg_verification.py",
               "python/test_discovery_core.py", "-p", "no:cacheprovider",
               f"--ignore={BUNDLE / '.pytest_cache'}"]
    try:
        done = subprocess.run(command, cwd=str(BUNDLE), capture_output=True,
                              text=True, encoding="utf-8", timeout=timeout,
                              check=False)
    except (subprocess.TimeoutExpired, OSError) as error:
        return Outcome(False, f"テストを走らせられませんでした: {error}", {})
    output = (done.stdout or "") + (done.stderr or "")
    tally = re.findall(r"(\d+) (passed|failed|error)s?\b", output)
    counts = {kind.rstrip("s"): int(number) for number, kind in tally}
    return Outcome(counts.get("failed", 0) == 0 and counts.get("error", 0) == 0
                   and counts.get("passed", 0) > 0,
                   output.strip().splitlines()[-1] if output.strip() else "",
                   {"counts": counts, "returncode": done.returncode})


def status() -> dict:
    records = corpus_records()
    domains: dict[str, int] = {}
    for record in records:
        domains[str(record.get("domain", "?"))] = domains.get(
            str(record.get("domain", "?")), 0) + 1
    return {
        "available": available(),
        "path": str(BUNDLE),
        "corpus": len(records),
        "domains": domains,
        "commands": [command for command, _ in THEOREM_FEATURE_COMMANDS],
    }


# --------------------------------------------------------------------------
# saying it without overstating it
# --------------------------------------------------------------------------

def _proof_caveat(summary: dict) -> list[str]:
    asserted = summary.get("asserted", 0)
    valid = summary.get("valid", 0)
    lean = summary.get("lean_calls", 0)
    lines = ["", f"**篩を通った {asserted} 本は「証明された」ではありません。**"]
    if lean == 0:
        lines.append("この実行で Lean は **1 回も呼ばれていません**"
                     f"（`lean_calls` = 0、`valid` = {valid}）。"
                     "有限モデルで反証されなかった、という意味だけです。")
    else:
        lines.append(f"Lean 呼び出し {lean} 回、証明済み `valid` = {valid} 本。"
                     "残りは篩を通っただけです。")
    lines.append("証明義務は `/theorem obligations` で Lean ファイルとして出せます。")
    return lines


def format_run(payload: dict) -> str:
    summary = payload.get("summary", {})
    asserted = payload.get("asserted", [])
    lines = ["**定理発見核を回しました**", "",
             "| 指標 | 値 |", "| --- | --- |",
             f"| cycles | {summary.get('cycles')} |",
             f"| asserted（篩を通った） | **{summary.get('asserted')}** |",
             f"| うち新規 | {summary.get('novel_asserted')} |",
             f"| valid（証明された） | **{summary.get('valid')}** |",
             f"| lean_calls | {summary.get('lean_calls')} |",
             f"| models_tested | {summary.get('models_tested')} |",
             f"| concept_coverage | {summary.get('concept_coverage')} "
             f"({summary.get('concept_coverage_ratio')}) |",
             f"| derivation_checks_pass | {summary.get('derivation_checks_pass')} |",
             f"| corpus pool | {payload.get('pool')} |"]
    if asserted:
        lines += ["", "| 周期 | 候補 | 由来 | 既知 | confidence |",
                  "| --- | --- | --- | --- | --- |"]
        for item in asserted:
            lines.append(f"| {item['cycle']} | `{item['name']}` | {item['mode']} "
                         f"| {'既知' if item['known'] else '新規'} "
                         f"| {item['confidence']} |")
    lines += _proof_caveat(summary)
    return "\n".join(lines)


def format_search(query: str, found: list[dict]) -> str:
    if not found:
        return (f"`{query}` に当たる定理はコーパスにありません"
                f"（{len(corpus_records())} 件中）。")
    lines = [f"**`{query}`: {len(found)} 件**", "",
             "| 名前 | 領域 | 前提 | 証明サイズ |", "| --- | --- | --- | --- |"]
    for item in found:
        lines.append(f"| `{item['name']}` | {item['domain']} "
                     f"| {'、'.join(item['premises']) or '-'} "
                     f"| {item['proof_size']} |")
    lines += ["", "命題（先頭のみ）", ""]
    for item in found[:3]:
        lines.append(f"- `{item['name']}`: `{item['statement']}`")
    return "\n".join(lines)


def format_status() -> str:
    found = status()
    if not found["available"]:
        return _missing().detail
    domains = "、".join(f"{name}: {count}"
                       for name, count in sorted(found["domains"].items()))
    lines = ["**定理発見システム**", "",
             f"- 場所: `{found['path']}`",
             f"- コーパス: {found['corpus']} 件（{domains}）", "",
             "| コマンド | 用途 |", "| --- | --- |"]
    for command, description in THEOREM_FEATURE_COMMANDS:
        lines.append(f"| `{command}` | {description} |")
    lines += ["", "**篩（asserted）と証明（valid）は別のものとして出します。**"
              "発見核の有限モデル篩は提案であって、権限は Lean の側にあります。"]
    return "\n".join(lines)


def handle_command(message: str) -> str | None:
    """Return markdown for a ``/theorem`` command, or ``None`` if it is not one."""
    text = message.strip()
    if not text.startswith("/theorem"):
        return None
    rest = text[len("/theorem"):].strip()
    if not rest:
        return format_status()
    if rest.startswith("search"):
        query = rest[len("search"):].strip()
        if not query:
            return "使い方: /theorem search <語>（例: /theorem search assoc）"
        return format_search(query, search(query))
    if rest.startswith("run"):
        argument = rest[len("run"):].strip()
        cycles = int(argument) if argument.isdigit() else DEFAULT_CYCLES
        outcome = run_discovery(cycles)
        if not outcome.ok:
            return f"**回せませんでした**: {outcome.detail}"
        return format_run(outcome.payload)
    if rest.startswith("obligations"):
        argument = rest[len("obligations"):].strip()
        cycles = int(argument) if argument.isdigit() else DEFAULT_CYCLES
        outcome = run_discovery(cycles)
        if not outcome.ok:
            return f"**出せませんでした**: {outcome.detail}"
        text_out = outcome.payload.get("obligations", "")
        body = text_out if len(text_out) < 4000 else text_out[:4000] + "\n-- …"
        return ("**証明義務（篩を通った辺だけ）**\n\n```lean\n" + body + "\n```\n\n"
                + "\n".join(_proof_caveat(outcome.payload.get("summary", {}))))
    if rest.startswith("verify"):
        outcome = verify()
        counts = outcome.payload.get("counts", {})
        head = "**バンドル自身のテスト**" if outcome.ok else "**テストが通りません**"
        return (f"{head}\n\n- 結果: `{outcome.detail}`\n"
                f"- passed {counts.get('passed', 0)} / "
                f"failed {counts.get('failed', 0)} / error {counts.get('error', 0)}")
    return format_status()
