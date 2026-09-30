"""L6.5: route execute requests through target resolution and subprocess run.

Search is an intermediate step here, not the final answer. README command
snippets and repository Python files become ExecutionCandidate objects, then a
validated ExecutionPlan is run and captured as an ExecutionResult.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
from typing import Callable, Sequence

from .process_execution import compatible_runner


@dataclass(frozen=True)
class ExecutionCandidate:
    script: str
    args: tuple[str, ...]
    evidence: str
    score: int


@dataclass(frozen=True)
class ExecutionPlan:
    candidate: ExecutionCandidate
    command: tuple[str, ...]
    cwd: str


@dataclass(frozen=True)
class ExecutionResult:
    command: tuple[str, ...]
    return_code: int
    stdout: str
    stderr: str
    elapsed_time: float
    generated_files: tuple[str, ...]
    evidence: str


def is_execute_request(text: str) -> bool:
    lower = text.lower()
    return any(term in text for term in ("実行", "動か", "走らせ", "試して")) or any(
        term in lower for term in ("run", "execute")
    )


def is_test_request(text: str) -> bool:
    lower = text.lower()
    return "テスト" in text or bool(re.search(r"\btests?\b", lower))


def _excluded(path: Path) -> bool:
    return bool({".git", ".venv", "__pycache__", ".pytest_cache", "node_modules"}.intersection(path.parts))


def _request_tokens(text: str) -> tuple[str, ...]:
    lower = text.lower()
    tokens = set(re.findall(r"[a-z][a-z0-9_-]{2,}", lower))
    if "シミュレーション" in text:
        tokens.update(("simulation", "sim"))
    if "python" in lower or "pythonファイル" in text:
        tokens.add("py")
    return tuple(tokens)


def _score_candidate(script: str, args: tuple[str, ...], evidence_text: str, request_tokens: Sequence[str], line_no: int) -> int:
    haystack = " ".join((script, " ".join(args), evidence_text)).lower()
    score = sum(4 for token in request_tokens if token != "py" and token in haystack)
    if script.lower().endswith(".py"):
        score += 2
    if any(word in evidence_text.lower() for word in ("quick", "standard", "basic", "通常", "標準")):
        score += 4
    if "--sweep" in args:
        score -= 2
    if args and "--sweep" not in args:
        score += 1
    if "test" in Path(script).name.lower():
        score -= 4
    score += max(0, 4 - min(line_no, 4))
    return score


def _command_candidates_from_text(root: Path, path: Path, request_tokens: Sequence[str]) -> tuple[ExecutionCandidate, ...]:
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return ()
    candidates: list[ExecutionCandidate] = []
    command_pattern = re.compile(r"(?:^|[\s`])(?:python|py)\s+([.\\/\w-]+\.py)([^\r\n`]*)", re.IGNORECASE)
    for line_no, line in enumerate(lines, 1):
        for match in command_pattern.finditer(line):
            script = match.group(1).replace("\\", "/").lstrip("./")
            args = tuple(shlex.split(match.group(2).strip(), posix=False)) if match.group(2).strip() else ()
            if not (root / script).is_file():
                continue
            evidence = f"{path.relative_to(root)}:{line_no}"
            score = _score_candidate(script, args, line, request_tokens, line_no)
            candidates.append(ExecutionCandidate(script, args, evidence, score))
    return tuple(candidates)


def discover_execution_candidates(root: Path, request: str) -> tuple[ExecutionCandidate, ...]:
    root = root.resolve()
    request_tokens = _request_tokens(request)
    candidates: list[ExecutionCandidate] = []
    docs = sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".md", ".txt"} and not _excluded(path)
    )
    for path in docs:
        candidates.extend(_command_candidates_from_text(root, path, request_tokens))
    explicit = re.search(r"([A-Za-z0-9_.-]+\.py)", request)
    if explicit and (root / explicit.group(1)).is_file():
        script = explicit.group(1)
        candidates.append(ExecutionCandidate(script, (), "request:explicit-path", 100))
    if not candidates:
        for path in sorted(root.rglob("*.py")):
            if _excluded(path):
                continue
            relative = path.relative_to(root).as_posix()
            score = _score_candidate(relative, (), relative, request_tokens, 1)
            if score > 1:
                candidates.append(ExecutionCandidate(relative, (), f"{relative}:1", score))
    unique: dict[tuple[str, tuple[str, ...]], ExecutionCandidate] = {}
    for candidate in candidates:
        key = (candidate.script, candidate.args)
        if key not in unique or candidate.score > unique[key].score:
            unique[key] = candidate
    return tuple(sorted(unique.values(), key=lambda item: (-item.score, item.script, item.args)))


def build_execution_plan(root: Path, request: str) -> ExecutionPlan:
    candidates = discover_execution_candidates(root, request)
    if not candidates:
        raise ValueError("実行候補のPythonファイルを特定できませんでした")
    candidate = candidates[0]
    script_path = (root / candidate.script).resolve()
    if root.resolve() not in script_path.parents and script_path != root.resolve():
        raise ValueError("workspace外のスクリプトは実行できません")
    return ExecutionPlan(candidate, (sys.executable, candidate.script, *candidate.args), str(root.resolve()))


def _snapshot_files(root: Path) -> dict[str, float]:
    snapshot = {}
    for path in root.rglob("*"):
        if path.is_file() and not _excluded(path):
            try:
                snapshot[path.relative_to(root).as_posix()] = path.stat().st_mtime
            except OSError:
                continue
    return snapshot


def execute_plan(
    plan: ExecutionPlan,
    *,
    timeout: float = 30.0,
    runner: Callable[..., subprocess.CompletedProcess[str]] = compatible_runner,
) -> ExecutionResult:
    root = Path(plan.cwd)
    before = _snapshot_files(root)
    started = time.perf_counter()
    completed = runner(
        list(plan.command),
        cwd=root,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    elapsed = time.perf_counter() - started
    after = _snapshot_files(root)
    generated = tuple(sorted(path for path in after if path not in before))
    return ExecutionResult(
        plan.command,
        completed.returncode,
        (completed.stdout or "")[-8000:],
        (completed.stderr or "")[-8000:],
        elapsed,
        generated,
        plan.candidate.evidence,
    )


def format_execution_result(result: ExecutionResult) -> str:
    command = " ".join(result.command)
    lines = [
        "【実行】",
        f"`{command}`",
        "",
        "【結果】",
        f"{'正常終了' if result.return_code == 0 else '失敗'}（return code {result.return_code}）",
        f"elapsed: {result.elapsed_time:.2f}s",
        f"evidence: `{result.evidence}`",
    ]
    if result.stdout.strip():
        lines.extend(["", "【stdout】", "```text", result.stdout.strip(), "```"])
    if result.stderr.strip():
        lines.extend(["", "【stderr】", "```text", result.stderr.strip(), "```"])
    if result.generated_files:
        lines.extend(["", "【生成物】", *[f"- `{item}`" for item in result.generated_files]])
    lines.extend(["", "【判断】", "シミュレーション/スクリプト実行は完了しました。" if result.return_code == 0 else "実行は失敗しました。stderrとreturn codeを確認してください。"])
    return "\n".join(lines)


def execute_request(root: Path, request: str, *, timeout: float = 30.0) -> ExecutionResult:
    return execute_plan(build_execution_plan(root, request), timeout=timeout)
