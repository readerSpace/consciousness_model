"""L8.2: synthesize and validate minimal patches before promoting them."""
from __future__ import annotations

from dataclasses import dataclass
import difflib
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Protocol

from .process_execution import combined_output, run_text
from .external_repair_l81_experiment import (
    PatchResult,
    RepairInstruction,
    RepairPlan,
    RegressionVerifier,
    VerificationResult,
)


@dataclass(frozen=True)
class PatchIntent:
    file: str
    symbol: str | None
    operation: str
    objective: str
    constraints: tuple[str, ...]
    evidence_ids: tuple[str, ...]


@dataclass(frozen=True)
class PatchCandidate:
    intent: PatchIntent
    original: str
    updated: str
    diff: str


@dataclass(frozen=True)
class AutonomousPatchResult:
    status: str
    intents: tuple[PatchIntent, ...]
    candidates: tuple[PatchCandidate, ...]
    patch: PatchResult
    static_validation: VerificationResult | None
    regression: VerificationResult | None
    message: str


class PatchSynthesizer(Protocol):
    def synthesize(self, root: Path, instruction: RepairInstruction, plan: RepairPlan) -> tuple[PatchCandidate, ...]: ...


def _diff(path: str, original: str, updated: str) -> str:
    return "".join(difflib.unified_diff(original.splitlines(True), updated.splitlines(True), fromfile=path, tofile=path))


class MinimalPatchSynthesizer:
    """Conservative deterministic synthesizer for explicit small edits."""

    def synthesize(self, root: Path, instruction: RepairInstruction, plan: RepairPlan) -> tuple[PatchCandidate, ...]:
        if not plan.resolved_files or plan.status.value != "proposal_applicable":
            return ()
        candidates: list[PatchCandidate] = []
        for change in instruction.requested_changes:
            function = re.search(r"(?:add|追加)\s+(?:function|関数)?\s*`?([A-Za-z_]\w*)`?", change, re.I)
            marker = re.search(r"(?:add marker|追加マーカー)\s*:?\s*(.+)", change, re.I)
            if function:
                path = root / plan.resolved_files[0]
                original = path.read_text(encoding="utf-8", errors="replace")
                name = function.group(1)
                if re.search(rf"^def\s+{re.escape(name)}\s*\(", original, re.M):
                    continue
                updated = original.rstrip() + f"\n\n\ndef {name}():\n    return None\n"
                intent = PatchIntent(plan.resolved_files[0], name, "insert", change, instruction.constraints, (f"{path.name}:new_function",))
                candidates.append(PatchCandidate(intent, original, updated, _diff(plan.resolved_files[0], original, updated)))
            elif marker:
                path = root / plan.resolved_files[0]
                original = path.read_text(encoding="utf-8", errors="replace")
                updated = original.rstrip() + f"\n\n# {marker.group(1).strip()}\n"
                intent = PatchIntent(plan.resolved_files[0], None, "insert", change, instruction.constraints, (f"{path.name}:end",))
                candidates.append(PatchCandidate(intent, original, updated, _diff(plan.resolved_files[0], original, updated)))
        return tuple(candidates)


class CompositePatchSynthesizer:
    """Try conservative synthesizers first, then the semantic L8.5 compiler."""

    def __init__(self, synthesizers: tuple[PatchSynthesizer, ...]):
        self.synthesizers = synthesizers

    def synthesize(self, root: Path, instruction: RepairInstruction, plan: RepairPlan) -> tuple[PatchCandidate, ...]:
        for synthesizer in self.synthesizers:
            candidates = synthesizer.synthesize(root, instruction, plan)
            if candidates:
                return candidates
        return ()

    @property
    def last_compilation(self):
        """Expose the L8.5 compilation so diagnosis can report why grounding failed."""
        compilations = [getattr(item, "last_compilation", None) for item in self.synthesizers]
        return next((item for item in reversed(compilations) if item is not None), None)


def default_synthesizer() -> PatchSynthesizer:
    """Minimal literal edits first; semantic changes fall through to L8.5."""
    from .semantic_change_compiler_l85_experiment import SemanticPatchSynthesizer

    return CompositePatchSynthesizer((MinimalPatchSynthesizer(), SemanticPatchSynthesizer()))


def _validate_candidate(candidate: PatchCandidate) -> VerificationResult:
    if not candidate.intent.file.endswith(".py"):
        return VerificationResult(True, (), "non-Python patch", 0)
    with tempfile.TemporaryDirectory(prefix="patch-static-") as directory:
        path = Path(directory) / Path(candidate.intent.file).name
        path.write_text(candidate.updated, encoding="utf-8")
        completed = run_text(["python", "-m", "py_compile", str(path)], timeout=30)
        return VerificationResult(completed.returncode == 0, ("python", "-m", "py_compile", candidate.intent.file), combined_output(completed, 4000), completed.returncode)


def apply_verified_candidates(root: Path, candidates: tuple[PatchCandidate, ...], regression: VerificationResult) -> PatchResult:
    if not candidates or not regression.passed:
        return PatchResult(False, (), "検証不合格のためパッチを反映しませんでした。")
    backups: dict[Path, str | None] = {}
    try:
        for candidate in candidates:
            path = (root / candidate.intent.file).resolve()
            if root.resolve() not in path.parents:
                return PatchResult(False, (), "workspace外のパスを拒否しました。")
            if path not in backups:
                # ``None`` marks a file the patch creates, so a rollback removes it.
                backups[path] = path.read_text(encoding="utf-8", errors="replace") if path.is_file() else None
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(candidate.updated, encoding="utf-8")
        return PatchResult(True, tuple(dict.fromkeys(candidate.intent.file for candidate in candidates)), "検証済みの最小差分を反映しました。")
    except OSError as error:
        for path, content in backups.items():
            if content is None:
                path.unlink(missing_ok=True)
            else:
                path.write_text(content, encoding="utf-8")
        return PatchResult(False, (), f"反映失敗のためロールバックしました: {error}")


def run_autonomous_patch_workflow(root: Path, instruction: RepairInstruction, plan: RepairPlan, synthesizer: PatchSynthesizer | None = None, verifier: RegressionVerifier | None = None) -> AutonomousPatchResult:
    synth = synthesizer or default_synthesizer()
    candidates = synth.synthesize(root, instruction, plan)
    intents = tuple(candidate.intent for candidate in candidates)
    if not candidates:
        return AutonomousPatchResult("synthesis_unavailable", intents, candidates, PatchResult(False, (), "提案から安全に生成できる最小差分がありません。"), None, None, "意味要求をCodeChangeIRへ接地できませんでした（対象/anchor/識別子が未解決）。")
    static_results = tuple(_validate_candidate(candidate) for candidate in candidates)
    static = static_results[0] if len(static_results) == 1 else VerificationResult(all(item.passed for item in static_results), ("python", "-m", "py_compile"), "\n".join(item.output for item in static_results), 0 if all(item.passed for item in static_results) else 1)
    if not static.passed:
        return AutonomousPatchResult("static_validation_failed", intents, candidates, PatchResult(False, (), "構文検証に失敗したため反映しませんでした。"), static, None, "SYNTAX_FAILURE")
    verifier = verifier or RegressionVerifier()
    with tempfile.TemporaryDirectory(prefix="repair-worktree-") as directory:
        worktree = Path(directory) / "workspace"
        shutil.copytree(root, worktree, ignore=shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache"))
        staged = apply_verified_candidates(worktree, candidates, VerificationResult(True, (), "staged", 0))
        if not staged.applied:
            return AutonomousPatchResult("temporary_apply_failed", intents, candidates, staged, static, None, "TEMPORARY_APPLY_FAILED")
        regression = verifier.verify(worktree, instruction.acceptance_tests)
    if not regression.passed:
        return AutonomousPatchResult("regression_failed", intents, candidates, PatchResult(False, (), "回帰テスト不合格のため反映しませんでした。"), static, regression, "REGRESSION")
    patch = apply_verified_candidates(root, candidates, regression)
    return AutonomousPatchResult("accepted" if patch.applied else "apply_failed", intents, candidates, patch, static, regression, patch.message)


def format_autonomous_patch(result: AutonomousPatchResult) -> str:
    lines = ["## Autonomous Patch Synthesis", "", f"【状態】`{result.status}`", f"【判断】{result.message}", "", "【Patch Intent】"]
    intents = [f"- `{item.operation}` `{item.file}` symbol=`{item.symbol or '-'}`: {item.objective}" for item in result.intents] or ["- なし"]
    lines.extend(intents)
    if result.candidates:
        lines.extend(["", "【差分】", "```diff", *[candidate.diff for candidate in result.candidates], "```"])
    for label, verification in (("静的検証", result.static_validation), ("回帰検証", result.regression)):
        if verification:
            lines.extend(["", f"【{label}】", f"- passed: `{verification.passed}`", f"- return code: `{verification.return_code}`", "```text", verification.output, "```"])
    return "\n".join(lines)
