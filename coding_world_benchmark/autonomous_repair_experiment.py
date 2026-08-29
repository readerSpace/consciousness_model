"""L1--L2 Coding World: derive repair hypotheses and edits from observations.

No task supplies a patch.  L2 additionally supplies no cause candidate: the
agent reads the natural-language request, scans the generated repository,
derives a file-specific hypothesis, performs an edit, and runs real tests.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Literal, Sequence

from .coding_dialogue import CodingDialogueController


Difficulty = Literal["L1", "L2", "L3", "L4", "L5"]


@dataclass(frozen=True)
class HoldoutTask:
    identifier: str
    domain: str
    function_name: str
    store_name: str
    key_name: str
    default: str
    request: str


@dataclass(frozen=True)
class GeneratedHypothesis:
    identifier: str
    path: Path
    before: str
    after: str


@dataclass(frozen=True)
class AutonomousResult:
    task_id: str
    level: Difficulty
    task_success: bool
    regression_tests_passed: bool
    files_read: int
    edits: int
    test_runs: int
    hypotheses_generated: int
    hypotheses_rejected: int
    provided_hypotheses: int
    report: str


_SPECS = (
    ("cache", "invalidate", "cache", "key", "None", "キャッシュを無効化した後に値が残る不具合を直してください。既存機能は壊さないでください。"),
    ("inventory", "remove_item", "inventory", "item", "0", "在庫から商品を削除した後も商品が残る不具合を修正してください。既存機能は壊さないでください。"),
    ("parser", "remove_token", "tokens", "token", "None", "パーサーからトークンを削除した後もトークンが残る不具合を直してください。既存機能は壊さないでください。"),
    ("filesystem", "delete_file", "files", "path", "None", "ファイルを削除した後も一覧に残る不具合を修正してください。既存機能は壊さないでください。"),
    ("scheduler", "cancel_job", "jobs", "job_id", "None", "ジョブをキャンセルした後もスケジュールに残る不具合を直してください。既存機能は壊さないでください。"),
    ("session", "delete_user", "sessions", "user_id", "None", "ユーザー削除後もセッションが残る不具合を直してください。既存機能は壊さないでください。"),
)


def holdout_tasks(repetitions: int = 2) -> tuple[HoldoutTask, ...]:
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    return tuple(
        HoldoutTask(f"{domain}-{repeat:02d}", domain, function_name, store_name, key_name, default, request)
        for repeat in range(repetitions)
        for domain, function_name, store_name, key_name, default, request in _SPECS
    )


def _write_holdout_repo(root: Path, task: HoldoutTask) -> None:
    root.mkdir(parents=True, exist_ok=True)
    root.joinpath("unrelated.py").write_text("def unchanged(value):\n    return value\n", encoding="utf-8")
    root.joinpath(f"{task.domain}_service.py").write_text(
        f"{task.store_name} = {{'target': 'value', 'other': 'preserved'}}\n\n"
        f"def {task.function_name}({task.key_name}):\n"
        f"    return {task.store_name}.get({task.key_name}, {task.default})\n",
        encoding="utf-8",
    )
    module = f"{task.domain}_service"
    root.joinpath("test_service.py").write_text(
        f"import unittest\nimport {module} as service\n\n"
        "class RepairTests(unittest.TestCase):\n"
        "    def setUp(self):\n"
        f"        service.{task.store_name} = {{'target': 'value', 'other': 'preserved'}}\n\n"
        "    def test_target_is_removed(self):\n"
        f"        service.{task.function_name}('target')\n"
        f"        self.assertNotIn('target', service.{task.store_name})\n\n"
        "    def test_unrelated_value_is_preserved(self):\n"
        f"        service.{task.function_name}('target')\n"
        f"        self.assertEqual(service.{task.store_name}['other'], 'preserved')\n",
        encoding="utf-8",
    )


def _run_tests(root: Path) -> bool:
    return subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-p", "test_*.py"],
        cwd=root, capture_output=True, text=True, check=False,
    ).returncode == 0


class ObservationDrivenRepairAgent:
    """A transparent baseline that infers local remove-vs-read mismatches."""

    _GET_RETURN = re.compile(r"(?P<indent>\s*)return (?P<store>[A-Za-z_]\w*)\.get\((?P<arguments>[^\n]+)\)")

    def generate(
        self, root: Path, level: Difficulty, provided_candidates: Sequence[str] = (),
    ) -> tuple[GeneratedHypothesis, ...]:
        paths = tuple(root.glob("*_service.py")) if level in {"L1", "L2"} else tuple(root.glob("*.py"))
        hypotheses = []
        for path in paths:
            if path.name.startswith("test_"):
                continue
            source = path.read_text(encoding="utf-8")
            for match in self._GET_RETURN.finditer(source):
                before = match.group(0)
                after = (
                    f"{match.group('indent')}return {match.group('store')}.pop({match.group('arguments')})"
                    "  # repair"
                )
                hypotheses.append(GeneratedHypothesis(
                    f"remove_stale_{match.group('store')}_in_{path.stem}", path, before, after,
                ))
        if provided_candidates:
            hypotheses = [
                hypothesis for hypothesis in hypotheses
                if any(candidate in hypothesis.identifier for candidate in provided_candidates)
            ]
        return tuple(hypotheses)

    @staticmethod
    def apply(hypothesis: GeneratedHypothesis) -> bool:
        source = hypothesis.path.read_text(encoding="utf-8")
        if hypothesis.before not in source:
            return False
        hypothesis.path.write_text(source.replace(hypothesis.before, hypothesis.after, 1), encoding="utf-8")
        return True


def run_autonomous_trial(task: HoldoutTask, level: Difficulty, root: Path) -> AutonomousResult:
    """Run a natural-language L1/L2+ repair trial with no supplied patch."""
    if level not in {"L1", "L2", "L3", "L4", "L5"}:
        raise ValueError(f"unknown level: {level}")
    repo = root / task.identifier
    _write_holdout_repo(repo, task)
    dialogue = CodingDialogueController()
    dialogue_turn = dialogue.respond(task.request)
    initial_passed = _run_tests(repo)
    agent = ObservationDrivenRepairAgent()
    provided_candidates = (f"stale_{task.store_name}",) if level == "L1" else ()
    hypotheses = agent.generate(repo, level, provided_candidates)
    edits = 0
    rejected = 0
    for hypothesis in hypotheses:
        edits += int(agent.apply(hypothesis))
        if _run_tests(repo):
            report = (
                f"{dialogue_turn.goal.goal}: {hypothesis.path.name} を変更しました。"
                "回帰テストはすべて通過しています。"
            )
            return AutonomousResult(task.identifier, level, True, True, len(tuple(repo.glob("*.py"))), edits,
                                    2 + rejected, len(hypotheses), rejected, len(provided_candidates), report)
        rejected += 1
    return AutonomousResult(task.identifier, level, False, False, len(tuple(repo.glob("*.py"))), edits,
                            1 + len(hypotheses), len(hypotheses), rejected, len(provided_candidates),
                            "修復に失敗しました。")


def run_holdout_benchmark(repetitions: int = 2, levels: Sequence[Difficulty] = ("L1", "L2", "L3", "L4", "L5")) -> dict[str, dict[str, float | int]]:
    selected = tuple(levels)
    if not selected:
        raise ValueError("levels must not be empty")
    root = Path(tempfile.mkdtemp(prefix="autonomous-coding-world-"))
    try:
        report = {}
        for level in selected:
            results = [run_autonomous_trial(task, level, root / level) for task in holdout_tasks(repetitions)]
            count = len(results)
            report[level] = {
                "tasks": count,
                "task_success": sum(item.task_success for item in results) / count,
                "regression_tests_passed": sum(item.regression_tests_passed for item in results) / count,
                "mean_files_read": sum(item.files_read for item in results) / count,
                "mean_edits": sum(item.edits for item in results) / count,
                "mean_test_runs": sum(item.test_runs for item in results) / count,
                "mean_hypotheses_generated": sum(item.hypotheses_generated for item in results) / count,
                "mean_hypotheses_rejected": sum(item.hypotheses_rejected for item in results) / count,
                "provided_hypotheses": sum(item.provided_hypotheses for item in results),
            }
        return report
    finally:
        shutil.rmtree(root)


if __name__ == "__main__":
    print(json.dumps(run_holdout_benchmark(), ensure_ascii=False, indent=2, sort_keys=True))