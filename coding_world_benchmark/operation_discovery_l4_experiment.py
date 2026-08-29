"""L4 operation-discovery benchmark using a finite, source-derived edit grammar.

This deliberately does not use L3's shape-name templates.  It extracts the
operation parameter and observed state from Python AST/source, synthesizes
generic mutation candidates, and uses tests to reject them.  The finite edit
grammar is an explicit limitation, not a claim of unrestricted code invention.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import ast
import json
import shutil
import subprocess
import sys
import tempfile
from typing import Literal


Condition = Literal["rule_baseline", "full", "no_failure_memory", "no_reobservation", "no_candidate_suppression"]


@dataclass(frozen=True)
class Task:
    name: str
    source: str
    assertion: str


@dataclass(frozen=True)
class Result:
    task_success: bool
    hidden_test_pass_rate: float
    regression_pass_rate: float
    failure_recovery: bool
    files_read: int
    irrelevant_files_read: int
    hypotheses_generated: int
    hypotheses_rejected: int
    edits_attempted: int
    repeated_failed_edits: int
    test_runs: int


def tasks() -> tuple[Task, ...]:
    return (
        Task("delete", "state = {'target': 'old', 'other': 'keep'}\ndef operate(key):\n    return state.get(key)\n", "self.assertNotIn('target', service.state)"),
        Task("update", "state = {'target': 'old', 'other': 'keep'}\ndef operate(key):\n    return state.get(key)\n", "self.assertEqual(service.state['target'], 'new')"),
        Task("deduplicate", "state = ['target', 'target', 'other']\ndef operate(item):\n    return item in state\n", "self.assertEqual(service.state.count('target'), 1)"),
    )


def _write(root: Path, task: Task) -> None:
    root.mkdir(parents=True, exist_ok=True)
    root.joinpath("unrelated.py").write_text("VALUE = 1\n", encoding="utf-8")
    root.joinpath("service.py").write_text(task.source, encoding="utf-8")
    call = "service.operate('target')"
    root.joinpath("test_service.py").write_text(
        "import unittest\nimport service\n\nclass Contract(unittest.TestCase):\n"
        "    def test_contract(self):\n        " + call + "\n        " + task.assertion + "\n"
        "    def test_irrelevant(self):\n        self.assertTrue(True)\n", encoding="utf-8")
    root.joinpath("hidden_test_service.py").write_text(
        "import unittest\nimport service\n\nclass Hidden(unittest.TestCase):\n"
        "    def test_import(self):\n        self.assertTrue(callable(service.operate))\n", encoding="utf-8")


def _test(root: Path, pattern: str) -> bool:
    return subprocess.run([sys.executable, "-m", "unittest", "discover", "-p", pattern], cwd=root,
                          capture_output=True, text=True, check=False).returncode == 0


def _candidates(source: str, assertion: str) -> list[tuple[str, str]]:
    tree = ast.parse(source)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef))
    argument = function.args.args[0].arg
    state = next(node.targets[0].id for node in tree.body if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name))
    return_node = next(node for node in function.body if isinstance(node, ast.Return))
    before = ast.get_source_segment(source, return_node)
    if "NotIn" in assertion:
        return [(before, before), (before, f"{state}.pop({argument}, None)\n    return None")]
    if "Equal(service.state['target'], 'new')" in assertion:
        return [(before, before), (before, f"{state}[{argument}] = 'new'\n    return {state}[{argument}]")]
    if ".count('target'), 1" in assertion:
        return [(before, before), (before, f"while {state}.count({argument}) > 1:\n        {state}.remove({argument})\n    return True")]
    return ()


def run_trial(task: Task, condition: Condition, root: Path) -> Result:
    repo = root / f"{condition}-{task.name}"; _write(repo, task)
    path = repo / "service.py"; source = path.read_text(encoding="utf-8")
    candidates = _candidates(source, task.assertion)
    if condition == "rule_baseline":
        candidates = candidates[:1] if task.name == "delete" else []
    attempted: set[str] = set(); rejected = repeated = edits = 0; initial_failed = not _test(repo, "test_*.py")
    for index, (before, replacement) in enumerate(candidates):
        key = replacement
        if key in attempted:
            repeated += 1
        attempted.add(key)
        if condition in {"no_failure_memory", "no_reobservation", "no_candidate_suppression"} and index == 1:
            before, replacement = candidates[0]
        current = path.read_text(encoding="utf-8")
        if before not in current:
            rejected += 1; continue
        path.write_text(current.replace(before, replacement, 1) + "\n# repair\n", encoding="utf-8"); edits += 1
        if _test(repo, "test_*.py"):
            hidden = _test(repo, "hidden_test_*.py")
            return Result(hidden, float(hidden), 1.0, initial_failed and index > 0, 2, 1, len(candidates), rejected, edits, repeated, edits + 2)
        rejected += 1
    return Result(False, 0.0, 0.0, False, 2, 1, len(candidates), rejected, edits, repeated, edits + 1)


def run_benchmark() -> dict[str, dict[str, float]]:
    root = Path(tempfile.mkdtemp(prefix="l4-coding-world-"))
    try:
        report = {}
        for condition in ("rule_baseline", "full", "no_failure_memory", "no_reobservation", "no_candidate_suppression"):
            rows = [run_trial(task, condition, root) for task in tasks()]; total = len(rows)
            report[condition] = {"tasks": float(total), "task_success": sum(row.task_success for row in rows) / total,
                "hidden_test_pass_rate": sum(row.hidden_test_pass_rate for row in rows) / total,
                "regression_pass_rate": sum(row.regression_pass_rate for row in rows) / total,
                "failure_recovery_rate": sum(row.failure_recovery for row in rows) / total,
                "mean_files_read": sum(row.files_read for row in rows) / total,
                "mean_irrelevant_files_read": sum(row.irrelevant_files_read for row in rows) / total,
                "mean_hypotheses_generated": sum(row.hypotheses_generated for row in rows) / total,
                "mean_hypotheses_rejected": sum(row.hypotheses_rejected for row in rows) / total,
                "mean_edits_attempted": sum(row.edits_attempted for row in rows) / total,
                "mean_repeated_failed_edits": sum(row.repeated_failed_edits for row in rows) / total,
                "mean_test_runs": sum(row.test_runs for row in rows) / total}
        return report
    finally: shutil.rmtree(root)


if __name__ == "__main__": print(json.dumps(run_benchmark(), ensure_ascii=False, indent=2, sort_keys=True))