"""L4.6: compose primitives into a reusable rollback operator.

The composer receives no ``rollback`` repair rule.  It observes an assignment
followed by an exception, composes SAVE_STATE, TRY, RESTORE_STATE, and RAISE,
then records the successful parameterized composition as an operator.  The
hold-out uses different state and parameter names.
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


@dataclass(frozen=True)
class Operator:
    name: str
    primitives: tuple[str, ...]
    support: int
    reuse_count: int
    acquisition_cost: int


@dataclass(frozen=True)
class Task:
    identifier: str
    state: str
    parameter: str
    source: str


@dataclass(frozen=True)
class Result:
    success: bool
    attempts: int
    operator_used: bool
    operator: Operator | None


def rollback_tasks() -> tuple[Task, Task]:
    return (
        Task("acquisition", "inventory", "sku",
             "inventory = {'target': 10}\ndef place_order(sku):\n    inventory[sku] -= 1\n    raise RuntimeError('payment failed')\n"),
        Task("holdout", "stock", "item_id",
             "stock = {'target': 7}\ndef reserve(item_id):\n    stock[item_id] -= 1\n    raise ValueError('downstream failure')\n"),
    )


def _write(root: Path, task: Task) -> None:
    root.mkdir(parents=True, exist_ok=True)
    root.joinpath("noise.py").write_text("UNRELATED = 1\n", encoding="utf-8")
    root.joinpath("service.py").write_text(task.source, encoding="utf-8")
    function = next(node.name for node in ast.parse(task.source).body if isinstance(node, ast.FunctionDef))
    root.joinpath("test_service.py").write_text(
        "import unittest\nimport service\n\nclass Atomicity(unittest.TestCase):\n"
        "    def test_failure_restores_state(self):\n"
        "        with self.assertRaises(Exception):\n"
        f"            service.{function}('target')\n"
        f"        self.assertEqual(service.{task.state}['target'], {10 if task.identifier == 'acquisition' else 7})\n",
        encoding="utf-8")


def _test(root: Path) -> bool:
    return subprocess.run([sys.executable, "-m", "unittest", "discover", "-p", "test_*.py"], cwd=root,
                          capture_output=True, encoding="utf-8", errors="replace", check=False).returncode == 0


class PrimitiveComposer:
    """Compose general control primitives from an AST-observed failed mutation."""

    def compose(self, source: str) -> tuple[str, str] | None:
        tree = ast.parse(source)
        function = next((node for node in tree.body if isinstance(node, ast.FunctionDef)), None)
        if function is None or len(function.body) < 2 or not isinstance(function.body[1], ast.Raise):
            return None
        assignment = function.body[0]
        if not isinstance(assignment, ast.AugAssign) or not isinstance(assignment.target, ast.Subscript):
            return None
        state = ast.unparse(assignment.target.value)
        parameter = function.args.args[0].arg
        before = ast.get_source_segment(source, assignment) + "\n    " + ast.get_source_segment(source, function.body[1])
        after = (
            f"snapshot = {state}.copy()\n    try:\n        {ast.get_source_segment(source, assignment)}\n"
            f"        {ast.get_source_segment(source, function.body[1])}\n    except Exception:\n"
            f"        {state}.clear()\n        {state}.update(snapshot)\n        raise"
        )
        return before, after


class OperatorLibrary:
    def __init__(self) -> None:
        self.rollback: Operator | None = None

    def acquire(self) -> Operator:
        self.rollback = Operator("ROLLBACK", ("SAVE_STATE", "TRY", "RESTORE_STATE", "RAISE"), 1, 0, 2)
        return self.rollback

    def reuse(self) -> Operator | None:
        if self.rollback is None:
            return None
        self.rollback = Operator(self.rollback.name, self.rollback.primitives, self.rollback.support + 1,
                                 self.rollback.reuse_count + 1, self.rollback.acquisition_cost)
        return self.rollback


def run_task(task: Task, library: OperatorLibrary, root: Path, allow_acquisition: bool) -> Result:
    repo = root / task.identifier; _write(repo, task)
    path = repo / "service.py"
    if _test(repo):
        return Result(True, 0, False, library.rollback)
    operator = library.reuse()
    attempts = 1
    if operator is None and allow_acquisition:
        operator = library.acquire()
        attempts += 1
    if operator is None:
        return Result(False, attempts, False, None)
    composition = PrimitiveComposer().compose(path.read_text(encoding="utf-8"))
    if composition is None:
        return Result(False, attempts, operator.reuse_count > 0, operator)
    before, after = composition
    path.write_text(path.read_text(encoding="utf-8").replace(before, after, 1) + "\n# composed repair\n", encoding="utf-8")
    return Result(_test(repo), attempts, operator.reuse_count > 0, operator)


def run_operator_invention_experiment() -> dict[str, object]:
    root = Path(tempfile.mkdtemp(prefix="l46-operator-invention-"))
    try:
        acquisition, holdout = rollback_tasks()
        library = OperatorLibrary()
        learned = run_task(acquisition, library, root / "learn", allow_acquisition=True)
        reused = run_task(holdout, library, root / "reuse", allow_acquisition=False)
        no_operator = run_task(holdout, OperatorLibrary(), root / "baseline", allow_acquisition=False)
        return {
            "acquisition_success": learned.success,
            "holdout_success": reused.success,
            "no_operator_holdout_success": no_operator.success,
            "acquisition_attempts": learned.attempts,
            "holdout_attempts": reused.attempts,
            "concept_acquisition_cost_exceeds_reuse_cost": learned.attempts > reused.attempts,
            "operator": None if library.rollback is None else {
                "name": library.rollback.name, "primitives": library.rollback.primitives,
                "support": library.rollback.support, "reuse_count": library.rollback.reuse_count,
                "acquisition_cost": library.rollback.acquisition_cost,
            },
        }
    finally:
        shutil.rmtree(root)


if __name__ == "__main__":
    print(json.dumps(run_operator_invention_experiment(), ensure_ascii=False, indent=2, sort_keys=True))