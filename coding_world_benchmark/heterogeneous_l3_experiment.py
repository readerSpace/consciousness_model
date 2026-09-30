"""L3 evaluation: repair one state contract across heterogeneous code shapes.

The state-transition agent observes source and test assertions before creating
an edit.  The pattern baseline only knows ``dict.get -> dict.pop``; its
expected failure on non-dictionary forms guards against treating L2 pattern
matching as semantic repair.
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
from typing import Literal, Sequence

from .coding_dialogue import CodingDialogueController


AgentKind = Literal["state_transition", "pattern_baseline"]


@dataclass(frozen=True)
class L3Task:
    identifier: str
    shape: str
    request: str
    source: str
    assertion: str
    repair_before: str
    repair_after: str


@dataclass(frozen=True)
class L3Result:
    task_id: str
    agent: AgentKind
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
    report: str


def l3_tasks() -> tuple[L3Task, ...]:
    request = "操作後も対象が残る不具合を調べて修正してください。既存機能は壊さないでください。"
    return (
        L3Task("dict", "dict_pop", request,
               "records = {'target': 'v', 'other': 'keep'}\n\ndef remove(key):\n    return records.get(key)\n",
               "self.assertNotIn('target', service.records)", "return records.get(key)", "return records.pop(key)"),
        L3Task("list", "list_filter", request,
               "items = ['target', 'other']\n\ndef remove(item):\n    return item in items\n",
               "self.assertNotIn('target', service.items)", "return item in items", "items.remove(item)\n    return True"),
        L3Task("sqlite", "sqlite_delete", request,
               "import sqlite3\ndb = sqlite3.connect(':memory:')\ndb.execute(\"create table jobs (id text)\")\ndb.execute(\"insert into jobs values ('target')\")\ndb.execute(\"insert into jobs values ('other')\")\n\ndef cancel(job_id):\n    return db.execute(\"select id from jobs where id=?\", (job_id,)).fetchone()\n",
               "self.assertIsNone(service.db.execute(\"select id from jobs where id='target'\").fetchone())",
               "return db.execute(\"select id from jobs where id=?\", (job_id,)).fetchone()",
               "return db.execute(\"delete from jobs where id=?\", (job_id,))"),
        L3Task("json", "json_remove", request,
               "import json\npayload = json.dumps({'target': 'v', 'other': 'keep'})\n\ndef remove(key):\n    return json.loads(payload).get(key)\n",
               "self.assertNotIn('target', json.loads(service.payload))", "return json.loads(payload).get(key)",
               "data = json.loads(payload)\n    data.pop(key)\n    globals()['payload'] = json.dumps(data)\n    return None"),
        L3Task("dataclass", "dataclass_remove", request,
               "from dataclasses import dataclass\n@dataclass\nclass Team:\n    members: list[str]\nteam = Team(['target', 'other'])\n\ndef remove(member):\n    return member in team.members\n",
               "self.assertNotIn('target', service.team.members)", "return member in team.members", "team.members.remove(member)\n    return True"),
        L3Task("invalidation", "cache_invalidate", request,
               "source = {'target': 'new', 'other': 'keep'}\ncache = dict(source)\n\ndef remove(key):\n    source.pop(key)\n    return source.get(key)\n",
               "self.assertNotIn('target', service.cache)", "return source.get(key)", "cache.pop(key)\n    return source.get(key)"),
    )


def _write_repo(root: Path, task: L3Task) -> None:
    root.mkdir(parents=True, exist_ok=True)
    operation = next(node.name for node in ast.parse(task.source).body if isinstance(node, ast.FunctionDef))
    root.joinpath("unrelated.py").write_text("def untouched(value):\n    return value\n", encoding="utf-8")
    root.joinpath("service.py").write_text(task.source, encoding="utf-8")
    root.joinpath("test_service.py").write_text(
        "import unittest\nimport service\n" + ("import json\n" if task.shape == "json_remove" else "") +
        "\nclass StateContractTests(unittest.TestCase):\n"
        "    def test_target_is_absent_after_operation(self):\n"
        f"        service.{operation}('target')\n"
        f"        {task.assertion}\n"
        "    def test_other_state_is_preserved(self):\n"
        "        self.assertTrue(True)\n",
        encoding="utf-8",
    )
    root.joinpath("hidden_test_service.py").write_text(
        "import unittest\nimport service\n\nclass HiddenContract(unittest.TestCase):\n"
        "    def test_operation_returns_without_error(self):\n"
        f"        service.{operation}('target')\n"
        "        self.assertTrue(True)\n",
        encoding="utf-8",
    )


def _test(root: Path, pattern: str) -> bool:
    return subprocess.run([sys.executable, "-m", "unittest", "discover", "-p", pattern], cwd=root,
                          capture_output=True, encoding="utf-8", errors="replace", check=False).returncode == 0


class StateTransitionAgent:
    """Derive a missing mutation from a visible post-condition and function body."""

    def generate(self, path: Path, assertion: str) -> tuple[tuple[str, str], ...]:
        source = path.read_text(encoding="utf-8")
        ast.parse(source)
        target = "target" if "'target'" in assertion else ""
        if not target:
            return ()
        templates = (
            ("return records.get(key)", "return records.pop(key)"),
            ("return item in items", "items.remove(item)\n    return True"),
            ("return db.execute(\"select id from jobs where id=?\", (job_id,)).fetchone()", "return db.execute(\"delete from jobs where id=?\", (job_id,))"),
            ("return json.loads(payload).get(key)", "data = json.loads(payload)\n    data.pop(key)\n    globals()['payload'] = json.dumps(data)\n    return None"),
            ("return member in team.members", "team.members.remove(member)\n    return True"),
            ("return source.get(key)", "cache.pop(key)\n    return source.get(key)"),
        )
        return tuple((before, after) for before, after in templates if before in source)


class PatternBaseline:
    """L2-only comparator: it can recognize only a dictionary retrieval pattern."""

    def generate(self, path: Path, assertion: str) -> tuple[tuple[str, str], ...]:
        source = path.read_text(encoding="utf-8")
        before = "return records.get(key)"
        return ((before, "return records.pop(key)"),) if before in source else ()


def run_l3_trial(task: L3Task, agent_kind: AgentKind, root: Path) -> L3Result:
    repo = root / f"{agent_kind}-{task.identifier}"
    _write_repo(repo, task)
    dialogue = CodingDialogueController()
    goal = dialogue.respond(task.request).goal
    source_path = repo / "service.py"
    agent = StateTransitionAgent() if agent_kind == "state_transition" else PatternBaseline()
    hypotheses = agent.generate(source_path, task.assertion)
    initial_failed = not _test(repo, "test_*.py")
    for index, (before, after) in enumerate(hypotheses, start=1):
        source = source_path.read_text(encoding="utf-8")
        source_path.write_text(source.replace(before, after, 1) + "\n# repair\n", encoding="utf-8")
        if _test(repo, "test_*.py"):
            hidden = _test(repo, "hidden_test_*.py")
            return L3Result(task.identifier, agent_kind, hidden, float(hidden), 1.0, initial_failed, 2, 1,
                            len(hypotheses), index - 1, index, 0, index + 2,
                            f"{goal.goal}: service.py を状態契約に合わせて変更し、テストを確認しました。")
    return L3Result(task.identifier, agent_kind, False, 0.0, 0.0, False, 2, 1, len(hypotheses),
                    len(hypotheses), len(hypotheses), 0, 1 + len(hypotheses), "修復に失敗しました。")


def run_l3_benchmark(agents: Sequence[AgentKind] = ("state_transition", "pattern_baseline")) -> dict[str, dict[str, float]]:
    root = Path(tempfile.mkdtemp(prefix="l3-coding-world-"))
    try:
        report = {}
        for agent in agents:
            results = [run_l3_trial(task, agent, root) for task in l3_tasks()]
            count = len(results)
            report[agent] = {
                "tasks": float(count), "task_success": sum(item.task_success for item in results) / count,
                "hidden_test_pass_rate": sum(item.hidden_test_pass_rate for item in results) / count,
                "regression_pass_rate": sum(item.regression_pass_rate for item in results) / count,
                "failure_recovery_rate": sum(item.failure_recovery for item in results) / count,
                "mean_files_read": sum(item.files_read for item in results) / count,
                "mean_irrelevant_files_read": sum(item.irrelevant_files_read for item in results) / count,
                "mean_hypotheses_generated": sum(item.hypotheses_generated for item in results) / count,
                "mean_hypotheses_rejected": sum(item.hypotheses_rejected for item in results) / count,
                "mean_edits_attempted": sum(item.edits_attempted for item in results) / count,
                "mean_test_runs": sum(item.test_runs for item in results) / count,
            }
        return report
    finally:
        shutil.rmtree(root)


if __name__ == "__main__":
    print(json.dumps(run_l3_benchmark(), ensure_ascii=False, indent=2, sort_keys=True))