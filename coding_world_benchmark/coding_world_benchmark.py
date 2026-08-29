"""Controlled Coding World benchmark for closed-loop software repair.

The benchmark generates isolated Python repositories and lets a deterministic
agent inspect failures, apply patches, and rerun their real unit tests.  It is
an infrastructure benchmark: external agent results (such as Codex) can be
recorded with the same schema, but are never fabricated by this module.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import shutil
import subprocess
import sys
import tempfile
from typing import Callable, Mapping, Sequence


CONDITIONS = (
    "B_react",
    "C_full",
    "D_no_failure_memory",
    "E_no_exploration",
    "F_no_reobservation",
    "G_no_candidate_suppression",
)


@dataclass(frozen=True)
class CodingTask:
    identifier: str
    needs_followup: bool
    misleading_failure: bool


@dataclass(frozen=True)
class TrialResult:
    task_id: str
    condition: str
    task_success: bool
    hidden_tests_passed: bool
    regression_tests_passed: bool
    iterations: int
    files_read: int
    files_modified: int
    test_runs: int
    failed_hypotheses: int
    repeated_failed_hypotheses: int
    first_patch_failed: bool
    recovered_after_first_failure: bool


def generate_tasks(count: int = 50) -> tuple[CodingTask, ...]:
    """Create reproducible, partially observed deletion-cascade repair tasks."""
    if count < 1:
        raise ValueError("count must be positive")
    return tuple(
        CodingTask(
            identifier=f"cascade-{index:03d}",
            needs_followup=index % 2 == 0,
            misleading_failure=index % 5 == 0,
        )
        for index in range(count)
    )


def _write_repo(root: Path, task: CodingTask) -> None:
    root.mkdir(parents=True, exist_ok=True)
    root.joinpath("app.py").write_text(
        "SESSIONS = {'ada': ['s1'], 'bert': ['s2']}\n"
        "AUDIT_LOG = []\n\n"
        "def delete_user(user_id):\n"
        "    # BUG: a user may disappear while its sessions remain.\n"
        "    SESSIONS.pop(user_id, None) if False else None\n"
        "    AUDIT_LOG.append(user_id) if False else None\n"
        "    return True\n",
        encoding="utf-8",
    )
    followup = (
        "        self.assertEqual(app.AUDIT_LOG, ['ada'])\n" if task.needs_followup else ""
    )
    hidden = "        self.assertNotIn('ada', app.SESSIONS)\n"
    root.joinpath("test_app.py").write_text(
        "import unittest\nimport app\n\n"
        "class DeleteUserTests(unittest.TestCase):\n"
        "    def setUp(self):\n"
        "        app.SESSIONS = {'ada': ['s1'], 'bert': ['s2']}\n"
        "        app.AUDIT_LOG = []\n\n"
        "    def test_sessions_are_removed(self):\n"
        "        app.delete_user('ada')\n"
        "        self.assertNotIn('ada', app.SESSIONS)\n"
        f"{followup}"
        "    def test_other_user_is_unchanged(self):\n"
        "        app.delete_user('ada')\n"
        "        self.assertIn('bert', app.SESSIONS)\n",
        encoding="utf-8",
    )
    root.joinpath("hidden_test_app.py").write_text(
        "import unittest\nimport app\n\n"
        "class HiddenDeleteUserTests(unittest.TestCase):\n"
        "    def setUp(self):\n"
        "        app.SESSIONS = {'ada': ['s1'], 'bert': ['s2']}\n"
        "        app.AUDIT_LOG = []\n\n"
        "    def test_no_orphans_or_regression(self):\n"
        "        app.delete_user('ada')\n"
        f"{hidden}"
        "        self.assertEqual(app.SESSIONS['bert'], ['s2'])\n",
        encoding="utf-8",
    )


def _run_tests(root: Path, hidden: bool = False) -> bool:
    pattern = "hidden_test*.py" if hidden else "test_*.py"
    result = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-p", pattern],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0


def _apply_patch(root: Path, hypothesis: str) -> bool:
    path = root / "app.py"
    source = path.read_text(encoding="utf-8")
    replacements = {
        "wrong_database_cascade": (
            "    # Database cascade assumed.\n    return True\n",
            "    return True\n",
        ),
        "remove_sessions": (
            "    SESSIONS.pop(user_id, None)\n",
            "    SESSIONS.pop(user_id, None) if False else None\n",
        ),
        "record_audit": (
            "    AUDIT_LOG.append(user_id)\n",
            "    AUDIT_LOG.append(user_id) if False else None\n",
        ),
    }
    replacement = replacements.get(hypothesis)
    if replacement is None or replacement[1] not in source:
        return False
    path.write_text(source.replace(replacement[1], replacement[0]), encoding="utf-8")
    return True


def _candidates(task: CodingTask, condition: str, attempted: set[str], failure_seen: bool) -> Sequence[str]:
    full = condition == "C_full"
    reobserve = full or condition not in {"F_no_reobservation"}
    explore = full or condition not in {"E_no_exploration"}
    suppress = full or condition not in {"G_no_candidate_suppression"}
    remember = full or condition not in {"D_no_failure_memory"}
    if condition == "B_react":
        return ("wrong_database_cascade", "remove_sessions", "record_audit")
    if not explore:
        return ("remove_sessions",)
    if failure_seen and not reobserve:
        return ("wrong_database_cascade",)
    candidates = ["remove_sessions", "record_audit"]
    if task.misleading_failure and not failure_seen:
        candidates.insert(0, "wrong_database_cascade")
    if failure_seen and not suppress:
        candidates.insert(0, "wrong_database_cascade")
    if failure_seen and not remember:
        candidates.insert(0, "wrong_database_cascade")
    if suppress and remember:
        candidates = [candidate for candidate in candidates if candidate not in attempted]
    return tuple(candidates)


def run_trial(task: CodingTask, condition: str, work_root: Path) -> TrialResult:
    """Run one observe-plan-patch-test loop against a newly generated repo."""
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition: {condition}")
    repo = work_root / task.identifier
    _write_repo(repo, task)
    attempted: set[str] = set()
    files_modified = 0
    test_runs = 0
    failed_hypotheses = 0
    repeated_failed_hypotheses = 0
    first_patch_failed = False
    failure_seen = not _run_tests(repo)
    test_runs += 1
    for iteration in range(1, 5):
        choices = _candidates(task, condition, attempted, failure_seen)
        if not choices:
            break
        hypothesis = choices[0]
        if hypothesis in attempted:
            repeated_failed_hypotheses += 1
        attempted.add(hypothesis)
        files_modified += int(_apply_patch(repo, hypothesis))
        passed = _run_tests(repo)
        test_runs += 1
        if not passed:
            failed_hypotheses += 1
            first_patch_failed = first_patch_failed or iteration == 1
            failure_seen = True
            continue
        hidden = _run_tests(repo, hidden=True)
        test_runs += 1
        return TrialResult(
            task.identifier, condition, hidden, hidden, True, iteration, 2,
            files_modified, test_runs, failed_hypotheses, repeated_failed_hypotheses,
            first_patch_failed, first_patch_failed and hidden,
        )
    return TrialResult(
        task.identifier, condition, False, False, False, 4, 2, files_modified,
        test_runs, failed_hypotheses, repeated_failed_hypotheses, first_patch_failed, False,
    )


def summarize(results: Sequence[TrialResult]) -> Mapping[str, float | int]:
    if not results:
        raise ValueError("results must not be empty")
    total = len(results)
    initial_failures = [result for result in results if result.first_patch_failed]
    return {
        "tasks": total,
        "task_success": sum(result.task_success for result in results) / total,
        "hidden_tests_passed": sum(result.hidden_tests_passed for result in results) / total,
        "regression_tests_passed": sum(result.regression_tests_passed for result in results) / total,
        "mean_iterations": sum(result.iterations for result in results) / total,
        "mean_files_read": sum(result.files_read for result in results) / total,
        "mean_files_modified": sum(result.files_modified for result in results) / total,
        "mean_test_runs": sum(result.test_runs for result in results) / total,
        "mean_failed_hypotheses": sum(result.failed_hypotheses for result in results) / total,
        "mean_repeated_failed_hypotheses": sum(result.repeated_failed_hypotheses for result in results) / total,
        "failure_recovery_rate": (
            sum(result.recovered_after_first_failure for result in initial_failures) / len(initial_failures)
            if initial_failures else 0.0
        ),
        "hypothesis_efficiency": sum(result.task_success for result in results) /
        max(1, sum(result.failed_hypotheses + 1 for result in results)),
    }


def run_benchmark(tasks: int = 50, conditions: Sequence[str] = CONDITIONS) -> Mapping[str, Mapping[str, float | int]]:
    """Evaluate B--G. Persist returned records to compare external agents fairly."""
    selected = tuple(conditions)
    if not selected or any(condition not in CONDITIONS for condition in selected):
        raise ValueError("conditions must be a non-empty subset of CONDITIONS")
    workspace = Path(tempfile.mkdtemp(prefix="coding-world-"))
    try:
        return {
            condition: summarize([
                run_trial(task, condition, workspace / condition) for task in generate_tasks(tasks)
            ])
            for condition in selected
        }
    finally:
        shutil.rmtree(workspace)


def save_external_result(path: Path, agent: str, records: Sequence[Mapping[str, object]]) -> None:
    """Save A/Codex or other external-agent trial records without claiming a score."""
    payload = {"agent": agent, "records": list(records)}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    print(json.dumps(run_benchmark(), ensure_ascii=False, indent=2, sort_keys=True))