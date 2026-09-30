"""Adversarial L4.5 contract hold-out for finding the edit-grammar boundary.

Task generation is isolated here: repair code receives only a generated repo
and its tests.  The four contracts intentionally require operations absent
from L4's mutation grammar, so failure is a measured boundary rather than a
benchmark failure hidden by a new task-specific repair template.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import random
import shutil
import subprocess
import sys
import tempfile

from .operation_discovery_l4_experiment import _candidates


@dataclass(frozen=True)
class HoldoutContract:
    identifier: str
    kind: str
    source: str
    assertion: str


def generate_adversarial_tasks(count: int = 100, seed: int = 41) -> tuple[HoldoutContract, ...]:
    """Generate seeded hold-outs without exposing generator metadata to repair."""
    if count < 1:
        raise ValueError("count must be positive")
    specs = (
        ("rollback", "state = {'target': 'old'}\ndef operate(key):\n    state[key] = 'pending'\n    raise ValueError('failed')\n    return state[key]\n", "self.assertEqual(service.state['target'], 'old')"),
        ("coherence", "persistent = {'target': 'old'}\ncache = dict(persistent)\ndef operate(key):\n    persistent[key] = 'new'\n    return persistent[key]\n", "self.assertEqual(service.cache['target'], 'new')"),
        ("idempotency", "state = ['target']\ndef operate(key):\n    state.append(key)\n    return state\n", "self.assertEqual(len(set(service.state)), len(service.state))"),
        ("cleanup", "resources = {'target': True}\ndef operate(key):\n    return resources[key]\n", "self.assertFalse(service.resources['target'])"),
    )
    rng = random.Random(seed)
    return tuple(HoldoutContract(f"{kind}-{index:03d}", kind, source, assertion)
                 for index in range(count) for kind, source, assertion in (rng.choice(specs),))


def _write_repo(root: Path, task: HoldoutContract) -> None:
    root.mkdir(parents=True, exist_ok=True)
    root.joinpath("noise.py").write_text("NOT_RELATED = True\n", encoding="utf-8")
    root.joinpath("service.py").write_text(task.source, encoding="utf-8")
    call = "with self.assertRaises(ValueError):\n            service.operate('target')" if task.kind == "rollback" else "service.operate('target')"
    root.joinpath("test_contract.py").write_text(
        "import unittest\nimport service\n\nclass Contract(unittest.TestCase):\n"
        "    def test_postcondition(self):\n        " + call + "\n        " + task.assertion + "\n",
        encoding="utf-8",
    )


def _test(root: Path) -> bool:
    return subprocess.run([sys.executable, "-m", "unittest", "discover", "-p", "test_*.py"], cwd=root,
                          capture_output=True, encoding="utf-8", errors="replace", check=False).returncode == 0


def evaluate_l4_grammar(tasks: tuple[HoldoutContract, ...]) -> dict[str, float | int]:
    """Measure L4 candidate coverage and repair rate on unseen contract types."""
    root = Path(tempfile.mkdtemp(prefix="l45-contract-holdout-"))
    try:
        covered = solved = 0
        by_kind: dict[str, list[bool]] = {}
        for task in tasks:
            repo = root / task.identifier
            _write_repo(repo, task)
            path = repo / "service.py"
            candidates = _candidates(path.read_text(encoding="utf-8"), task.assertion)
            covered += bool(candidates)
            for before, after in candidates:
                source = path.read_text(encoding="utf-8")
                path.write_text(source.replace(before, after, 1) + "\n# candidate\n", encoding="utf-8")
                if _test(repo):
                    solved += 1
                    break
            by_kind.setdefault(task.kind, []).append(_test(repo))
        total = len(tasks)
        return {
            "tasks": total,
            "candidate_coverage": covered / total,
            "task_success": solved / total,
            "rollback_success": sum(by_kind.get("rollback", ())) / max(1, len(by_kind.get("rollback", ()))),
            "coherence_success": sum(by_kind.get("coherence", ())) / max(1, len(by_kind.get("coherence", ()))),
            "idempotency_success": sum(by_kind.get("idempotency", ())) / max(1, len(by_kind.get("idempotency", ()))),
            "cleanup_success": sum(by_kind.get("cleanup", ())) / max(1, len(by_kind.get("cleanup", ()))),
        }
    finally:
        shutil.rmtree(root)


if __name__ == "__main__":
    print(json.dumps(evaluate_l4_grammar(generate_adversarial_tasks()), ensure_ascii=False, indent=2, sort_keys=True))