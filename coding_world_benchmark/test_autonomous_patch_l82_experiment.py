from pathlib import Path
import subprocess

from coding_world_benchmark.autonomous_patch_l82_experiment import (
    format_autonomous_patch,
    run_autonomous_patch_workflow,
)
from coding_world_benchmark.external_repair_l81_experiment import parse_repair_proposal, plan_repair, RegressionVerifier


def _runner(passed: bool):
    def run(command, cwd):
        return subprocess.CompletedProcess(command, 0 if passed else 1, "1 passed" if passed else "failure", "")
    return RegressionVerifier(run)


def _proposal():
    return parse_repair_proposal(
        """Problem: add a small health helper
Target files: target.py
Suggested changes:
- add function health_check
Regression tests:
- test_target.py
"""
    )


def test_autonomous_patch_stages_validates_and_promotes_minimal_diff(tmp_path: Path):
    target = tmp_path / "target.py"
    target.write_text("VALUE = 1\n", encoding="utf-8")
    instruction = _proposal()
    plan = plan_repair(tmp_path, instruction)

    result = run_autonomous_patch_workflow(tmp_path, instruction, plan, verifier=_runner(True))

    assert result.status == "accepted"
    assert result.patch.applied
    assert "def health_check" in target.read_text(encoding="utf-8")
    assert result.static_validation and result.static_validation.passed
    assert result.regression and result.regression.passed
    assert "diff" in format_autonomous_patch(result)


def test_regression_failure_does_not_modify_original(tmp_path: Path):
    target = tmp_path / "target.py"
    original = "VALUE = 1\n"
    target.write_text(original, encoding="utf-8")
    instruction = _proposal()
    plan = plan_repair(tmp_path, instruction)

    result = run_autonomous_patch_workflow(tmp_path, instruction, plan, verifier=_runner(False))

    assert result.status == "regression_failed"
    assert not result.patch.applied
    assert target.read_text(encoding="utf-8") == original


def test_unsupported_semantic_change_is_not_fabricated(tmp_path: Path):
    target = tmp_path / "target.py"
    target.write_text("VALUE = 1\n", encoding="utf-8")
    instruction = parse_repair_proposal(
        """Problem: redesign algorithm
Target files: target.py
Suggested changes:
- route the algorithm through a new semantic planner
Regression tests:
- test_target.py
"""
    )
    plan = plan_repair(tmp_path, instruction)

    result = run_autonomous_patch_workflow(tmp_path, instruction, plan, verifier=_runner(True))

    assert result.status == "synthesis_unavailable"
    assert not result.patch.applied
    assert target.read_text(encoding="utf-8") == "VALUE = 1\n"
