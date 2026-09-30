from pathlib import Path
import subprocess

from coding_world_benchmark.autonomous_repair_l84_experiment import FailureType, run_autonomous_repair_loop
from coding_world_benchmark.external_repair_l81_experiment import RegressionVerifier, RepairInstruction


def runner_sequence():
    calls = {"count": 0}

    def run(command, cwd):
        calls["count"] += 1
        passed = calls["count"] > 1
        return subprocess.CompletedProcess(command, 0 if passed else 1, "1 passed" if passed else "1 failed", "")

    return RegressionVerifier(run)


def runner_always_fail():
    def run(command, cwd):
        return subprocess.CompletedProcess(command, 1, "1 failed", "")
    return RegressionVerifier(run)


class RetryOnce:
    def propose(self, instruction, plan, diagnosis, attempts):
        return RepairInstruction(instruction.problem, instruction.target_files, instruction.target_symbols, ("add function recovered",), instruction.constraints, instruction.acceptance_tests, instruction.source)


class SameRetry:
    def propose(self, instruction, plan, diagnosis, attempts):
        return instruction


def proposal():
    return """Problem: recover helper
Target files: target.py
Suggested changes:
- add function first_attempt
Regression tests:
- test_target.py
"""


def test_repair_loop_retries_with_new_instruction_and_accepts(tmp_path: Path):
    (tmp_path / "target.py").write_text("VALUE = 1\n", encoding="utf-8")
    result = run_autonomous_repair_loop(tmp_path, proposal(), RetryOnce(), runner_sequence())

    assert result.outcome == "accepted"
    assert len(result.attempts) == 2
    assert result.diagnoses[0].failure_type is FailureType.BEHAVIORAL_REGRESSION
    assert "def recovered" in (tmp_path / "target.py").read_text(encoding="utf-8")


def test_same_failure_signature_stops_without_infinite_retry(tmp_path: Path):
    (tmp_path / "target.py").write_text("VALUE = 1\n", encoding="utf-8")
    result = run_autonomous_repair_loop(tmp_path, proposal(), SameRetry(), runner_always_fail(), max_attempts=3)

    assert result.outcome == "repeated_failure"
    assert len(result.attempts) == 2
    assert (tmp_path / "target.py").read_text(encoding="utf-8") == "VALUE = 1\n"
