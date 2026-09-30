"""L4.8: scientific result diagnosis for coding-agent experiments.

L4.7 shows that a bounded, family-agnostic composer can acquire and reuse
operators.  This follow-on tests a different failure mode: after a coding agent
has produced and run code, can a critic avoid treating every bad result as a
code bug to be patched?

The experiment keeps the world deliberately small and deterministic.  Each
case writes a real Python repository, runs a real ``unittest`` subprocess, then
diagnoses the experimental evidence into one of six outcomes.  The measured
claim is not unrestricted scientific reasoning; it is the narrower control
primitive needed before adding web search or external LLM review to the loop.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import json
import math
import shutil
import subprocess
import sys
import tempfile


UNKNOWN_UNCERTAINTY = 1_000_000.0


class ResultDiagnosis(Enum):
    IMPLEMENTATION_FAILURE = "implementation_failure"
    INSTRUMENT_FAILURE = "instrument_failure"
    LOW_POWER = "low_power"
    CONFOUNDED = "confounded"
    NEGATIVE_RESULT = "negative_result"
    POSITIVE_RESULT = "positive_result"


@dataclass(frozen=True)
class Evidence:
    implementation_ok: bool
    metric_available: bool
    effect: float | None
    control_effect: float | None
    samples: int
    threshold: float
    uncertainty: float


@dataclass(frozen=True)
class DiagnosisCase:
    name: str
    source: str
    test_body: str
    observations: tuple[float, ...]
    control_observations: tuple[float, ...]
    metric_available: bool
    threshold: float
    expected: ResultDiagnosis


def cases() -> tuple[DiagnosisCase, ...]:
    """Six controlled outcomes sharing the same repository/testing interface."""
    passing_source = (
        "def transform(value):\n"
        "    return value + 1\n"
    )
    passing_test = "self.assertEqual(service.transform(1), 2)"
    return (
        DiagnosisCase(
            "implementation_failure",
            "def transform(value):\n    return value + missing_name\n",
            passing_test,
            (0.4, 0.5, 0.6, 0.5, 0.4),
            (0.0, 0.1, 0.0, -0.1, 0.0),
            True,
            0.5,
            ResultDiagnosis.IMPLEMENTATION_FAILURE,
        ),
        DiagnosisCase(
            "instrument_failure",
            passing_source,
            passing_test,
            (),
            (0.0, 0.0, 0.0, 0.0, 0.0),
            False,
            0.5,
            ResultDiagnosis.INSTRUMENT_FAILURE,
        ),
        DiagnosisCase(
            "low_power",
            passing_source,
            passing_test,
            (0.65, 0.35),
            (0.0, 0.1),
            True,
            0.5,
            ResultDiagnosis.LOW_POWER,
        ),
        DiagnosisCase(
            "confounded",
            passing_source,
            passing_test,
            (0.8, 0.7, 0.9, 0.8, 0.7, 0.9),
            (0.7, 0.8, 0.7, 0.9, 0.8, 0.7),
            True,
            0.5,
            ResultDiagnosis.CONFOUNDED,
        ),
        DiagnosisCase(
            "negative_result",
            passing_source,
            passing_test,
            (0.1, 0.2, 0.0, 0.1, -0.1, 0.0),
            (0.0, 0.1, 0.0, -0.1, 0.1, 0.0),
            True,
            0.5,
            ResultDiagnosis.NEGATIVE_RESULT,
        ),
        DiagnosisCase(
            "positive_result",
            passing_source,
            passing_test,
            (0.9, 0.8, 0.7, 0.8, 0.9, 0.8),
            (0.0, 0.1, -0.1, 0.0, 0.1, 0.0),
            True,
            0.5,
            ResultDiagnosis.POSITIVE_RESULT,
        ),
    )


def _write_repo(root: Path, case: DiagnosisCase) -> None:
    root.mkdir(parents=True, exist_ok=True)
    root.joinpath("service.py").write_text(case.source, encoding="utf-8")
    root.joinpath("test_service.py").write_text(
        "import unittest\nimport service\n\nclass Contract(unittest.TestCase):\n"
        "    def test_contract(self):\n"
        f"        {case.test_body}\n",
        encoding="utf-8",
    )


def _unittest(root: Path) -> bool:
    return subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-p", "test_*.py"],
        cwd=root,
        capture_output=True,
        encoding="utf-8", errors="replace",
        check=False,
    ).returncode == 0


def _mean(values: tuple[float, ...]) -> float:
    return sum(values) / len(values) if values else math.nan


def _standard_error(values: tuple[float, ...]) -> float:
    if len(values) < 2:
        return math.inf
    mean = _mean(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance / len(values))


def collect_evidence(case: DiagnosisCase, root: Path) -> Evidence:
    _write_repo(root, case)
    implementation_ok = _unittest(root)
    if not implementation_ok or not case.metric_available:
        return Evidence(implementation_ok, case.metric_available, None, None, len(case.observations),
                        case.threshold, UNKNOWN_UNCERTAINTY)
    effect = _mean(case.observations)
    control = _mean(case.control_observations)
    uncertainty = max(_standard_error(case.observations), _standard_error(case.control_observations))
    return Evidence(implementation_ok, True, effect, control, len(case.observations),
                    case.threshold, uncertainty)


def diagnose(evidence: Evidence, *, minimum_samples: int = 5, confidence_margin: float = 0.15) -> ResultDiagnosis:
    if not evidence.implementation_ok:
        return ResultDiagnosis.IMPLEMENTATION_FAILURE
    if not evidence.metric_available or evidence.effect is None or evidence.control_effect is None:
        return ResultDiagnosis.INSTRUMENT_FAILURE
    if evidence.samples < minimum_samples or evidence.uncertainty > confidence_margin:
        return ResultDiagnosis.LOW_POWER
    if evidence.control_effect >= evidence.threshold:
        return ResultDiagnosis.CONFOUNDED
    if evidence.effect >= evidence.threshold:
        return ResultDiagnosis.POSITIVE_RESULT
    return ResultDiagnosis.NEGATIVE_RESULT


def next_action(diagnosis: ResultDiagnosis) -> str:
    return {
        ResultDiagnosis.IMPLEMENTATION_FAILURE: "repair_code",
        ResultDiagnosis.INSTRUMENT_FAILURE: "repair_measurement",
        ResultDiagnosis.LOW_POWER: "increase_samples",
        ResultDiagnosis.CONFOUNDED: "revise_controls",
        ResultDiagnosis.NEGATIVE_RESULT: "record_falsification",
        ResultDiagnosis.POSITIVE_RESULT: "record_support",
    }[diagnosis]


def naive_success_only_action(evidence: Evidence) -> str:
    """A coding-agent baseline that cannot separate scientific failure causes."""
    if not evidence.implementation_ok:
        return "repair_code"
    if evidence.effect is not None and evidence.effect >= evidence.threshold:
        return "record_support"
    return "repair_code"


def run_scientific_diagnosis_experiment() -> dict[str, object]:
    root = Path(tempfile.mkdtemp(prefix="l48-scientific-diagnosis-"))
    try:
        rows = []
        for case in cases():
            evidence = collect_evidence(case, root / case.name)
            diagnosis = diagnose(evidence)
            rows.append({
                "case": case.name,
                "expected": case.expected.value,
                "diagnosis": diagnosis.value,
                "correct": diagnosis == case.expected,
                "next_action": next_action(diagnosis),
                "naive_action": naive_success_only_action(evidence),
                "implementation_ok": evidence.implementation_ok,
                "metric_available": evidence.metric_available,
                "effect": evidence.effect,
                "control_effect": evidence.control_effect,
                "samples": evidence.samples,
                "uncertainty": evidence.uncertainty,
            })
        total = len(rows)
        naive_correct = sum(row["naive_action"] == row["next_action"] for row in rows)
        return {
            "diagnosis_labels": [entry.value for entry in ResultDiagnosis],
            "cases": rows,
            "diagnostic_accuracy": sum(bool(row["correct"]) for row in rows) / total,
            "naive_action_agreement": naive_correct / total,
            "separates_scientific_failure_from_code_failure": all(bool(row["correct"]) for row in rows)
            and naive_correct < total,
        }
    finally:
        shutil.rmtree(root)


if __name__ == "__main__":
    print(json.dumps(run_scientific_diagnosis_experiment(), ensure_ascii=False, indent=2, sort_keys=True))
