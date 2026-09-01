"""Reject generic or structurally invalid experiment designs before ranking."""

from __future__ import annotations

from .models import ExperimentProposal, ExperimentReview


GENERIC_METRICS = {"task success rate", "effect size", "variance"}


def review_experiment(proposal: ExperimentProposal) -> ExperimentReview:
    problems: list[str] = []
    has_control = proposal.intervention is not None and proposal.control_condition is not None
    if not proposal.procedure:
        problems.append("手順がない。")
    if not proposal.metrics or set(proposal.metrics) == GENERIC_METRICS:
        problems.append("仮説固有の測定指標がない。")
    if not proposal.falsification_condition:
        problems.append("具体的な反証条件がない。")
    if proposal.experiment_type in {"controlled_experiment", "benchmark", "ablation"} and not has_control:
        problems.append("比較形式に必要な介入または対照条件がない。")
    return ExperimentReview(
        testable=bool(proposal.procedure), actually_tests_hypothesis=bool(proposal.target_hypothesis and proposal.expected_result),
        falsifiable=bool(proposal.falsification_condition), measurable=bool(proposal.metrics),
        has_valid_control=has_control if proposal.experiment_type in {"controlled_experiment", "benchmark", "ablation"} else None,
        duplicates_existing_test=False, problems=problems,
    )