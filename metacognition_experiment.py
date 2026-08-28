"""Metacognition test: calibrated knowledge reports from workspace evidence.

The planner receives no task-type label.  It infers confidence from the
confidence stored on retrieved concepts and from competing transitions in the
currently broadcast world model.  It may answer, defer for more observation,
or report insufficient evidence.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import random
from typing import Dict, Mapping, Sequence, Tuple

from Consciousness_model import CompressedWorkspace, Relation
from world_model_planning_experiment import simulate


@dataclass(frozen=True)
class TrialReport:
    confidence: float
    report: str
    actual_success: float
    correct_metareport: float


def _observations(kind: str) -> Tuple[Relation, ...]:
    base = (("outside", "get-key", "key"), ("key", "unlock", "hall"))
    if kind == "known":
        return base + (("hall", "follow-map", "treasure"),)
    if kind == "ambiguous":
        return base + (("key", "unlock", "cave"), ("hall", "follow-map", "treasure"))
    if kind == "unknown":
        return base + (("hall", "wander", "cave"),)
    raise ValueError("kind must be known, ambiguous, or unknown")


def _confidence(workspace: CompressedWorkspace, actions: Sequence[str], states: Sequence[str]) -> float:
    if not actions:
        return 0.0
    edges = {
        relation
        for identifier in workspace.items
        for relation in workspace.long_term[identifier].relations
    }
    confidence = 1.0
    for source, action, destination in zip(states, actions, states[1:]):
        supporting_confidences = [
            concept.confidence
            for identifier in workspace.items
            for concept in (workspace.long_term[identifier],)
            if (source, action, destination) in concept.relations
        ]
        alternatives = {target for edge_source, edge_action, target in edges if edge_source == source and edge_action == action}
        if not supporting_confidences or not alternatives:
            return 0.0
        confidence *= max(supporting_confidences) / len(alternatives)
    return confidence


def _single_trial(kind: str, seed: int, capacity: int | None = 16) -> TrialReport:
    workspace = CompressedWorkspace(capacity)
    workspace.ingest(_observations(kind))
    actions, states = simulate(workspace, "outside", "treasure")
    confidence = _confidence(workspace, actions, states)
    # Answer only when the accumulated path evidence clears the declared
    # decision-risk threshold; otherwise request more evidence.
    if confidence >= 0.70:
        report = "known"
    elif confidence > 0.0:
        report = "need_more_evidence"
    else:
        report = "insufficient_evidence"
    # Ground truth is external to the model.  In an ambiguous transition the
    # observed model licenses two equally plausible outcomes.
    actual_success = 1.0 if kind == "known" else 0.0 if kind == "unknown" else float(random.Random(seed).randrange(2) == 0)
    expected = {"known": "known", "ambiguous": "need_more_evidence", "unknown": "insufficient_evidence"}[kind]
    return TrialReport(confidence, report, actual_success, float(report == expected))


def run(trials_per_condition: int = 100, capacity: int | None = 16) -> Mapping[str, object]:
    reports = {
        kind: [_single_trial(kind, seed, capacity) for seed in range(trials_per_condition)]
        for kind in ("known", "ambiguous", "unknown")
    }
    all_reports = [report for rows in reports.values() for report in rows]
    brier = sum((report.confidence - report.actual_success) ** 2 for report in all_reports) / len(all_reports)
    by_kind = {
        kind: {
            "mean_confidence": sum(report.confidence for report in rows) / len(rows),
            "success_rate": sum(report.actual_success for report in rows) / len(rows),
            "metareport_accuracy": sum(report.correct_metareport for report in rows) / len(rows),
            "report": rows[0].report,
        }
        for kind, rows in reports.items()
    }
    return {"brier_score": brier, "metareport_accuracy": sum(report.correct_metareport for report in all_reports) / len(all_reports), "conditions": by_kind}


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2, sort_keys=True))
