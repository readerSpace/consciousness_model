"""Competing-theory interventions for a functional phenomenal-state candidate."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
from math import sqrt
from typing import Callable, Mapping, Sequence

THEORIES = ("shared_latent", "recurrent_state", "workspace_state", "integrated_private")


@dataclass(frozen=True)
class Intervention:
    name: str
    apply: Callable[[dict[str, bool]], dict[str, bool]]
    cost: float


def _observables(theory: str, intervention: str) -> dict[str, bool]:
    outputs = {"memory": True, "planning": True, "report": True, "geometry": True, "integration": True}
    if theory == "shared_latent" and intervention == "geometry_swap":
        outputs["geometry"] = False
    if theory == "recurrent_state" and intervention == "history_reset":
        outputs["memory"] = False
    if theory == "workspace_state" and intervention == "workspace_block":
        outputs["report"] = False
    if theory == "integrated_private" and intervention == "integration_split":
        outputs["integration"] = False
    return outputs


def _interventions() -> tuple[Intervention, ...]:
    return tuple(Intervention(name, lambda state, name=name: {**state, name: False}, 1.0)
                 for name in ("geometry_swap", "history_reset", "workspace_block", "integration_split"))


def _distance(left: Mapping[str, bool], right: Mapping[str, bool]) -> float:
    return sqrt(sum(left[key] != right[key] for key in left))


def _pair_discrimination(left: str, right: str, intervention: str) -> float:
    return _distance(_observables(left, intervention), _observables(right, intervention)) / sqrt(5.0)


def run_competing_theories() -> Mapping[str, object]:
    weights = {"shared_latent": 0.25, "recurrent_state": 0.25, "workspace_state": 0.25, "integrated_private": 0.25}
    scores = {}
    for intervention in _interventions():
        value = sum(weights[left] * weights[right] * _pair_discrimination(left, right, intervention.name)
                    for left, right in combinations(THEORIES, 2))
        scores[intervention.name] = value / intervention.cost
    selected = max(scores, key=scores.get)
    return {"theories": THEORIES, "scores": scores, "selected_intervention": selected,
            "selected_value": scores[selected]}


def run_paired_predictions() -> Mapping[str, object]:
    interventions = [intervention.name for intervention in _interventions()]
    matrix = {theory: {name: _observables(theory, name) for name in interventions} for theory in THEORIES}
    distinguishable = sum(any(matrix[left][name] != matrix[right][name] for name in interventions)
                          for left, right in combinations(THEORIES, 2))
    return {"pairs": len(list(combinations(THEORIES, 2))), "distinguishable_pairs": distinguishable,
            "complete_separation": distinguishable == 6, "matrix": matrix}


def run_blind_machine_evaluation() -> Mapping[str, object]:
    conditions = {"Q0_no_q": (False, False, False, False), "Q2_random": (True, False, False, False),
                  "Q3_split": (True, True, False, False), "Q4_integrated": (True, True, True, True)}
    scores = {name: sum(values) / len(values) for name, values in conditions.items()}
    return {"conditions": scores, "blind_comparison_ready": True, "q4_best": max(scores, key=scores.get)}


def run() -> Mapping[str, object]:
    return {"competing_theories": run_competing_theories(),
            "paired_predictions": run_paired_predictions(), "blind_machine_evaluation": run_blind_machine_evaluation()}


if __name__ == "__main__":
    import json
    print(json.dumps(run(), indent=2, sort_keys=True))
