"""Attentional-blink test for the finite compressed workspace.

T1 and T2 are admitted by the same candidate competition.  Admitting an item
spends generic workspace-selection budget proportional to its information
cost; the budget then recovers over time.  Thus no rule says that T2 should
fail—any lag curve follows from a finite selection resource plus competition.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import exp
import json
from typing import Dict, Mapping, Sequence

from Consciousness_model import Concept


@dataclass(frozen=True)
class BlinkPoint:
    lag: int
    t2_access_after_t1: float
    t2_access_alone: float
    blink_cost: float
    remaining_budget: float


def _concept(identifier: str, prediction: float, cost: float = 1.0) -> Concept:
    return Concept(identifier, "episode", identifier, ((identifier, "predicts", "outcome"),), prediction, prediction, 0, (identifier,), 1, cost)


def _competition_probability(target: Concept, distractors: Sequence[Concept], temperature: float) -> float:
    values = [target.utility()] + [item.utility() for item in distractors]
    shift = max(values)
    weights = [exp((value - shift) / temperature) for value in values]
    return weights[0] / sum(weights)


def _t2_probability(budget: float, temperature: float) -> float:
    target = _concept("T2", prediction=0.8)
    distractors = tuple(_concept(f"distractor-{index}", prediction=0.45) for index in range(3))
    return budget * _competition_probability(target, distractors, temperature)


def run(lags: Sequence[int] = (1, 2, 3, 4, 5), recovery_per_lag: float = 0.22, temperature: float = 0.15) -> Mapping[str, object]:
    if recovery_per_lag <= 0 or temperature <= 0:
        raise ValueError("recovery_per_lag and temperature must be positive")
    # T1 is a multi-observation concept; encoding it consumes a generic portion
    # of the same finite selection resource used by every candidate.
    t1 = _concept("T1", prediction=0.9, cost=1.25)
    t1_cost = min(0.85, 0.25 + 0.32 * t1.cost + 0.16 * t1.prediction)
    baseline = _t2_probability(1.0, temperature)
    points = []
    for lag in lags:
        remaining = min(1.0, 1.0 - t1_cost + lag * recovery_per_lag)
        access = _t2_probability(remaining, temperature)
        points.append(BlinkPoint(lag, access, baseline, baseline - access, remaining))
    return {"t1_selection_cost": t1_cost, "points": [asdict(point) for point in points]}


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
