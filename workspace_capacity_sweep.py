"""Capacity sweep for the compressed workspace.

Run with ``python workspace_capacity_sweep.py``.  The held-out tasks are new
paths; no C3 task outcome is encoded as a success rule.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import random
from typing import Dict, Iterable, Mapping, Sequence, Tuple

from Consciousness_model import CompressedWorkspace, Relation


@dataclass(frozen=True)
class CapacityResult:
    capacity: str
    planning: float
    prediction: float
    global_reuse: float
    abstraction: float
    mean: float


def _trial(seed: int, capacity: int | None) -> Tuple[float, float, float, float]:
    source = random.Random(seed)
    target = source.randrange(6)
    paths = list(range(6))
    source.shuffle(paths)
    observations: list[Relation] = []
    for number in paths:
        observations.extend(((f"start-{number}", "leads-to", f"middle-{number}"), (f"middle-{number}", "leads-to", f"goal-{number}")))
    # Repeated exemplars support an abstraction that will be queried on a new colour.
    observations.extend((("red:apple", "is", "fruit"), ("green:apple", "is", "fruit")))
    workspace = CompressedWorkspace(capacity)
    workspace.ingest(observations)

    plan = workspace.query("planner", (f"start-{target}", f"goal-{target}"))
    predicted = workspace.query("predictor", (f"start-{target}", f"goal-{target}"))
    remembered = workspace.query("memory", (f"start-{target}", f"goal-{target}"))
    abstraction = workspace.query("reasoner", ("*:apple", "fruit"))
    planning_ok = float(any((f"start-{target}", "leads-to", f"goal-{target}") in item.relations for item in plan))
    prediction_ok = float(any((f"start-{target}", "leads-to", f"goal-{target}") in item.relations for item in predicted))
    global_reuse = float(
        planning_ok
        and bool(predicted)
        and bool(remembered)
        and plan[0].identifier == predicted[0].identifier == remembered[0].identifier
    )
    abstraction_ok = float(any(item.kind == "abstract" for item in abstraction))
    return planning_ok, prediction_ok, global_reuse, abstraction_ok


def run(capacities: Sequence[int | None] = (2, 4, 8, 16, 32, 64, None), trials: int = 96) -> Mapping[str, Mapping[str, float | str]]:
    results: Dict[str, Mapping[str, float | str]] = {}
    for capacity in capacities:
        values = [_trial(seed, capacity) for seed in range(trials)]
        means = [sum(row[index] for row in values) / trials for index in range(4)]
        label = "infinity" if capacity is None else str(capacity)
        result = CapacityResult(label, *means, sum(means) / 4.0)
        results[label] = asdict(result)
    return results


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2, sort_keys=True))
