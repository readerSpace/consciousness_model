"""Stress-test global availability under capacity limits and retrieval noise."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import random
from typing import Dict, Mapping, Sequence

from Consciousness_model import CompressedWorkspace, Relation


@dataclass(frozen=True)
class RobustnessResult:
    capacity: str
    target_present: float
    planner_access: float
    all_module_access: float
    matching_candidates: float


def _probability(workspace: CompressedWorkspace, terms: Sequence[str], expected: str) -> float:
    return sum(probability for concept, probability in workspace.attention_distribution(terms, focus=0.8) if concept.identifier == expected)


def _trial(seed: int, capacity: int | None, noise_per_goal: int) -> tuple[float, float, float, float]:
    rng = random.Random(seed)
    target = rng.randrange(6)
    observations: list[Relation] = []
    paths = list(range(6))
    rng.shuffle(paths)
    for number in paths:
        observations.extend(((f"start-{number}", "leads-to", f"middle-{number}"), (f"middle-{number}", "leads-to", f"goal-{number}")))
    noise = [(f"noise-{number}-{copy}", "leads-to", f"goal-{number}") for number in range(6) for copy in range(noise_per_goal)]
    rng.shuffle(noise)
    workspace = CompressedWorkspace(capacity)
    workspace.ingest(observations + noise)
    expected = f"compose:start-{target}:leads-to:goal-{target}"
    module_queries = {
        "planner": (f"start-{target}", f"goal-{target}"),
        "predictor": (f"start-{target}", f"goal-{target}", "leads-to"),
        "memory": (f"start-{target}", f"goal-{target}"),
        "report": (f"start-{target}", f"goal-{target}"),
    }
    access = {module: _probability(workspace, terms, expected) for module, terms in module_queries.items()}
    candidate_count = len(workspace.attention_distribution(module_queries["planner"], focus=0.8))
    return float(expected in workspace.items), access["planner"], access["planner"] * access["predictor"] * access["memory"] * access["report"], float(candidate_count)


def run(capacities: Sequence[int | None] = (2, 4, 8, 16, 32, 64, 128, None), trials: int = 24, noise_per_goal: int = 16) -> Mapping[str, Mapping[str, float | str]]:
    results: Dict[str, Mapping[str, float | str]] = {}
    for capacity in capacities:
        values = [_trial(seed, capacity, noise_per_goal) for seed in range(trials)]
        label = "infinity" if capacity is None else str(capacity)
        results[label] = asdict(RobustnessResult(label, *(sum(row[index] for row in values) / trials for index in range(4))))
    return results


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2, sort_keys=True))
