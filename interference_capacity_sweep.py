"""Test whether a finite workspace has an intermediate optimum under interference.

It adds low-utility but query-overlapping routes to the same environment used
by ``workspace_capacity_sweep``.  The model has one normalised attention budget
per query, so more matching concepts dilute retrieval without deleting the
correct concept from long-term memory.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import random
from typing import Dict, Mapping, Sequence

from Consciousness_model import CompressedWorkspace, Relation


@dataclass(frozen=True)
class InterferenceResult:
    capacity: str
    target_present: float
    retrieval_accuracy: float
    matching_candidates: float


def _trial(seed: int, capacity: int | None, noise_per_goal: int) -> tuple[float, float, float]:
    random_source = random.Random(seed)
    target = random_source.randrange(6)
    paths = list(range(6))
    random_source.shuffle(paths)
    observations: list[Relation] = []
    for number in paths:
        observations.extend(((f"start-{number}", "leads-to", f"middle-{number}"), (f"middle-{number}", "leads-to", f"goal-{number}")))
    # These routes share a destination cue but cannot solve the start-target task.
    # They are lower utility individually than a two-step compressed route.
    noise = [(f"noise-{number}-{copy}", "leads-to", f"goal-{number}") for number in range(6) for copy in range(noise_per_goal)]
    random_source.shuffle(noise)
    workspace = CompressedWorkspace(capacity)
    workspace.ingest(observations + noise)
    correct_id = f"compose:start-{target}:leads-to:goal-{target}"
    distribution = workspace.attention_distribution((f"start-{target}", f"goal-{target}"), focus=0.8)
    target_present = float(correct_id in workspace.items)
    retrieval_accuracy = sum(probability for concept, probability in distribution if concept.identifier == correct_id)
    return target_present, retrieval_accuracy, float(len(distribution))


def run(capacities: Sequence[int | None] = (2, 4, 8, 16, 32, 64, 128, None), trials: int = 24, noise_per_goal: int = 16) -> Mapping[str, Mapping[str, float | str]]:
    results: Dict[str, Mapping[str, float | str]] = {}
    for capacity in capacities:
        values = [_trial(seed, capacity, noise_per_goal) for seed in range(trials)]
        means = [sum(row[index] for row in values) / trials for index in range(3)]
        label = "infinity" if capacity is None else str(capacity)
        results[label] = asdict(InterferenceResult(label, *means))
    return results


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2, sort_keys=True))
