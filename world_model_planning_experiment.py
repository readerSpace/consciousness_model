"""Test multi-step internal simulation over compressed workspace concepts.

The environment supplies only local state --action--> state transitions.  The
planner reads the capacity-limited workspace, expands future states internally,
and chooses the first action of a trajectory reaching the goal.  There is no
stored start-to-goal answer.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import random
from typing import Dict, Iterable, Mapping, Sequence, Tuple

from Consciousness_model import CompressedWorkspace, Relation


@dataclass(frozen=True)
class PlanningResult:
    capacity: str
    success_rate: float
    mean_simulated_depth: float


def _world(seed: int) -> Tuple[Tuple[Relation, ...], str, str, Tuple[str, ...]]:
    """A shuffled observation stream; only local transitions are supplied."""
    goal_path = (
        ("outside", "get-key", "key"),
        ("key", "unlock-door", "hall"),
        ("hall", "take-map", "map-room"),
        ("map-room", "follow-map", "treasure"),
    )
    distractors = (
        ("outside", "enter-cave", "cave"),
        ("cave", "wander", "cave"),
        ("key", "drop-key", "outside"),
        ("hall", "return", "outside"),
        ("map-room", "ignore-map", "cave"),
        ("garden", "pick-flower", "garden"),
    )
    observations = list(goal_path + distractors)
    random.Random(seed).shuffle(observations)
    return tuple(observations), "outside", "treasure", tuple(edge[1] for edge in goal_path)


def simulate(workspace: CompressedWorkspace, start: str, goal: str, horizon: int = 6) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """Breadth-first rollout using only currently broadcast workspace items."""
    edges = {
        relation
        for identifier in workspace.items
        for relation in workspace.long_term[identifier].relations
    }
    frontier = [(start, (), (start,))]
    visited = {start}
    while frontier:
        state, actions, states = frontier.pop(0)
        if state == goal:
            return actions, states
        if len(actions) == horizon:
            continue
        for source, action, destination in sorted(edges):
            if source == state and destination not in visited:
                visited.add(destination)
                frontier.append((destination, actions + (action,), states + (destination,)))
    return (), (start,)


def _trial(seed: int, capacity: int | None) -> Tuple[float, float]:
    observations, start, goal, expected_actions = _world(seed)
    workspace = CompressedWorkspace(capacity)
    workspace.ingest(observations)
    actions, states = simulate(workspace, start, goal)
    success = float(actions == expected_actions and states[-1] == goal)
    return success, float(len(actions))


def run(capacities: Sequence[int | None] = (2, 4, 8, 16, None), trials: int = 64) -> Mapping[str, Mapping[str, float | str]]:
    results: Dict[str, Mapping[str, float | str]] = {}
    for capacity in capacities:
        values = [_trial(seed, capacity) for seed in range(trials)]
        label = "infinity" if capacity is None else str(capacity)
        results[label] = asdict(PlanningResult(label, sum(row[0] for row in values) / trials, sum(row[1] for row in values) / trials))
    return results


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2, sort_keys=True))
