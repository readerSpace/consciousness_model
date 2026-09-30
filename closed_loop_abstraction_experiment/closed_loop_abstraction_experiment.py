"""Derive a closed-loop abstraction from composed raw 2D U(1) link paths.

No cycle or plaquette candidate is supplied. Four direction primitives are
composed into paths. Only after enumeration does the experiment inspect each
path's endpoint displacement; repeated non-backtracking paths with displacement
zero are compressed into a ``closed_loop`` category. Gauge interventions test
whether that derived category predicts invariance.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import itertools
import json
from pathlib import Path
import sys
from time import perf_counter
from typing import Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "two_dimensional_gauge_cycle_experiment"))

from Consciousness_model import CompressedWorkspace, ConsciousnessController
from gauge_cycle_discovery_experiment import State, gauge_transform, make_state, step


DIRECTIONS = ("+x", "-x", "+y", "-y")
VECTORS = {"+x": (1, 0), "-x": (-1, 0), "+y": (0, 1), "-y": (0, -1)}
OPPOSITE = {"+x": "-x", "-x": "+x", "+y": "-y", "-y": "+y"}
KNOWN_CLOSED_PATH = ("+x", "+y", "-x", "-y")


@dataclass(frozen=True)
class Config:
    train_sizes: tuple[int, ...] = (4, 6)
    test_sizes: tuple[int, ...] = (8, 10, 12, 16)
    train_steps: int = 20
    test_steps: int = 28
    dt: float = 0.035
    selected_count: int = 3


@dataclass(frozen=True)
class Report:
    primitive_directions: tuple[str, ...]
    composed_path_count: int
    zero_displacement_paths: tuple[str, ...]
    compressed_concepts: tuple[str, ...]
    selected_paths: tuple[str, ...]
    selected_closed_count: int
    gauge_invariance_precision: float
    gauge_invariance_recall: float
    size_results: tuple[Mapping[str, float | int], ...]
    closed_path_ablation_rmse_increase: float
    broadcast_concepts: tuple[str, ...]


def displacement(path: Sequence[str]) -> tuple[int, int]:
    return tuple(sum(VECTORS[direction][axis] for direction in path) for axis in range(2))  # type: ignore[return-value]


def composed_paths(length: int = 4) -> tuple[tuple[str, ...], ...]:
    """Generate paths with composition alone, excluding immediate cancellation."""
    return tuple(path for path in itertools.product(DIRECTIONS, repeat=length) if all(right != OPPOSITE[left] for left, right in zip(path, path[1:])))


def path_name(path: Sequence[str]) -> str:
    return "compose(" + ",".join(path) + ")"


def closed_paths() -> tuple[tuple[str, ...], ...]:
    return tuple(path for path in composed_paths() if displacement(path) == (0, 0))


def raw_relations() -> tuple[tuple[str, str, str], ...]:
    return tuple((f"path:{path_name(path)}", "composes", "direction") for _ in range(8) for path in composed_paths())


def compressed_concepts() -> tuple[str, ...]:
    """Compress repeated zero-displacement constructions after path generation."""
    workspace = CompressedWorkspace(capacity=None)
    workspace.ingest(raw_relations())
    abstracted = [concept for concept in workspace.long_term.values() if concept.kind == "abstract"]
    zero_names = {path_name(path) for path in closed_paths()}
    discovered = {concept.relations[0][0].removeprefix("*:") for concept in abstracted}
    if not zero_names <= discovered:
        raise RuntimeError("path grammar failed to preserve repeated constructions")
    return ("closed_loop",) if len(zero_names) > 1 else ()


def _path_angle(state: State, path: Sequence[str]) -> np.ndarray:
    size = state.x_link.shape[0]
    angle = np.zeros((size, size))
    row, column = np.indices((size, size))
    for direction in path:
        if direction == "+x":
            angle += state.x_link[row, column]
            row = (row + 1) % size
        elif direction == "-x":
            row = (row - 1) % size
            angle -= state.x_link[row, column]
        elif direction == "+y":
            angle += state.y_link[row, column]
            column = (column + 1) % size
        else:
            column = (column - 1) % size
            angle -= state.y_link[row, column]
    return angle


def path_values(state: State) -> dict[str, float]:
    values = {path_name(path): float(np.mean(np.cos(_path_angle(state, path)))) for path in composed_paths()}
    values["electric_energy"] = float(np.mean((state.x_electric ** 2 + state.y_electric ** 2) / 2.0))
    return values


def _target(values: Mapping[str, float]) -> np.ndarray:
    return np.asarray([values[path_name(KNOWN_CLOSED_PATH)], values["electric_energy"]])


def trajectory(size: int, seed: int, steps: int, config: Config) -> tuple[dict[str, float], ...]:
    state = make_state(size, seed)
    result = [path_values(state)]
    for _ in range(steps):
        state = step(state, 1.0, 1.0, config.dt)
        result.append(path_values(state))
    return tuple(result)


def _pairs(trajectories: Sequence[Sequence[Mapping[str, float]]]) -> list[tuple[Mapping[str, float], Mapping[str, float]]]:
    return [pair for trajectory in trajectories for pair in zip(trajectory, trajectory[1:])]


class Rule:
    def __init__(self, paths: Sequence[str], coefficients: np.ndarray) -> None:
        self.paths, self.coefficients = tuple(paths), coefficients

    @classmethod
    def fit(cls, rows: Sequence[tuple[Mapping[str, float], Mapping[str, float]]], paths: Sequence[str]) -> "Rule":
        design = np.asarray([[1.0, *(before[path] for path in paths)] for before, _ in rows])
        coefficients, *_ = np.linalg.lstsq(design, np.asarray([_target(after) for _, after in rows]), rcond=None)
        return cls(paths, coefficients)

    def predict(self, values: Mapping[str, float]) -> dict[str, float]:
        prediction = np.asarray([1.0, *(values[path] for path in self.paths)]) @ self.coefficients
        result = dict(values)
        result[path_name(KNOWN_CLOSED_PATH)] = float(prediction[0])
        result["electric_energy"] = float(prediction[1])
        return result


def _rollout_error(rule: Rule, reference: Sequence[Mapping[str, float]], steps: int) -> float:
    predicted = [dict(reference[0])]
    for _ in range(steps):
        predicted.append(rule.predict(predicted[-1]))
    return float(np.sqrt(np.mean((np.asarray([_target(row) for row in predicted]) - np.asarray([_target(row) for row in reference])) ** 2)))


def _training(config: Config) -> list[tuple[Mapping[str, float], Mapping[str, float]]]:
    return _pairs([trajectory(size, 100 * size + seed, config.train_steps, config) for size in config.train_sizes for seed in range(6)])


def select_paths(config: Config) -> tuple[str, ...]:
    rows = _training(config)
    boundary = int(0.7 * len(rows))
    train, validation = rows[:boundary], rows[boundary:]
    candidates = tuple(path_name(path) for path in composed_paths()) + ("electric_energy",)
    selected: list[str] = []
    for _ in range(config.selected_count):
        scores = []
        for candidate in candidates:
            if candidate in selected:
                continue
            rule = Rule.fit(train, (*selected, candidate))
            error = float(np.sqrt(np.mean((np.asarray([_target(rule.predict(before)) for before, _ in validation]) - np.asarray([_target(after) for _, after in validation])) ** 2)))
            scores.append((error, candidate))
        _, winner = min(scores)
        selected.append(winner)
    return tuple(selected)


def invariance_metrics() -> tuple[float, float]:
    state, transformed = make_state(8, 44), gauge_transform(make_state(8, 44), 45)
    original, changed = path_values(state), path_values(transformed)
    candidates = tuple(path_name(path) for path in composed_paths())
    actual_closed = {path_name(path) for path in closed_paths()}
    invariant = {name for name in candidates if abs(original[name] - changed[name]) < 1e-10}
    return len(invariant & actual_closed) / len(invariant), len(invariant & actual_closed) / len(actual_closed)


def run(config: Config = Config()) -> Report:
    selected = select_paths(config)
    rule = Rule.fit(_training(config), selected)
    controller = ConsciousnessController(capacity=len(selected))
    controller.perceive(tuple(("composed", "is", path) for path in selected))
    controller.integrate(selected)
    results = []
    for size in config.test_sizes:
        reference = trajectory(size, 900 + size, config.test_steps, config)
        start = perf_counter()
        error = _rollout_error(rule, reference, config.test_steps)
        elapsed = perf_counter() - start
        results.append({"size": size, "rmse": error, "concept_seconds": elapsed})
    reference = trajectory(10, 910, config.test_steps, config)
    full_error = _rollout_error(rule, reference, config.test_steps)
    closed_selected = tuple(path for path in selected if path in {path_name(value) for value in closed_paths()})
    without_closed = tuple(path for path in selected if path not in closed_selected)
    ablation = _rollout_error(Rule.fit(_training(config), without_closed), reference, config.test_steps) - full_error if closed_selected else 0.0
    precision, recall = invariance_metrics()
    return Report(DIRECTIONS, len(composed_paths()), tuple(path_name(path) for path in closed_paths()), compressed_concepts(), selected, len(closed_selected), precision, recall, tuple(results), ablation, controller.broadcast("predictor").concepts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="write the JSON report to this path")
    arguments = parser.parse_args()
    encoded = json.dumps(asdict(run()), ensure_ascii=False, indent=2)
    print(encoded)
    if arguments.output:
        arguments.output.write_text(encoded + "\n", encoding="utf-8")