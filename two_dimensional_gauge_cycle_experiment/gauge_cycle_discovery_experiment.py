"""Discover a local closed-link concept in a 2D compact U(1) gauge proxy.

The grammar receives oriented raw links only. It enumerates short paths and
cycles, without a ``plaquette`` label or formula. Candidate choice uses only
held-out next-step prediction error. The classical reference is a 2D compact
U(1) Hamiltonian lattice, suitable for a representation-discovery test but not
a quantum lattice-gauge simulation.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from math import pi
from pathlib import Path
import sys
from time import perf_counter
from typing import Mapping, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Consciousness_model import CompressedWorkspace, ConsciousnessController


CANDIDATES = (
    "path(+x)", "path(+y)", "path(+x,+y)", "path(+x,-y)",
    "cycle(+x,+y,-x,-y)", "cycle(+x,-y,-x,+y)", "mean(square(electric))",
)
TARGETS = ("cycle(+x,+y,-x,-y)", "mean(square(electric))")
CYCLE = "cycle(+x,+y,-x,-y)"


@dataclass(frozen=True)
class Config:
    train_sizes: tuple[int, ...] = (4, 6)
    test_sizes: tuple[int, ...] = (8, 10, 12, 16)
    train_steps: int = 20
    test_steps: int = 28
    dt: float = 0.035
    latent_dimension: int = 3


@dataclass(frozen=True)
class Report:
    grammar_candidates: tuple[str, ...]
    selected_concepts: tuple[str, ...]
    validation_rmse: tuple[float, ...]
    size_results: tuple[Mapping[str, float | int], ...]
    gauge_cycle_error: float
    ablation_rmse_increase: Mapping[str, float]
    broadcast_concepts: tuple[str, ...]


@dataclass(frozen=True)
class State:
    x_link: np.ndarray
    y_link: np.ndarray
    x_electric: np.ndarray
    y_electric: np.ndarray


def _wrap(values: np.ndarray) -> np.ndarray:
    return (values + pi) % (2.0 * pi) - pi


def make_state(size: int, seed: int) -> State:
    source = np.random.default_rng(seed)
    return State(
        source.uniform(-pi, pi, (size, size)), source.uniform(-pi, pi, (size, size)),
        source.normal(0.0, 0.25, (size, size)), source.normal(0.0, 0.25, (size, size)),
    )


def cycle_angle(state: State) -> np.ndarray:
    return _wrap(state.x_link + np.roll(state.y_link, -1, axis=0) - np.roll(state.x_link, -1, axis=1) - state.y_link)


def step(state: State, coupling: float, magnetic: float, dt: float) -> State:
    """Leapfrog evolution of electric fields and compact link phases."""
    def force(values: State) -> tuple[np.ndarray, np.ndarray]:
        sine = magnetic * np.sin(cycle_angle(values))
        return -sine + np.roll(sine, 1, axis=1), -sine + np.roll(sine, 1, axis=0)

    force_x, force_y = force(state)
    electric_x = state.x_electric + 0.5 * dt * force_x
    electric_y = state.y_electric + 0.5 * dt * force_y
    middle = State(_wrap(state.x_link + dt * coupling * coupling * electric_x), _wrap(state.y_link + dt * coupling * coupling * electric_y), electric_x, electric_y)
    force_x, force_y = force(middle)
    return State(middle.x_link, middle.y_link, electric_x + 0.5 * dt * force_x, electric_y + 0.5 * dt * force_y)


def gauge_transform(state: State, seed: int) -> State:
    alpha = np.random.default_rng(seed).uniform(-pi, pi, state.x_link.shape)
    return State(
        _wrap(state.x_link + alpha - np.roll(alpha, -1, axis=0)),
        _wrap(state.y_link + alpha - np.roll(alpha, -1, axis=1)),
        state.x_electric.copy(), state.y_electric.copy(),
    )


def values(state: State) -> dict[str, float]:
    closed = cycle_angle(state)
    electric = (state.x_electric ** 2 + state.y_electric ** 2) / 2.0
    return {
        "path(+x)": float(np.mean(np.cos(state.x_link))),
        "path(+y)": float(np.mean(np.cos(state.y_link))),
        "path(+x,+y)": float(np.mean(np.cos(state.x_link + np.roll(state.y_link, -1, axis=0)))),
        "path(+x,-y)": float(np.mean(np.cos(state.x_link - state.y_link))),
        CYCLE: float(np.mean(np.cos(closed))),
        "cycle(+x,-y,-x,+y)": float(np.mean(np.sin(closed))),
        "mean(square(electric))": float(np.mean(electric)),
    }


def grammar_relations(size: int) -> tuple[tuple[str, str, str], ...]:
    return tuple((f"local:{candidate}", "composes", "raw_link_path") for _ in range(size * size) for candidate in CANDIDATES)


def generated_grammar() -> tuple[str, ...]:
    workspace = CompressedWorkspace(capacity=None)
    workspace.ingest(grammar_relations(4))
    result = [concept.relations[0][0].removeprefix("*:") for concept in workspace.long_term.values() if concept.kind == "abstract"]
    return tuple(sorted(result))


def trajectory(size: int, coupling: float, magnetic: float, seed: int, steps: int, config: Config) -> tuple[dict[str, float], ...]:
    state = make_state(size, seed)
    result = [values(state)]
    for _ in range(steps):
        state = step(state, coupling, magnetic, config.dt)
        result.append(values(state))
    return tuple(result)


def _pairs(trajectories: Sequence[Sequence[Mapping[str, float]]]) -> list[tuple[Mapping[str, float], Mapping[str, float]]]:
    return [pair for trajectory in trajectories for pair in zip(trajectory, trajectory[1:])]


def _matrix(rows: Sequence[Mapping[str, float]]) -> np.ndarray:
    return np.asarray([[row[name] for name in TARGETS] for row in rows])


class Rule:
    def __init__(self, concepts: Sequence[str], coefficients: np.ndarray) -> None:
        self.concepts, self.coefficients = tuple(concepts), coefficients

    @classmethod
    def fit(cls, rows: Sequence[tuple[Mapping[str, float], Mapping[str, float]]], concepts: Sequence[str]) -> "Rule":
        design = np.asarray([[1.0, *(left[name] for name in concepts)] for left, _ in rows])
        coefficients, *_ = np.linalg.lstsq(design, _matrix([right for _, right in rows]), rcond=None)
        return cls(concepts, coefficients)

    def predict(self, source: Mapping[str, float]) -> dict[str, float]:
        result = dict(source)
        output = np.asarray([1.0, *(source[name] for name in self.concepts)]) @ self.coefficients
        result.update({name: float(value) for name, value in zip(TARGETS, output)})
        return result


def _rmse(rule: Rule, reference: Sequence[Mapping[str, float]], steps: int) -> float:
    predicted = [dict(reference[0])]
    for _ in range(steps):
        predicted.append(rule.predict(predicted[-1]))
    return float(np.sqrt(np.mean((_matrix(predicted) - _matrix(reference)) ** 2)))


def _training(config: Config, steps: int) -> list[tuple[Mapping[str, float], Mapping[str, float]]]:
    conditions = ((0.6, 0.5), (1.0, 1.0), (1.4, 1.5))
    return _pairs([trajectory(size, coupling, magnetic, 1000 * size + 100 * index + seed, steps, config) for size in config.train_sizes for index, (coupling, magnetic) in enumerate(conditions) for seed in range(3)])


def select_concepts(config: Config) -> tuple[tuple[str, ...], tuple[float, ...]]:
    rows = _training(config, config.train_steps)
    split = int(0.7 * len(rows))
    train, validation = rows[:split], rows[split:]
    selected: list[str] = []
    errors: list[float] = []
    for _ in range(config.latent_dimension):
        scored = []
        for candidate in generated_grammar():
            if candidate in selected:
                continue
            rule = Rule.fit(train, (*selected, candidate))
            error = float(np.sqrt(np.mean((_matrix([rule.predict(left) for left, _ in validation]) - _matrix([right for _, right in validation])) ** 2)))
            scored.append((error, candidate))
        error, selected_candidate = min(scored)
        selected.append(selected_candidate)
        errors.append(error)
    return tuple(selected), tuple(errors)


def run(config: Config = Config()) -> Report:
    selected, validation = select_concepts(config)
    rule = Rule.fit(_training(config, config.train_steps), selected)
    controller = ConsciousnessController(capacity=len(selected))
    controller.perceive(tuple(("grammar", "composes", concept) for concept in selected))
    controller.integrate(selected)
    results = []
    for size in config.test_sizes:
        reference = trajectory(size, 1.0, 1.0, 800 + size, config.test_steps, config)
        start = perf_counter()
        error = _rmse(rule, reference, config.test_steps)
        concept_seconds = perf_counter() - start
        raw_state = make_state(size, 800 + size)
        start = perf_counter()
        for _ in range(config.test_steps):
            raw_state = step(raw_state, 1.0, 1.0, config.dt)
        exact_seconds = perf_counter() - start
        results.append({"size": size, "rmse": error, "concept_seconds": concept_seconds, "reference_seconds": exact_seconds})
    reference = trajectory(10, 1.0, 1.0, 810, config.test_steps, config)
    full_error = _rmse(rule, reference, config.test_steps)
    ablation = {concept: _rmse(Rule.fit(_training(config, config.train_steps), tuple(value for value in selected if value != concept)), reference, config.test_steps) - full_error for concept in selected}
    original, transformed = make_state(8, 88), gauge_transform(make_state(8, 88), 89)
    cycle_error = abs(values(original)[CYCLE] - values(transformed)[CYCLE])
    return Report(generated_grammar(), selected, validation, tuple(results), cycle_error, ablation, controller.broadcast("predictor").concepts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="write the JSON report to this path")
    arguments = parser.parse_args()
    encoded = json.dumps(asdict(run()), ensure_ascii=False, indent=2)
    print(encoded)
    if arguments.output:
        arguments.output.write_text(encoded + "\n", encoding="utf-8")