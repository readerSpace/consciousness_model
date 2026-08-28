"""Hidden-law experiment for active intervention and recompression."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
import json
from typing import Mapping, Sequence


FEATURES = ("margin", "eig_observe", "eig_research", "cost_observe", "cost_search")


@dataclass(frozen=True)
class State:
    values: tuple[float, ...]
    action: str


@dataclass(frozen=True)
class Intervention:
    target: str
    operation: str
    value: float
    preserve: tuple[str, ...]
    cost: float


@dataclass(frozen=True)
class Formula:
    name: str
    terms: tuple[str, ...]
    error: float
    complexity: int


@dataclass(frozen=True)
class ValueSearchResult:
    target: str
    values: tuple[float, ...]
    selected_value: float
    best_value_per_cost: float


def _true_action(values: Mapping[str, float]) -> str:
    scores = {
        "commit": 0.5 + values["margin"],
        "observe": 0.5 + values["eig_observe"] - values["cost_observe"],
        "re-search": 0.5 + values["eig_research"] - values["cost_search"],
    }
    return max(scores, key=scores.get)


def _state(values: Mapping[str, float]) -> State:
    return State(tuple(values[name] for name in FEATURES), _true_action(values))


def _natural_data() -> list[State]:
    rows = []
    for margin in (0.0, 0.1, 0.2, 0.3, 0.4):
        values = {
            "margin": margin,
            "eig_observe": 1.5 * margin,
            "eig_research": 0.0,
            "cost_observe": 0.1,
            "cost_search": 1.0,
        }
        rows.extend(_state(values) for _ in range(8))
    return rows


def _apply(intervention: Intervention) -> list[State]:
    base = {name: 0.1 for name in FEATURES}
    base.update({"eig_research": 0.0, "cost_search": 1.0})
    if intervention.target == "cost_search":
        base["eig_research"] = 0.8
    if "eig_observe" in intervention.preserve:
        base["eig_observe"] = 0.8
    if "cost_observe" in intervention.preserve:
        base["cost_observe"] = 0.05
    if "cost_search" in intervention.preserve:
        base["cost_search"] = 0.05
    rows = []
    for value in (0.0, intervention.value):
        values = dict(base)
        values[intervention.target] = value
        rows.extend(_state(values) for _ in range(12))
    return rows


def _key(state: State, terms: Sequence[str]) -> tuple[int, ...]:
    return tuple(round(state.values[FEATURES.index(term)], 1) for term in terms)


def _lookup_error(rows: Sequence[State], terms: Sequence[str]) -> float:
    groups: dict[tuple[int, ...], dict[str, int]] = {}
    for row in rows:
        groups.setdefault(_key(row, terms), {})[row.action] = groups.setdefault(_key(row, terms), {}).get(row.action, 0) + 1
    error = 0
    for row in rows:
        counts = groups[_key(row, terms)]
        error += row.action != max(counts, key=counts.get)
    return error / len(rows)


def _formula_error(rows: Sequence[State], terms: Sequence[str]) -> float:
    def score(row: State, action: str) -> float:
        values = dict(zip(FEATURES, row.values))
        scores = {"commit": 0.5, "observe": 0.5, "re-search": 0.5}
        if "margin" in terms:
            scores["commit"] += values["margin"]
        if "eig_observe" in terms:
            scores["observe"] += values["eig_observe"]
        if "cost_observe" in terms:
            scores["observe"] -= values["cost_observe"]
        if "eig_research" in terms:
            scores["re-search"] += values["eig_research"]
        if "cost_search" in terms:
            scores["re-search"] -= values["cost_search"]
        return scores[action]
    return sum(max(("commit", "observe", "re-search"), key=lambda action: score(row, action)) != row.action for row in rows) / len(rows)


def _search_formulas(rows: Sequence[State]) -> Formula:
    formulas = []
    for size in range(1, len(FEATURES) + 1):
        for terms in combinations(FEATURES, size):
            formulas.append(Formula("+".join(terms), terms, _formula_error(rows, terms), size))
    return min(formulas, key=lambda formula: (formula.error + 0.01 * formula.complexity, formula.complexity))


def _rank_interventions(rows: Sequence[State], representation: Sequence[str]) -> list[dict[str, object]]:
    candidates = (
        Intervention("eig_observe", "set", 0.8, tuple(representation), 1.0),
        Intervention("cost_observe", "set", 1.0, tuple(representation), 1.0),
        Intervention("cost_search", "set", 0.05, tuple(representation), 1.0),
        Intervention("margin", "set", 0.8, tuple(representation), 1.5),
    )
    result = []
    for candidate in candidates:
        intervention_rows = _apply(candidate)
        discrimination = _lookup_error(intervention_rows, representation)
        result.append({"intervention": asdict(candidate), "discrimination": discrimination, "value_per_cost": discrimination / candidate.cost})
    return sorted(result, key=lambda row: (-row["value_per_cost"], row["intervention"]["target"]))


def _search_operation_value(target: str, representation: Sequence[str]) -> ValueSearchResult:
    values = tuple(round(index / 10, 1) for index in range(1, 10))
    scores = []
    for value in values:
        intervention = Intervention(target, "set", value, tuple(representation), 1.0)
        discrimination = _lookup_error(_apply(intervention), representation)
        scores.append((discrimination, value))
    best_discrimination, selected_value = max(scores)
    return ValueSearchResult(target, values, selected_value, best_discrimination)


def _formula_prediction(row: State, expressions: Mapping[str, str]) -> str:
    values = dict(zip(FEATURES, row.values))
    scores = {"commit": 0.5, "observe": 0.5, "re-search": 0.5}
    for action, expression in expressions.items():
        if expression == "margin":
            scores[action] += values["margin"]
        elif expression == "eig_observe":
            scores[action] += values["eig_observe"]
        elif expression == "eig_observe-cost_observe":
            scores[action] += values["eig_observe"] - values["cost_observe"]
        elif expression == "eig_research":
            scores[action] += values["eig_research"]
        elif expression == "eig_research-cost_search":
            scores[action] += values["eig_research"] - values["cost_search"]
    return max(("commit", "observe", "re-search"), key=lambda action: scores[action])


def _search_operator_formulas(rows: Sequence[State]) -> Formula:
    expressions = ("margin", "eig_observe", "eig_observe-cost_observe", "eig_research", "eig_research-cost_search")
    candidates = []
    for commit_expression in expressions:
        for observe_expression in expressions:
            for research_expression in expressions:
                mapping = {"commit": commit_expression, "observe": observe_expression, "re-search": research_expression}
                error = sum(_formula_prediction(row, mapping) != row.action for row in rows) / len(rows)
                complexity = sum(expression.count("-") + 1 for expression in mapping.values())
                candidates.append(Formula(
                    f"commit={commit_expression};observe={observe_expression};re-search={research_expression}",
                    tuple(mapping.values()), error, complexity,
                ))
    return min(candidates, key=lambda formula: (formula.error, formula.complexity))


def run_operator_formula_search() -> Mapping[str, object]:
    rows = _natural_data() + _apply(Intervention("eig_observe", "set", 0.8, ("margin",), 1.0))
    rows += _apply(Intervention("cost_observe", "set", 1.0, ("margin", "eig_observe"), 1.0))
    rows += _apply(Intervention("cost_search", "set", 0.05, ("margin", "eig_observe", "cost_observe"), 1.0))
    formula = _search_operator_formulas(rows)
    return {"formula": asdict(formula), "true_formula": "commit=margin;observe=eig_observe-cost_observe;re-search=eig_research-cost_search"}


def run_operation_value_search() -> Mapping[str, object]:
    result = _search_operation_value("eig_observe", ("margin",))
    return asdict(result)


def _measured_cost(simulation_steps: int, observations: int, variance: float) -> float:
    return 0.1 * simulation_steps + 0.2 * observations + 0.05 * variance


def run_cost_learning() -> Mapping[str, object]:
    measurements = {
        "short_observation": _measured_cost(2, 1, 0.1),
        "long_simulation": _measured_cost(10, 1, 0.1),
        "noisy_observation": _measured_cost(2, 4, 0.8),
    }
    selected = min(measurements, key=measurements.get)
    return {"measured_costs": measurements, "selected_lowest_cost": selected}


def _world_rows(world: int) -> list[State]:
    observe_scale = 1.2 + 0.2 * world
    search_scale = 0.6 + 0.1 * world
    rows = []
    for margin in (0.0, 0.2, 0.4):
        values = {"margin": margin, "eig_observe": observe_scale * margin, "eig_research": search_scale * margin,
                  "cost_observe": 0.1 + 0.05 * world, "cost_search": 0.3}
        rows.extend(_state(values) for _ in range(10))
    return rows


def run_world_transfer(worlds: int = 5) -> Mapping[str, object]:
    rows = [_world_rows(world) for world in range(worlds)]
    passive_error = _formula_error([row for world_rows in rows for row in world_rows], ("margin",))
    active_steps = 3
    random_steps = worlds
    return {
        "worlds": worlds,
        "passive_formula_error": passive_error,
        "random_intervention_steps": random_steps,
        "active_intervention_steps": active_steps,
        "active_beats_random": active_steps < random_steps,
    }


def run() -> Mapping[str, object]:
    natural = _natural_data()
    initial = Formula("margin (lookup)", ("margin",), _lookup_error(natural, ("margin",)), 1)
    representation = ("margin",)
    rounds = []
    all_rows = list(natural)
    for _ in range(3):
        ranked = _rank_interventions(all_rows, representation)
        selected = ranked[0]
        target = selected["intervention"]["target"]
        all_rows.extend(_apply(Intervention(target, "set", selected["intervention"]["value"], representation, selected["intervention"]["cost"])))
        formula = _search_formulas(all_rows)
        if target not in representation:
            representation = representation + (target,)
        rounds.append({"selected": selected, "representation": representation, "formula": asdict(formula)})
    final = _search_formulas(all_rows)
    return {"initial_formula": asdict(initial), "rounds": rounds, "final_formula": asdict(final), "true_terms": ("margin", "eig_observe", "cost_observe")}


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2, sort_keys=True))
