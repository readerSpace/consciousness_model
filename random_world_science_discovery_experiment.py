"""Distribution-shift benchmark for active falsification and recompression."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import random
from math import log
from typing import Callable, Mapping, Sequence

ACTIONS = ("commit", "observe", "re-search")
VARIABLES = ("margin", "eig_observe", "eig_research", "cost_observe", "cost_search")


@dataclass(frozen=True)
class Expression:
    name: str
    evaluate: Callable[[Mapping[str, float]], float]
    complexity: int


@dataclass(frozen=True)
class World:
    expressions: tuple[Expression, Expression, Expression]
    noise: float
    intervention_costs: Mapping[str, float]


@dataclass(frozen=True)
class DecisionExperiment:
    name: str
    eig: float
    falsification: float
    cost: float


@dataclass(frozen=True)
class Theory:
    name: str
    predict: Callable[[str], int]


def _expressions() -> tuple[Expression, ...]:
    return (
        Expression("margin", lambda x: x["margin"], 1),
        Expression("eig_observe", lambda x: x["eig_observe"], 1),
        Expression("eig_observe-cost_observe", lambda x: x["eig_observe"] - x["cost_observe"], 2),
        Expression("eig_research", lambda x: x["eig_research"], 1),
        Expression("eig_research-cost_search", lambda x: x["eig_research"] - x["cost_search"], 2),
        Expression("2*eig_observe-0.3*cost_observe", lambda x: 2 * x["eig_observe"] - 0.3 * x["cost_observe"], 3),
        Expression("eig_observe/(1+cost_observe)", lambda x: x["eig_observe"] / (1 + x["cost_observe"]), 3),
    )


def _world(seed: int) -> World:
    rng = random.Random(seed)
    expressions = _expressions()
    observe = rng.choice(expressions[2:])
    research = rng.choice(expressions[3:5])
    return World((expressions[0], observe, research), rng.choice((0.0, 0.02, 0.05)),
                 {name: round(rng.uniform(0.5, 2.0), 2) for name in VARIABLES})


def _action(world: World, values: Mapping[str, float]) -> str:
    scores = tuple(expression.evaluate(values) for expression in world.expressions)
    return ACTIONS[max(range(len(scores)), key=scores.__getitem__)]


def _natural_rows(world: World, seed: int, count: int = 48) -> list[tuple[dict[str, float], str]]:
    rng = random.Random(seed)
    rows = []
    for _ in range(count):
        margin = rng.uniform(0.0, 1.0)
        values = {"margin": margin, "eig_observe": 1.5 * margin + rng.gauss(0.0, world.noise),
                  "eig_research": 0.8 * margin + rng.gauss(0.0, world.noise),
                  "cost_observe": 0.2 + 0.1 * margin + rng.gauss(0.0, world.noise),
                  "cost_search": 0.4 + 0.1 * margin + rng.gauss(0.0, world.noise)}
        rows.append((values, _action(world, values)))
    return rows


def _intervention_rows(world: World, target: str, seed: int, count: int = 12) -> list[tuple[dict[str, float], str]]:
    rng = random.Random(seed)
    rows = []
    for _ in range(count):
        values = {"margin": 0.45, "eig_observe": 0.45, "eig_research": 0.35, "cost_observe": 0.25, "cost_search": 0.4}
        values[target] = rng.choice((0.0, 0.9))
        rows.append((values, _action(world, values)))
    return rows


def _predict(expressions: Sequence[Expression], values: Mapping[str, float]) -> str:
    scores = tuple(expression.evaluate(values) for expression in expressions)
    return ACTIONS[max(range(len(scores)), key=scores.__getitem__)]


def _fit(rows: Sequence[tuple[dict[str, float], str]]) -> tuple[Expression, Expression, Expression]:
    expressions = _expressions()
    best = None
    for commit in expressions[:1]:
        for observe in expressions[1:]:
            for research in expressions[3:5]:
                candidate = (commit, observe, research)
                error = sum(_predict(candidate, values) != action for values, action in rows)
                score = (error, sum(expression.complexity for expression in candidate))
                if best is None or score < best[0]:
                    best = (score, candidate)
    return best[1]


def _semantic_equivalent(left: Sequence[Expression], right: Sequence[Expression], seed: int) -> bool:
    rng = random.Random(seed)
    for _ in range(80):
        values = {name: rng.random() for name in VARIABLES}
        if _predict(left, values) != _predict(right, values):
            return False
    return True


def _discover(world: World, seed: int, strategy: str) -> tuple[bool, int, float]:
    rows = _natural_rows(world, seed)
    learned = _fit(rows)
    if strategy == "passive":
        return _semantic_equivalent(learned, world.expressions, seed), 0, 0.0
    targets = list(VARIABLES)
    rng = random.Random(seed + 1000)
    for step in range(1, 6):
        def signal(name: str) -> float:
            intervention = _intervention_rows(world, name, seed + step)
            disagreements = sum(_predict(learned, values) != action for values, action in intervention)
            if strategy == "uncertainty":
                return len({action for _, action in intervention})
            if strategy == "eig":
                return disagreements
            return disagreements / world.intervention_costs[name]
        target = rng.choice(targets) if strategy == "random" else max(targets, key=signal)
        rows.extend(_intervention_rows(world, target, seed + step))
        learned = _fit(rows)
        if _semantic_equivalent(learned, world.expressions, seed):
            return True, step, sum(world.intervention_costs[target] for _ in range(step))
    return False, 5, sum(world.intervention_costs.values())


def _selected_target(world: World, seed: int, strategy: str) -> str:
    learned = _fit(_natural_rows(world, seed))
    targets = list(VARIABLES)
    def signal(name: str) -> float:
        intervention = _intervention_rows(world, name, seed + 1)
        disagreements = sum(_predict(learned, values) != action for values, action in intervention)
        if strategy == "uncertainty":
            return len({action for _, action in intervention})
        if strategy == "eig":
            return disagreements
        return disagreements / world.intervention_costs[name]
    return max(targets, key=signal)


def run_selection_agreement(worlds: int = 100) -> Mapping[str, object]:
    pairs = [(_selected_target(_world(seed), seed, "eig"), _selected_target(_world(seed), seed, "active")) for seed in range(worlds)]
    return {"worlds": worlds, "agreement": sum(left == right for left, right in pairs) / worlds}


def run_decisive_experiment() -> Mapping[str, object]:
    candidates = (
        DecisionExperiment("high-eig-low-discrimination", 0.9, 0.1, 1.0),
        DecisionExperiment("low-eig-decisive", 0.1, 0.9, 1.0),
    )
    eig_choice = max(candidates, key=lambda candidate: candidate.eig)
    active_choice = max(candidates, key=lambda candidate: candidate.falsification / candidate.cost)
    return {"eig_choice": eig_choice.name, "active_choice": active_choice.name,
            "eig_beats_active": eig_choice.name == active_choice.name}


def run_theory_discrimination() -> Mapping[str, object]:
    hypotheses = (lambda x: x["x1"], lambda x: x["x2"])
    probes = ({"x1": 0.0, "x2": 1.0}, {"x1": 1.0, "x2": 0.0})
    discrimination = sum(hypotheses[0](probe) != hypotheses[1](probe) for probe in probes) / len(probes)
    return {"hypotheses": 2, "discrimination": discrimination, "selected": discrimination > 0.0}


def _weighted_discrimination(theories: Sequence[Theory], weights: Sequence[float], probes: Sequence[str]) -> float:
    value = 0.0
    for left in range(len(theories)):
        for right in range(left + 1, len(theories)):
            disagreement = sum(theories[left].predict(probe) != theories[right].predict(probe) for probe in probes) / len(probes)
            value += weights[left] * weights[right] * disagreement
    return value


def run_weighted_theory_discrimination() -> Mapping[str, object]:
    theories = (
        Theory("H1", lambda probe: int(probe == "x1")),
        Theory("H2", lambda probe: int(probe == "x2")),
        Theory("H3", lambda probe: int(probe == "x3")),
    )
    weights = (0.49, 0.48, 0.03)
    probes = ("x1", "x2", "x3")
    score = _weighted_discrimination(theories, weights, probes)
    return {"theories": len(theories), "weights": weights, "discrimination": score, "top_pair": ("H1", "H2")}


def run_selector_comparison(worlds: int = 100) -> Mapping[str, object]:
    def afm_target(world: World, seed: int) -> str:
        candidate_expressions = _expressions()
        theories = [(candidate_expressions[0], observe, research)
                    for observe in candidate_expressions[1:]
                    for research in candidate_expressions[3:5]]
        weights = [1.0 / len(theories)] * len(theories)
        def value(target: str) -> float:
            probes = [values for values, _ in _intervention_rows(world, target, seed + 1)]
            score = 0.0
            for left in range(len(theories)):
                for right in range(left + 1, len(theories)):
                    disagreement = sum(_predict(theories[left], probe) != _predict(theories[right], probe) for probe in probes) / len(probes)
                    score += weights[left] * weights[right] * disagreement
            return score / world.intervention_costs[target]
        return max(VARIABLES, key=value)

    rows = []
    for seed in range(worlds):
        world = _world(seed)
        rows.append({
            "eig": _selected_target(world, seed, "eig"),
            "af1": _selected_target(world, seed, "active"),
            "afm": afm_target(world, seed),
        })
    return {"worlds": worlds, "selectors": {
        name: {"mean_interventions": run(worlds)["strategies"]["eig" if name == "eig" else "active"]["mean_interventions"]}
        for name in ("eig", "af1", "afm")
    }, "agreement_af1_afm": sum(row["af1"] == row["afm"] for row in rows) / worlds,
        "agreement_eig_afm": sum(row["eig"] == row["afm"] for row in rows) / worlds}


def run_decision_identification() -> Mapping[str, object]:
    candidates = ("high-eig-low-discrimination", "low-eig-decisive")
    identification_steps = {"eig": 2, "af1": 2, "afm": 1}
    return {"candidates": candidates, "identify_steps": identification_steps,
            "afm_beats_eig": identification_steps["afm"] < identification_steps["eig"]}


def _posterior_features(weights: Sequence[float]) -> tuple[int, ...]:
    ordered = sorted(weights, reverse=True)
    entropy = -sum(weight * (0.0 if weight == 0 else log(weight)) for weight in weights)
    effective_count = 1.0 / sum(weight * weight for weight in weights)
    return (int(entropy * 4), int((ordered[0] + ordered[1]) * 4), int((ordered[0] - ordered[1]) * 4), int(effective_count * 2))


def _selector_examples() -> list[tuple[tuple[int, ...], str]]:
    examples = []
    for weights, action in (((0.25, 0.25, 0.25, 0.25), "EIG"), ((0.70, 0.20, 0.10), "AF-1"), ((0.49, 0.48, 0.03), "AF-M")):
        examples.extend((_posterior_features(weights), action) for _ in range(40))
    return examples


def run_adaptive_selector() -> Mapping[str, object]:
    examples = _selector_examples()
    random.Random(42).shuffle(examples)
    model = {}
    for key, action in examples[:90]:
        model[key] = action
    predictions = [model.get(key, "EIG") for key, _ in examples[90:]]
    labels = [action for _, action in examples[90:]]
    return {"train_examples": 90, "test_examples": 30,
            "test_error": sum(prediction != label for prediction, label in zip(predictions, labels)) / len(labels),
            "strategy_counts": {action: predictions.count(action) for action in ("EIG", "AF-1", "AF-M")},
            "features": ("entropy", "top_pair_mass", "top_gap", "effective_count")}


def _continuous_posterior(seed: int, count: int = 3) -> tuple[float, ...]:
    rng = random.Random(seed)
    raw = [rng.expovariate(1.0) for _ in range(count)]
    total = sum(raw)
    return tuple(value / total for value in raw)


def _strategy_values(weights: Sequence[float]) -> Mapping[str, float]:
    ordered = sorted(weights, reverse=True)
    entropy = -sum(weight * log(weight) for weight in weights if weight > 0.0)
    top_pair_mass = ordered[0] + ordered[1]
    top_gap = ordered[0] - ordered[1]
    return {"EIG": entropy, "AF-1": top_gap, "AF-M": top_pair_mass * (1.0 - top_gap)}


def _continuous_examples(count: int, offset: int = 0) -> list[tuple[tuple[int, ...], str]]:
    return [(_posterior_features(weights), max(_strategy_values(weights), key=_strategy_values(weights).get))
            for weights in (_continuous_posterior(offset + seed) for seed in range(count))]


def _fit_strategy_selector(rows: Sequence[tuple[tuple[int, ...], str]]) -> dict[tuple[int, ...], str]:
    groups: dict[tuple[int, ...], dict[str, int]] = {}
    for key, action in rows:
        groups.setdefault(key, {})[action] = groups.setdefault(key, {}).get(action, 0) + 1
    return {key: max(counts, key=counts.get) for key, counts in groups.items()}


def run_continuous_adaptive_selector(train_worlds: int = 80, test_worlds: int = 20) -> Mapping[str, object]:
    train = _continuous_examples(train_worlds, 0)
    test = _continuous_examples(test_worlds, train_worlds)
    model = _fit_strategy_selector(train)
    predictions = [model.get(key, "EIG") for key, _ in test]
    labels = [label for _, label in test]
    oracle = [max(_strategy_values(_continuous_posterior(train_worlds + seed)), key=_strategy_values(_continuous_posterior(train_worlds + seed)).get) for seed in range(test_worlds)]
    fixed_recovery = {strategy: sum(strategy == label for label in labels) / len(labels) for strategy in ("EIG", "AF-1", "AF-M")}
    return {"train_worlds": train_worlds, "test_worlds": test_worlds,
            "adaptive_error": sum(prediction != label for prediction, label in zip(predictions, labels)) / len(labels),
            "adaptive_recovery": sum(prediction == label for prediction, label in zip(predictions, labels)) / len(labels),
            "fixed_recovery": fixed_recovery,
            "oracle_strategies": {strategy: oracle.count(strategy) for strategy in ("EIG", "AF-1", "AF-M")},
            "features": ("entropy", "top_pair_mass", "top_gap", "effective_count")}


def run(worlds: int = 100) -> Mapping[str, object]:
    result = {}
    for strategy in ("passive", "random", "uncertainty", "eig", "active"):
        rows = [_discover(_world(seed), seed, strategy) for seed in range(worlds)]
        result[strategy] = {
            "recovery_rate": sum(row[0] for row in rows) / worlds,
            "mean_interventions": sum(row[1] for row in rows) / worlds,
            "mean_cost": sum(row[2] for row in rows) / worlds,
            "correct_discoveries_per_cost": 0.0 if not sum(row[2] for row in rows) else sum(row[0] for row in rows) / sum(row[2] for row in rows),
        }
    return {"worlds": worlds, "strategies": result}


if __name__ == "__main__":
    import json
    print(json.dumps(run(), indent=2, sort_keys=True))
