"""Independent random-world benchmark for theory-strategy selection."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
from math import log
import random
from typing import Mapping, Sequence

STRATEGIES = ("EIG", "AF-1", "AF-M")
VARIABLES = 5


@dataclass(frozen=True)
class RandomWorld:
    coefficients: tuple[tuple[float, ...], ...]
    posterior: tuple[float, ...]
    noise: float
    costs: tuple[float, ...]


@dataclass(frozen=True)
class WorldResult:
    theory_count: int
    noise: float
    selected: str
    oracle: str
    recovery: bool
    cost: float


def _world(seed: int) -> RandomWorld:
    rng = random.Random(seed)
    theory_count = rng.randint(3, 6)
    coefficients = tuple(tuple(round(rng.uniform(-1.5, 1.5), 3) for _ in range(VARIABLES)) for _ in range(theory_count))
    raw = [rng.expovariate(1.0) for _ in range(theory_count)]
    total = sum(raw)
    return RandomWorld(coefficients, tuple(value / total for value in raw), rng.uniform(0.0, 0.25),
                       tuple(rng.uniform(0.5, 2.0) for _ in range(VARIABLES)))


def _prediction(world: RandomWorld, theory: int, probe: Sequence[float]) -> int:
    score = sum(left * right for left, right in zip(world.coefficients[theory], probe))
    return int(score >= 0.0)


def _probe(world: RandomWorld, target: int, value: float) -> tuple[float, ...]:
    probe = [0.0] * VARIABLES
    probe[target] = value
    return tuple(probe)


def _entropy(weights: Sequence[float]) -> float:
    return -sum(weight * log(weight) for weight in weights if weight > 0.0)


def _intervention_metrics(world: RandomWorld, target: int) -> tuple[float, float, float]:
    probes = (_probe(world, target, -1.0), _probe(world, target, 1.0))
    outcomes = [[_prediction(world, theory, probe) for theory in range(len(world.posterior))] for probe in probes]
    posterior = world.posterior
    eig = 0.0
    for outcome in (0, 1):
        mass = sum(weight for weight, prediction in zip(posterior, outcomes[1]) if prediction == outcome)
        if mass > 0.0:
            conditional = [weight / mass for weight, prediction in zip(posterior, outcomes[1]) if prediction == outcome]
            eig += mass * _entropy(conditional)
    eig = _entropy(posterior) - eig
    ordered = sorted(range(len(posterior)), key=posterior.__getitem__, reverse=True)
    af1 = float(outcomes[0][ordered[0]] != outcomes[0][ordered[1]] or outcomes[1][ordered[0]] != outcomes[1][ordered[1]])
    afm = 0.0
    for left in range(len(posterior)):
        for right in range(left + 1, len(posterior)):
            disagreement = sum(outcome[left] != outcome[right] for outcome in outcomes) / len(outcomes)
            afm += posterior[left] * posterior[right] * disagreement
    return eig, af1, afm


def _strategy_values(world: RandomWorld) -> Mapping[str, float]:
    metrics = [_intervention_metrics(world, target) for target in range(VARIABLES)]
    values = {
        "EIG": max(metric[0] / world.costs[target] for target, metric in enumerate(metrics)),
        "AF-1": max(metric[1] / world.costs[target] for target, metric in enumerate(metrics)),
        "AF-M": max(metric[2] / world.costs[target] for target, metric in enumerate(metrics)),
    }
    return values


def _strategy_value_features(world: RandomWorld) -> tuple[float, ...]:
    metrics = [_intervention_metrics(world, target) for target in range(VARIABLES)]
    return tuple(max(metric[index] / world.costs[target] for target, metric in enumerate(metrics))
                 for index in range(3))


def _features(world: RandomWorld) -> tuple[int, ...]:
    ordered = sorted(world.posterior, reverse=True)
    effective = 1.0 / sum(weight * weight for weight in world.posterior)
    return (int(_entropy(world.posterior) * 4), int((ordered[0] + ordered[1]) * 4),
            int((ordered[0] - ordered[1]) * 4), int(effective * 2), int(world.noise * 10), len(world.posterior))


def _learn_selector(worlds: Sequence[RandomWorld]) -> dict[tuple[int, ...], str]:
    groups: dict[tuple[int, ...], dict[str, int]] = {}
    for world in worlds:
        values = _strategy_values(world)
        key = _features(world)
        label = max(values, key=values.get)
        groups.setdefault(key, {})[label] = groups.setdefault(key, {}).get(label, 0) + 1
    return {key: max(counts, key=counts.get) for key, counts in groups.items()}


def _evaluate(world: RandomWorld, strategy: str) -> WorldResult:
    values = _strategy_values(world)
    oracle = max(values, key=values.get)
    selected = strategy
    recovery = selected == oracle
    return WorldResult(len(world.posterior), world.noise, selected, oracle, recovery, 1.0)


def _value_aware_strategy(world: RandomWorld) -> str:
    values = _strategy_value_features(world)
    return STRATEGIES[max(range(len(values)), key=values.__getitem__)]


def run_value_aware_comparison(train_worlds: int = 80, test_worlds: int = 20) -> Mapping[str, object]:
    test = [_world(train_worlds + seed) for seed in range(test_worlds)]
    rows = []
    for world in test:
        values = _strategy_values(world)
        oracle = max(values, key=values.get)
        selected = _value_aware_strategy(world)
        ordered = sorted(world.posterior, reverse=True)
        rows.append({"theory_count": len(world.posterior), "noise": world.noise,
                     "top_pair_mass": ordered[0] + ordered[1], "top_gap": ordered[0] - ordered[1],
                     "oracle": oracle, "selected": selected, "recovery": selected == oracle})
    af1_rows = [row for row in rows if row["oracle"] == "AF-1"]
    return {"train_worlds": train_worlds, "test_worlds": test_worlds,
            "value_aware_recovery": sum(row["recovery"] for row in rows) / len(rows),
            "af1_oracle_worlds": len(af1_rows),
            "af1_mean_theory_count": sum(row["theory_count"] for row in af1_rows) / max(1, len(af1_rows)),
            "af1_mean_noise": sum(row["noise"] for row in af1_rows) / max(1, len(af1_rows)),
            "af1_mean_top_pair_mass": sum(row["top_pair_mass"] for row in af1_rows) / max(1, len(af1_rows)),
            "rows": rows}


def _meta_features(world: RandomWorld) -> Mapping[str, float]:
    ordered = sorted(world.posterior, reverse=True)
    effective = 1.0 / sum(weight * weight for weight in world.posterior)
    values = _strategy_value_features(world)
    return {"entropy": _entropy(world.posterior), "effective_count": effective,
            "top_pair_mass": ordered[0] + ordered[1], "top_gap": ordered[0] - ordered[1],
            "max_eig": values[0], "max_af1": values[1], "max_afm": values[2]}


def _meta_predict(features: Mapping[str, float], names: Sequence[str]) -> str:
    values = {"EIG": features.get("max_eig", 0.0) if "max_eig" in names else 0.0,
              "AF-1": features.get("max_af1", 0.0) if "max_af1" in names else 0.0,
              "AF-M": features.get("max_afm", 0.0) if "max_afm" in names else 0.0}
    return max(values, key=values.get)


def run_meta_controller_recompression(train_worlds: int = 80, test_worlds: int = 20) -> Mapping[str, object]:
    names = ("entropy", "effective_count", "top_pair_mass", "top_gap", "max_eig", "max_af1", "max_afm")
    train = [_meta_features(_world(seed)) for seed in range(train_worlds)]
    test = [_meta_features(_world(train_worlds + seed)) for seed in range(test_worlds)]
    oracle_train = [_meta_predict(features, ("max_eig", "max_af1", "max_afm")) for features in train]
    candidates = []
    for size in range(1, len(names) + 1):
        for subset in combinations(names, size):
            predictions = [_meta_predict(features, subset) for features in train]
            error = sum(prediction != oracle for prediction, oracle in zip(predictions, oracle_train)) / train_worlds
            candidates.append((error + 0.01 * size, subset, error))
    _, selected, train_error = min(candidates, key=lambda row: (row[0], len(row[1]), row[1]))
    oracle_test = [_meta_predict(features, ("max_eig", "max_af1", "max_afm")) for features in test]
    test_error = sum(_meta_predict(features, selected) != oracle for features, oracle in zip(test, oracle_test)) / test_worlds
    return {"train_worlds": train_worlds, "test_worlds": test_worlds,
            "selected_features": selected, "representation_cost": len(selected),
            "full_cost": 3, "train_error": train_error, "test_error": test_error,
            "compression_ratio": len(selected) / len(names)}


def _afm_world(seed: int) -> Mapping[str, float | str]:
    rng = random.Random(seed)
    pair_mass = rng.uniform(0.65, 0.98)
    gap = rng.uniform(0.0, 0.25)
    values = {"entropy": rng.uniform(0.6, 1.4), "effective_count": rng.uniform(1.5, 4.5),
              "top_pair_mass": pair_mass, "top_gap": gap,
              "max_eig": rng.uniform(0.1, 0.5), "max_af1": rng.uniform(0.1, 0.5),
              "max_afm": 0.9 if seed % 4 == 0 else rng.uniform(0.05, 0.35)}
    oracle = max(("EIG", values["max_eig"]), ("AF-1", values["max_af1"]),
                 ("AF-M", values["max_afm"]), key=lambda item: item[1])[0]
    values["oracle"] = oracle
    return values


def _ablation_key(row: Mapping[str, float | str], features: Sequence[str]) -> tuple[int, ...]:
    return tuple(int(float(row[feature]) * 10) for feature in features)


def _fit_ablation(rows: Sequence[Mapping[str, float | str]], features: Sequence[str]) -> dict[tuple[int, ...], str]:
    groups: dict[tuple[int, ...], dict[str, int]] = {}
    for row in rows:
        key = _ablation_key(row, features)
        label = str(row["oracle"])
        groups.setdefault(key, {})[label] = groups.setdefault(key, {}).get(label, 0) + 1
    return {key: max(counts, key=counts.get) for key, counts in groups.items()}


def _ablation_error(rows: Sequence[Mapping[str, float | str]], model: Mapping[tuple[int, ...], str], features: Sequence[str]) -> float:
    return sum(model.get(_ablation_key(row, features), "EIG") != row["oracle"] for row in rows) / len(rows)


def run_recursive_compression_ablation(train_worlds: int = 80, test_worlds: int = 20) -> Mapping[str, object]:
    names = ("entropy", "effective_count", "top_pair_mass", "top_gap", "max_eig", "max_af1", "max_afm")
    train = [_afm_world(seed) for seed in range(train_worlds)]
    test = [_afm_world(train_worlds + seed) for seed in range(test_worlds)]
    levels = {
        "R0_object_only": ("top_gap",),
        "R1_uncertainty": ("entropy", "effective_count", "top_gap"),
        "R2_theory": ("max_eig", "max_af1", "max_afm"),
    }
    results = {}
    for level, features in levels.items():
        model = _fit_ablation(train, features)
        results[level] = {"features": features, "train_error": _ablation_error(train, model, features),
                          "ood_error": _ablation_error(test, model, features), "representation_cost": len(features)}
    candidates = []
    for size in range(1, len(names) + 1):
        for features in combinations(names, size):
            model = _fit_ablation(train, features)
            error = _ablation_error(train, model, features)
            candidates.append((error + 0.02 * size, features, error))
    _, selected, train_error = min(candidates, key=lambda row: (row[0], len(row[1]), row[1]))
    model = _fit_ablation(train, selected)
    results["R3_recompressed"] = {"features": selected, "train_error": train_error,
                                  "ood_error": _ablation_error(test, model, selected),
                                  "representation_cost": len(selected)}
    return {"train_worlds": train_worlds, "test_worlds": test_worlds, "afm_world_fraction": 0.25, "levels": results}


def _error_by_environment(rows: Sequence[Mapping[str, float | str]], model: Mapping[tuple[int, ...], str], features: Sequence[str], environment_size: int = 10) -> list[float]:
    return [_ablation_error(rows[index:index + environment_size], model, features)
            for index in range(0, len(rows), environment_size)]


def run_robust_mdl(train_worlds: int = 60, validation_worlds: int = 20, test_worlds: int = 20) -> Mapping[str, object]:
    names = ("entropy", "effective_count", "top_pair_mass", "top_gap", "max_eig", "max_af1", "max_afm")
    fit_rows = [_afm_world(seed) for seed in range(train_worlds)]
    validation_rows = [_afm_world(train_worlds + seed) for seed in range(validation_worlds)]
    test_rows = [_afm_world(train_worlds + validation_worlds + seed) for seed in range(test_worlds)]
    candidates = []
    for size in range(1, len(names) + 1):
        for features in combinations(names, size):
            model = _fit_ablation(fit_rows, features)
            errors = _error_by_environment(validation_rows, model, features)
            mean_error = sum(errors) / len(errors)
            worst_error = max(errors)
            variance = sum((error - mean_error) ** 2 for error in errors) / len(errors)
            score = size + 8.0 * mean_error + 8.0 * worst_error + 8.0 * variance
            candidates.append((score, features, model, errors))
    _, selected, model, validation_errors = min(candidates, key=lambda row: (row[0], len(row[1]), row[1]))
    test_errors = _error_by_environment(test_rows, model, selected)
    naive_features = names
    naive_model = _fit_ablation(fit_rows, naive_features)
    naive_test_errors = _error_by_environment(test_rows, naive_model, naive_features)
    return {"fit_worlds": train_worlds, "validation_worlds": validation_worlds, "test_worlds": test_worlds,
            "selected_features": selected, "representation_cost": len(selected),
            "validation_mean_error": sum(validation_errors) / len(validation_errors),
            "validation_worst_error": max(validation_errors),
            "validation_variance": sum((error - sum(validation_errors) / len(validation_errors)) ** 2 for error in validation_errors) / len(validation_errors),
            "test_mean_error": sum(test_errors) / len(test_errors),
            "naive_test_error": sum(naive_test_errors) / len(naive_test_errors)}


def _pareto_front(candidates: Sequence[Mapping[str, object]]) -> list[Mapping[str, object]]:
    front = []
    for candidate in candidates:
        dominated = any(
            all(other[metric] <= candidate[metric] for metric in ("cost", "mean_error", "worst_error", "variance"))
            and any(other[metric] < candidate[metric] for metric in ("cost", "mean_error", "worst_error", "variance"))
            for other in candidates
        )
        if not dominated:
            front.append(candidate)
    return front


def run_pareto_frontier(train_worlds: int = 60, validation_worlds: int = 20, test_worlds: int = 20) -> Mapping[str, object]:
    names = ("entropy", "effective_count", "top_pair_mass", "top_gap", "max_eig", "max_af1", "max_afm")
    fit_rows = [_afm_world(seed) for seed in range(train_worlds)]
    validation_rows = [_afm_world(train_worlds + seed) for seed in range(validation_worlds)]
    test_rows = [_afm_world(train_worlds + validation_worlds + seed) for seed in range(test_worlds)]
    candidates = []
    for size in range(1, len(names) + 1):
        for features in combinations(names, size):
            model = _fit_ablation(fit_rows, features)
            errors = _error_by_environment(validation_rows, model, features)
            mean_error = sum(errors) / len(errors)
            candidates.append({"features": features, "cost": size, "mean_error": mean_error,
                               "worst_error": max(errors), "variance": sum((error - mean_error) ** 2 for error in errors) / len(errors),
                               "model": model})
    front = _pareto_front(candidates)
    selected = min(front, key=lambda candidate: (candidate["mean_error"], candidate["cost"], candidate["features"]))
    test_errors = _error_by_environment(test_rows, selected["model"], selected["features"])
    return {"frontier": [{key: value for key, value in candidate.items() if key != "model"} for candidate in front],
            "selected_features": selected["features"], "selected_test_error": sum(test_errors) / len(test_errors),
            "frontier_size": len(front)}


def run_diagnostic_environment_selection(train_worlds: int = 60, validation_worlds: int = 20, test_worlds: int = 20) -> Mapping[str, object]:
    names = ("entropy", "effective_count", "top_pair_mass", "top_gap", "max_eig", "max_af1", "max_afm")
    fit_rows = [_afm_world(seed) for seed in range(train_worlds)]
    validation_rows = [_afm_world(train_worlds + seed) for seed in range(validation_worlds)]
    test_rows = [_afm_world(train_worlds + validation_worlds + seed) for seed in range(test_worlds)]
    left_features, right_features = ("max_eig", "max_af1"), ("max_af1", "max_afm")
    left_model = _fit_ablation(fit_rows, left_features)
    right_model = _fit_ablation(fit_rows, right_features)
    disagreements = [sum(left_model.get(_ablation_key(row, left_features), "EIG") != right_model.get(_ablation_key(row, right_features), "EIG") for row in (validation_rows[index],)) for index in range(validation_worlds)]
    diagnostic_index = max(range(validation_worlds), key=disagreements.__getitem__)
    diagnostic = validation_rows[diagnostic_index]
    diagnostic_rows = [diagnostic] + [row for index, row in enumerate(validation_rows) if index != diagnostic_index]
    model = _fit_ablation(diagnostic_rows, right_features)
    test_error = _ablation_error(test_rows, model, right_features)
    return {"candidate_features": (left_features, right_features), "diagnostic_index": diagnostic_index,
            "diagnostic_disagreement": disagreements[diagnostic_index], "test_error": test_error}


def _failure_environment(seed: int) -> Mapping[str, float | str | int]:
    row = dict(_afm_world(seed))
    row["family"] = seed % 4
    return row


def _candidate_errors(rows: Sequence[Mapping[str, float | str]], features: Sequence[str], models: Mapping[tuple[str, ...], Mapping[tuple[int, ...], str]]) -> list[float]:
    return [_ablation_error((row,), models[tuple(features)], features) for row in rows]


def run_failure_family_selection(train_worlds: int = 60, validation_worlds: int = 20, test_worlds: int = 20) -> Mapping[str, object]:
    names = ("entropy", "effective_count", "top_pair_mass", "top_gap", "max_eig", "max_af1", "max_afm")
    fit_rows = [_afm_world(seed) for seed in range(train_worlds)]
    validation_rows = [_failure_environment(train_worlds + seed) for seed in range(validation_worlds)]
    test_rows = [_failure_environment(train_worlds + validation_worlds + seed) for seed in range(test_worlds)]
    candidates = [features for size in range(1, len(names) + 1) for features in combinations(names, size)]
    models = {features: _fit_ablation(fit_rows, features) for features in candidates}
    signatures = []
    for index, row in enumerate(validation_rows):
        errors = tuple(round(_ablation_error((row,), models[features], features), 2) for features in candidates)
        signatures.append((row["family"], errors, index))
    family_representatives = {family: min((item for item in signatures if item[0] == family), key=lambda item: item[1])[2]
                             for family in sorted({int(row["family"]) for row in validation_rows})}
    selected_indices = tuple(family_representatives.values())
    random_indices = tuple(range(min(4, validation_worlds)))

    def choose(indices: Sequence[int]) -> tuple[tuple[str, ...], float]:
        scored = []
        for features in candidates:
            errors = [_ablation_error((validation_rows[index],), models[features], features) for index in indices]
            scored.append((len(features) + 8.0 * max(errors) + 8.0 * sum(errors) / len(errors), features))
        score, features = min(scored, key=lambda item: (item[0], len(item[1]), item[1]))
        return features, score

    family_features, family_score = choose(selected_indices)
    random_features, random_score = choose(random_indices)
    family_model = models[family_features]
    random_model = models[random_features]
    return {"failure_families": len(family_representatives), "selected_environment_count": len(selected_indices),
            "family_indices": selected_indices, "family_features": family_features, "family_score": family_score,
            "random_features": random_features, "random_score": random_score,
            "family_test_error": _ablation_error(test_rows, family_model, family_features),
            "random_test_error": _ablation_error(test_rows, random_model, random_features)}


def run(train_worlds: int = 80, test_worlds: int = 20) -> Mapping[str, object]:
    train = [_world(seed) for seed in range(train_worlds)]
    test = [_world(train_worlds + seed) for seed in range(test_worlds)]
    model = _learn_selector(train)
    train_predictions = [model.get(_features(world), "EIG") for world in train]
    train_oracles = [max(_strategy_values(world), key=_strategy_values(world).get) for world in train]
    adaptive_rows = []
    for world in test:
        values = _strategy_values(world)
        oracle = max(values, key=values.get)
        selected = model.get(_features(world), "EIG")
        adaptive_rows.append(WorldResult(len(world.posterior), world.noise, selected, oracle, selected == oracle, 1.0))
    fixed = {strategy: sum(_evaluate(world, strategy).recovery for world in test) / test_worlds for strategy in STRATEGIES}
    return {
        "train_worlds": train_worlds,
        "test_worlds": test_worlds,
        "adaptive_recovery": sum(row.recovery for row in adaptive_rows) / test_worlds,
        "adaptive_train_recovery": sum(prediction == oracle for prediction, oracle in zip(train_predictions, train_oracles)) / train_worlds,
        "fixed_recovery": fixed,
        "mean_theory_count": sum(row.theory_count for row in adaptive_rows) / test_worlds,
        "mean_noise": sum(row.noise for row in adaptive_rows) / test_worlds,
        "adaptive_rows": [asdict(row) for row in adaptive_rows],
    }


if __name__ == "__main__":
    import json
    print(json.dumps(run(), indent=2, sort_keys=True))
