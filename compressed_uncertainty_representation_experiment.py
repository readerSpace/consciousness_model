"""MDL-style search for a compact uncertainty representation."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from itertools import combinations
import json
import random
from typing import Dict, Mapping, Sequence

from metacognitive_summary_experiment import _distributions
from partial_observation_experiment import _workspace
from uncertainty_encoder_experiment import UncertaintyState, _choose, _semantic_rows, _state


FEATURES = (
    "entropy",
    "margin",
    "tail_mass",
    "effective_count",
    "disagreement",
    "prediction_error",
    "eig_research",
    "eig_observe",
    "cost_observe",
    "cost_search",
)


@dataclass(frozen=True)
class Example:
    values: tuple[float, ...]
    action: str


@dataclass(frozen=True)
class RepresentationResult:
    features: tuple[str, ...]
    train_error: float
    test_error: float
    representation_cost: float
    mdl: float


@dataclass(frozen=True)
class InterventionResult:
    examples: int
    margin_only_error: float
    full_representation_error: float
    distinct_actions: tuple[str, ...]


@dataclass(frozen=True)
class InterventionCandidate:
    feature: str
    discrimination: float
    cost: float
    value_per_cost: float


def _bin(feature: str, value: float) -> int:
    if feature in {"entropy", "effective_count"}:
        return min(7, int(value * 2.0))
    return min(7, int(value * 10.0))


def _key(example: Example, feature_indexes: Sequence[int]) -> tuple[int, ...]:
    return tuple(_bin(FEATURES[index], example.values[index]) for index in feature_indexes)


def _examples() -> list[Example]:
    rows: list[Example] = []
    for noise in range(0, 21, 2):
        for seed in range(12):
            naive, corrected, _ = _distributions(noise, seed)
            probabilities = [corrected.top_probability, max(0.0, 1.0 - corrected.top_probability - corrected.other_mass)]
            state = _state(probabilities, eig_observe=corrected.other_mass)
            rows.append(Example(
                (state.entropy, state.margin, corrected.other_mass, corrected.effective_count,
                 state.disagreement, state.prediction_error, state.eig_research, state.eig_observe,
                 state.cost_observe, state.cost_search),
                _choose(state),
            ))
    return rows


def _intervention_examples() -> list[Example]:
    rows: list[Example] = []
    for eig_observe in (0.0, 0.8):
        action = _choose(UncertaintyState(1.0, 0.1, 0.5, 0.5, eig_observe, 0.0, 0.08, 0.12))
        rows.extend(Example(
            (1.0, 0.1, 0.2, 2.0, 0.5, 0.5, 0.0, eig_observe, 0.08, 0.12), action,
        ) for _ in range(20))
    return rows


def _cost_intervention_examples() -> list[Example]:
    rows: list[Example] = []
    for cost_observe in (0.05, 1.0):
        state = UncertaintyState(1.0, 0.1, 0.5, 0.5, 0.8, 0.0, cost_observe, 0.12)
        action = _choose(state)
        rows.extend(Example(
            (1.0, 0.1, 0.2, 2.0, 0.5, 0.5, 0.0, 0.8, cost_observe, 0.12), action,
        ) for _ in range(20))
    return rows


def _candidate_intervention(feature: str) -> list[Example]:
    if feature == "eig_observe":
        states = (UncertaintyState(1.0, 0.1, 0.5, 0.5, 0.0, 0.0, 0.08, 0.12),
                  UncertaintyState(1.0, 0.1, 0.5, 0.5, 0.8, 0.0, 0.08, 0.12))
    elif feature == "cost_observe":
        states = (UncertaintyState(1.0, 0.1, 0.5, 0.5, 0.8, 0.0, 0.05, 0.12),
                  UncertaintyState(1.0, 0.1, 0.5, 0.5, 0.8, 0.0, 1.0, 0.12))
    elif feature == "cost_search":
        states = (UncertaintyState(1.0, 0.1, 0.5, 0.5, 0.0, 0.8, 0.08, 0.05),
                  UncertaintyState(1.0, 0.1, 0.5, 0.5, 0.0, 0.8, 0.08, 1.0))
    else:
        states = (UncertaintyState(1.0, 0.1, 0.5, 0.5, 0.0, 0.0, 0.08, 0.12),) * 2
    return [Example(
        (state.entropy, state.margin, 0.2, 2.0, state.disagreement, state.prediction_error,
         state.eig_research, state.eig_observe, state.cost_observe, state.cost_search),
        _choose(state),
    ) for state in states for _ in range(20)]


def _fit(train: Sequence[Example], feature_indexes: Sequence[int]) -> Dict[tuple[int, ...], str]:
    groups: Dict[tuple[int, ...], Counter[str]] = {}
    for example in train:
        groups.setdefault(_key(example, feature_indexes), Counter())[example.action] += 1
    return {key: counts.most_common(1)[0][0] for key, counts in groups.items()}


def _error(rows: Sequence[Example], model: Mapping[tuple[int, ...], str], feature_indexes: Sequence[int]) -> float:
    errors = 0
    for example in rows:
        prediction = model.get(_key(example, feature_indexes), "commit")
        errors += prediction != example.action
    return errors / len(rows)


def _select_intervention(examples: Sequence[Example], representation: Sequence[str]) -> Mapping[str, object]:
    indexes = tuple(FEATURES.index(feature) for feature in representation)
    model = _fit(examples, indexes)
    candidates = []
    for feature, cost in (("eig_observe", 1.0), ("cost_observe", 1.0), ("cost_search", 1.0), ("tail_mass", 1.5)):
        intervention = _candidate_intervention(feature)
        discrimination = _error(intervention, model, indexes)
        candidates.append(InterventionCandidate(feature, discrimination, cost, discrimination / cost))
    selected = max(candidates, key=lambda candidate: (candidate.value_per_cost, candidate.feature))
    return {
        "current_representation": tuple(representation),
        "selected": asdict(selected),
        "candidates": [asdict(candidate) for candidate in candidates],
    }


def run_active_intervention_loop() -> Mapping[str, object]:
    examples = _examples()
    initial = run(train_fraction=0.7)
    representation = tuple(initial["best"]["features"])
    rounds = []
    for _ in range(2):
        decision = _select_intervention(examples, representation)
        feature = decision["selected"]["feature"]
        examples.extend(_candidate_intervention(feature))
        recompressed = run_invariant_mdl_for_examples(examples)
        representation = tuple(recompressed["best"]["features"])
        rounds.append({"intervention": decision, "recompressed": recompressed})
    return {"initial_representation": tuple(initial["best"]["features"]), "rounds": rounds}


def run_intervention() -> Mapping[str, object]:
    examples = _intervention_examples()
    margin_index = (FEATURES.index("margin"),)
    full_indexes = tuple(range(len(FEATURES)))
    margin_model = _fit(examples, margin_index)
    full_model = _fit(examples, full_indexes)
    result = InterventionResult(
        len(examples),
        _error(examples, margin_model, margin_index),
        _error(examples, full_model, full_indexes),
        tuple(sorted({example.action for example in examples})),
    )
    return asdict(result)


def run_cost_intervention() -> Mapping[str, object]:
    examples = _cost_intervention_examples()
    eig_index = (FEATURES.index("eig_observe"),)
    cost_index = (FEATURES.index("cost_observe"),)
    full_indexes = tuple(range(len(FEATURES)))
    return {
        "examples": len(examples),
        "eig_only_error": _error(examples, _fit(examples, eig_index), eig_index),
        "cost_only_error": _error(examples, _fit(examples, cost_index), cost_index),
        "full_representation_error": _error(examples, _fit(examples, full_indexes), full_indexes),
        "distinct_actions": tuple(sorted({example.action for example in examples})),
    }


def _domain_example(domain: str, seed: int) -> Example:
    if domain == "physics":
        _, corrected, _ = _distributions(seed % 21, seed)
        probabilities = [corrected.top_probability, max(0.0, 1.0 - corrected.top_probability - corrected.other_mass)]
        eig_observe = corrected.other_mass
    elif domain == "partial":
        rows = _semantic_rows(_workspace().attention_distribution(("door", "go"), focus=0.8))
        probabilities = [probability for _, probability in rows]
        eig_observe = 1.0
    else:
        raise ValueError("unknown domain")
    state = _state(probabilities, eig_observe=eig_observe)
    return Example(
        (state.entropy, state.margin, eig_observe, 1.0 / sum(value * value for value in probabilities),
         state.disagreement, state.prediction_error, state.eig_research, state.eig_observe,
         state.cost_observe, state.cost_search),
        _choose(state),
    )


def run_domain_holdout() -> Mapping[str, object]:
    train = [_domain_example("physics", seed) for seed in range(48)]
    test = [_domain_example("partial", seed) for seed in range(48)]
    margin_indexes = (FEATURES.index("margin"),)
    full_indexes = tuple(range(len(FEATURES)))
    margin_model = _fit(train, margin_indexes)
    full_model = _fit(train, full_indexes)
    return {
        "train_domain": "physics",
        "test_domain": "partial",
        "margin_only_error": _error(test, margin_model, margin_indexes),
        "full_representation_error": _error(test, full_model, full_indexes),
        "test_actions": dict(Counter(example.action for example in test)),
    }


def _continuous_action(example: Example) -> str:
    accuracy = 1.0 - example.values[5]
    values = {
        "commit": accuracy,
        "re-search": min(1.0, accuracy + example.values[6]) - example.values[9],
        "observe": min(1.0, accuracy + example.values[7]) - example.values[8],
    }
    return max(values, key=values.get)


def run_continuous_domain_holdout() -> Mapping[str, object]:
    test = [_domain_example("partial", seed) for seed in range(48)]
    return {
        "test_domain": "partial",
        "continuous_error": sum(_continuous_action(example) != example.action for example in test) / len(test),
        "lookup_error": run_domain_holdout()["full_representation_error"],
    }


def run_operator_search() -> Mapping[str, object]:
    examples = _intervention_examples()
    operators = {
        "margin": lambda example: example.values[1],
        "eig_observe": lambda example: example.values[7],
        "margin_over_tail": lambda example: example.values[1] / (example.values[2] + 1e-6),
        "entropy": lambda example: example.values[0],
    }
    scores = {}
    for name, operator in operators.items():
        transformed = [Example((operator(example),), example.action) for example in examples]
        model = _fit(transformed, (0,))
        scores[name] = _error(transformed, model, (0,))
    best = min(scores, key=scores.get)
    return {"best_operator": best, "errors": scores}


def run_invariant_mdl_for_examples(examples: Sequence[Example]) -> Mapping[str, object]:
    examples = list(examples)
    random.Random(42).shuffle(examples)
    split = int(len(examples) * 0.7)
    train, test = examples[:split], examples[split:]
    results = []
    for size in range(1, len(FEATURES) + 1):
        for indexes in combinations(range(len(FEATURES)), size):
            model = _fit(train, indexes)
            results.append(RepresentationResult(
                tuple(FEATURES[index] for index in indexes),
                _error(train, model, indexes), _error(test, model, indexes), float(size),
                float(size) + 32.0 * _error(train, model, indexes),
            ))
    best = min(results, key=lambda result: (result.mdl, result.test_error, result.representation_cost))
    return {"examples": len(examples), "best": asdict(best), "action_set": sorted({example.action for example in examples})}


def run_invariant_mdl() -> Mapping[str, object]:
    return run_invariant_mdl_for_examples(_examples() + _intervention_examples() + _cost_intervention_examples())


def run(train_fraction: float = 0.7, error_weight: float = 8.0) -> Mapping[str, object]:
    examples = _examples()
    random.Random(42).shuffle(examples)
    split = int(len(examples) * train_fraction)
    train, test = examples[:split], examples[split:]
    candidates: list[RepresentationResult] = []
    for size in range(1, len(FEATURES) + 1):
        for indexes in combinations(range(len(FEATURES)), size):
            model = _fit(train, indexes)
            train_error = _error(train, model, indexes)
            test_error = _error(test, model, indexes)
            cost = float(size)
            candidates.append(RepresentationResult(
                tuple(FEATURES[index] for index in indexes), train_error, test_error, cost,
                cost + error_weight * train_error,
            ))
    best = min(candidates, key=lambda result: (result.mdl, result.test_error, result.representation_cost))
    full = next(result for result in candidates if len(result.features) == len(FEATURES))
    return {
        "examples": len(examples),
        "compression_ratio": len(best.features) / len(FEATURES),
        "best": asdict(best),
        "full": asdict(full),
        "all_candidates": [asdict(result) for result in candidates],
    }


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2, sort_keys=True))
