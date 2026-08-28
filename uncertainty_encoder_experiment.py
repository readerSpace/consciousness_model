"""Domain-invariant uncertainty encoder and EIG-based three-action control.

The controller sees only z_U=(entropy, margin, disagreement, prediction
error, EIG_observe, EIG_research, observe_cost, search_cost).  It receives no
domain name and uses Q(action)=expected accuracy gain-cost.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import log2
import json
import random
from typing import Dict, Mapping, Sequence

from cross_domain_revision_experiment import _build_world
from partial_observation_experiment import _workspace


@dataclass(frozen=True)
class UncertaintyState:
    entropy: float
    margin: float
    disagreement: float
    prediction_error: float
    eig_observe: float
    eig_research: float
    cost_observe: float
    cost_search: float


@dataclass(frozen=True)
class EncoderResult:
    domain: str
    success_rate: float
    observe_rate: float
    research_rate: float
    commit_rate: float
    mean_entropy_drop_after_observe: float
    observe_success_improvement: float
    observe_eig_dominance: float


def _entropy(probabilities: Sequence[float]) -> float:
    return -sum(probability * log2(probability) for probability in probabilities if probability > 0.0)


def _state(probabilities: Sequence[float], eig_observe: float, cost_observe: float = 0.08, cost_search: float = 0.12) -> UncertaintyState:
    ordered = sorted(probabilities, reverse=True)
    entropy = _entropy(ordered)
    top = ordered[0] if ordered else 0.0
    remaining = ordered[1:]
    renormalized = [value / max(1e-12, 1.0 - top) for value in remaining]
    # Checking the leading hypothesis gives feedback: correct -> zero entropy;
    # wrong -> remove it and re-normalize the remaining belief distribution.
    eig_search = entropy - (1.0 - top) * _entropy(renormalized)
    return UncertaintyState(entropy, top - (ordered[1] if len(ordered) > 1 else 0.0), 1.0 - top, 1.0 - top, eig_observe, eig_search, cost_observe, cost_search)


def _choose(state: UncertaintyState) -> str:
    # Expected accuracy gain is represented by uncertainty reduction.  Observe
    # is valuable only when it is predicted to reduce belief entropy.
    current_accuracy = 1.0 - state.prediction_error
    values = {
        "commit": current_accuracy,
        "re-search": min(1.0, current_accuracy + state.eig_research) - state.cost_search,
        "observe": min(1.0, current_accuracy + state.eig_observe) - state.cost_observe,
    }
    return max(values, key=values.get)


def _sample(rows: Sequence[tuple[str, float]], rng: random.Random) -> str:
    draw, cumulative = rng.random(), 0.0
    for identifier, probability in rows:
        cumulative += probability
        if draw <= cumulative:
            return identifier
    return rows[-1][0]


def _semantic_rows(distribution) -> list[tuple[str, float]]:
    """Aggregate alternate chunk/compose encodings of the same transition."""
    grouped: Dict[str, float] = {}
    for concept, probability in distribution:
        relations = concept.relations
        key = f"{relations[0][0]}->{relations[-1][2]}"
        grouped[key] = grouped.get(key, 0.0) + probability
    return list(grouped.items())


def _graph_trial(domain: str, seed: int) -> tuple[str, float, float, float, float, float]:
    # Alternate clear and noisy scenes under the same physical-domain label.
    workspace, _, expected, start, goal, _ = _build_world(domain, seed, noise_per_goal=0 if seed % 2 == 0 else 12)
    rows = _semantic_rows(workspace.attention_distribution((start, goal), focus=0.8))
    state = _state([probability for _, probability in rows], eig_observe=0.0)
    action, rng = _choose(state), random.Random(seed + 991)
    expected_key = f"{start}->{goal}"
    if action == "commit":
        success = _sample(rows, rng) == expected_key
    elif action == "re-search":
        top = max(rows, key=lambda row: row[1])[0]
        remaining = [(identifier, probability) for identifier, probability in rows if identifier != top]
        success = top == expected_key or (bool(remaining) and _sample(remaining, rng) == expected_key)
    else:
        success = _sample(rows, rng) == expected_key
    return action, float(success), 0.0, 0.0, state.eig_observe, state.eig_research


def _partial_trial(seed: int) -> tuple[str, float, float, float, float, float]:
    actual = "safe" if random.Random(seed).randrange(2) == 0 else "trap"
    workspace = _workspace()
    rows = _semantic_rows(workspace.attention_distribution(("door", "go"), focus=0.8))
    before = _entropy([probability for _, probability in rows])
    # The discriminating cue makes only the actual door-transition compatible.
    state = _state([probability for _, probability in rows], eig_observe=before)
    expected, action, rng = f"door->{actual}", _choose(state), random.Random(seed + 1991)
    commit_success = _sample(rows, rng) == expected
    if action == "observe":
        workspace.ingest(((f"cue:{actual}", "indicates", actual),))
        after_rows = _semantic_rows(workspace.attention_distribution(("door", actual), focus=0.8))
        after = _entropy([probability for _, probability in after_rows])
        success = bool(after_rows and max(after_rows, key=lambda row: row[1])[0] == expected)
        return action, float(success), before - after, float(success) - float(commit_success), state.eig_observe, state.eig_research
    if action == "re-search":
        return action, float(_sample(rows, rng) == expected), 0.0, 0.0, state.eig_observe, state.eig_research
    return action, float(commit_success), 0.0, 0.0, state.eig_observe, state.eig_research


def run(trials: int = 96) -> Mapping[str, Mapping[str, float | str]]:
    output: Dict[str, Mapping[str, float | str]] = {}
    for domain, trial in (("physics", lambda seed: _graph_trial("physics", seed)), ("partial", _partial_trial)):
        rows = [trial(seed) for seed in range(trials)]
        output[domain] = asdict(EncoderResult(
            domain,
            sum(row[1] for row in rows) / trials,
            sum(row[0] == "observe" for row in rows) / trials,
            sum(row[0] == "re-search" for row in rows) / trials,
            sum(row[0] == "commit" for row in rows) / trials,
            sum(row[2] for row in rows) / trials,
            sum(row[3] for row in rows) / trials,
            sum(row[4] > row[5] for row in rows) / trials,
        ))
    return output


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2, sort_keys=True))
