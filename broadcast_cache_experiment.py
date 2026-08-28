"""B0--B3 broadcast-cache ablation under noisy workspace retrieval.

B0: independent search; B1: exact-match cache; B2: semantic cache but search
remains independent; B3: semantic cache is authoritative after a confidence
and entropy gate.  The experiment reports both efficiency and the cost of
propagating a wrong shared interpretation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import log2
import json
import random
from typing import Dict, Iterable, Mapping, Sequence, Tuple

from Consciousness_model import CompressedWorkspace, Relation


@dataclass
class CacheEntry:
    query: frozenset[str]
    concept_id: str
    confidence: float
    entropy: float
    ttl: int


@dataclass(frozen=True)
class CacheMetrics:
    condition: str
    module_accuracy: float
    global_success: float
    consistency: float
    workspace_searches: float
    candidate_evaluations: float
    cache_hits: float
    error_propagation: float


QUERIES = {
    "planner": ("start", "goal"),
    "predictor": ("start", "goal", "leads-to"),
    "memory": ("start", "goal", "remember"),
    "report": ("start", "goal", "report"),
}


def _make_workspace(seed: int, capacity: int = 16, noise_per_goal: int = 16) -> tuple[CompressedWorkspace, int, str]:
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
    return workspace, target, f"compose:start-{target}:leads-to:goal-{target}"


def _terms(template: Sequence[str], target: int) -> tuple[str, ...]:
    return tuple(f"{term}-{target}" if term in {"start", "goal"} else term for term in template)


def _resolve(workspace: CompressedWorkspace, terms: Sequence[str], rng: random.Random) -> tuple[str | None, float, float, int]:
    distribution = workspace.attention_distribution(terms, focus=0.8)
    if not distribution:
        return None, 0.0, 0.0, 0
    draw = rng.random()
    cumulative = 0.0
    concept_id, confidence = distribution[-1][0].identifier, distribution[-1][1]
    for concept, probability in distribution:
        cumulative += probability
        if draw <= cumulative:
            concept_id, confidence = concept.identifier, probability
            break
    entropy = -sum(probability * log2(probability) for _, probability in distribution if probability > 0)
    return concept_id, confidence, entropy, len(distribution)


def _similar(left: frozenset[str], right: frozenset[str]) -> float:
    return len(left & right) / len(left | right)


def _cache_hit(cache: Iterable[CacheEntry], terms: frozenset[str], mode: str) -> CacheEntry | None:
    for entry in cache:
        if entry.ttl <= 0:
            continue
        if mode == "exact" and entry.query == terms:
            return entry
        if mode == "semantic" and _similar(entry.query, terms) >= 0.60:
            return entry
    return None


def _trial(seed: int, condition: str, confidence_gate: float, entropy_gate: float) -> tuple[float, float, float, float, float, float, float]:
    workspace, target, expected = _make_workspace(seed)
    rng = random.Random(seed + 10_000)
    cache: list[CacheEntry] = []
    answers: list[str | None] = []
    searches = evaluated = hits = 0
    propagated_error = False
    mode = "exact" if condition == "B1" else "semantic"
    for _, template in QUERIES.items():
        terms = frozenset(_terms(template, target))
        entry = _cache_hit(cache, terms, mode) if condition != "B0" else None
        # Only B3 treats a semantic cache hit as the current shared resolution.
        if condition == "B3" and entry is not None:
            answer = entry.concept_id
            hits += 1
            if answer != expected:
                propagated_error = True
        else:
            if entry is not None:
                # B1/B2 can observe a matching cached resolution, but retain
                # independent search by design; count this potential reuse.
                hits += 1
            answer, confidence, entropy, evaluated_now = _resolve(workspace, tuple(terms), rng)
            searches += 1
            evaluated += evaluated_now
            # B1/B2 record possible shared resolutions, but do not prohibit a
            # later independent search.  B3 uses the record authoritatively.
            if condition != "B0" and confidence >= confidence_gate and entropy <= entropy_gate:
                cache.insert(0, CacheEntry(terms, answer or "", confidence, entropy, ttl=4))
        answers.append(answer)
        for entry in cache:
            entry.ttl -= 1
    accuracy = sum(answer == expected for answer in answers) / len(answers)
    global_success = float(all(answer == expected for answer in answers))
    consistency = float(len(set(answers)) == 1)
    return accuracy, global_success, consistency, float(searches), float(evaluated), float(hits), float(propagated_error)


def run(trials: int = 64, confidence_gate: float = 0.30, entropy_gate: float = 2.0) -> Mapping[str, Mapping[str, float | str]]:
    if not 0.0 <= confidence_gate <= 1.0 or entropy_gate <= 0.0:
        raise ValueError("invalid cache-gate parameters")
    results: Dict[str, Mapping[str, float | str]] = {}
    for condition in ("B0", "B1", "B2", "B3"):
        rows = [_trial(seed, condition, confidence_gate, entropy_gate) for seed in range(trials)]
        means = [sum(row[index] for row in rows) / trials for index in range(7)]
        results[condition] = asdict(CacheMetrics(condition, *means))
    return results


def unsafe_gate_stress_test(trials: int = 64) -> Mapping[str, Mapping[str, float | str]]:
    """A zero confidence gate quantifies the predicted error-propagation risk."""
    return run(trials=trials, confidence_gate=0.0, entropy_gate=10.0)


if __name__ == "__main__":
    print(json.dumps({
        "permissive_gate": run(confidence_gate=0.30),
        "conservative_gate": run(confidence_gate=0.50),
        "unsafe_gate": unsafe_gate_stress_test(),
    }, ensure_ascii=False, indent=2, sort_keys=True))
