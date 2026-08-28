"""B4: adaptive broadcast commitment with feedback-driven belief revision.

The initial query is tentative.  Prediction feedback can challenge and revoke
the selected concept; a re-search then excludes falsified concepts.  An online
policy decides whether to commit based on confidence, entropy, margin, novelty,
and empirical success history for similar query contexts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import log2
import json
import random
from typing import Dict, Mapping, Sequence

from broadcast_cache_experiment import QUERIES, _make_workspace, _terms
from Consciousness_model import CompressedWorkspace


@dataclass(frozen=True)
class RevisionMetrics:
    condition: str
    module_accuracy: float
    global_success: float
    workspace_searches: float
    candidate_evaluations: float
    commitment_rate: float
    revocation_rate: float
    final_error_propagation: float


class AdaptivePolicy:
    """Beta-Bernoulli reliability estimates conditioned on query features."""

    def __init__(self) -> None:
        self.history: Dict[tuple[str, str, bool], list[float]] = {}
        self.seen_targets: set[int] = set()

    def key(self, noise: str, confidence: float, novel: bool) -> tuple[str, str, bool]:
        return noise, "high" if confidence >= 0.30 else "low", novel

    def expected_error(self, key: tuple[str, str, bool]) -> float:
        successes, failures = self.history.get(key, [2.0, 2.0])
        return failures / (successes + failures)

    def should_commit(self, key: tuple[str, str, bool], confidence: float, entropy: float, margin: float) -> bool:
        # Saved searches and consistency are valuable; likely shared error is
        # costly.  Low margins / high entropy discount the confidence signal.
        saved_search_value = 3.0
        consistency_value = 0.8
        uncertainty_discount = 0.10 * min(1.0, entropy / 2.0) + 0.05 * max(0.0, 0.20 - margin)
        # Blend empirical reliability with the current posterior instead of
        # treating entropy as an automatic veto.  This permits tentative
        # commitments early on and lets feedback learn when they are costly.
        error_risk = 0.55 * self.expected_error(key) + 0.35 * (1.0 - confidence) + uncertainty_discount
        return saved_search_value + consistency_value - 8.0 * error_risk > 0.0

    def update(self, key: tuple[str, str, bool], correct: bool) -> None:
        values = self.history.setdefault(key, [2.0, 2.0])
        values[0 if correct else 1] += 1.0


def _distribution(workspace: CompressedWorkspace, terms: Sequence[str], excluded: set[str]) -> list[tuple[str, float]]:
    rows = [(concept.identifier, probability) for concept, probability in workspace.attention_distribution(terms, focus=0.8) if concept.identifier not in excluded]
    total = sum(probability for _, probability in rows)
    return [(identifier, probability / total) for identifier, probability in rows] if total else []


def _resolve(workspace: CompressedWorkspace, terms: Sequence[str], rng: random.Random, excluded: set[str]) -> tuple[str | None, float, float, float, int]:
    rows = _distribution(workspace, terms, excluded)
    if not rows:
        return None, 0.0, 0.0, 0.0, 0
    draw, total = rng.random(), 0.0
    answer, confidence = rows[-1]
    for identifier, probability in rows:
        total += probability
        if draw <= total:
            answer, confidence = identifier, probability
            break
    entropy = -sum(probability * log2(probability) for _, probability in rows if probability > 0.0)
    ordered = sorted((probability for _, probability in rows), reverse=True)
    margin = ordered[0] - ordered[1] if len(ordered) > 1 else ordered[0]
    return answer, confidence, entropy, margin, len(rows)


def _independent_answers(workspace: CompressedWorkspace, target: int, rng: random.Random) -> tuple[list[str | None], int, int]:
    answers: list[str | None] = []
    searches = evaluated = 0
    for template in QUERIES.values():
        answer, _, _, _, count = _resolve(workspace, _terms(template, target), rng, set())
        answers.append(answer)
        searches += 1
        evaluated += count
    return answers, searches, evaluated


def _trial_b3(seed: int) -> tuple[float, float, float, float, float, float, float]:
    workspace, target, expected = _make_workspace(seed, noise_per_goal=16)
    answer, _, _, _, evaluated = _resolve(workspace, _terms(QUERIES["planner"], target), random.Random(seed + 5000), set())
    correct = float(answer == expected)
    return correct, correct, 1.0, float(evaluated), 1.0, 0.0, float(not correct)


def _trial_b4(seed: int, policy: AdaptivePolicy) -> tuple[float, float, float, float, float, float, float]:
    # Alternate familiar low-noise and novel/high-noise query contexts.
    high_noise = seed % 2 == 1
    noise_per_goal = 16 if high_noise else 4
    workspace, target, expected = _make_workspace(seed, noise_per_goal=noise_per_goal)
    rng = random.Random(seed + 5000)
    terms = _terms(QUERIES["planner"], target)
    answer, confidence, entropy, margin, evaluated = _resolve(workspace, terms, rng, set())
    novelty = target not in policy.seen_targets
    policy.seen_targets.add(target)
    key = policy.key("high" if high_noise else "low", confidence, novelty)
    searches, committed, revocations = 1, False, 0
    if policy.should_commit(key, confidence, entropy, margin):
        committed = True  # status: tentative
        excluded: set[str] = set()
        # Prediction feedback validates each tentative hypothesis.  A failed
        # hypothesis is revoked, excluded, and the workspace is searched again.
        while answer is not None and answer != expected and len(excluded) < 3:
            policy.update(key, False)
            excluded.add(answer)  # tentative -> challenged -> revoked
            revocations += 1
            answer, confidence, entropy, margin, count = _resolve(workspace, terms, rng, excluded)
            searches += 1
            evaluated += count
        correct = answer == expected
        policy.update(key, correct)
        if correct:
            # revised concept becomes committed and all modules use it.
            answers = [answer] * len(QUERIES)
        else:
            answers, extra_searches, extra_evaluated = _independent_answers(workspace, target, rng)
            searches += extra_searches
            evaluated += extra_evaluated
    else:
        # No temporary commitment: each module retains independent search.
        policy.update(key, answer == expected)
        answers, extra_searches, extra_evaluated = _independent_answers(workspace, target, rng)
        searches += extra_searches
        evaluated += extra_evaluated
    accuracy = sum(item == expected for item in answers) / len(answers)
    global_success = float(all(item == expected for item in answers))
    # Wrong concepts are never left shared after feedback/revision.
    final_error_propagation = float(len(set(answers)) == 1 and answers[0] != expected)
    return accuracy, global_success, float(searches), float(evaluated), float(committed), float(revocations > 0), final_error_propagation


def run(trials: int = 96) -> Mapping[str, Mapping[str, float | str]]:
    policy = AdaptivePolicy()
    result: Dict[str, Mapping[str, float | str]] = {}
    for condition in ("B3", "B4"):
        rows = [(_trial_b3(seed) if condition == "B3" else _trial_b4(seed, policy)) for seed in range(trials)]
        means = [sum(row[index] for row in rows) / trials for index in range(7)]
        result[condition] = asdict(RevisionMetrics(condition, *means))
    return result


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2, sort_keys=True))
