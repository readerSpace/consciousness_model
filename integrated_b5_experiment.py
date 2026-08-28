"""Integrated B5 evaluation using actual workspace and text-belief features.

Unlike the earlier meta-control sandbox, action outcomes here are computed by
the existing compressed workspace's retrieval distribution, the partial-
observation cue, and the controlled text belief state.  B5 trains without a
domain label on maze/text and is tested on physics/partial observation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import log2
import json
import random
from typing import Callable, Dict, Mapping, Sequence

from adaptive_broadcast_revision_experiment import AdaptivePolicy
from cross_domain_revision_experiment import _build_world
from domain_invariant_meta_control_experiment import B5Controller, Context
from partial_observation_experiment import _workspace
from text_reasoning_experiment import TextBeliefState


@dataclass(frozen=True)
class IntegratedResult:
    controller: str
    success_rate: float
    commit_rate: float
    research_rate: float
    observe_rate: float


def _features(probabilities: Sequence[float], information_gain: float) -> Context:
    ordered = sorted(probabilities, reverse=True)
    confidence = ordered[0] if ordered else 0.0
    entropy = -sum(probability * log2(probability) for probability in ordered if probability > 0.0)
    margin = confidence - ordered[1] if len(ordered) > 1 else confidence
    return Context("unused", confidence, entropy, margin, 1.0 - confidence, information_gain, 1.0 - confidence, {})


class ActualTask:
    def __init__(self, features: Context, commit: Callable[[], bool], research: Callable[[], bool], observe: Callable[[], bool]) -> None:
        self.features, self._commit, self._research, self._observe = features, commit, research, observe

    def act(self, action: str) -> bool:
        return {"commit": self._commit, "re-search": self._research, "observe": self._observe}[action]()


def _sample(rows: Sequence[tuple[str, float]], rng: random.Random, excluded: set[str] | None = None) -> str | None:
    allowed = [(identifier, probability) for identifier, probability in rows if identifier not in (excluded or set())]
    total = sum(probability for _, probability in allowed)
    if total <= 0:
        return None
    draw, cumulative = rng.random(), 0.0
    for identifier, probability in allowed:
        cumulative += probability / total
        if draw <= cumulative:
            return identifier
    return allowed[-1][0]


def _graph_task(domain: str, seed: int) -> ActualTask:
    workspace, _, expected, start, goal, _ = _build_world(domain, seed, noise_per_goal=12)
    rows = [(concept.identifier, probability) for concept, probability in workspace.attention_distribution((start, goal), focus=0.8)]
    features = _features([probability for _, probability in rows], information_gain=0.03)
    rng = random.Random(seed + 44_000)
    def commit() -> bool:
        return _sample(rows, rng) == expected
    def research() -> bool:
        excluded: set[str] = set()
        for _ in range(3):
            answer = _sample(rows, rng, excluded)
            if answer == expected:
                return True
            if answer is not None:
                excluded.add(answer)
        return False
    return ActualTask(features, commit, research, commit)


def _partial_task(seed: int) -> ActualTask:
    actual = "safe" if random.Random(seed).randrange(2) == 0 else "trap"
    workspace, rng = _workspace(), random.Random(seed + 55_000)
    rows = [(concept.identifier, probability) for concept, probability in workspace.attention_distribution(("door", "go"), focus=0.8)]
    expected = f"episode:door:go:{actual}"
    features = _features([probability for _, probability in rows], information_gain=0.48)
    def commit() -> bool:
        return _sample(rows, rng) == expected
    def research() -> bool:
        return _sample(rows, rng) == expected
    def observe() -> bool:
        workspace.ingest(((f"cue:{actual}", "indicates", actual),))
        resolved = workspace.attention_distribution(("door", actual), focus=0.8)
        return bool(resolved and resolved[0][0].identifier == expected)
    return ActualTask(features, commit, research, observe)


def _text_task(seed: int) -> ActualTask:
    contradiction = seed % 2 == 1
    state = TextBeliefState()
    for sentence in ("アオイはキツネである。", "すべてのキツネは速い。", "すべての速いものは警戒している。"):
        state.add(sentence)
    expected = "不明" if contradiction else "真"
    current = state.answer("アオイ", "警戒している")
    features = Context("unused", 0.729, 0.30, 0.43, 0.271, 0.62 if contradiction else 0.02, 0.25, {})
    def commit() -> bool:
        return current == expected
    def research() -> bool:
        return current == expected
    def observe() -> bool:
        if contradiction:
            state.add("アオイは警戒していない。")
        return state.answer("アオイ", "警戒している") == expected
    return ActualTask(features, commit, research, observe)


def _task(domain: str, seed: int) -> ActualTask:
    if domain in {"maze", "physics"}:
        return _graph_task(domain, seed)
    if domain == "partial":
        return _partial_task(seed)
    if domain == "text":
        return _text_task(seed)
    raise ValueError("unknown domain")


def _run_b5(train_domains: Sequence[str], test_domains: Sequence[str], train_trials: int, test_trials: int) -> IntegratedResult:
    controller, rng = B5Controller(), random.Random(91_100)
    for step in range(train_trials):
        task = _task(train_domains[step % len(train_domains)], step // len(train_domains))
        action = controller.choose(task.features, rng, explore=max(0.05, 0.25 * (1.0 - step / train_trials)))
        controller.update(task.features, action, task.act(action))
    rows = []
    for step in range(test_trials):
        task = _task(test_domains[step % len(test_domains)], step // len(test_domains) + 100_000)
        action = controller.choose(task.features, rng, explore=0.0)
        rows.append((action, float(task.act(action))))
    return IntegratedResult("B5", sum(value for _, value in rows) / len(rows), *(sum(action == wanted for action, _ in rows) / len(rows) for wanted in ("commit", "re-search", "observe")))


def _run_b4(test_domains: Sequence[str], test_trials: int) -> IntegratedResult:
    policy = AdaptivePolicy()
    rows = []
    for step in range(test_trials):
        task = _task(test_domains[step % len(test_domains)], step // len(test_domains) + 100_000)
        key = policy.key("domain-free", task.features.confidence, novel=step < 2)
        action = "commit" if policy.should_commit(key, task.features.confidence, task.features.entropy, task.features.margin) else "re-search"
        result = task.act(action)
        policy.update(key, result)
        rows.append((action, float(result)))
    return IntegratedResult("B4", sum(value for _, value in rows) / len(rows), *(sum(action == wanted for action, _ in rows) / len(rows) for wanted in ("commit", "re-search", "observe")))


def run(train_trials: int = 360, test_trials: int = 180) -> Mapping[str, Mapping[str, float | str]]:
    # Train on actual maze/text states; test on unseen physical/partial states.
    results = (_run_b4(("physics", "partial"), test_trials), _run_b5(("maze", "text"), ("physics", "partial"), train_trials, test_trials))
    return {result.controller: asdict(result) for result in results}


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2, sort_keys=True))
