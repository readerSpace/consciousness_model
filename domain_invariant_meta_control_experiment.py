"""B5 domain-invariant meta-control: commit, re-search, or observe more.

The controller never receives a domain label.  It learns action values from
only confidence, entropy, margin, prediction-error risk, expected information
gain, and candidate disagreement.  Evaluation domains are held out by name.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import random
from typing import Dict, Mapping, Sequence

from adaptive_broadcast_revision_experiment import AdaptivePolicy


@dataclass(frozen=True)
class Context:
    domain: str
    confidence: float
    entropy: float
    margin: float
    prediction_error: float
    information_gain: float
    disagreement: float
    success: Mapping[str, float]


@dataclass(frozen=True)
class ControllerResult:
    controller: str
    success_rate: float
    commit_rate: float
    research_rate: float
    observe_rate: float
    commit_predicted_correctness: float
    commit_actual_correctness: float


ACTION_COST = {"commit": 0.0, "re-search": 0.12, "observe": 0.08}


def _context(domain: str, seed: int) -> Context:
    # Domain names select environment dynamics only; neither controller sees it.
    if domain in {"maze", "physics"}:
        clear = seed % 2 == 0
        if clear:
            return Context(domain, 0.90, 0.20, 0.55, 0.10, 0.03, 0.10, {"commit": 0.90, "re-search": 0.92, "observe": 0.90})
        return Context(domain, 0.45, 1.25, 0.06, 0.55, 0.08, 0.80, {"commit": 0.45, "re-search": 0.82, "observe": 0.46})
    if domain in {"text", "occluded-text"}:
        if domain == "text":
            return Context(domain, 0.88, 0.24, 0.52, 0.12, 0.04, 0.10, {"commit": 0.88, "re-search": 0.91, "observe": 0.88})
        return Context(domain, 0.50, 1.05, 0.03, 0.50, 0.46, 0.92, {"commit": 0.50, "re-search": 0.64, "observe": 0.98})
    if domain == "partial":
        return Context(domain, 0.50, 1.05, 0.03, 0.50, 0.46, 0.92, {"commit": 0.50, "re-search": 0.64, "observe": 0.98})
    raise ValueError("unknown domain")


class B5Controller:
    def __init__(self) -> None:
        self.values: Dict[tuple[str, str, str, str, str, str], Dict[str, list[float]]] = {}

    @staticmethod
    def key(context: Context) -> tuple[str, str, str, str, str, str]:
        return (
            "high" if context.confidence >= 0.75 else "low",
            "high" if context.entropy >= 0.80 else "low",
            "high" if context.margin >= 0.25 else "low",
            "high" if context.prediction_error >= 0.35 else "low",
            "high" if context.information_gain >= 0.25 else "low",
            "high" if context.disagreement >= 0.50 else "low",
        )

    def choose(self, context: Context, rng: random.Random, explore: float) -> str:
        if rng.random() < explore:
            return rng.choice(("commit", "re-search", "observe"))
        rows = self.values.get(self.key(context), {})
        return max(("commit", "re-search", "observe"), key=lambda action: rows.get(action, [0.0, 0.0])[0] / max(1.0, rows.get(action, [0.0, 0.0])[1]))

    def update(self, context: Context, action: str, success: bool) -> None:
        values = self.values.setdefault(self.key(context), {})
        total, count = values.setdefault(action, [0.0, 0.0])
        values[action] = [total + float(success) - ACTION_COST[action], count + 1.0]

    def estimate_commit(self, context: Context) -> float:
        total, count = self.values.get(self.key(context), {}).get("commit", [context.confidence, 1.0])
        return total / max(1.0, count)


def _run_b5(train_domains: Sequence[str], test_domains: Sequence[str], train_trials: int, test_trials: int) -> ControllerResult:
    controller, rng = B5Controller(), random.Random(8801)
    for step in range(train_trials):
        context = _context(train_domains[step % len(train_domains)], step // len(train_domains))
        action = controller.choose(context, rng, explore=max(0.05, 0.25 * (1.0 - step / train_trials)))
        success = rng.random() < context.success[action]
        controller.update(context, action, success)
    rows = []
    for step in range(test_trials):
        context = _context(test_domains[step % len(test_domains)], step // len(test_domains) + 100_000)
        action = controller.choose(context, rng, explore=0.0)
        success = rng.random() < context.success[action]
        rows.append((context, action, float(success)))
    commits = [(context, success) for context, action, success in rows if action == "commit"]
    return ControllerResult(
        "B5", sum(success for _, _, success in rows) / len(rows),
        sum(action == "commit" for _, action, _ in rows) / len(rows),
        sum(action == "re-search" for _, action, _ in rows) / len(rows),
        sum(action == "observe" for _, action, _ in rows) / len(rows),
        sum(controller.estimate_commit(context) for context, _ in commits) / max(1, len(commits)),
        sum(success for _, success in commits) / max(1, len(commits)),
    )


def _run_b4(test_domains: Sequence[str], test_trials: int) -> ControllerResult:
    policy, rng = AdaptivePolicy(), random.Random(8801)
    rows = []
    for step in range(test_trials):
        context = _context(test_domains[step % len(test_domains)], step // len(test_domains) + 100_000)
        key = policy.key("domain-free", context.confidence, novel=step < 2)
        action = "commit" if policy.should_commit(key, context.confidence, context.entropy, context.margin) else "re-search"
        success = float(rng.random() < context.success[action])
        policy.update(key, bool(success))
        rows.append((context, action, success))
    commits = [(context, success) for context, action, success in rows if action == "commit"]
    return ControllerResult(
        "B4", sum(success for _, _, success in rows) / len(rows),
        sum(action == "commit" for _, action, _ in rows) / len(rows),
        sum(action == "re-search" for _, action, _ in rows) / len(rows), 0.0,
        sum(context.confidence for context, _ in commits) / max(1, len(commits)),
        sum(success for _, success in commits) / max(1, len(commits)),
    )


def run(train_trials: int = 480, test_trials: int = 240) -> Mapping[str, Mapping[str, float | str]]:
    # B5 learns on maze/text/occluded-text and is evaluated on unseen physical
    # transition and partial-observation task labels.
    b4 = _run_b4(("physics", "partial"), test_trials)
    b5 = _run_b5(("maze", "text", "occluded-text"), ("physics", "partial"), train_trials, test_trials)
    return {result.controller: asdict(result) for result in (b4, b5)}


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2, sort_keys=True))
