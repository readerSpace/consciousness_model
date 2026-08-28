"""Frozen-B4 generalization and belief-revision quality across two domains.

No B4 policy parameter is tuned here.  The same AdaptivePolicy is applied to
a maze-like navigation graph and a simple physical-transition graph.  Feedback
is 95% reliable, permitting measurement of false revocations as well as
recovery from genuinely wrong initial broadcasts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import random
from typing import Dict, Mapping, Sequence

from adaptive_broadcast_revision_experiment import AdaptivePolicy, _resolve
from Consciousness_model import CompressedWorkspace, Relation


@dataclass(frozen=True)
class DomainResult:
    domain: str
    condition: str
    module_accuracy: float
    global_success: float
    workspace_searches: float
    false_revoke_rate: float
    recovery_rate: float
    mean_revision_depth: float


def _build_world(domain: str, seed: int, noise_per_goal: int = 12) -> tuple[CompressedWorkspace, int, str, str, str, str]:
    rng = random.Random(seed)
    target = rng.randrange(6)
    if domain == "maze":
        start = lambda number: f"maze:{number}:entrance"
        middle = lambda number: f"maze:{number}:junction"
        goal = lambda number: f"maze:{number}:exit"
        verb = "moves-to"
    elif domain == "physics":
        start = lambda number: f"ball:{number}:ramp"
        middle = lambda number: f"ball:{number}:bottom"
        goal = lambda number: f"bell:{number}:ring"
        verb = "evolves-to"
    else:
        raise ValueError("domain must be maze or physics")
    observations: list[Relation] = []
    order = list(range(6))
    rng.shuffle(order)
    for number in order:
        observations.extend(((start(number), verb, middle(number)), (middle(number), verb, goal(number))))
    noise = [(f"{domain}:noise:{number}:{copy}", verb, goal(number)) for number in range(6) for copy in range(noise_per_goal)]
    rng.shuffle(noise)
    workspace = CompressedWorkspace(capacity=16)
    workspace.ingest(observations + noise)
    expected = f"compose:{start(target)}:{verb}:{goal(target)}"
    return workspace, target, expected, start(target), goal(target), verb


def _queries(start: str, goal: str, verb: str) -> tuple[tuple[str, ...], ...]:
    return ((start, goal), (start, goal, verb), (start, goal, "remember"), (start, goal, "report"))


def _independent(workspace: CompressedWorkspace, queries: Sequence[Sequence[str]], expected: str, rng: random.Random) -> tuple[float, float, int]:
    answers, searches = [], 0
    for terms in queries:
        answer, _, _, _, _ = _resolve(workspace, terms, rng, set())
        answers.append(answer)
        searches += 1
    return sum(answer == expected for answer in answers) / len(answers), float(all(answer == expected for answer in answers)), searches


def _b3_trial(domain: str, seed: int) -> tuple[float, float, float, float, float, float, float]:
    workspace, _, expected, start, goal, verb = _build_world(domain, seed)
    answer, _, _, _, _ = _resolve(workspace, _queries(start, goal, verb)[0], random.Random(seed + 90_000), set())
    correct = float(answer == expected)
    return correct, correct, 1.0, 0.0, 0.0, 0.0, float(not correct)


def _b4_trial(domain: str, seed: int, policy: AdaptivePolicy, feedback_error_rate: float) -> tuple[float, float, float, float, float, float, float]:
    workspace, target, expected, start, goal, verb = _build_world(domain, seed)
    rng = random.Random(seed + 90_000)
    query_set = _queries(start, goal, verb)
    answer, confidence, entropy, margin, _ = _resolve(workspace, query_set[0], rng, set())
    novelty = (domain, target) not in getattr(policy, "seen_domain_targets", set())
    if not hasattr(policy, "seen_domain_targets"):
        policy.seen_domain_targets = set()  # type: ignore[attr-defined]
    policy.seen_domain_targets.add((domain, target))  # type: ignore[attr-defined]
    key = policy.key(f"{domain}:noise", confidence, novelty)
    searches, depth, false_revoke, initially_wrong, recovered = 1, 0, False, answer != expected, False
    if policy.should_commit(key, confidence, entropy, margin):
        excluded: set[str] = set()
        # tentative -> feedback -> challenged/revoked -> re-search
        while answer is not None and depth < 3:
            actual_error = answer != expected
            reported_error = actual_error if rng.random() >= feedback_error_rate else not actual_error
            if not reported_error:
                break  # committed
            if not actual_error:
                false_revoke = True
            policy.update(key, not actual_error)
            excluded.add(answer)
            depth += 1
            answer, confidence, entropy, margin, _ = _resolve(workspace, query_set[0], rng, excluded)
            searches += 1
        recovered = initially_wrong and answer == expected
        policy.update(key, answer == expected)
        if answer == expected:
            accuracy, global_success = 1.0, 1.0
        else:
            accuracy, global_success, extra = _independent(workspace, query_set, expected, rng)
            searches += extra
    else:
        policy.update(key, answer == expected)
        accuracy, global_success, extra = _independent(workspace, query_set, expected, rng)
        searches += extra
    return accuracy, global_success, float(searches), float(false_revoke), float(recovered), float(depth) if initially_wrong else 0.0, float(initially_wrong)


def _aggregate(domain: str, condition: str, rows: Sequence[tuple[float, float, float, float, float, float, float]]) -> DomainResult:
    total = len(rows)
    initially_wrong_count = sum(row[6] for row in rows)
    return DomainResult(
        domain, condition,
        sum(row[0] for row in rows) / total,
        sum(row[1] for row in rows) / total,
        sum(row[2] for row in rows) / total,
        sum(row[3] for row in rows) / total,
        sum(row[4] for row in rows) / max(1.0, initially_wrong_count),
        sum(row[5] for row in rows) / max(1.0, initially_wrong_count),
    )


def run(trials: int = 96, feedback_error_rate: float = 0.05) -> Mapping[str, Mapping[str, Mapping[str, float | str]]]:
    if not 0.0 <= feedback_error_rate < 0.5:
        raise ValueError("feedback_error_rate must be in [0, 0.5)")
    output: Dict[str, Mapping[str, Mapping[str, float | str]]] = {}
    for domain in ("maze", "physics"):
        b3 = [_b3_trial(domain, seed) for seed in range(trials)]
        policy = AdaptivePolicy()  # frozen parameters; fresh learned history per domain
        b4 = [_b4_trial(domain, seed, policy, feedback_error_rate) for seed in range(trials)]
        output[domain] = {
            "B3": asdict(_aggregate(domain, "B3", b3)),
            "B4": asdict(_aggregate(domain, "B4", b4)),
        }
    return output


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2, sort_keys=True))
