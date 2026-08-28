"""Selection-bias test: Top-K workspace versus pre-selection uncertainty summary."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import exp, log2
import json
import random
from typing import Dict, Mapping, Sequence

from cross_domain_revision_experiment import _build_world


@dataclass(frozen=True)
class Summary:
    entropy: float
    top_probability: float
    margin: float
    other_mass: float
    effective_count: float
    map_hypothesis: str


def _key(concept) -> str:
    return f"{concept.relations[0][0]}->{concept.relations[-1][2]}"


def _normalize(weights: Mapping[str, float]) -> Dict[str, float]:
    total = sum(weights.values())
    return {key: value / total for key, value in weights.items()} if total else {}


def _summary(distribution: Mapping[str, float], other_mass: float = 0.0) -> Summary:
    rows = sorted(distribution.items(), key=lambda row: row[1], reverse=True)
    probabilities = [value for _, value in rows] + ([other_mass] if other_mass > 1e-12 else [])
    entropy = -sum(value * log2(value) for value in probabilities if value > 0)
    top = rows[0][1] if rows else 0.0
    margin = top - (rows[1][1] if len(rows) > 1 else 0.0)
    return Summary(entropy, top, margin, other_mass, 1.0 / sum(value * value for value in probabilities) if probabilities else 0.0, rows[0][0] if rows else "")


def _selected_ids(workspace, capacity: int, rng: random.Random | None = None) -> tuple[str, ...]:
    """Select a finite workspace, optionally adding small ranking noise."""
    if rng is None:
        workspace._select()
        return workspace.items
    pool = list(workspace.long_term.values())
    selected = []
    while pool and len(selected) < min(capacity, len(pool)):
        best = max(
            pool,
            key=lambda candidate: (
                candidate.utility() + rng.uniform(-0.02, 0.02),
                candidate.identifier,
            ),
        )
        selected.append(best)
        pool.remove(best)
    return tuple(concept.identifier for concept in selected)


def _distributions(
    noise: int,
    seed: int,
    capacity: int = 8,
    selection_seed: int | None = None,
) -> tuple[Summary, Summary, bool]:
    workspace, _, _, start, goal, _ = _build_world("physics", seed, noise_per_goal=noise)
    workspace.capacity = capacity
    selection_rng = random.Random(selection_seed) if selection_seed is not None else None
    selected_ids = _selected_ids(workspace, capacity, selection_rng)

    raw: Dict[str, float] = {}
    for concept in workspace.long_term.values():
        raw[_key(concept)] = raw.get(_key(concept), 0.0) + exp(concept.utility())
    full = _normalize(raw)
    selected_raw: Dict[str, float] = {}
    for identifier in selected_ids:
        concept = workspace.long_term[identifier]
        selected_raw[_key(concept)] = selected_raw.get(_key(concept), 0.0) + exp(concept.utility())
    naive = _summary(_normalize(selected_raw))
    selected_keys = set(selected_raw)
    visible = {key: probability for key, probability in full.items() if key in selected_keys}
    other = 1.0 - sum(visible.values())
    corrected = _summary(visible, other)
    return naive, corrected, naive.map_hypothesis == f"{start}->{goal}"


def _action(summary: Summary) -> str:
    # A compact confidence-aware controller.  OTHER blocks commitment even if
    # the visible top candidate looks confident after Top-K re-normalization.
    return "commit" if summary.top_probability >= 0.10 and summary.other_mass <= 0.20 else "re-search"


def _action_signature(summary: Summary) -> str:
    """Treat committing to a different hypothesis as a different action."""
    return f"{_action(summary)}:{summary.map_hypothesis}"


def _brier(rows: Sequence[tuple[Summary, bool]]) -> float:
    return sum((summary.top_probability - float(correct)) ** 2 for summary, correct in rows) / len(rows)


def _flip_rate(actions: Sequence[str]) -> float:
    return sum(left != right for left, right in zip(actions, actions[1:])) / max(1, len(actions) - 1)


def run(noise_values: Sequence[int] = tuple(range(21)), perturbations: int = 100) -> Mapping[str, object]:
    sweep = []
    all_naive: list[tuple[Summary, bool]] = []
    all_corrected: list[tuple[Summary, bool]] = []
    for noise in noise_values:
        rows = [_distributions(noise, seed) for seed in range(24)]
        naive_rows = [(naive, correct) for naive, _, correct in rows]
        corrected_rows = [(corrected, correct) for _, corrected, correct in rows]
        all_naive.extend(naive_rows)
        all_corrected.extend(corrected_rows)
        sweep.append({
            "noise": noise,
            "naive_entropy": sum(row[0].entropy for row in rows) / len(rows),
            "corrected_entropy": sum(row[1].entropy for row in rows) / len(rows),
            "semantic_entropy": sum(row[1].entropy for row in rows) / len(rows),
            "semantic_margin": sum(row[1].margin for row in rows) / len(rows),
            "tail_mass": sum(row[1].other_mass for row in rows) / len(rows),
            "effective_count": sum(row[1].effective_count for row in rows) / len(rows),
            "naive_commit_rate": sum(_action(row[0]) == "commit" for row in rows) / len(rows),
            "corrected_commit_rate": sum(_action(row[1]) == "commit" for row in rows) / len(rows),
        })
    perturb_rows = [_distributions(10, 0, selection_seed=seed) for seed in range(perturbations)]
    return {
        "brier_naive": _brier(all_naive),
        "brier_corrected": _brier(all_corrected),
        "flip_rate_naive": _flip_rate([_action_signature(row[0]) for row in perturb_rows]),
        "flip_rate_corrected": _flip_rate([_action_signature(row[1]) for row in perturb_rows]),
        "decision_flip_rate_naive": _flip_rate([_action(row[0]) for row in perturb_rows]),
        "decision_flip_rate_corrected": _flip_rate([_action(row[1]) for row in perturb_rows]),
        "sweep": sweep,
    }


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2, sort_keys=True))
