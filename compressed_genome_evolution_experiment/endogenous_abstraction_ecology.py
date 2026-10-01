"""exp604 -- Endogenous Abstraction Ecology.

Tasks, retained instruction bases, and descriptions co-evolve in a small typed
language.  The environment selects the hardest currently reachable candidate
whose new operation can repay its description cost.  Adaptive language then
chooses retain/delete subsets by MDL; fixed, no-retention, and keep-all isolate
the roles of invention, memory, and forgetting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, Iterable, List, Sequence, Tuple

import cumulative_instruction_bootstrapping as B
import numpy as np


ARITIES = (2, 4, 8)
CAPACITY = 64
OPERATION_BY_ARITY = {
    2: B.O1,
    4: B.O2,
    8: B.Operation("O3", 8),
}


@dataclass
class Life:
    arity: int
    births: List[int] = field(default_factory=list)
    prunes: List[int] = field(default_factory=list)
    uses: int = 0
    reuses: int = 0
    retained_steps: int = 0


@dataclass
class EcologyState:
    condition: str
    retained: Tuple[int, ...] = ()
    ever_invented: set[int] = field(default_factory=set)
    lives: Dict[int, Life] = field(default_factory=lambda: {arity: Life(arity) for arity in ARITIES})
    records: List[dict] = field(default_factory=list)


def _history(retained: Iterable[int]) -> B.History:
    arities = tuple(sorted(retained))
    operations = tuple(OPERATION_BY_ARITY[arity] for arity in arities)
    return B.History("basis_" + "_".join(map(str, arities)) if arities else "P0",
                     operations, language_bits(arities))


def language_bits(retained: Iterable[int]) -> int:
    """Dependency-aware maintenance: each definition is encoded from lower retained operations."""
    prior: Tuple[int, ...] = ()
    total = 0
    for arity in sorted(retained):
        cost, _ = B.semantic_complexity(arity, _history(prior))
        total += cost
        prior += (arity,)
    return total


def semantic_cost(arity: int, retained: Iterable[int]) -> int:
    return B.semantic_complexity(arity, _history(tuple(sorted(retained))))[0]


def _subsets(items: Iterable[int]) -> Iterable[Tuple[int, ...]]:
    ordered = tuple(sorted(items))
    for count in range(len(ordered) + 1):
        yield from combinations(ordered, count)


def _best_retained(available: Iterable[int], arity: int, workload: int) -> Tuple[int, ...]:
    return min(_subsets(available), key=lambda subset: language_bits(subset) + workload * semantic_cost(arity, subset))


def _birth_workload(retained: Tuple[int, ...], arity: int) -> int | None:
    """Smallest reuse level at which adding this operation lowers full MDL."""
    before_language = language_bits(retained)
    before_task = semantic_cost(arity, retained)
    after = tuple(sorted((*retained, arity)))
    after_language = language_bits(after)
    after_task = semantic_cost(arity, after)
    if after_language > CAPACITY:
        return None
    for workload in range(1, 13):
        if after_language + workload * after_task < before_language + workload * before_task:
            return workload
    return None


def _select_environment(state: EcologyState, rng: np.random.Generator) -> Tuple[int, int, bool]:
    """Select a ZPD task from the current basis, not a preset stage schedule."""
    candidates = []
    retained = state.retained
    for arity in ARITIES:
        # A larger operation is environmentally reachable only after the prior
        # retained abstraction exposes its construction path.  This is a basis
        # constraint, not a time-indexed curriculum.
        prerequisite = {2: None, 4: 2, 8: 4}[arity]
        if prerequisite is not None and prerequisite not in retained:
            continue
        if state.condition != "fixed_language" and arity not in retained:
            birth_workload = _birth_workload(retained, arity)
            if birth_workload is not None:
                # New reachable operation dominates: basis growth opens the next task.
                candidates.append((100 + arity, arity, birth_workload, True))
        current = semantic_cost(arity, retained)
        if current <= CAPACITY:
            # Mature ecology: small stochastic workloads expose retain/prune tradeoffs.
            workload = int(rng.choice((1, 3, 5, 8)))
            candidates.append((current - 0.35 * workload, arity, workload, False))
    if not candidates:
        return 2, 1, False
    _, arity, workload, offers_birth = max(candidates, key=lambda row: row[0])
    return arity, workload, offers_birth


def _transition(state: EcologyState, arity: int, workload: int, offers_birth: bool, step: int) -> Tuple[List[int], List[int]]:
    before = set(state.retained)
    born: List[int] = []
    if offers_birth and state.condition != "fixed_language":
        born = [arity]
        state.ever_invented.add(arity)
        state.lives[arity].births.append(step)
        if state.condition != "no_retention":
            state.retained = tuple(sorted((*state.retained, arity)))
    if state.condition == "adaptive_language":
        state.retained = _best_retained(state.ever_invented, arity, workload)
    elif state.condition == "keep_all":
        state.retained = tuple(sorted(state.ever_invented))
    elif state.condition in ("fixed_language", "no_retention"):
        state.retained = ()
    after = set(state.retained)
    pruned = sorted(before - after)
    for operation in pruned:
        state.lives[operation].prunes.append(step)
    return born, pruned


def run_ecology(condition: str, steps: int = 30, seed: int = 1) -> Dict[str, object]:
    if condition not in {"adaptive_language", "fixed_language", "no_retention", "keep_all"}:
        raise ValueError(condition)
    rng = np.random.default_rng(seed)
    state = EcologyState(condition)
    for step in range(steps):
        arity, workload, offers_birth = _select_environment(state, rng)
        before = state.retained
        base_cost = semantic_cost(arity, ())
        current_cost = semantic_cost(arity, before)
        born, pruned = _transition(state, arity, workload, offers_birth, step)
        active = state.retained if condition != "no_retention" else before
        # A no-retention invention is usable during this task but vanishes afterwards.
        if condition == "no_retention" and born:
            active = tuple(sorted((*before, arity)))
        task_cost = semantic_cost(arity, active)
        for operation in active:
            life = state.lives[operation]
            if life.uses:
                life.reuses += 1
            life.uses += 1
        for operation in state.retained:
            state.lives[operation].retained_steps += 1
        record = {
            "t": step, "arity": arity, "workload": workload, "offers_birth": offers_birth,
            "born": born, "pruned": pruned, "basis_before": list(before),
            "basis_after": list(state.retained), "active_basis": list(active),
            "D_language": language_bits(state.retained),
            "K_P0": base_cost, "K_current": task_cost,
            "A_advantage": base_cost - task_cost,
            "description_total": language_bits(state.retained) + workload * task_cost,
        }
        state.records.append(record)
    late = state.records[len(state.records) // 2:]
    early_operations = {arity for arity, item in state.lives.items()
                        if item.births and min(item.births) < len(state.records) // 2}
    life = {f"O{index + 1}": {"birth": item.births, "prune": item.prunes,
                                "use": item.uses, "reuse": item.reuses,
                                "retained_steps": item.retained_steps}
            for index, item in enumerate(state.lives.values())}
    summary = {
        "late_births": sum(len(record["born"]) for record in late),
        "late_reuse": sum(sum(operation in record["active_basis"] for operation in early_operations)
                  for record in late),
        "n_birth": sum(len(item.births) for item in state.lives.values()),
        "n_pruned": sum(len(item.prunes) for item in state.lives.values()),
        "n_retained_final": len(state.retained),
        "mean_A_late": float(np.mean([record["A_advantage"] for record in late])),
        "mean_description": float(np.mean([record["description_total"] for record in state.records])),
        "max_D_language": max(record["D_language"] for record in state.records),
        "final_D_language": language_bits(state.retained),
        "max_arity": max(record["arity"] for record in state.records),
    }
    return {"condition": condition, "records": state.records, "life_history": life, "summary": summary}


def run_conditions(steps: int = 30, seed: int = 1) -> Dict[str, Dict[str, object]]:
    return {condition: run_ecology(condition, steps, seed) for condition in
            ("adaptive_language", "fixed_language", "no_retention", "keep_all")}