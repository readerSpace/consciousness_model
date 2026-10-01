"""exp605 -- Scaling / Plateau Audit for the bounded abstraction ecology.

The experiment deliberately keeps a finite typed operation catalog.  It asks
whether more time or capacity produces continuing *functional semantic novelty*,
or whether apparent activity is explained by a capacity boundary, exhausting the
catalog, or reinvention cycles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, Iterable, List, Tuple

import cumulative_instruction_bootstrapping as B
import numpy as np


CONDITIONS = ("adaptive_language", "fixed_language", "no_retention", "keep_all")


@dataclass(frozen=True)
class AuditConfig:
    steps: int = 30
    capacity: int = 64
    max_arity: int = 8
    max_birth_workload: int = 4096

    def __post_init__(self) -> None:
        if self.steps < 2 or self.capacity <= 0 or self.max_arity < 2:
            raise ValueError("steps >= 2, capacity > 0, and max_arity >= 2 are required")
        if self.max_arity & (self.max_arity - 1):
            raise ValueError("max_arity must be a power of two")

    @property
    def arities(self) -> Tuple[int, ...]:
        return tuple(2 ** exponent for exponent in range(1, self.max_arity.bit_length()))


@dataclass
class Life:
    arity: int
    births: List[int] = field(default_factory=list)
    prunes: List[int] = field(default_factory=list)
    uses: int = 0
    reuses: int = 0


@dataclass
class State:
    condition: str
    retained: Tuple[int, ...] = ()
    ever_invented: set[int] = field(default_factory=set)
    semantic_history: set[int] = field(default_factory=set)
    lives: Dict[int, Life] = field(default_factory=dict)
    records: List[dict] = field(default_factory=list)


class Ecology:
    """A finite catalog ecology whose reachability is gated by retained basis."""

    def __init__(self, config: AuditConfig):
        self.config = config
        self.operations = {arity: B.Operation(f"O{arity}", arity) for arity in config.arities}
        self._language_cache: Dict[Tuple[int, ...], int] = {}
        self._semantic_cache: Dict[Tuple[int, Tuple[int, ...]], int] = {}
        self._semantic_tables: Dict[Tuple[int, ...], List[int]] = {}

    def history(self, retained: Iterable[int]) -> B.History:
        arities = tuple(sorted(retained))
        return B.History("basis_" + "_".join(map(str, arities)) if arities else "P0",
                         tuple(self.operations[arity] for arity in arities), 0)

    def language_bits(self, retained: Iterable[int]) -> int:
        arities = tuple(sorted(retained))
        if arities in self._language_cache:
            return self._language_cache[arities]
        prior: Tuple[int, ...] = ()
        total = 0
        for arity in arities:
            total += self.semantic_cost(arity, prior)
            prior += (arity,)
        self._language_cache[arities] = total
        return total

    def semantic_cost(self, arity: int, retained: Iterable[int]) -> int:
        arities = tuple(sorted(retained))
        key = (arity, arities)
        if key not in self._semantic_cache:
            if arities not in self._semantic_tables:
                self._semantic_tables[arities] = self._build_semantic_table(arities)
            self._semantic_cache[key] = self._semantic_tables[arities][arity]
        return self._semantic_cache[key]

    def _build_semantic_table(self, arities: Tuple[int, ...]) -> List[int]:
        """Exact scalar dynamic program for every target arity under one basis."""
        maximum = self.config.max_arity
        costs = [0, B.REF_BITS] + [10 ** 9] * (maximum - 1)
        operation_costs = [(2, B.RAW_JOIN_BITS)] + [(item, B.DERIVED_CALL_BITS)
                                                     for item in arities]
        for width in range(2, maximum + 1):
            best = 10 ** 9
            for operation_arity, operation_bits in operation_costs:
                if operation_arity > width:
                    continue
                partitions = np.full(width + 1, 10 ** 9, dtype=np.int64)
                partitions[0] = 0
                for _ in range(operation_arity):
                    next_partitions = np.full(width + 1, 10 ** 9, dtype=np.int64)
                    for total in range(1, width + 1):
                        next_partitions[total] = np.min(
                            partitions[:total] + np.asarray(costs[total:0:-1], dtype=np.int64))
                    partitions = next_partitions
                best = min(best, operation_bits + int(partitions[width]))
            costs[width] = best
        return costs

    @staticmethod
    def semantic_distance(first_arity: int, second_arity: int) -> float:
        """Typed functional distance: concatenation functions differ by required arity."""
        return 0.0 if first_arity == second_arity else 1.0

    def _subsets(self, arities: Iterable[int]) -> Iterable[Tuple[int, ...]]:
        ordered = tuple(sorted(arities))
        for count in range(len(ordered) + 1):
            yield from combinations(ordered, count)

    def best_retained(self, available: Iterable[int], arity: int, workload: int) -> Tuple[int, ...]:
        return min(self._subsets(available),
                   key=lambda subset: self.language_bits(subset) + workload * self.semantic_cost(arity, subset))

    def birth_workload(self, retained: Tuple[int, ...], arity: int) -> int | None:
        before_language = self.language_bits(retained)
        before_task = self.semantic_cost(arity, retained)
        after = tuple(sorted((*retained, arity)))
        after_language = self.language_bits(after)
        after_task = self.semantic_cost(arity, after)
        if after_language > self.config.capacity:
            return None
        saving_per_use = before_task - after_task
        if saving_per_use <= 0:
            return None
        workload = max(1, (after_language - before_language) // saving_per_use + 1)
        return workload if workload <= self.config.max_birth_workload else None

    def select_environment(self, state: State, rng: np.random.Generator) -> Tuple[int, int, bool]:
        candidates = []
        for index, arity in enumerate(self.config.arities):
            prerequisite = self.config.arities[index - 1] if index else None
            if prerequisite is not None and prerequisite not in state.retained:
                continue
            if state.condition != "fixed_language" and arity not in state.retained:
                workload = self.birth_workload(state.retained, arity)
                if workload is not None:
                    candidates.append((100_000 + arity, arity, workload, True))
            current = self.semantic_cost(arity, state.retained)
            if current <= self.config.capacity:
                workload = int(rng.choice((1, 3, 5, 8)))
                candidates.append((current - 0.35 * workload, arity, workload, False))
        if not candidates:
            return self.config.arities[0], 1, False
        _, arity, workload, offers_birth = max(candidates, key=lambda row: row[0])
        return arity, workload, offers_birth

    def transition(self, state: State, arity: int, workload: int, offers_birth: bool, step: int) -> Tuple[List[int], List[int]]:
        before = set(state.retained)
        born: List[int] = []
        if offers_birth and state.condition != "fixed_language":
            born = [arity]
            state.ever_invented.add(arity)
            state.lives[arity].births.append(step)
            if state.condition != "no_retention":
                state.retained = tuple(sorted((*state.retained, arity)))
        if state.condition == "adaptive_language":
            state.retained = self.best_retained(state.ever_invented, arity, workload)
        elif state.condition == "keep_all":
            state.retained = tuple(sorted(state.ever_invented))
        elif state.condition in ("fixed_language", "no_retention"):
            state.retained = ()
        pruned = sorted(before - set(state.retained))
        for operation in pruned:
            state.lives[operation].prunes.append(step)
        return born, pruned

    def run(self, condition: str, seed: int = 1) -> Dict[str, object]:
        if condition not in CONDITIONS:
            raise ValueError(condition)
        rng = np.random.default_rng(seed)
        state = State(condition, lives={arity: Life(arity) for arity in self.config.arities})
        for step in range(self.config.steps):
            arity, workload, offers_birth = self.select_environment(state, rng)
            before = state.retained
            base_cost = self.semantic_cost(arity, ())
            prior_cost = self.semantic_cost(arity, before)
            already_known = arity in state.semantic_history
            born, pruned = self.transition(state, arity, workload, offers_birth, step)
            active = state.retained if condition != "no_retention" else before
            if condition == "no_retention" and born:
                active = tuple(sorted((*before, arity)))
            task_cost = self.semantic_cost(arity, active)
            novelty_distance = min((self.semantic_distance(arity, old) for old in state.semantic_history), default=1.0)
            functional_novelty = bool(born and not already_known and novelty_distance >= 0.5
                                      and task_cost < prior_cost)
            if functional_novelty:
                state.semantic_history.add(arity)
            for operation in active:
                life = state.lives[operation]
                life.reuses += int(life.uses > 0)
                life.uses += 1
            record = {
                "t": step, "arity": arity, "workload": workload, "offers_birth": offers_birth,
                "born": born, "pruned": pruned, "basis_before": list(before),
                "basis_after": list(state.retained), "active_basis": list(active),
                "D_language": self.language_bits(state.retained), "K_P0": base_cost,
                "K_current": task_cost, "A_advantage": base_cost - task_cost,
                "description_total": self.language_bits(state.retained) + workload * task_cost,
                "semantic_distance": novelty_distance,
                "functional_semantic_novelty": functional_novelty,
            }
            state.records.append(record)
        return self._result(state)

    def _result(self, state: State) -> Dict[str, object]:
        split = len(state.records) // 2
        late = state.records[split:]
        late_advantages = [record["A_advantage"] for record in late]
        slope = float(np.polyfit(np.arange(len(late_advantages)), late_advantages, 1)[0])
        life = {f"O{arity}": {"birth": item.births, "prune": item.prunes,
                                "use": item.uses, "reuse": item.reuses}
                for arity, item in state.lives.items()}
        total_births = sum(len(item.births) for item in state.lives.values())
        novelty = sum(record["functional_semantic_novelty"] for record in state.records)
        summary = {
            "n_birth": total_births,
            "n_reinvent": total_births - novelty,
            "n_pruned": sum(len(item.prunes) for item in state.lives.values()),
            "n_semantic_novelty": novelty,
            "late_births": sum(len(record["born"]) for record in late),
            "late_semantic_novelty": sum(record["functional_semantic_novelty"] for record in late),
            "remaining_catalog": len(set(self.config.arities) - state.semantic_history),
            "catalog_exhausted": len(state.semantic_history) == len(self.config.arities),
            "mean_A_late": float(np.mean(late_advantages)),
            "late_A_slope": slope,
            "max_D_language": max(record["D_language"] for record in state.records),
            "final_D_language": self.language_bits(state.retained),
            "capacity_fraction": max(record["D_language"] for record in state.records) / self.config.capacity,
            "mean_description": float(np.mean([record["description_total"] for record in state.records])),
            "max_arity": max(record["arity"] for record in state.records),
        }
        return {"config": self.config, "condition": state.condition, "records": state.records,
                "life_history": life, "summary": summary}


def run_audit(condition: str, config: AuditConfig = AuditConfig(), seed: int = 1) -> Dict[str, object]:
    return Ecology(config).run(condition, seed)
