"""exp607 -- Multidimensional Semantic Innovation.

Environment programs are recursively growing semantic DAGs.  They are not a
task catalog: every step introduces a new variable-depth program.  The agent
starts with only raw tree spelling and can invent generic operator abstractions
when their definition cost is amortized by future use.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Tuple

import numpy as np


CONDITIONS = ("adaptive_language", "fixed_language", "no_retention", "keep_all")
PARAMETRIC = "parametric_only"
COMPOSITIONAL = "compositional_generative"
OPERATOR_BITS = {"ITERATE": 20, "REVERSE": 14, "INTERLEAVE": 22, "MAP_FLIP": 14}
OPERATOR_ORDER = tuple(OPERATOR_BITS)


@dataclass(frozen=True)
class Config:
    steps: int = 300
    capacity: int = 96
    manageable_cost: int = 128
    reuse_horizon: int = 8

    def __post_init__(self) -> None:
        if self.steps < 2 or self.capacity <= 0 or self.manageable_cost <= 0:
            raise ValueError("steps >= 2, capacity > 0, manageable_cost > 0 are required")


@dataclass(frozen=True)
class Task:
    depth: int
    operators: Tuple[str, ...]

    def structural_signature(self) -> Tuple[str, ...]:
        return self.operators

    def probe_signature(self) -> Tuple[int, Tuple[str, ...]]:
        """Exact abstract interpretation of all registered nonempty probes.

        Repeat changes output length by $2^depth$; reverse/map/interleave are
        retained as an ordered transducer trace.  No exponential output is
        materialized, but equal summaries have equal outputs on every probe in
        this restricted grammar.
        """
        return self.depth, self.operators


def semantic_distance(first: Task, second: Task) -> float:
    return 0.0 if first.probe_signature() == second.probe_signature() else 1.0


def structural_distance(first: Task, second: Task) -> float:
    return 0.0 if first.structural_signature() == second.structural_signature() else 1.0


def _generator(mode: str, step: int, seed: int) -> Task:
    if mode == PARAMETRIC:
        return Task(step + 1, ("ITERATE",))
    if mode != COMPOSITIONAL:
        raise ValueError(mode)
    # The DAG keeps growing after all classes have appeared: it recursively
    # nests a distinct structural operator at every level.
    cycle = ("REVERSE", "INTERLEAVE", "MAP_FLIP")
    layers = ("ITERATE",) + tuple(cycle[index % len(cycle)] for index in range(step))
    return Task(step + 1, layers)


def raw_cost(task: Task) -> int:
    return 10 * task.depth + 9 * (len(task.operators) - 1) + 4


def encoded_cost(task: Task, retained: Iterable[str]) -> int:
    retained_set = set(retained)
    if "ITERATE" in retained_set:
        cost = 2 + (2 * (task.depth + 1).bit_length() - 1)
    else:
        cost = 10 * task.depth + 4
    for operator, multiplicity in Counter(task.operators).items():
        if operator == "ITERATE":
            continue
        cost += (2 + (2 * multiplicity.bit_length() - 1)
                 if operator in retained_set else 9 * multiplicity)
    return cost


def language_bits(retained: Iterable[str]) -> int:
    return sum(OPERATOR_BITS[operator] for operator in retained)


def operator_semantic_class(operator: str) -> Tuple[str, int]:
    """Probe semantics cluster operator names by their transducer behavior."""
    behaviors = {"ITERATE": ("length_scale", 2), "REVERSE": ("permutation", -1),
                 "INTERLEAVE": ("merge", 2), "MAP_FLIP": ("value_map", 1)}
    return behaviors[operator]


@dataclass
class State:
    condition: str
    retained: Tuple[str, ...] = ()
    ever_invented: set[str] = field(default_factory=set)
    semantic_history: List[Task] = field(default_factory=list)
    structural_history: List[Task] = field(default_factory=list)
    semantic_signatures: set[Tuple[int, Tuple[str, ...]]] = field(default_factory=set)
    structural_signatures: set[Tuple[str, ...]] = field(default_factory=set)
    records: List[dict] = field(default_factory=list)
    births: Dict[str, List[int]] = field(default_factory=lambda: {operator: [] for operator in OPERATOR_ORDER})
    uses: Dict[str, int] = field(default_factory=lambda: {operator: 0 for operator in OPERATOR_ORDER})
    reuses: Dict[str, int] = field(default_factory=lambda: {operator: 0 for operator in OPERATOR_ORDER})


def _needed_inventions(task: Task, state: State, config: Config) -> List[str]:
    needed = []
    before = encoded_cost(task, state.retained)
    for operator in dict.fromkeys(task.operators):
        if operator in state.ever_invented:
            continue
        after_retained = tuple((*state.retained, operator))
        after = OPERATOR_BITS[operator] + config.reuse_horizon * encoded_cost(task, after_retained)
        if after < config.reuse_horizon * before and language_bits(after_retained) <= config.capacity:
            needed.append(operator)
    return needed


def _transition(state: State, inventions: List[str], task: Task, config: Config, step: int) -> Tuple[str, ...]:
    if state.condition == "fixed_language":
        return ()
    for operator in inventions:
        state.ever_invented.add(operator)
        state.births[operator].append(step)
    if state.condition == "adaptive_language":
        # A definition is retained exactly while it shortens this reachable DAG.
        state.retained = tuple(operator for operator in state.ever_invented if operator in task.operators)
    elif state.condition == "keep_all":
        state.retained = tuple(operator for operator in OPERATOR_ORDER if operator in state.ever_invented)
    else:  # no-retention
        state.retained = ()
    return tuple((*state.retained, *inventions)) if state.condition == "no_retention" else state.retained


def run_innovation(mode: str, condition: str, config: Config = Config(), seed: int = 1) -> Dict[str, object]:
    if condition not in CONDITIONS:
        raise ValueError(condition)
    state = State(condition)
    for step in range(config.steps):
        proposal = _generator(mode, step, seed)
        inventions = _needed_inventions(proposal, state, config)
        active = _transition(state, inventions, proposal, config, step)
        cost = encoded_cost(proposal, active)
        reachable = cost <= config.manageable_cost
        task = proposal if reachable else state.semantic_history[-1]
        semantic_new = task.probe_signature() not in state.semantic_signatures
        structural_new = (semantic_new and task.operators != ("ITERATE",)
                          and task.structural_signature() not in state.structural_signatures)
        if semantic_new:
            state.semantic_history.append(task)
            state.semantic_signatures.add(task.probe_signature())
        if structural_new:
            state.structural_history.append(task)
            state.structural_signatures.add(task.structural_signature())
        for operator in active:
            state.reuses[operator] += int(state.uses[operator] > 0)
            state.uses[operator] += 1
        operator_classes = {operator_semantic_class(operator) for operator in state.ever_invented}
        state.records.append({
            "t": step, "generated_depth": proposal.depth, "task_depth": task.depth,
            "structural_depth": len(task.operators) - 1,
            "active_operator_classes": len(set(active)), "generated": reachable, "born": inventions,
            "retained": list(state.retained), "active": list(active),
            "parametric_novelty": semantic_new and not structural_new,
            "structural_novelty": structural_new, "semantic_novelty": semantic_new,
            "operator_classes": len(operator_classes), "D_language": language_bits(state.retained),
            "K_P0": raw_cost(task), "K_current": encoded_cost(task, active),
        })
    return _result(mode, state, config)


def _result(mode: str, state: State, config: Config) -> Dict[str, object]:
    late = state.records[len(state.records) // 2:]
    summary = {
        "n_parametric_novelty": sum(row["parametric_novelty"] for row in state.records),
        "n_structural_novelty": sum(row["structural_novelty"] for row in state.records),
        "late_parametric_novelty": sum(row["parametric_novelty"] for row in late),
        "late_structural_novelty": sum(row["structural_novelty"] for row in late),
        "structural_novelty_rate_late": sum(row["structural_novelty"] for row in late) / len(late),
        "operator_classes": max(row["operator_classes"] for row in state.records),
        "n_birth": sum(len(steps) for steps in state.births.values()),
        "n_retained_final": len(state.retained),
        "total_reuses": sum(state.reuses.values()),
        "frontier_depth": max(row["task_depth"] for row in state.records),
        "max_D_language": max(row["D_language"] for row in state.records),
        "capacity_fraction": max(row["D_language"] for row in state.records) / config.capacity,
    }
    return {"mode": mode, "condition": state.condition, "config": config, "records": state.records,
            "births": state.births, "reuses": state.reuses, "summary": summary}


def run_conditions(mode: str, config: Config = Config(), seed: int = 1) -> Dict[str, Dict[str, object]]:
    return {condition: run_innovation(mode, condition, config, seed) for condition in CONDITIONS}
