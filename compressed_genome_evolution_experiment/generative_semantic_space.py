"""exp606 -- Generative Semantic Space.

Tasks are variable-length semantic programs, not members of a finite catalog.
The generator grows the tree REPEAT2^d(MAP/REVERSE(x)) without an upper bound
on d.  A generic ITERATE_REPEAT2 instruction may be invented by MDL, allowing
the adaptive language to encode a previously unreachable semantic frontier.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np


PROBES = ((0, 1, 1), (1, 0, 0), (0, 1, 0, 1))
ITERATE_BITS = 20
CALL_BITS = 2
RAW_JOIN_BITS = 8
REF_BITS = 3


def gamma_bits(value: int) -> int:
    if value < 1:
        raise ValueError(value)
    return 2 * value.bit_length() - 1


@dataclass(frozen=True)
class Config:
    steps: int = 100
    capacity: int = 64
    manageable_cost: int = 64
    reuse_horizon: int = 8

    def __post_init__(self) -> None:
        if self.steps < 2 or self.capacity <= 0 or self.manageable_cost <= 0:
            raise ValueError("steps >= 2, capacity > 0, manageable_cost > 0 are required")


@dataclass(frozen=True)
class Task:
    """Canonical form of the variable-length tree REPEAT2^depth(base(x))."""

    depth: int
    reverse: bool
    flip: bool

    def factor(self) -> int:
        return 1 << self.depth

    def semantic_signature(self) -> Tuple[int, bool, bool]:
        return self.factor(), self.reverse, self.flip

    def evaluate(self, probe: Tuple[int, ...]) -> Tuple[int, ...]:
        value = tuple(1 - bit for bit in probe) if self.flip else probe
        value = tuple(reversed(value)) if self.reverse else value
        return value * self.factor()


def probe_distance(first: Task, second: Task) -> float:
    """Exact probe distance in this canonical DSL without materializing long outputs."""
    if first.semantic_signature() == second.semantic_signature():
        return 0.0
    # Every registered probe is nonempty.  A distinct repeat factor changes
    # output length; a changed map/reverse flag differs on at least one probe.
    # This is exactly the result of the finite probe suite, without expanding
    # a length 2^depth output.
    return 1.0


def raw_cost(task: Task) -> int:
    """Cost of spelling out each repeat node in the variable-length syntax tree."""
    return 10 * task.depth + 4


def iterated_cost(task: Task) -> int:
    return CALL_BITS + gamma_bits(task.depth + 1) + 2


@dataclass
class State:
    condition: str
    retained_iterate: bool = False
    invented: bool = False
    history: List[Task] = field(default_factory=list)
    records: List[dict] = field(default_factory=list)
    uses: int = 0
    reuses: int = 0
    birth_steps: List[int] = field(default_factory=list)


def description_cost(task: Task, retained_iterate: bool) -> int:
    return iterated_cost(task) if retained_iterate else raw_cost(task)


def invention_pays(task: Task, config: Config) -> bool:
    before = config.reuse_horizon * raw_cost(task)
    after = ITERATE_BITS + config.reuse_horizon * iterated_cost(task)
    return after < before and ITERATE_BITS <= config.capacity


def _candidate(step: int, seed: int) -> Task:
    # Depth is unbounded in time.  The two base transformations prevent the
    # sequence from reducing to output length alone while preserving canonical
    # semantic comparison on probes.
    return Task(depth=step + 1, reverse=bool((step + seed) % 2), flip=bool((step // 2 + seed) % 2))


def run_space(condition: str, config: Config = Config(), seed: int = 1) -> Dict[str, object]:
    if condition not in {"adaptive_language", "fixed_language"}:
        raise ValueError(condition)
    state = State(condition)
    for step in range(config.steps):
        proposal = _candidate(step, seed)
        prior_cost = description_cost(proposal, state.retained_iterate)
        born = False
        if (condition == "adaptive_language" and not state.invented
                and invention_pays(proposal, config)):
            state.invented = state.retained_iterate = True
            state.birth_steps.append(step)
            born = True
        current_cost = description_cost(proposal, state.retained_iterate)
        reachable = current_cost <= config.manageable_cost
        task = proposal if reachable else state.history[-1]
        is_new = not state.history or probe_distance(task, state.history[-1]) > 0.5
        if is_new:
            state.history.append(task)
        if state.retained_iterate and reachable:
            state.reuses += int(state.uses > 0)
            state.uses += 1
        record = {
            "t": step, "generated_depth": proposal.depth, "task_depth": task.depth,
            "generated": reachable, "born": born, "retained_iterate": state.retained_iterate,
            "K_P0": raw_cost(task), "K_current": description_cost(task, state.retained_iterate),
            "A_advantage": raw_cost(task) - description_cost(task, state.retained_iterate),
            "semantic_novelty": is_new, "D_language": ITERATE_BITS if state.retained_iterate else 0,
        }
        state.records.append(record)
    late = state.records[len(state.records) // 2:]
    summary = {
        "n_semantic_novelty": sum(record["semantic_novelty"] for record in state.records),
        "late_semantic_novelty": sum(record["semantic_novelty"] for record in late),
        "novelty_rate_late": sum(record["semantic_novelty"] for record in late) / len(late),
        "n_birth": len(state.birth_steps), "n_retained": int(state.retained_iterate),
        "uses": state.uses, "reuses": state.reuses,
        "frontier_depth": max(record["task_depth"] for record in state.records),
        "generated_frontier_depth": max(record["generated_depth"] for record in state.records if record["generated"]),
        "mean_A_late": float(np.mean([record["A_advantage"] for record in late])),
        "max_D_language": max(record["D_language"] for record in state.records),
        "capacity_fraction": max(record["D_language"] for record in state.records) / config.capacity,
    }
    return {"condition": condition, "config": config, "records": state.records,
            "semantic_history": [task.semantic_signature() for task in state.history], "summary": summary}


def run_conditions(config: Config = Config(), seed: int = 1) -> Dict[str, Dict[str, object]]:
    return {condition: run_space(condition, config, seed) for condition in
            ("adaptive_language", "fixed_language")}
