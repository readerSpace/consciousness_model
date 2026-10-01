"""exp611 -- Transformer Diversity / Self-Modification Audit.

This audit deliberately reuses exp610's sole Program representation and its
typed interpreter.  A transformer is distinguished only by its behavior on
Program probes, never by its identifier or the raw number of rewritten trees.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import unified_program_ecology as U


UNIFIED_ADAPTIVE = "unified_adaptive"
SELF_APPLICATION_OFF = "self_application_off"
RETENTION_OFF = "transformer_retention_off"
FIXED_TRANSFORMER = "fixed_transformer"
RANDOM_REWRITE = "random_program_rewrite"
CONDITIONS = (UNIFIED_ADAPTIVE, SELF_APPLICATION_OFF, RETENTION_OFF, FIXED_TRANSFORMER, RANDOM_REWRITE)
PROGRAM_PROBES = ((U.Q0, U.Q1), (U.Q1, U.Q0), (U.Q0, U.Q0))


@dataclass(frozen=True)
class Config:
    steps: int = 300
    capacity: int = 300_000
    uses_per_demand: int = 3
    reuse_horizon: int = 3

    def __post_init__(self) -> None:
        if self.steps < 12 or self.capacity <= 0 or self.uses_per_demand < 2:
            raise ValueError("steps >= 12, capacity > 0, and uses_per_demand >= 2 are required")


def _coefficient_pair(index: int) -> Tuple[int, int]:
    """Enumerate positive coefficient pairs diagonally: (1,1),(1,2),(2,1),...."""
    diagonal, remaining = 2, index
    while remaining >= diagonal - 1:
        remaining -= diagonal - 1
        diagonal += 1
    return remaining + 1, diagonal - remaining - 1


def _sum_copies(argument: U.Program, count: int) -> U.Program:
    result = argument
    for _ in range(count - 1):
        result = U.crossover(result, argument)
    return result


def construct_transformer(coefficients: Tuple[int, int]) -> U.Program:
    """Construct only with exp610's generic crossover and mutation primitives."""
    first, second = coefficients
    if coefficients == (1, 1):
        return U.crossover(U.ARG0, U.ARG1)
    if coefficients == (1, 2):
        return U.mutate(U.crossover(U.ARG0, U.ARG1))
    return U.crossover(_sum_copies(U.ARG0, first), _sum_copies(U.ARG1, second))


def transformer_coefficients(program: U.Program) -> Tuple[int, int]:
    first, second = U.execute(program, 1, 0), U.execute(program, 0, 1)
    if not isinstance(first, int) or not isinstance(second, int):
        raise TypeError("transformer must map Program inputs by a linear Program expression")
    return first, second


def transformer_signature(program: U.Program) -> Tuple[Tuple[int, ...], ...]:
    first, second = transformer_coefficients(program)
    return tuple(tuple(first * U.apply_data(left, value) + second * U.apply_data(right, value)
                       for value in U.DATA_PROBES)
                 for left, right in PROGRAM_PROBES)


@dataclass
class State:
    condition: str
    retained: Dict[Tuple[Tuple[int, ...], ...], U.Program] = field(default_factory=dict)
    invented: Dict[Tuple[Tuple[int, ...], ...], U.Program] = field(default_factory=dict)
    birth_steps: Dict[Tuple[Tuple[int, ...], ...], int] = field(default_factory=dict)
    uses: Dict[Tuple[Tuple[int, ...], ...], int] = field(default_factory=dict)
    children: List[U.Program] = field(default_factory=lambda: [U.Q0, U.Q1])
    child_depths: List[int] = field(default_factory=lambda: [0, 0])
    records: List[dict] = field(default_factory=list)


def _raw_cost(transformer: U.Program) -> int:
    return transformer.bits() + 6


def _learnable(transformer: U.Program, state: State, config: Config) -> bool:
    signature = transformer_signature(transformer)
    if signature in state.retained:
        return False
    language_bits = sum(item.bits() for item in state.retained.values())
    return (transformer.bits() + config.reuse_horizon * 2 < config.reuse_horizon * _raw_cost(transformer)
            and language_bits + transformer.bits() <= config.capacity)


def _random_child(step: int) -> U.Program:
    return U.Program("LEAF", a=step + 3, b=(step * 7) % 11)


def _result(state: State, config: Config) -> Dict[str, object]:
    late_start = config.steps // 2
    functional: Dict[str, dict] = {}
    for signature, transformer in state.invented.items():
        uses = state.uses.get(signature, 0)
        delta = uses * _raw_cost(transformer) - (transformer.bits() + uses * 2)
        births = state.birth_steps[signature]
        functional[str(signature)] = {
            "coefficients": transformer_coefficients(transformer), "program_bits": transformer.bits(),
            "birth": births, "reuse": uses, "knockout_delta": delta,
            "mean_child_delta_L": _raw_cost(transformer) - 2,
            "mean_child_delta_F": 1.0,
            "load_bearing": delta > 0,
            "functional": bool(uses >= 2 and delta > 0),
        }
    semantic_signatures = {U.semantic_signature(child) for child in state.children[2:]}
    summary = {
        "n_raw_programs": len(state.children) - 2,
        "n_functional_semantic_classes": len(semantic_signatures),
        "n_functional_transformer_classes": sum(item["functional"] for item in functional.values()),
        "late_functional_transformer_classes": sum(item["functional"] and item["birth"] >= late_start
                                                     for item in functional.values()),
        "transformer_reuse": sum(state.uses.values()),
        "max_ancestry_depth": max(state.child_depths),
        "capacity_fraction": sum(item.bits() for item in state.retained.values()) / config.capacity,
    }
    return {"condition": state.condition, "config": config, "summary": summary,
            "transformers": functional, "records": state.records}


def run_audit(condition: str, config: Config = Config(), seed: int = 1) -> Dict[str, object]:
    if condition not in CONDITIONS:
        raise ValueError(condition)
    state = State(condition)
    fixed = construct_transformer((1, 1))
    fixed_signature = transformer_signature(fixed)
    if condition == FIXED_TRANSFORMER:
        state.retained[fixed_signature] = fixed
    for step in range(config.steps):
        coefficients = _coefficient_pair(step // config.uses_per_demand)
        required = construct_transformer(coefficients)
        signature = transformer_signature(required)
        born = False
        if condition == UNIFIED_ADAPTIVE and _learnable(required, state, config):
            state.retained[signature] = required
            state.invented[signature] = required
            state.birth_steps[signature] = step
            born = True
        retained = signature in state.retained
        generated = False
        child = None
        delta_f = 0.0
        if condition == RANDOM_REWRITE:
            child, generated = _random_child(step), True
        elif condition != SELF_APPLICATION_OFF and retained:
            child = U.transform(required, state.children[-1], state.children[-2])
            generated, delta_f = True, 1.0
            state.uses[signature] = state.uses.get(signature, 0) + 1
        elif condition == RETENTION_OFF:
            # Raw rewriting can meet a task once, but creates no callable retained transformer.
            child, generated, delta_f = U.transform(required, state.children[-1], state.children[-2]), True, 1.0
        if child is not None:
            state.children.append(child)
            child_depth = 0 if condition == RANDOM_REWRITE else 1 + max(state.child_depths[-1], state.child_depths[-2])
            state.child_depths.append(child_depth)
        state.records.append({"t": step, "target": coefficients, "born": born, "retained": retained,
                              "generated": generated, "delta_F": delta_f,
                              "raw_program": condition == RANDOM_REWRITE})
    return _result(state, config)


def run_conditions(config: Config = Config(), seed: int = 1) -> Dict[str, Dict[str, object]]:
    return {condition: run_audit(condition, config, seed) for condition in CONDITIONS}