"""exp610 -- Unified Program Ecology.

There is one first-class ``Program`` representation.  The same evaluator
applies it to integers (an operator role) or to Programs (a transformer role).
No meta-program type exists: roles arise solely from the type of application.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Union


SPECIALIZED = "specialized_two_layer"
UNIFIED = "unified_program_ecology"
DATA_PROBES = (-2, -1, 0, 1, 2)


@dataclass(frozen=True)
class Config:
    steps: int = 300
    capacity: int = 30_000
    switch_fraction: float = 0.5
    reuse_horizon: int = 4

    def __post_init__(self) -> None:
        if self.steps < 12 or self.capacity <= 0 or not 0 < self.switch_fraction < 1:
            raise ValueError("steps >= 12, capacity > 0, and switch_fraction in (0, 1) are required")


@dataclass(frozen=True)
class Program:
    tag: str
    left: "Program | None" = None
    right: "Program | None" = None
    a: int = 0
    b: int = 0

    def bits(self) -> int:
        if self.tag == "LEAF":
            return 6 + abs(self.a).bit_length() + abs(self.b).bit_length()
        if self.tag in {"ARG0", "ARG1"}:
            return 2
        return 2 + self.left.bits() + self.right.bits()


ARG0 = Program("ARG0")
ARG1 = Program("ARG1")
Q0 = Program("LEAF", a=1, b=1)
Q1 = Program("LEAF", a=2, b=0)


def add(left: Program, right: Program) -> Program:
    return Program("ADD", left, right)


def crossover(left: Program, right: Program) -> Program:
    """Generic Program crossover: construct an ADD node for any Programs."""
    return add(left, right)


def mutate(program: Program) -> Program:
    """Generic subtree duplication mutation, independent of a Program's role."""
    if program.tag in {"ARG0", "ARG1", "LEAF"}:
        return add(program, program)
    return add(program.left, mutate(program.right))


R1 = crossover(ARG0, ARG1)
R2 = mutate(R1)


Value = Union[int, Program]


def execute(program: Program, first: Value, second: Value) -> Value:
    """The sole typed interpreter used for data and program application."""
    if program.tag == "ARG0":
        return first
    if program.tag == "ARG1":
        return second
    if program.tag == "LEAF":
        if not isinstance(first, int):
            return program
        return program.a * first + program.b
    if program.tag == "ADD":
        left, right = execute(program.left, first, second), execute(program.right, first, second)
        if isinstance(left, int) and isinstance(right, int):
            return left + right
        if isinstance(left, Program) and isinstance(right, Program):
            return add(left, right)
    raise TypeError("ADD operands must share a type")


def apply_data(program: Program, value: int) -> int:
    result = execute(program, value, 0)
    if not isinstance(result, int):
        raise TypeError("program did not produce data")
    return result


def transform(transformer: Program, first: Program, second: Program) -> Program:
    result = execute(transformer, first, second)
    if not isinstance(result, Program):
        raise TypeError("program did not produce a program")
    return result


def semantic_signature(program: Program, cache: Dict[int, Tuple[int, ...]] | None = None) -> Tuple[int, ...]:
    """Evaluate a shared Program DAG once per node, not once per expansion."""
    cache = {} if cache is None else cache
    stack = [program]
    while stack:
        node = stack[-1]
        node_id = id(node)
        if node_id in cache:
            stack.pop()
        elif node.tag == "LEAF":
            cache[node_id] = tuple(node.a * value + node.b for value in DATA_PROBES)
            stack.pop()
        elif node.tag == "ADD" and id(node.left) in cache and id(node.right) in cache:
            left, right = cache[id(node.left)], cache[id(node.right)]
            cache[node_id] = tuple(first + second for first, second in zip(left, right))
            stack.pop()
        elif node.tag == "ADD":
            stack.extend((node.right, node.left))
        else:
            raise TypeError("only data Programs have semantic signatures")
    return cache[id(program)]


def transformer_signature(program: Program) -> Tuple[Tuple[int, ...], ...]:
    return tuple(semantic_signature(transform(program, left, right))
                 for left, right in ((Q0, Q1), (Q1, Q0), (Q0, Q0)))


def ancestry_depth(program: Program) -> int:
    depth, stack = 0, [(program, 0)]
    while stack:
        node, node_depth = stack.pop()
        depth = max(depth, node_depth)
        if node.tag == "ADD":
            stack.extend(((node.left, node_depth + 1), (node.right, node_depth + 1)))
    return depth


def _needed_transformer(step: int, switch_step: int) -> Program:
    return R1 if step < switch_step or step % 2 == 0 else R2


@dataclass
class State:
    mode: str
    retained_transformers: Dict[Tuple[Tuple[int, ...], ...], Program] = field(default_factory=dict)
    invented_transformers: Dict[Tuple[Tuple[int, ...], ...], Program] = field(default_factory=dict)
    operators: List[Program] = field(default_factory=lambda: [Q0, Q1])
    operator_depths: List[int] = field(default_factory=lambda: [0, 0])
    transformer_uses: Dict[Tuple[Tuple[int, ...], ...], int] = field(default_factory=dict)
    birth_steps: Dict[Tuple[Tuple[int, ...], ...], int] = field(default_factory=dict)
    records: List[dict] = field(default_factory=list)


def _raw_transform_cost(program: Program) -> int:
    return program.bits() + 6


def _learnable(program: Program, state: State, config: Config) -> bool:
    signature = transformer_signature(program)
    if signature in state.retained_transformers:
        return False
    raw = config.reuse_horizon * _raw_transform_cost(program)
    encoded = program.bits() + config.reuse_horizon * 2
    language = sum(item.bits() for item in state.retained_transformers.values())
    return encoded < raw and language + program.bits() <= config.capacity


def _specialized_transform(required: Program, first: Program, second: Program) -> Program:
    """Two-layer control: the rule is external, not a first-class Program."""
    if required == R1:
        return add(first, second)
    return add(first, add(second, second))


def _result(state: State, config: Config) -> Dict[str, object]:
    late_start = config.steps // 2
    functional: Dict[str, dict] = {}
    for signature, transformer in state.invented_transformers.items():
        uses = state.transformer_uses.get(signature, 0)
        delta = uses * _raw_transform_cost(transformer) - (transformer.bits() + uses * 2)
        functional[str(signature)] = {
            "program_bits": transformer.bits(), "signature": signature,
            "birth": state.birth_steps[signature], "reuse": uses, "knockout_delta": delta,
            "load_bearing": delta > 0, "functional": bool(uses >= 2 and delta > 0),
        }
    semantic_cache: Dict[int, Tuple[int, ...]] = {}
    seen, late_seen = set(), set()
    for index, program in enumerate(state.operators[2:]):
        signature = semantic_signature(program, semantic_cache)
        if index >= late_start and signature not in seen:
            late_seen.add(signature)
        seen.add(signature)
    max_depth = max(state.operator_depths)
    summary = {
        "n_functional_semantic_classes": len(seen),
        "late_functional_semantic_classes": len(late_seen),
        "n_functional_transformer_classes": sum(item["functional"] for item in functional.values()),
        "late_functional_transformer_classes": sum(item["functional"] and item["birth"] >= late_start
                                                     for item in functional.values()),
        "max_ancestry_depth": max_depth,
        "transformer_reuse": sum(state.transformer_uses.values()),
        "frontier_program": len(state.operators) - 1,
        "capacity_fraction": sum(item.bits() for item in state.retained_transformers.values()) / config.capacity,
    }
    return {"mode": state.mode, "config": config, "summary": summary, "transformers": functional,
            "operator_lineage": [{"signature": semantic_signature(program, semantic_cache), "depth": depth}
                                  for program, depth in zip(state.operators[2:], state.operator_depths[2:])],
            "records": state.records}


def run_ecology(mode: str, config: Config = Config(), seed: int = 1) -> Dict[str, object]:
    if mode not in {SPECIALIZED, UNIFIED}:
        raise ValueError(mode)
    state = State(mode)
    switch_step = int(config.steps * config.switch_fraction)
    if mode == SPECIALIZED:
        state.retained_transformers[transformer_signature(R1)] = R1
    for step in range(config.steps):
        required = _needed_transformer(step, switch_step)
        signature = transformer_signature(required)
        born = False
        if mode == UNIFIED and _learnable(required, state, config):
            state.retained_transformers[signature] = required
            state.invented_transformers[signature] = required
            state.birth_steps[signature] = step
            born = True
        available = signature in state.retained_transformers if mode == UNIFIED else (
            required == R1 or required == R2)
        program = None
        if available:
            if mode == UNIFIED:
                program = transform(required, state.operators[-1], state.operators[-2])
            else:
                program = _specialized_transform(required, state.operators[-1], state.operators[-2])
            state.operators.append(program)
            state.operator_depths.append(1 + max(state.operator_depths[-1], state.operator_depths[-2]))
            state.transformer_uses[signature] = state.transformer_uses.get(signature, 0) + 1
        state.records.append({"t": step, "required_bits": required.bits(), "generated": available,
                              "transformer_born": born,
                              "operator_depth": state.operator_depths[-1] if program else None})
    return _result(state, config)


def run_conditions(config: Config = Config(), seed: int = 1) -> Dict[str, Dict[str, object]]:
    return {mode: run_ecology(mode, config, seed) for mode in (SPECIALIZED, UNIFIED)}