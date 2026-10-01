"""exp609 -- Meta-Operator Evolution.

Operators are affine actions.  Meta-rules that synthesize them are not named
primitives: each is a small stack program over LEFT, RIGHT, DUP, and ADD.
An adaptive individual may retain a meta-rule only when repeated raw uses repay
its MDL definition; fixed-meta-rule receives only the early rule R1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Tuple


META_PROBES = ((1, 0), (0, 1), (1, 1), (2, -1), (-1, 2))
FIXED_META = "fixed_meta_rule"
ADAPTIVE_META = "adaptive_meta_rule"


def gamma_bits(value: int) -> int:
    return 2 * value.bit_length() - 1


@dataclass(frozen=True)
class Config:
    steps: int = 300
    capacity: int = 30_000
    rule_switch_fraction: float = 0.5
    reuse_horizon: int = 4

    def __post_init__(self) -> None:
        if self.steps < 12 or self.capacity <= 0 or not 0 < self.rule_switch_fraction < 1:
            raise ValueError("steps >= 12, capacity > 0, and a switch fraction in (0, 1) are required")


@dataclass(frozen=True)
class MetaRule:
    identifier: int
    program: Tuple[str, ...]

    def apply_pair(self, left: int, right: int) -> int:
        stack: List[int] = []
        for instruction in self.program:
            if instruction == "LEFT":
                stack.append(left)
            elif instruction == "RIGHT":
                stack.append(right)
            elif instruction == "DUP":
                stack.append(stack[-1])
            elif instruction == "ADD":
                right_value, left_value = stack.pop(), stack.pop()
                stack.append(left_value + right_value)
            else:
                raise ValueError(instruction)
        if len(stack) != 1:
            raise ValueError("meta-rule must leave one result")
        return stack[0]

    def probe_signature(self) -> Tuple[int, ...]:
        return tuple(self.apply_pair(left, right) for left, right in META_PROBES)

    def definition_bits(self) -> int:
        return 6 + 2 * len(self.program)

    def raw_cost(self) -> int:
        return 6 + 2 * len(self.program)


R1 = MetaRule(1, ("LEFT", "RIGHT", "ADD"))
R2 = MetaRule(2, ("LEFT", "RIGHT", "DUP", "ADD", "ADD"))


@dataclass(frozen=True)
class Operator:
    identifier: int
    a: int
    b: int
    parents: Tuple[int, int]
    rule: int

    def probe_signature(self) -> Tuple[int, ...]:
        return tuple(self.a * value + self.b for value in (-2, -1, 0, 1, 2))

    def definition_bits(self) -> int:
        return 6 + sum(gamma_bits(parent + 1) for parent in self.parents) + gamma_bits(self.rule)


def meta_rule_distance(first: MetaRule, second: MetaRule) -> float:
    return sum(left != right for left, right in zip(first.probe_signature(), second.probe_signature())) / len(META_PROBES)


def _required_rule(step: int, switch_step: int) -> MetaRule:
    # R1 remains useful after the change, so fixed-meta continues growing but
    # cannot cover all demands.  R2 first appears strictly in the late half.
    return R1 if step < switch_step or step % 2 == 0 else R2


def _synthesize(identifier: int, rule: MetaRule, left: Operator, right: Operator) -> Operator:
    return Operator(identifier, rule.apply_pair(left.a, right.a), rule.apply_pair(left.b, right.b),
                    (left.identifier, right.identifier), rule.identifier)


@dataclass
class State:
    mode: str
    retained_rules: Dict[int, MetaRule] = field(default_factory=dict)
    invented_rules: Dict[int, MetaRule] = field(default_factory=dict)
    operators: List[Operator] = field(default_factory=list)
    rule_uses: Dict[int, int] = field(default_factory=dict)
    records: List[dict] = field(default_factory=list)


def _rule_learnable(rule: MetaRule, state: State, config: Config) -> bool:
    if rule.identifier in state.retained_rules:
        return False
    if rule.probe_signature() in {item.probe_signature() for item in state.retained_rules.values()}:
        return False
    raw = config.reuse_horizon * rule.raw_cost()
    encoded = rule.definition_bits() + config.reuse_horizon * 2
    language = sum(item.definition_bits() for item in state.retained_rules.values())
    return encoded < raw and language + rule.definition_bits() <= config.capacity


def _result(state: State, config: Config) -> Dict[str, object]:
    functional_rules: Dict[int, dict] = {}
    for identifier, rule in state.invented_rules.items():
        birth = next(record for record in state.records if record["rule_born"] == identifier)
        uses = state.rule_uses.get(identifier, 0)
        knockout_delta = uses * rule.raw_cost() - (rule.definition_bits() + uses * 2)
        functional_rules[identifier] = {
            "program": list(rule.program), "signature": rule.probe_signature(), "birth": birth["t"],
            "reuse": uses, "knockout_delta": knockout_delta,
            "load_bearing": knockout_delta > 0,
            "functional": bool(uses >= 2 and knockout_delta > 0),
        }
    late_start = config.steps // 2
    operator_signatures = set()
    late_operator_signatures = set()
    for operator in state.operators[2:]:
        signature = operator.probe_signature()
        if operator.identifier - 2 >= late_start and signature not in operator_signatures:
            late_operator_signatures.add(signature)
        operator_signatures.add(signature)
    summary = {
        "n_operator_classes": len(operator_signatures),
        "late_operator_classes": len(late_operator_signatures),
        "n_meta_rule_classes": len(state.retained_rules),
        "n_functional_meta_rules": sum(item["functional"] for item in functional_rules.values()),
        "late_functional_meta_rules": sum(item["functional"] and item["birth"] >= late_start
                                           for item in functional_rules.values()),
        "meta_rule_reuse": sum(state.rule_uses.values()),
        "frontier_operator": state.operators[-1].identifier,
        "max_D_meta": sum(rule.definition_bits() for rule in state.retained_rules.values()),
        "capacity_fraction": sum(rule.definition_bits() for rule in state.retained_rules.values()) / config.capacity,
    }
    records = [{key: (value.identifier if key == "operator" and value is not None else value)
                for key, value in record.items()}
               for record in state.records]
    return {"mode": state.mode, "config": config, "records": records,
            "meta_lineage": functional_rules, "operator_lineage": [
                {"id": operator.identifier, "parents": list(operator.parents), "rule": operator.rule}
                for operator in state.operators[2:]], "summary": summary}


def run_meta_evolution(mode: str, config: Config = Config(), seed: int = 1) -> Dict[str, object]:
    if mode not in {FIXED_META, ADAPTIVE_META}:
        raise ValueError(mode)
    state = State(mode)
    if mode == FIXED_META:
        state.retained_rules[R1.identifier] = R1
    state.operators = [Operator(0, 1, 1, (0, 0), 0), Operator(1, 2, 0, (1, 1), 0)]
    switch_step = int(config.steps * config.rule_switch_fraction)
    for step in range(config.steps):
        needed = _required_rule(step, switch_step)
        born = None
        if mode == ADAPTIVE_META and _rule_learnable(needed, state, config):
            state.retained_rules[needed.identifier] = needed
            state.invented_rules[needed.identifier] = needed
            born = needed.identifier
        generated = needed.identifier in state.retained_rules
        operator = None
        if generated:
            operator = _synthesize(len(state.operators), needed, state.operators[-1], state.operators[-2])
            state.operators.append(operator)
            state.rule_uses[needed.identifier] = state.rule_uses.get(needed.identifier, 0) + 1
        state.records.append({"t": step, "required_rule": needed.identifier, "generated": generated,
                              "rule_born": born, "operator": operator,
                              "D_meta": sum(rule.definition_bits() for rule in state.retained_rules.values())})
    return _result(state, config)


def run_conditions(config: Config = Config(), seed: int = 1) -> Dict[str, Dict[str, object]]:
    return {mode: run_meta_evolution(mode, config, seed) for mode in (FIXED_META, ADAPTIVE_META)}