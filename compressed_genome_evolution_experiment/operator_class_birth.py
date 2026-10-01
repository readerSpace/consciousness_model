"""exp608 -- Operator-Class Birth.

The environment generates affine semantic operators q(x)=a*x+b from prior
operators.  Names never define classes: probe actions do.  An individual class
counts only if it is novel, MDL-learnable, retained, reused as a parent by later
operators, and load-bearing under a description-length knockout.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


PROBES = (-2, -1, 0, 1, 2)
FIXED_ENV = "fixed_environment_operators"
GENERATIVE_ENV = "generative_environment_operators"
FIXED_LANGUAGE = "fixed_language"
ADAPTIVE_LANGUAGE = "adaptive_language"


def gamma_bits(value: int) -> int:
    return 2 * value.bit_length() - 1


@dataclass(frozen=True)
class Config:
    steps: int = 300
    capacity: int = 20_000
    manageable_raw_cost: int = 80
    reuse_horizon: int = 3

    def __post_init__(self) -> None:
        if self.steps < 4 or self.capacity <= 0:
            raise ValueError("steps >= 4 and capacity > 0 are required")


@dataclass(frozen=True)
class Operator:
    identifier: int
    a: int
    b: int
    parents: Tuple[int, ...] = ()

    def probe_signature(self) -> Tuple[int, ...]:
        return tuple(self.a * value + self.b for value in PROBES)

    def definition_bits(self) -> int:
        return 6 + sum(gamma_bits(parent + 1) for parent in self.parents)


def operator_distance(first: Operator, second: Operator) -> float:
    first_values, second_values = first.probe_signature(), second.probe_signature()
    return sum(left != right for left, right in zip(first_values, second_values)) / len(PROBES)


def compose(identifier: int, first: Operator, second: Operator) -> Operator:
    """Synthesize a new affine action by adding two parent actions.

    This is an operator-construction primitive, rather than ordinary function
    composition.  It keeps the semantic coefficients compact: their bit length
    grows linearly with the lineage depth rather than exponentially.
    """
    return Operator(identifier, first.a + second.a, first.b + second.b,
                    (first.identifier, second.identifier))


def _environment_operator(environment: str, step: int, operators: List[Operator]) -> Operator:
    if environment == FIXED_ENV:
        return operators[0]
    if environment != GENERATIVE_ENV:
        raise ValueError(environment)
    # Start q1,q2 -> q3 then q3,q1 -> q4.  Afterwards q_t uses q_{t-1}
    # and q_{t-3}, so each mature class becomes a parent twice.
    while len(operators) <= step + 1:
        identifier = len(operators)
        if identifier == 2:
            parents = (operators[1], operators[0])
        elif identifier == 3:
            parents = (operators[1], operators[2])
        elif identifier == 4:
            parents = (operators[3], operators[1])
        else:
            parents = (operators[-1], operators[-3])
        operators.append(compose(identifier, *parents))
    return operators[step + 1]


def raw_cost(operator: Operator) -> int:
    return 12 + 10 * len(operator.parents) + 8 * max(operator.identifier - 1, 0)


def language_bits(retained: Dict[int, Operator]) -> int:
    return sum(operator.definition_bits() for operator in retained.values())


@dataclass
class State:
    language: str
    retained: Dict[int, Operator] = field(default_factory=dict)
    invented: Dict[int, Operator] = field(default_factory=dict)
    records: List[dict] = field(default_factory=list)
    parent_uses: Dict[int, int] = field(default_factory=dict)


def _learnable(operator: Operator, state: State, config: Config) -> bool:
    if operator.identifier in state.invented or state.language == FIXED_LANGUAGE:
        return False
    before = config.reuse_horizon * raw_cost(operator)
    after = operator.definition_bits() + config.reuse_horizon * 2
    return after < before and language_bits(state.retained) + operator.definition_bits() <= config.capacity


def _active_cost(operator: Operator, state: State) -> int:
    return 2 if operator.identifier in state.retained else raw_cost(operator)


def run_birth(environment: str, language: str, config: Config = Config(), seed: int = 1) -> Dict[str, object]:
    if language not in {FIXED_LANGUAGE, ADAPTIVE_LANGUAGE}:
        raise ValueError(language)
    # q0(x)=x+1 and q1(x)=2x provide only primitive semantics; all q_t (t>=2)
    # are generated compositions rather than a fixed operator catalog.
    environment_ops = [Operator(0, 1, 1), Operator(1, 2, 0)]
    state = State(language)
    seen_signatures: set[Tuple[int, ...]] = set()
    for step in range(config.steps):
        proposal = _environment_operator(environment, step, environment_ops)
        born = _learnable(proposal, state, config)
        if born:
            state.invented[proposal.identifier] = proposal
            state.retained[proposal.identifier] = proposal
        reachable = _active_cost(proposal, state) <= config.manageable_raw_cost
        task = proposal if reachable else state.records[-1]["operator"]
        novel = task.probe_signature() not in seen_signatures
        if novel:
            seen_signatures.add(task.probe_signature())
        if reachable and task.parents:
            for parent in task.parents:
                if parent in state.retained:
                    state.parent_uses[parent] = state.parent_uses.get(parent, 0) + 1
        state.records.append({"t": step, "operator": task, "generated": reachable, "born": born,
                              "novel": novel, "D_language": language_bits(state.retained),
                              "K_raw": raw_cost(task), "K_current": _active_cost(task, state)})
    return _result(environment, state, config)


def _result(environment: str, state: State, config: Config) -> Dict[str, object]:
    late = state.records[len(state.records) // 2:]
    functional: Dict[int, dict] = {}
    for identifier, operator in state.invented.items():
        birth_record = next(record for record in state.records if record["born"] and record["operator"].identifier == identifier)
        reuse = state.parent_uses.get(identifier, 0)
        knockout_cost = raw_cost(birth_record["operator"])
        load_bearing = knockout_cost > birth_record["K_current"]
        functional[identifier] = {"signature": operator.probe_signature(), "parents": list(operator.parents),
                                  "birth": birth_record["t"], "reuse": reuse,
                                  "load_bearing": load_bearing,
                                  "functional": bool(reuse >= 2 and load_bearing)}
    late_functional = sum(item["functional"] and item["birth"] >= len(state.records) // 2
                          for item in functional.values())
    summary = {
        "n_operator_classes": len(functional),
        "n_functional_classes": sum(item["functional"] for item in functional.values()),
        "late_functional_classes": late_functional,
        "functional_class_rate_late": late_functional / (len(state.records) / 2),
        "total_parent_reuse": sum(state.parent_uses.values()),
        "frontier_operator": max(record["operator"].identifier for record in state.records),
        "max_D_language": max(record["D_language"] for record in state.records),
        "capacity_fraction": max(record["D_language"] for record in state.records) / config.capacity,
    }
    compact_records = [{key: (value.identifier if key == "operator" else value)
                        for key, value in record.items()} for record in state.records]
    return {"environment": environment, "language": state.language, "config": config,
            "records": compact_records, "lineage": functional, "summary": summary}


def run_conditions(config: Config = Config(), seed: int = 1) -> Dict[str, Dict[str, object]]:
    return {
        "fixed_environment_operators": run_birth(FIXED_ENV, ADAPTIVE_LANGUAGE, config, seed),
        "generative_environment_operators": run_birth(GENERATIVE_ENV, ADAPTIVE_LANGUAGE, config, seed),
        "generative_fixed_individual_language": run_birth(GENERATIVE_ENV, FIXED_LANGUAGE, config, seed),
        "generative_adaptive_individual_language": run_birth(GENERATIVE_ENV, ADAPTIVE_LANGUAGE, config, seed),
    }
