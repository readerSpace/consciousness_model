"""Workspace-controlled proof search with a small trusted propositional kernel.

The planner is allowed to fail, revise its strategy, and retain lightweight
strategy statistics.  Only ``ProofKernel`` decides whether a constructed proof
is valid; this module makes no claim about subjective experience.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Sequence, Tuple, Union

from Consciousness_model import ConsciousnessController, IntegratedState


@dataclass(frozen=True)
class Atom:
    name: str


@dataclass(frozen=True)
class And:
    left: Formula
    right: Formula


@dataclass(frozen=True)
class Implies:
    premise: Formula
    conclusion: Formula


Formula = Union[Atom, And, Implies]


def render(formula: Formula) -> str:
    if isinstance(formula, Atom):
        return formula.name
    if isinstance(formula, And):
        return f"({render(formula.left)} & {render(formula.right)})"
    return f"({render(formula.premise)} -> {render(formula.conclusion)})"


@dataclass(frozen=True)
class ProofStep:
    rule: str
    conclusion: Formula
    premises: Tuple[ProofStep, ...] = ()


class ProofKernel:
    """A trusted checker for a deliberately small natural-deduction calculus."""

    def check(self, proof: ProofStep, assumptions: Sequence[Formula]) -> bool:
        context = tuple(assumptions)
        if proof.rule == "assumption":
            return not proof.premises and proof.conclusion in context
        if proof.rule == "and_intro" and isinstance(proof.conclusion, And) and len(proof.premises) == 2:
            left, right = proof.premises
            return (
                left.conclusion == proof.conclusion.left
                and right.conclusion == proof.conclusion.right
                and self.check(left, context)
                and self.check(right, context)
            )
        if proof.rule in {"and_elim_left", "and_elim_right"} and len(proof.premises) == 1:
            source = proof.premises[0]
            if not isinstance(source.conclusion, And) or not self.check(source, context):
                return False
            expected = source.conclusion.left if proof.rule == "and_elim_left" else source.conclusion.right
            return proof.conclusion == expected
        if proof.rule == "imp_intro" and isinstance(proof.conclusion, Implies) and len(proof.premises) == 1:
            body = proof.premises[0]
            return body.conclusion == proof.conclusion.conclusion and self.check(body, context + (proof.conclusion.premise,))
        if proof.rule == "imp_elim" and len(proof.premises) == 2:
            implication, premise = proof.premises
            return (
                isinstance(implication.conclusion, Implies)
                and implication.conclusion.premise == premise.conclusion
                and implication.conclusion.conclusion == proof.conclusion
                and self.check(implication, context)
                and self.check(premise, context)
            )
        return False


@dataclass(frozen=True)
class ProofState:
    goal: Formula
    active_facts: Tuple[Formula, ...]
    workspace: IntegratedState
    uncertainty: float


@dataclass(frozen=True)
class ProofResult:
    proof: ProofStep | None
    state: ProofState
    control: str
    explored_nodes: int


class ProofPlanner:
    """Uses finite q to select proof facts, then delegates validity to a kernel."""

    def __init__(self, capacity: int = 4, max_depth: int = 8) -> None:
        self.controller = ConsciousnessController(capacity=capacity)
        self.kernel = ProofKernel()
        self.max_depth = max_depth
        self.strategy_history: Dict[str, Tuple[int, int]] = {}

    def prove(self, assumptions: Iterable[Formula], goal: Formula) -> ProofResult:
        context = tuple(assumptions)
        self.controller.perceive(("fact", "available", render(formula)) for formula in context)
        q = self.controller.integrate((render(goal), "available"))
        active = tuple(formula for formula in context if render(formula) in self._active_strings(q))
        state = ProofState(goal, active, q, q.uncertainty)
        proof, explored = self._search(goal, active, self.max_depth)
        valid = proof is not None and self.kernel.check(proof, context)
        if valid:
            control = "commit"
            self._record("commit", True)
        else:
            control = "backtrack" if explored else "retrieve"
            self._record(control, False)
            self.controller.observe_outcome(self.controller.select_action(), succeeded=False)
            self.controller.reflect()
        return ProofResult(proof if valid else None, state, control, explored)

    def strategy_confidence(self, strategy: str) -> float:
        successes, attempts = self.strategy_history.get(strategy, (0, 0))
        return successes / attempts if attempts else 0.0

    def _search(self, goal: Formula, context: Tuple[Formula, ...], depth: int) -> Tuple[ProofStep | None, int]:
        if depth < 0:
            return None, 0
        if goal in context:
            return ProofStep("assumption", goal), 1
        if isinstance(goal, And):
            left, left_nodes = self._search(goal.left, context, depth - 1)
            right, right_nodes = self._search(goal.right, context, depth - 1)
            if left and right:
                return ProofStep("and_intro", goal, (left, right)), left_nodes + right_nodes + 1
            return None, left_nodes + right_nodes
        if isinstance(goal, Implies):
            body, nodes = self._search(goal.conclusion, context + (goal.premise,), depth - 1)
            return (ProofStep("imp_intro", goal, (body,)), nodes + 1) if body else (None, nodes)
        nodes = 0
        for fact in context:
            if isinstance(fact, And) and fact.left == goal:
                return ProofStep("and_elim_left", goal, (ProofStep("assumption", fact),)), nodes + 2
            if isinstance(fact, And) and fact.right == goal:
                return ProofStep("and_elim_right", goal, (ProofStep("assumption", fact),)), nodes + 2
            if isinstance(fact, Implies) and fact.conclusion == goal:
                premise, premise_nodes = self._search(fact.premise, context, depth - 1)
                nodes += premise_nodes
                if premise:
                    return ProofStep("imp_elim", goal, (ProofStep("assumption", fact), premise)), nodes + 2
        return None, nodes

    def _record(self, strategy: str, succeeded: bool) -> None:
        successes, attempts = self.strategy_history.get(strategy, (0, 0))
        self.strategy_history[strategy] = (successes + int(succeeded), attempts + 1)

    def _active_strings(self, q: IntegratedState) -> set[str]:
        return {
            relation[2]
            for identifier in q.concepts
            for relation in self.controller.workspace.long_term[identifier].relations
            if relation[0] == "fact" and relation[1] == "available"
        }