"""L8.37: the same three states at any number of unknowns, or the rule was ad hoc.

L8.34 solved two unknown words together and the gain was real, but two is the
number where a cartesian product is small enough that nothing has to be clever.
So the question this layer answers is not "does it scale" -- it is whether the
*rule* generalises::

    B_0 = M_1 x ... x M_n,     B_{t+1} = { b in B_t : C_t(b) }

with execution gated on ``|B| = 1`` and nothing else.  **Adding no new safety rule
as n grows is itself the thing under test.**  A layer that needed a fresh
threshold at n=4 would have shown that L8.21's uniqueness rule was a
two-variable convenience rather than a principle.

The family that decides it is ``GLOBAL_ONLY``: every single-word marginal is
ambiguous, every two-word marginal is ambiguous, and the whole assignment is
unique.  Such a world cannot be solved by being good at words or at pairs, so
solving it is evidence of constraint satisfaction rather than of a generalised
special case.  It needs a genuinely n-ary constraint, and the domain supplies
one -- 「1つ目が残りの合計より大きい」 relates all n at once and decomposes into no
set of binary constraints.

**Candidate-space growth matters more than n.**  Succeeding at n=5 proves little
when |M|^n is still enumerable, so the suite does both: at n=3..5 the propagating
solver is checked against an **exhaustive oracle** for exact agreement, and at
n=6, 8, 10 it is asked for the same answers while the states it visits are
counted against |M|^n.  Agreeing with brute force and then not doing brute force
is what separates a constraint solver from a loop over a product.

``SYMMETRIC`` is the control that cannot be passed by trying harder.  When the
constraints are invariant under permuting the meanings, the solutions come in
orbits and no amount of n or evidence breaks the tie.  Staying ambiguous there is
correct, and a solver that ever returns one of a symmetric pair has invented the
distinction rather than found it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product

from .cross_context_identification_l829_experiment import (
    FieldFacts, Observation, constrain,
)
from .joint_inference_l834_experiment import DISTINCT, EXCEEDS, SAME_TYPE, holds
from .joint_inference_l834_experiment import Relation as PairRelation

RESOLVED, AMBIGUOUS, CONTRADICTION = "RESOLVED", "AMBIGUOUS", "CONTRADICTION"

#: n-ary constraint kinds. ``ALL_DIFFERENT`` is decomposable into pairs and is
#: kept as one for propagation's sake; ``DOMINATES_SUM`` is not decomposable at
#: all, which is exactly why ``GLOBAL_ONLY`` needs it.
ALL_DIFFERENT, DOMINATES_SUM = "ALL_DIFFERENT", "DOMINATES_SUM"


@dataclass(frozen=True)
class Nary:
    kind: str
    indices: tuple[int, ...]
    note: str = ""


def _facts(universe, name) -> FieldFacts:
    return next(f for f in universe if f.name == name)


def nary_holds(constraint: Nary, assignment: tuple[str, ...], universe) -> bool:
    picked = [assignment[i] for i in constraint.indices]
    if constraint.kind == ALL_DIFFERENT:
        return len(set(picked)) == len(picked)
    if constraint.kind == DOMINATES_SUM:
        # 「1つ目が残りの合計より大きい」: true when the first can exceed the sum of
        # the rest given what the store recorded. Relates all n at once -- no set
        # of binary constraints expresses it.
        head, rest = picked[0], picked[1:]
        if len(set(picked)) != len(picked):
            return False
        top = _facts(universe, head).high
        floor = sum(_facts(universe, name).low for name in rest)
        return top is not None and top > floor
    raise ValueError(f"unknown n-ary constraint: {constraint.kind}")


@dataclass
class Problem:
    universe: tuple[FieldFacts, ...]
    size: int
    unary: dict[int, tuple[Observation, ...]] = field(default_factory=dict)
    binary: tuple[PairRelation, ...] = ()
    nary: tuple[Nary, ...] = ()

    @property
    def meanings(self) -> tuple[str, ...]:
        return tuple(facts.name for facts in self.universe)

    def domains(self) -> list[set[str]]:
        found = []
        for index in range(self.size):
            live = set(self.meanings)
            for observation in self.unary.get(index, ()):
                live &= constrain(observation, self.universe)
            found.append(live)
        return found

    def satisfied(self, assignment: tuple[str, ...]) -> bool:
        for index, observations in self.unary.items():
            for observation in observations:
                if assignment[index] not in constrain(observation, self.universe):
                    return False
        for relation in self.binary:
            if not holds(relation, assignment, self.universe):
                return False
        for constraint in self.nary:
            if not nary_holds(constraint, assignment, self.universe):
                return False
        return True


# --------------------------------------------------------------------------
# the oracle
# --------------------------------------------------------------------------

def exhaustive(problem: Problem, cap: int = 3) -> tuple[list[tuple[str, ...]], int]:
    """Every assignment, tested. The ground truth the solver has to match.

    ``cap`` stops the collection, not the enumeration: the state count is the
    honest |M|^n so the comparison against propagation means something.
    """
    found, states = [], 0
    for assignment in product(problem.meanings, repeat=problem.size):
        states += 1
        if problem.satisfied(assignment):
            if len(found) < cap:
                found.append(assignment)
    return found, states


# --------------------------------------------------------------------------
# propagation
# --------------------------------------------------------------------------

def _binary_supported(relation: PairRelation, index: int, value: str,
                      domains: list[set[str]], universe) -> bool:
    """Is there a partner in the other domain that makes this relation hold?"""
    other = relation.right if index == relation.left else relation.left
    for partner in domains[other]:
        pair = [""] * (max(relation.left, relation.right) + 1)
        pair[index], pair[other] = value, partner
        if holds(relation, tuple(pair), universe):
            return True
    return False


def propagate(problem: Problem) -> list[set[str]] | None:
    """Arc consistency over the binary constraints, plus what the n-ary ones say.

    Returns ``None`` when a domain empties, which is a contradiction found without
    touching the product. The n-ary filtering is deliberately weak -- it prunes a
    value only when no completion of the *other* domains supports it -- because a
    strong special-purpose propagator would make the search count meaningless.
    """
    domains = problem.domains()
    changed = True
    while changed:
        changed = False
        for relation in problem.binary:
            for index in (relation.left, relation.right):
                doomed = {value for value in domains[index]
                          if not _binary_supported(relation, index, value, domains,
                                                   problem.universe)}
                if doomed:
                    domains[index] -= doomed
                    changed = True
                    if not domains[index]:
                        return None
        for constraint in problem.nary:
            for position, index in enumerate(constraint.indices):
                doomed = set()
                for value in domains[index]:
                    if not _nary_supported(constraint, position, value, domains,
                                           problem.universe):
                        doomed.add(value)
                if doomed:
                    domains[index] -= doomed
                    changed = True
                    # Bail the moment a domain empties rather than at the end of
                    # the sweep: the remaining filters read their partners'
                    # extremes, and an empty partner has none. L8.37.1 never hit
                    # this because its contradiction arrives through
                    # ALL_DIFFERENT, whose filter tolerates an empty domain;
                    # L8.41's does, through a unary observation under
                    # DOMINATES_SUM.
                    if not domains[index]:
                        return None
        if any(not domain for domain in domains):
            return None
    return domains


def _nary_supported(constraint: Nary, position: int, value: str,
                    domains: list[set[str]], universe) -> bool:
    """Could some choice for the other positions make this constraint hold?

    Checked against the domains' extremes rather than by enumeration, so the
    propagator stays cheap and the state count keeps its meaning.
    """
    if constraint.kind == ALL_DIFFERENT:
        others = [domains[i] for p, i in enumerate(constraint.indices) if p != position]
        # A value survives unless removing it leaves too few values to go round.
        available = set().union(*others) if others else set()
        return len(available - {value}) >= len(others)
    if constraint.kind == DOMINATES_SUM:
        head_index = constraint.indices[0]
        rest = [i for i in constraint.indices[1:]]
        if position == 0:
            top = _facts(universe, value).high
            floor = sum(min(_facts(universe, n).low for n in domains[i]) for i in rest)
            return top is not None and top > floor
        best_head = max((_facts(universe, n).high or 0.0) for n in domains[head_index])
        floor = _facts(universe, value).low
        others = sum(min(_facts(universe, n).low for n in domains[i])
                     for i in rest if i != constraint.indices[position])
        return best_head > floor + others
    return True


def solve(problem: Problem, cap: int = 3) -> tuple[list[tuple[str, ...]], int]:
    """Propagate, then search, counting every node the search actually enters."""
    states = 0
    domains = propagate(problem)
    if domains is None:
        return [], states

    order = sorted(range(problem.size), key=lambda i: len(domains[i]))
    found: list[tuple[str, ...]] = []

    def walk(assignment: dict[int, str]) -> None:
        nonlocal states
        if len(found) >= cap:
            return
        if len(assignment) == problem.size:
            candidate = tuple(assignment[i] for i in range(problem.size))
            if problem.satisfied(candidate):
                found.append(candidate)
            return
        index = next(i for i in order if i not in assignment)
        for value in sorted(domains[index]):
            states += 1
            assignment[index] = value
            if _partial_ok(problem, assignment):
                walk(assignment)
            del assignment[index]
            if len(found) >= cap:
                return

    walk({})
    return found, states


def _partial_ok(problem: Problem, assignment: dict[int, str]) -> bool:
    """Reject as early as a constraint can be evaluated, and not before."""
    for relation in problem.binary:
        if relation.left in assignment and relation.right in assignment:
            pair = [""] * problem.size
            for index, value in assignment.items():
                pair[index] = value
            if not holds(relation, tuple(pair), problem.universe):
                return False
    for constraint in problem.nary:
        if all(index in assignment for index in constraint.indices):
            full = [""] * problem.size
            for index, value in assignment.items():
                full[index] = value
            if not nary_holds(constraint, tuple(full), problem.universe):
                return False
        elif constraint.kind == ALL_DIFFERENT:
            picked = [assignment[i] for i in constraint.indices if i in assignment]
            if len(set(picked)) != len(picked):
                return False
    return True


# --------------------------------------------------------------------------
# the same three states as everywhere else
# --------------------------------------------------------------------------

def verdict(solutions: list[tuple[str, ...]]) -> str:
    if not solutions:
        return CONTRADICTION
    if len(solutions) == 1:
        return RESOLVED
    return AMBIGUOUS


def marginals(solutions: list[tuple[str, ...]], size: int) -> tuple[frozenset[str], ...]:
    return tuple(frozenset(s[i] for s in solutions) for i in range(size))


def pair_marginals(solutions: list[tuple[str, ...]], size: int) -> dict[tuple[int, int], int]:
    found: dict[tuple[int, int], int] = {}
    for left in range(size):
        for right in range(left + 1, size):
            found[(left, right)] = len({(s[left], s[right]) for s in solutions})
    return found
