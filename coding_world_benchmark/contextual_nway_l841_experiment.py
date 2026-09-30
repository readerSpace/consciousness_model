"""L8.41: n unknown words whose admissible *combinations* are selected by the context.

L8.37 solved n words at once and L8.38 let the context choose which pairs were
admissible.  Running both is not the same as needing both, and a suite that only
ran them side by side would show two capabilities standing next to each other.
The world this layer is built around needs them **simultaneously**::

    B_c = { b in M^n : shared constraints,  and the context's own n-ary one }

with execution still gated per context on ``|B_c| = 1`` -- no new rule, which is
the same thing L8.37 was testing when n grew and L8.38 when contexts appeared.

The decisive family is ``CONTEXT_GLOBAL_ONLY`` and it is defined by what it
denies, so each denial is measured rather than asserted:

* **no single word is determined** and **no pair is determined** once the
  non-decomposable n-ary constraint is removed -- 「1つ目が残りの合計より大きい」
  relates all n at once and decomposes into no set of binary constraints, which is
  L8.37's reason for using it;
* **erasing the context leaves it undetermined** -- each episode resolves to a
  *different* assignment, so the union an unlabelled account is left with has two,
  which is L8.38's erasure ablation lifted from pairs to n;
* **with both present each episode is unique**;
* **one contradicting observation empties that episode** and is reported rather
  than repaired.

Two axes, four cells, and the controls occupy the other three: a world the pairs
finish without any n-ary constraint, a world the n-ary constraint finishes
identically in every context, and a symmetric world nothing finishes.  A headline
family alone would not distinguish "needs both" from "needs whatever it is given".
"""
from __future__ import annotations

from dataclasses import dataclass, replace

from .cross_context_identification_l829_experiment import FieldFacts, Observation
from .nway_semantics_l837_experiment import (
    AMBIGUOUS, CONTRADICTION, DOMINATES_SUM, RESOLVED, Nary, Problem, exhaustive,
    marginals, pair_marginals, solve, verdict,
)

Assignment = tuple[str, ...]


@dataclass(frozen=True)
class ContextualProblem:
    """One constraint problem per context, over the same n surfaces.

    The contexts share a universe and a number of surfaces and nothing else: what
    an episode selects is which assignments are admissible in it, and the most
    direct way to say that is to let each episode carry its own constraints.
    """

    size: int
    problems: tuple[tuple[str, Problem], ...]

    @property
    def contexts(self) -> tuple[str, ...]:
        return tuple(label for label, _ in self.problems)

    def problem(self, context: str) -> Problem:
        return dict(self.problems)[context]

    def solve_in(self, context: str, cap: int = 3):
        return solve(self.problem(context), cap)

    def solutions(self, cap: int = 3) -> dict[str, list[Assignment]]:
        return {context: self.solve_in(context, cap)[0] for context in self.contexts}

    def states(self, cap: int = 3) -> int:
        return sum(self.solve_in(context, cap)[1] for context in self.contexts)

    def verdict_in(self, context: str, cap: int = 3) -> str:
        return verdict(self.solve_in(context, cap)[0])

    def meanings_in(self, context: str) -> Assignment | None:
        found = self.solve_in(context)[0]
        return found[0] if len(found) == 1 else None

    @property
    def all_resolved(self) -> bool:
        return all(self.verdict_in(context) == RESOLVED for context in self.contexts)

    # -- the two ablations ---------------------------------------------
    def erased(self, cap: int = 3) -> list[Assignment]:
        """The same evidence with the episode labels removed.

        The union, for L8.38's reason: with no label there is nothing to say
        which episode an assignment belonged to, so all of them remain on the
        table.  When the episodes resolve to *different* assignments this is
        exactly where the information the labels carried shows up as a loss.
        """
        found: list[Assignment] = []
        for context in self.contexts:
            for assignment in self.solve_in(context, cap)[0]:
                if assignment not in found:
                    found.append(assignment)
        return found

    def erased_verdict(self, cap: int = 3) -> str:
        return verdict(self.erased(cap))

    def without(self, kind: str = DOMINATES_SUM) -> "ContextualProblem":
        """The same world with one kind of n-ary constraint dropped everywhere."""
        return ContextualProblem(self.size, tuple(
            (label, replace(problem, nary=tuple(
                constraint for constraint in problem.nary
                if constraint.kind != kind)))
            for label, problem in self.problems))

    def observed(self, context: str, index: int,
                 observation: Observation) -> "ContextualProblem":
        return ContextualProblem(self.size, tuple(
            (label, replace(problem, unary={
                **problem.unary,
                index: problem.unary.get(index, ()) + (observation,)})
             if label == context else problem)
            for label, problem in self.problems))

    # -- the oracle ----------------------------------------------------
    def exhaustive(self, cap: int = 3) -> dict[str, list[Assignment]]:
        return {context: exhaustive(self.problem(context), cap)[0]
                for context in self.contexts}

    def product_size(self) -> int:
        return len(self.problems[0][1].meanings) ** self.size


@dataclass(frozen=True)
class Analysis:
    """What each ingredient is doing, as four verdicts rather than one claim."""

    full: dict[str, str]
    without_nary: dict[str, str]
    erased: str
    erased_without_nary: str
    unary_marginals_ambiguous: bool
    pair_marginals_ambiguous: bool

    @property
    def resolved(self) -> bool:
        return all(state == RESOLVED for state in self.full.values())

    @property
    def context_needed(self) -> bool:
        """Every episode unique, and the unlabelled union is not."""
        return self.resolved and self.erased != RESOLVED

    @property
    def nary_needed(self) -> bool:
        """Every episode unique, and dropping the n-ary constraint loses that."""
        return self.resolved and any(state != RESOLVED
                                     for state in self.without_nary.values())

    @property
    def both_needed(self) -> bool:
        return self.context_needed and self.nary_needed


def analyse(problem: ContextualProblem, cap: int = 3) -> Analysis:
    stripped = problem.without()
    per_context = {c: problem.verdict_in(c, cap) for c in problem.contexts}
    without = {c: stripped.verdict_in(c, cap) for c in problem.contexts}

    # "No single word decides it, no pair decides it" is a claim about what the
    # lower-arity constraints can reach, so it is read off the world with the
    # n-ary one removed -- with it in place every episode is a singleton and the
    # marginals are trivially unique, which would prove nothing.
    unary_ambiguous, pair_ambiguous = True, True
    for context in problem.contexts:
        found = stripped.solve_in(context, cap=64)[0]
        if not found:
            continue
        if any(len(m) < 2 for m in marginals(found, problem.size)):
            unary_ambiguous = False
        if any(count < 2 for count in pair_marginals(found, problem.size).values()):
            pair_ambiguous = False

    return Analysis(per_context, without, problem.erased_verdict(cap),
                    stripped.erased_verdict(cap), unary_ambiguous, pair_ambiguous)


def agrees_with_oracle(problem: ContextualProblem, cap: int = 3) -> bool:
    """Propagation and brute force, compared per context and exactly."""
    solved, oracle = problem.solutions(cap), problem.exhaustive(cap)
    return all(sorted(solved[c]) == sorted(oracle[c]) for c in problem.contexts)
