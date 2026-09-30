"""L8.41.1: the world that needs the n-ary constraint *and* the episode label.

``CONTEXT_GLOBAL_ONLY`` is built so that each ingredient can be taken away and
the loss measured.  The episodes share everything except which surface is the one
「残りの合計より大きい」 -- in ``sim`` it is the first, in ``lab`` the second -- and
that single difference is enough to make the two episodes resolve to *different*
assignments.  So:

* remove the n-ary constraint and every episode goes ambiguous (the binary chain
  cannot say which surface is the dominant one);
* remove the episode labels and the union of two different unique answers is
  ambiguous;
* keep both and each episode is unique;
* add one observation that contradicts an episode's answer and that episode is
  empty, reported rather than repaired.

The other three cells of the same 2x2 are occupied by controls, because a suite
containing only the headline cannot tell "needs both" from "uses whatever it is
handed".  ``CONTEXT_PAIRWISE`` needs the labels and no n-ary constraint at all,
``NARY_ONLY`` is L8.37's world with the episodes agreeing, and ``NEITHER`` is
pinned by each surface's own observations.  ``SYMMETRIC`` resolves in no episode
however much is added, which is the correct answer and the safety case.

The truth is not read back from the solver.  For this family it is stated in
closed form -- the dominant surface takes the largest field and the rest descend
-- and at n = 3, 4, 5 the propagating solver is additionally checked against an
exhaustive oracle per episode, exactly as L8.37 did before trusting itself at
larger n.
"""
from __future__ import annotations

from dataclasses import dataclass

from .contextual_nway_l841_experiment import (
    Analysis, ContextualProblem, agrees_with_oracle, analyse,
)
from .cross_context_identification_l829_experiment import (
    Observation, synthetic_universe,
)
from .joint_inference_l834_experiment import EXCEEDS, Relation
from .nway_semantics_l837_experiment import (
    ALL_DIFFERENT, AMBIGUOUS, CONTRADICTION, DOMINATES_SUM, RESOLVED, Nary,
    Problem,
)

ORACLE_SIZES = (3, 4, 5)
SCALE_SIZES = (6, 8, 10)
GEOMETRY = "dominant"
SIM, LAB = "sim", "lab"


def _equals(name: str) -> Observation:
    return Observation("EQUALS", name)


def _dominant(universe, size: int, head: int) -> Problem:
    """One surface dominates the sum of the rest; the rest descend in order."""
    rest = tuple(index for index in range(size) if index != head)
    return Problem(
        universe, size,
        binary=tuple(Relation(EXCEEDS, rest[k], rest[k + 1])
                     for k in range(len(rest) - 1)),
        nary=(Nary(ALL_DIFFERENT, tuple(range(size))),
              Nary(DOMINATES_SUM, (head,) + rest)))


def expected_assignment(universe, size: int, head: int) -> tuple[str, ...]:
    """Stated in closed form, not read back from the solver.

    Only the largest field can exceed the sum of the others' minima in this
    geometry, so the dominant surface takes it; the descending chain then forces
    the remaining surfaces onto the remaining fields in order.
    """
    names = [facts.name for facts in universe][:size]
    rest = [index for index in range(size) if index != head]
    found = [""] * size
    found[head] = names[-1]
    for position, index in enumerate(rest):
        found[index] = names[size - 2 - position]
    return tuple(found)


# --------------------------------------------------------------------------
# the families
# --------------------------------------------------------------------------

def _context_global_only(universe, size) -> ContextualProblem:
    return ContextualProblem(size, ((SIM, _dominant(universe, size, 0)),
                                    (LAB, _dominant(universe, size, 1))))


def _nary_only(universe, size) -> ContextualProblem:
    return ContextualProblem(size, ((SIM, _dominant(universe, size, 0)),
                                    (LAB, _dominant(universe, size, 0))))


def _chain(universe, size, ascending: bool) -> Problem:
    pairs = [(i + 1, i) if ascending else (i, i + 1) for i in range(size - 1)]
    return Problem(universe, size,
                   binary=tuple(Relation(EXCEEDS, left, right)
                                for left, right in pairs),
                   nary=(Nary(ALL_DIFFERENT, tuple(range(size))),))


def _context_pairwise(universe, size) -> ContextualProblem:
    return ContextualProblem(size, ((SIM, _chain(universe, size, False)),
                                    (LAB, _chain(universe, size, True))))


def _neither(universe, size) -> ContextualProblem:
    names = [facts.name for facts in universe]
    pinned = Problem(universe, size,
                     unary={i: (_equals(names[i]),) for i in range(size)})
    return ContextualProblem(size, ((SIM, pinned), (LAB, pinned)))


def _symmetric(universe, size) -> ContextualProblem:
    loose = Problem(universe, size,
                    nary=(Nary(ALL_DIFFERENT, tuple(range(size))),))
    return ContextualProblem(size, ((SIM, loose), (LAB, loose)))


@dataclass(frozen=True)
class Family:
    name: str
    build: object
    expect_context: bool
    expect_nary: bool
    expect_state: str
    note: str


def build_families() -> tuple[Family, ...]:
    return (
        Family("CONTEXT_GLOBAL_ONLY", _context_global_only, True, True, RESOLVED,
               "neither the pairs nor the unlabelled evidence reach it"),
        Family("CONTEXT_PAIRWISE", _context_pairwise, True, False, RESOLVED,
               "control: the labels matter and no n-ary constraint is used"),
        Family("NARY_ONLY", _nary_only, False, True, RESOLVED,
               "control: L8.37's world, with the episodes agreeing"),
        Family("NEITHER", _neither, False, False, RESOLVED,
               "control: each surface's own observations pin it"),
        Family("SYMMETRIC", _symmetric, False, False, AMBIGUOUS,
               "safety: a permutation symmetry is not broken by trying harder"),
    )


def _ratio(numerator, denominator):
    if denominator == 0:
        return "n/a"
    return round(numerator / denominator, 3)


def run_family(family: Family, size: int) -> dict:
    universe = synthetic_universe(size, geometry=GEOMETRY)
    problem = family.build(universe, size)
    found: Analysis = analyse(problem)

    truth, wrong = None, False
    if family.name in ("CONTEXT_GLOBAL_ONLY", "NARY_ONLY"):
        heads = {SIM: 0, LAB: 1 if family.name == "CONTEXT_GLOBAL_ONLY" else 0}
        truth = {c: expected_assignment(universe, size, head)
                 for c, head in heads.items()}
        wrong = any(problem.meanings_in(c) is not None
                    and problem.meanings_in(c) != truth[c] for c in truth)

    oracle = agrees_with_oracle(problem) if size in ORACLE_SIZES else None

    return {
        "name": family.name, "size": size, "note": family.note,
        "full": found.full, "without_nary": found.without_nary,
        "erased": found.erased, "erased_without_nary": found.erased_without_nary,
        "context_needed": found.context_needed,
        "nary_needed": found.nary_needed,
        "both_needed": found.both_needed,
        "cell_correct": (found.context_needed == family.expect_context
                         and found.nary_needed == family.expect_nary),
        "state_correct": all(state == family.expect_state
                             for state in found.full.values()),
        "unary_marginals_ambiguous": found.unary_marginals_ambiguous,
        "pair_marginals_ambiguous": found.pair_marginals_ambiguous,
        "wrong": wrong, "truth": truth,
        "oracle_agreement": oracle,
        "states": problem.states(), "product": problem.product_size(),
    }


def run_contradiction(size: int) -> dict:
    """One observation against an episode's own answer empties that episode."""
    universe = synthetic_universe(size, geometry=GEOMETRY)
    problem = _context_global_only(universe, size)
    truth = expected_assignment(universe, size, 0)
    other = next(facts.name for facts in universe if facts.name != truth[0])
    broken = problem.observed(SIM, 0, _equals(other))
    return {
        "size": size,
        "sim": broken.verdict_in(SIM), "lab": broken.verdict_in(LAB),
        "correct": (broken.verdict_in(SIM) == CONTRADICTION
                    and broken.verdict_in(LAB) == RESOLVED),
        "resolved_anyway": broken.meanings_in(SIM) is not None,
    }


def run_experiment() -> dict:
    rows = [run_family(family, size)
            for size in ORACLE_SIZES + SCALE_SIZES
            for family in build_families()]
    contradictions = [run_contradiction(size) for size in ORACLE_SIZES]
    headline = [r for r in rows if r["name"] == "CONTEXT_GLOBAL_ONLY"]
    oracle_rows = [r for r in rows if r["oracle_agreement"] is not None]

    return {
        "rows": rows,
        "contradictions": contradictions,
        "metrics": {
            "false_joint_resolution_rate": _ratio(
                sum(1 for r in rows if r["wrong"]), len(rows)),
            "cell_accuracy": _ratio(
                sum(1 for r in rows if r["cell_correct"]), len(rows)),
            "state_accuracy": _ratio(
                sum(1 for r in rows if r["state_correct"]), len(rows)),
            "oracle_agreement": _ratio(
                sum(1 for r in oracle_rows if r["oracle_agreement"]),
                len(oracle_rows)),
            "both_needed_at_every_n": all(r["both_needed"] for r in headline),
            "no_unary_suffices": all(r["unary_marginals_ambiguous"]
                                     for r in headline),
            "no_pair_suffices": all(r["pair_marginals_ambiguous"]
                                    for r in headline),
            "contradiction_reported": all(r["correct"] for r in contradictions),
            "cells_occupied": len({(r["context_needed"], r["nary_needed"])
                                   for r in rows if r["state_correct"]
                                   and r["full"][SIM] == RESOLVED}),
            "headline_states_over_product": max(
                round(r["states"] / r["product"], 10) for r in headline),
            "worst_states_over_product": max(
                round(r["states"] / r["product"], 4) for r in rows),
            "worst_states_family": max(
                rows, key=lambda r: r["states"] / r["product"])["name"],
            "largest_product": max(r["product"] for r in rows),
        },
    }


def format_experiment(report) -> str:
    found = report["metrics"]
    lines = ["## Context-conditioned n-way semantics (L8.41.1)", "",
             "| metric | value |", "| --- | --- |",
             f"| **false_joint_resolution_rate** | **{found['false_joint_resolution_rate']}** |",
             f"| **both_needed at every n** | **{found['both_needed_at_every_n']}** |",
             f"| no_unary_suffices | {found['no_unary_suffices']} |",
             f"| no_pair_suffices | {found['no_pair_suffices']} |",
             f"| cell_accuracy | {found['cell_accuracy']} |",
             f"| state_accuracy | {found['state_accuracy']} |",
             f"| oracle_agreement (n=3,4,5) | {found['oracle_agreement']} |",
             f"| contradiction_reported | {found['contradiction_reported']} |",
             f"| cells of the 2x2 occupied | {found['cells_occupied']}/4 |",
             f"| states / \\|M\\|^n on the headline family "
             f"| {found['headline_states_over_product']} "
             f"(largest product {found['largest_product']}) |",
             f"| worst over all families | {found['worst_states_over_product']} "
             f"({found['worst_states_family']}) |",
             "", "### the ablation, on CONTEXT_GLOBAL_ONLY", "",
             "| n | removed | sim | lab | unlabelled union |",
             "| --- | --- | --- | --- | --- |"]
    for row in [r for r in report["rows"] if r["name"] == "CONTEXT_GLOBAL_ONLY"]:
        lines.append(f"| {row['size']} | nothing | {row['full'][SIM]} "
                     f"| {row['full'][LAB]} | {row['erased']} |")
        lines.append(f"| {row['size']} | the n-ary constraint "
                     f"| {row['without_nary'][SIM]} | {row['without_nary'][LAB]} "
                     f"| {row['erased_without_nary']} |")
    for run in report["contradictions"]:
        lines.append(f"| {run['size']} | (one contradicting observation in sim) "
                     f"| {run['sim']} | {run['lab']} | - |")

    lines += ["", "### the four cells", "",
              "| family | context needed | n-ary needed | state | ok |",
              "| --- | --- | --- | --- | --- |"]
    seen = set()
    for row in report["rows"]:
        if row["name"] in seen:
            continue
        seen.add(row["name"])
        lines.append(f"| {row['name']} | {row['context_needed']} "
                     f"| {row['nary_needed']} | {row['full'][SIM]} "
                     f"| {row['cell_correct'] and row['state_correct']} |")

    lines += ["", "### the solver, at each n", "",
              "| n | states | \\|M\\|^n | oracle agreement |",
              "| --- | --- | --- | --- |"]
    for row in [r for r in report["rows"] if r["name"] == "CONTEXT_GLOBAL_ONLY"]:
        agreement = "-" if row["oracle_agreement"] is None else row["oracle_agreement"]
        lines.append(f"| {row['size']} | {row['states']} | {row['product']} "
                     f"| {agreement} |")
    lines += ["",
              f"`{found['worst_states_family']}` is the honest counterweight to "
              f"the column above: with nothing to propagate the search enters "
              f"more nodes than the product has points at n=3, because a node is "
              f"a *partial* assignment. Propagation buys nothing where there is "
              f"no constraint to propagate, and the ratio on the headline family "
              f"is a statement about that world rather than about the solver.",
              "",
              "The gate is unchanged at every n and in every episode: `|B_c| = 1` "
              "executes, `0` contradicts, `>= 2` abstains. No rule was added for "
              "contexts and none for n."]
    return "\n".join(lines)


def main() -> None:  # pragma: no cover
    print(format_experiment(run_experiment()))


if __name__ == "__main__":  # pragma: no cover
    main()
