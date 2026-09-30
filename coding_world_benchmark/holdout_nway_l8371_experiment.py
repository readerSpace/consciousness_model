"""L8.37.1: agree with brute force, then stop doing brute force.

Two phases, because succeeding at a size you could have enumerated proves
nothing about the method.

**Phase one, n = 3..5.**  The propagating solver is checked against an exhaustive
oracle for *exact* agreement -- same solutions, same verdict.  This is the phase
that says the solver is correct.

**Phase two, n = 6, 8, 10.**  The same problems grow past enumeration and only
the states visited are compared against |M|^n.  This is the phase that says the
solver is a solver.  At n=10 the product holds 10^10 assignments.

The families separate what is doing the work:

``INDEPENDENT``   each word's own observations pin it
``PAIRWISE_ONLY`` no word alone, but the binary relations finish it
``GLOBAL_ONLY``   **every** single-word marginal and **every** two-word marginal
                  is ambiguous, and the whole assignment is unique. Stated
                  precisely, since it is the claim that matters: with the n-ary
                  constraint removed the problem has several solutions whose
                  marginals are all plural, and restoring it leaves exactly one.
                  A world like that cannot be solved by being good at words or at
                  pairs.
``SYMMETRIC``     the constraints are invariant under permuting meanings, so the
                  solutions come in orbits and no amount of n or evidence breaks
                  the tie. Staying ambiguous is correct; resolving would mean
                  inventing the distinction.
``CONTRADICTION`` nothing satisfies them, reported rather than approximated.

``GLOBAL_ONLY`` runs on a ``dominant`` geometry rather than the ``disjoint`` one
the others use, and the reason is worth recording: with evenly spaced disjoint
ranges the sum constraint becomes unsatisfiable from n=5 -- the highest field's
maximum stops exceeding the sum of the others' minima -- so the family collapsed
into CONTRADICTION. That was the world outgrowing the constraint, not the solver
failing, and the fix is a layout where exactly one field can dominate at any n.

The invariant is ``false_joint_resolution_rate`` and, beside it, the fact that no
new rule was needed: ``|B| = 1`` executes, ``|B| = 0`` contradicts, ``|B| >= 2``
abstains, at every n. A threshold appearing at n=4 would have shown that L8.21's
uniqueness rule was a two-variable convenience.
"""
from __future__ import annotations

from dataclasses import dataclass

from .cross_context_identification_l829_experiment import (
    Observation, synthetic_universe, universe_from,
)
from .holdout_binding_l8211_experiment import build_holdout
from .holdout_lexicon_l8281_experiment import widen_store
from .joint_inference_l834_experiment import DISTINCT, EXCEEDS, Relation
from .nway_semantics_l837_experiment import (
    ALL_DIFFERENT, AMBIGUOUS, CONTRADICTION, DOMINATES_SUM, RESOLVED, Nary, Problem,
    exhaustive, marginals, pair_marginals, propagate, solve, verdict,
)

ORACLE_SIZES = (3, 4, 5)
SCALE_SIZES = (6, 8, 10)


def _equals(name: str) -> Observation:
    return Observation("EQUALS", name)


@dataclass(frozen=True)
class Family:
    name: str
    build: object  # (universe, n) -> Problem
    expect: str
    note: str


def _independent(universe, size):
    names = [f.name for f in universe]
    return Problem(universe, size,
                   unary={i: (_equals(names[i]),) for i in range(size)})


def _pairwise(universe, size):
    return Problem(universe, size,
                   binary=tuple(Relation(EXCEEDS, i, i + 1) for i in range(size - 1)),
                   nary=(Nary(ALL_DIFFERENT, tuple(range(size))),))


#: The layout ``GLOBAL_ONLY`` needs; see the module docstring.
GLOBAL_GEOMETRY = "dominant"


def _global_only(universe, size):
    return Problem(universe, size,
                   binary=tuple(Relation(EXCEEDS, i, i + 1) for i in range(1, size - 1)),
                   nary=(Nary(ALL_DIFFERENT, tuple(range(size))),
                         Nary(DOMINATES_SUM, tuple(range(size)))))


def _global_only_without_nary(universe, size):
    """The same world with the n-ary constraint removed, for the comparison."""
    return Problem(universe, size,
                   binary=tuple(Relation(EXCEEDS, i, i + 1) for i in range(1, size - 1)),
                   nary=(Nary(ALL_DIFFERENT, tuple(range(size))),))


def _symmetric(universe, size):
    return Problem(universe, size, nary=(Nary(ALL_DIFFERENT, tuple(range(size))),))


def _contradiction(universe, size):
    names = [f.name for f in universe]
    return Problem(universe, size,
                   unary={i: (_equals(names[0]),) for i in range(size)},
                   nary=(Nary(ALL_DIFFERENT, tuple(range(size))),))


def build_families() -> tuple[Family, ...]:
    return (
        Family("INDEPENDENT", _independent, RESOLVED,
               "each word's own observations pin it"),
        Family("PAIRWISE_ONLY", _pairwise, RESOLVED,
               "no word alone; the binary relations finish it"),
        Family("GLOBAL_ONLY", _global_only, RESOLVED,
               "every one- and two-word marginal ambiguous; the n-ary decides"),
        Family("SYMMETRIC", _symmetric, AMBIGUOUS,
               "permutation symmetry: staying ambiguous is the correct answer"),
        Family("CONTRADICTION", _contradiction, CONTRADICTION,
               "nothing satisfies them; report it"),
    )


def _ratio(numerator, denominator):
    if denominator == 0:
        return "n/a"
    return round(numerator / denominator, 3)


def run_oracle_phase() -> dict:
    """n = 3..5: exact agreement with brute force."""
    rows = []
    for size in ORACLE_SIZES:
        for family in build_families():
            universe = synthetic_universe(
                size, geometry=GLOBAL_GEOMETRY if family.name == "GLOBAL_ONLY"
                else "disjoint")
            problem = family.build(universe, size)
            found, states = solve(problem)
            truth, product_states = exhaustive(problem)
            rows.append({
                "family": family.name, "size": size,
                "verdict": verdict(found), "oracle_verdict": verdict(truth),
                "agree": found == truth and verdict(found) == verdict(truth),
                "expected": family.expect,
                "correct": verdict(found) == family.expect,
                "states": states, "product": product_states,
            })
    return {
        "rows": rows,
        "oracle_agreement": _ratio(sum(1 for r in rows if r["agree"]), len(rows)),
        "verdict_accuracy": _ratio(sum(1 for r in rows if r["correct"]), len(rows)),
    }


def run_scale_phase() -> dict:
    """n = 6, 8, 10: same answers, and nothing like the product visited."""
    rows = []
    for size in SCALE_SIZES:
        for family in build_families():
            universe = synthetic_universe(
                size, geometry=GLOBAL_GEOMETRY if family.name == "GLOBAL_ONLY"
                else "disjoint")
            problem = family.build(universe, size)
            found, states = solve(problem)
            product = len(universe) ** size
            rows.append({
                "family": family.name, "size": size,
                "verdict": verdict(found), "expected": family.expect,
                "correct": verdict(found) == family.expect,
                "states": states, "product": product,
                "fraction": states / product,
            })
    return {
        "rows": rows,
        "verdict_accuracy": _ratio(sum(1 for r in rows if r["correct"]), len(rows)),
        "worst_fraction": max(r["fraction"] for r in rows),
    }


def run_global_only_evidence() -> dict:
    """The GLOBAL_ONLY claim, stated as the comparison that establishes it."""
    rows = []
    for size in (3, 4, 5):
        universe = synthetic_universe(size, geometry=GLOBAL_GEOMETRY)
        without, _ = exhaustive(_global_only_without_nary(universe, size), cap=999)
        with_nary, _ = solve(_global_only(universe, size))
        rows.append({
            "size": size,
            "without_nary_solutions": len(without),
            "singleton_marginals": sum(
                1 for m in marginals(without, size) if len(m) == 1),
            "singleton_pairs": sum(
                1 for count in pair_marginals(without, size).values() if count == 1),
            "with_nary": verdict(with_nary),
            "global_only": (len(without) > 1
                            and all(len(m) > 1 for m in marginals(without, size))
                            and all(c > 1 for c in pair_marginals(without, size).values())
                            and verdict(with_nary) == RESOLVED),
        })
    return {"rows": rows,
            "holds_at_every_size": all(r["global_only"] for r in rows)}


def run_symmetry_pressure() -> dict:
    """More n, more evidence of the same symmetric kind. It must not break."""
    rows = []
    for size in (3, 4, 5, 6):
        universe = synthetic_universe(size, geometry="disjoint")
        base = _symmetric(universe, size)
        extra = Problem(universe, size, nary=(
            Nary(ALL_DIFFERENT, tuple(range(size))),
            Nary(ALL_DIFFERENT, tuple(range(size - 1))),
            Nary(ALL_DIFFERENT, tuple(range(1, size))),
        ))
        rows.append({
            "size": size,
            "base": verdict(solve(base)[0]),
            "with_more_evidence": verdict(solve(extra)[0]),
        })
    return {"rows": rows,
            "never_broken": all(r["base"] == r["with_more_evidence"] == AMBIGUOUS
                                for r in rows)}


def run_real_universe() -> dict:
    """The same shapes on the three fields the store actually records."""
    universe = universe_from(widen_store(build_holdout(24)[0]))
    problem = _global_only(universe, 3)
    found, states = solve(problem)
    truth, product = exhaustive(problem)
    return {
        "meanings": [f.name for f in universe],
        "solution": found[0] if found else None,
        "verdict": verdict(found),
        "agrees_with_oracle": found == truth,
        "states": states, "product": product,
    }


def run_experiment() -> dict:
    oracle = run_oracle_phase()
    scale = run_scale_phase()
    rows = oracle["rows"] + scale["rows"]
    wrong = sum(1 for r in rows
                if r["verdict"] == RESOLVED and r["expected"] != RESOLVED)
    return {
        "oracle": oracle, "scale": scale,
        "global_only": run_global_only_evidence(),
        "symmetry": run_symmetry_pressure(),
        "real": run_real_universe(),
        "metrics": {
            "false_joint_resolution_rate": _ratio(wrong, len(rows)),
            "oracle_agreement": oracle["oracle_agreement"],
            "verdict_accuracy": _ratio(sum(1 for r in rows if r["correct"]), len(rows)),
            "worst_states_fraction": scale["worst_fraction"],
            "symmetry_broken_rate": 0.0 if run_symmetry_pressure()["never_broken"] else 1.0,
        },
    }


def format_experiment(report) -> str:
    found = report["metrics"]
    lines = ["## N-way joint semantics (L8.37.1)", "",
             "| metric | value |", "| --- | --- |",
             f"| **false_joint_resolution_rate** | **{found['false_joint_resolution_rate']}** |",
             f"| **oracle_agreement (n=3..5)** | **{found['oracle_agreement']}** |",
             f"| verdict_accuracy | {found['verdict_accuracy']} |",
             f"| **symmetry_broken_rate** | **{found['symmetry_broken_rate']}** |",
             f"| worst states / \\|M\\|^n | {found['worst_states_fraction']:.2e} |",
             "", "### phase two: the same answers without the product", "",
             "| family | n | \\|M\\|^n | states visited | verdict |",
             "| --- | --- | --- | --- | --- |"]
    for row in report["scale"]["rows"]:
        mark = "" if row["correct"] else " (NG)"
        lines.append(f"| {row['family']} | {row['size']} | {row['product']:,} "
                     f"| {row['states']} | {row['verdict']}{mark} |")

    lines += ["", "### the GLOBAL_ONLY claim, stated as its comparison", "",
              "| n | solutions without the n-ary | singleton 1-marginals "
              "| singleton 2-marginals | with it |", "| --- | --- | --- | --- | --- |"]
    for row in report["global_only"]["rows"]:
        lines.append(f"| {row['size']} | {row['without_nary_solutions']} "
                     f"| {row['singleton_marginals']} | {row['singleton_pairs']} "
                     f"| {row['with_nary']} |")
    lines.append("")
    lines.append("Every one- and two-word marginal is plural without the n-ary "
                 "constraint, and the whole assignment is unique with it. Being good "
                 "at words or at pairs cannot solve that world.")

    lines += ["", "### symmetry under pressure", "",
              "| n | ALL_DIFFERENT alone | plus more of the same evidence |",
              "| --- | --- | --- |"]
    for row in report["symmetry"]["rows"]:
        lines.append(f"| {row['size']} | {row['base']} | {row['with_more_evidence']} |")

    real = report["real"]
    lines += ["", "### on the three fields the store records", "",
              f"- meanings: {real['meanings']}",
              f"- solution: `{real['solution']}` ({real['verdict']}), "
              f"agrees with the oracle: {real['agrees_with_oracle']}",
              f"- {real['states']} states against a product of {real['product']}"]
    return "\n".join(lines)


def main() -> None:  # pragma: no cover
    print(format_experiment(run_experiment()))


if __name__ == "__main__":  # pragma: no cover
    main()
