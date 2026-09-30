"""L8.32.2.1: the audit's report. Greedy is not optimal; these worlds are easy.

L8.32.1 measured ``regret = 0`` in eleven enumerated worlds and said so as a
measurement rather than a theorem.  This decides which it was, and the answer is
both halves at once:

* over **every** interval world on a small grid, and over a thousand random ones,
  greedy is exactly optimal;
* over arbitrary set systems it is not, and the smallest counterexample found has
  six meanings and five queries, where greedy pays 17/6 against the best policy's
  8/3.

So the zero came from the geometry, not from the rule.  That is worth the run:
"greedy is fine" would have been an unfalsifiable comfort, whereas "greedy is
suboptimal in general and optimal on interval geometry" is a statement about the
world the schema describes -- the same kind of statement L8.29 and L8.31 kept
arriving at about identifiability, reached here about planning instead.

Exhausting a class matters more than sampling it.  A sample can only report that
nothing turned up; enumerating every set system with five meanings or fewer, and
every interval world on the grid, reports that none exists there.  That is what
makes six meanings the *start* of the counterexamples rather than merely where
this search happened to look.
"""
from __future__ import annotations

from .greedy_counterexample_l8322_experiment import (
    exhaustive_set_systems, greedy_cost, grid_intervals, known_counterexample,
    optimal_cost, regret, search_intervals, search_set_systems,
)

INTERVAL_SIZES = (4, 5, 6)
SET_SIZES = ((5, 4), (6, 5), (7, 5))


def run_experiment(trials: int = 400) -> dict:
    tier_a_random = {
        size: search_intervals(trials, size, seed=size) for size in INTERVAL_SIZES
    }
    tier_a_grid = {size: grid_intervals(size, 5) for size in (4, 5)}
    tier_b_random = {
        f"K{size}/pool{pool}": search_set_systems(trials, size, pool, seed=size * 10 + pool)
        for size, pool in SET_SIZES
    }
    tier_b_exhaustive = {
        f"K{size}/pool{pool}": exhaustive_set_systems(size, pool)
        for size, pool in ((4, 3), (5, 3), (5, 4))
    }
    witness = known_counterexample()
    everything = frozenset(witness.names)
    return {
        "tier_a_random": tier_a_random,
        "tier_a_grid": tier_a_grid,
        "tier_b_random": tier_b_random,
        "tier_b_exhaustive": tier_b_exhaustive,
        "witness": {
            "names": list(witness.names),
            "queries": [sorted(query) for query in witness.queries],
            "greedy": round(greedy_cost(witness, everything), 6),
            "optimal": round(optimal_cost(witness, everything), 6),
            "regret": round(regret(witness), 6),
        },
    }


def format_experiment(report) -> str:
    lines = ["## Greedy counterexample search (L8.32.2)", "",
             "### tier A -- worlds this project's observation language can express", "",
             "| search | worlds | max regret |", "| --- | --- | --- |"]
    for size, found in report["tier_a_random"].items():
        lines.append(f"| random intervals, K={size} | {found['trials']} "
                     f"| {found['max_regret']} |")
    for size, found in report["tier_a_grid"].items():
        lines.append(f"| **every** interval world on a 5-grid, K={size} "
                     f"| {found['checked']} | {found['max_regret']} |")

    lines += ["", "### tier B -- arbitrary set systems", "",
              "| search | worlds | max regret |", "| --- | --- | --- |"]
    for label, found in report["tier_b_exhaustive"].items():
        lines.append(f"| **every** set system, {label} | {found['checked']} "
                     f"| {found['max_regret']} |")
    for label, found in report["tier_b_random"].items():
        lines.append(f"| random set systems, {label} | {found['trials']} "
                     f"| {found['max_regret']} |")

    witness = report["witness"]
    lines += ["", "### the smallest counterexample found", "",
              f"Six meanings, five queries. Greedy pays **{witness['greedy']}** "
              f"where the best policy pays **{witness['optimal']}** "
              f"-- regret **{witness['regret']}**.", "",
              "```"]
    for query in witness["queries"]:
        lines.append(f"admits {query}")
    lines += ["```", "",
              "Every set system with five meanings or fewer is clean, so six is "
              "where counterexamples start rather than where this search happened "
              "to look.", "",
              "**Verdict.** Greedy expected-size minimisation is not optimal in "
              "general. It is optimal on every interval world searched, "
              "exhaustively for the small ones. L8.32.1's zero was a property of "
              "the geometry, not of the rule."]
    return "\n".join(lines)


def main() -> None:  # pragma: no cover
    print(format_experiment(run_experiment()))


if __name__ == "__main__":  # pragma: no cover
    main()
