"""L8.32.2: is greedy optimal, or were those worlds easy?

L8.32.1 enumerated eleven decision trees exactly and found ``regret = 0`` in every
one.  That was reported as a measurement about those worlds rather than a
property of the rule, and this layer is the audit that decides which it was.  It
adds no capability.  Its only output is a fact about the previous layer.

The search maximises

    R = E[T_greedy] - E[T_optimal]

over generated worlds, and it is run at two tiers because the interesting answer
is different at each.

**Tier A -- worlds this project can actually express.**  Meanings are intervals
and queries are the RANGE and CONTRAST probes L8.29 generates from them.  A
counterexample here would mean the agent as built can be led astray.  A clean
sweep here means something narrower and more useful than "greedy is fine": it
means *interval* geometry is well behaved, which is a claim about the world the
schema describes.

**Tier B -- arbitrary set systems.**  A query is any subset of the meanings, with
no geometric constraint at all.  Minimising expected size one step at a time is
known not to be optimal for general split problems, so a counterexample is
expected here.  Finding one is what turns Tier A's clean sweep into a real
result: it shows the zero came from the geometry and not from the rule, which is
exactly the distinction L8.29 and L8.31 kept having to make about identifiability.

The abstract search runs on its own tiny reimplementation of the two policies,
because running the audit through the code being audited would prove nothing
about the code.  ``test_the_abstract_search_agrees_with_the_real_planner`` pins
the two together on interval worlds so the generalisation is faithful rather than
merely similar.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from itertools import combinations

from .cross_context_identification_l829_experiment import (
    FieldFacts, available_probes, constrain,
)
from .typed_operation_l819_experiment import ValueType


#: The smallest counterexample this search found, pinned so the claim does not
#: depend on a seed.  Six meanings, five queries: greedy pays 17/6 where the best
#: policy pays 8/3.  Exhausting every set system with five meanings or fewer
#: finds none, so six is where it starts.
KNOWN_COUNTEREXAMPLE: tuple[tuple[str, ...], tuple[tuple[str, ...], ...]] = (
    ("m0", "m1", "m2", "m3", "m4", "m5"),
    (("m0", "m1", "m2", "m3", "m5"),
     ("m0", "m1", "m3", "m4"),
     ("m1", "m2", "m3", "m4"),
     ("m1", "m3", "m4", "m5"),
     ("m2", "m3")),
)


@dataclass(frozen=True)
class SetWorld:
    """Meanings, and the subset each query admits.  No geometry assumed."""

    names: tuple[str, ...]
    queries: tuple[frozenset[str], ...]

    def branches(self, candidates: frozenset[str], query: frozenset[str]):
        return candidates & query, candidates - query


def expected_size(world: SetWorld, candidates: frozenset[str], query) -> float:
    yes, no = world.branches(candidates, query)
    return (len(yes) ** 2 + len(no) ** 2) / len(candidates)


def greedy_cost(world: SetWorld, candidates: frozenset[str]) -> float:
    """E[T] when each step minimises the expected surviving size."""
    if len(candidates) <= 1:
        return 0.0
    usable = [q for q in world.queries if world.branches(candidates, q)[0]
              and world.branches(candidates, q)[1]]
    if not usable:
        return float("inf")
    chosen = min(
        ((expected_size(world, candidates, q), index, q) for index, q in enumerate(usable)),
        key=lambda item: (item[0], item[1]))[2]
    yes, no = world.branches(candidates, chosen)
    return 1.0 + (len(yes) * greedy_cost(world, yes)
                  + len(no) * greedy_cost(world, no)) / len(candidates)


def optimal_cost(world: SetWorld, candidates: frozenset[str],
                 memo: dict | None = None) -> float:
    memo = {} if memo is None else memo
    if len(candidates) <= 1:
        return 0.0
    if candidates in memo:
        return memo[candidates]
    memo[candidates] = float("inf")
    best = float("inf")
    for query in world.queries:
        yes, no = world.branches(candidates, query)
        if not yes or not no:
            continue
        best = min(best, 1.0 + (len(yes) * optimal_cost(world, yes, memo)
                                + len(no) * optimal_cost(world, no, memo)) / len(candidates))
    memo[candidates] = best
    return best


def regret(world: SetWorld) -> float:
    """R = E[T_greedy] - E[T_optimal], or 0.0 when the world is unresolvable."""
    everything = frozenset(world.names)
    optimal = optimal_cost(world, everything)
    if optimal == float("inf"):
        return 0.0  # nothing to compare: neither policy resolves it
    return greedy_cost(world, everything) - optimal


# --------------------------------------------------------------------------
# tier A: worlds this project's observation language can express
# --------------------------------------------------------------------------

def known_counterexample() -> SetWorld:
    names, queries = KNOWN_COUNTEREXAMPLE
    return SetWorld(names, tuple(frozenset(q) for q in queries))


def interval_world(bounds: tuple[tuple[float, float], ...]) -> SetWorld:
    """An interval universe, with exactly the probes L8.29 would generate for it."""
    universe = tuple(
        FieldFacts(f"m{index}", ValueType.NUMBER, low, high)
        for index, (low, high) in enumerate(bounds))
    names = tuple(facts.name for facts in universe)
    queries = []
    for probe in available_probes(universe):
        admitted = constrain(probe, universe)
        if admitted and admitted != frozenset(names):
            queries.append(admitted)
    return SetWorld(names, tuple(dict.fromkeys(queries)))


def search_intervals(trials: int, size: int, seed: int = 0) -> dict:
    """Random interval universes, scored for regret."""
    rng = random.Random(seed)
    worst, found = 0.0, []
    for _ in range(trials):
        bounds = []
        for _ in range(size):
            low = round(rng.uniform(0.0, 10.0), 2)
            bounds.append((low, round(low + rng.uniform(0.5, 6.0), 2)))
        world = interval_world(tuple(bounds))
        value = regret(world)
        if value > 1e-9:
            found.append({"bounds": bounds, "regret": round(value, 6)})
        worst = max(worst, value)
    return {"trials": trials, "size": size, "max_regret": round(worst, 6),
            "counterexamples": found}


# --------------------------------------------------------------------------
# tier B: arbitrary set systems
# --------------------------------------------------------------------------

def grid_intervals(size: int, grid: int = 5) -> dict:
    """Every interval world on a small integer grid, not a sample of them.

    Sampling can only ever say "we did not happen to find one".  Exhausting a
    class says the class contains none, which is the statement Tier A needs if it
    is to mean anything against Tier B's counterexample.
    """
    spans = [(low, high) for low in range(grid) for high in range(low + 1, grid + 1)]
    worst, witness, checked = 0.0, None, 0
    for bounds in combinations(spans, size):
        checked += 1
        value = regret(interval_world(tuple((float(a), float(b)) for a, b in bounds)))
        if value > worst:
            worst, witness = value, bounds
    return {"size": size, "grid": grid, "checked": checked,
            "max_regret": round(worst, 6), "witness": witness}


def search_set_systems(trials: int, size: int, queries: int, seed: int = 0) -> dict:
    rng = random.Random(seed)
    names = tuple(f"m{index}" for index in range(size))
    worst, best_world, found = 0.0, None, []
    for _ in range(trials):
        pool = set()
        while len(pool) < queries:
            subset = frozenset(name for name in names if rng.random() < 0.5)
            if subset and subset != frozenset(names):
                pool.add(subset)
        world = SetWorld(names, tuple(sorted(pool, key=lambda s: sorted(s))))
        value = regret(world)
        if value > 1e-9:
            found.append({"queries": [sorted(q) for q in world.queries],
                          "regret": round(value, 6)})
            if value > worst:
                best_world = world
        worst = max(worst, value)
    return {"trials": trials, "size": size, "pool": queries,
            "max_regret": round(worst, 6), "counterexamples": found[:3],
            "witness": best_world}


def exhaustive_set_systems(size: int, queries: int) -> dict:
    """Every set system of this shape, for sizes small enough to enumerate.

    A clean sweep over an exhaustive class is a much stronger statement than a
    clean sweep over samples, so it is worth paying for where it is affordable.
    """
    names = tuple(f"m{index}" for index in range(size))
    subsets = [frozenset(combination)
               for count in range(1, size)
               for combination in combinations(names, count)]
    worst, witness, checked = 0.0, None, 0
    for pool in combinations(subsets, queries):
        world = SetWorld(names, pool)
        checked += 1
        value = regret(world)
        if value > worst:
            worst, witness = value, world
    return {"size": size, "pool": queries, "checked": checked,
            "max_regret": round(worst, 6),
            "witness": [sorted(q) for q in witness.queries] if witness else None}
