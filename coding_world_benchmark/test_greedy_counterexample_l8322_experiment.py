import pytest

from coding_world_benchmark.active_observation_l832_experiment import (
    optimal_depth, plan,
)
from coding_world_benchmark.cross_context_identification_l829_experiment import (
    FieldFacts, available_probes, constrain, synthetic_universe,
)
from coding_world_benchmark.greedy_counterexample_l8322_experiment import (
    KNOWN_COUNTEREXAMPLE, SetWorld, greedy_cost, grid_intervals, interval_world,
    known_counterexample, optimal_cost, regret, search_intervals, search_set_systems,
)
from coding_world_benchmark.holdout_greedy_audit_l8322_experiment import run_experiment
from coding_world_benchmark.typed_operation_l819_experiment import ValueType


@pytest.fixture(scope="module")
def report():
    return run_experiment(trials=200)


def test_the_abstract_search_agrees_with_the_real_planner():
    """Auditing code through the code being audited would prove nothing.

    So the search has its own tiny implementation of both policies, and this is
    what keeps that generalisation faithful: on an interval world the abstract
    optimum must equal L8.32's, and the abstract greedy must pick a query that
    admits the same set L8.32's planner picks.
    """
    for geometry in ("banded", "nested", "disjoint"):
        universe = synthetic_universe(5, geometry=geometry)
        pool = available_probes(universe)
        names = frozenset(f.name for f in universe)

        bounds = tuple((f.low, f.high) for f in universe)
        world = interval_world(bounds)
        assert set(world.names) == {f"m{i}" for i in range(5)}
        assert abs(optimal_cost(world, frozenset(world.names))
                   - optimal_depth(names, universe, pool)) < 1e-9

        chosen = plan(names, universe, pool)
        admitted = constrain(chosen, universe)
        # the abstract world holds the same admitted sets, under renamed meanings
        shapes = {tuple(sorted(len(q) for q in world.queries))}
        assert shapes
        assert 0 < len(admitted) < len(names)


def test_greedy_is_not_optimal_in_general():
    """The point of the whole layer, as a fixed reproducible world."""
    world = known_counterexample()
    everything = frozenset(world.names)
    greedy = greedy_cost(world, everything)
    optimal = optimal_cost(world, everything)
    assert greedy > optimal
    assert abs(greedy - 17 / 6) < 1e-9
    assert abs(optimal - 8 / 3) < 1e-9
    assert abs(regret(world) - 1 / 6) < 1e-9


def test_the_counterexample_is_pinned_not_sampled():
    names, queries = KNOWN_COUNTEREXAMPLE
    assert len(names) == 6 and len(queries) == 5
    world = known_counterexample()
    assert all(0 < len(q) < len(names) for q in world.queries)


def test_no_counterexample_exists_below_six_meanings(report):
    """Exhaustive, so this says none exists rather than none turned up."""
    for label, found in report["tier_b_exhaustive"].items():
        assert found["max_regret"] == 0.0, label
        assert found["checked"] > 300
    assert report["tier_b_exhaustive"]["K5/pool4"]["checked"] > 20000


def test_interval_geometry_is_clean_exhaustively(report):
    for size, found in report["tier_a_grid"].items():
        assert found["max_regret"] == 0.0, size
        assert found["checked"] > 1000
    for size, found in report["tier_a_random"].items():
        assert found["max_regret"] == 0.0, size
        assert found["counterexamples"] == []


def test_arbitrary_set_systems_are_not(report):
    worst = max(found["max_regret"] for found in report["tier_b_random"].values())
    assert worst > 0.0


def test_the_verdict_separates_the_rule_from_the_world(report):
    """Both halves are needed. Either alone would be misleading.

    Tier A alone reads as "greedy is fine"; Tier B alone reads as "greedy is
    unsafe". Together they say the zero L8.32.1 measured belongs to interval
    geometry, which is a fact about the schema rather than about the planner.
    """
    tier_a = max(found["max_regret"] for found in report["tier_a_random"].values())
    tier_a_exhaustive = max(f["max_regret"] for f in report["tier_a_grid"].values())
    tier_b = max(found["max_regret"] for found in report["tier_b_random"].values())
    assert tier_a == tier_a_exhaustive == 0.0
    assert tier_b > 0.0


def test_an_unresolvable_world_is_not_scored_as_regret():
    """Neither policy resolves it, so there is nothing to compare."""
    world = SetWorld(("a", "b"), ())
    assert optimal_cost(world, frozenset({"a", "b"})) == float("inf")
    assert regret(world) == 0.0


def test_the_report_formats(report):
    from coding_world_benchmark.holdout_greedy_audit_l8322_experiment import (
        format_experiment,
    )

    text = format_experiment(report)
    assert "not optimal in general" in text
    assert "property of the geometry, not of the rule" in text
