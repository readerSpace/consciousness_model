"""Tests for L8.41: n unknowns whose admissible combinations are context-selected."""
from __future__ import annotations

import pytest

from coding_world_benchmark.contextual_nway_l841_experiment import (
    ContextualProblem, agrees_with_oracle, analyse,
)
from coding_world_benchmark.cross_context_identification_l829_experiment import (
    Observation, synthetic_universe,
)
from coding_world_benchmark.holdout_contextual_nway_l8411_experiment import (
    GEOMETRY, LAB, ORACLE_SIZES, SCALE_SIZES, SIM, _context_global_only,
    _dominant, _symmetric, build_families, expected_assignment, run_experiment,
)
from coding_world_benchmark.nway_semantics_l837_experiment import (
    ALL_DIFFERENT, AMBIGUOUS, CONTRADICTION, DOMINATES_SUM, RESOLVED, Nary,
    Problem, propagate,
)


@pytest.fixture(scope="module")
def report():
    return run_experiment()


def _world(size):
    universe = synthetic_universe(size, geometry=GEOMETRY)
    return universe, _context_global_only(universe, size)


# -- the headline ----------------------------------------------------------

def test_each_episode_is_unique_and_the_episodes_disagree():
    for size in (3, 4, 5):
        universe, problem = _world(size)
        assert problem.all_resolved
        assert problem.meanings_in(SIM) != problem.meanings_in(LAB)


def test_removing_the_context_labels_loses_it(report):
    for row in [r for r in report["rows"] if r["name"] == "CONTEXT_GLOBAL_ONLY"]:
        assert row["erased"] == AMBIGUOUS


def test_removing_the_nary_constraint_loses_it(report):
    for row in [r for r in report["rows"] if r["name"] == "CONTEXT_GLOBAL_ONLY"]:
        assert row["without_nary"][SIM] == AMBIGUOUS
        assert row["without_nary"][LAB] == AMBIGUOUS


def test_both_are_needed_at_every_n(report):
    assert report["metrics"]["both_needed_at_every_n"]
    sizes = {r["size"] for r in report["rows"] if r["name"] == "CONTEXT_GLOBAL_ONLY"}
    assert sizes == set(ORACLE_SIZES + SCALE_SIZES)


def test_no_single_word_and_no_pair_decides_it(report):
    assert report["metrics"]["no_unary_suffices"]
    assert report["metrics"]["no_pair_suffices"]


def test_the_erased_view_is_the_union_of_the_episodes():
    _, problem = _world(4)
    erased = problem.erased()
    assert sorted(erased) == sorted({problem.meanings_in(SIM),
                                     problem.meanings_in(LAB)})


# -- safety ----------------------------------------------------------------

def test_the_answer_is_the_one_stated_in_closed_form(report):
    assert report["metrics"]["false_joint_resolution_rate"] == 0.0
    for size in (3, 5, 8):
        universe, problem = _world(size)
        assert problem.meanings_in(SIM) == expected_assignment(universe, size, 0)
        assert problem.meanings_in(LAB) == expected_assignment(universe, size, 1)


def test_a_contradiction_is_confined_to_its_own_episode(report):
    assert report["metrics"]["contradiction_reported"]
    for run in report["contradictions"]:
        assert run["sim"] == CONTRADICTION and run["lab"] == RESOLVED
        assert not run["resolved_anyway"]


def test_a_symmetric_world_resolves_in_no_episode():
    for size in (3, 4, 5):
        universe = synthetic_universe(size, geometry=GEOMETRY)
        problem = _symmetric(universe, size)
        for context in problem.contexts:
            assert problem.verdict_in(context) == AMBIGUOUS


def test_the_solver_agrees_with_brute_force(report):
    assert report["metrics"]["oracle_agreement"] == 1.0
    for size in ORACLE_SIZES:
        assert agrees_with_oracle(_world(size)[1])


# -- the other three cells -------------------------------------------------

def test_every_cell_of_the_two_by_two_is_occupied(report):
    assert report["metrics"]["cells_occupied"] == 4
    assert report["metrics"]["cell_accuracy"] == 1.0


def test_each_family_lands_in_the_cell_it_claims(report):
    expected = {f.name: (f.expect_context, f.expect_nary) for f in build_families()}
    for row in report["rows"]:
        assert (row["context_needed"], row["nary_needed"]) == expected[row["name"]]


# -- the machinery ---------------------------------------------------------

def test_without_drops_only_the_non_decomposable_constraint():
    universe, problem = _world(4)
    stripped = problem.without()
    for _, sub in stripped.problems:
        kinds = {constraint.kind for constraint in sub.nary}
        assert DOMINATES_SUM not in kinds and ALL_DIFFERENT in kinds


def test_an_observation_reaches_only_the_episode_it_was_made_in():
    universe, problem = _world(4)
    before = problem.meanings_in(LAB)
    names = [facts.name for facts in universe]
    changed = problem.observed(SIM, 0, Observation("EQUALS", names[0]))
    assert changed.meanings_in(LAB) == before
    assert changed.verdict_in(SIM) == CONTRADICTION


def test_propagation_stops_the_moment_a_domain_empties():
    """The crash L8.41 found in L8.37's propagator, pinned.

    A unary observation can empty a domain in the middle of a sweep, and the
    DOMINATES_SUM filter reads its partners' extremes -- an empty partner has
    none. L8.37.1 never reached it because its contradiction arrives through
    ALL_DIFFERENT, whose filter tolerates an empty domain.
    """
    universe = synthetic_universe(4, geometry=GEOMETRY)
    names = [facts.name for facts in universe]
    problem = _dominant(universe, 4, 0)
    problem.unary = {0: (Observation("EQUALS", names[0]),)}
    assert propagate(problem) is None


def test_the_gate_is_still_only_the_size_of_the_belief():
    universe = synthetic_universe(3, geometry=GEOMETRY)
    loose = Problem(universe, 3, nary=(Nary(ALL_DIFFERENT, (0, 1, 2)),))
    problem = ContextualProblem(3, ((SIM, loose),))
    found = analyse(problem)
    assert found.full[SIM] == AMBIGUOUS
    assert not found.resolved and not found.context_needed and not found.nary_needed
