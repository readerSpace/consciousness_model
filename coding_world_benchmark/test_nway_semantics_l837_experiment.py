import inspect

import pytest

from coding_world_benchmark.cross_context_identification_l829_experiment import (
    Observation, synthetic_universe, universe_from,
)
from coding_world_benchmark.holdout_binding_l8211_experiment import build_holdout
from coding_world_benchmark.holdout_lexicon_l8281_experiment import widen_store
from coding_world_benchmark.holdout_nway_l8371_experiment import (
    GLOBAL_GEOMETRY, ORACLE_SIZES, SCALE_SIZES, build_families, run_experiment,
)
from coding_world_benchmark.joint_inference_l834_experiment import (
    EXCEEDS, JOINT_RESOLVED, MARGINAL_ONLY, Relation,
)
from coding_world_benchmark.nway_semantics_l837_experiment import (
    ALL_DIFFERENT, AMBIGUOUS, CONTRADICTION, DOMINATES_SUM, RESOLVED, Nary, Problem,
    exhaustive, marginals, nary_holds, pair_marginals, propagate, solve, verdict,
)


@pytest.fixture(scope="module")
def report():
    return run_experiment()


# --------------------------------------------------------------------------
# no new rule
# --------------------------------------------------------------------------

def test_the_same_three_states_at_every_n():
    """Adding no safety rule as n grows is the thing under test.

    A threshold appearing at n=4 would have shown that L8.21's uniqueness rule
    was a two-variable convenience rather than a principle.
    """
    assert verdict([]) == CONTRADICTION
    assert verdict([("a",)]) == RESOLVED
    assert verdict([("a",), ("b",)]) == AMBIGUOUS
    source = inspect.getsource(verdict)
    for smell in ("threshold", "0.5", "if n ", "size >"):
        assert smell not in source


def test_the_gate_is_uniqueness_and_nothing_else(report):
    for row in report["oracle"]["rows"] + report["scale"]["rows"]:
        if row["verdict"] == RESOLVED:
            assert row["expected"] == RESOLVED, row


# --------------------------------------------------------------------------
# phase one: agree with brute force
# --------------------------------------------------------------------------

def test_exact_agreement_with_the_oracle(report):
    assert report["oracle"]["oracle_agreement"] == 1.0
    assert len(report["oracle"]["rows"]) == len(ORACLE_SIZES) * len(build_families())
    for row in report["oracle"]["rows"]:
        assert row["verdict"] == row["oracle_verdict"], row


def test_the_oracle_really_enumerates_the_product(report):
    for row in report["oracle"]["rows"]:
        assert row["product"] == row["size"] ** row["size"]


# --------------------------------------------------------------------------
# phase two: then stop doing brute force
# --------------------------------------------------------------------------

def test_the_answers_hold_past_enumeration(report):
    assert report["scale"]["verdict_accuracy"] == 1.0
    assert set(SCALE_SIZES) == {r["size"] for r in report["scale"]["rows"]}


def test_the_product_is_not_visited(report):
    """At n=10 the product holds 10^10 assignments."""
    assert report["metrics"]["worst_states_fraction"] < 1e-3
    biggest = [r for r in report["scale"]["rows"] if r["size"] == 10]
    assert biggest and all(r["product"] == 10 ** 10 for r in biggest)
    for row in biggest:
        assert row["states"] < 200, row


def test_resolution_costs_about_one_state_per_word(report):
    """The signature of propagation rather than of search."""
    for row in report["scale"]["rows"]:
        if row["verdict"] == RESOLVED:
            assert row["states"] == row["size"], row


# --------------------------------------------------------------------------
# the family that decides it
# --------------------------------------------------------------------------

def test_global_only_is_global_only(report):
    """Every one- and two-word marginal plural, and the whole thing unique.

    A world like that cannot be solved by being good at words or at pairs, so
    solving it is evidence of constraint satisfaction rather than of a
    generalised special case.
    """
    assert report["global_only"]["holds_at_every_size"]
    for row in report["global_only"]["rows"]:
        assert row["without_nary_solutions"] > 1
        assert row["singleton_marginals"] == 0
        assert row["singleton_pairs"] == 0
        assert row["with_nary"] == RESOLVED


def test_the_n_ary_constraint_decomposes_into_no_pairs():
    """DOMINATES_SUM relates all n at once; that is why it is needed."""
    universe = synthetic_universe(4, geometry=GLOBAL_GEOMETRY)
    names = [f.name for f in universe]
    top = names[-1]
    assert nary_holds(Nary(DOMINATES_SUM, (0, 1, 2, 3)),
                      (top, names[0], names[1], names[2]), universe)
    assert not nary_holds(Nary(DOMINATES_SUM, (0, 1, 2, 3)),
                          (names[0], top, names[1], names[2]), universe)
    # and it is not implied by any pair: every *pair* of those meanings is fine
    assert nary_holds(Nary(DOMINATES_SUM, (0, 1)), (names[1], names[0]), universe)


def test_the_geometry_was_the_problem_not_the_solver():
    """A disclosure kept in code: the disjoint layout outgrows a sum constraint.

    With evenly spaced ranges the highest field's maximum stops exceeding the sum
    of the others' minima from n=5, so GLOBAL_ONLY collapsed to CONTRADICTION
    there. The dominant layout keeps exactly one field able to dominate at any n.
    """
    flat = synthetic_universe(5, geometry="disjoint")
    names = [f.name for f in flat]
    assert not nary_holds(Nary(DOMINATES_SUM, tuple(range(5))),
                          tuple(reversed(names)), flat)
    steep = synthetic_universe(5, geometry=GLOBAL_GEOMETRY)
    steep_names = [f.name for f in steep]
    assert nary_holds(Nary(DOMINATES_SUM, tuple(range(5))),
                      tuple([steep_names[-1]] + steep_names[:-1]), steep)


# --------------------------------------------------------------------------
# the control that cannot be passed by trying harder
# --------------------------------------------------------------------------

def test_symmetry_is_never_broken(report):
    assert report["metrics"]["symmetry_broken_rate"] == 0.0
    assert report["symmetry"]["never_broken"]
    for row in report["symmetry"]["rows"]:
        assert row["base"] == AMBIGUOUS
        assert row["with_more_evidence"] == AMBIGUOUS


def test_more_of_the_same_evidence_does_not_help_a_symmetric_world():
    universe = synthetic_universe(4, geometry="disjoint")
    base = Problem(universe, 4, nary=(Nary(ALL_DIFFERENT, (0, 1, 2, 3)),))
    assert verdict(solve(base)[0]) == AMBIGUOUS
    piled = Problem(universe, 4, nary=tuple(
        Nary(ALL_DIFFERENT, (0, 1, 2, 3)) for _ in range(5)))
    assert verdict(solve(piled)[0]) == AMBIGUOUS


# --------------------------------------------------------------------------
# propagation itself
# --------------------------------------------------------------------------

def test_a_contradiction_is_found_without_touching_the_product():
    universe = synthetic_universe(6, geometry="disjoint")
    names = [f.name for f in universe]
    problem = Problem(universe, 6,
                      unary={i: (Observation("EQUALS", names[0]),) for i in range(6)},
                      nary=(Nary(ALL_DIFFERENT, tuple(range(6))),))
    assert propagate(problem) is None
    found, states = solve(problem)
    assert found == [] and states == 0


def test_the_real_universe_agrees_with_the_oracle(report):
    real = report["real"]
    assert real["verdict"] == RESOLVED
    assert real["agrees_with_oracle"]
    assert real["states"] < real["product"]


def test_the_invariant(report):
    assert report["metrics"]["false_joint_resolution_rate"] == 0.0
    assert report["metrics"]["verdict_accuracy"] == 1.0


def test_the_report_formats(report):
    from coding_world_benchmark.holdout_nway_l8371_experiment import format_experiment

    text = format_experiment(report)
    assert "GLOBAL_ONLY" in text and "10,000,000,000" in text
