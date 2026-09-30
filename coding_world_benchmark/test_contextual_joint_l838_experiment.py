import pytest

from coding_world_benchmark.contextual_joint_l838_experiment import (
    NO_EFFECT, RELATIONAL_EFFECT, UNARY_EFFECT, ContextualJoint, erased_verdict,
    per_word_sees_context, per_word_view,
)
from coding_world_benchmark.cross_context_identification_l829_experiment import (
    Observation, universe_from,
)
from coding_world_benchmark.holdout_binding_l8211_experiment import build_holdout
from coding_world_benchmark.holdout_contextual_joint_l8381_experiment import (
    CONTEXTS, ENERGY, PROBE, PROBE_INDEX, RATE, RUNTIME, SURFACES, build_families,
    run_experiment,
)
from coding_world_benchmark.holdout_lexicon_l8281_experiment import widen_store
from coding_world_benchmark.nway_semantics_l837_experiment import (
    AMBIGUOUS, CONTRADICTION, RESOLVED,
)


@pytest.fixture(scope="module")
def universe():
    return universe_from(widen_store(build_holdout(24)[0]))


@pytest.fixture(scope="module")
def report():
    return run_experiment()


def relational() -> ContextualJoint:
    return ContextualJoint(SURFACES, (
        ("sim", frozenset({(RUNTIME, RATE), (RATE, ENERGY)})),
        ("lab", frozenset({(RUNTIME, ENERGY), (RATE, RATE)}))))


# --------------------------------------------------------------------------
# the constraint on the criterion itself
# --------------------------------------------------------------------------

def test_identical_marginals_and_a_unique_solution_cannot_coexist():
    """A singleton's marginals *are* its assignment.

    So "every unary marginal identical across contexts" and "|B_c| = 1 in every
    context" describe the same context twice. The identity has to hold at the
    level of the belief sets, and what has to be shown is that the difference
    between them is usable -- which is what the decisive family does.
    """
    same = ContextualJoint(SURFACES, (
        ("sim", frozenset({(RUNTIME, RATE)})),
        ("lab", frozenset({(RUNTIME, RATE)}))))
    assert same.marginals_agree and same.sets_agree
    differing = ContextualJoint(SURFACES, (
        ("sim", frozenset({(RUNTIME, RATE)})),
        ("lab", frozenset({(RATE, ENERGY)}))))
    assert not differing.marginals_agree  # unique per context => marginals differ


# --------------------------------------------------------------------------
# what a per-word account can and cannot see
# --------------------------------------------------------------------------

def test_the_relational_family_is_invisible_to_a_per_word_view():
    joint = relational()
    assert joint.marginals_agree
    assert not joint.sets_agree
    assert joint.effect == RELATIONAL_EFFECT
    assert per_word_sees_context(joint) is False
    view = per_word_view(joint)
    for sets in view.values():
        assert sets["sim"] == sets["lab"]


def test_one_observation_lands_on_different_meanings(universe):
    """The demonstration. "The sets differ" on its own is bookkeeping."""
    joint = relational()
    after = joint.observe(PROBE_INDEX, PROBE, universe)
    assert after.verdict_in("sim") == RESOLVED
    assert after.verdict_in("lab") == RESOLVED
    assert after.meanings_in("sim")[SURFACES[1]] == RATE
    assert after.meanings_in("lab")[SURFACES[1]] == ENERGY


def test_erasing_the_labels_loses_exactly_that(universe):
    joint = relational()
    after = joint.observe(PROBE_INDEX, PROBE, universe)
    from coding_world_benchmark.nway_semantics_l837_experiment import verdict

    assert verdict(sorted(after.erased())) == AMBIGUOUS
    assert erased_verdict(joint) == AMBIGUOUS


def test_an_independent_polysemy_is_visible_per_word():
    """The control that stops the layer claiming credit for L8.35's work."""
    joint = ContextualJoint(SURFACES, (
        ("sim", frozenset({(RUNTIME, RATE)})),
        ("lab", frozenset({(ENERGY, ENERGY)}))))
    assert joint.effect == UNARY_EFFECT
    assert per_word_sees_context(joint) is True


def test_identical_sets_are_reported_as_no_effect():
    live = frozenset({(RUNTIME, RATE), (RATE, ENERGY)})
    joint = ContextualJoint(SURFACES, (("sim", live), ("lab", live)))
    assert joint.effect == NO_EFFECT
    assert joint.sets_agree and joint.marginals_agree


# --------------------------------------------------------------------------
# the gate is unchanged
# --------------------------------------------------------------------------

def test_the_same_three_states_per_context():
    joint = ContextualJoint(SURFACES, (
        ("sim", frozenset({(RUNTIME, RATE)})),
        ("lab", frozenset({(RUNTIME, RATE), (RATE, ENERGY)})),
        ("void", frozenset())))
    assert joint.verdict_in("sim") == RESOLVED
    assert joint.verdict_in("lab") == AMBIGUOUS
    assert joint.verdict_in("void") == CONTRADICTION
    assert joint.meanings_in("lab") is None
    assert joint.meanings_in("void") is None


def test_an_observation_can_be_scoped_to_one_context(universe):
    joint = relational()
    only_sim = joint.observe(PROBE_INDEX, PROBE, universe, context="sim")
    assert only_sim.verdict_in("sim") == RESOLVED
    assert only_sim.verdict_in("lab") == AMBIGUOUS


# --------------------------------------------------------------------------
# the hold-out
# --------------------------------------------------------------------------

def test_the_invariant(report):
    assert report["metrics"]["false_joint_resolution_rate"] == 0.0


def test_the_gain_is_not_reachable_by_the_two_layers_below(report):
    """A solver that ran L8.35 per word and then L8.37 on the pool scores zero."""
    assert report["metrics"]["relational_only_gain"] == 1
    assert report["metrics"]["marginal_identity_verified"] is True
    relational_runs = [r for r in report["runs"] if r["effect"] == RELATIONAL_EFFECT]
    assert relational_runs
    for run in relational_runs:
        assert run["marginals_agree"]
        assert not run["per_word_sees_context"]
        assert run["different_readings"]


def test_erasure_breaks_only_the_relational_family(report):
    assert report["metrics"]["erasure_breaks_only_relational"] is True
    for run in report["runs"]:
        if run["effect"] == RELATIONAL_EFFECT:
            assert run["erasure_breaks_it"]
        else:
            assert not run["erasure_breaks_it"]


def test_every_family_is_classified_correctly(report):
    assert report["metrics"]["effect_accuracy"] == 1.0
    assert report["metrics"]["context_conditioned_joint_accuracy"] == 1.0
    assert len(build_families()) == 5


def test_symmetry_inside_a_context_is_not_broken_by_the_probe(report):
    runs = {run["name"]: run for run in report["runs"]}
    symmetric = runs["SYMMETRIC_WITHIN_CONTEXT"]
    assert all(value is None for value in symmetric["readings"].values())
    assert all(v == AMBIGUOUS for v in symmetric["after_probe"].values())


def test_an_empty_context_is_reported_not_filled_in(report):
    runs = {run["name"]: run for run in report["runs"]}
    assert runs["CONTRADICTION"]["verdicts"]["lab"] == CONTRADICTION
    assert runs["CONTRADICTION"]["readings"]["lab"] is None


def test_the_report_formats(report):
    from coding_world_benchmark.holdout_contextual_joint_l8381_experiment import (
        format_experiment,
    )

    text = format_experiment(report)
    assert "per-word account to find" in text
    assert "RELATIONAL_EFFECT" in text
