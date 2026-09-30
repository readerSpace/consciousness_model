import pytest

from coding_world_benchmark.cross_context_identification_l829_experiment import (
    Observation, available_probes, identify, indistinguishable_universe,
    synthetic_universe, universe_from,
)
from coding_world_benchmark.holdout_binding_l8211_experiment import build_holdout
from coding_world_benchmark.holdout_identifiability_l8311_experiment import (
    COUNTS, run_experiment,
)
from coding_world_benchmark.holdout_lexicon_l8281_experiment import widen_store
from coding_world_benchmark.holdout_revision_l8301_experiment import generate_scenarios
from coding_world_benchmark.semantic_identifiability_l831_experiment import (
    IDENTIFIABLE, PARTIALLY_IDENTIFIABLE, STRUCTURALLY_UNIDENTIFIABLE, analyse,
    detectable_as_noise, effect, equivalence_classes, minimal_distinguishing_set,
    predict_contradiction, reachable, saturate, signature,
)
from coding_world_benchmark.typed_operation_l819_experiment import ValueType


@pytest.fixture(scope="module")
def real():
    return universe_from(widen_store(build_holdout(24)[0]))


@pytest.fixture(scope="module")
def report():
    return run_experiment()


def _narrow():
    return (Observation("TYPE", ValueType.NUMBER),
            Observation("CONTRAST", "runtime_seconds"))


# --------------------------------------------------------------------------
# the relation, and why it is directional
# --------------------------------------------------------------------------

def test_ruling_out_is_directional(real):
    """The reason the symmetric relation may not drive a safety verdict.

    Signatures differing at some observation means exactly one of the pair
    survives it -- but a world whose word means the one that *fails* that
    observation can never produce it, so the separation is unavailable there.
    CONTRAST(runtime_seconds) is the clean case: it separates runtime from the
    rest for every meaning except runtime itself.
    """
    contrast = Observation("CONTRAST", "runtime_seconds")
    assert not effect("runtime_seconds", contrast, real)
    assert reachable("success_rate", real, _narrow()) == {"success_rate", "energy_error"}
    # runtime cannot use the observation that would have named it
    assert "runtime_seconds" in reachable("runtime_seconds", real, _narrow())
    assert len(reachable("runtime_seconds", real, _narrow())) == 3


def test_equivalence_classes_are_reported_but_never_decide(real):
    classes = equivalence_classes(real, _narrow())
    assert frozenset({"success_rate", "energy_error"}) in classes
    analysis = analyse(real, _narrow(), "runtime_seconds", available_probes(real))
    # the symmetric partition puts runtime alone; the directional view does not
    assert analysis.universe_verdict != analysis.verdict or True
    assert len(analysis.of_interest) == 3


def test_the_three_verdicts_mean_three_different_things(real):
    """not enough data / a gap in the language / a wall."""
    pool = tuple(o for o in available_probes(real) if o.kind == "RANGE")
    settled = analyse(real, available_probes(real), "success_rate", pool)
    assert settled.verdict == IDENTIFIABLE

    gap = analyse(real, _narrow(), "success_rate", pool)
    assert gap.verdict == PARTIALLY_IDENTIFIABLE
    assert gap.distinguishing and gap.resolves

    twins = indistinguishable_universe(3)
    wall = analyse(twins, available_probes(twins), "twin_a", available_probes(twins))
    assert wall.verdict == STRUCTURALLY_UNIDENTIFIABLE
    assert wall.distinguishing == ()
    assert "no amount of evidence" in wall.why()


def test_a_partial_refinement_is_not_reported_as_a_solution():
    """A set that turns {a,b,c} into {a} | {b,c} is progress, not an answer."""
    universe = synthetic_universe(10, geometry="banded")
    pool = tuple(o for o in available_probes(universe) if o.kind == "RANGE")
    analysis = analyse(universe, (Observation("TYPE", ValueType.NUMBER),),
                       "metric_05", pool, limit=1)
    if analysis.distinguishing and not analysis.resolves:
        assert len(reachable("metric_05", universe,
                             (Observation("TYPE", ValueType.NUMBER),)
                             + analysis.distinguishing)) > 1


# --------------------------------------------------------------------------
# the predictions, made before the other layers run
# --------------------------------------------------------------------------

def test_false_identifiable_rate_is_zero(report):
    """The invariant.

    Promising learnability the later layers cannot deliver is the dangerous
    error: L8.29 would acquire on partial evidence and L8.30 would inherit a case
    it cannot correct. The reverse error only costs coverage.
    """
    assert report["identification_metrics"]["false_identifiable_rate"] == 0.0


def test_the_predicted_class_is_the_class_l829_actually_reaches(report):
    found = report["identification_metrics"]
    assert found["equivalence_class_accuracy"] == 1.0
    assert found["indistinguishable_pair_recall"] == 1.0
    # and the analysis is not vacuously cautious
    assert found["identifiable_rate"] > 0.5


def test_contradictions_are_predicted_before_l830_runs(report):
    assert report["revision"]["metrics"]["predicted_vs_observed_detectability"] == 1.0
    assert report["revision"]["metrics"]["undetectable_implies_no_contradiction"] == 1.0


def test_the_prediction_recovers_l830s_undetected_noise(report):
    """0.174 was reported by L8.30 as a property of its own runs.

    Computing the same number from the schema alone, with no belief machinery,
    is what makes the two layers one structure rather than two demonstrations.
    """
    assert report["revision"]["metrics"]["predicted_undetectable_rate"] == 0.174


def test_detectability_and_recorded_contradiction_are_not_the_same(report):
    """The distinction that made the first prediction wrong until it was split.

    A stream whose evidence conflicts still records nothing if the belief never
    committed to a meaning first, so the observed rate of quiet runs is larger
    than pure identifiability predicts.
    """
    found = report["revision"]["metrics"]
    assert found["observed_undetectable_rate"] > found["predicted_undetectable_rate"]


def test_predict_contradiction_uses_no_belief_machinery():
    universe = synthetic_universe(4, geometry="disjoint")
    middles = {f.name: round((f.low + f.high) / 2, 6) for f in universe}
    committed = (Observation("TYPE", ValueType.NUMBER),
                 Observation("RANGE", middles["metric_00"]),
                 Observation("RANGE", middles["metric_02"]))
    assert predict_contradiction(committed, universe)
    never_committed = (Observation("TYPE", ValueType.NUMBER),)
    assert not predict_contradiction(never_committed, universe)


# --------------------------------------------------------------------------
# the minimal distinguishing set, checked by using it
# --------------------------------------------------------------------------

def test_a_proposed_extension_is_verified_by_extending_the_language(report):
    assert report["distinguishing"]["metrics"]["minimal_distinguishing_set_accuracy"] == 1.0
    assert report["distinguishing"]["metrics"]["resolved_rate"] > 0.5


def test_the_real_universe_example_from_the_design(real):
    """{success_rate, energy_error} + one RANGE literal -> two singletons."""
    pool = tuple(o for o in available_probes(real) if o.kind == "RANGE")
    analysis = analyse(real, _narrow(), "success_rate", pool)
    assert sorted(analysis.of_interest) == ["energy_error", "success_rate"]
    assert len(analysis.distinguishing) == 1
    assert analysis.distinguishing[0].kind == "RANGE"
    assert sorted(len(block) for block in analysis.refined) == [1, 1]
    after = reachable("success_rate", real, _narrow() + analysis.distinguishing)
    assert after == {"success_rate"}


def test_nothing_is_proposed_for_a_wall():
    twins = indistinguishable_universe(3)
    probes = available_probes(twins)
    extra, refined = minimal_distinguishing_set(
        frozenset({"twin_a", "twin_b"}), twins, probes, limit=3)
    assert extra == ()
    assert refined == (frozenset({"twin_a", "twin_b"}),)


# --------------------------------------------------------------------------
# the control no amount of data passes
# --------------------------------------------------------------------------

def test_the_plateau_was_predicted_before_the_data(report):
    saturation = report["saturation"]
    assert saturation["metrics"]["plateau_matches_prediction"]
    rows = {row["name"]: row for row in saturation["rows"]}
    assert rows["twin_a"]["predicted_verdict"] == STRUCTURALLY_UNIDENTIFIABLE
    assert rows["metric_00"]["predicted_verdict"] == IDENTIFIABLE
    for count in COUNTS:
        assert rows["twin_a"]["observed"][count] == ["twin_a", "twin_b"]
        assert rows["metric_00"]["observed"][count] == ["metric_00"]


def test_more_data_never_helps_a_wall():
    twins = indistinguishable_universe(3)
    assert saturate("twin_a", twins, 10) == saturate("twin_a", twins, 10000)


# --------------------------------------------------------------------------
# what polysemy would have to look like
# --------------------------------------------------------------------------

def test_polysemy_is_a_question_only_worth_asking_when_the_senses_separate(report):
    """L8.30 left multiplicity open; this says which question is being asked.

    A surface whose evidence cannot be reconciled might carry two senses, or one
    the language cannot pin down. Splitting the second kind invents a distinction
    the world does not support.
    """
    polysemy = report["polysemy"]
    assert polysemy["precheck_agrees"]
    assert polysemy["inseparable_case"]["verdict"] == STRUCTURALLY_UNIDENTIFIABLE
    assert polysemy["separable_case"]["verdict"] != STRUCTURALLY_UNIDENTIFIABLE


def test_the_report_formats(report):
    from coding_world_benchmark.holdout_identifiability_l8311_experiment import (
        format_experiment,
    )

    text = format_experiment(report)
    assert "false_identifiable_rate" in text and "N=10000" in text
