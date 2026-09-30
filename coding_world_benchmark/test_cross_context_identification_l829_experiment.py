import math

import pytest

from coding_world_benchmark.cross_context_identification_l829_experiment import (
    CONTRADICTION, DECIDABLE, UNRESOLVED, AcquiredLexicon, Belief, FieldFacts,
    Observation, available_probes, choose_probe, common_word,
    expected_information_gain, identify, indistinguishable_universe,
    numeric_literals, observations_from_request, run_acquired_query,
    synthetic_universe, universe_from,
)
from coding_world_benchmark.evidence_graph_l827_experiment import run_graph_query
from coding_world_benchmark.holdout_binding_l8211_experiment import build_holdout
from coding_world_benchmark.holdout_identification_l8291_experiment import (
    ARMS, CONTEXTS, TARGET, TRUTH, run_experiment,
)
from coding_world_benchmark.holdout_lexicon_l8281_experiment import widen_store
from coding_world_benchmark.semantic_proposer_l822_experiment import Proposer
from coding_world_benchmark.span_grounded_proposer_l823_experiment import SpanProposer
from coding_world_benchmark.structural_sensor_l826_experiment import StructuralSensor
from coding_world_benchmark.typed_operation_l819_experiment import ValueType


@pytest.fixture(scope="module")
def store():
    return widen_store(build_holdout(24)[0])


@pytest.fixture(scope="module")
def sensors():
    return SpanProposer.trained(), Proposer.trained(), StructuralSensor()


@pytest.fixture(scope="module")
def report():
    return run_experiment(24)


# --------------------------------------------------------------------------
# the state machine has no thresholds in it
# --------------------------------------------------------------------------

def test_the_three_states_are_read_off_the_set_size():
    universe = synthetic_universe(3)
    belief = Belief.prior("w", universe)
    assert belief.state == UNRESOLVED and belief.meaning is None
    one = Belief("w", frozenset({"metric_00"}))
    assert one.state == DECIDABLE and one.meaning == "metric_00"
    assert Belief("w", frozenset()).state == CONTRADICTION


def test_intersection_never_loses_the_truth(): 
    """H_{t+1} = H_t ∩ C(e_t) over truthful contexts keeps the answer in the set.

    This is why L8.28's recall 1.0 with decidable 0 was an unfinished measurement
    and not a failure: narrowing can only remove candidates the world excluded.
    """
    universe = synthetic_universe(10)
    for facts in universe:
        run = identify("w", facts.name, universe, "cross")
        assert run.state != CONTRADICTION
        assert run.meaning in (None, facts.name)


# --------------------------------------------------------------------------
# the K sweep
# --------------------------------------------------------------------------

def test_single_context_reproduces_l828(report):
    """The check that this suite measures the same thing the last one did."""
    sweep = report["sweeps"]["banded"]
    assert sweep[2]["single"]["decidable_rate"] == 1.0
    for count in (3, 5, 10, 20, 50):
        assert sweep[count]["single"]["decidable_rate"] == 0.0


def test_cross_context_identifies_where_one_context_cannot(report):
    for geometry, sweep in report["sweeps"].items():
        for count, arms in sweep.items():
            assert arms["cross"]["decidable_rate"] == 1.0, (geometry, count)
            assert arms["cross"]["decidable_rate"] >= arms["single"]["decidable_rate"]


def test_false_acquisition_is_zero_everywhere(report):
    """The invariant. Identification may stall; it may not be wrong."""
    for sweep in report["sweeps"].values():
        for arms in sweep.values():
            for arm in ARMS:
                assert arms[arm]["false_acquisition_rate"] == 0.0


def test_information_gain_beats_a_fixed_order(report):
    for geometry, sweep in report["sweeps"].items():
        for count, arms in sweep.items():
            if count == 2:
                continue
            assert arms["eig"]["mean_observations"] < arms["cross"]["mean_observations"], (
                geometry, count)


def test_the_scaling_law_belongs_to_the_world_not_the_algorithm(report):
    """The layer's real finding, pinned so it cannot quietly change.

    In ``banded`` every literal localises to a constant-size block, so EIG's cost
    stops growing once K passes it. In ``nested`` a field whose range sits inside
    every other field's cannot be separated by any literal at all -- every value
    it could truthfully be compared against is admitted by everything above it --
    so only CONTRAST separates those, one at a time, and the cost stays near
    linear. Same algorithm, different worlds.
    """
    banded = report["sweeps"]["banded"]
    nested = report["sweeps"]["nested"]
    assert banded[50]["eig"]["mean_observations"] == banded[20]["eig"]["mean_observations"]
    assert nested[50]["eig"]["mean_observations"] > nested[20]["eig"]["mean_observations"]
    mix = nested[50]["eig"]["observation_mix"]
    assert mix["CONTRAST"] > mix.get("RANGE", 0)


def test_contrast_shaves_and_range_splits():
    """Why the two sources have different scaling, in one assertion."""
    universe = synthetic_universe(8)
    belief = Belief.prior("w", universe)
    contrast = Observation("CONTRAST", "metric_00")
    best_range = max(
        (p for p in available_probes(universe) if p.kind == "RANGE"),
        key=lambda p: expected_information_gain(belief, p, universe))
    shaved = expected_information_gain(belief, contrast, universe)
    split = expected_information_gain(belief, best_range, universe)
    assert split > shaved
    assert shaved < 1.0 and math.isclose(split, 1.0, abs_tol=0.35)


# --------------------------------------------------------------------------
# contradiction and the pair nothing can separate
# --------------------------------------------------------------------------

def test_contradiction_is_detected_and_not_invented(report):
    assert report["contradiction"]["contradiction_detection_rate"] == 1.0
    assert report["contradiction"]["false_contradiction"] is False


def test_an_indistinguishable_pair_stays_unresolved(report):
    """The control that makes the rest of the numbers mean something.

    Two fields sharing a type and a range, nameable by nothing, cannot be told
    apart by any observation this world produces. Staying UNRESOLVED however many
    episodes arrive is the correct behaviour, and an ordinary field in the same
    universe still resolves, so the universe is not degenerate.
    """
    control = report["control"]
    for arm in ARMS:
        assert control["twin_a"][arm] == UNRESOLVED
        assert control["twin_b"][arm] == UNRESOLVED
    assert control["metric_00"]["cross"] == DECIDABLE


def test_an_unnameable_field_is_never_offered_as_a_contrast():
    """The bug this control caught: probes were generated for fields no sentence
    can name, which resolved the twins and made the control vacuous."""
    universe = indistinguishable_universe(3)
    named = {p.payload for p in available_probes(universe) if p.kind == "CONTRAST"}
    assert "twin_a" not in named and "twin_b" not in named


# --------------------------------------------------------------------------
# grounding: the constraints come out of real sentences
# --------------------------------------------------------------------------

def test_both_constraint_kinds_are_read_off_real_requests(store, sensors):
    span, observations = observations_from_request(CONTEXTS[0], store, *sensors)
    kinds = {item.kind for item in observations}
    assert span and kinds == {"TYPE", "CONTRAST"}
    span, observations = observations_from_request(CONTEXTS[1], store, *sensors)
    assert span and {item.kind for item in observations} == {"TYPE", "RANGE"}


def test_a_categorical_literal_is_not_read_as_a_magnitude(store):
    assert numeric_literals("64x64のケースをならして", store) == ()
    assert numeric_literals("0.5以上のフガ率をならして", store) == (0.5,)


def test_the_recurring_word_is_found_not_assumed():
    assert common_word(("0.5以上のフガ率", "超えたフガ率", "フガ率")) == "フガ率"
    assert common_word(("あいう", "かきく")) == ""


def test_the_universe_comes_from_what_episodes_recorded(store):
    universe = {facts.name: facts for facts in universe_from(store)}
    assert set(universe) == {"runtime_seconds", "success_rate", "energy_error"}
    assert universe["success_rate"].low >= 0.0 and universe["success_rate"].high <= 1.0


# --------------------------------------------------------------------------
# end to end: acquisition that is used, and can be taken back
# --------------------------------------------------------------------------

def test_two_requests_identify_what_one_could_not(report):
    end = report["end_to_end"]
    assert end["single_context_state"] == UNRESOLVED
    assert end["acquired"] and end["meaning"] == TRUTH
    assert end["false_acquisition"] is False


def test_acquisition_recovers_a_refused_request_without_a_wrong_answer(report):
    end = report["end_to_end"]
    assert end["before"] == "ABSTAIN"
    assert end["after"] == "ANSWER"
    assert end["recovered"] is True
    assert end["wrong_answer"] is False


def test_acquisition_is_reversible(report):
    """A promotion that cannot be taken back is not safe to make."""
    assert report["end_to_end"]["reversible"] is True


def test_an_unacquired_word_changes_nothing(store, sensors):
    empty = AcquiredLexicon()
    for request in CONTEXTS + (TARGET,):
        assert run_acquired_query(store, request, empty, *sensors).decision == \
               run_graph_query(store, request, *sensors).decision


def test_only_a_singleton_belief_is_acquirable():
    lexicon = AcquiredLexicon()
    from coding_world_benchmark.cross_context_identification_l829_experiment import (
        Identification,
    )
    assert not lexicon.acquire(Identification("w", "a", UNRESOLVED, None, 3, ()))
    assert not lexicon.acquire(Identification("w", "a", CONTRADICTION, None, 3, ()))
    assert lexicon.acquire(Identification("w", "a", DECIDABLE, "a", 3, ()))
    assert lexicon.entries == {"w": "a"}


def test_the_report_formats(report):
    from coding_world_benchmark.holdout_identification_l8291_experiment import (
        format_experiment,
    )

    text = format_experiment(report)
    assert "false_acq" in text and "UNRESOLVED" in text and "reversible" in text
