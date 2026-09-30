import pytest

from coding_world_benchmark.active_observation_l832_experiment import (
    RESOLVED, STRUCTURALLY_UNRESOLVABLE, answer_as_observation, branches,
    expected_size, optimal_depth, plan, question_for, render_query, run_active,
    useful,
)
from coding_world_benchmark.cross_context_identification_l829_experiment import (
    Observation, available_probes, constrain, indistinguishable_universe,
    synthetic_universe, universe_from,
)
from coding_world_benchmark.holdout_active_l8321_experiment import (
    OPTIMAL_LIMIT, run_experiment, worlds,
)
from coding_world_benchmark.holdout_binding_l8211_experiment import build_holdout
from coding_world_benchmark.holdout_lexicon_l8281_experiment import widen_store
from coding_world_benchmark.semantic_identifiability_l831_experiment import (
    effect, reachable,
)
from coding_world_benchmark.typed_operation_l819_experiment import ValueType


@pytest.fixture(scope="module")
def real():
    return universe_from(widen_store(build_holdout(24)[0]))


@pytest.fixture(scope="module")
def report():
    return run_experiment()


# --------------------------------------------------------------------------
# a negative answer is evidence, and it has to be exact
# --------------------------------------------------------------------------

def test_a_no_answer_asserts_the_complement(real):
    """The bug this caught, pinned.

    An earlier draft approximated the negative branch with a hand-picked
    CONTRAST and resolved the word to the wrong field -- a false resolution, the
    one thing this layer claims cannot happen. It cannot happen only because the
    negation is exact, so the exactness is the test.
    """
    query = Observation("RANGE", 0.5)
    negated = answer_as_observation(query, False)
    assert negated.kind == "NOT"
    everything = frozenset(f.name for f in real)
    assert constrain(negated, real) == everything - constrain(query, real)
    assert answer_as_observation(query, True) is query


def test_both_branches_keep_the_truth(real):
    """Why false_resolution_rate is structural rather than lucky."""
    candidates = frozenset(f.name for f in real)
    for query in available_probes(real):
        yes, no = branches(query, candidates, real)
        for name in candidates:
            assert name in (yes if effect(name, query, real) else no)


# --------------------------------------------------------------------------
# uselessness is an equality, not a threshold
# --------------------------------------------------------------------------

def test_a_query_that_does_not_split_leaves_the_expected_size_unchanged():
    universe = synthetic_universe(4, geometry="disjoint")
    candidates = frozenset({"metric_00", "metric_01"})
    useless = Observation("TYPE", ValueType.NUMBER)
    assert not useful(useless, candidates, universe)
    assert expected_size(useless, candidates, universe) == len(candidates)


def test_no_useful_query_means_stop_not_search_harder():
    twins = indistinguishable_universe(3)
    pool = available_probes(twins)
    assert plan(frozenset({"twin_a", "twin_b"}), twins, pool) is None
    run = run_active("twin_a", twins, pool)
    assert run.state == STRUCTURALLY_UNRESOLVABLE
    assert run.meaning is None
    assert run.wasted == 0


def test_the_planner_and_the_exact_optimum_agree_on_impossibility():
    """``plan`` returning None and ``optimal_depth`` returning inf are the same
    fact reached by different routes, which is what makes one a check on the other."""
    twins = indistinguishable_universe(3)
    pool = available_probes(twins)
    assert optimal_depth(frozenset({"twin_a", "twin_b"}), twins, pool) == float("inf")


# --------------------------------------------------------------------------
# the hold-out
# --------------------------------------------------------------------------

def test_false_resolution_rate_is_zero(report):
    assert report["metrics"]["false_resolution_rate"] == 0.0
    assert report["metrics"]["wasted_queries"] == 0


def test_it_stops_instead_of_asking_forever(report):
    assert report["metrics"]["impossible_query_avoidance_rate"] == 1.0
    twins = next(w for w in report["worlds"] if w["world"] == "twins/K3")
    assert twins["unresolvable"] == twins["stopped_cleanly"] == 2
    assert twins["resolved"] == 3  # the ordinary fields in the same world still resolve


def test_it_resolves_everything_that_is_resolvable(report):
    assert report["metrics"]["active_resolution_rate"] > 0.95
    for world in report["worlds"]:
        if world["world"] != "twins/K3":
            assert world["resolved"] == world["size"], world["world"]


def test_greedy_is_measured_against_optimal_not_assumed_equal_to_it(report):
    """Regret is 0 in these worlds. That is a measurement, not a theorem."""
    assert report["metrics"]["greedy_vs_optimal_regret"] == 0.0
    compared = [w for w in report["worlds"]
                if w["optimal"] is not None and w["mean_steps"]]
    assert len(compared) >= 8
    for world in compared:
        assert world["size"] <= OPTIMAL_LIMIT
        assert world["mean_steps"] >= world["optimal"]


def test_not_enumerated_is_not_reported_as_unresolvable(report):
    """Two different facts that printed the same way until they were separated."""
    big = [w for w in report["worlds"] if w["size"] > OPTIMAL_LIMIT]
    assert big
    for world in big:
        assert world["optimal"] is None
        assert "not enumerated" in world["optimal_note"]
    twins = next(w for w in report["worlds"] if w["world"] == "twins/K3")
    assert twins["optimal_note"] == "unresolvable"


def test_the_predicted_reduction_matches_what_happens(report):
    assert report["metrics"]["predicted_vs_actual_reduction"] == 0.0


# --------------------------------------------------------------------------
# asking is stronger than being told
# --------------------------------------------------------------------------

def test_asking_reaches_meanings_being_told_cannot(report):
    """L8.31's ``reachable`` is the passive regime; a query has two outcomes.

    Measured on magnitude comparisons alone, because the CONTRAST probes make
    every meaning passively identifiable and the gap would read as zero for the
    wrong reason.
    """
    assert report["metrics"]["active_vs_passive_gain"] > 20
    nested = next(w for w in report["worlds"] if w["world"] == "nested/K10")
    assert len(nested["passive_gain"]) >= 5


def test_the_nested_world_is_exactly_l829s_finding():
    """A meaning contained in every other's range cannot be narrowed passively.

    Every literal it could truthfully be compared against is admitted by
    everything above it -- so only a *negative* answer separates it, and only an
    active learner can obtain one.
    """
    universe = synthetic_universe(6, geometry="nested")
    magnitudes = tuple(q for q in available_probes(universe) if q.kind == "RANGE")
    deep = "metric_05"
    assert len(reachable(deep, universe, magnitudes)) > 1
    assert run_active(deep, universe, magnitudes).state == RESOLVED


def test_the_twins_are_unmoved_by_asking():
    twins = indistinguishable_universe(3)
    magnitudes = tuple(q for q in available_probes(twins) if q.kind == "RANGE")
    assert run_active("twin_a", twins, magnitudes).state != RESOLVED


# --------------------------------------------------------------------------
# the question, in words, and the loop it closes
# --------------------------------------------------------------------------

def test_a_contrast_question_is_phrased_the_way_its_answer_is_read(real):
    """CONTRAST(X) constrains to "not X", so yes must mean "not X"."""
    query = Observation("CONTRAST", "success_rate")
    assert "とは別の指標ですか" in render_query(query, "フガ率")
    assert not effect("success_rate", query, real)


def test_there_is_no_question_when_there_is_nothing_to_ask():
    twins = indistinguishable_universe(3)
    text, query = question_for(frozenset({"twin_a", "twin_b"}), twins,
                               available_probes(twins), "ホゲ")
    assert query is None
    assert "これ以上絞れません" in text


def test_the_loop_closes_on_a_real_request(report):
    """不明 -> なぜ不明か -> 何を訊けば分かるか -> 訊く -> 実行が変わる."""
    end = report["end_to_end"]
    assert end["before"] == "ABSTAIN"
    assert end["stuck_candidates"] == ["energy_error", "success_rate"]
    assert end["diagnosis"] == "PARTIALLY_IDENTIFIABLE"
    assert end["question"].startswith("「フガ率」")
    assert end["questions_asked"] == 1
    assert end["settled_meaning"] == "success_rate"
    assert end["after"] == "ANSWER"
    assert end["correct"] is True


def test_the_report_formats(report):
    from coding_world_benchmark.holdout_active_l8321_experiment import format_experiment

    text = format_experiment(report)
    assert "false_resolution_rate" in text and "not a theorem" in text
