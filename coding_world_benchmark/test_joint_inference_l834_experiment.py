import pytest

from coding_world_benchmark.cross_context_identification_l829_experiment import (
    Observation, available_probes, universe_from,
)
from coding_world_benchmark.holdout_binding_l8211_experiment import build_holdout
from coding_world_benchmark.holdout_joint_l8341_experiment import (
    SURFACES, build_scenarios, run_experiment,
)
from coding_world_benchmark.holdout_lexicon_l8281_experiment import widen_store
from coding_world_benchmark.joint_inference_l834_experiment import (
    CONTRADICTION, DISTINCT, EXCEEDS, JOINT_RESOLVED, MARGINAL_ONLY, SAME_TYPE,
    UNRESOLVED, JointBelief, Relation, holds, independent, joint_queries,
    plan_joint, relations_from_request, resolve,
)
from coding_world_benchmark.typed_operation_l819_experiment import ValueType


@pytest.fixture(scope="module")
def universe():
    return universe_from(widen_store(build_holdout(24)[0]))


@pytest.fixture(scope="module")
def report():
    return run_experiment()


def _number():
    return Observation("TYPE", ValueType.NUMBER)


# --------------------------------------------------------------------------
# a unique marginal is not a unique assignment
# --------------------------------------------------------------------------

def test_marginals_can_be_narrow_while_nothing_is_known(universe):
    """The safety rule of the layer, as the smallest possible example.

    Both surfaces are "narrowed to two" and the correspondence between them is
    exactly what is undetermined. Reading the marginals would report progress.
    """
    belief = JointBelief.prior(SURFACES, universe)
    for index in (0, 1):
        belief = belief.observe(index, _number())
        belief = belief.observe(index, Observation("CONTRAST", "energy_error"))
    belief = belief.relate(Relation(DISTINCT, 0, 1))
    assert belief.assignments == frozenset({
        ("runtime_seconds", "success_rate"), ("success_rate", "runtime_seconds")})
    assert belief.marginal(0) == belief.marginal(1) == {"runtime_seconds", "success_rate"}
    assert belief.state == MARGINAL_ONLY
    assert belief.meanings is None


def test_only_a_single_assignment_yields_meanings(universe):
    belief = JointBelief.prior(SURFACES, universe)
    assert belief.state == UNRESOLVED and belief.meanings is None
    settled = JointBelief(SURFACES, frozenset({("success_rate", "energy_error")}), universe)
    assert settled.state == JOINT_RESOLVED
    assert settled.meanings == {"フガ率": "success_rate", "ホゲ尺": "energy_error"}
    assert JointBelief(SURFACES, frozenset(), universe).state == CONTRADICTION


# --------------------------------------------------------------------------
# the relation is evidence
# --------------------------------------------------------------------------

def test_the_order_relation_comes_from_the_store_not_a_vocabulary(universe):
    """runtime 8.0--51.75, success 0.40--0.75, energy 0.001--0.024 is a strict order."""
    def exceeds(left, right):
        return holds(Relation(EXCEEDS, 0, 1), (left, right), universe)

    assert exceeds("runtime_seconds", "success_rate")
    assert exceeds("success_rate", "energy_error")
    assert not exceeds("success_rate", "runtime_seconds")
    assert not exceeds("energy_error", "success_rate")
    assert not exceeds("success_rate", "success_rate")  # and never itself


def test_relations_are_read_off_syntax_only():
    ordered = relations_from_request("フガ率がホゲ尺より大きいケースの平均", SURFACES)
    assert [r.kind for r in ordered] == [SAME_TYPE, EXCEEDS]
    assert (ordered[1].left, ordered[1].right) == (0, 1)
    swapped = relations_from_request("ホゲ尺がフガ率より大きいケース", SURFACES)
    assert (swapped[1].left, swapped[1].right) == (1, 0)
    listed = relations_from_request("フガ率とホゲ尺をそれぞれならして", SURFACES)
    assert [r.kind for r in listed] == [SAME_TYPE, DISTINCT]
    assert relations_from_request("フガ率をならして", SURFACES) == ()


def test_the_pair_resolves_where_neither_word_could(universe):
    """The reason the layer exists, stated as one assertion.

    Each surface's own evidence leaves the same two candidates. The order admits
    only one of the two pairings, so the assignment is unique while both
    independent marginals are not.
    """
    belief = JointBelief.prior(SURFACES, universe)
    evidence = (_number(), Observation("CONTRAST", "runtime_seconds"))
    for index in (0, 1):
        for observation in evidence:
            belief = belief.observe(index, observation)
    alone = independent(belief, {0: evidence, 1: evidence})
    assert alone == (frozenset({"success_rate", "energy_error"}),) * 2

    for relation in relations_from_request(
            "フガ率がホゲ尺より大きいケースの平均", SURFACES):
        belief = belief.relate(relation)
    assert belief.state == JOINT_RESOLVED
    assert belief.meanings == {"フガ率": "success_rate", "ホゲ尺": "energy_error"}


# --------------------------------------------------------------------------
# planning over the pair
# --------------------------------------------------------------------------

def test_a_relational_question_is_a_candidate_beside_the_per_word_ones(universe):
    belief = JointBelief.prior(SURFACES, universe)
    pool = tuple(p for p in available_probes(universe) if p.kind != "TYPE")
    kinds = {query.kind for query in joint_queries(belief, pool)}
    assert kinds == {"SURFACE", "RELATION"}


def test_planning_over_the_joint_can_beat_planning_over_each_word(report):
    """Whether it does is left to the planner; here it does, and by one question."""
    planning = report["planning"]
    assert planning["first_is_relational"]
    assert planning["questions_joint"] < planning["questions_surface_only"]
    assert planning["both_correct"]


def test_the_marginal_trap_is_escapable_by_asking_about_the_pair(universe):
    """What is undetermined is the correspondence, so that is what to ask about.

    With no per-word questions available at all, one relational question still
    resolves it -- which is the positive form of the same fact the trap states
    negatively.
    """
    belief = JointBelief(SURFACES, frozenset({
        ("runtime_seconds", "success_rate"), ("success_rate", "runtime_seconds")}),
        universe)
    assert belief.state == MARGINAL_ONLY
    settled, asked = resolve(belief, {"フガ率": "runtime_seconds",
                                      "ホゲ尺": "success_rate"}, ())
    assert asked == 1
    assert settled.state == JOINT_RESOLVED
    assert settled.meanings == {"フガ率": "runtime_seconds", "ホゲ尺": "success_rate"}


def test_resolution_stops_when_no_question_would_help(universe):
    """Two assignments no relation can separate: stop, do not pick."""
    belief = JointBelief(SURFACES, frozenset({
        ("success_rate", "success_rate"), ("energy_error", "energy_error")}),
        universe)
    settled, asked = resolve(belief, {"フガ率": "success_rate",
                                      "ホゲ尺": "success_rate"}, ())
    assert asked == 0
    assert settled.state == MARGINAL_ONLY and settled.meanings is None


# --------------------------------------------------------------------------
# the hold-out
# --------------------------------------------------------------------------

def test_the_invariant(report):
    assert report["metrics"]["false_joint_resolution_rate"] == 0.0


def test_the_gain_is_not_zero(report):
    """A solver that merely ran L8.29 twice would score zero here.

    The families are built so that being good at single-word inference cannot
    produce this number.
    """
    assert report["metrics"]["independent_vs_joint_gain"] == 2
    joint_only = [r for r in report["runs"] if r["joint_only"]]
    assert len(joint_only) == 2
    for run in joint_only:
        assert all(len(m) > 1 for m in run["independent"])


def test_the_marginal_trap_is_refused(report):
    runs = {run["name"]: run for run in report["runs"]}
    trap = runs["MARGINAL_TRAP"]
    assert trap["state"] == MARGINAL_ONLY
    assert trap["meanings"] is None
    assert len(trap["joint"]) == 2
    assert report["metrics"]["marginal_trap_abstention"] == 1.0


def test_the_controls_hold(report):
    runs = {run["name"]: run for run in report["runs"]}
    assert runs["INDEPENDENT_ENOUGH"]["solved_independently"]
    assert runs["INDEPENDENT_ENOUGH"]["meanings_correct"]
    assert runs["IMPOSSIBLE"]["state"] == CONTRADICTION
    assert report["metrics"]["joint_resolution_accuracy"] == 1.0


def test_two_words_acquired_at_once_change_what_executes(report):
    end = report["end_to_end"]
    assert end["meanings"] == {"フガ率": "success_rate", "ホゲ尺": "energy_error"}
    assert end["before"] == "ABSTAIN"
    assert end["after"] == "ANSWER"
    assert end["correct"] is True


def test_the_report_formats(report):
    from coding_world_benchmark.holdout_joint_l8341_experiment import format_experiment

    text = format_experiment(report)
    assert "independent_vs_joint_gain" in text and "MARGINAL_TRAP" in text
