import pytest

from coding_world_benchmark.cross_context_identification_l829_experiment import (
    Observation, run_acquired_query, synthetic_universe, universe_from,
)
from coding_world_benchmark.evidence_graph_l827_experiment import run_graph_query
from coding_world_benchmark.holdout_binding_l8211_experiment import build_holdout
from coding_world_benchmark.holdout_lexicon_l8281_experiment import widen_store
from coding_world_benchmark.holdout_revision_l8301_experiment import (
    SURFACE, TARGET, build_scenarios, generate_scenarios, run_experiment, run_scenario,
)
from coding_world_benchmark.lexical_revision_l830_experiment import (
    ACQUIRED, AUTHORITATIVE, DISPUTED, HYPOTHESIS, REVISED, REVOKED, LexicalBelief,
    RevisableLexicon, repairs,
)
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
def universe(store):
    return universe_from(store)


@pytest.fixture(scope="module")
def report():
    return run_experiment(24)


def _type():
    return Observation("TYPE", ValueType.NUMBER)


def stream(universe, *observations) -> LexicalBelief:
    belief = LexicalBelief(SURFACE, universe)
    for observation in observations:
        belief = belief.observe(observation)
    return belief


# --------------------------------------------------------------------------
# the rule the layer is built on
# --------------------------------------------------------------------------

def test_a_contradiction_suspends_and_does_not_rewrite(universe):
    """contradiction != immediate remapping.

    The newest observation is the one corroborated least, so it may not by itself
    replace a meaning. It withdraws authority and the belief waits.
    """
    belief = stream(universe, _type(), Observation("CONTRAST", "runtime_seconds"),
                    Observation("RANGE", 0.5))
    assert belief.state == ACQUIRED and belief.meaning == "success_rate"
    disputed = belief.observe(Observation("RANGE", 0.01))
    assert disputed.state == DISPUTED
    assert disputed.meaning is None
    assert not disputed.authoritative
    # and it still knows what it is disputing
    assert disputed.believed == "success_rate"


def test_one_disagreement_suspends_two_agreeing_ones_revise(universe):
    """The property the design wanted, emergent from cardinality alone.

    Nothing in the code counts to two. Discarding one observation is cheaper than
    discarding two, so a lone contradiction ties and suspends while a corroborated
    one wins outright.
    """
    base = (_type(), Observation("CONTRAST", "runtime_seconds"), Observation("RANGE", 0.5))
    once = stream(universe, *base, Observation("RANGE", 0.01))
    assert once.state == DISPUTED
    twice = once.observe(Observation("RANGE", 0.005))
    assert twice.state == REVISED and twice.meaning == "energy_error"


def test_corroborating_the_original_restores_it(universe):
    """The same shape with the other truth: maintenance, not correction."""
    base = (_type(), Observation("CONTRAST", "runtime_seconds"), Observation("RANGE", 0.5))
    disputed = stream(universe, *base, Observation("RANGE", 0.01))
    restored = disputed.observe(Observation("RANGE", 0.6))
    assert restored.state == REVISED and restored.meaning == "success_rate"


def test_only_acquired_and_revised_carry_authority(universe):
    for state in (ACQUIRED, REVISED):
        assert state in AUTHORITATIVE
    for state in (HYPOTHESIS, DISPUTED, REVOKED):
        assert state not in AUTHORITATIVE


def test_a_revision_produces_a_new_belief_base(universe):
    """The defect this caught, pinned.

    Without it the raw intersection stays empty forever, so the very next
    *agreeing* observation re-fires the contradiction and the belief oscillates
    REVISED -> DISPUTED -> REVISED without end. The discard is provisional: the
    full log is kept and the next contradiction reconsiders all of it.
    """
    base = (_type(), Observation("CONTRAST", "runtime_seconds"), Observation("RANGE", 0.5))
    revised = stream(universe, *base, Observation("RANGE", 0.01), Observation("RANGE", 0.6))
    assert revised.state == REVISED
    assert revised.discarded == ("RANGE(0.01)",)
    agreeing = revised.observe(Observation("RANGE", 0.7))
    assert agreeing.state == ACQUIRED and agreeing.meaning == "success_rate"
    assert agreeing.authoritative


def test_balanced_evidence_stays_suspended(universe):
    """A belief may be suspended forever, and that is the correct end state."""
    belief = stream(universe, _type(), Observation("CONTRAST", "runtime_seconds"),
                    Observation("RANGE", 0.5), Observation("RANGE", 0.01),
                    Observation("RANGE", 0.6), Observation("RANGE", 0.005))
    assert belief.state == DISPUTED and not belief.authoritative


def test_revoked_when_the_old_meaning_is_ruled_out_with_no_replacement(universe):
    belief = stream(universe, _type(), Observation("RANGE", 0.5),
                    Observation("CONTRAST", "success_rate"),
                    Observation("CONTRAST", "success_rate"))
    assert belief.state == REVOKED
    assert belief.meaning is None and belief.believed == "success_rate"
    assert sorted(belief.repaired) == ["energy_error", "runtime_seconds"]


def test_revision_never_fires_without_a_contradiction(universe):
    belief = stream(universe, _type(), Observation("CONTRAST", "runtime_seconds"),
                    Observation("RANGE", 0.5), Observation("RANGE", 0.6),
                    Observation("RANGE", 0.7))
    assert belief.state == ACQUIRED and belief.meaning == "success_rate"
    assert all("DISPUTED" not in entry for entry in belief.history)


def test_repairs_are_cardinality_maximal(universe):
    observations = (_type(), Observation("RANGE", 0.5), Observation("RANGE", 0.01))
    found = repairs(observations, universe)
    assert found and all(len(kept) == len(observations) - 1 for kept, _ in found)
    assert len(found) == 2  # a tie, which is why this state suspends


# --------------------------------------------------------------------------
# the executor sees only authoritative beliefs
# --------------------------------------------------------------------------

def test_the_lexicon_hides_everything_that_is_not_authoritative(universe):
    lexicon = RevisableLexicon(universe)
    for observation in (_type(), Observation("CONTRAST", "runtime_seconds"),
                        Observation("RANGE", 0.5)):
        lexicon.observe(SURFACE, observation)
    assert lexicon.lookup(f"…{SURFACE}を…") == "success_rate"
    lexicon.observe(SURFACE, Observation("RANGE", 0.01))
    assert lexicon.lookup(f"…{SURFACE}を…") is None


def test_the_executor_is_unchanged_by_revision(store, sensors, universe):
    """``run_acquired_query`` is L8.29's and never learns that revision exists."""
    lexicon = RevisableLexicon(universe)
    assert run_acquired_query(store, TARGET, lexicon, *sensors).decision == \
           run_graph_query(store, TARGET, *sensors).decision


# --------------------------------------------------------------------------
# the hold-out
# --------------------------------------------------------------------------

def test_the_invariants_hold_everywhere(report):
    """Neither of these may move, in any suite."""
    for key in ("scenario_metrics",):
        assert report[key]["wrong_answer_during_dispute"] == 0
        assert report[key]["irreversible_false_learning_rate"] == 0.0
    for sweep in report["sweeps"].values():
        assert sweep["metrics"]["wrong_answer_during_dispute"] == 0
        assert sweep["metrics"]["irreversible_false_learning_rate"] in (0.0, "n/a")


def test_both_halves_are_present_and_both_work(report):
    """Maintenance and correction, same shape, different truth."""
    runs = {run["name"]: run for run in report["scenarios"]}
    assert runs["MAINTAIN"]["ended_correct"] and not runs["MAINTAIN"]["started_wrong"]
    assert runs["CORRECT"]["ended_correct"] and runs["CORRECT"]["started_wrong"]
    assert runs["STALEMATE"]["disputed"] and not runs["STALEMATE"]["authoritative"]
    assert runs["REVOKE"]["revoked"] and not runs["REVOKE"]["authoritative"]
    assert runs["CLEAN"]["ended_correct"] and not runs["CLEAN"]["had_contradiction"]


def test_authority_is_withdrawn_in_the_same_step(report):
    assert report["scenario_metrics"]["time_to_suspend"] == 0.0
    assert report["scenario_metrics"]["contradiction_detection_rate"] == 1.0
    assert report["sweeps"]["disjoint"]["metrics"]["contradiction_detection_rate"] == 1.0


def test_revision_and_reacquisition_are_accurate(report):
    assert report["scenario_metrics"]["revision_accuracy"] == 1.0
    disjoint = report["sweeps"]["disjoint"]["metrics"]
    assert disjoint["revision_accuracy"] == 1.0
    assert disjoint["correct_reacquisition_rate"] == 1.0
    # In the real universe REVOKE deliberately ends without a replacement, so one
    # of the two runs that started wrong does not reacquire. That is a correct
    # refusal, and ``false_revocation_rate`` 0.0 says no revocation was wrong.
    assert report["scenario_metrics"]["correct_reacquisition_rate"] == 0.5
    assert report["scenario_metrics"]["false_revocation_rate"] == 0.0


def test_undetectable_noise_is_reported_as_identifiability_not_revision(report):
    """L8.29's lesson again: noise you cannot detect is noise you cannot revise.

    In the banded sweep a stray observation can be consistent with the truth, so
    it never contradicts and nothing can be revised. That shows up as
    ``undetected_noise_rate``, and it shrinks as the universe grows because a
    wider universe makes a stray value less likely to land on a single survivor.
    """
    banded = report["sweeps"]["banded"]["by_size"]
    assert banded["K3"]["undetected_noise_rate"] > banded["K10"]["undetected_noise_rate"]
    assert report["sweeps"]["disjoint"]["metrics"]["undetected_noise_rate"] == 0.0


def test_the_generated_sweep_slides_the_noise(report):
    scenarios = generate_scenarios(sizes=(3,))
    assert len({tuple(o.payload for o in s.stream) for s in scenarios}) == len(scenarios)
    assert len({s.truth for s in scenarios}) == 3


def test_every_scenario_queries_after_every_observation(report):
    for run in report["scenarios"]:
        assert all(step["decision"] in ("ANSWER", "ABSTAIN", "NOT_AN_OPERATION")
                   for step in run["steps"])
        for step in run["steps"]:
            if not step["authoritative"]:
                assert step["decision"] != "ANSWER", (run["name"], step)


def test_the_report_formats(report):
    from coding_world_benchmark.holdout_revision_l8301_experiment import format_experiment

    text = format_experiment(report)
    assert "wrong_answer_during_dispute" in text and "REVOKE" in text
