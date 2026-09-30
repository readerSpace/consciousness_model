import pytest

from coding_world_benchmark.evidence_arbitration_l824_experiment import (
    WHOLE_STRONG, SlotEvidence, arbitrate, gather_evidence, run_arbitrated_query,
)
from coding_world_benchmark.holdout_arbitration_l8241_experiment import run_experiment
from coding_world_benchmark.holdout_binding_l8211_experiment import build_holdout
from coding_world_benchmark.query_algebra_l820_experiment import canonical_render
from coding_world_benchmark.semantic_proposer_l822_experiment import Proposer
from coding_world_benchmark.span_grounded_proposer_l823_experiment import SpanProposer

DILUTED = "64x64で走らせた分のランタイムの値をアベレージ"
COMPLEMENTARY = "64x64のケースの成功の比率をならした値は？"


@pytest.fixture(scope="module")
def store():
    return build_holdout(24)[0]


@pytest.fixture(scope="module")
def sensors():
    return SpanProposer.trained(), Proposer.trained()


@pytest.fixture(scope="module")
def report():
    return run_experiment(24)


def test_a_contended_sentence_reading_is_not_evidence(store, sensors):
    # MAX 0.516 against MEAN 0.484 is two labels arguing over one string. Adding
    # it back beside the span's MEAN is what a union would do, and it restores
    # the ambiguity grounding resolved.
    spans, sentence = sensors
    evidence = {(e.slot, e.candidate): e for e in gather_evidence(DILUTED, spans, sentence)}
    contended = evidence[("OPERATOR", "MAX")]

    assert contended.whole_score < WHOLE_STRONG
    assert contended.span_coverage == 0.0
    assert not contended.admitted


def test_a_decisive_sentence_reading_is_evidence(store, sensors):
    spans, sentence = sensors
    evidence = {(e.slot, e.candidate): e for e in gather_evidence(COMPLEMENTARY, spans, sentence)}
    field = evidence[("FIELD", "success_rate")]

    assert field.span_coverage == 0.0
    assert field.whole_score >= WHOLE_STRONG
    assert field.sources == ("whole",)


def test_a_program_can_be_assembled_from_two_sensors(store, sensors):
    """The case neither sensor could do alone.

    The operator is visible only to the span (「ならした」) and the field only to
    the sentence (「成功の比率」 breaks the n-grams that 成功比率 trained), so the
    program exists only if the two are kept side by side instead of one being
    chosen.
    """
    spans, sentence = sensors
    operators, fields = arbitrate(gather_evidence(COMPLEMENTARY, spans, sentence))

    assert operators == ("MEAN",)
    assert fields == ("success_rate",)

    result = run_arbitrated_query(store, COMPLEMENTARY, spans, sentence)
    assert result.decision == "ANSWER"
    assert result.source == "span+whole"
    assert canonical_render(result.expr) == (
        "MEAN(PROJECT(FILTER(episodes, lattice_size == 64x64), success_rate))"
    )


def test_conflicting_sensors_both_stay_in_the_candidate_set():
    """Arbitration on constructed evidence, because natural conflicts are rare.

    Two sensors reading the same string mostly agree or both fall silent, so a
    rule only ever exercised on agreement has not been exercised. Given a span
    that says MAX strongly and a sentence that says MEAN strongly, neither wins:
    both are admitted, two well-typed programs survive downstream, and the
    uniqueness rule refuses.
    """
    conflict = (
        SlotEvidence("OPERATOR", "MAX", 1.0, 1.0, 0.10),
        SlotEvidence("OPERATOR", "MEAN", 0.0, 0.0, 0.90),
    )
    operators, _ = arbitrate(conflict)

    assert set(operators) == {"MAX", "MEAN"}


def test_two_admitted_operators_refuse(store, sensors, report):
    spans, sentence = sensors
    result = run_arbitrated_query(store, "64x64のならならアベレのピークの値", spans, sentence)

    assert result.decision == "ABSTAIN"
    assert result.value is None


def test_arbitration_solves_everything_both_sensors_solve(report):
    complementarity = report["complementarity"]

    # Each sensor alone answers one query the other cannot.
    assert complementarity["l822\\l823"] == 1
    assert complementarity["l823\\l822"] == 1
    # Arbitration loses nothing from either and gains from both.
    assert complementarity["l822\\l824"] == 0
    assert complementarity["l823\\l824"] == 0
    assert complementarity["l824\\l822"] >= 1
    assert complementarity["l824\\l823"] >= 1


def test_coverage_rises_above_both_sensors(report):
    runs = report["runs"]

    assert runs["l824"]["metrics"]["execution_accuracy"] > runs["l822"]["metrics"]["execution_accuracy"]
    assert runs["l824"]["metrics"]["execution_accuracy"] > runs["l823"]["metrics"]["execution_accuracy"]
    assert runs["l824"]["metrics"]["safe_coverage_gain"] > runs["l822"]["metrics"]["safe_coverage_gain"]


def test_safety_is_unchanged_for_every_pipeline(report):
    for run in report["runs"].values():
        assert run["metrics"]["wrong_answer_rate"] == 0.0
        assert run["metrics"]["unsafe_execution_rate"] == 0.0
        assert run["metrics"]["refusal_accuracy"] == 1.0


def test_requests_the_cue_tables_read_never_reach_a_sensor(store, sensors, report):
    spans, sentence = sensors
    result = run_arbitrated_query(store, "64x64を使った実験の平均実行時間は？", spans, sentence)

    assert result.decision == "ANSWER"
    assert result.source == "symbolic"


def test_controls_survive(store, sensors):
    spans, sentence = sensors

    assert run_arbitrated_query(store, "さっきの続きをやって", spans, sentence).decision == "NOT_AN_OPERATION"
    gate = run_arbitrated_query(store, "実行時間の中央値は？", spans, sentence)
    assert gate.decision == "ABSTAIN" and gate.stage == "gate"


def test_results_are_deterministic():
    left, right = run_experiment(16), run_experiment(16)

    assert left["complementarity"] == right["complementarity"]
    assert left["runs"]["l824"]["metrics"] == right["runs"]["l824"]["metrics"]
