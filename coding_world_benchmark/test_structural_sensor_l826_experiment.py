import inspect

import pytest

from coding_world_benchmark.evidence_arbitration_l824_experiment import (
    SlotEvidence, run_arbitrated_query,
)
from coding_world_benchmark.holdout_binding_l8211_experiment import build_holdout
from coding_world_benchmark.holdout_structural_l8261_experiment import run_experiment
from coding_world_benchmark.query_algebra_l820_experiment import canonical_render
from coding_world_benchmark.semantic_proposer_l822_experiment import (
    FIELD_FORMS, OPERATOR_FORMS, Proposer,
)
from coding_world_benchmark.span_grounded_proposer_l823_experiment import SpanProposer
from coding_world_benchmark.structural_sensor_l826_experiment import (
    FusedEvidence, StructuralSensor, arbitrate_fused, fuse, run_structural_query,
)

TWO_FIELDS = "64x64のケースで処理時間ではなく成功比率をならした"
CONFLICT = "64x64のケースの成功比率を処理時間のかわりにならした"


@pytest.fixture(scope="module")
def store():
    return build_holdout(24)[0]


@pytest.fixture(scope="module")
def sensors():
    return SpanProposer.trained(), Proposer.trained(), StructuralSensor()


@pytest.fixture(scope="module")
def report():
    return run_experiment(24)


def test_the_structural_sensor_knows_no_labels():
    """The property that makes it a different sensor rather than a second opinion.

    If it could name MEAN or success_rate it would be doing the lexical sensors'
    job with extra steps; its whole contribution is saying which candidate the
    head governs without knowing what any of them mean.
    """
    source = inspect.getsource(StructuralSensor)
    labels = set(OPERATOR_FORMS) | set(FIELD_FORMS)

    assert labels
    assert not any(label in source for label in labels)
    forms = {form for split in OPERATOR_FORMS.values() for form in split["train"]}
    assert not any(form in source for form in forms)


def test_attachment_strength_is_graded_not_bimodal(report, sensors):
    # The lexical scores sit near 1.0 or near 0.5, which is why calibration had
    # nothing to do in L8.25. Attachment varies continuously with distance.
    values = report["structure_score_values"]

    assert len(values) >= 4
    assert min(values) < 0.5 < max(values)


def test_structure_resolves_what_the_lexical_pair_cannot(store, sensors):
    spans, sentence, structure = sensors

    assert run_arbitrated_query(store, TWO_FIELDS, spans, sentence).decision == "ABSTAIN"

    result = run_structural_query(store, TWO_FIELDS, spans, sentence, structure)
    assert result.decision == "ANSWER"
    assert canonical_render(result.expr) == (
        "MEAN(PROJECT(FILTER(episodes, lattice_size == 64x64), success_rate))"
    )


def test_the_head_governs_the_nearest_accusative_argument(sensors):
    _, _, structure = sensors
    binds = structure.bindings(TWO_FIELDS)

    assert binds[0].text == "成功比率"
    assert binds[0].strength > binds[1].strength


def test_a_dominated_candidate_is_dropped(sensors):
    spans, sentence, structure = sensors
    fused = {
        (item.slot, item.candidate): item for item in fuse(TWO_FIELDS, spans, sentence, structure)
    }
    winner = fused[("FIELD", "success_rate")]
    loser = fused[("FIELD", "runtime_seconds")]

    assert loser.sources < winner.sources  # strict subset
    _, fields = arbitrate_fused(tuple(fused.values()))
    assert fields == ("success_rate",)


def test_equal_but_disagreeing_support_is_kept():
    # Dominance is a strict-subset test, so two candidates backed by the same
    # sensors both survive and the uniqueness rule refuses. Nothing outranks.
    left = FusedEvidence("FIELD", "runtime_seconds", SlotEvidence("FIELD", "runtime_seconds", 1.0, 1.0, 0.0), 1.0)
    right = FusedEvidence("FIELD", "success_rate", SlotEvidence("FIELD", "success_rate", 1.0, 1.0, 0.0), 0.97)

    _, fields = arbitrate_fused((left, right))
    assert set(fields) == {"runtime_seconds", "success_rate"}


def test_conflict_is_now_refused_on_a_real_sentence(store, sensors):
    """The cell L8.24 could only reach with constructed evidence.

    Two lexical sensors reading one string agree or fall silent together, so
    disagreement had to be synthesised. A sensor that reads relations instead of
    vocabulary disagrees on its own: here both fields are strongly attached,
    neither dominates, and the request is refused.
    """
    spans, sentence, structure = sensors
    fused = [item for item in fuse(CONFLICT, spans, sentence, structure) if item.slot == "FIELD"]

    assert len(fused) == 2
    assert all("structure" in item.sources for item in fused)

    result = run_structural_query(store, CONFLICT, spans, sentence, structure)
    assert result.decision == "ABSTAIN"
    assert result.value is None


def test_structure_contributes_queries_nothing_else_can(report):
    contribution = report["contribution"]

    assert len(contribution["structure"]) >= 2
    assert all(text in report["runs"]["l826"]["solved"] for text in contribution["structure"])


def test_the_whole_sentence_sensor_contributes_nothing_here(report):
    # Reported as measured: on this probe set every query the sentence reading
    # carries is also carried by a span, so its unique contribution is empty.
    assert report["contribution"]["whole"] == []


def test_coverage_rises_without_losing_safety(report):
    before = report["runs"]["l824"]["metrics"]
    after = report["runs"]["l826"]["metrics"]

    assert after["execution_accuracy"] > before["execution_accuracy"]
    assert after["refusal_accuracy"] == 1.0
    assert after["wrong_answer_rate"] == 0.0
    assert after["unsafe_execution_rate"] == 0.0


def test_controls_survive(store, sensors):
    spans, sentence, structure = sensors

    assert run_structural_query(store, "さっきの続きをやって", spans, sentence, structure).decision == "NOT_AN_OPERATION"
    gate = run_structural_query(store, "実行時間の中央値は？", spans, sentence, structure)
    assert gate.decision == "ABSTAIN" and gate.stage == "gate"
    symbolic = run_structural_query(store, "64x64を使った実験の平均実行時間は？", spans, sentence, structure)
    assert symbolic.source == "symbolic"


def test_results_are_deterministic():
    left, right = run_experiment(16), run_experiment(16)

    assert left["contribution"] == right["contribution"]
    assert left["runs"]["l826"]["metrics"] == right["runs"]["l826"]["metrics"]
