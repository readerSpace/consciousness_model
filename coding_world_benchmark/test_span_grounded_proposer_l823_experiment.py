import pytest

from coding_world_benchmark.holdout_binding_l8211_experiment import build_holdout
from coding_world_benchmark.holdout_span_l8231_experiment import run_holdout
from coding_world_benchmark.query_algebra_l820_experiment import canonical_render
from coding_world_benchmark.semantic_proposer_l822_experiment import (
    Proposer, assert_holdout_is_unseen, run_proposed_query,
)
from coding_world_benchmark.span_grounded_proposer_l823_experiment import (
    MIN_SPAN, SPAN_COVERAGE, SpanProposer, build_aligned_training, run_span_query,
)


@pytest.fixture(scope="module")
def store():
    return build_holdout(24)[0]


@pytest.fixture(scope="module")
def spans():
    return SpanProposer.trained()


@pytest.fixture(scope="module")
def sentence():
    return Proposer.trained()


@pytest.fixture(scope="module")
def cold():
    return run_holdout(24, grounded=True)


@pytest.fixture(scope="module")
def cold_base():
    return run_holdout(24, grounded=False)


DILUTED = "64x64で走らせた分のランタイムの値をアベレージ"


def test_span_labels_come_from_the_renderer_not_from_annotation():
    operators, fields = build_aligned_training()

    assert ("ならす", "MEAN") in operators
    assert ("ランタイム", "runtime_seconds") in fields
    # Every pair is a (span, label) the renderer emitted; none was written by hand.
    assert all(len(span) <= 12 for span, _ in operators + fields)
    assert_holdout_is_unseen()


def test_evidence_is_localised(spans):
    detected = {(span.text, span.kind) for span in spans.detect(DILUTED)}

    assert ("ランタイム", "FIELD") in detected
    assert ("アベレージ", "OPERATOR") in detected


def test_coverage_is_required_not_just_a_relative_score(spans):
    # The classifier's score is relative across labels, so a span with one known
    # n-gram scores 1.0 when only one label matches. Absolute coverage is what
    # keeps 「で走らせた分のラン」 out.
    assert spans.field_model.coverage("ランタイム", "runtime_seconds") >= SPAN_COVERAGE
    assert spans.field_model.coverage("で走らせた分のラン", "runtime_seconds") < SPAN_COVERAGE
    assert all(len(span.text) >= MIN_SPAN for span in spans.detect(DILUTED))


def test_short_fragments_are_excluded_by_length(spans):
    # 「の値」 is fully covered by MAX (頭打ちの値 is a training form) and carries no
    # evidence; coverage alone cannot reject it, so the length floor does.
    assert spans.operator_model.coverage("の値", "MAX") == 1.0
    assert not any(span.text == "の値" for span in spans.detect(DILUTED))


def test_grounding_resolves_the_case_that_dilution_refused(store, spans, sentence):
    assert run_proposed_query(store, DILUTED, sentence).decision == "ABSTAIN"

    result = run_span_query(store, DILUTED, spans, sentence)
    assert result.decision == "ANSWER"
    assert result.source == "span"
    assert canonical_render(result.expr) == (
        "MEAN(PROJECT(FILTER(episodes, lattice_size == 64x64), runtime_seconds))"
    )


def test_thin_overlap_still_falls_back_to_the_sentence(store, spans, sentence):
    # Span evidence is stricter, so a wording that no span covers would be lost
    # if grounding simply replaced the sentence reading.
    result = run_span_query(store, "64x64のケースの数を教えて", spans, sentence)

    assert result.decision == "ANSWER"
    assert result.source == "sentence"


def test_a_span_that_keeps_two_labels_still_refuses(store, spans, sentence):
    result = run_span_query(store, "64x64のケースの処理時間をピークをならして", spans, sentence)

    assert result.decision == "ABSTAIN"
    assert result.value is None


def test_controls_survive(store, spans, sentence):
    assert run_span_query(store, "さっきの続きをやって", spans, sentence).decision == "NOT_AN_OPERATION"
    gate = run_span_query(store, "実行時間の中央値は？", spans, sentence)
    assert gate.decision == "ABSTAIN" and gate.stage == "gate"
    assert run_span_query(store, "64x64を使った実験の平均実行時間は？", spans, sentence).source == "symbolic"


def test_safety_is_unchanged_on_fresh_probes(cold, cold_base):
    for metrics in (cold["metrics"], cold_base["metrics"]):
        assert metrics["wrong_answer_rate"] == 0.0
        assert metrics["unsafe_execution_rate"] == 0.0
        assert metrics["refusal_accuracy"] == 1.0


def test_grounding_changes_which_cases_work_rather_than_how_many(cold, cold_base):
    """The cold result, stated as measured rather than as hoped.

    On fresh probes the two proposers tie: execution accuracy and
    safe_coverage_gain are identical, and only the attribution differs
    (span_grounded_rate 0.6 against 0.0). Grounding fixed the dilution case and
    broke a different one, so the honest summary is that it changed which
    requests work, not how many.
    """
    assert cold["metrics"]["execution_accuracy"] == cold_base["metrics"]["execution_accuracy"]
    assert cold["metrics"]["safe_coverage_gain"] == cold_base["metrics"]["safe_coverage_gain"]
    assert cold["metrics"]["span_grounded_rate"] > cold_base["metrics"]["span_grounded_rate"]


def test_the_two_witnesses_are_complementary_per_slot(cold, spans, sentence):
    """Why the tie happened, asserted so the next layer starts from the cause.

    In 「64x64のケースの成功の比率をならした値は？」 the spans find the operator and no
    field -- 成功の比率 breaks the n-grams that 成功比率 trained -- while the
    sentence finds both. Precedence is currently applied to the whole proposal,
    so the span reading wins and takes its empty field set with it. Applying
    precedence per slot is the obvious next move and is left to be measured on a
    new hold-out rather than tuned against this one.
    """
    text = "64x64のケースの成功の比率をならした値は？"
    span_operators, span_fields = spans.propose(text)

    assert span_operators == ("MEAN",)
    assert span_fields == ()
    assert sentence.fields(text) == ("success_rate",)

    failure = next(
        r for r in cold["records"]
        if r["text"] == text
    )
    assert failure["decision"] == "ABSTAIN"
    assert not failure["wrong"]


def test_holdout_is_deterministic():
    assert run_holdout(16)["metrics"] == run_holdout(16)["metrics"]
