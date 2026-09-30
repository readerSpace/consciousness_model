import inspect

import pytest

from coding_world_benchmark.evidence_graph_l827_experiment import run_graph_query
from coding_world_benchmark.holdout_binding_l8211_experiment import build_holdout
from coding_world_benchmark.holdout_lexicon_l8281_experiment import (
    INVENTED, assert_zero_overlap, build_probes, generate_probes, run_experiment,
    run_suite, widen_store,
)
from coding_world_benchmark.lexical_hypothesis_l828_experiment import (
    admitted_by_type, excluded_by_contrast, propose_hypotheses, run_hypothesis_query,
    subword_overlap,
)
from coding_world_benchmark.semantic_proposer_l822_experiment import (
    FIELD_FORMS, OPERATOR_FORMS, Proposer,
)
from coding_world_benchmark.span_grounded_proposer_l823_experiment import SpanProposer
from coding_world_benchmark.structural_sensor_l826_experiment import StructuralSensor

#: A word the system has never seen, in a construction that excludes its sibling.
ZERO_CONTRAST = "64x64のケースで処理時間ではなくフガ率をならして"
#: The same word with no contrastive relation to lean on.
ZERO_BARE = "処理時間を見ながら64x64のケースのフガ率をならして"
OVERLAP_CONTRAST = "64x64のケースで処理時間ではなく成功の比率をならして"
READABLE = "64x64のケースの処理時間をならす形に"


@pytest.fixture(scope="module")
def store():
    return build_holdout(24)[0]


@pytest.fixture(scope="module")
def sensors():
    return SpanProposer.trained(), Proposer.trained(), StructuralSensor()


@pytest.fixture(scope="module")
def report():
    return run_experiment(24)


# --------------------------------------------------------------------------
# the separation this layer is for
# --------------------------------------------------------------------------

def test_generation_cannot_change_a_single_decision(store, sensors):
    """Generation and promotion are split, and the split is control flow.

    Every probe runs through both paths and the decision must be identical --
    not merely usually identical. A hypothesis that could move an answer would
    make the two capabilities unmeasurable apart, which is the failure this
    layer exists to avoid.
    """
    for probe in build_probes() + generate_probes():
        baseline = run_graph_query(store, probe.text, *sensors)
        outcome = run_hypothesis_query(store, probe.text, *sensors)
        assert outcome.result.decision == baseline.decision, probe.text
        assert outcome.result.reason == baseline.reason, probe.text


def test_nothing_is_written_back_to_the_lexicon(store, sensors):
    """The classifiers and the form banks are inputs here, never outputs."""
    spans = sensors[0]
    before = {label: dict(counts) for label, counts in spans.field_model.counts.items()}
    forms = {label: dict(split) for label, split in FIELD_FORMS.items()}
    operators = {label: dict(split) for label, split in OPERATOR_FORMS.items()}
    for probe in build_probes():
        run_hypothesis_query(store, probe.text, *sensors)
    assert {label: dict(c) for label, c in spans.field_model.counts.items()} == before
    assert {label: dict(s) for label, s in FIELD_FORMS.items()} == forms
    assert {label: dict(s) for label, s in OPERATOR_FORMS.items()} == operators


def test_no_hypothesis_where_nothing_was_unnamed(store, sensors):
    """A generator that speculates about phrases the sensors read is noise."""
    assert run_hypothesis_query(store, READABLE, *sensors).hypotheses == ()
    assert run_hypothesis_query(store, "さっきの続きをやって", *sensors).hypotheses == ()


# --------------------------------------------------------------------------
# the central claim: structure, not spelling
# --------------------------------------------------------------------------

def test_the_invented_forms_carry_no_subword_evidence():
    """Checked, not asserted -- otherwise the ZERO cells prove nothing."""
    assert_zero_overlap()
    spans = SpanProposer.trained()
    for forms in INVENTED.values():
        for form in forms:
            assert subword_overlap(form, spans) == {}


def test_an_unseen_word_is_named_from_structure_alone(store, sensors):
    outcome = run_hypothesis_query(store, ZERO_CONTRAST, *sensors)
    assert outcome.result.decision == "ABSTAIN"
    assert outcome.decidable
    assert outcome.top.label == "success_rate"
    assert outcome.top.overlap == 0.0
    assert "subword" not in outcome.top.sources


def test_removing_spelling_does_not_cost_decidability(report):
    """The ablation that answers what the next layer needs to know."""
    full = report["generated"]["metrics"]
    without = report["ablations"]["subword"]["metrics"]
    assert without["decidable_rate"] == full["decidable_rate"]
    assert without["decidable_precision"] == full["decidable_precision"]


def test_removing_the_type_hole_or_the_contrast_does(report):
    full = report["generated"]["metrics"]
    for source in ("type", "contrast", "attestation"):
        assert report["ablations"][source]["metrics"]["decidable_rate"] < full["decidable_rate"]


def test_the_sources_are_label_free(store, sensors):
    """Type and contrast read the graph and the particles, never a vocabulary."""
    source = inspect.getsource(excluded_by_contrast) + inspect.getsource(admitted_by_type)
    for form in ("ならす", "成功比率", "処理時間", "サクセスレート", "ピーク"):
        assert form not in source


# --------------------------------------------------------------------------
# where it stops working, measured rather than caveated
# --------------------------------------------------------------------------

def test_a_third_number_field_collapses_decidability(report):
    """The honest limit.

    With two attested Number fields, "exclude one" and "name the other" are the
    same operation, so decidable_precision 1.0 was partly counting the schema.
    Attesting a third collapses decidability to zero -- and recall stays 1.0, so
    the generator degrades into indecision rather than into error. That is the
    safe direction, and it says precisely what promotion still needs: evidence
    that separates candidates *within* the type-admissible pool.
    """
    narrow = report["generated"]["metrics"]
    wide = report["widened"]["metrics"]
    assert narrow["decidable_rate"] > 0.0
    assert wide["decidable_rate"] == 0.0
    assert wide["hypothesis_recall"] == 1.0


def test_bare_zero_overlap_refuses_to_be_decidable(store, sensors):
    """No spelling and no contrast leaves nothing to choose with, and it doesn't."""
    outcome = run_hypothesis_query(store, ZERO_BARE, *sensors)
    assert len(outcome.hypotheses) == 2
    assert not outcome.decidable
    assert {item.label for item in outcome.hypotheses} == {"runtime_seconds", "success_rate"}


def test_widening_is_a_real_change(store):
    assert "energy_error" not in {key for values in store.fields.values() for key in values}
    widened = widen_store(store)
    assert all("energy_error" in values for values in widened.fields.values())


# --------------------------------------------------------------------------
# the hold-out
# --------------------------------------------------------------------------

def test_the_headline_numbers(report):
    metrics = report["generated"]["metrics"]
    assert metrics["hypothesis_recall"] == 1.0
    assert metrics["false_hypothesis_rate"] == 0.0
    assert metrics["execution_invariance"] == 1.0
    assert metrics["wrong_answer_rate"] == 0.0
    assert metrics["decidable_precision"] == 1.0


def test_every_cell_is_populated(report):
    cells = report["generated_by_family"]
    for name in ("OVERLAP_CONTRAST", "OVERLAP_BARE", "ZERO_CONTRAST", "ZERO_BARE"):
        assert cells[name]["hypothesis_recall"] == 1.0


def test_contrast_excludes_only_a_labelled_sibling(store, sensors):
    spans, sentence, structure = sensors
    from coding_world_benchmark.evidence_graph_l827_experiment import build_graph

    graph = build_graph(OVERLAP_CONTRAST, store, spans, sentence, structure)
    assert excluded_by_contrast(OVERLAP_CONTRAST, graph, structure) == {"runtime_seconds"}
    bare = build_graph(ZERO_BARE, store, spans, sentence, structure)
    assert excluded_by_contrast(ZERO_BARE, bare, structure) == frozenset()


def test_hypotheses_are_ranked_by_evidence_not_by_name(store, sensors):
    outcome = run_hypothesis_query(store, "処理時間を見ながら64x64のケースの成功の比率をならして", *sensors)
    assert outcome.top.label == "success_rate"
    assert "subword" in outcome.top.sources


def test_the_report_formats(report):
    from coding_world_benchmark.holdout_lexicon_l8281_experiment import format_experiment

    text = format_experiment(report)
    assert "decidable_precision" in text and "3 Number fields" in text


def test_the_suite_runs_with_every_source_disabled(store):
    run = run_suite(generate_probes(), 24, disable=("type", "contrast", "attestation", "subword"))
    assert run["metrics"]["execution_invariance"] == 1.0
