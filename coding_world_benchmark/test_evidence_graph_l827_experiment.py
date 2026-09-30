import pytest

from coding_world_benchmark.evidence_graph_l827_experiment import (
    EvidenceGraph, build_graph, needs_field, phrases, readings, run_graph_query,
)
from coding_world_benchmark.holdout_binding_l8211_experiment import build_holdout
from coding_world_benchmark.holdout_graph_l8271_experiment import (
    build_probes, generate_probes, run_experiment,
)
from coding_world_benchmark.query_algebra_l820_experiment import canonical_render
from coding_world_benchmark.semantic_proposer_l822_experiment import Proposer
from coding_world_benchmark.span_grounded_proposer_l823_experiment import SpanProposer
from coding_world_benchmark.structural_sensor_l826_experiment import (
    StructuralSensor, run_structural_query,
)

#: The sentence this layer exists for: the head governs 「成功の比率」, which no
#: lexical sensor can read, while the only legible field is the excluded one.
UNNAMED = "64x64のケースでランタイムの値ではなく成功の比率をならして"
OBLIQUE = "64x64のケースの成功比率を処理時間のかわりにならした"
COORDINATED = "64x64のケースで処理時間と成功比率をならした"
MODIFIER = "64x64のケースのサクセスレートの値をアベレージ"


@pytest.fixture(scope="module")
def store():
    return build_holdout(24)[0]


@pytest.fixture(scope="module")
def sensors():
    return SpanProposer.trained(), Proposer.trained(), StructuralSensor()


@pytest.fixture(scope="module")
def report():
    return run_experiment(24)


def graph_for(store, sensors, text) -> EvidenceGraph:
    return build_graph(text, store, *sensors)


# --------------------------------------------------------------------------
# the representation
# --------------------------------------------------------------------------

def test_nodes_are_grounded_so_the_same_label_twice_is_two_nodes(store, sensors):
    """A set keys on the label; the graph keys on the position.

    This is the whole difference. Without it there is nowhere to record that the
    head governs *this* occurrence and not that one.
    """
    graph = graph_for(store, sensors, OBLIQUE)
    fields = [node for node in graph.of_kind("FIELD") if node.start >= 0]
    assert fields, "the span sensor read at least one field"
    assert all(node.start >= 0 and node.end > node.start for node in fields)


def test_every_edge_records_which_sensors_witnessed_it(store, sensors):
    graph = graph_for(store, sensors, MODIFIER)
    bind = graph.binds()
    assert bind, "a bind edge was built"
    assert "structure" in bind[0].witnesses
    # The lexical sensors that labelled the argument are on the edge too, so the
    # edge says who supported it and not merely that something did.
    assert bind[0].witnesses & {"span", "whole"}


def test_the_graph_renders_the_shape_the_layer_is_named_for(store, sensors):
    drawing = graph_for(store, sensors, MODIFIER).render()
    assert "--bind[" in drawing and "--filter[" in drawing
    assert "success_rate" in drawing and "64x64" in drawing


# --------------------------------------------------------------------------
# what the edges buy
# --------------------------------------------------------------------------

def test_an_unnamed_governed_argument_is_refused_not_substituted(store, sensors):
    """The failure that motivated the layer, stated as a test.

    L8.24 and L8.26 answer this with the *excluded* field, confidently and with a
    well-typed program. The evidence to refuse was always present; only a set
    could not hold it.
    """
    before = run_structural_query(store, UNNAMED, *sensors)
    after = run_graph_query(store, UNNAMED, *sensors)
    assert before.decision == "ANSWER"
    assert "runtime_seconds" in canonical_render(before.expr)
    assert after.decision == "ABSTAIN"
    assert "成功の比率" in after.reason


def test_accusative_marking_beats_proximity(store, sensors):
    """「Aを Bのかわりに ならす」 is about A, however near B sits to the head."""
    result = run_graph_query(store, OBLIQUE, *sensors)
    assert result.decision == "ANSWER"
    assert "success_rate" in canonical_render(result.expr)


def test_a_modifier_chain_is_one_argument(store, sensors):
    """Without の-merging the light noun 「値」 wins every attachment contest."""
    segmented = phrases(StructuralSensor(), MODIFIER)
    assert any("サクセスレート" in phrase.text and "値" in phrase.text for phrase in segmented)
    result = run_graph_query(store, MODIFIER, *sensors)
    assert result.decision == "ANSWER"
    assert "success_rate" in canonical_render(result.expr)


def test_a_coordinated_argument_is_refused(store, sensors):
    """The algebra has no plural projection node, so it must not pick one."""
    assert graph_for(store, sensors, COORDINATED).coordinated
    result = run_graph_query(store, COORDINATED, *sensors)
    assert result.decision == "ABSTAIN"


def test_coordination_survives_an_unreadable_conjunct(store, sensors):
    """The と can end up in a phrase the accusative rule then discards.

    「成功の割合と処理にかかった時間を」 splits at the clause-internal に, so the
    coordinated phrase loses and only the legible conjunct reaches the head.
    Coordination is therefore read off the clause, not off the winning phrase.
    """
    text = "64x64のケースで成功の割合と処理にかかった時間をならして"
    assert graph_for(store, sensors, text).coordinated
    assert run_graph_query(store, text, *sensors).decision == "ABSTAIN"


def test_structure_and_sentence_complete_each_other(store, sensors):
    """Where says nothing about what; what says nothing about where.

    A request whose only field is unreadable as a span still resolves, because
    exactly one ungrounded whole-sentence node exists to fill the edge. This is
    the repair, and it is available only when that node is unique.
    """
    text = "64x64のケースの成功の割合をならして"
    result = run_graph_query(store, text, *sensors)
    assert result.decision == "ANSWER"
    assert "success_rate" in canonical_render(result.expr)
    assert "structure" in result.source and "whole" in result.source


def test_the_repair_is_unavailable_when_two_fields_contend(store, sensors):
    """UNNAMED has two fields, so the sentence splits and nothing floats free."""
    graph = graph_for(store, sensors, UNNAMED)
    assert [node for node in graph.of_kind("FIELD") if node.start < 0] == []
    assert readings(graph)[1] is not None


# --------------------------------------------------------------------------
# the fail-closed default
# --------------------------------------------------------------------------

def test_an_operator_the_registry_never_heard_of_still_needs_an_argument():
    """L8.19's OPERATIONS predates MAX.

    Keying the guard on that table answered False for MAX, so both refusals
    silently did not apply to it -- six wrong answers on the sweep. The test
    pins the direction the default must fail in.
    """
    assert needs_field("MAX") and needs_field("MEAN")
    assert needs_field("AN_OPERATION_INVENTED_LATER")
    assert not needs_field("COUNT")


# --------------------------------------------------------------------------
# the hold-out
# --------------------------------------------------------------------------

def test_the_generated_sweep_is_not_hand_written():
    probes = generate_probes()
    assert len(probes) >= 60
    assert len({probe.family for probe in probes}) == 5


def test_wrong_answer_rate_is_zero_on_both_suites(report):
    """The invariant. Coverage may be traded; this may not."""
    for suite in ("generated", "families"):
        assert report[suite]["l827"]["metrics"]["wrong_answer_rate"] == 0.0
        assert report[suite]["l827"]["metrics"]["unsafe_execution_rate"] == 0.0


def test_the_graph_beats_its_predecessors_on_the_generated_sweep(report):
    generated = report["generated"]
    assert generated["l827"]["metrics"]["coverage"] > generated["l826"]["metrics"]["coverage"]
    assert generated["l826"]["metrics"]["wrong_answer_rate"] > 0.0
    assert generated["l827"]["metrics"]["calibration"] == 1.0


def test_abstention_is_reported_beside_coverage(report):
    """Either number alone is trivially satisfiable; the pair is the claim."""
    for suite in ("generated", "families"):
        metrics = report[suite]["l827"]["metrics"]
        assert "unnecessary_abstention_rate" in metrics
        assert metrics["unnecessary_abstention_rate"] < 0.2


def test_controls_are_untouched(report):
    families = build_probes()
    records = report["families"]["l827"]["records"]
    for probe, record in zip(families, records):
        if probe.family.startswith("CONTROL") or probe.family == "NO_EVIDENCE":
            assert record["correct"], probe.text


def test_the_report_formats(report):
    from coding_world_benchmark.holdout_graph_l8271_experiment import format_experiment

    text = format_experiment(report)
    assert "wrong_answer" in text and "--bind[" in text
