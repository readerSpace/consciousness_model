import pytest

from coding_world_benchmark.argument_binding_l821_experiment import (
    collect_mentions,
    compile_bound,
    run_bound_query,
)
from coding_world_benchmark.holdout_binding_l8211_experiment import build_holdout
from coding_world_benchmark.holdout_binding_l8211_experiment import run_holdout as run_binding
from coding_world_benchmark.holdout_program_l8201_experiment import run_holdout as run_program
from coding_world_benchmark.query_algebra_l820_experiment import (
    Filter, Source, canonical_render, run_query,
)


@pytest.fixture(scope="module")
def store():
    return build_holdout(24)[0]


@pytest.fixture(scope="module")
def cold():
    return run_binding(24, bound=True)


@pytest.fixture(scope="module")
def cold_l820():
    return run_binding(24, bound=False)


def test_the_l8201_binding_defect_is_closed(store):
    text = "格子サイズ64x64の実行時間の平均は？"

    assert run_query(store, text).decision == "ABSTAIN"

    bound = run_bound_query(store, text)
    assert bound.decision == "ANSWER"
    assert canonical_render(bound.expr) == (
        "MEAN(PROJECT(FILTER(episodes, lattice_size == 64x64), runtime_seconds))"
    )


def test_a_value_can_name_its_own_field(store):
    # Nothing in this request says 格子サイズ; the filter is forced because 64x64
    # occurs in exactly one field's domain.  A proximity rule has nothing to
    # work from here.
    mentions = collect_mentions("64x64を使った実験の平均成功率は？", store)

    assert [m.candidate_fields for m in mentions.values] == [("lattice_size",)]
    result = run_bound_query(store, "64x64を使った実験の平均成功率は？")
    assert "FILTER(episodes, lattice_size == 64x64)" in canonical_render(result.expr)


def test_the_value_decides_which_field_is_filtered(store, cold):
    record = next(r for r in cold["records"] if r["family"] == "OTHER_FIELD_VALUE")

    assert record["correct"]
    assert "seed ==" in record["got"]
    assert "lattice_size ==" not in record["got"]


def test_an_underdetermined_request_refuses_instead_of_choosing(store):
    result = run_bound_query(store, "実行時間と成功率の平均は？")

    assert result.decision == "ABSTAIN"
    assert result.stage == "binding"
    assert "does not choose" in result.reason


def test_that_same_request_is_answered_by_the_previous_compiler(store, cold_l820):
    # The point of keeping L8.20 alongside: it returns a number, nothing in the
    # answer says it picked one of two fields, and this is the first wrong
    # answer any layer has produced since L8.18.
    assert run_query(store, "実行時間と成功率の平均は？").decision == "ANSWER"
    assert cold_l820["metrics"]["wrong_answer_rate"] > 0.0


def test_a_named_field_that_cannot_fill_the_hole_is_a_type_error(store):
    result = run_bound_query(store, "実験名の平均は？")

    assert result.decision == "ABSTAIN"
    assert result.stage == "type"
    assert "cannot take" in result.reason


def test_inheritance_outranks_that_type_error(store):
    # In 「格子サイズ32x32と比べて…」 the baseline names lattice_size for its *filter*
    # and inherits the projection from its sibling.  Treating the named field as
    # a failed projection would report a hole it was never offered for.
    text = "格子サイズ32x32と比べて格子サイズ64x64の実行時間の平均は大きい？"
    result = run_bound_query(store, text)

    assert result.decision == "ANSWER"
    assert canonical_render(result.expr).startswith("COMPARE(MEAN(")


def test_cold_holdout_binds_correctly_without_gaining_a_wrong_answer(cold):
    metrics = cold["metrics"]

    assert metrics["binding_accuracy"] == 1.0
    assert metrics["shape_accuracy"] == 1.0
    assert metrics["wrong_answer_rate"] == 0.0
    assert metrics["ambiguous_abstention_rate"] == 1.0
    assert metrics["unnecessary_abstention_rate"] == 0.0
    assert metrics["control_accuracy"] == 1.0


def test_binding_beats_the_previous_compiler_on_both_axes(cold, cold_l820):
    assert cold["metrics"]["binding_accuracy"] > cold_l820["metrics"]["binding_accuracy"]
    assert cold["metrics"]["wrong_answer_rate"] < cold_l820["metrics"]["wrong_answer_rate"]


def test_no_regression_on_the_program_holdout():
    metrics = run_program(24, bound=True)["metrics"]

    assert metrics["held_out_shape_accuracy"] == 1.0
    assert metrics["development_shape_accuracy"] == 1.0
    assert metrics["ast_exact_match"] == 1.0
    assert metrics["wrong_answer_rate"] == 0.0


def test_commuted_filter_chains_compare_equal():
    # Conjunctive filters commute, so AST equality is taken over the canonical
    # form; comparing raw renderings measured emission order, not meaning.
    left = Filter(Filter(Source(), "seed", "1"), "lattice_size", "64x64")
    right = Filter(Filter(Source(), "lattice_size", "64x64"), "seed", "1")

    assert canonical_render(left) == canonical_render(right)


def test_controls_are_untouched(store):
    assert run_bound_query(store, "さっきの続きをやって").decision == "NOT_AN_OPERATION"
    gate = run_bound_query(store, "実行時間の中央値は？")
    assert gate.decision == "ABSTAIN" and gate.stage == "gate"


def test_holdout_is_deterministic():
    assert run_binding(16)["metrics"] == run_binding(16)["metrics"]
