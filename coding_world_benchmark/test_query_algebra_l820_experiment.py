import pytest

from coding_world_benchmark.holdout_composition_l8191_experiment import build_holdout_store
from coding_world_benchmark.holdout_program_l8201_experiment import (
    DEVELOPMENT_SHAPES,
    run_holdout,
)
from coding_world_benchmark.query_algebra_l820_experiment import (
    Compare,
    Count,
    Filter,
    Kind,
    Mean,
    Project,
    Source,
    TypeError_,
    compile_query,
    infer,
    render,
    run_query,
)
from coding_world_benchmark.typed_operation_l819_experiment import execute_typed


@pytest.fixture(scope="module")
def store():
    return build_holdout_store(24)[0]


@pytest.fixture(scope="module")
def cold():
    return run_holdout(24)


NESTED = "32x32と比べて64x64の平均実行時間は大きい？"


def test_the_l8191_failure_is_closed(store):
    # L8.19's flat parser read this as MEAN with two contradictory filters.
    flat = execute_typed(store, NESTED)
    nested = run_query(store, NESTED)

    assert flat.decision == "ABSTAIN"
    assert nested.decision == "ANSWER"
    assert render(nested.expr) == render(
        Compare(
            Mean(Project(Filter(Source(), "lattice_size", "32x32"), "runtime_seconds")),
            Mean(Project(Filter(Source(), "lattice_size", "64x64"), "runtime_seconds")),
        )
    )


def test_compare_binds_loosest(store):
    # The whole fix: the outermost operator is taken off first, so 平均 cannot
    # capture a request whose top-level operator is a comparison.
    compiled = compile_query(NESTED, store)

    assert isinstance(compiled.expr, Compare)


def test_an_operand_inherits_an_elided_aggregate_and_field(store):
    # 「32x32と」 states neither the aggregate nor the field; both appear once, on
    # the other operand. Without inheritance the baseline is a bare set and the
    # comparison is ill-typed.
    compiled = compile_query(NESTED, store)

    assert isinstance(compiled.expr.baseline, Mean)
    assert compiled.expr.baseline.source.field == "runtime_seconds"


def test_types_are_inferred_bottom_up_and_name_the_node():
    assert infer(Count(Filter(Source(), "lattice_size", "64x64"))) is Kind.INTEGER

    with pytest.raises(TypeError_) as error:
        infer(Mean(Project(Source(), "experiment_name")))
    assert "Collection[Number]" in str(error.value)


def test_the_l818_gate_still_runs_first(store):
    result = run_query(store, "実行時間の中央値は？")

    assert result.decision == "ABSTAIN"
    assert result.stage == "gate"


def test_an_ordinary_reference_is_not_a_query(store):
    assert run_query(store, "さっきの続きをやって").decision == "NOT_AN_OPERATION"


def test_composition_generalizes_to_shapes_never_written_against(cold):
    """The claim this hold-out exists to test.

    Not a learning result: the compiler is hand-written, so 'held out' means
    absent from the cases it was written against, and the question is whether
    the recursion works in general or whether the nesting fix was special-cased
    around the example that motivated it.
    """
    assert cold["metrics"]["held_out_shape_accuracy"] == 1.0

    held_out = {r["shape"] for r in cold["records"] if r["held_out"]}
    assert held_out and not (held_out & DEVELOPMENT_SHAPES)
    assert all(r["ast_match"] for r in cold["records"] if r["held_out"])


def test_nothing_is_answered_wrongly(cold):
    assert cold["metrics"]["wrong_answer_rate"] == 0.0


def test_an_unsupported_language_fails_without_computing(cold):
    english = [r for r in cold["records"] if r["language"] == "en"]

    assert english
    assert all(r["decision"] != "ANSWER" for r in english)
    assert cold["metrics"]["unsupported_language_wrong_rate"] == 0.0


def test_field_binding_is_still_first_match_wins(store, cold):
    """The defect this hold-out surfaced, one level below composition.

    「格子サイズ64x64の実行時間の平均を教えて」 mentions two fields: 格子サイズ belongs to
    the filter and 実行時間 to the projection.  The parser takes the first cue it
    finds, projects lattice_size, and MEAN then receives Collection[Categorical].

    Composition generalised; argument binding did not -- the same first-match
    mistake as L8.19's operator selection, one level down.  It fails safely
    because the type check refuses it: without types this would have averaged
    lattice-size strings.  Left unfixed; binding is L8.21's subject.
    """
    result = run_query(store, "格子サイズ64x64の実行時間の平均を教えて")

    assert result.decision == "ABSTAIN"
    assert result.stage == "type"
    assert "Categorical" in result.reason
    assert cold["metrics"]["development_shape_accuracy"] < 1.0


def test_held_out_shapes_outscore_the_paraphrase_gap(cold):
    metrics = cold["metrics"]

    assert metrics["held_out_shape_accuracy"] > metrics["development_shape_accuracy"]
    assert metrics["execution_match"] >= 0.9


def test_holdout_is_deterministic():
    assert run_holdout(16)["metrics"] == run_holdout(16)["metrics"]
