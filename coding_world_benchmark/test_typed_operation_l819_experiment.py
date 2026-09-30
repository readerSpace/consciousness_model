import pytest

from coding_world_benchmark.holdout_composition_l8191_experiment import (
    build_holdout_store,
    run_holdout,
)
from coding_world_benchmark.typed_operation_l819_experiment import (
    OPERATIONS,
    ValueType,
    execute_typed,
    parse_typed_query,
    type_check,
)


@pytest.fixture(scope="module")
def store():
    return build_holdout_store(24)[0]


@pytest.fixture(scope="module")
def cold():
    return run_holdout(24)


def test_signatures_are_declared_not_implied():
    names = {spec.name for spec in OPERATIONS}

    assert names == {"COUNT", "MEAN", "MODE", "ARGMAX", "COMPARE"}
    mean = next(spec for spec in OPERATIONS if spec.name == "MEAN")
    assert mean.argument_types == (ValueType.NUMBER,)
    assert mean.result_type is ValueType.NUMBER


def test_a_fully_resolved_request_can_still_be_rejected_on_type(store):
    # Every slot resolves -- MEAN is found, experiment_name is found -- and the
    # request is still nonsense.  A string-matching implementation computes
    # something here; this is the case type checking exists for.
    outcome = execute_typed(store, "実験名の平均は？")

    assert outcome.decision == "ABSTAIN"
    assert outcome.stage == "type"
    assert "Text" in outcome.reason


def test_argmax_over_a_categorical_field_is_a_type_error(store):
    outcome = execute_typed(store, "格子サイズが一番大きい実験は？")

    assert outcome.decision == "ABSTAIN"
    assert outcome.stage == "type"


def test_preconditions_are_separate_from_types(store):
    # energy_error is declared Number, so ARGMAX admits it and the type check
    # passes.  It is simply never recorded.  Collapsing the two checks would
    # report this as a type error and hide which one actually failed.
    assert type_check(parse_typed_query("エネルギー誤差が一番大きい実験は？", store)) is None

    outcome = execute_typed(store, "エネルギー誤差が一番大きい実験は？")
    assert outcome.decision == "ABSTAIN"
    assert outcome.stage == "precondition"


def test_the_l818_gate_still_holds_unimplemented_operations(store):
    outcome = execute_typed(store, "実行時間の中央値は？")

    assert outcome.decision == "ABSTAIN"
    assert outcome.stage == "gate"


def test_an_ordinary_reference_is_not_captured_as_a_computation(store):
    assert execute_typed(store, "さっきの続きをやって").decision == "NOT_AN_OPERATION"
    assert execute_typed(store, "64x64を使った実験を続けて").decision == "NOT_AN_OPERATION"


def test_flat_compositions_execute(cold):
    families = {record["family"]: record for record in cold["records"]}

    assert families["COUNT_FILTERED"]["correct"]
    assert families["MEAN_FILTERED"]["correct"]
    assert families["ARGMAX_FILTERED"]["correct"]
    assert families["MODE_PLAIN"]["correct"]


def test_nested_composition_is_not_supported_but_fails_safely(store, cold):
    """The limitation this hold-out was built to find.

    「32x32と比べて64x64の平均実行時間は大きい？」 is COMPARE(MEAN(A), MEAN(B)).  The
    parser is flat and first-match-wins, so 「平均」 wins over 「と比べて」 and the
    request is read as MEAN with two contradictory filters.  Both operands are
    recovered; only the nesting of the operators is lost -- the same shape as
    L8.18's lesson, one level up.

    It refuses rather than answering, and it refuses because L8.18's
    precondition catches a filter that selects nothing, on a case that rule was
    never written for.  Left unfixed: found on a cold hold-out, and nesting is
    L8.20's subject.
    """
    parsed = parse_typed_query("32x32と比べて64x64の平均実行時間は大きい？", store)

    assert parsed.operation == "MEAN"
    assert {item.value for item in parsed.filters} == {"32x32", "64x64"}

    outcome = execute_typed(store, "32x32と比べて64x64の平均実行時間は大きい？")
    assert outcome.decision == "ABSTAIN"
    assert outcome.stage == "precondition"


def test_nothing_is_ever_computed_wrongly(cold):
    metrics = cold["metrics"]

    assert metrics["wrong_operation_rate"] == 0.0
    assert metrics["type_error_escape_rate"] == 0.0
    assert metrics["invalid_execution_abstention_rate"] == 1.0


def test_capability_and_refusal_are_reported_as_a_pair(cold):
    metrics = cold["metrics"]

    assert metrics["valid_execution_rate"] >= 0.8
    assert metrics["unnecessary_abstention_rate"] <= 0.15


def test_the_three_refusal_stages_are_distinguishable(cold):
    stages = {record["family"]: record["stage"] for record in cold["records"] if record["abstained"]}

    assert stages["TYPE_ERROR"] == "type"
    assert stages["PRECONDITION"] == "precondition"
    assert stages["UNKNOWN_OPERATION"] == "gate"


def test_holdout_is_deterministic():
    assert run_holdout(16)["metrics"] == run_holdout(16)["metrics"]
