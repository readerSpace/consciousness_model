import pytest

from coding_world_benchmark.evidence_calibration_l825_experiment import (
    ENTANGLED, MIN_SUPPORT, RELIABILITY_FLOOR, SEPARATED, ReliabilityTable, SlotEvidence,
    band, calibrate, calibration_renderings, run_calibrated_query,
)
from coding_world_benchmark.holdout_binding_l8211_experiment import build_holdout
from coding_world_benchmark.holdout_calibration_l8251_experiment import (
    WHOLE_STRONG_L824, run_holdout,
)
from coding_world_benchmark.query_algebra_l820_experiment import canonical_render


@pytest.fixture(scope="module")
def store():
    return build_holdout(24)[0]


@pytest.fixture(scope="module")
def report():
    return run_holdout(24)


def test_reliability_is_estimated_from_program_supervised_renderings():
    # Each observation is a sensor proposal checked against the node the
    # renderer emitted; nothing here was labelled by hand.
    rows = calibration_renderings(SEPARATED)

    assert rows
    for text, operator_label, field_label in rows[:5]:
        assert operator_label in SEPARATED.operators
        assert field_label in SEPARATED.fields
        assert "64x64" in text


def test_the_hard_coded_constant_is_conservative_by_one_band(report):
    """The finding: the data puts the boundary lower than 0.75.

    Whole-sentence operator readings in the 0.6 band are right 95% of the time,
    and L8.24's fixed threshold rejects them. The constant was not wrong, it was
    chosen by eye from a plot, and the estimate says where the boundary actually
    sits.
    """
    separated = report["diagnostics"]["separated"]
    key = ("whole", "OPERATOR", 0.6, 0.0)

    assert separated[key] >= RELIABILITY_FLOOR
    assert key[2] < WHOLE_STRONG_L824


def test_reliability_differs_by_sensor_which_one_threshold_cannot_say(report):
    separated = report["diagnostics"]["separated"]

    assert separated[("span", "FIELD", 1.0, 1.0)] >= 0.9
    assert separated[("whole", "FIELD", 0.4, 0.0)] <= 0.6


def test_reliability_differs_by_vocabulary(report):
    """What a fixed constant assumes away.

    In the separated vocabulary a whole-sentence field reading reaches the 1.0
    band and is trustworthy. In the entangled one -- 実行時間 beside 実行成功率 --
    every field reading lands at 0.4 and is a coin flip, so fields can only come
    from spans there. The same threshold would describe both.
    """
    separated = report["diagnostics"]["separated"]
    entangled = report["diagnostics"]["entangled"]

    assert ("whole", "FIELD", 1.0, 0.0) in separated
    assert ("whole", "FIELD", 1.0, 0.0) not in entangled
    assert entangled[("whole", "FIELD", 0.4, 0.0)] <= 0.6


def test_nothing_is_lost_on_the_original_vocabulary(report):
    metrics = report["runs"]["separated"]["metrics"]

    assert metrics["accuracy"] == 1.0
    assert metrics["wrong_answer_rate"] == 0.0
    assert metrics["unsafe_execution_rate"] == 0.0


def test_the_entangled_vocabulary_works_through_spans(report):
    metrics = report["runs"]["entangled"]["metrics"]

    assert metrics["accuracy"] == 1.0
    assert metrics["wrong_answer_rate"] == 0.0
    assert metrics["unsafe_execution_rate"] == 0.0


def test_the_measured_outcome_is_agreement_not_a_coverage_gain(report):
    """Reported as it came out.

    The whole-sentence score is close to bimodal -- near 1.0 when one label has
    evidence, near 0.5 when two contend -- so the band where the two admission
    rules disagree is barely populated by real requests. Calibration buys a
    derived and re-derivable boundary here, not coverage.
    """
    assert report["runs"]["separated"]["metrics"]["agreement_with_fixed_threshold"] == 1.0
    assert all(record["agrees_with_fixed"] for record in report["runs"]["separated"]["records"])


def test_thin_bands_are_not_trusted_in_either_direction():
    table = ReliabilityTable()
    for _ in range(MIN_SUPPORT - 1):
        table.observe(("whole", "OPERATOR", 1.0, 0.0), True)

    assert table.reliability(("whole", "OPERATOR", 1.0, 0.0)) is None
    admitted, sources = table.admits(SlotEvidence("OPERATOR", "MEAN", 0.0, 0.0, 1.0))
    assert not admitted and sources == ()


def test_calibration_is_admission_only(store, report):
    # A candidate admitted by reliability still has to survive typing and
    # uniqueness; the ambiguous probe is refused exactly as before.
    table = report["tables"]["separated"]
    result = run_calibrated_query(
        store, "64x64のケースの処理時間をピークをならして", SEPARATED, table
    )

    assert result.decision == "ABSTAIN"
    assert result.value is None


def test_a_calibrated_execution_still_produces_the_right_program(store, report):
    table = report["tables"]["separated"]
    result = run_calibrated_query(store, "64x64のケースの処理時間をならす形に", SEPARATED, table)

    assert result.decision == "ANSWER"
    assert canonical_render(result.expr) == (
        "MEAN(PROJECT(FILTER(episodes, lattice_size == 64x64), runtime_seconds))"
    )


def test_bands_are_monotone_and_bounded():
    assert band(0.0) == 0.0
    assert band(0.59) == 0.4
    assert band(0.61) == 0.6
    assert band(1.0) == 1.0


def test_results_are_deterministic():
    left, right = run_holdout(16), run_holdout(16)

    assert left["diagnostics"] == right["diagnostics"]
    assert left["runs"]["entangled"]["metrics"] == right["runs"]["entangled"]["metrics"]
