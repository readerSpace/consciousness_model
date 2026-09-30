import ast
from pathlib import Path

import pytest

from coding_world_benchmark.holdout_faults_l861_experiment import (
    COLD_MEASUREMENT,
    HOLDOUT_FIXTURES,
    classify_holdout_outcome,
    compare_known_and_holdout,
    format_holdout_comparison,
    verify_holdout_is_well_formed,
)


@pytest.fixture(scope="module")
def comparison(tmp_path_factory):
    return compare_known_and_holdout(tmp_path_factory.mktemp("l861"))


def _outcome(result, identifier: str):
    return next(item for item in result.outcomes if item.fixture_id == identifier)


# --------------------------------------------------------------------------
# fixture hygiene: a hold-out only means something if it isolates one fault
# --------------------------------------------------------------------------

@pytest.mark.parametrize("fixture", HOLDOUT_FIXTURES, ids=[item.id for item in HOLDOUT_FIXTURES])
def test_holdout_isolates_exactly_one_regression(fixture, tmp_path: Path):
    healthy_passes, probe_fails, baseline_holds = verify_holdout_is_well_formed(tmp_path, fixture)

    assert healthy_passes, "未改変のモジュールが probe/baseline を通らない"
    assert probe_fails, "改変しても probe が落ちない"
    assert baseline_holds, "改変が baseline まで壊しており、故障が分離できていない"


@pytest.mark.parametrize("fixture", HOLDOUT_FIXTURES, ids=[item.id for item in HOLDOUT_FIXTURES])
def test_corruption_is_a_single_auditable_substitution(fixture):
    before, after = fixture.corruption

    assert fixture.healthy_module.count(before) == 1, "改変対象が一意でない"
    assert fixture.broken_module != fixture.healthy_module, "改変が何も変えていない"
    # The fault must be behavioural, not a syntax error the compiler would catch.
    ast.parse(fixture.broken_module)


# --------------------------------------------------------------------------
# what generalizes and what does not
# --------------------------------------------------------------------------

def test_behavioural_detection_generalizes_to_unseen_faults(comparison):
    # Detection is a probe run, not a pattern match, so it transfers intact.
    assert comparison.holdout.metrics.fault_detection_rate == 1.0
    assert comparison.known.metrics.fault_detection_rate == 1.0


def test_repair_does_not_generalize_and_the_gap_is_reported(comparison):
    assert comparison.generalization_gap >= 0.0
    assert comparison.known.metrics.repair_success_rate >= comparison.holdout.metrics.repair_success_rate
    assert "generalization_gap" in format_holdout_comparison(comparison)


def test_cold_measurement_is_unchanged_or_the_holdout_has_been_burned(comparison):
    measured = {
        "repair_success_rate": comparison.holdout.metrics.repair_success_rate,
        "fault_detection_rate": comparison.holdout.metrics.fault_detection_rate,
        "false_repair_count": comparison.holdout.metrics.false_repair_count,
        "breakdown": dict(comparison.breakdown),
    }

    assert measured == COLD_MEASUREMENT, (
        "hold-out の結果が記録値と異なります。診断器が改善したのであれば H1-H6 は既知故障に変わったので、"
        "新しい hold-out を作って再測定し、COLD_MEASUREMENT と TEST_MATRIX.md を更新してください。"
    )


# --------------------------------------------------------------------------
# safety must hold even where repair fails
# --------------------------------------------------------------------------

def test_no_wrong_repair_survives_the_regression_gate(comparison):
    assert comparison.holdout.metrics.false_repair_count == 0
    assert comparison.holdout.metrics.regression_introduction_rate == 0.0
    for outcome in comparison.holdout.outcomes:
        assert outcome.changed_files == (), f"{outcome.fixture_id} が誤った編集を残した"


def test_misdiagnosis_is_caught_rather_than_applied(comparison):
    reasons = dict(comparison.breakdown)
    rejected = [identifier for identifier, reason in reasons.items()
                if reason == "misdiagnosed_then_rejected_by_regression_gate"]

    assert rejected, "誤診が一度も起きていないなら、この hold-out は gate を検証できていない"
    for identifier in rejected:
        outcome = _outcome(comparison.holdout, identifier)
        assert outcome.loop and outcome.loop.outcome != "accepted"
        assert not outcome.probe_passed_after
        assert outcome.changed_files == ()


def test_holdout_control_is_neither_detected_nor_edited(comparison):
    control = _outcome(comparison.holdout, "H0")

    assert not control.detected
    assert control.changed_files == ()
    assert control.decision_correct
    assert classify_holdout_outcome(control) == "undetected"
