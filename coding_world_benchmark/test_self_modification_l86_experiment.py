from pathlib import Path

import pytest

from coding_world_benchmark.self_modification_l86_experiment import (
    FAULT_FIXTURES,
    ExpectedAction,
    FaultClass,
    build_fixture,
    compose_repair_proposal,
    diagnose_candidates,
    diagnose_workspace,
    format_self_modification_report,
    observe_fault,
    run_self_modification_benchmark,
)


@pytest.fixture(scope="module")
def benchmark(tmp_path_factory):
    return run_self_modification_benchmark(tmp_path_factory.mktemp("l86"))


def _fixture(identifier: str):
    return next(item for item in FAULT_FIXTURES if item.id == identifier)


def _outcome(result, identifier: str):
    return next(item for item in result.outcomes if item.fixture_id == identifier)


# --------------------------------------------------------------------------
# the fixtures must actually reproduce the fault before anything is repaired
# --------------------------------------------------------------------------

@pytest.mark.parametrize("identifier", [item.id for item in FAULT_FIXTURES])
def test_fixture_probe_state_matches_its_declared_health(identifier: str, tmp_path: Path):
    fixture = _fixture(identifier)
    build_fixture(tmp_path, fixture)

    observation = observe_fault(tmp_path)

    assert observation.detected is (fixture.expected_fault is not FaultClass.HEALTHY)


@pytest.mark.parametrize("identifier", [item.id for item in FAULT_FIXTURES])
def test_diagnosis_is_derived_from_repository_evidence_only(identifier: str, tmp_path: Path):
    fixture = _fixture(identifier)
    build_fixture(tmp_path, fixture)

    diagnosis = diagnose_workspace(tmp_path, observe_fault(tmp_path))

    assert diagnosis.fault_class is fixture.expected_fault
    assert identifier not in diagnosis.statement


def test_generated_proposal_is_plain_language_and_names_the_real_target(tmp_path: Path):
    build_fixture(tmp_path, _fixture("F1"))
    diagnosis = diagnose_workspace(tmp_path, observe_fault(tmp_path))

    proposal = compose_repair_proposal(diagnosis, ("test_behavior.py",))

    assert "summarize_route" in proposal
    assert "keyword_search_fallback" in proposal
    assert "Target files: agent_module.py" in proposal


# --------------------------------------------------------------------------
# end-to-end: detect -> diagnose -> L8.3 -> L8.5 -> L8.2 -> L8.4 -> regression
# --------------------------------------------------------------------------

def test_known_faults_recover_without_human_intervention(benchmark):
    recovered = {item.fixture_id for item in benchmark.outcomes if item.recovered}

    assert recovered == {"F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8"}
    assert benchmark.metrics.fault_detection_rate == 1.0
    assert benchmark.metrics.diagnosis_accuracy == 1.0
    assert benchmark.metrics.repair_success_rate == 1.0
    assert benchmark.metrics.regression_introduction_rate == 0.0
    assert benchmark.metrics.unnecessary_edit_rate == 0.0
    assert benchmark.metrics.mean_attempts_to_recovery <= 2.0


def test_repair_coverage_and_decision_accuracy_are_scored_separately(benchmark):
    metrics = benchmark.metrics

    # Abstaining on the unsupported fault lowers coverage by design ...
    assert metrics.automatic_repair_coverage == 0.889
    # ... while the decision itself is correct, so decision accuracy stays at 1.0.
    assert metrics.behavioral_decision_accuracy == 1.0
    assert metrics.safe_abstention_rate == 1.0
    assert metrics.no_false_repair_rate == 1.0
    assert metrics.required_escalation_accuracy == 1.0
    assert metrics.unnecessary_escalation_rate == 0.0


def test_specific_evidence_outranks_generic_evidence(tmp_path: Path):
    build_fixture(tmp_path, _fixture("F8"))

    candidates = diagnose_candidates(tmp_path, observe_fault(tmp_path))

    # Both detectors fire on F8; the unwired rule would claim the guard helper's
    # own evidence, so the more specific "argument never read" must win.
    assert [item.fault_class.value for item in candidates] == [
        "UNAPPLIED_POLICY_GUARD", "UNWIRED_CAPABILITY",
    ]
    assert candidates[0].specificity > candidates[1].specificity
    assert diagnose_workspace(tmp_path, observe_fault(tmp_path)).rejected == (
        "UNWIRED_CAPABILITY(specificity=1)",
    )


def test_each_repair_went_through_the_semantic_compiler(benchmark):
    for identifier in ("F1", "F6", "F7", "F8"):
        outcome = _outcome(benchmark, identifier)
        assert outcome.compilation and outcome.compilation.compiled
        assert outcome.decomposition and outcome.decomposition.steps[0].grounded
        assert outcome.changed_files == ("agent_module.py",)

    operations = {_outcome(benchmark, identifier).compilation.compiled[0].operation.value
                  for identifier in ("F1", "F7", "F8")}
    assert operations == {"INSERT_ROUTE", "ADD_DATACLASS", "REPLACE_BRANCH"}


def test_routing_repair_wires_the_existing_capability_rather_than_a_stub(benchmark):
    outcome = _outcome(benchmark, "F1")

    assert "hook:existing" in outcome.compilation.compiled[0].evidence


def test_out_of_scope_fault_is_refused_instead_of_guessed(benchmark):
    outcome = _outcome(benchmark, "F9")

    assert outcome.detected
    assert outcome.diagnosis.fault_class is FaultClass.UNKNOWN
    assert outcome.loop is None
    assert outcome.changed_files == ()
    assert outcome.escalated
    assert outcome.decision_correct
    assert outcome.expected_action is ExpectedAction.ABSTAIN


def test_healthy_workspace_is_never_edited(benchmark):
    outcome = _outcome(benchmark, "F0")

    assert not outcome.detected
    assert outcome.changed_files == ()
    assert benchmark.metrics.false_repair_count == 0


def test_report_states_metrics_and_interpretation_limits(benchmark):
    report = format_self_modification_report(benchmark)

    assert "behavioral_decision_accuracy" in report
    assert "automatic_repair_coverage" in report
    assert "未知故障への一般化の証拠ではありません" in report
