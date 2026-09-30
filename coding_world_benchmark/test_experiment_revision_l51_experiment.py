from coding_world_benchmark.evidence_grounded_l50_experiment import EvidenceStatus
from coding_world_benchmark.experiment_revision_l51_experiment import (
    RevisionDecision,
    RevisionProposal,
    apply_revision,
    cases,
    propose_revision,
    run_experiment_revision_experiment,
    run_policy,
)
from coding_world_benchmark.scientific_diagnosis_l48_experiment import ResultDiagnosis


REPORT = run_experiment_revision_experiment()


def test_l51_covers_revision_routing_and_hold_cases():
    all_cases = cases()
    assert {case.diagnosis for case in all_cases} >= {
        ResultDiagnosis.LOW_POWER,
        ResultDiagnosis.INSTRUMENT_FAILURE,
        ResultDiagnosis.CONFOUNDED,
        ResultDiagnosis.IMPLEMENTATION_FAILURE,
        ResultDiagnosis.NEGATIVE_RESULT,
    }
    assert any(case.evidence_status is EvidenceStatus.CONFLICTING for case in all_cases)
    assert any(case.evidence_status is EvidenceStatus.UNRESOLVED for case in all_cases)
    assert len(all_cases) == 7


def test_revision_proposal_contains_evidence_attribution_and_rationale():
    case = next(item for item in cases() if item.name == "low_power_increase_samples")
    decision, proposal = propose_revision(case)
    assert decision is RevisionDecision.REVISE_EXPERIMENT
    assert isinstance(proposal, RevisionProposal)
    assert proposal.target == "sample_size"
    assert proposal.change_type == "increase"
    assert proposal.rationale_claims
    assert proposal.evidence_ids == tuple(item.source_url for item in case.evidence)
    assert proposal.confidence > 0.8


def test_supported_evidence_changes_only_the_relevant_plan_field():
    low_power = next(item for item in cases() if item.name == "low_power_increase_samples")
    _, low_power_proposal = propose_revision(low_power)
    revised = apply_revision(low_power.plan, RevisionDecision.REVISE_EXPERIMENT, low_power_proposal)
    assert revised.sample_size > low_power.plan.sample_size
    assert revised.measurement == low_power.plan.measurement
    assert revised.controls == low_power.plan.controls

    instrument = next(item for item in cases() if item.name == "instrument_failure_change_measurement")
    _, instrument_proposal = propose_revision(instrument)
    revised = apply_revision(instrument.plan, RevisionDecision.REVISE_EXPERIMENT, instrument_proposal)
    assert revised.measurement == "nanmean"
    assert revised.sample_size == instrument.plan.sample_size


def test_code_failures_and_negative_results_are_not_experiment_tweaks():
    implementation = next(item for item in cases() if item.name == "implementation_failure_route_to_code")
    decision, proposal = propose_revision(implementation)
    assert decision is RevisionDecision.ROUTE_TO_CODE_REPAIR
    assert proposal is None

    negative = next(item for item in cases() if item.name == "negative_result_abandon")
    decision, proposal = propose_revision(negative)
    assert decision is RevisionDecision.ABANDON_HYPOTHESIS
    assert proposal is not None


def test_conflicting_and_unresolved_evidence_hold_revision():
    for name in ("conflicting_evidence_hold", "unresolved_evidence_hold"):
        case = next(item for item in cases() if item.name == name)
        decision, proposal = propose_revision(case)
        assert decision is RevisionDecision.GATHER_MORE_EVIDENCE
        assert proposal is None
        row = run_policy(case, "evidence_grounded_revision")
        assert not row["revised"]
        assert row["conclusion"] == "no_conclusion"


def test_evidence_grounded_revision_has_no_harmful_or_unnecessary_revisions():
    metrics = REPORT["policies"]["evidence_grounded_revision"]["metrics"]
    assert metrics["revision_precision"] == 1.0
    assert metrics["revision_recall"] == 1.0
    assert metrics["harmful_revision_rate"] == 0.0
    assert metrics["unnecessary_revision_rate"] == 0.0
    assert metrics["evidence_attribution_accuracy"] == 1.0
    assert metrics["post_revision_experiment_success"] == 1.0
    assert metrics["post_revision_conclusion_accuracy"] == 1.0


def test_grounded_revision_beats_revision_without_grounding_and_no_revision():
    policies = REPORT["policies"]
    grounded = policies["evidence_grounded_revision"]["metrics"]
    assert grounded["post_revision_conclusion_accuracy"] > policies["no_revision"]["metrics"]["post_revision_conclusion_accuracy"]
    assert grounded["harmful_revision_rate"] < policies["retrieval_without_grounding"]["metrics"]["harmful_revision_rate"]
    assert grounded["unnecessary_revision_rate"] < policies["diagnosis_only_revision"]["metrics"]["unnecessary_revision_rate"]
    assert grounded["post_revision_experiment_success"] > policies["retrieval_without_grounding"]["metrics"]["post_revision_experiment_success"]
