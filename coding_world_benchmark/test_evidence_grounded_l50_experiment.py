from coding_world_benchmark.evidence_grounded_l50_experiment import (
    EvidenceStatus,
    GroundedEvidence,
    cases,
    retrieve_evidence,
    run_evidence_grounded_retrieval_experiment,
    run_policy,
)
from coding_world_benchmark.knowledge_gap_l49_experiment import CandidateAction


REPORT = run_evidence_grounded_retrieval_experiment()


def test_l50_has_the_five_required_retrieval_failure_modes():
    assert [case.name for case in cases()] == [
        "multi_source_power",
        "misleading_top_snippet",
        "conflicting_sources",
        "unresolved_gap",
        "irrelevant_true_result",
    ]
    assert [case.expected_status for case in cases()] == [
        EvidenceStatus.SUPPORTED,
        EvidenceStatus.SUPPORTED,
        EvidenceStatus.CONFLICTING,
        EvidenceStatus.UNRESOLVED,
        EvidenceStatus.UNRESOLVED,
    ]


def test_grounded_evidence_carries_source_and_uncertainty_fields():
    case = next(item for item in cases() if item.name == "multi_source_power")
    evidence = retrieve_evidence(case, "evidence_grounded")
    assert len(evidence) >= 2
    assert all(isinstance(item, GroundedEvidence) for item in evidence)
    assert all(item.source_url.startswith("https://example.test/") for item in evidence)
    assert all(item.retrieved_at.startswith("2026-09-13") for item in evidence)
    assert all(item.quoted_span for item in evidence)
    assert all(0.0 <= item.uncertainty <= 1.0 for item in evidence)


def test_evidence_grounded_policy_recovers_supported_claims_without_unsupported_claims():
    metrics = REPORT["policies"]["evidence_grounded"]["metrics"]
    assert metrics["claim_precision"] == 1.0
    assert metrics["claim_recall"] == 1.0
    assert metrics["unsupported_claim_rate"] == 0.0
    assert metrics["source_attribution_accuracy"] == 1.0


def test_evidence_grounded_policy_detects_conflicts_and_unresolved_gaps():
    metrics = REPORT["policies"]["evidence_grounded"]["metrics"]
    assert metrics["conflict_detection_accuracy"] == 1.0
    assert metrics["unresolved_gap_accuracy"] == 1.0

    conflict = run_policy(next(case for case in cases() if case.name == "conflicting_sources"), "evidence_grounded")
    assert conflict["status"] == "conflicting_evidence"
    assert conflict["accepted_claim"] is None
    assert conflict["action"] == CandidateAction.ASK_EXTERNAL_REASONER.value

    unresolved = run_policy(next(case for case in cases() if case.name == "unresolved_gap"), "evidence_grounded")
    assert unresolved["status"] == "unresolved_gap"
    assert unresolved["accepted_claim"] is None


def test_irrelevant_true_sources_are_rejected_by_relevance_filtering():
    case = next(item for item in cases() if item.name == "irrelevant_true_result")
    top1 = run_policy(case, "top1_snippet")
    grounded = run_policy(case, "evidence_grounded")
    assert top1["accepted_claim"] is not None
    assert top1["unsupported_claim"]
    assert grounded["status"] == "unresolved_gap"
    assert grounded["accepted_claim"] is None
    assert grounded["action"] == CandidateAction.REVISE_HYPOTHESIS.value


def test_evidence_grounded_beats_snippet_majority_and_authority_baselines():
    policies = REPORT["policies"]
    grounded = policies["evidence_grounded"]["metrics"]
    assert grounded["unsupported_claim_rate"] < policies["top1_snippet"]["metrics"]["unsupported_claim_rate"]
    assert grounded["conflict_detection_accuracy"] > policies["top1_snippet"]["metrics"]["conflict_detection_accuracy"]
    assert grounded["unresolved_gap_accuracy"] > policies["authority_weighted"]["metrics"]["unresolved_gap_accuracy"]
    assert grounded["action_accuracy_after_evidence"] > policies["top_k_majority"]["metrics"]["action_accuracy_after_evidence"]
