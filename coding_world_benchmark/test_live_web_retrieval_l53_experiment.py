from coding_world_benchmark.live_web_retrieval_l53_experiment import (
    LiveRetrievalStatus,
    cases,
    retrieve_live,
    run_live_web_retrieval_experiment,
)
from coding_world_benchmark.evidence_grounded_l50_experiment import EvidenceStatus
from coding_world_benchmark.knowledge_gap_l49_experiment import CandidateAction


REPORT = run_live_web_retrieval_experiment()


def test_l53_covers_all_live_retrieval_statuses():
    statuses = {case.primary_status for case in cases()}
    assert statuses == set(LiveRetrievalStatus)


def test_grounded_retrieval_filters_stale_and_parse_failed_content():
    stale = next(case for case in cases() if case.name == "stale_source_rejected")
    parsed = next(case for case in cases() if case.name == "parse_failure_recovers_with_fallback")

    stale_result = retrieve_live(stale, "live_grounded")
    parsed_result = retrieve_live(parsed, "live_grounded")

    assert stale_result["status"] == EvidenceStatus.UNRESOLVED.value
    assert stale_result["stale_rejected"]
    assert stale_result["evidence"] == []
    assert parsed_result["status"] == EvidenceStatus.UNRESOLVED.value
    assert parsed_result["evidence"] == []


def test_fallback_recovers_timeout_and_parse_failure():
    recoverable = {
        case.name: retrieve_live(case, "live_grounded_with_fallback")
        for case in cases()
        if case.expected_recovery
    }

    assert set(recoverable) == {
        "timeout_then_official_fallback",
        "parse_failure_recovers_with_fallback",
    }
    assert all(row["failure_recovered"] for row in recoverable.values())
    assert all(row["status"] == EvidenceStatus.SUPPORTED.value for row in recoverable.values())
    assert all(row["status_correct"] for row in recoverable.values())


def test_live_grounded_stops_safely_on_transport_missing_conflict_and_insufficient_cases():
    expected = {
        "browser_failure_unresolved": EvidenceStatus.UNRESOLVED.value,
        "source_not_found_unresolved": EvidenceStatus.UNRESOLVED.value,
        "stale_source_rejected": EvidenceStatus.UNRESOLVED.value,
        "conflicting_live_sources": EvidenceStatus.CONFLICTING.value,
        "insufficient_evidence_unresolved": EvidenceStatus.UNRESOLVED.value,
    }
    results = {
        name: retrieve_live(next(case for case in cases() if case.name == name), "live_grounded")
        for name in expected
    }

    assert {name: row["status"] for name, row in results.items()} == expected
    assert results["conflicting_live_sources"]["action"] == CandidateAction.ASK_EXTERNAL_REASONER.value
    assert all(not row["unsupported_claim"] for row in results.values())


def test_grounded_fallback_improves_resolution_without_unsupported_claims():
    policies = REPORT["policies"]
    grounded = policies["live_grounded_with_fallback"]["metrics"]
    top1 = policies["live_top1"]["metrics"]
    search_only = policies["live_search_only"]["metrics"]

    assert grounded["gap_resolution_rate"] > top1["gap_resolution_rate"]
    assert grounded["unsupported_claim_rate"] == 0.0
    assert search_only["unsupported_claim_rate"] > grounded["unsupported_claim_rate"]
    assert grounded["conflict_detection_rate"] == 1.0
    assert grounded["stale_source_rejection_rate"] == 1.0
    assert grounded["retrieval_failure_recovery_rate"] == 1.0
    assert grounded["post_retrieval_action_accuracy"] == 1.0


def test_report_contains_the_requested_policy_comparison():
    assert set(REPORT["policies"]) == {
        "sealed_grounded",
        "live_top1",
        "live_search_only",
        "live_grounded",
        "live_grounded_with_fallback",
    }
    for policy in REPORT["policies"].values():
        assert {
            "gap_resolution_rate",
            "supported_claim_rate",
            "unsupported_claim_rate",
            "source_attribution_accuracy",
            "conflict_detection_rate",
            "stale_source_rejection_rate",
            "retrieval_failure_recovery_rate",
            "post_retrieval_action_accuracy",
        } <= set(policy["metrics"])
