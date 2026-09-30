from coding_world_benchmark.citation_grounding_l533_experiment import (
    IntegratedEvidenceStatus,
    cases,
    run_citation_grounding_experiment,
    run_citation_pipeline,
)


REPORT = run_citation_grounding_experiment()


def test_pipeline_preserves_citation_chain_from_ranked_source_to_document():
    row = next(row for row in REPORT["rows"] if row["case"] == "multi_source_agreement")
    assert row["status"] == IntegratedEvidenceStatus.SUPPORTED.value
    assert row["citation_count"] == 2
    assert row["independent_source_count"] == 2
    assert all(item["source_url"].startswith("https://") for item in row["citation_chain"])
    assert all(item["retrieval_status"] == "success" for item in row["citation_chain"])


def test_duplicate_sources_are_removed_before_fetching():
    row = next(row for row in REPORT["rows"] if row["case"] == "duplicate_removed_before_fetch")
    assert row["raw_result_count"] == 2
    assert row["duplicate_count"] == 1
    assert row["ranked_count"] == 1
    assert row["fetched_count"] == 1
    assert row["citation_count"] == 1
    assert row["duplicate_fetch_suppressed"]


def test_independent_conflicting_claims_are_not_fused_into_support():
    row = next(row for row in REPORT["rows"] if row["case"] == "conflicting_claims")
    assert row["status"] == IntegratedEvidenceStatus.CONFLICTING.value
    assert row["claim"] is None
    assert row["citation_count"] == 2
    assert row["independent_source_count"] == 2


def test_stale_document_is_removed_from_final_evidence():
    row = next(row for row in REPORT["rows"] if row["case"] == "stale_not_grounded")
    assert row["status"] == IntegratedEvidenceStatus.UNRESOLVED.value
    assert row["citation_count"] == 0
    assert row["unsupported_claim"] is False


def test_browser_only_source_enters_the_same_grounded_citation_chain():
    case = next(case for case in cases() if case.name == "browser_only_source")
    row = run_citation_pipeline(case)
    assert row["status"] == IntegratedEvidenceStatus.SUPPORTED.value
    assert row["browser_fallback"]
    assert row["citation_chain"][0]["source_type"] == "official_docs"


def test_l533_metrics_show_grounded_multi_source_and_zero_unsupported_claims():
    metrics = REPORT["metrics"]
    assert metrics["citation_chain_accuracy"] == 1.0
    assert metrics["citation_completeness"] == 1.0
    assert metrics["multi_source_integration_accuracy"] == 1.0
    assert metrics["duplicate_fetch_suppression_rate"] == 1.0
    assert metrics["unsupported_claim_rate"] == 0.0
    assert metrics["browser_fallback_rate"] > 0.0
