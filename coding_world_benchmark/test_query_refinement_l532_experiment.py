from coding_world_benchmark.query_refinement_l532_experiment import (
    QueryStage,
    canonical_url,
    cases,
    generate_query,
    rank_sources,
    run_query_experiment,
)


REPORT = run_query_experiment()


def test_knowledge_gaps_generate_category_specific_queries():
    assert all(generate_query(case.gap).gap_category == case.gap.category for case in cases())
    assert all(generate_query(case.gap).stage is QueryStage.INITIAL for case in cases())
    assert REPORT["metrics"]["query_generation_accuracy"] == 1.0


def test_refinement_runs_only_for_weak_initial_result_sets():
    rows = {row["case"]: row for row in REPORT["rows"]}
    assert rows["refine_power_query"]["refined"]
    assert rows["refine_domain_query"]["refined"]
    assert not rows["rank_api_sources"]["refined"]
    assert not rows["dedup_search_results"]["refined"]
    assert REPORT["metrics"]["query_refinement_accuracy"] == 1.0
    assert REPORT["metrics"]["query_budget_violation_rate"] == 0.0


def test_ranking_prefers_authoritative_relevant_source_over_weak_snippet():
    row = next(row for row in REPORT["rows"] if row["case"] == "rank_api_sources")
    assert row["top1_gold"]
    assert row["ranked_sources"][0]["source_type"] == "paper" or row["ranked_sources"][0]["source_type"] == "official_docs"
    assert REPORT["metrics"]["top1_ranking_accuracy"] == 1.0


def test_url_tracking_variants_are_deduplicated_before_grounding():
    row = next(row for row in REPORT["rows"] if row["case"] == "dedup_search_results")
    assert row["raw_result_count"] == 2
    assert row["unique_source_count"] == 1
    assert row["duplicate_count"] == 1
    assert canonical_url("HTTPS://SEARCH.EXAMPLE.TEST/control/control-paper/?utm_source=x#section") == "https://search.example.test/control/control-paper"
    assert REPORT["metrics"]["deduplication_accuracy"] == 1.0


def test_refined_results_recover_gold_sources_without_exceeding_budget():
    metrics = REPORT["metrics"]
    assert metrics["refinement_success_rate"] == 1.0
    assert metrics["top1_ranking_accuracy"] == 1.0
    assert metrics["duplicate_reduction_rate"] > 0.0
