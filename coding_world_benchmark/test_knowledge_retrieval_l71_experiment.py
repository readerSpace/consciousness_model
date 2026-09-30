from coding_world_benchmark.coding_agent_app import LocalWorkspaceAgent
from coding_world_benchmark.knowledge_retrieval_l71_experiment import (
    KnowledgeGapDetector,
    KnowledgeGap,
    QueryGenerator,
    ScientificRetriever,
)


REQUEST = "量子情報から時空が創発するか検証するために論文を検索してそれを参考に検証して"


def test_gap_detection_and_canonical_query_generation():
    gaps = KnowledgeGapDetector().detect(REQUEST)
    assert gaps
    assert gaps[0].blocking is True
    queries = QueryGenerator().generate(gaps[0])
    assert queries[0] == "quantum information entanglement emergent spacetime review"
    assert all(query.isascii() for query in queries)


def test_retriever_fetches_extracts_ranks_and_integrates_evidence():
    outcomes = ScientificRetriever().resolve_task(REQUEST)
    assert len(outcomes) == 2
    assert all(outcome.status == "SUPPORTED" for outcome in outcomes)
    assert any(evidence.source_type == "paper" for outcome in outcomes for evidence in outcome.evidence)
    assert any(evidence.equations for outcome in outcomes for evidence in outcome.evidence)
    assert all(outcome.queries for outcome in outcomes)


def test_unresolved_gap_stops_without_unsupported_claim():
    outcome = ScientificRetriever().resolve_gap(KnowledgeGap("unmapped mechanism", "not indexed", True, 0.1))
    assert outcome.status == "UNRESOLVED"
    assert not outcome.evidence
    assert outcome.unresolved_questions


def test_app_exposes_search_queries_and_evidence_path_before_repo_search(tmp_path):
    result = LocalWorkspaceAgent(tmp_path).handle(REQUEST)
    assert "【検索・Evidence統合】" in result.text
    assert "quantum information entanglement" in result.text
    assert "候補ファイル" not in result.text
