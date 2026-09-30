"""L5.3.3: ranked-source acquisition, citation chains, and evidence fusion.

L5.3.2 chooses sources and L5.3.1 acquires pages.  This layer preserves the
lineage between those steps, removes URL duplicates before fetching, and fuses
independent fetched documents only when their claims agree.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
from typing import Callable

from .knowledge_gap_l49_experiment import KnowledgeGap
from .live_retrieval_adapter_l531_experiment import (
    BrowserRetrievalAdapter,
    HttpRetrievalAdapter,
    RawResponse,
    RetrievalRequest,
    RetrievalResult,
    fetch_with_browser_fallback,
)
from .live_web_retrieval_l53_experiment import LiveRetrievalStatus
from .query_refinement_l532_experiment import (
    SearchQuery,
    SearchResult,
    canonical_url,
    rank_sources,
    search_with_refinement,
)


class IntegratedEvidenceStatus(Enum):
    SUPPORTED = "supported"
    CONFLICTING = "conflicting_evidence"
    UNRESOLVED = "unresolved_gap"


@dataclass(frozen=True)
class CitationNode:
    citation_id: str
    source_url: str
    canonical_source_url: str
    source_type: str
    independence_group: str
    claim: str
    supports: bool | None
    retrieval_status: LiveRetrievalStatus
    quoted_span: str


@dataclass(frozen=True)
class IntegratedEvidence:
    status: IntegratedEvidenceStatus
    claim: str | None
    citation_ids: tuple[str, ...]
    source_urls: tuple[str, ...]
    independent_source_count: int


@dataclass(frozen=True)
class CitationCase:
    name: str
    gap: KnowledgeGap
    search_results: tuple[SearchResult, ...]
    responses: dict[str, RawResponse]
    browser_responses: dict[str, RawResponse]
    expected_status: IntegratedEvidenceStatus
    expected_citations: int
    expected_independent_sources: int
    expected_browser_fallback: bool = False


def _claim(text: str) -> tuple[str, bool | None]:
    claim = text.split("Claim:", 1)[-1].strip()
    lowered = claim.lower()
    if "does not" in lowered or "not require" in lowered or "fails to" in lowered:
        return claim, False
    return claim, True


def build_citation_chain(results: tuple[SearchResult, ...], fetched: tuple[RetrievalResult, ...]) -> tuple[CitationNode, ...]:
    nodes = []
    for index, (result, retrieval) in enumerate(zip(results, fetched), start=1):
        document = retrieval.document
        if retrieval.status is not LiveRetrievalStatus.SUCCESS or document is None:
            continue
        claim, supports = _claim(document.text)
        nodes.append(CitationNode(
            f"cite-{index}", result.url, canonical_url(result.url), result.source_type,
            result.independence_group, claim, supports, retrieval.status, document.text,
        ))
    return tuple(nodes)


def integrate_evidence(nodes: tuple[CitationNode, ...]) -> IntegratedEvidence:
    if not nodes:
        return IntegratedEvidence(IntegratedEvidenceStatus.UNRESOLVED, None, (), (), 0)
    claims = {node.claim for node in nodes}
    support_values = {node.supports for node in nodes if node.supports is not None}
    citation_ids = tuple(node.citation_id for node in nodes)
    urls = tuple(node.source_url for node in nodes)
    independent = len({node.independence_group for node in nodes})
    if len(claims) > 1 or len(support_values) > 1:
        return IntegratedEvidence(IntegratedEvidenceStatus.CONFLICTING, None, citation_ids, urls, independent)
    return IntegratedEvidence(IntegratedEvidenceStatus.SUPPORTED, nodes[0].claim, citation_ids, urls, independent)


def _response_for(case: CitationCase, url: str, browser: bool = False) -> RawResponse:
    responses = case.browser_responses if browser else case.responses
    return responses.get(url, RawResponse(404, {"Content-Type": "text/html"}, "", url))


def run_citation_pipeline(case: CitationCase, *, max_sources: int = 5) -> dict[str, object]:
    def search(_query: SearchQuery) -> tuple[SearchResult, ...]:
        return case.search_results

    search_row = search_with_refinement(case_to_query_case(case), search, max_queries=1)
    ranked = rank_sources(case.search_results, limit=max_sources)
    fetched = []
    for result in ranked:
        request = RetrievalRequest(result.url, search_row["queries"][0], ("evidence",), max_age_days=365)
        http = HttpRetrievalAdapter(lambda url, case=case: _response_for(case, url))
        browser = BrowserRetrievalAdapter(lambda url, case=case: _response_for(case, url, browser=True))
        fetched.append(fetch_with_browser_fallback(request, http, browser))
    chain = build_citation_chain(ranked, tuple(fetched))
    integrated = integrate_evidence(chain)
    browser_used = any(item.used_browser_fallback for item in fetched)
    return {
        "case": case.name,
        "query": search_row["queries"][0],
        "ranked_count": len(ranked),
        "fetched_count": len(fetched),
        "raw_result_count": len(case.search_results),
        "duplicate_count": len(case.search_results) - len({canonical_url(item.url) for item in case.search_results}),
        "citation_chain": [item.__dict__ | {"retrieval_status": item.retrieval_status.value} for item in chain],
        "status": integrated.status.value,
        "expected_status": case.expected_status.value,
        "status_correct": integrated.status is case.expected_status,
        "claim": integrated.claim,
        "citation_count": len(integrated.citation_ids),
        "expected_citations": case.expected_citations,
        "citation_complete": len(integrated.citation_ids) == case.expected_citations,
        "independent_source_count": integrated.independent_source_count,
        "expected_independent_sources": case.expected_independent_sources,
        "independence_complete": integrated.independent_source_count >= case.expected_independent_sources,
        "duplicate_fetch_suppressed": len(fetched) < len(case.search_results) if len(case.search_results) > len(ranked) else True,
        "browser_fallback": browser_used,
        "unsupported_claim": integrated.status is IntegratedEvidenceStatus.SUPPORTED and not chain,
    }


def case_to_query_case(case: CitationCase):
    from .query_refinement_l532_experiment import QueryCase
    return QueryCase(case.name, case.gap, (), case.search_results, (), False)


def _gap(category: str, question: str) -> KnowledgeGap:
    return KnowledgeGap(question, "L5.3.3 citation grounding case.", True, 0.25, 0.9, category)


def _result(identifier: str, category: str, *, authority: float, relevance: float,
            independent: str, source_type: str = "paper", tracking: str = "") -> SearchResult:
    return SearchResult(identifier, f"https://citation.example.test/{category}/{identifier}{tracking}", "evidence", authority, relevance, "2026-09-01", source_type, independent, True)


def _html(claim: str, *, published: str = "2026-09-01") -> RawResponse:
    return RawResponse(200, {"Content-Type": "text/html"}, f'<html><title>Source</title><body data-published="{published}">Claim: {claim} evidence</body></html>', "")


def cases() -> tuple[CitationCase, ...]:
    power = _gap("power_analysis", "What evidence guides sample-size planning?")
    measure = _gap("measurement_method", "Does the estimator require independent samples?")
    return (
        CitationCase("multi_source_agreement", power,
                     (_result("guide", "power", authority=.95, relevance=.95, independent="guide", source_type="official_docs"),
                      _result("paper", "power", authority=.88, relevance=.9, independent="paper")),
                     {"https://citation.example.test/power/guide": _html("sample size uses effect size and target power"),
                      "https://citation.example.test/power/paper": _html("sample size uses effect size and target power")}, {}, IntegratedEvidenceStatus.SUPPORTED, 2, 2),
        CitationCase("duplicate_removed_before_fetch", power,
                     (_result("guide", "power", authority=.95, relevance=.95, independent="guide", source_type="official_docs"),
                      _result("guide", "power", authority=.9, relevance=.9, independent="mirror", tracking="?utm_source=mirror")),
                     {"https://citation.example.test/power/guide": _html("sample size uses effect size and target power")}, {}, IntegratedEvidenceStatus.SUPPORTED, 1, 1),
        CitationCase("conflicting_claims", measure,
                     (_result("yes", "measurement", authority=.9, relevance=.95, independent="guide"),
                      _result("no", "measurement", authority=.9, relevance=.95, independent="paper")),
                     {"https://citation.example.test/measurement/yes": _html("estimator requires independent samples"),
                      "https://citation.example.test/measurement/no": _html("estimator does not require independent samples")}, {}, IntegratedEvidenceStatus.CONFLICTING, 2, 2),
        CitationCase("stale_not_grounded", power,
                     (_result("old", "power", authority=.95, relevance=.95, independent="old"),),
                     {"https://citation.example.test/power/old": _html("sample size uses effect size", published="2020-01-01")}, {}, IntegratedEvidenceStatus.UNRESOLVED, 0, 0),
        CitationCase("browser_only_source", power,
                     (_result("rendered", "power", authority=.95, relevance=.95, independent="browser", source_type="official_docs"),),
                     {"https://citation.example.test/power/rendered": RawResponse(200, {"Content-Type": "text/html"}, "<body>Enable JavaScript to continue.</body>", "https://citation.example.test/power/rendered")},
                     {"https://citation.example.test/power/rendered": _html("sample size uses effect size and target power")}, IntegratedEvidenceStatus.SUPPORTED, 1, 1, True),
    )


def run_citation_grounding_experiment() -> dict[str, object]:
    rows = [run_citation_pipeline(case) for case in cases()]
    return {"rows": rows, "metrics": {
        "citation_chain_accuracy": sum(row["status_correct"] for row in rows) / len(rows),
        "citation_completeness": sum(row["citation_complete"] for row in rows) / len(rows),
        "multi_source_integration_accuracy": sum(row["independence_complete"] for row in rows) / len(rows),
        "duplicate_fetch_suppression_rate": sum(row["duplicate_fetch_suppressed"] for row in rows if row["duplicate_count"] > 0) / max(1, sum(row["duplicate_count"] > 0 for row in rows)),
        "grounded_support_rate": sum(row["status"] == IntegratedEvidenceStatus.SUPPORTED.value for row in rows) / len(rows),
        "unsupported_claim_rate": sum(row["unsupported_claim"] for row in rows) / len(rows),
        "browser_fallback_rate": sum(row["browser_fallback"] for row in rows) / len(rows),
    }}


if __name__ == "__main__":
    print(json.dumps(run_citation_grounding_experiment(), ensure_ascii=False, indent=2, sort_keys=True))
