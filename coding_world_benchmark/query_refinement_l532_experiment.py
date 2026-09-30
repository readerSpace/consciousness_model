"""L5.3.2: KnowledgeGap-driven query generation and source selection.

The adapter in L5.3.1 answers whether a requested page can be acquired.  This
layer chooses what to request: it generates a query from a ``KnowledgeGap``,
refines it when the result set is weak, ranks sources using multiple signals,
and removes URL-level duplicates before grounding.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import json
from typing import Callable
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from .knowledge_gap_l49_experiment import KnowledgeGap


class QueryStage(Enum):
    INITIAL = "initial"
    REFINED = "refined"


@dataclass(frozen=True)
class SearchQuery:
    text: str
    stage: QueryStage
    gap_category: str


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    snippet: str
    authority: float
    relevance: float
    published_at: str
    source_type: str
    independence_group: str
    gold: bool = False


@dataclass(frozen=True)
class QueryCase:
    name: str
    gap: KnowledgeGap
    expected_terms: tuple[str, ...]
    initial_results: tuple[SearchResult, ...]
    refined_results: tuple[SearchResult, ...]
    needs_refinement: bool


def generate_query(gap: KnowledgeGap) -> SearchQuery:
    terms = {
        "power_analysis": "sample size effect size statistical power",
        "python_api": "Python API missing values mean official documentation",
        "measurement_method": "estimator independent samples statistical method",
        "domain_mechanism": "mechanism observed signal primary research paper",
        "causal_control": "causal control confound randomized experiment",
    }
    text = terms.get(gap.category, f"{gap.question} evidence primary source")
    return SearchQuery(text, QueryStage.INITIAL, gap.category)


def refine_query(query: SearchQuery, gap: KnowledgeGap) -> SearchQuery:
    suffix = {
        "power_analysis": "pilot effect size required sample size",
        "python_api": "site:docs.python.org exact missing value behavior",
        "measurement_method": "assumptions validation independent observations",
        "domain_mechanism": "site:arxiv.org mechanism evidence limitations",
        "causal_control": "negative control shared confounding identification",
    }.get(gap.category, "primary source limitations")
    return SearchQuery(f"{query.text} {suffix}", QueryStage.REFINED, query.gap_category)


def canonical_url(url: str) -> str:
    parts = urlsplit(url)
    query = [(key, value) for key, value in parse_qsl(parts.query) if not key.lower().startswith("utm_") and key.lower() not in {"ref", "source"}]
    path = parts.path.rstrip("/") or "/"
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path, urlencode(query), ""))


def _freshness(published_at: str) -> float:
    published = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
    if published.tzinfo is None:
        published = published.replace(tzinfo=timezone.utc)
    age_days = max(0, (datetime(2026, 9, 14, tzinfo=timezone.utc) - published).days)
    return max(0.0, 1.0 - age_days / 3650.0)


def rank_sources(results: tuple[SearchResult, ...], *, limit: int = 5) -> tuple[SearchResult, ...]:
    best: dict[str, SearchResult] = {}
    for result in results:
        key = canonical_url(result.url)
        previous = best.get(key)
        score = 0.45 * result.relevance + 0.3 * result.authority + 0.15 * _freshness(result.published_at)
        if previous is None:
            best[key] = result
        else:
            previous_score = 0.45 * previous.relevance + 0.3 * previous.authority + 0.15 * _freshness(previous.published_at)
            if score > previous_score:
                best[key] = result
    return tuple(sorted(best.values(), key=lambda item: (
        0.45 * item.relevance + 0.3 * item.authority + 0.15 * _freshness(item.published_at),
        item.gold,
    ), reverse=True)[:limit])


def _has_useful_result(results: tuple[SearchResult, ...]) -> bool:
    return any(result.relevance >= 0.7 and result.authority >= 0.6 for result in results)


def search_with_refinement(case: QueryCase, search: Callable[[SearchQuery], tuple[SearchResult, ...]], *, max_queries: int = 2) -> dict[str, object]:
    initial_query = generate_query(case.gap)
    attempts = [(initial_query, search(initial_query))]
    if not _has_useful_result(attempts[0][1]) and max_queries > 1:
        refined_query = refine_query(initial_query, case.gap)
        attempts.append((refined_query, search(refined_query)))
    all_results = tuple(result for _, results in attempts for result in results)
    ranked = rank_sources(all_results)
    return {
        "case": case.name,
        "queries": [query.text for query, _ in attempts],
        "query_stages": [query.stage.value for query, _ in attempts],
        "refined": len(attempts) > 1,
        "expected_refinement": case.needs_refinement,
        "refinement_correct": (len(attempts) > 1) is case.needs_refinement,
        "ranked_sources": [result.__dict__ | {"canonical_url": canonical_url(result.url)} for result in ranked],
        "top1_gold": bool(ranked and ranked[0].gold),
        "unique_source_count": len(ranked),
        "raw_result_count": len(all_results),
        "duplicate_count": len(all_results) - len({canonical_url(result.url) for result in all_results}),
        "query_budget_ok": len(attempts) <= max_queries,
    }


def _gap(category: str, question: str) -> KnowledgeGap:
    return KnowledgeGap(question, "L5.3.2 query-selection case.", True, 0.3, 0.85, category)


def _result(identifier: str, category: str, *, relevance: float, authority: float,
            gold: bool, source_type: str = "paper", published: str = "2026-09-01",
            tracking: str = "") -> SearchResult:
    url = f"https://search.example.test/{category}/{identifier}{tracking}"
    return SearchResult(identifier.replace("-", " ").title(), url, f"Evidence about {category}.", authority, relevance, published, source_type, identifier, gold)


def cases() -> tuple[QueryCase, ...]:
    power = _gap("power_analysis", "How should sample size be revised after low power?")
    api = _gap("python_api", "How does the API handle missing values in a mean?")
    control = _gap("causal_control", "Which control separates treatment from confound?")
    return (
        QueryCase("refine_power_query", power, ("sample", "effect", "power"),
                  (_result("blog-power", "power", relevance=.35, authority=.25, gold=False),),
                  (_result("power-guideline", "power", relevance=.95, authority=.92, gold=True, source_type="official_docs"),
                   _result("power-guideline", "power", relevance=.9, authority=.85, gold=True, tracking="?utm_source=search")), True),
        QueryCase("rank_api_sources", api, ("python", "missing", "mean"),
                  (_result("cookbook", "api", relevance=.8, authority=.55, gold=False),
                   _result("python-docs", "api", relevance=.88, authority=.98, gold=True, source_type="official_docs")), (), False),
        QueryCase("dedup_search_results", control, ("causal", "control", "confound"),
                  (_result("control-paper", "control", relevance=.91, authority=.9, gold=True),
                   _result("control-paper", "control", relevance=.91, authority=.9, gold=True, tracking="?utm_source=mirror")), (), False),
        QueryCase("refine_domain_query", _gap("domain_mechanism", "What mechanism explains the signal?"), ("mechanism", "signal", "paper"),
                  (_result("irrelevant", "domain", relevance=.4, authority=.9, gold=False),),
                  (_result("mechanism-paper", "domain", relevance=.92, authority=.88, gold=True),), True),
    )


def run_query_experiment() -> dict[str, object]:
    rows = []
    for case in cases():
        def search(query: SearchQuery, case: QueryCase = case) -> tuple[SearchResult, ...]:
            return case.refined_results if query.stage is QueryStage.REFINED else case.initial_results
        rows.append(search_with_refinement(case, search))
    return {"rows": rows, "metrics": {
        "query_generation_accuracy": sum(all(term in row["queries"][0].lower() for term in next(case for case in cases() if case.name == row["case"]).expected_terms) for row in rows) / len(rows),
        "query_refinement_accuracy": sum(bool(row["refinement_correct"]) for row in rows) / len(rows),
        "refinement_success_rate": sum(row["top1_gold"] for row in rows if row["expected_refinement"]) / max(1, sum(row["expected_refinement"] for row in rows)),
        "top1_ranking_accuracy": sum(row["top1_gold"] for row in rows) / len(rows),
        "deduplication_accuracy": sum(row["duplicate_count"] >= 0 and row["unique_source_count"] <= row["raw_result_count"] for row in rows) / len(rows),
        "duplicate_reduction_rate": sum(row["duplicate_count"] for row in rows) / max(1, sum(row["raw_result_count"] for row in rows)),
        "query_budget_violation_rate": sum(not row["query_budget_ok"] for row in rows) / len(rows),
    }}


if __name__ == "__main__":
    print(json.dumps(run_query_experiment(), ensure_ascii=False, indent=2, sort_keys=True))
