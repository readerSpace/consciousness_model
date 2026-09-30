"""L7.1: scientific knowledge retrieval as a grounded, testable loop.

All external boundaries are injectable. The default adapter is an offline
replay corpus so the coding system remains deterministic and API-free, while
the interfaces can later be backed by web_search/web_fetch/paper_search.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class KnowledgeGap:
    question: str
    reason: str
    blocking: bool
    confidence: float


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    snippet: str
    source_type: str
    authority: float
    published_year: int | None = None


@dataclass(frozen=True)
class Evidence:
    claim: str
    source_url: str
    source_type: str
    quote_or_span: str
    equations: tuple[str, ...]
    relevance: float
    authority: float
    recency: float
    confidence: float
    supports: bool | None


@dataclass(frozen=True)
class RetrievalOutcome:
    status: str
    evidence: tuple[Evidence, ...]
    queries: tuple[str, ...]
    unresolved_questions: tuple[str, ...]
    suggested_actions: tuple[str, ...]
    conflicts: tuple[str, ...]


def format_retrieval_outcome(outcome: RetrievalOutcome) -> str:
    """Render retrieval state without turning missing evidence into a claim."""
    lines = [f"- status: `{outcome.status}`", "- queries:"]
    lines.extend(f"  - `{query}`" for query in outcome.queries)
    lines.append("- evidence:")
    if outcome.evidence:
        for item in outcome.evidence:
            lines.append(f"  - `{item.source_url}` relevance={item.relevance:.2f} authority={item.authority:.2f} confidence={item.confidence:.2f}")
    else:
        lines.append("  - none")
    if outcome.unresolved_questions:
        lines.append("- unresolved: " + "; ".join(outcome.unresolved_questions))
    return "\n".join(lines)


class SearchAdapter(Protocol):
    def search(self, query: str) -> list[SearchResult]: ...


class PageFetcher(Protocol):
    def fetch(self, url: str) -> str: ...


class KnowledgeGapDetector:
    def detect(self, task: str) -> tuple[KnowledgeGap, ...]:
        lower = task.lower()
        gaps: list[KnowledgeGap] = []
        if any(word in task for word in ("量子情報", "時空", "創発")) or any(
            word in lower for word in ("quantum information", "emergent spacetime")
        ):
            gaps.append(KnowledgeGap("entanglement and geometry relation", "required to construct a testable spacetime model", True, 0.2))
            gaps.append(KnowledgeGap("candidate finite toy models", "required to choose an executable experiment", True, 0.15))
        elif any(word in task for word in ("荷電粒子", "放射", "電磁波")) or "radiation" in lower:
            gaps.append(KnowledgeGap("relativistic radiation power formula", "required to design the simulation", True, 0.2))
        return tuple(gaps)


class QueryGenerator:
    def generate(self, gap: KnowledgeGap) -> tuple[str, ...]:
        if "entanglement" in gap.question or "geometry" in gap.question:
            return (
                "quantum information entanglement emergent spacetime review",
                "holographic entanglement entropy geometry arXiv",
                "tensor network holographic quantum error correction toy model",
            )
        if "radiation" in gap.question:
            return (
                "relativistic accelerated charge radiation formula",
                "Lienard formula accelerated charge radiation",
                "relativistic Larmor formula review",
            )
        return (gap.question,)


class OfflineSearchAdapter:
    def __init__(self, results: tuple[SearchResult, ...] | None = None):
        self.results = results or _DEFAULT_RESULTS

    def search(self, query: str) -> list[SearchResult]:
        tokens = {token.lower() for token in query.split() if len(token) > 3}
        ranked = sorted(self.results, key=lambda item: sum(token in (item.title + " " + item.snippet).lower() for token in tokens), reverse=True)
        return [item for item in ranked if any(token in (item.title + " " + item.snippet).lower() for token in tokens)]


class OfflinePageFetcher:
    def __init__(self, pages: dict[str, str] | None = None):
        self.pages = pages or {item.url: item.snippet for item in _DEFAULT_RESULTS}

    def fetch(self, url: str) -> str:
        return self.pages.get(url, "")


class EvidenceExtractor:
    def extract(self, result: SearchResult, content: str, gap: KnowledgeGap) -> Evidence | None:
        if not content or not any(token in content.lower() for token in gap.question.split() if len(token) > 3):
            return None
        claim = result.snippet
        equations: tuple[str, ...] = ()
        if "entanglement entropy" in content.lower():
            equations = (r"S_A = \frac{\mathrm{Area}(\gamma_A)}{4G_N}",)
        relevance = 0.95 if "quantum" in content.lower() or "entanglement" in content.lower() else 0.7
        recency = 0.7 if result.published_year and result.published_year >= 2010 else 0.5
        confidence = relevance * result.authority
        return Evidence(claim, result.url, result.source_type, content[:320], equations, relevance, result.authority, recency, confidence, True)


class SourceRanker:
    def rank(self, evidence: tuple[Evidence, ...]) -> tuple[Evidence, ...]:
        return tuple(sorted(evidence, key=lambda item: item.relevance * item.authority * item.recency, reverse=True))


class KnowledgeStore:
    def __init__(self):
        self._evidence: list[Evidence] = []

    def add(self, evidence: tuple[Evidence, ...]) -> None:
        known = {item.source_url for item in self._evidence}
        self._evidence.extend(item for item in evidence if item.source_url not in known)

    def evidence(self) -> tuple[Evidence, ...]:
        return tuple(self._evidence)


class ScientificRetriever:
    def __init__(self, search: SearchAdapter | None = None, fetcher: PageFetcher | None = None):
        self.search = search or OfflineSearchAdapter()
        self.fetcher = fetcher or OfflinePageFetcher()
        self.detector = KnowledgeGapDetector()
        self.queries = QueryGenerator()
        self.extractor = EvidenceExtractor()
        self.ranker = SourceRanker()
        self.store = KnowledgeStore()

    def resolve_gap(self, gap: KnowledgeGap) -> RetrievalOutcome:
        queries = self.queries.generate(gap)
        extracted: list[Evidence] = []
        for query in queries:
            for result in self.search.search(query):
                content = self.fetcher.fetch(result.url)
                evidence = self.extractor.extract(result, content, gap)
                if evidence:
                    extracted.append(evidence)
        ranked = self.ranker.rank(tuple(extracted))
        self.store.add(ranked)
        unique_claims = {item.claim for item in ranked}
        conflicts = tuple("conflicting_evidence" for _ in [] if False)
        if not ranked:
            return RetrievalOutcome("UNRESOLVED", (), queries, (gap.question,), ("refine query", "request human clarification"), conflicts)
        status = "SUPPORTED" if len(unique_claims) >= 1 else "UNRESOLVED"
        return RetrievalOutcome(status, ranked, queries, (), ("integrate evidence into ResearchPlan",), conflicts)

    def resolve_task(self, task: str) -> tuple[KnowledgeGap, ...] | tuple[RetrievalOutcome, ...]:
        gaps = self.detector.detect(task)
        if not gaps:
            return (RetrievalOutcome("UNRESOLVED", (), (), ("scientific topic",), ("clarify domain",), ()),)
        return tuple(self.resolve_gap(gap) for gap in gaps)


_DEFAULT_RESULTS = (
    SearchResult("Holographic Derivation of Entanglement Entropy", "https://arxiv.org/abs/hep-th/0603001", "Entanglement entropy in a conformal field theory can be obtained from the area of a minimal surface in the bulk.", "paper", 0.98, 2006),
    SearchResult("Building up spacetime with quantum entanglement", "https://arxiv.org/abs/1005.3035", "The emergence of classically connected spacetime is related to quantum entanglement between degrees of freedom.", "paper", 0.95, 2010),
    SearchResult("Holographic quantum error-correcting codes", "https://arxiv.org/abs/1503.06237", "Tensor-network quantum error-correcting codes give toy models of bulk and boundary information encoding.", "paper", 0.97, 2015),
)
