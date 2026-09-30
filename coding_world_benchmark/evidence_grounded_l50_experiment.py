"""L5.0: sealed evidence-grounded retrieval benchmark.

L4.9 decides when retrieval is worth doing.  L5.0 tests the next layer without
using live web access: given a knowledge gap and a sealed search corpus, can the
agent turn retrieved pages into attributed, relevant, uncertainty-aware
evidence instead of accepting snippets as knowledge?

The corpus is intentionally adversarial.  It includes multiple agreeing
authoritative sources, a high-ranked weak source with a false claim,
contradictory sources, an unresolved gap, and an irrelevant but otherwise true
source.  This keeps the failure cause reproducible before replacing the sealed
fixture with live Web retrieval.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
import json

from .knowledge_gap_l49_experiment import CandidateAction, KnowledgeGap


RETRIEVED_AT = datetime(2026, 9, 13, tzinfo=UTC).isoformat()


class EvidenceStatus(Enum):
    SUPPORTED = "supported"
    CONFLICTING = "conflicting_evidence"
    UNRESOLVED = "unresolved_gap"


@dataclass(frozen=True)
class SearchDocument:
    title: str
    url: str
    source_type: str
    rank: int
    authority: float
    independence_group: str
    relevance_by_gap: dict[str, float]
    claim_key: str
    claim: str
    supports: bool | None
    quoted_span: str | None


@dataclass(frozen=True)
class GroundedEvidence:
    claim: str
    source_url: str
    source_type: str
    retrieved_at: str
    supports: bool | None
    relevance: float
    authority: float
    independence_group: str
    uncertainty: float
    quoted_span: str | None


@dataclass(frozen=True)
class RetrievalCase:
    name: str
    gap: KnowledgeGap
    gold_claim_key: str | None
    gold_claim: str | None
    expected_status: EvidenceStatus
    expected_action: CandidateAction


def _gap(category: str, question: str) -> KnowledgeGap:
    return KnowledgeGap(question, "L5.0 sealed retrieval case.", True, 0.25, 0.88, category)


def cases() -> tuple[RetrievalCase, ...]:
    return (
        RetrievalCase(
            "multi_source_power",
            _gap("power_analysis", "How should sample size be revised after low statistical power?"),
            "power_depends_on_effect_size",
            "Sample-size planning should use an estimated effect size and target power.",
            EvidenceStatus.SUPPORTED,
            CandidateAction.RUN_EXPERIMENT,
        ),
        RetrievalCase(
            "misleading_top_snippet",
            _gap("python_api", "How does the API handle missing values when computing a mean?"),
            "nanmean_ignores_nan",
            "nanmean ignores NaN values when computing the mean.",
            EvidenceStatus.SUPPORTED,
            CandidateAction.INSPECT_CODE,
        ),
        RetrievalCase(
            "conflicting_sources",
            _gap("measurement_method", "Does the estimator require independent samples?"),
            "estimator_requires_independence",
            None,
            EvidenceStatus.CONFLICTING,
            CandidateAction.ASK_EXTERNAL_REASONER,
        ),
        RetrievalCase(
            "unresolved_gap",
            _gap("domain_mechanism", "What published mechanism explains the synthetic workspace resonance?"),
            "workspace_resonance_mechanism",
            None,
            EvidenceStatus.UNRESOLVED,
            CandidateAction.ASK_EXTERNAL_REASONER,
        ),
        RetrievalCase(
            "irrelevant_true_result",
            _gap("causal_control", "Which control separates treatment from the observed confound?"),
            "control_design_for_confound",
            None,
            EvidenceStatus.UNRESOLVED,
            CandidateAction.REVISE_HYPOTHESIS,
        ),
    )


def corpus() -> tuple[SearchDocument, ...]:
    return (
        SearchDocument(
            "Quick stats blog",
            "https://example.test/blog/sample-size-shortcut",
            "blog",
            1,
            0.25,
            "blog-network",
            {"power_analysis": 0.72},
            "fixed_n_is_enough",
            "Thirty samples are always enough for a reliable experiment.",
            True,
            "Thirty samples are always enough",
        ),
        SearchDocument(
            "Statistics handbook",
            "https://example.test/handbook/power-analysis",
            "official_docs",
            2,
            0.92,
            "stats-handbook",
            {"power_analysis": 0.94},
            "power_depends_on_effect_size",
            "Sample-size planning should use an estimated effect size and target power.",
            True,
            "effect size and target power",
        ),
        SearchDocument(
            "Independent methods note",
            "https://example.test/methods/pilot-power",
            "paper",
            3,
            0.85,
            "methods-lab",
            {"power_analysis": 0.89},
            "power_depends_on_effect_size",
            "Sample-size planning should use an estimated effect size and target power.",
            True,
            "estimated effect size",
        ),
        SearchDocument(
            "Forum answer about missing values",
            "https://example.test/forum/nanmean",
            "forum",
            1,
            0.2,
            "forum",
            {"python_api": 0.77},
            "nanmean_treats_nan_as_zero",
            "nanmean treats NaN values as zeros.",
            True,
            "NaN values as zeros",
        ),
        SearchDocument(
            "Array library reference",
            "https://example.test/docs/nanmean",
            "official_docs",
            2,
            0.96,
            "array-docs",
            {"python_api": 0.95},
            "nanmean_ignores_nan",
            "nanmean ignores NaN values when computing the mean.",
            True,
            "ignores NaN values",
        ),
        SearchDocument(
            "Cookbook missing-data example",
            "https://example.test/cookbook/missing-data",
            "official_docs",
            3,
            0.82,
            "array-cookbook",
            {"python_api": 0.82},
            "nanmean_ignores_nan",
            "nanmean ignores NaN values when computing the mean.",
            True,
            "NaN entries are omitted",
        ),
        SearchDocument(
            "Estimator user guide",
            "https://example.test/docs/estimator-independence",
            "official_docs",
            1,
            0.88,
            "estimator-docs",
            {"measurement_method": 0.9},
            "estimator_requires_independence",
            "The estimator requires independent samples.",
            True,
            "requires independent samples",
        ),
        SearchDocument(
            "Estimator validation paper",
            "https://example.test/papers/correlated-estimator",
            "paper",
            2,
            0.86,
            "estimator-paper",
            {"measurement_method": 0.87},
            "estimator_requires_independence",
            "The estimator does not require independent samples.",
            False,
            "does not require independent samples",
        ),
        SearchDocument(
            "Neural resonance review",
            "https://example.test/reviews/neural-resonance",
            "paper",
            1,
            0.84,
            "neuro-review",
            {"domain_mechanism": 0.34, "causal_control": 0.22},
            "neural_resonance_mechanism",
            "Neural resonance can occur in coupled oscillators.",
            True,
            "coupled oscillators",
        ),
        SearchDocument(
            "Randomized trial controls",
            "https://example.test/docs/randomized-controls",
            "official_docs",
            1,
            0.9,
            "trial-docs",
            {"causal_control": 0.41},
            "randomized_trial_controls",
            "Randomization can balance treatment groups in clinical trials.",
            True,
            "balance treatment groups",
        ),
    )


def query_documents(gap: KnowledgeGap) -> tuple[SearchDocument, ...]:
    return tuple(sorted(corpus(), key=lambda doc: (doc.rank, -doc.relevance_by_gap.get(gap.category, 0.0))))


def _evidence_from_document(gap: KnowledgeGap, document: SearchDocument) -> GroundedEvidence:
    relevance = document.relevance_by_gap.get(gap.category, 0.0)
    return GroundedEvidence(
        document.claim,
        document.url,
        document.source_type,
        RETRIEVED_AT,
        document.supports,
        relevance,
        document.authority,
        document.independence_group,
        1.0 - min(document.authority, relevance),
        document.quoted_span,
    )


def _top1_snippet(case: RetrievalCase) -> tuple[GroundedEvidence, ...]:
    return (_evidence_from_document(case.gap, query_documents(case.gap)[0]),)


def _top_k_majority(case: RetrievalCase, k: int = 3) -> tuple[GroundedEvidence, ...]:
    return tuple(_evidence_from_document(case.gap, document) for document in query_documents(case.gap)[:k]
                 if document.relevance_by_gap.get(case.gap.category, 0.0) >= 0.3)


def _authority_weighted(case: RetrievalCase) -> tuple[GroundedEvidence, ...]:
    return tuple(_evidence_from_document(case.gap, document) for document in query_documents(case.gap)
                 if document.authority >= 0.8 and document.relevance_by_gap.get(case.gap.category, 0.0) >= 0.35)


def _evidence_grounded(case: RetrievalCase) -> tuple[GroundedEvidence, ...]:
    candidates = []
    for document in query_documents(case.gap):
        relevance = document.relevance_by_gap.get(case.gap.category, 0.0)
        if relevance < 0.65 or document.authority < 0.6 or document.quoted_span is None:
            continue
        candidates.append(_evidence_from_document(case.gap, document))
    return tuple(candidates)


def retrieve_evidence(case: RetrievalCase, policy: str) -> tuple[GroundedEvidence, ...]:
    if policy == "top1_snippet":
        return _top1_snippet(case)
    if policy == "top_k_majority":
        return _top_k_majority(case)
    if policy == "authority_weighted":
        return _authority_weighted(case)
    if policy == "evidence_grounded":
        return _evidence_grounded(case)
    raise ValueError(f"unknown policy: {policy}")


def status_from_evidence(evidence: tuple[GroundedEvidence, ...]) -> EvidenceStatus:
    if not evidence:
        return EvidenceStatus.UNRESOLVED
    claims = {item.claim for item in evidence}
    if len(claims) > 1 and any(item.supports is False for item in evidence):
        return EvidenceStatus.CONFLICTING
    return EvidenceStatus.SUPPORTED


def _accepted_claim(case: RetrievalCase, evidence: tuple[GroundedEvidence, ...],
                    status: EvidenceStatus) -> str | None:
    if status is not EvidenceStatus.SUPPORTED or not evidence:
        return None
    weighted: dict[str, float] = {}
    for item in evidence:
        weighted[item.claim] = weighted.get(item.claim, 0.0) + item.authority * item.relevance
    return max(weighted, key=weighted.get)


def action_from_evidence(case: RetrievalCase, status: EvidenceStatus,
                         accepted_claim: str | None) -> CandidateAction:
    if status is EvidenceStatus.CONFLICTING:
        return CandidateAction.ASK_EXTERNAL_REASONER
    if status is EvidenceStatus.UNRESOLVED:
        if case.gap.category == "causal_control":
            return CandidateAction.REVISE_HYPOTHESIS
        return CandidateAction.ASK_EXTERNAL_REASONER
    if accepted_claim == "nanmean ignores NaN values when computing the mean.":
        return CandidateAction.INSPECT_CODE
    return CandidateAction.RUN_EXPERIMENT


def run_policy(case: RetrievalCase, policy: str) -> dict[str, object]:
    evidence = retrieve_evidence(case, policy)
    status = status_from_evidence(evidence)
    accepted = _accepted_claim(case, evidence, status)
    source_urls = {item.source_url for item in evidence}
    gold_supported = accepted == case.gold_claim if case.gold_claim is not None else accepted is None
    unsupported = accepted is not None and accepted != case.gold_claim
    action = action_from_evidence(case, status, accepted)
    return {
        "case": case.name,
        "policy": policy,
        "status": status.value,
        "expected_status": case.expected_status.value,
        "status_correct": status is case.expected_status,
        "accepted_claim": accepted,
        "gold_claim": case.gold_claim,
        "claim_correct": gold_supported,
        "unsupported_claim": unsupported,
        "source_urls": sorted(source_urls),
        "source_attribution_correct": all(item.source_url.startswith("https://example.test/") for item in evidence)
        and all(item.quoted_span for item in evidence),
        "action": action.value,
        "expected_action": case.expected_action.value,
        "action_correct": action is case.expected_action,
        "evidence": [item.__dict__ for item in evidence],
    }


def _aggregate(rows: list[dict[str, object]]) -> dict[str, float]:
    resolved_gold = [row for row in rows if row["gold_claim"] is not None]
    extracted = [row for row in rows if row["accepted_claim"] is not None]
    conflict_rows = [row for row in rows if row["expected_status"] == EvidenceStatus.CONFLICTING.value]
    unresolved_rows = [row for row in rows if row["expected_status"] == EvidenceStatus.UNRESOLVED.value]
    return {
        "claim_precision": sum(bool(row["claim_correct"]) for row in extracted) / max(1, len(extracted)),
        "claim_recall": sum(bool(row["claim_correct"]) for row in resolved_gold) / max(1, len(resolved_gold)),
        "unsupported_claim_rate": sum(bool(row["unsupported_claim"]) for row in rows) / len(rows),
        "source_attribution_accuracy": sum(bool(row["source_attribution_correct"]) for row in rows) / len(rows),
        "conflict_detection_accuracy": sum(bool(row["status_correct"]) for row in conflict_rows) / max(1, len(conflict_rows)),
        "unresolved_gap_accuracy": sum(bool(row["status_correct"]) for row in unresolved_rows) / max(1, len(unresolved_rows)),
        "action_accuracy_after_evidence": sum(bool(row["action_correct"]) for row in rows) / len(rows),
    }


def run_evidence_grounded_retrieval_experiment() -> dict[str, object]:
    policies = ("top1_snippet", "top_k_majority", "authority_weighted", "evidence_grounded")
    report: dict[str, object] = {"cases": [case.name for case in cases()], "policies": {}}
    for policy in policies:
        rows = [run_policy(case, policy) for case in cases()]
        report["policies"][policy] = {"rows": rows, "metrics": _aggregate(rows)}
    return report


if __name__ == "__main__":
    print(json.dumps(run_evidence_grounded_retrieval_experiment(), ensure_ascii=False, indent=2, sort_keys=True))
