"""L5.3: live-Web retrieval disturbance benchmark.

L5.2 verifies the sealed autonomous loop.  L5.3 exposes the retrieval layer to
the kinds of disturbances that a live Web connector can return, while keeping
the test deterministic.  The fixture models timeout, transport failure, missing
sources, stale sources, source conflict, parse failure, insufficient evidence,
and successful evidence extraction.

The benchmark does not depend on live network access.  A real adapter can later
emit the same ``LiveRetrievalStatus`` values, so failures from Web instability
stay separated from failures in the scientific loop.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json

from .evidence_grounded_l50_experiment import (
    EvidenceStatus,
    GroundedEvidence,
    status_from_evidence,
)
from .knowledge_gap_l49_experiment import CandidateAction, KnowledgeGap


class LiveRetrievalStatus(Enum):
    RETRIEVAL_TIMEOUT = "retrieval_timeout"
    HTTP_OR_BROWSER_FAILURE = "http_or_browser_failure"
    SOURCE_NOT_FOUND = "source_not_found"
    SOURCE_STALE = "source_stale"
    SOURCE_CONFLICT = "source_conflict"
    CONTENT_PARSE_FAILURE = "content_parse_failure"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    SUCCESS = "success"


@dataclass(frozen=True)
class LiveDocument:
    title: str
    url: str
    source_type: str
    fetched_at: str
    published_at: str
    authority: float
    relevance: float
    independence_group: str
    claim: str | None
    supports: bool | None
    quoted_span: str | None
    stale: bool = False
    parse_ok: bool = True


@dataclass(frozen=True)
class LiveRetrievalCase:
    name: str
    gap: KnowledgeGap
    primary_status: LiveRetrievalStatus
    fallback_status: LiveRetrievalStatus
    primary_documents: tuple[LiveDocument, ...]
    fallback_documents: tuple[LiveDocument, ...]
    expected_status: EvidenceStatus
    expected_action: CandidateAction
    expected_recovery: bool


def _gap(category: str, question: str) -> KnowledgeGap:
    return KnowledgeGap(question, "L5.3 live-Web disturbance case.", True, 0.28, 0.86, category)


def _doc(identifier: str, category: str, claim: str, *, relevance: float = 0.9, authority: float = 0.86,
         source_type: str = "official_docs", supports: bool | None = True, stale: bool = False,
         parse_ok: bool = True, quoted_span: str | None = None) -> LiveDocument:
    return LiveDocument(
        title=identifier.replace("-", " ").title(),
        url=f"https://live.example.test/{category}/{identifier}",
        source_type=source_type,
        fetched_at="2026-09-14T00:00:00+00:00",
        published_at="2026-09-01T00:00:00+00:00" if not stale else "2020-01-01T00:00:00+00:00",
        authority=authority,
        relevance=relevance,
        independence_group=identifier,
        claim=claim,
        supports=supports,
        quoted_span=quoted_span if quoted_span is not None else claim[:48],
        stale=stale,
        parse_ok=parse_ok,
    )


def cases() -> tuple[LiveRetrievalCase, ...]:
    power = _gap("power_analysis", "How should sample size be revised after low statistical power?")
    api = _gap("python_api", "How does the API handle missing values when computing a mean?")
    measurement = _gap("measurement_method", "Does the estimator require independent samples?")
    domain = _gap("domain_mechanism", "What mechanism explains the observed signal?")
    control = _gap("causal_control", "Which control separates treatment from confound?")
    return (
        LiveRetrievalCase(
            "timeout_then_official_fallback",
            power,
            LiveRetrievalStatus.RETRIEVAL_TIMEOUT,
            LiveRetrievalStatus.SUCCESS,
            (),
            (_doc("pilot-power", "power", "Sample-size planning should use pilot effect size estimates."),),
            EvidenceStatus.SUPPORTED,
            CandidateAction.RUN_EXPERIMENT,
            True,
        ),
        LiveRetrievalCase(
            "browser_failure_unresolved",
            api,
            LiveRetrievalStatus.HTTP_OR_BROWSER_FAILURE,
            LiveRetrievalStatus.HTTP_OR_BROWSER_FAILURE,
            (),
            (),
            EvidenceStatus.UNRESOLVED,
            CandidateAction.ASK_EXTERNAL_REASONER,
            False,
        ),
        LiveRetrievalCase(
            "source_not_found_unresolved",
            domain,
            LiveRetrievalStatus.SOURCE_NOT_FOUND,
            LiveRetrievalStatus.SOURCE_NOT_FOUND,
            (),
            (),
            EvidenceStatus.UNRESOLVED,
            CandidateAction.ASK_EXTERNAL_REASONER,
            False,
        ),
        LiveRetrievalCase(
            "stale_source_rejected",
            power,
            LiveRetrievalStatus.SOURCE_STALE,
            LiveRetrievalStatus.INSUFFICIENT_EVIDENCE,
            (_doc("old-rule", "power", "Ten samples are enough for this task.", stale=True),),
            (),
            EvidenceStatus.UNRESOLVED,
            CandidateAction.ASK_EXTERNAL_REASONER,
            False,
        ),
        LiveRetrievalCase(
            "conflicting_live_sources",
            measurement,
            LiveRetrievalStatus.SOURCE_CONFLICT,
            LiveRetrievalStatus.SOURCE_CONFLICT,
            (
                _doc("estimator-guide", "measurement", "The estimator requires independent samples.", supports=True),
                _doc("correlated-paper", "measurement", "The estimator does not require independent samples.",
                     source_type="paper", supports=False),
            ),
            (),
            EvidenceStatus.CONFLICTING,
            CandidateAction.ASK_EXTERNAL_REASONER,
            False,
        ),
        LiveRetrievalCase(
            "parse_failure_recovers_with_fallback",
            api,
            LiveRetrievalStatus.CONTENT_PARSE_FAILURE,
            LiveRetrievalStatus.SUCCESS,
            (_doc("script-heavy-doc", "api", "nanmean ignores NaN values when computing the mean.",
                  parse_ok=False, quoted_span=None),),
            (_doc("api-reference", "api", "nanmean ignores NaN values when computing the mean.", authority=0.95),),
            EvidenceStatus.SUPPORTED,
            CandidateAction.INSPECT_CODE,
            True,
        ),
        LiveRetrievalCase(
            "insufficient_evidence_unresolved",
            control,
            LiveRetrievalStatus.INSUFFICIENT_EVIDENCE,
            LiveRetrievalStatus.INSUFFICIENT_EVIDENCE,
            (_doc("broad-control-note", "control", "Randomization is often useful.",
                  relevance=0.32, authority=0.82),),
            (),
            EvidenceStatus.UNRESOLVED,
            CandidateAction.REVISE_HYPOTHESIS,
            False,
        ),
        LiveRetrievalCase(
            "live_success",
            control,
            LiveRetrievalStatus.SUCCESS,
            LiveRetrievalStatus.SUCCESS,
            (_doc("shuffled-control", "control", "A shuffled negative control separates treatment from shared drift."),),
            (),
            EvidenceStatus.SUPPORTED,
            CandidateAction.REVISE_HYPOTHESIS,
            False,
        ),
    )


def _ground(document: LiveDocument, gap: KnowledgeGap) -> GroundedEvidence | None:
    if document.stale or not document.parse_ok or document.claim is None or document.quoted_span is None:
        return None
    if document.relevance < 0.65 or document.authority < 0.6:
        return None
    return GroundedEvidence(
        claim=document.claim,
        source_url=document.url,
        source_type=document.source_type,
        retrieved_at=document.fetched_at,
        supports=document.supports,
        relevance=document.relevance,
        authority=document.authority,
        independence_group=document.independence_group,
        uncertainty=1.0 - min(document.relevance, document.authority),
        quoted_span=document.quoted_span,
    )


def _status_from_live(status: LiveRetrievalStatus, evidence: tuple[GroundedEvidence, ...]) -> EvidenceStatus:
    if status is LiveRetrievalStatus.SOURCE_CONFLICT:
        return EvidenceStatus.CONFLICTING if evidence else EvidenceStatus.UNRESOLVED
    if status is LiveRetrievalStatus.SUCCESS and evidence:
        return status_from_evidence(evidence)
    return EvidenceStatus.UNRESOLVED


def _action_from_status(gap: KnowledgeGap, status: EvidenceStatus,
                        evidence: tuple[GroundedEvidence, ...]) -> CandidateAction:
    if status is EvidenceStatus.CONFLICTING:
        return CandidateAction.ASK_EXTERNAL_REASONER
    if status is EvidenceStatus.UNRESOLVED:
        if gap.category == "causal_control":
            return CandidateAction.REVISE_HYPOTHESIS
        return CandidateAction.ASK_EXTERNAL_REASONER
    if any("nanmean" in item.claim for item in evidence):
        return CandidateAction.INSPECT_CODE
    if gap.category == "causal_control":
        return CandidateAction.REVISE_HYPOTHESIS
    return CandidateAction.RUN_EXPERIMENT


def _live_attempt(case: LiveRetrievalCase, *, fallback: bool = False,
                  grounded: bool = True) -> tuple[LiveRetrievalStatus, tuple[GroundedEvidence, ...]]:
    status = case.fallback_status if fallback else case.primary_status
    documents = case.fallback_documents if fallback else case.primary_documents
    if not grounded:
        evidence = tuple(
            GroundedEvidence(
                document.claim or "",
                document.url,
                document.source_type,
                document.fetched_at,
                document.supports,
                document.relevance,
                document.authority,
                document.independence_group,
                1.0 - min(document.relevance, document.authority),
                document.quoted_span,
            )
            for document in documents if document.claim is not None
        )
    else:
        evidence = tuple(item for document in documents for item in (_ground(document, case.gap),) if item is not None)
    return status, evidence


def retrieve_live(case: LiveRetrievalCase, policy: str) -> dict[str, object]:
    if policy == "sealed_grounded":
        # The sealed control uses the same case documents, but bypasses live
        # transport failures so the policy comparison isolates grounding.
        documents = case.primary_documents or case.fallback_documents
        evidence = tuple(item for document in documents for item in (_ground(document, case.gap),) if item is not None)
        status = _status_from_live(LiveRetrievalStatus.SUCCESS, evidence)
        failure_status = LiveRetrievalStatus.SUCCESS if evidence else case.primary_status
        recovered = False
    elif policy == "live_top1":
        failure_status, evidence = _live_attempt(case, grounded=False)
        evidence = evidence[:1]
        status = _status_from_live(failure_status, evidence)
        recovered = False
    elif policy == "live_search_only":
        failure_status, evidence = _live_attempt(case, grounded=False)
        status = EvidenceStatus.SUPPORTED if evidence else EvidenceStatus.UNRESOLVED
        recovered = False
    elif policy == "live_grounded":
        failure_status, evidence = _live_attempt(case, grounded=True)
        status = _status_from_live(failure_status, evidence)
        recovered = False
    elif policy == "live_grounded_with_fallback":
        failure_status, evidence = _live_attempt(case, grounded=True)
        status = _status_from_live(failure_status, evidence)
        recovered = False
        if status is EvidenceStatus.UNRESOLVED and case.fallback_status is LiveRetrievalStatus.SUCCESS:
            fallback_status, fallback_evidence = _live_attempt(case, fallback=True, grounded=True)
            if fallback_evidence:
                failure_status = fallback_status
                evidence = fallback_evidence
                status = _status_from_live(fallback_status, fallback_evidence)
                recovered = True
    else:
        raise ValueError(f"unknown policy: {policy}")

    unsupported = status is EvidenceStatus.SUPPORTED and status is not case.expected_status
    action = _action_from_status(case.gap, status, evidence)
    return {
        "case": case.name,
        "policy": policy,
        "live_status": failure_status.value,
        "status": status.value,
        "expected_status": case.expected_status.value,
        "status_correct": status is case.expected_status,
        "action": action.value,
        "expected_action": case.expected_action.value,
        "action_correct": action is case.expected_action,
        "gap_resolved": status is not EvidenceStatus.UNRESOLVED,
        "supported_claim": status is EvidenceStatus.SUPPORTED,
        "unsupported_claim": unsupported,
        "source_attribution_correct": all(item.source_url.startswith("https://") and item.quoted_span for item in evidence),
        "conflict_detected": status is EvidenceStatus.CONFLICTING,
        "stale_rejected": case.primary_status is not LiveRetrievalStatus.SOURCE_STALE
        or status is EvidenceStatus.UNRESOLVED,
        "failure_recovered": recovered,
        "evidence": [item.__dict__ for item in evidence],
    }


def _aggregate(rows: list[dict[str, object]]) -> dict[str, float]:
    stale = [row for row in rows if row["case"] == "stale_source_rejected"]
    recoverable = [row for row in rows if row["case"] in {"timeout_then_official_fallback", "parse_failure_recovers_with_fallback"}]
    return {
        "gap_resolution_rate": sum(bool(row["gap_resolved"]) for row in rows) / len(rows),
        "supported_claim_rate": sum(bool(row["supported_claim"]) for row in rows) / len(rows),
        "unsupported_claim_rate": sum(bool(row["unsupported_claim"]) for row in rows) / len(rows),
        "source_attribution_accuracy": sum(bool(row["source_attribution_correct"]) for row in rows) / len(rows),
        "conflict_detection_rate": sum(bool(row["conflict_detected"]) for row in rows if row["expected_status"] == EvidenceStatus.CONFLICTING.value)
        / max(1, sum(row["expected_status"] == EvidenceStatus.CONFLICTING.value for row in rows)),
        "stale_source_rejection_rate": sum(bool(row["stale_rejected"]) for row in stale) / max(1, len(stale)),
        "retrieval_failure_recovery_rate": sum(bool(row["failure_recovered"]) for row in recoverable) / max(1, len(recoverable)),
        "post_retrieval_action_accuracy": sum(bool(row["action_correct"]) for row in rows) / len(rows),
    }


def run_live_web_retrieval_experiment() -> dict[str, object]:
    policies = ("sealed_grounded", "live_top1", "live_search_only", "live_grounded", "live_grounded_with_fallback")
    report: dict[str, object] = {"cases": [case.name for case in cases()], "policies": {}}
    for policy in policies:
        rows = [retrieve_live(case, policy) for case in cases()]
        report["policies"][policy] = {"rows": rows, "metrics": _aggregate(rows)}
    return report


if __name__ == "__main__":
    print(json.dumps(run_live_web_retrieval_experiment(), ensure_ascii=False, indent=2, sort_keys=True))
