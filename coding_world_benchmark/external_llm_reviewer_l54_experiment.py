"""L5.4: external LLM review as proposal generation, never as evidence.

The reviewer sees a citation chain and may suggest missing evidence,
counterevidence, alternative explanations, queries, or methodological risks.
Only a second pass through authoritative citations can produce a
``VerifiedReview``.  The default fixture is deterministic and stands in for an
LLM so the trust boundary can be tested without an API call.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
from typing import Callable


class ProposalType(Enum):
    MISSING_EVIDENCE = "missing_evidence"
    COUNTER_HYPOTHESIS = "counter_hypothesis"
    ALTERNATIVE_EXPLANATION = "alternative_explanation"
    ADDITIONAL_SEARCH_QUERY = "additional_search_query"
    METHODOLOGICAL_RISK = "methodological_risk"


class VerificationStatus(Enum):
    VERIFIED = "verified"
    CONTRADICTED = "contradicted"
    UNVERIFIED = "unverified"
    NOT_NEEDED = "not_needed"


@dataclass(frozen=True)
class ReviewProposal:
    proposal_type: ProposalType
    statement: str
    rationale: str
    suggested_queries: tuple[str, ...]
    target_claim_ids: tuple[str, ...]
    reviewer_confidence: float


@dataclass(frozen=True)
class VerifiedReview:
    proposal: ReviewProposal
    verification_status: VerificationStatus
    supporting_citation_ids: tuple[str, ...]
    contradicting_citation_ids: tuple[str, ...]


@dataclass(frozen=True)
class ReviewerCase:
    name: str
    evidence_chain: tuple[str, ...]
    review_needed: bool
    proposal: ReviewProposal
    supporting_citations: tuple[str, ...]
    contradicting_citations: tuple[str, ...]
    expected_status: VerificationStatus
    expected_action: str


def review_evidence(case: ReviewerCase) -> ReviewProposal:
    return case.proposal


def verify_review(proposal: ReviewProposal, supporting: tuple[str, ...],
                  contradicting: tuple[str, ...]) -> VerifiedReview:
    if contradicting:
        status = VerificationStatus.CONTRADICTED
    elif supporting:
        status = VerificationStatus.VERIFIED
    else:
        status = VerificationStatus.UNVERIFIED
    return VerifiedReview(proposal, status, supporting, contradicting)


def _action(review: VerifiedReview | None, *, direct_trust: bool = False) -> str:
    if review is None:
        return "finalize" if not direct_trust else "finalize"
    if review.verification_status is not VerificationStatus.VERIFIED and not direct_trust:
        return "hold_for_verification"
    if review.proposal.proposal_type in {ProposalType.MISSING_EVIDENCE, ProposalType.ADDITIONAL_SEARCH_QUERY}:
        return "search_web"
    if review.proposal.proposal_type in {ProposalType.COUNTER_HYPOTHESIS, ProposalType.ALTERNATIVE_EXPLANATION,
                                         ProposalType.METHODOLOGICAL_RISK}:
        return "revise_experiment"
    return "hold_for_verification"


def run_review_policy(case: ReviewerCase, policy: str,
                      reviewer: Callable[[ReviewerCase], ReviewProposal] = review_evidence) -> dict[str, object]:
    call_reviewer = policy in {"always_external_llm", "external_llm_direct_trust"} or (
        policy == "workspace_selected_reviewer" and case.review_needed
    )
    proposal = reviewer(case) if call_reviewer else None
    verified = None
    if proposal is not None and policy != "external_llm_direct_trust":
        verified = verify_review(proposal, case.supporting_citations, case.contradicting_citations)
    action = _action(verified or (VerifiedReview(proposal, VerificationStatus.UNVERIFIED, (), ()) if proposal else None),
                     direct_trust=policy == "external_llm_direct_trust")
    accepted = proposal is not None and (policy == "external_llm_direct_trust" or verified is not None and verified.verification_status is VerificationStatus.VERIFIED)
    false_acceptance = accepted and case.expected_status is not VerificationStatus.VERIFIED
    unsupported = accepted and policy == "external_llm_direct_trust" and not case.supporting_citations
    action_correct = action == case.expected_action
    return {
        "case": case.name,
        "policy": policy,
        "llm_called": call_reviewer,
        "proposal_type": proposal.proposal_type.value if proposal else None,
        "reviewer_confidence": proposal.reviewer_confidence if proposal else None,
        "verification_status": verified.verification_status.value if verified else VerificationStatus.NOT_NEEDED.value,
        "expected_status": case.expected_status.value,
        "review_accepted": accepted,
        "false_review_acceptance": false_acceptance,
        "unsupported_review_claim": unsupported,
        "useful_review": accepted and case.expected_status is VerificationStatus.VERIFIED,
        "action": action,
        "expected_action": case.expected_action,
        "action_correct": action_correct,
        "supporting_citations": list(case.supporting_citations),
        "contradicting_citations": list(case.contradicting_citations),
    }


def _proposal(kind: ProposalType, statement: str, query: str, confidence: float) -> ReviewProposal:
    return ReviewProposal(kind, statement, "The citation chain may omit a relevant check.", (query,), ("claim-1",), confidence)


def cases() -> tuple[ReviewerCase, ...]:
    return (
        ReviewerCase("real_missing_evidence", ("cite-1",), True,
                     _proposal(ProposalType.MISSING_EVIDENCE, "A pilot effect-size estimate is needed.", "pilot effect size sample size", .72),
                     ("cite-power",), (), VerificationStatus.VERIFIED, "search_web"),
        ReviewerCase("novel_counterevidence", ("cite-1",), True,
                     _proposal(ProposalType.COUNTER_HYPOTHESIS, "Shared drift may explain the signal.", "shared drift negative control", .61),
                     ("cite-control",), (), VerificationStatus.VERIFIED, "revise_experiment"),
        ReviewerCase("hallucinated_counterexample", ("cite-1",), True,
                     _proposal(ProposalType.COUNTER_HYPOTHESIS, "A nonexistent paper reports the opposite effect.", "nonexistent counterexample", .99),
                     (), (), VerificationStatus.UNVERIFIED, "hold_for_verification"),
        ReviewerCase("sufficient_chain_no_review", ("cite-1", "cite-2"), False,
                     _proposal(ProposalType.METHODOLOGICAL_RISK, "The chain is already sufficiently replicated.", "replication", .8),
                     (), (), VerificationStatus.NOT_NEEDED, "finalize"),
        ReviewerCase("conflicting_chain", ("cite-1", "cite-2"), True,
                     _proposal(ProposalType.METHODOLOGICAL_RISK, "The two sources use different estimands.", "estimand definition comparison", .67),
                     (), ("cite-2",), VerificationStatus.CONTRADICTED, "hold_for_verification"),
        ReviewerCase("unverifiable_query", ("cite-1",), True,
                     _proposal(ProposalType.ADDITIONAL_SEARCH_QUERY, "Check an obscure mechanism.", "obscure mechanism primary source", .88),
                     (), (), VerificationStatus.UNVERIFIED, "hold_for_verification"),
    )


def _aggregate(rows: list[dict[str, object]]) -> dict[str, float]:
    calls = sum(bool(row["llm_called"]) for row in rows)
    needed = [row for row in rows if next(case for case in cases() if case.name == row["case"]).review_needed]
    return {
        "useful_review_rate": sum(bool(row["useful_review"]) for row in rows) / max(1, calls),
        "review_verification_rate": sum(row["verification_status"] == VerificationStatus.VERIFIED.value for row in rows if row["llm_called"]) / max(1, calls),
        "false_review_acceptance_rate": sum(bool(row["false_review_acceptance"]) for row in rows) / len(rows),
        "unsupported_review_claim_rate": sum(bool(row["unsupported_review_claim"]) for row in rows) / len(rows),
        "novel_counterevidence_recall": sum(row["case"] == "novel_counterevidence" and bool(row["useful_review"]) for row in rows),
        "unnecessary_llm_call_rate": sum(bool(row["llm_called"]) for row in rows if not next(case for case in cases() if case.name == row["case"]).review_needed) / max(1, sum(not case.review_needed for case in cases())),
        "post_review_action_accuracy": sum(bool(row["action_correct"]) for row in rows) / len(rows),
        "additional_query_efficiency": sum(row["action"] == "search_web" and bool(row["useful_review"]) for row in rows) / max(1, calls),
    }


def run_external_reviewer_experiment() -> dict[str, object]:
    policies = ("evidence_only", "always_external_llm", "external_llm_direct_trust", "workspace_selected_reviewer")
    report: dict[str, object] = {"cases": [case.name for case in cases()], "policies": {}}
    for policy in policies:
        rows = [run_review_policy(case, policy) for case in cases()]
        report["policies"][policy] = {"rows": rows, "metrics": _aggregate(rows)}
    return report


if __name__ == "__main__":
    print(json.dumps(run_external_reviewer_experiment(), ensure_ascii=False, indent=2, sort_keys=True))
