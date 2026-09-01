"""Find hypotheses and questions without recorded experimental support."""

from __future__ import annotations

from .models import AtomicClaim, Concept, ResearchGap


def detect_gaps(claims: list[AtomicClaim], concepts: list[Concept]) -> list[ResearchGap]:
    claim_to_concept = {claim_id: concept.id for concept in concepts for claim_id in concept.member_claim_ids}
    experiments = [claim.text.lower() for claim in claims if claim.claim_type in {"experiment", "result"}]
    targets = [claim for claim in claims if claim.claim_type in {"hypothesis", "question"}]
    gaps: list[ResearchGap] = []
    for index, claim in enumerate(targets, start=1):
        words = [word for word in claim.text.lower().split() if len(word) > 3]
        is_tested = any(any(word in experiment for word in words) for experiment in experiments)
        if not is_tested:
            gaps.append(ResearchGap(
                id=f"G{index:02d}", description=f"未検証: {claim.text}", target_hypothesis=claim.text,
                concept_id=claim_to_concept.get(claim.id), uncertainty=1.0 - claim.confidence,
                importance=0.8, information_gain=0.9, novelty=0.6,
                dependency_impact=0.7, estimated_cost=0.4,
            ))
    return gaps