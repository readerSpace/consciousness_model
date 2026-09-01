"""Compress related atomic claims into reusable concepts."""

from __future__ import annotations

from collections import defaultdict

from .concept_matcher import matching_pairs, tokens
from .models import AtomicClaim, Concept


def compress_claims(claims: list[AtomicClaim], threshold: float = 0.3) -> list[Concept]:
    parents = {claim.id: claim.id for claim in claims}

    def root(identifier: str) -> str:
        while parents[identifier] != identifier:
            parents[identifier] = parents[parents[identifier]]
            identifier = parents[identifier]
        return identifier

    for first, second in matching_pairs(claims, threshold):
        first_root, second_root = root(first), root(second)
        if first_root != second_root:
            parents[second_root] = first_root

    groups: dict[str, list[AtomicClaim]] = defaultdict(list)
    for claim in claims:
        groups[root(claim.id)].append(claim)

    concepts: list[Concept] = []
    for index, members in enumerate(groups.values(), start=1):
        terms = sorted(set().union(*(tokens(member.text) for member in members)), key=lambda term: (-sum(term in member.text.lower() for member in members), term))
        name = " / ".join(terms[:3]) or members[0].text[:30]
        predictions = [member.text for member in members if member.claim_type == "hypothesis"]
        questions = [member.text for member in members if member.claim_type == "question"]
        concepts.append(Concept(
            id=f"C{index:02d}", name=name, description=members[0].text,
            member_claim_ids=[member.id for member in members], abstraction_level=1,
            confidence=sum(member.confidence for member in members) / len(members),
            support_count=len(members), predictions=predictions, unresolved_questions=questions,
        ))
    return concepts