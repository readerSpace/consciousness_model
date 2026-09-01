"""Minimal serializable research graph and state classifier."""

from __future__ import annotations

from dataclasses import dataclass, field

from .models import AtomicClaim, Concept, ResearchState


@dataclass
class ResearchGraph:
    nodes: dict[str, str] = field(default_factory=dict)
    edges: list[tuple[str, str, str]] = field(default_factory=list)

    def add(self, identifier: str, node_type: str) -> None:
        self.nodes[identifier] = node_type

    def link(self, source: str, relation: str, target: str) -> None:
        self.edges.append((source, relation, target))


def build_graph(claims: list[AtomicClaim], concepts: list[Concept]) -> ResearchGraph:
    graph = ResearchGraph()
    for claim in claims:
        graph.add(claim.id, claim.claim_type.upper())
    for concept in concepts:
        graph.add(concept.id, "CONCEPT")
        for claim_id in concept.member_claim_ids:
            graph.link(claim_id, "DERIVED_FROM", concept.id)
    return graph


def classify_state(claims: list[AtomicClaim]) -> ResearchState:
    state = ResearchState()
    for claim in claims:
        if claim.claim_type == "result":
            state.established.append(claim.text)
        elif claim.claim_type == "hypothesis":
            state.hypotheses.append(claim.text)
        elif claim.claim_type == "question":
            state.unresolved.append(claim.text)
    result_terms = " ".join(state.established).lower()
    for hypothesis in state.hypotheses:
        if any(term in result_terms for term in hypothesis.lower().split() if len(term) > 3):
            state.contradictions.append(hypothesis)
    return state