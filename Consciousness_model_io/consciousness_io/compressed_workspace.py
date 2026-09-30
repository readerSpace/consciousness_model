"""Finite, self-editing compressed workspace for functional experiments.

The module keeps all observed concepts in a long-term store, but exposes only
the highest-utility, non-redundant concepts through a capacity-limited query
layer.  It is deliberately symbolic and dependency-free: this is a test of
the computational hypothesis, not a claim that symbols alone are conscious.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import exp, log2
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


Relation = Tuple[str, str, str]


@dataclass(frozen=True)
class Concept:
    identifier: str
    kind: str
    representation: str
    relations: Tuple[Relation, ...]
    prediction: float
    confidence: float
    reuse_count: int
    provenance: Tuple[str, ...]
    support: int
    cost: float = 1.0

    def utility(self, alpha: float = 1.2, beta: float = 0.7, gamma: float = 0.8, penalty: float = 0.35) -> float:
        compression_gain = log2(self.support + 1.0) * (1.3 if self.kind != "episode" else 1.0)
        transfer = 1.0 if self.kind in {"abstract", "compose", "chunk"} else 0.0
        return compression_gain + alpha * self.prediction + beta * log2(self.reuse_count + 1.0) + gamma * transfer - penalty * self.cost


class CompressedWorkspace:
    """A K-concept workspace with long-term storage and query-based broadcast."""

    def __init__(self, capacity: int | None = 16) -> None:
        if capacity is not None and capacity < 1:
            raise ValueError("capacity must be positive or None for unbounded")
        self.capacity = capacity
        self.long_term: Dict[str, Concept] = {}
        self.items: Tuple[str, ...] = ()

    def ingest(self, observations: Iterable[Relation]) -> None:
        episodes = self._episodes(observations)
        candidates = episodes + self._merge(episodes) + self._abstract(episodes) + self._chunk(episodes) + self._compose(episodes)
        for concept in candidates:
            previous = self.long_term.get(concept.identifier)
            if previous is None:
                self.long_term[concept.identifier] = concept
            else:
                self.long_term[concept.identifier] = replace(
                    previous,
                    support=previous.support + concept.support,
                    confidence=max(previous.confidence, concept.confidence),
                    provenance=tuple(sorted(set(previous.provenance + concept.provenance))),
                )
        self._select()

    def query(self, module: str, terms: Sequence[str], limit: int = 1) -> Tuple[Concept, ...]:
        """Retrieve concepts from the finite workspace, not from the full DB.

        Different modules issue different terms.  A returned concept gains reuse
        value, changing the next self-editing selection step.
        """
        if module not in {"planner", "predictor", "memory", "reasoner", "report"}:
            raise ValueError("unknown module")
        wanted = set(terms)
        ranked = sorted(
            (self.long_term[key] for key in self.items),
            key=lambda concept: (self._relevance(concept, wanted), concept.utility(), concept.identifier),
            reverse=True,
        )
        found = tuple(concept for concept in ranked if self._relevance(concept, wanted) > 0)[:limit]
        for concept in found:
            self.long_term[concept.identifier] = replace(concept, reuse_count=concept.reuse_count + 1)
        if found:
            self._select()
        return found

    def attention_distribution(self, terms: Sequence[str], focus: float = 1.0) -> Tuple[Tuple[Concept, float], ...]:
        """Return the finite-attention distribution for one ambiguous query.

        This is not a random-error shortcut.  Each matching concept consumes a
        share of one normalised retrieval budget; adding similarly relevant
        concepts therefore reduces the probability of retrieving any one item.
        ``focus`` controls how strongly query-feature overlap is weighted.
        """
        if focus <= 0:
            raise ValueError("focus must be positive")
        wanted = set(terms)
        candidates = [self.long_term[key] for key in self.items if self._relevance(self.long_term[key], wanted) > 0]
        if not candidates:
            return ()
        activations = [focus * self._relevance(concept, wanted) + 0.15 * concept.utility() for concept in candidates]
        shift = max(activations)
        weights = [exp(value - shift) for value in activations]
        normalizer = sum(weights)
        return tuple((concept, weight / normalizer) for concept, weight in zip(candidates, weights))

    def _select(self) -> None:
        pool = list(self.long_term.values())
        selected: List[Concept] = []
        max_items = len(pool) if self.capacity is None else self.capacity
        while pool and len(selected) < max_items:
            best = max(pool, key=lambda candidate: candidate.utility() - self._redundancy(candidate, selected))
            selected.append(best)
            pool.remove(best)
        self.items = tuple(concept.identifier for concept in selected)

    @staticmethod
    def _relevance(concept: Concept, terms: set[str]) -> int:
        return sum(token in terms for relation in concept.relations for token in relation) + sum(token in concept.representation for token in terms)

    @staticmethod
    def _redundancy(candidate: Concept, selected: Sequence[Concept]) -> float:
        candidate_terms = set(token for relation in candidate.relations for token in relation)
        if not candidate_terms:
            return 0.0
        return max(
            (len(candidate_terms & {token for relation in other.relations for token in relation}) / len(candidate_terms | {token for relation in other.relations for token in relation}))
            for other in selected
        ) if selected else 0.0

    @staticmethod
    def _episodes(observations: Iterable[Relation]) -> List[Concept]:
        result = []
        for index, relation in enumerate(observations):
            source, verb, target = relation
            result.append(Concept(
                f"episode:{source}:{verb}:{target}", "episode", f"{source} {verb} {target}", (relation,),
                prediction=0.75, confidence=0.9, reuse_count=0, provenance=(f"obs:{index}",), support=1,
            ))
        return result

    @staticmethod
    def _merge(concepts: Sequence[Concept]) -> List[Concept]:
        # Equivalent observations become one higher-support candidate.
        grouped: Dict[Relation, List[Concept]] = {}
        for concept in concepts:
            grouped.setdefault(concept.relations[0], []).append(concept)
        return [Concept(
            f"merge:{':'.join(relation)}", "merge", " / ".join(item.representation for item in group), (relation,),
            prediction=0.8, confidence=max(item.confidence for item in group), reuse_count=0,
            provenance=tuple(source for item in group for source in item.provenance), support=len(group),
        ) for relation, group in grouped.items() if len(group) > 1]

    @staticmethod
    def _abstract(concepts: Sequence[Concept]) -> List[Concept]:
        # Sources such as red:apple and green:apple yield *:apple -> target.
        grouped: Dict[Tuple[str, str, str], List[Concept]] = {}
        for concept in concepts:
            source, verb, target = concept.relations[0]
            if ":" in source:
                _, family = source.split(":", 1)
                grouped.setdefault((family, verb, target), []).append(concept)
        return [Concept(
            f"abstract:{family}:{verb}:{target}", "abstract", f"*: {family} {verb} {target}", ((f"*:{family}", verb, target),),
            prediction=0.85, confidence=sum(item.confidence for item in group) / len(group), reuse_count=0,
            provenance=tuple(source for item in group for source in item.provenance), support=len(group),
        ) for (family, verb, target), group in grouped.items() if len(group) >= 2]

    @staticmethod
    def _chunk(concepts: Sequence[Concept]) -> List[Concept]:
        # A two-step path is also a reusable sequential chunk.
        chunks = []
        for first in concepts:
            a, verb_a, middle = first.relations[0]
            for second in concepts:
                source_b, verb_b, target = second.relations[0]
                if middle == source_b:
                    chunks.append(Concept(
                        f"chunk:{a}:{verb_a}:{middle}:{target}", "chunk", f"{a}→{middle}→{target}", ((a, verb_a, middle), (middle, verb_b, target)),
                        prediction=0.9, confidence=first.confidence * second.confidence, reuse_count=0,
                        provenance=first.provenance + second.provenance, support=2, cost=1.0,
                    ))
        return chunks

    @staticmethod
    def _compose(concepts: Sequence[Concept]) -> List[Concept]:
        composed = []
        for first in concepts:
            source, verb, middle = first.relations[0]
            for second in concepts:
                other_source, other_verb, target = second.relations[0]
                if middle == other_source and verb == other_verb:
                    relation = (source, verb, target)
                    composed.append(Concept(
                        f"compose:{source}:{verb}:{target}", "compose", f"{source} {verb} {target}", (relation,),
                        prediction=0.92, confidence=first.confidence * second.confidence, reuse_count=0,
                        provenance=first.provenance + second.provenance, support=2,
                    ))
        return composed
