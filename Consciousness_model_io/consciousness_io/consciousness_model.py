"""Functional conscious-access control built on a finite compressed workspace.

This module models functional control only.  Its private integrated state is a
capacity-limited data structure whose contents can be broadcast to independent
consumers; it makes no claim about subjective experience.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, log2
from typing import Dict, Iterable, Mapping, Sequence, Tuple

from .compressed_workspace import CompressedWorkspace, Relation


@dataclass(frozen=True)
class IntegratedState:
    """The current private workspace state q, valid for one control cycle."""

    cycle: int
    concepts: Tuple[str, ...]
    confidence: float
    uncertainty: float
    goal: Tuple[str, ...]


@dataclass(frozen=True)
class ConceptualAction:
    """An action policy, deliberately separated from its final rendering."""

    kind: str
    proposition: Tuple[Relation, ...]
    confidence: float
    risk: float


@dataclass(frozen=True)
class SelfObservation:
    cycle: int
    predicted_confidence: float
    succeeded: bool
    prediction_error: float


class ConsciousnessController:
    """Select, integrate, broadcast, and reflect over a finite state q.

    Long-term concepts remain private to the workspace.  Consumers can receive
    only the most recently integrated state through :meth:`broadcast`.
    """

    def __init__(self, capacity: int = 4, ignition_threshold: float = 0.15) -> None:
        if capacity < 1:
            raise ValueError("capacity must be positive")
        if not 0.0 < ignition_threshold <= 1.0:
            raise ValueError("ignition_threshold must be in (0, 1]")
        self.workspace = CompressedWorkspace(capacity=capacity)
        self.ignition_threshold = ignition_threshold
        self.q: IntegratedState | None = None
        self.risk = 0.5
        self._cycle = 0
        self._observations: list[SelfObservation] = []

    def perceive(self, observations: Iterable[Relation]) -> None:
        """Store observations without making them globally available yet."""
        self.workspace.ingest(observations)

    def integrate(self, goal: Sequence[str], context: Sequence[str] = ()) -> IntegratedState:
        """Create q through relevance competition within the fixed capacity."""
        terms = tuple(goal) + tuple(context)
        wanted = set(terms)
        candidates = [
            concept for concept in self.workspace.long_term.values()
            if self.workspace._relevance(concept, wanted) > 0
        ]
        activations = [self.workspace._relevance(concept, wanted) + 0.15 * concept.utility() for concept in candidates]
        if activations:
            shift = max(activations)
            weights = [exp(value - shift) for value in activations]
            normalizer = sum(weights)
            distribution = tuple((concept, weight / normalizer) for concept, weight in zip(candidates, weights))
        else:
            distribution = ()
        winners = tuple(
            concept
            for concept, probability in sorted(distribution, key=lambda row: row[1], reverse=True)[:self.workspace.capacity]
            if probability >= self.ignition_threshold
        )
        if not winners and distribution:
            winners = (max(distribution, key=lambda row: row[1])[0],)
        confidence = sum(concept.confidence for concept in winners) / len(winners) if winners else 0.0
        uncertainty = self._entropy(distribution)
        self._cycle += 1
        self.q = IntegratedState(
            cycle=self._cycle,
            concepts=tuple(concept.identifier for concept in winners),
            confidence=confidence,
            uncertainty=uncertainty,
            goal=tuple(goal),
        )
        return self.q

    def broadcast(self, module: str) -> IntegratedState:
        """Expose the same current q to a named functional consumer."""
        if module not in {"planner", "predictor", "memory", "reasoner", "report"}:
            raise ValueError("unknown module")
        if self.q is None:
            raise RuntimeError("integrate before broadcast")
        return self.q

    def select_action(self) -> ConceptualAction:
        """Choose an evidence-backed policy from q, accounting for risk."""
        q = self.broadcast("planner")
        relations = tuple(
            relation
            for identifier in q.concepts
            for relation in self.workspace.long_term[identifier].relations
        )
        adjusted_confidence = max(0.0, q.confidence - self.risk * q.uncertainty)
        if not relations or adjusted_confidence < 0.25:
            return ConceptualAction("REQUEST_EVIDENCE", (), adjusted_confidence, self.risk)
        return ConceptualAction("ACT", relations, adjusted_confidence, self.risk)

    def report(self) -> Mapping[str, float | str]:
        """Give a calibrated availability report from the broadcast state."""
        q = self.broadcast("report")
        confidence = max(0.0, q.confidence - self.risk * q.uncertainty)
        status = "known" if confidence >= 0.70 else "need_more_evidence" if confidence > 0.0 else "unavailable"
        return {"status": status, "confidence": confidence, "uncertainty": q.uncertainty}

    def observe_outcome(self, action: ConceptualAction, succeeded: bool) -> SelfObservation:
        """Record a self-observation; call reflect to update persistent control."""
        observation = SelfObservation(
            cycle=self.broadcast("memory").cycle,
            predicted_confidence=action.confidence,
            succeeded=succeeded,
            prediction_error=abs(action.confidence - float(succeeded)),
        )
        self._observations.append(observation)
        return observation

    def reflect(self, window: int = 8) -> float:
        """Update risk from recent calibration error, preserving bounded control."""
        if window < 1:
            raise ValueError("window must be positive")
        recent = self._observations[-window:]
        if recent:
            mean_error = sum(item.prediction_error for item in recent) / len(recent)
            self.risk = min(1.0, max(0.0, 0.75 * self.risk + 0.5 * mean_error))
        return self.risk

    @staticmethod
    def _entropy(distribution: Sequence[Tuple[object, float]]) -> float:
        if len(distribution) <= 1:
            return 0.0
        entropy = -sum(probability * log2(probability) for _, probability in distribution if probability > 0.0)
        return entropy / log2(len(distribution))