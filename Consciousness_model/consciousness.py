"""Portable computational primitives for the functional consciousness model.

The module contains functional mechanisms only. It does not claim to model
subjective experience.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import log2
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class WorkspaceItem:
    """A candidate that may be admitted to the finite workspace."""

    identifier: str
    value: float
    payload: object = None


class FiniteWorkspace:
    """Capacity-limited broadcast buffer with a separate candidate store."""

    def __init__(self, capacity: int = 8) -> None:
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._candidates: dict[str, WorkspaceItem] = {}
        self._items: tuple[str, ...] = ()

    @property
    def candidates(self) -> tuple[WorkspaceItem, ...]:
        return tuple(self._candidates.values())

    @property
    def items(self) -> tuple[WorkspaceItem, ...]:
        return tuple(self._candidates[identifier] for identifier in self._items)

    def ingest(self, candidates: Iterable[WorkspaceItem]) -> tuple[WorkspaceItem, ...]:
        for candidate in candidates:
            previous = self._candidates.get(candidate.identifier)
            if previous is None or candidate.value >= previous.value:
                self._candidates[candidate.identifier] = candidate
        self._items = tuple(
            candidate.identifier
            for candidate in sorted(self._candidates.values(), key=lambda item: (-item.value, item.identifier))[: self.capacity]
        )
        return self.items

    def broadcast(self) -> tuple[WorkspaceItem, ...]:
        return self.items


@dataclass(frozen=True)
class UncertaintySummary:
    """Selection-independent uncertainty statistics."""

    entropy: float
    top_probability: float
    margin: float
    other_mass: float
    effective_count: float


def summarize_uncertainty(probabilities: Sequence[float], visible_count: int | None = None) -> UncertaintySummary:
    """Summarize a full probability distribution without Top-K renormalization."""
    if not probabilities:
        raise ValueError("probabilities must be non-empty")
    if any(probability < 0.0 for probability in probabilities):
        raise ValueError("probabilities must be non-negative")
    total = sum(probabilities)
    if total <= 0.0:
        raise ValueError("probabilities must have positive mass")
    normalized = sorted((probability / total for probability in probabilities), reverse=True)
    if visible_count is None:
        visible_count = len(normalized)
    if visible_count < 1:
        raise ValueError("visible_count must be positive")
    visible_count = min(visible_count, len(normalized))
    other_mass = sum(normalized[visible_count:])
    summary_distribution = normalized[:visible_count] + ([other_mass] if other_mass > 0.0 else [])
    entropy = -sum(probability * log2(probability) for probability in summary_distribution if probability > 0.0)
    top = normalized[0]
    second = normalized[1] if len(normalized) > 1 else 0.0
    return UncertaintySummary(entropy, top, top - second, other_mass, 1.0 / sum(probability * probability for probability in summary_distribution))


@dataclass(frozen=True)
class PhenomenalState:
    """Private integrated state candidate derived from perception and context."""

    values: tuple[float, ...]
    source: str = field(default="private", compare=True)

    @classmethod
    def integrate(cls, sensory: Sequence[float], context: Sequence[float], history: Sequence[float] = ()) -> "PhenomenalState":
        values = tuple(float(value) for value in tuple(sensory) + tuple(context) + tuple(history))
        if not values:
            raise ValueError("integrated state requires at least one value")
        mean = sum(values) / len(values)
        centered = tuple(value - mean for value in values)
        return cls((mean,) + centered, "private")

    def distance(self, other: "PhenomenalState") -> float:
        if len(self.values) != len(other.values):
            raise ValueError("states must have equal dimensions")
        return sum((left - right) ** 2 for left, right in zip(self.values, other.values)) ** 0.5


@dataclass(frozen=True)
class CognitiveReadout:
    values: Mapping[str, float]
    report: str | None = None


def readout(state: PhenomenalState, report: bool = True) -> CognitiveReadout:
    """Expose multiple downstream values while allowing report ablation."""
    first = state.values[0]
    values = {"memory": first, "attention": sum(abs(value) for value in state.values[1:])}
    return CognitiveReadout(values, "present" if report else None)
