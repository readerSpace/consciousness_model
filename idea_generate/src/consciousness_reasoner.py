"""Translate research gaps into Consciousness_model control cycles."""

from __future__ import annotations

from dataclasses import dataclass

from Consciousness_model import ConceptualAction, ConsciousnessController, IntegratedState

from .models import ResearchGap


@dataclass(frozen=True)
class ResearchInference:
    """The broadcast state and selected gap order for one research cycle."""

    state: IntegratedState
    action: ConceptualAction
    ranked_gap_ids: tuple[str, ...]


def infer_next_experiments(gaps: list[ResearchGap], capacity: int = 8) -> ResearchInference:
    """Select experiment gaps through the capacity-limited consciousness model.

    Repeated observations encode independently useful evidence. The compressed
    workspace turns their support into concept utility before integration.
    """
    controller = ConsciousnessController(capacity=capacity)
    observations: list[tuple[str, str, str]] = []
    for gap in gaps:
        evidence = gap.uncertainty + gap.information_gain + gap.novelty + gap.dependency_impact - gap.estimated_cost
        repetitions = max(1, round(1 + evidence * 3))
        observations.extend([(gap.id, "next_experiment", gap.target_hypothesis)] * repetitions)
    controller.perceive(observations)
    state = controller.integrate(("next_experiment",))
    action = controller.select_action()
    ranked_gap_ids: list[str] = []
    for relation in action.proposition:
        gap_id = relation[0]
        if gap_id not in ranked_gap_ids:
            ranked_gap_ids.append(gap_id)
    remaining = [gap.id for gap in gaps if gap.id not in ranked_gap_ids]
    return ResearchInference(state, action, tuple(ranked_gap_ids + remaining))