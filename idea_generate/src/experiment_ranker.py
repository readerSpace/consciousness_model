"""Score experiment candidates using the weights specified in the design."""

from __future__ import annotations

from .consciousness_reasoner import infer_next_experiments
from .models import ExperimentProposal, ResearchGap


DEFAULT_WEIGHTS = {"uncertainty": 1.0, "information_gain": 1.5, "novelty": 0.5, "dependency": 1.0, "cost": 0.7}


def rank_experiments(
    gaps: list[ResearchGap], proposals: list[ExperimentProposal], workspace_capacity: int = 8
) -> list[ExperimentProposal]:
    if not gaps:
        return []
    inference = infer_next_experiments(gaps, capacity=workspace_capacity)
    rank_by_gap_id = {gap_id: index for index, gap_id in enumerate(inference.ranked_gap_ids)}
    gap_by_id = {gap.id: gap for gap in gaps}
    for proposal, gap in zip(proposals, gaps):
        rank = rank_by_gap_id[gap.id]
        proposal.score = round(max(0.0, inference.action.confidence) + (len(gaps) - rank) / len(gaps), 3)
        ranking_rationale = (
            f"ConsciousnessControllerのcycle {inference.state.cycle}で"
            f"有限Workspaceへ統合され、行動候補としてbroadcastされた。"
        )
        proposal.rationale = f"{proposal.rationale} {ranking_rationale}".strip()
    return sorted(proposals, key=lambda proposal: proposal.score, reverse=True)