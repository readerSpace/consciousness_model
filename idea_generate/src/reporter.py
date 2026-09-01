"""Human-readable research digest formatter."""

from __future__ import annotations

from .models import Concept, ExperimentProposal, ResearchState


def _items(values: list[str], empty: str = "なし") -> str:
    return "\n".join(f"- {value}" for value in values) if values else f"- {empty}"


def render_digest(concepts: list[Concept], state: ResearchState, experiments: list[ExperimentProposal]) -> str:
    concept_lines = [f"{concept.id} {concept.name}" for concept in concepts]
    experiment_lines = [f"{index}. {proposal.title}\nScore: {proposal.score:.2f}\n理由: {proposal.rationale}" for index, proposal in enumerate(experiments, start=1)]
    return "\n".join([
        "================================", "RESEARCH DIGEST", "================================", "",
        "主要概念", "--------------------------------", _items(concept_lines), "",
        "確認済み", "--------------------------------", _items(state.established), "",
        "主要仮説", "--------------------------------", _items(state.hypotheses), "",
        "未検証", "--------------------------------", _items(state.unresolved), "",
        "矛盾 / 弱点", "--------------------------------", _items(state.contradictions), "",
        "================================", "NEXT EXPERIMENTS", "================================", _items(experiment_lines), "",
    ])