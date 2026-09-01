"""Finite salience-limited workspace for experiment decisions."""

from __future__ import annotations

from .models import ExperimentProposal


def select_workspace(proposals: list[ExperimentProposal], capacity: int = 8) -> list[ExperimentProposal]:
    if capacity < 1:
        raise ValueError("Workspace capacity must be at least one.")
    return sorted(proposals, key=lambda proposal: proposal.score, reverse=True)[:capacity]