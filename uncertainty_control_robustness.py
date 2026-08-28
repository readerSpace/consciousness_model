"""Sensitivity tests for the z_U uncertainty encoder and three-action rule."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from math import log2
from typing import Dict, Mapping, Sequence

from cross_domain_revision_experiment import _build_world
from partial_observation_experiment import _workspace
from uncertainty_encoder_experiment import _entropy, _semantic_rows, _state


@dataclass(frozen=True)
class DecisionPoint:
    label: str
    action: str
    entropy: float
    eig_observe: float
    eig_research: float
    observe_cost: float
    search_cost: float


def _decide(state, eig_scale: float = 1.0, entropy_scale: float = 1.0, observe_cost: float | None = None, search_cost: float | None = None) -> str:
    # entropy_scale models a calibration error in the common uncertainty scale;
    # EIG is scaled consistently, so only relative action values matter.
    current_accuracy = 1.0 - state.prediction_error
    observe_cost = state.cost_observe if observe_cost is None else observe_cost
    search_cost = state.cost_search if search_cost is None else search_cost
    values = {
        "commit": current_accuracy,
        "re-search": min(1.0, current_accuracy + eig_scale * entropy_scale * state.eig_research) - search_cost,
        "observe": min(1.0, current_accuracy + eig_scale * entropy_scale * state.eig_observe) - observe_cost,
    }
    return max(values, key=values.get)


def _physics_state(noise: int):
    workspace, _, _, start, goal, _ = _build_world("physics", seed=1, noise_per_goal=noise)
    rows = _semantic_rows(workspace.attention_distribution((start, goal), focus=0.8))
    return _state([probability for _, probability in rows], eig_observe=0.0)


def _partial_state(cue_reliability: float = 1.0):
    rows = _semantic_rows(_workspace().attention_distribution(("door", "go"), focus=0.8))
    entropy = _entropy([probability for _, probability in rows])
    return _state([probability for _, probability in rows], eig_observe=cue_reliability * entropy)


def run() -> Mapping[str, object]:
    clear, noisy, partial = _physics_state(0), _physics_state(12), _partial_state()
    coefficient_grid = []
    for scale in (0.5, 1.0, 1.5, 2.0):
        coefficient_grid.append({
            "scale": scale,
            "physics_clear": _decide(clear, eig_scale=scale, entropy_scale=scale),
            "physics_noisy": _decide(noisy, eig_scale=scale, entropy_scale=scale),
            "partial": _decide(partial, eig_scale=scale, entropy_scale=scale),
        })
    noise_sweep = []
    for noise in (0, 2, 4, 8, 12, 20):
        state = _physics_state(noise)
        noise_sweep.append(asdict(DecisionPoint(f"noise={noise}", _decide(state), state.entropy, state.eig_observe, state.eig_research, state.cost_observe, state.cost_search)))
    cost_sweep = []
    for observe_cost, search_cost in ((0.08, 0.12), (0.25, 0.12), (0.80, 0.60)):
        cost_sweep.append(asdict(DecisionPoint(
            f"observe={observe_cost},search={search_cost}", _decide(partial, observe_cost=observe_cost, search_cost=search_cost),
            partial.entropy, partial.eig_observe, partial.eig_research, observe_cost, search_cost,
        )))
    cue_sweep = []
    for reliability in (1.0, 0.8, 0.6, 0.4, 0.2):
        state = _partial_state(reliability)
        cue_sweep.append({
            "cue_reliability": reliability,
            "action": _decide(state),
            "eig_observe": state.eig_observe,
            "expected_observe_success": reliability,
        })
    return {"coefficient_grid": coefficient_grid, "noise_sweep": noise_sweep, "cost_sweep": cost_sweep, "cue_reliability_sweep": cue_sweep}


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2, sort_keys=True))
