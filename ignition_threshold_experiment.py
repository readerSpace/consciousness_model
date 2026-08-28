"""Measure global-access ignition from finite-workspace candidate competition.

Stimulus strength changes only the candidate's predictive value.  Admission is
then determined by the same compression/prediction/reuse utility as the
workspace model, with a softmax competition rule.  A winning candidate becomes
queryable by planner, predictor, memory, and report modules.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import exp
import json
from typing import Dict, Mapping, Sequence

from Consciousness_model import Concept


@dataclass(frozen=True)
class IgnitionPoint:
    strength: float
    global_access_probability: float
    winner_utility: float
    distractor_utility: float


def _candidate(identifier: str, strength: float) -> Concept:
    return Concept(
        identifier=identifier,
        kind="episode",
        representation=identifier,
        relations=((identifier, "predicts", "outcome"),),
        prediction=strength,
        confidence=strength,
        reuse_count=0,
        provenance=(identifier,),
        support=1,
    )


def global_access_probability(strength: float, distractors: int = 4, distractor_strength: float = 0.5, temperature: float = 0.12) -> IgnitionPoint:
    """Probability that one stimulus wins finite-workspace admission.

    The background candidates share identical utility but distinct provenance;
    their number captures local competition without hard-coding an access
    outcome.  A low temperature represents a sharper winner-take-most gate.
    """
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be within [0, 1]")
    if distractors < 1 or temperature <= 0:
        raise ValueError("distractors and temperature must be positive")
    winner = _candidate("stimulus", strength)
    distractor = _candidate("distractor", distractor_strength)
    winner_utility, distractor_utility = winner.utility(), distractor.utility()
    numerator = exp(winner_utility / temperature)
    denominator = numerator + distractors * exp(distractor_utility / temperature)
    return IgnitionPoint(strength, numerator / denominator, winner_utility, distractor_utility)


def run(strengths: Sequence[float] = tuple(step / 10.0 for step in range(1, 11))) -> Mapping[str, object]:
    points = [global_access_probability(strength) for strength in strengths]
    threshold = next((point.strength for point in points if point.global_access_probability >= 0.5), None)
    jumps = [later.global_access_probability - earlier.global_access_probability for earlier, later in zip(points, points[1:])]
    return {
        "points": [asdict(point) for point in points],
        "access_threshold_at_p_ge_0_5": threshold,
        "maximum_step_change": max(jumps, default=0.0),
    }


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
