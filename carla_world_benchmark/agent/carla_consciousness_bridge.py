"""CARLA-independent bridge between driving perception and the consciousness model.

This module deliberately imports nothing from ``carla``.  It converts a list of
symbolic hazard observations plus ego/route state into

* a set of :class:`WorkspaceItem` candidates ranked by salience,
* a bounded selection produced by :class:`FiniteWorkspace`,
* an integrated :class:`PhenomenalState` and its ``memory`` / ``attention`` readout,
* a longitudinal driving decision (target speed, brake override, goal mode).

Keeping it free of the simulator makes the selection mechanism unit-testable
without a GPU, which is the only part of the stack that can be verified offline.
No claim about subjective experience is made anywhere in this file.
"""

from __future__ import annotations

import math
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Dict, Iterable, Mapping, Sequence, Tuple

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from Consciousness_model.consciousness import (  # noqa: E402
    FiniteWorkspace,
    PhenomenalState,
    WorkspaceItem,
    readout,
)

INF = float("inf")

# Prior weight per hazard class.  Vulnerable road users outrank vehicles, which
# outrank regulatory elements, which outrank static geometry.
KIND_PRIOR: Mapping[str, float] = {
    "walker": 1.30,
    "bicycle": 1.20,
    "vehicle": 1.00,
    "traffic_light": 0.85,
    "stop_sign": 0.75,
    "static": 0.55,
}

SELECTION_MODES = ("workspace", "all", "random", "blind")


@dataclass(frozen=True)
class HazardObservation:
    """One symbolic hazard, expressed in the ego frame."""

    identifier: str
    kind: str
    distance: float          # metres ahead along the ego path, >= 0
    lateral: float           # metres of |offset| from the ego path
    relative_speed: float    # m/s, positive means closing
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.identifier:
            raise ValueError("identifier must be non-empty")
        if self.kind not in KIND_PRIOR:
            raise ValueError("unknown hazard kind: {}".format(self.kind))
        for name in ("distance", "lateral", "relative_speed"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError("{} must be finite".format(name))
        if self.distance < 0.0 or self.lateral < 0.0:
            raise ValueError("distance and lateral must be non-negative")

    @property
    def time_to_collision(self) -> float:
        """Constant-velocity TTC.  ``inf`` when the hazard is not closing."""
        if self.relative_speed <= 1e-3:
            return INF
        return self.distance / self.relative_speed


def corridor_weight(lateral: float, corridor_half_width: float = 1.75) -> float:
    """Relevance of a lateral offset to the ego path, in (0, 1]."""
    if corridor_half_width <= 0.0:
        raise ValueError("corridor_half_width must be positive")
    return math.exp(-0.5 * (lateral / corridor_half_width) ** 2)


def salience(observation: HazardObservation, ego_speed: float, risk: float = 0.0) -> float:
    """Deterministic salience used to rank workspace candidates.

    Urgency dominates: a closing hazard three seconds away outranks a stationary
    one at the same distance.  ``risk`` is the controller's own calibration term,
    so a badly calibrated agent widens its attention rather than narrowing it.
    """
    if ego_speed < 0.0:
        raise ValueError("ego_speed must be non-negative")
    if not 0.0 <= risk <= 1.0:
        raise ValueError("risk must be in [0, 1]")
    ttc = observation.time_to_collision
    urgency = 1.0 / (1.0 + ttc) if math.isfinite(ttc) else 0.0
    proximity = 1.0 / (1.0 + observation.distance / 10.0)
    approach = math.tanh(max(0.0, observation.relative_speed) / 8.0)
    speed_gate = 0.5 + 0.5 * math.tanh(ego_speed / 8.0)
    prior = KIND_PRIOR[observation.kind]
    base = 2.2 * urgency + 0.9 * proximity + 0.6 * approach
    return prior * corridor_weight(observation.lateral) * speed_gate * base * (1.0 + 0.5 * risk)


def to_candidates(
    observations: Iterable[HazardObservation],
    ego_speed: float,
    risk: float = 0.0,
) -> Tuple[WorkspaceItem, ...]:
    """Encode hazards as valued workspace candidates."""
    return tuple(
        WorkspaceItem(observation.identifier, salience(observation, ego_speed, risk), observation)
        for observation in observations
    )


@dataclass(frozen=True)
class DrivingDecision:
    """Everything the CARLA agent needs, plus the diagnostics an experiment needs."""

    target_speed_kmh: float
    brake: float
    hazard_stop: bool
    goal: str
    admitted: Tuple[str, ...]
    dropped: Tuple[str, ...]
    memory_signal: float
    attention_signal: float
    risk: float
    min_ttc_admitted: float
    min_ttc_true: float
    missed_critical: Tuple[str, ...]

    def as_record(self) -> Dict[str, object]:
        return {
            "target_speed_kmh": self.target_speed_kmh,
            "brake": self.brake,
            "hazard_stop": self.hazard_stop,
            "goal": self.goal,
            "admitted": list(self.admitted),
            "dropped": list(self.dropped),
            "memory_signal": self.memory_signal,
            "attention_signal": self.attention_signal,
            "risk": self.risk,
            "min_ttc_admitted": None if self.min_ttc_admitted == INF else self.min_ttc_admitted,
            "min_ttc_true": None if self.min_ttc_true == INF else self.min_ttc_true,
            "missed_critical": list(self.missed_critical),
        }


@dataclass(frozen=True)
class PolicyConfig:
    capacity: int = 4
    selection: str = "workspace"
    base_target_speed_kmh: float = 30.0
    min_target_speed_kmh: float = 6.0
    safety_enter_ttc: float = 3.0
    safety_release_ttc: float = 5.0
    brake_ttc: float = 1.5
    stop_distance: float = 4.0
    corridor_half_width: float = 1.75
    risk_window: int = 20
    seed: int = 0

    def __post_init__(self) -> None:
        if self.capacity < 1:
            raise ValueError("capacity must be positive")
        if self.selection not in SELECTION_MODES:
            raise ValueError("selection must be one of {}".format(SELECTION_MODES))
        if not 0.0 < self.min_target_speed_kmh <= self.base_target_speed_kmh:
            raise ValueError("min_target_speed_kmh must be in (0, base_target_speed_kmh]")
        if not 0.0 < self.brake_ttc < self.safety_enter_ttc < self.safety_release_ttc:
            raise ValueError("thresholds must satisfy 0 < brake < enter < release")


class ConsciousnessDrivingPolicy:
    """Finite-capacity hazard selection driving a longitudinal policy.

    One control cycle: encode hazards, admit at most ``capacity`` of them, integrate
    an internal state from ego/route/history features, read out ``memory`` and
    ``attention``, then choose goal mode, target speed and brake override from the
    admitted set only.  Hazards that were not admitted cannot influence the
    decision -- that restriction is the hypothesis under test, so it is enforced
    structurally rather than by convention.
    """

    def __init__(self, config: PolicyConfig | None = None) -> None:
        self.config = config or PolicyConfig()
        self.workspace = FiniteWorkspace(capacity=self.config.capacity)
        self.risk = 0.0
        self.goal = "CRUISE"
        self.cycle = 0
        self._random = random.Random(self.config.seed)
        self._previous_dropped: Tuple[str, ...] = ()
        self._previous_target = self.config.base_target_speed_kmh
        self._previous_brake = 0.0
        self._miss_history: list[float] = []

    # -- selection -----------------------------------------------------------

    def _select(
        self,
        observations: Sequence[HazardObservation],
        ego_speed: float,
    ) -> Tuple[Tuple[HazardObservation, ...], Tuple[HazardObservation, ...]]:
        by_id = {observation.identifier: observation for observation in observations}
        mode = self.config.selection
        if mode == "blind":
            return (), tuple(observations)
        if mode == "all":
            return tuple(observations), ()
        if mode == "random":
            identifiers = sorted(by_id)
            self._random.shuffle(identifiers)
            chosen = set(identifiers[: self.config.capacity])
            return (
                tuple(by_id[key] for key in by_id if key in chosen),
                tuple(by_id[key] for key in by_id if key not in chosen),
            )
        # ``workspace``: the model under test.  A fresh buffer per cycle keeps the
        # decision a function of the current scene; persistence across cycles is
        # carried by ``risk``, not by stale candidates.
        self.workspace = FiniteWorkspace(capacity=self.config.capacity)
        admitted_items = self.workspace.ingest(to_candidates(observations, ego_speed, self.risk))
        admitted_ids = {item.identifier for item in admitted_items}
        return (
            tuple(by_id[key] for key in by_id if key in admitted_ids),
            tuple(by_id[key] for key in by_id if key not in admitted_ids),
        )

    # -- calibration ---------------------------------------------------------

    def _update_risk(self, observations: Sequence[HazardObservation]) -> Tuple[str, ...]:
        """Charge the controller for hazards it dropped and then had to face.

        A hazard that was excluded last cycle and is critical this cycle is a
        capacity-induced miss.  Repeated misses raise ``risk``, which both widens
        salience and lowers the target speed.  This is the only feedback path from
        outcome to control, so an ablation that removes it is meaningful.
        """
        critical = {
            observation.identifier
            for observation in observations
            if observation.time_to_collision < self.config.brake_ttc
            or observation.distance < self.config.stop_distance
        }
        missed = tuple(sorted(critical & set(self._previous_dropped)))
        self._miss_history.append(1.0 if missed else 0.0)
        window = self._miss_history[-self.config.risk_window :]
        miss_rate = sum(window) / len(window)
        self.risk = min(1.0, max(0.0, 0.7 * self.risk + 0.6 * miss_rate))
        return missed

    # -- one cycle -----------------------------------------------------------

    def step(
        self,
        ego_speed: float,
        hazards: Sequence[HazardObservation],
        route_curvature: float = 0.0,
        distance_to_goal: float = 0.0,
        speed_limit_kmh: float | None = None,
    ) -> DrivingDecision:
        if ego_speed < 0.0:
            raise ValueError("ego_speed must be non-negative")
        if not 0.0 <= route_curvature <= 1.0:
            raise ValueError("route_curvature must be in [0, 1]")
        self.cycle += 1
        hazards = tuple(hazards)
        missed = self._update_risk(hazards)
        admitted, dropped = self._select(hazards, ego_speed)

        min_ttc_admitted = min((item.time_to_collision for item in admitted), default=INF)
        min_ttc_true = min((item.time_to_collision for item in hazards), default=INF)
        nearest_admitted = min((item.distance for item in admitted), default=INF)

        ceiling = self.config.base_target_speed_kmh
        if speed_limit_kmh is not None and speed_limit_kmh > 0.0:
            ceiling = min(ceiling, float(speed_limit_kmh))

        state = PhenomenalState.integrate(
            sensory=(
                min(ego_speed / 20.0, 2.0),
                1.0 / (1.0 + min_ttc_admitted) if math.isfinite(min_ttc_admitted) else 0.0,
                1.0 / (1.0 + nearest_admitted) if math.isfinite(nearest_admitted) else 0.0,
                min(len(admitted) / max(1, self.config.capacity), 1.0),
            ),
            context=(route_curvature, min(distance_to_goal / 100.0, 2.0), ceiling / 50.0),
            history=(self._previous_target / 50.0, self._previous_brake, self.risk),
        )
        signals = readout(state)
        memory_signal = float(signals.values["memory"])
        attention_signal = float(signals.values["attention"])

        # Goal arbitration with hysteresis: a safety goal suspends cruising and is
        # only released once the admitted set has been clear for a wider margin.
        if min_ttc_admitted < self.config.safety_enter_ttc or nearest_admitted < self.config.stop_distance:
            self.goal = "SAFETY"
        elif min_ttc_admitted > self.config.safety_release_ttc and nearest_admitted > 2.0 * self.config.stop_distance:
            self.goal = "CRUISE"

        attention_factor = 1.0 / (1.0 + 0.25 * attention_signal)
        curvature_factor = 1.0 - 0.45 * route_curvature
        risk_factor = 1.0 - 0.35 * self.risk
        target = ceiling * attention_factor * curvature_factor * risk_factor
        if self.goal == "SAFETY":
            target *= 0.45
        target = max(self.config.min_target_speed_kmh, min(ceiling, target))

        hazard_stop = (
            min_ttc_admitted < self.config.brake_ttc or nearest_admitted < self.config.stop_distance
        )
        if hazard_stop:
            brake = 1.0
        elif min_ttc_admitted < self.config.safety_enter_ttc:
            span = self.config.safety_enter_ttc - self.config.brake_ttc
            brake = 0.6 * (self.config.safety_enter_ttc - min_ttc_admitted) / span
        else:
            brake = 0.0
        brake = max(0.0, min(1.0, brake))

        self._previous_dropped = tuple(item.identifier for item in dropped)
        self._previous_target = target
        self._previous_brake = brake

        return DrivingDecision(
            target_speed_kmh=target,
            brake=brake,
            hazard_stop=hazard_stop,
            goal=self.goal,
            admitted=tuple(item.identifier for item in admitted),
            dropped=self._previous_dropped,
            memory_signal=memory_signal,
            attention_signal=attention_signal,
            risk=self.risk,
            min_ttc_admitted=min_ttc_admitted,
            min_ttc_true=min_ttc_true,
            missed_critical=missed,
        )
