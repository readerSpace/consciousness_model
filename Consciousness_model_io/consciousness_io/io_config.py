"""Configurable input/output for the set-theoretic consciousness agent.

The whole point of this package over the base ``consciousness_model`` is that
**what the agent senses and what it can do are supplied by the user, not baked
in**.  Three declarative objects do that:

* :class:`SensorChannel` — an input port.  Raw data from one sensor (camera,
  microphone, range finder, joint encoder, a text string, a dict from a
  simulator ...) is turned into observation :class:`~consciousness_io.facts.Fact`
  s by a user-supplied ``encode`` callable.
* :class:`ActionSchema` — an output port.  It declares an action's cost, how it
  is *predicted* to change beliefs (``effect``), and, optionally, how it is
  *actually executed* in the world (``execute``).
* :class:`AgentConfig` — the scalar knobs of the control loop (workspace size,
  question threshold θ, curiosity weight λ, planning horizon, policy mode).

None of these depend on a particular robot or simulator, so the same agent can
be pointed at a toy 2-D world, a CARLA scene, or a text environment just by
swapping the channel and action lists.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Hashable, Iterable, Mapping, Optional, Sequence, Tuple

from .facts import Fact


# --- Input -----------------------------------------------------------------

RawReading = Any
Encoder = Callable[[RawReading], Iterable[Fact]]


@dataclass(frozen=True)
class SensorChannel:
    """One configurable input port: ``raw sensor data -> observation facts``.

    Parameters
    ----------
    name:
        Identifier used as the key in the per-step raw-input dict.
    encode:
        Callable turning this channel's raw reading into an iterable of
        :class:`Fact`.  Kept application-specific on purpose; the agent never
        looks inside the raw reading itself.
    optional:
        If ``True`` the channel may be absent from a given step's input without
        raising.  Useful for event sensors (e.g. a bump sensor).
    """

    name: str
    encode: Encoder
    optional: bool = False

    def read(self, raw_inputs: Mapping[str, RawReading]) -> Tuple[Fact, ...]:
        if self.name not in raw_inputs:
            if self.optional:
                return ()
            raise KeyError(f"missing input for sensor channel '{self.name}'")
        produced = tuple(self.encode(raw_inputs[self.name]))
        for item in produced:
            if not isinstance(item, Fact):
                raise TypeError(
                    f"sensor '{self.name}' encode() must yield Fact, got {type(item).__name__}"
                )
        return produced


# --- Output ----------------------------------------------------------------

# Predicted effect of an action on the current belief set.  Returns the facts
# the action is expected to add or overwrite (NOT the whole next belief set),
# together with a confidence in [0, 1] about that prediction.  Low confidence
# feeds the question set Q_t.
EffectModel = Callable[[frozenset[Fact], "ActionBinding"], "PredictedEffect"]

# Real-world executor.  Given the chosen binding it drives the actuator and may
# return raw sensor deltas that the caller merges into the next observation.
# Optional: in a simulated / externally-driven loop the environment supplies the
# next observation instead.
Executor = Callable[["ActionBinding"], Optional[Mapping[str, RawReading]]]


@dataclass(frozen=True)
class PredictedEffect:
    """``P(a, B_t)`` for one action: the facts it should produce, plus certainty."""

    added: Tuple[Fact, ...] = ()
    confidence: float = 1.0

    def apply(self, beliefs: frozenset[Fact]) -> frozenset[Fact]:
        from .facts import BeliefStore

        store = BeliefStore().update(beliefs).update(self.added)
        return store.as_set()


@dataclass(frozen=True)
class ActionSchema:
    """One configurable output port / action template.

    Parameters
    ----------
    name:
        Action identifier, e.g. ``"GRASP"``, ``"LOOK"``, ``"PUSH"``, ``"ASK"``.
    effect:
        ``EffectModel`` giving the predicted belief change and its confidence.
        This is what makes the action *plannable*: goal-directed selection
        compares ``effect(...)`` against the goal, and curiosity selection reads
        its ``confidence`` as an uncertainty signal.
    cost:
        Scalar action cost used by the curiosity trade-off ``info_gain - λ·cost``.
    execute:
        Optional real-world executor.  If ``None`` the action is planning-only
        and the environment is expected to advance the world.
    parameters:
        Optional fixed keyword arguments merged into every binding of this
        action (e.g. which gripper, default speed).
    """

    name: str
    effect: EffectModel
    cost: float = 1.0
    execute: Optional[Executor] = None
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def bind(self, **kwargs: Any) -> "ActionBinding":
        merged = dict(self.parameters)
        merged.update(kwargs)
        return ActionBinding(self, merged)


@dataclass(frozen=True)
class ActionBinding:
    """A concrete action = schema + bound arguments, i.e. an element of ``A_t``."""

    schema: "ActionSchema"
    arguments: Mapping[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.schema.name

    @property
    def cost(self) -> float:
        return self.schema.cost

    def predict(self, beliefs: frozenset[Fact]) -> PredictedEffect:
        return self.schema.effect(beliefs, self)

    def run(self) -> Optional[Mapping[str, RawReading]]:
        if self.schema.execute is None:
            return None
        return self.schema.execute(self)

    def label(self) -> str:
        if not self.arguments:
            return self.name
        inner = ",".join(f"{key}={value}" for key, value in sorted(self.arguments.items()))
        return f"{self.name}({inner})"


# --- Loop configuration ----------------------------------------------------

@dataclass(frozen=True)
class AgentConfig:
    """Scalar knobs of the ``C_{t+1} = F(C_t, O_{t+1}, a_t)`` control loop."""

    workspace_capacity: int = 7
    """|W_t| upper bound — the finite set of items 'currently in mind'."""

    compressed_capacity: int = 16
    """Capacity of the MDL compressed workspace producing Z_t."""

    question_threshold: float = 0.5
    """θ.  An aspect enters Q_t when its predictive uncertainty H(...) exceeds θ."""

    curiosity_weight: float = 0.3
    """λ in the curiosity objective ``I(Q_t; O_{t+1}|a) - λ·Cost(a)``."""

    policy: str = "auto"
    """'auto' | 'goal' | 'curiosity'.  'auto' exploits a reachable goal, else asks."""

    goal_commit_margin: float = 1e-9
    """An action must cut goal distance by more than this to count as progress."""

    def __post_init__(self) -> None:
        if self.workspace_capacity < 1:
            raise ValueError("workspace_capacity must be positive")
        if self.compressed_capacity < 1:
            raise ValueError("compressed_capacity must be positive")
        if not 0.0 <= self.question_threshold <= 1.0:
            raise ValueError("question_threshold must be in [0, 1]")
        if self.curiosity_weight < 0.0:
            raise ValueError("curiosity_weight must be non-negative")
        if self.policy not in {"auto", "goal", "curiosity"}:
            raise ValueError("policy must be 'auto', 'goal', or 'curiosity'")
