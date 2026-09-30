"""Set-theoretic consciousness agent with configurable I/O.

Implements the state and cycle from the design note::

    C_t = (O_t, B_t, P_t, G_t, Q_t, A_t, W_t)                      (state)
    C_{t+1} = F(C_t, O_{t+1}, a_t)                                 (transition)

    O_t --compression--> Z_t --prediction--> P_t
    Z_t, P_t --mismatch--> Q_t
    Z_t, Q_t, G_t --attention--> W_t
    W_t --action selection--> a_t
    a_t --world--> O_{t+1}

Operational definition used here:

    consciousness = the finite set of information the agent can currently access
    that is needed for prediction, goals, questions, and action (i.e. W_t).

The agent does no learning by itself; it is a deterministic controller wired
from user-supplied sensor channels and action schemas.  Compression reuses the
MDL-style :class:`~consciousness_io.compressed_workspace.CompressedWorkspace`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import log2
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .compressed_workspace import CompressedWorkspace
from .facts import BeliefStore, Fact, KeyFn, goal_distance
from .io_config import (
    ActionBinding,
    ActionSchema,
    AgentConfig,
    PredictedEffect,
    RawReading,
    SensorChannel,
)


# --- Prediction & question records -----------------------------------------

@dataclass(frozen=True)
class Prediction:
    """One element of ``P_t``: an action and its predicted consequence."""

    action: ActionBinding
    effect: PredictedEffect
    goal_distance: float
    """D(P(a, B_t), G_t) — smaller is closer to the goal."""

    @property
    def uncertainty(self) -> float:
        """H proxy in [0, 1]: how unsure the effect model is about this action."""

        return 1.0 - self.effect.confidence


@dataclass(frozen=True)
class Question:
    """One element of ``Q_t``: an aspect whose next-step outcome is uncertain."""

    subject: str
    uncertainty: float
    origin: str  # "unpredictable_action" | "unmet_goal" | custom
    resolving_actions: Tuple[str, ...] = ()

    def __str__(self) -> str:
        return f"?{self.subject}[{self.origin},H={self.uncertainty:.2f}]"


@dataclass(frozen=True)
class Decision:
    """The selected action ``a_t`` plus why it was chosen."""

    action: Optional[ActionBinding]
    mode: str  # "goal" | "curiosity" | "noop"
    score: float
    rationale: str

    @property
    def label(self) -> str:
        return self.action.label() if self.action is not None else "NOOP"


@dataclass(frozen=True)
class ConsciousState:
    """A snapshot of ``C_t = (O_t, Z_t, B_t, P_t, G_t, Q_t, A_t, W_t)``."""

    cycle: int
    observations: Tuple[Fact, ...]          # O_t
    compressed: Tuple[str, ...]             # Z_t  (concept representations)
    beliefs: Tuple[Fact, ...]               # B_t
    predictions: Tuple[Prediction, ...]     # P_t
    goals: Tuple[Fact, ...]                 # G_t
    questions: Tuple[Question, ...]         # Q_t
    actions: Tuple[ActionBinding, ...]      # A_t
    workspace: Tuple[Fact, ...]             # W_t
    decision: Decision
    goal_distance: float

    def summary(self) -> str:
        lines = [
            f"cycle {self.cycle}: D(B,G)={self.goal_distance:.2f}  decision={self.decision.label} ({self.decision.mode})",
            f"  O_t ({len(self.observations)}): " + ", ".join(str(f) for f in self.observations),
            f"  Z_t ({len(self.compressed)}): " + ", ".join(self.compressed),
            f"  B_t ({len(self.beliefs)}): " + ", ".join(str(f) for f in self.beliefs),
            f"  Q_t ({len(self.questions)}): " + ", ".join(str(q) for q in self.questions),
            f"  W_t ({len(self.workspace)}): " + ", ".join(str(f) for f in self.workspace),
        ]
        return "\n".join(lines)


# --- The agent -------------------------------------------------------------

class ConsciousAgent:
    """A configurable, set-theoretic conscious-access controller.

    Parameters
    ----------
    sensors:
        Input ports (``SensorChannel``).  Their union defines what can enter O_t.
    actions:
        Output ports (``ActionSchema``).  Bound at each step into the action set
        ``A_t`` via :meth:`available_actions` (override for parameterised actions).
    goals:
        Initial goal facts ``G_t``.  Replaceable at runtime with :meth:`set_goals`.
    config:
        Scalar loop knobs (:class:`AgentConfig`).
    belief_key_fn:
        Optional custom identity for belief overwrite (see ``BeliefStore``).
    """

    def __init__(
        self,
        sensors: Sequence[SensorChannel],
        actions: Sequence[ActionSchema],
        goals: Sequence[Fact] = (),
        config: AgentConfig | None = None,
        belief_key_fn: KeyFn | None = None,
    ) -> None:
        if not sensors:
            raise ValueError("at least one sensor channel is required")
        if not actions:
            raise ValueError("at least one action schema is required")
        names = [s.name for s in sensors]
        if len(names) != len(set(names)):
            raise ValueError("sensor channel names must be unique")
        self.config = config or AgentConfig()
        self.sensors: Tuple[SensorChannel, ...] = tuple(sensors)
        self.actions: Tuple[ActionSchema, ...] = tuple(actions)
        self.beliefs = BeliefStore(belief_key_fn)
        self.goals: Tuple[Fact, ...] = tuple(goals)
        self._cycle = 0
        self.history: List[ConsciousState] = []

    # -- configuration at runtime -----------------------------------------

    def set_goals(self, goals: Sequence[Fact]) -> None:
        self.goals = tuple(goals)

    def available_actions(self) -> Tuple[ActionBinding, ...]:
        """Build ``A_t``.  Default: one zero-argument binding per schema.

        Override (or pass parameterised schemas) when actions need arguments
        grounded in current beliefs, e.g. ``GRASP(cup)`` for each visible object.
        """

        return tuple(schema.bind() for schema in self.actions)

    # -- the cycle F -------------------------------------------------------

    def step(self, raw_inputs: Mapping[str, RawReading]) -> ConsciousState:
        """Run one control cycle ``C_{t+1} = F(C_t, O_{t+1}, a_t)``.

        Returns the new :class:`ConsciousState`.  If the selected action has a
        configured executor, it is run and any raw sensor deltas it returns are
        *not* auto-fed back here — the caller decides how the world advances
        (call :meth:`step` again with the next reading).
        """

        self._cycle += 1

        # O_t : encode every sensor channel into observation facts.
        observations = self._observe(raw_inputs)

        # B_{t+1} = B_t ∪ f(O_t)
        self.beliefs.update(observations)
        belief_set = self.beliefs.as_set()

        # Z_t : MDL compression of the current beliefs.
        compressed = self._compress(belief_set)

        # A_t and P_t : predicted effect of each available action.
        actions = self.available_actions()
        current_distance = goal_distance(self.beliefs, self.goals)
        predictions = self._predict(actions, belief_set)

        # Q_t : mismatch / high predictive uncertainty.
        questions = self._questions(predictions, belief_set)

        # W_t : finite attention over O ∪ B ∪ P ∪ G ∪ Q, focused by G ∪ Q.
        workspace = self._attend(observations, belief_set, predictions, questions)

        # a_t : action selection from the workspace.
        decision = self._select(predictions, questions, current_distance)

        # a_t --world--> executor (optional side effect).
        if decision.action is not None:
            decision.action.run()

        state = ConsciousState(
            cycle=self._cycle,
            observations=observations,
            compressed=compressed,
            beliefs=tuple(sorted(belief_set, key=str)),
            predictions=predictions,
            goals=self.goals,
            questions=questions,
            actions=actions,
            workspace=workspace,
            decision=decision,
            goal_distance=current_distance,
        )
        self.history.append(state)
        return state

    def run(self, stream: Iterable[Mapping[str, RawReading]], max_steps: int | None = None) -> List[ConsciousState]:
        """Drive the loop over an iterable of raw readings."""

        states: List[ConsciousState] = []
        for index, raw in enumerate(stream):
            if max_steps is not None and index >= max_steps:
                break
            states.append(self.step(raw))
        return states

    # -- stages ------------------------------------------------------------

    def _observe(self, raw_inputs: Mapping[str, RawReading]) -> Tuple[Fact, ...]:
        collected: List[Fact] = []
        for channel in self.sensors:
            collected.extend(channel.read(raw_inputs))
        return tuple(collected)

    def _compress(self, beliefs: frozenset[Fact]) -> Tuple[str, ...]:
        workspace = CompressedWorkspace(capacity=self.config.compressed_capacity)
        relations = [f.to_relation() for f in beliefs]
        if relations:
            workspace.ingest(relations)
        return tuple(
            workspace.long_term[identifier].representation
            for identifier in workspace.items
        )

    def _predict(self, actions: Sequence[ActionBinding], beliefs: frozenset[Fact]) -> Tuple[Prediction, ...]:
        predictions: List[Prediction] = []
        for action in actions:
            effect = action.predict(beliefs)
            predicted_beliefs = effect.apply(beliefs)
            distance = goal_distance(predicted_beliefs, self.goals)
            predictions.append(Prediction(action, effect, distance))
        return tuple(predictions)

    def _questions(self, predictions: Sequence[Prediction], beliefs: frozenset[Fact]) -> Tuple[Question, ...]:
        """Build Q_t = { x | H(O_{t+1}|Z_t, x) > θ }.

        Two default sources of high entropy:

        * an action whose predicted effect the model is unsure about
          (``uncertainty > θ``) — the outcome of taking it is a question;
        * a goal fact not currently entailed and not reachable in one step by any
          confident action — *how to achieve it* is a question.
        """

        theta = self.config.question_threshold
        questions: List[Question] = []

        # (a) unpredictable actions
        for prediction in predictions:
            if prediction.uncertainty > theta:
                questions.append(Question(
                    subject=prediction.action.label(),
                    uncertainty=prediction.uncertainty,
                    origin="unpredictable_action",
                    resolving_actions=(prediction.action.label(),),
                ))

        # (b) unmet, not-obviously-reachable goals
        store = BeliefStore().update(beliefs)
        for goal in self.goals:
            if store.satisfies(goal):
                continue
            resolvers = tuple(
                p.action.label()
                for p in predictions
                if p.goal_distance < goal_distance(beliefs, self.goals) and p.effect.confidence >= (1.0 - theta)
            )
            if not resolvers:
                questions.append(Question(
                    subject=str(goal),
                    uncertainty=1.0,
                    origin="unmet_goal",
                ))
        return tuple(questions)

    def _attend(
        self,
        observations: Tuple[Fact, ...],
        beliefs: frozenset[Fact],
        predictions: Sequence[Prediction],
        questions: Sequence[Question],
    ) -> Tuple[Fact, ...]:
        """Select W_t: the top-``capacity`` facts most relevant to G_t ∪ Q_t."""

        focus_terms: set[str] = set()
        for goal in self.goals:
            focus_terms.update(goal.tokens())
        for question in questions:
            focus_terms.update(question.subject.replace("(", " ").replace(")", " ").replace(",", " ").split())

        # Candidate pool = O ∪ B ∪ G ∪ (facts predicted by the best actions).
        pool: Dict[Tuple[str, Tuple[Any, ...]], Fact] = {}
        for source in (tuple(beliefs), observations, self.goals):
            for f in source:
                pool[(f.predicate, f.args)] = f
        for prediction in sorted(predictions, key=lambda p: p.goal_distance)[:2]:
            for f in prediction.effect.added:
                pool[(f.predicate, f.args)] = f

        scored = sorted(
            pool.values(),
            key=lambda f: (self._relevance(f, focus_terms), f.confidence, str(f)),
            reverse=True,
        )
        return tuple(scored[: self.config.workspace_capacity])

    @staticmethod
    def _relevance(fact: Fact, focus_terms: set[str]) -> int:
        return sum(token in focus_terms for token in fact.tokens())

    def _select(self, predictions: Sequence[Prediction], questions: Sequence[Question], current_distance: float) -> Decision:
        """Choose ``a_t`` per the configured policy.

        goal      : a* = argmin_a D(P(a, B_t), G_t)
        curiosity : a* = argmax_a [ I(Q_t; O_{t+1}|a) - λ·Cost(a) ]
        auto      : take the goal action if it makes progress; else be curious.
        """

        policy = self.config.policy
        goal_choice = self._best_goal_action(predictions)
        curiosity_choice = self._best_curiosity_action(predictions, questions)

        if policy == "goal":
            return goal_choice or Decision(None, "noop", 0.0, "no action defined")
        if policy == "curiosity":
            return curiosity_choice or Decision(None, "noop", 0.0, "nothing to be curious about")

        # auto
        makes_progress = (
            goal_choice is not None
            and goal_choice.action is not None
            and (current_distance - goal_choice.score) > self.config.goal_commit_margin
        )
        if makes_progress:
            return goal_choice
        if questions and curiosity_choice is not None:
            return curiosity_choice
        if goal_choice is not None and current_distance <= self.config.goal_commit_margin:
            return Decision(None, "noop", 0.0, "goal already satisfied")
        return goal_choice or Decision(None, "noop", 0.0, "no progressing or curious action")

    def _best_goal_action(self, predictions: Sequence[Prediction]) -> Optional[Decision]:
        if not predictions:
            return None
        best = min(predictions, key=lambda p: (p.goal_distance, p.uncertainty, p.action.cost))
        return Decision(
            action=best.action,
            mode="goal",
            score=best.goal_distance,  # D(P(a,B),G): lower is better
            rationale=f"argmin D(P(a,B),G)={best.goal_distance:.2f}",
        )

    def _best_curiosity_action(self, predictions: Sequence[Prediction], questions: Sequence[Question]) -> Optional[Decision]:
        if not predictions:
            return None
        resolvable = {name for question in questions for name in question.resolving_actions}
        lam = self.config.curiosity_weight
        best: Optional[Tuple[float, Prediction]] = None
        for prediction in predictions:
            # Expected information gain proxy: an action's own outcome entropy,
            # boosted when it is known to resolve a live question.
            info_gain = prediction.uncertainty
            if prediction.action.label() in resolvable:
                info_gain += 1.0
            objective = info_gain - lam * prediction.action.cost
            if best is None or objective > best[0]:
                best = (objective, prediction)
        if best is None or best[0] <= 0.0:
            return None
        objective, prediction = best
        return Decision(
            action=prediction.action,
            mode="curiosity",
            score=objective,  # I - λ·cost: higher is better
            rationale=f"argmax I-λcost={objective:.2f}",
        )
