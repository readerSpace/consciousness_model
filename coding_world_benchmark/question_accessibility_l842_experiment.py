"""L8.42: a question the person cannot answer is not an expensive question.

L8.39 chose between three kinds of question by what they leave standing and, on
a tie, by a cost vector.  Every question it could think of, it could also ask.
That is the assumption this layer removes.  Some questions are **not in the
action space at all**: the person cannot recall an episode they have not opened,
and cannot compare two episodes one of which they cannot refer to.

The design decision is the whole layer.  ``askable`` does **not** join the cost
vector::

    Q_available(s) = { q in Q : askable(q, c, s) = 1 }

is formed **first**, and then L8.36's chooser runs over it **unchanged** -- the
same `ask_pool` seam L8.39 opened, given a smaller pool.  Modelling "cannot ask"
as "very expensive" would let a large enough tie-break, or a world with no
alternative, talk the planner into asking it anyway; there is no penalty value
here to be outweighed.  This is L8.36's own distinction, from the other side:
capability was a **veto** and cost was a comparison, and accessibility belongs
with the veto.

The consequence worth having is a state the earlier layers could not express.
``IMPOSSIBLE`` used to mean "nothing would help".  With accessibility it splits:

``IMPOSSIBLE``
    no question is informative at any level of access.  More access changes
    nothing, and abandoning the goal is correct.

``BLOCKED``
    an informative question exists and is out of reach right now.  Reporting this
    as ``IMPOSSIBLE`` would permanently abandon a task that one further action --
    opening the episode -- would finish.

The distinction is made falsifiable rather than asserted: every ``BLOCKED`` claim
is checked by re-running the *same* planner with full access, and every
``IMPOSSIBLE`` claim by checking that full access still does not finish it.
"""
from __future__ import annotations

from dataclasses import dataclass

from .cost_aware_dialogue_l836_experiment import (
    ASK, IMPOSSIBLE, MULTIPLE_OPTIMAL, RESOLVED, Question, answer, choose,
)
from .cost_aware_joint_l839_experiment import (
    CONTEXT, World, context_questions, joint_questions, relation_questions,
    unary_questions,
)
from .nway_semantics_l837_experiment import CONTRADICTION

BLOCKED = "BLOCKED"


@dataclass(frozen=True)
class Access:
    """What the person can currently refer to, and what they can currently recall.

    Two levels, because the questions need different things.  Naming an episode
    at all is enough to be asked whether two of them behave the same; saying what
    a word meant *in* an episode needs its details in front of them.
    """

    known: frozenset[str]
    detail: frozenset[str]

    @classmethod
    def of(cls, known=(), detail=()) -> "Access":
        known = frozenset(known) | frozenset(detail)
        return cls(known, frozenset(detail))

    @classmethod
    def full(cls, contexts) -> "Access":
        return cls.of(contexts, contexts)

    def opening(self, context: str) -> "Access":
        """The person opens an episode: they can now recall its details."""
        return Access(self.known | {context}, self.detail | {context})


def named_contexts(question: Question, contexts) -> frozenset[str]:
    """Which episodes the question mentions, read off the question itself.

    Derived from the text rather than from the pool it came out of, so the audit
    below is independent of the construction it is auditing.
    """
    return frozenset(context for context in contexts
                     if f"「{context}」" in question.text)


def askable(question: Question, contexts, access: Access) -> bool:
    named = named_contexts(question, contexts)
    if question.kind == CONTEXT:
        # A question about the shape of the usage across episodes needs every
        # episode it compares to be referable. One that names none compares them
        # all, so it needs them all.
        return (named or frozenset(contexts)) <= access.known
    return bool(named) and named <= access.detail


def accessible_pool(surfaces, access: Access):
    """L8.39's question space, built only over the episodes the state allows.

    The restriction is applied by handing the builders fewer contexts, not by
    scoring their output, which is what makes "not in the action space" literal.
    """
    def pool(space, contexts, universe):
        detail = tuple(c for c in contexts if c in access.detail)
        known = tuple(c for c in contexts if c in access.known)
        found = (unary_questions(space, detail, universe, surfaces)
                 + relation_questions(space, detail, universe, surfaces)
                 + context_questions(space, known, universe, surfaces))
        return tuple(question for question in found if question.informative)
    return pool


@dataclass(frozen=True)
class Diagnosis:
    state: str
    questions: tuple[Question, ...]
    informative: int   # ignoring access entirely
    available: int     # after the filter
    unlockable: bool   # would the same planner, with full access, finish it?


def diagnose(space, contexts, universe, surfaces, access: Access,
             truth: World | None = None, budget: int = 8) -> Diagnosis:
    everything = joint_questions(space, contexts, universe, surfaces)
    reachable = accessible_pool(surfaces, access)(space, contexts, universe)

    if not space:
        return Diagnosis(CONTRADICTION, (), len(everything), len(reachable), False)
    if len(space) <= 1:
        return Diagnosis(RESOLVED, (), len(everything), len(reachable), True)

    choice = choose(space, contexts, universe,
                    ask_pool=accessible_pool(surfaces, access))
    if choice.state != IMPOSSIBLE:
        return Diagnosis(choice.state, choice.questions,
                         len(everything), len(reachable), True)

    # Nothing reachable finishes the job. Whether that is a wall or a door is
    # settled by running the same planner with the restriction lifted, not by
    # counting questions -- a question can be available and still lead nowhere.
    unlocked = False
    if truth is not None:
        session = run_accessible(space, contexts, universe, surfaces, truth,
                                 Access.full(contexts), budget)
        unlocked = session.resolved
    else:
        unlocked = choose(space, contexts, universe,
                          ask_pool=accessible_pool(surfaces,
                                                   Access.full(contexts))
                          ).state != IMPOSSIBLE
    return Diagnosis(BLOCKED if unlocked else IMPOSSIBLE, (),
                     len(everything), len(reachable), unlocked)


@dataclass(frozen=True)
class AccessSession:
    state: str
    asked: tuple[Question, ...]
    remaining: frozenset
    truth: World
    access: Access
    contexts: tuple[str, ...]

    @property
    def turns(self) -> int:
        return len(self.asked)

    @property
    def resolved(self) -> bool:
        return len(self.remaining) == 1

    @property
    def wrong(self) -> bool:
        return self.resolved and next(iter(self.remaining)) != self.truth

    @property
    def unaskable_asked(self) -> int:
        """Audited against ``askable`` directly, not against the pool used."""
        return sum(1 for question in self.asked
                   if not askable(question, self.contexts, self.access))


def run_accessible(space, contexts, universe, surfaces, truth: World,
                   access: Access, budget: int = 8) -> AccessSession:
    asked: list[Question] = []
    state = CONTRADICTION if not space else RESOLVED
    while True:
        if not space:
            state = CONTRADICTION
            break
        if len(space) <= 1:
            state = RESOLVED
            break
        choice = choose(space, contexts, universe,
                        ask_pool=accessible_pool(surfaces, access))
        state = choice.state
        if choice.state in (RESOLVED, IMPOSSIBLE):
            break
        if len(asked) >= budget:
            break
        question = choice.questions[0]
        space = answer(space, question, truth)
        asked.append(question)
    return AccessSession(state, tuple(asked), space, truth, access, tuple(contexts))


def cheapest_unavailable(space, contexts, universe, surfaces,
                         access: Access) -> tuple[Question, ...]:
    """Informative questions the state rules out, kept so their cost can be shown.

    The point of keeping them is the control: an unavailable question may be
    *cheaper* than the one that gets asked, and if accessibility had been folded
    into the cost vector that could never happen.
    """
    reachable = {q.text for q in accessible_pool(surfaces, access)(
        space, contexts, universe)}
    return tuple(q for q in joint_questions(space, contexts, universe, surfaces)
                 if q.text not in reachable)
