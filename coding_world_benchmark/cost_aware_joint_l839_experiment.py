"""L8.39: three kinds of question over a context-conditioned joint, one rule.

L8.38 left the belief as one set per context, ``B_c subset of M^n``, and showed
that the context can change nothing but the *correspondence*.  Asking about that
is the job here, and the question space splits three ways:

* **unary** -- 「sim の場面で フガ率 は runtime_seconds のことですか？」
* **relation** -- 「sim の場面で フガ率 は ホゲ尺 より大きいですか？」
* **context / structure** -- 「この対応は sim と lab で同じですか？」

No kind is given priority.  A question is scored by what it leaves standing over
the possible worlds

    W = { w : C -> M^n  |  w(c) in B_c }

and chosen by L8.36's rule, unchanged and literally the same code: solvability is
a **veto**, then expected remaining ambiguity, then Pareto-minimal cost with the
cost still a vector.  ``ask_pool`` is the only thing L8.36 gained, and all it does
is let this layer hand in a different space of questions.

**What separates the kinds is not price, it is what they are able to name.**  A
unary question has to name a meaning; the project has modelled since L8.29 that
some fields cannot be named (``FieldFacts.nameable``), and a word is coined
precisely when the person has no name for the quantity.  A relation question
names only the two surfaces the person already used, and a context question names
only the episodes.  So in a world whose meanings are unnameable the unary pool is
*empty* -- not merely expensive -- while the other two still work.  That is where
``Q_unary informative = 0, Q_relation informative > 0`` comes from, and it is a
fact about what can be asked rather than a scoring trick.

The negative half is stated in the same breath, because it bounds the claim.  If
every meaning *is* nameable, a world with more than one possibility and no
informative unary question **cannot exist**:

    every unary question uninformative
      <=> |pi_i(B_c)| = 1 for every context c and index i
      ==> B_c subset of the product of its own singleton marginals, so |B_c| = 1
      ==> |W| = 1.

``search_unary_blind`` looks for a counterexample exhaustively over the small
worlds and finds none, which is what makes the constructed zero above a
restriction with a reason rather than a rigged pool.

``false_joint_resolution_rate`` stays zero for L8.32's structural reason: the
outcomes of every question partition the worlds, the branch kept is the one
holding the answer actually given, so the truth never leaves the surviving set
and a singleton can only be it.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product

from .contextual_joint_l838_experiment import Assignment, ContextualJoint
from .cost_aware_dialogue_l836_experiment import (
    ASK, IMPOSSIBLE, MULTIPLE_OPTIMAL, RESOLVED, Choice, Cost, Question, answer,
    choose,
)
from .cross_context_identification_l829_experiment import FieldFacts
from .joint_inference_l834_experiment import (
    DISTINCT, EXCEEDS, SAME_TYPE, Relation, holds,
)
from .nway_semantics_l837_experiment import CONTRADICTION

UNARY, RELATION, CONTEXT = "UNARY", "RELATION", "CONTEXT"
ALL_KINDS = (UNARY, RELATION, CONTEXT)
BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"

#: One complete lexicon: an assignment for every context, in context order.
World = tuple[tuple[str, Assignment], ...]


def worlds(joint: ContextualJoint) -> frozenset[World]:
    """Every way of choosing one admissible assignment per context.

    The product rather than anything cleverer: L8.38 gives no constraint tying
    the contexts together, so refusing to enumerate their combinations would be
    assuming the coupling that the questions are there to find out about.
    """
    pools = [sorted(joint.belief(context)) for context in joint.contexts]
    if any(not pool for pool in pools):
        return frozenset()
    return frozenset(tuple(zip(joint.contexts, chosen))
                     for chosen in product(*pools))


def reading(world: World, context: str) -> Assignment:
    return dict(world)[context]


# --------------------------------------------------------------------------
# the three kinds of question
# --------------------------------------------------------------------------

def _split(space: frozenset[World], keep) -> tuple[frozenset[World], frozenset[World]]:
    yes = frozenset(world for world in space if keep(world))
    return yes, space - yes


def unary_questions(space: frozenset[World], contexts, universe, surfaces) -> list[Question]:
    """Questions that name a meaning -- and so can only ask about nameable ones."""
    named = {facts.name for facts in universe if facts.nameable}
    found: list[Question] = []
    for context in contexts:
        for index, surface in enumerate(surfaces):
            options = sorted({reading(world, context)[index] for world in space})
            askable = [name for name in options if name in named]
            for meaning in askable:
                yes, no = _split(space,
                                 lambda w, c=context, i=index, m=meaning:
                                 reading(w, c)[i] == m)
                found.append(Question(
                    UNARY,
                    f"「{context}」の場面で {surface} は {meaning} のことですか？",
                    Cost(1, 2, True), (("yes", yes), ("no", no))))
            # A choice question only partitions the space when every option can
            # be offered, so it is withheld as soon as one of them is unnameable.
            if len(askable) > 2 and len(askable) == len(options):
                found.append(Question(
                    UNARY,
                    f"「{context}」の場面で {surface} は "
                    f"{'／'.join(askable)} のどれですか？",
                    Cost(1, len(askable), True),
                    tuple((meaning,
                           frozenset(w for w in space
                                     if reading(w, context)[index] == meaning))
                          for meaning in askable)))
    return found


_RELATION_TEXT = {
    EXCEEDS: "{c} の場面で {a} は {b} より大きいですか？",
    SAME_TYPE: "{c} の場面で {a} と {b} は同じ種類の値ですか？",
    DISTINCT: "{c} の場面で {a} と {b} は別の指標ですか？",
}


def relation_questions(space: frozenset[World], contexts, universe, surfaces) -> list[Question]:
    """Questions about the pair, naming only the surfaces the person used."""
    found: list[Question] = []
    for context in contexts:
        for left, right in combinations(range(len(surfaces)), 2):
            for kind in (EXCEEDS, SAME_TYPE, DISTINCT):
                relation = Relation(kind, left, right)
                yes, no = _split(space,
                                 lambda w, c=context, r=relation:
                                 holds(r, reading(w, c), universe))
                found.append(Question(
                    RELATION,
                    _RELATION_TEXT[kind].format(
                        c=f"「{context}」", a=surfaces[left], b=surfaces[right]),
                    Cost(1, 2, True), (("yes", yes), ("no", no))))
    return found


def context_questions(space: frozenset[World], contexts, universe, surfaces) -> list[Question]:
    """Questions about the shape of the usage, naming neither meaning nor value.

    ``needs_context`` is false for the same reason L8.36 gave its structural
    questions: the person answers from how they use the word, not by recalling
    what a particular episode recorded.
    """
    found: list[Question] = []
    for one, other in combinations(contexts, 2):
        yes, no = _split(space,
                         lambda w, a=one, b=other: reading(w, a) == reading(w, b))
        found.append(Question(
            CONTEXT, f"この対応は「{one}」と「{other}」で同じですか？",
            Cost(1, 2, False), (("yes", yes), ("no", no))))
        for index, surface in enumerate(surfaces):
            yes, no = _split(space,
                             lambda w, a=one, b=other, i=index:
                             reading(w, a)[i] == reading(w, b)[i])
            found.append(Question(
                CONTEXT,
                f"{surface} の意味は「{one}」と「{other}」で同じですか？",
                Cost(1, 2, False), (("yes", yes), ("no", no))))
    if len(contexts) > 2:
        yes, no = _split(space,
                         lambda w: len({reading(w, c) for c in contexts}) == 1)
        found.append(Question(
            CONTEXT, "この対応はどの場面でも同じですか？",
            Cost(1, 2, False), (("yes", yes), ("no", no))))
    return found


_BUILDERS = {UNARY: unary_questions, RELATION: relation_questions,
             CONTEXT: context_questions}


def joint_questions(space, contexts, universe, surfaces,
                    kinds=ALL_KINDS) -> tuple[Question, ...]:
    found: list[Question] = []
    for kind in kinds:
        found += _BUILDERS[kind](space, contexts, universe, surfaces)
    return tuple(question for question in found if question.informative)


def pool_for(surfaces, kinds=ALL_KINDS):
    """A question space in the shape L8.36's chooser expects."""
    def pool(space, contexts, universe):
        return joint_questions(space, contexts, universe, surfaces, kinds)
    return pool


def informative_by_kind(space, contexts, universe, surfaces) -> dict[str, int]:
    return {kind: len(joint_questions(space, contexts, universe, surfaces, (kind,)))
            for kind in ALL_KINDS}


# --------------------------------------------------------------------------
# choosing and asking
# --------------------------------------------------------------------------

def choose_joint(space, contexts, universe, surfaces, kinds=ALL_KINDS) -> Choice:
    """L8.36's rule verbatim, over worlds instead of over lexical hypotheses.

    The one guard added is for the empty set: nothing to ask about is not the
    same as nothing left to ask, and reporting a contradiction as ``RESOLVED``
    would be the exact failure L8.34 built ``MARGINAL_ONLY`` to avoid.
    """
    if not space:
        return Choice(CONTRADICTION, (), space)
    return choose(space, contexts, universe, ask_pool=pool_for(surfaces, kinds))


@dataclass(frozen=True)
class Session:
    state: str
    asked: tuple[Question, ...]
    remaining: frozenset
    truth: World

    @property
    def turns(self) -> int:
        return sum(question.cost.turns for question in self.asked)

    @property
    def context_recalls(self) -> int:
        return sum(1 for question in self.asked if question.cost.needs_context)

    @property
    def kinds(self) -> tuple[str, ...]:
        return tuple(question.kind for question in self.asked)

    @property
    def resolved(self) -> bool:
        return len(self.remaining) == 1

    @property
    def wrong(self) -> bool:
        return self.resolved and next(iter(self.remaining)) != self.truth

    @property
    def uninformative_asked(self) -> int:
        return sum(1 for question in self.asked if not question.informative)


def run_joint_dialogue(space: frozenset[World], contexts, universe, surfaces,
                       truth: World, kinds=ALL_KINDS, budget: int = 8) -> Session:
    asked: list[Question] = []
    state = CONTRADICTION if not space else RESOLVED
    while True:
        choice = choose_joint(space, contexts, universe, surfaces, kinds)
        state = choice.state
        if choice.state in (RESOLVED, IMPOSSIBLE, CONTRADICTION):
            break
        if len(asked) >= budget:
            state = BUDGET_EXHAUSTED
            break
        question = choice.questions[0]
        space = answer(space, question, truth)
        asked.append(question)
    return Session(state, tuple(asked), space, truth)


# --------------------------------------------------------------------------
# the negative half: is a unary-blind world constructible at all?
# --------------------------------------------------------------------------

def unary_blind(space: frozenset[World], contexts, universe, surfaces) -> bool:
    return (len(space) > 1
            and not joint_questions(space, contexts, universe, surfaces, (UNARY,)))


def search_unary_blind(universe: tuple[FieldFacts, ...], surfaces=("a", "b"),
                       contexts=("c1", "c2"), max_set: int = 3) -> list:
    """Exhaustive hunt for a world with no informative unary question.

    Only over universes whose fields are all nameable -- with an unnameable one
    the answer is immediate and uninteresting.  The claim being checked is that
    the restriction L8.39 leans on is *necessary*, so a counterexample here would
    retire the headline rather than decorate it.
    """
    assert all(facts.nameable for facts in universe)
    names = [facts.name for facts in universe]
    everything = list(product(names, repeat=len(surfaces)))
    subsets = [frozenset(choice)
               for size in range(1, max_set + 1)
               for choice in combinations(everything, size)]
    found = []
    for chosen in product(subsets, repeat=len(contexts)):
        joint = ContextualJoint(surfaces, tuple(zip(contexts, chosen)))
        space = worlds(joint)
        if unary_blind(space, contexts, universe, surfaces):
            found.append(joint)
    return found
