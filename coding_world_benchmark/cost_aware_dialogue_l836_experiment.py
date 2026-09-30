"""L8.36: pick the cheapest question that loses nothing, and say so when two tie.

L8.32 chose questions by how much they shrink the candidate set, which is right
and incomplete: a yes/no and a five-way choice are both "one question" and are
not the same thing to answer.  The obvious repair -- ``gain - lambda * cost`` --
smuggles in a constant nobody can justify, and this project has spent a dozen
layers removing exactly those.  So there is no lambda.  Questions are compared
**lexicographically**::

    q  |->  ( resolution capability,  expected remaining ambiguity,  cost )

First, do not ask something that makes a solvable problem unsolvable.  Second,
among those, shrink the hypothesis space most.  Only where those tie does cost
decide -- and cost is not a number either.  It stays a vector::

    C(q) = ( turns,  |answer space|,  needs the user to recall a context )

compared by Pareto dominance.  A three-way choice answered in one turn and two
yes/no questions are genuinely incomparable, and inventing a weight to rank them
would be inventing the answer.  When several questions are non-dominated the
planner reports ``MULTIPLE_OPTIMAL`` and hands them over rather than picking.

**The new thing L8.35 makes possible is a question about the model.**  Until now
every question asked what a word *means*.  In L8.35's ``AMBIGUOUS`` state -- where
time and context explain the same split -- no question about meaning can help at
all, because every surviving hypothesis agrees about every meaning that was
observed.  What they disagree about is *which variable explains the change*.
Asking that directly settles it in one turn::

    「この語の意味は、時期によって変わりましたか、それとも場面によって違いますか？」

So the target of questioning widens from meaning to model structure, and the
capability term is what expresses it: meaning questions there have capability
zero, so they lose on the first comparison rather than on price.  That is a
stronger statement than "the structural question is cheaper".

``false_resolution_rate`` remains the invariant, for the same structural reason
as in L8.32: every branch of every question retains the hypotheses consistent
with the answer that was actually given, so the surviving set always contains the
truth and a singleton can only be the right one.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

from .contextual_polysemy_l835_experiment import (
    MONOSEMY, NOISE, POLYSEMY, REVISION, ContextualObservation, _live,
)
from .cross_context_identification_l829_experiment import FieldFacts, constrain

RESOLVED = "RESOLVED"
ASK = "ASK"
MULTIPLE_OPTIMAL = "MULTIPLE_OPTIMAL"
IMPOSSIBLE = "IMPOSSIBLE"

MEANING, STRUCTURE = "MEANING", "STRUCTURE"


@dataclass(frozen=True)
class Cost:
    """What answering costs, kept as a vector because it is not one quantity.

    ``turns`` is how many exchanges it takes, ``answer_space`` how many options
    the person has to choose between, and ``needs_context`` whether answering
    requires recalling a particular episode rather than answering from the
    request in front of them.
    """

    turns: int
    answer_space: int
    needs_context: bool

    @property
    def vector(self) -> tuple[int, int, int]:
        return (self.turns, self.answer_space, int(self.needs_context))

    def dominates(self, other: "Cost") -> bool:
        mine, theirs = self.vector, other.vector
        return all(a <= b for a, b in zip(mine, theirs)) and mine != theirs


@dataclass(frozen=True)
class Hypothesis:
    """One complete account of the evidence: a model kind, its senses, and what
    it predicts for each context that was actually observed.

    ``predictions`` is the part that matters for planning and it is computed from
    the evidence rather than from the model's own labels.  Under REVISION the
    meaning depends on *when*, so what it predicts for a context is whatever it
    says about the times that context appeared at -- and when a context happens to
    sit entirely on one side of the change, REVISION and POLYSEMY predict exactly
    the same thing everywhere.  That identity is the reason no question about
    meaning can separate them, and storing the predictions is what lets the
    planner see it instead of being told.
    """

    model: str  # MONOSEMY | NOISE | REVISION | POLYSEMY
    senses: tuple[tuple[str, str], ...]  # (block label, meaning), sorted
    predictions: tuple[tuple[str, str], ...] = ()  # (context, meaning), sorted

    @property
    def sense_map(self) -> dict[str, str]:
        return dict(self.senses)

    def meaning_in(self, context: str) -> str | None:
        return dict(self.predictions).get(context)


def _blocks(evidence, key) -> dict[str, tuple[int, ...]]:
    found: dict[str, list[int]] = {}
    for index, item in enumerate(evidence):
        found.setdefault(key(item), []).append(index)
    return {label: tuple(indices) for label, indices in found.items()}


def _predict(evidence, blocks: dict[str, tuple[int, ...]],
             senses: dict[str, str]) -> tuple[tuple[str, str], ...]:
    """What this account says about each observed context.

    A context inherits the meaning of the block its observations fall in.  A
    context spread across blocks that disagree predicts nothing for it, which is
    the honest answer and keeps such a context out of the meaning questions.
    """
    by_context: dict[str, set[str]] = {}
    for label, indices in blocks.items():
        for index in indices:
            by_context.setdefault(evidence[index].context, set()).add(senses[label])
    return tuple(sorted((context, next(iter(values)))
                        for context, values in by_context.items()
                        if len(values) == 1))


def hypotheses(evidence: tuple[ContextualObservation, ...],
               universe: tuple[FieldFacts, ...],
               min_block: int = 2) -> frozenset[Hypothesis]:
    """Every account that fits the evidence, not only the one L8.35 would pick.

    L8.35 selects; planning needs the space it selected from, because a question
    is worth asking exactly insofar as it separates accounts that are still
    standing.
    """
    found: set[Hypothesis] = set()

    contexts = tuple(sorted({item.context for item in evidence}))

    def flat(model: str, meaning: str) -> Hypothesis:
        return Hypothesis(model, (("*", meaning),),
                          tuple((context, meaning) for context in contexts))

    live = _live(evidence, universe)
    for meaning in live:
        found.add(flat(MONOSEMY, meaning))

    total = len(evidence)
    for dropped in range(1, max(1, total - min_block + 1)):
        for kept in combinations(range(total), total - dropped):
            survivors = _live(tuple(evidence[i] for i in kept), universe)
            if len(survivors) == 1:
                found.add(flat(NOISE, next(iter(survivors))))
        if any(h.model == NOISE for h in found):
            break

    for label, key in ((REVISION, None), (POLYSEMY, "context")):
        if label == POLYSEMY:
            blocks = _blocks(evidence, lambda item: item.context)
            candidates = [blocks]
        else:
            times = sorted({item.time for item in evidence})
            candidates = [
                _blocks(evidence,
                        lambda item, cut=cut: "before" if item.time < cut else "after")
                for cut in times[1:]
            ]
        for blocks in candidates:
            if len(blocks) < 2 or any(len(i) < min_block for i in blocks.values()):
                continue
            senses = {}
            ok = True
            for name, indices in blocks.items():
                survivors = _live(tuple(evidence[i] for i in indices), universe)
                if len(survivors) != 1:
                    ok = False
                    break
                senses[name] = next(iter(survivors))
            if ok and len(set(senses.values())) > 1:
                found.add(Hypothesis(label, tuple(sorted(senses.items())),
                                     _predict(evidence, blocks, senses)))

    # L8.30's rule, applied to the space rather than to one belief: an account
    # that discards observations is strictly worse than one that explains all of
    # them, so a NOISE reading survives only while no conditioning fits. Keeping
    # it otherwise would let a worse account dilute every question's value.
    if any(h.model in (REVISION, POLYSEMY) for h in found):
        found = {h for h in found if h.model != NOISE}
    return frozenset(found)


# --------------------------------------------------------------------------
# questions
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Question:
    kind: str  # MEANING | STRUCTURE
    text: str
    cost: Cost
    #: label -> the hypotheses an answer of that label would leave standing.
    outcomes: tuple[tuple[str, frozenset[Hypothesis]], ...]

    @property
    def informative(self) -> bool:
        live = [block for _, block in self.outcomes if block]
        return len(live) > 1

    def expected_size(self) -> float:
        total = sum(len(block) for _, block in self.outcomes)
        if not total:
            return 0.0
        return sum(len(block) ** 2 for _, block in self.outcomes) / total


def _meaning_questions(space: frozenset[Hypothesis], contexts: tuple[str, ...],
                       meanings: tuple[str, ...]) -> list[Question]:
    found: list[Question] = []
    for context in contexts:
        for meaning in meanings:
            yes = frozenset(h for h in space if h.meaning_in(context) == meaning)
            no = space - yes
            found.append(Question(
                MEANING, f"「{context}」の場面では {meaning} のことですか？",
                Cost(1, 2, True), (("yes", yes), ("no", no))))
        options = tuple(sorted({h.meaning_in(context) for h in space
                                if h.meaning_in(context)}))
        if len(options) > 2:
            found.append(Question(
                MEANING,
                f"「{context}」の場面では {'／'.join(options)} のどれですか？",
                Cost(1, len(options), True),
                tuple((option, frozenset(h for h in space
                                         if h.meaning_in(context) == option))
                      for option in options)))
    return found


def _structure_questions(space: frozenset[Hypothesis]) -> list[Question]:
    """Questions about which variable explains the evidence, not about meanings."""
    models = tuple(sorted({h.model for h in space}))
    if len(models) < 2:
        return []
    found = [Question(
        STRUCTURE,
        "この語の意味は、時期によって変わりましたか、それとも場面によって違いますか？",
        Cost(1, len(models), True),
        tuple((model, frozenset(h for h in space if h.model == model))
              for model in models))]
    if REVISION in models:
        changed = frozenset(h for h in space if h.model == REVISION)
        found.append(Question(
            STRUCTURE, "この語の意味は途中で変わりましたか？",
            Cost(1, 2, False), (("yes", changed), ("no", space - changed))))
    if POLYSEMY in models:
        split = frozenset(h for h in space if h.model == POLYSEMY)
        found.append(Question(
            STRUCTURE, "この語は場面によって別の意味になりますか？",
            Cost(1, 2, False), (("yes", split), ("no", space - split))))
    return found


def all_questions(space: frozenset[Hypothesis], contexts, meanings) -> tuple[Question, ...]:
    return tuple(q for q in _meaning_questions(space, contexts, meanings)
                 + _structure_questions(space) if q.informative)


# --------------------------------------------------------------------------
# lexicographic choice, with the cost left as a vector
# --------------------------------------------------------------------------

def preserves_solvability(question: Question, contexts, meanings,
                          ask_pool=None) -> bool:
    """Does every answer leave a state that can still be finished?

    ``ask_pool`` lets a later layer supply a different question space without
    touching the rule; L8.39 passes its own and the comparison below is then
    literally the same code, which is the point.

    This is the first comparison and it is a **veto, not an optimisation**.  A
    question that is informative and leads to a branch nothing can narrow has
    spent a turn to arrive somewhere permanently stuck, and no amount of
    cheapness redeems that.  It is also how the structural questions win in
    L8.35's AMBIGUOUS state: meaning questions there leave POLYSEMY and REVISION
    together forever, because the two predict the same meaning for every context
    that was observed.
    """
    for _, block in question.outcomes:
        if len(block) <= 1:
            continue
        if not (ask_pool or all_questions)(block, contexts, meanings):
            return False
    return True


@dataclass(frozen=True)
class Choice:
    state: str
    questions: tuple[Question, ...]
    space: frozenset[Hypothesis]

    @property
    def question(self) -> Question | None:
        return self.questions[0] if len(self.questions) == 1 else None


def choose(space: frozenset[Hypothesis], contexts, meanings,
           ask_pool=None) -> Choice:
    """Capability, then ambiguity, then Pareto-minimal cost. No weights anywhere."""
    if len(space) <= 1:
        return Choice(RESOLVED, (), space)
    pool = (ask_pool or all_questions)(space, contexts, meanings)
    if not pool:
        return Choice(IMPOSSIBLE, (), space)

    strongest = [q for q in pool
                 if preserves_solvability(q, contexts, meanings, ask_pool)]
    if not strongest:
        # Nothing finishes the job. Say so rather than spending a turn on the
        # least bad option and arriving stuck anyway.
        return Choice(IMPOSSIBLE, (), space)

    least = min(q.expected_size() for q in strongest)
    tied = [q for q in strongest if abs(q.expected_size() - least) < 1e-9]

    cheapest = [q for q in tied
                if not any(other.cost.dominates(q.cost) for other in tied)]
    # Identical costs are not competing accounts of the same question; keep one
    # of each distinct cost/text pair so MULTIPLE_OPTIMAL means what it says.
    unique: dict[tuple, Question] = {}
    for question in cheapest:
        unique.setdefault((question.cost.vector, question.text), question)
    remaining = tuple(unique.values())
    if len(remaining) == 1:
        return Choice(ASK, remaining, space)
    return Choice(MULTIPLE_OPTIMAL, remaining, space)


def answer(space: frozenset[Hypothesis], question: Question,
           truth: Hypothesis) -> frozenset[Hypothesis]:
    for _, block in question.outcomes:
        if truth in block:
            return block
    return frozenset()


def run_dialogue(space: frozenset[Hypothesis], contexts, meanings,
                 truth: Hypothesis, budget: int = 8):
    """Ask until one hypothesis stands, or until nothing left would separate them."""
    asked: list[Question] = []
    while len(space) > 1 and len(asked) < budget:
        choice = choose(space, contexts, meanings)
        if choice.state in (IMPOSSIBLE, RESOLVED):
            break
        question = choice.questions[0]
        space = answer(space, question, truth)
        asked.append(question)
    return space, tuple(asked)


def total_cost(questions: tuple[Question, ...]) -> Cost:
    return Cost(sum(q.cost.turns for q in questions),
                max((q.cost.answer_space for q in questions), default=0),
                any(q.cost.needs_context for q in questions))
