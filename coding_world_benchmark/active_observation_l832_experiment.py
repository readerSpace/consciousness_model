"""L8.32: stop collecting evidence and start choosing it -- or stop entirely.

L8.31 can say, before any learning, that a word's meaning sits in a class of two
and which single observation would split it.  That makes the next step a
planning problem rather than a data problem: not "gather more" but "gather
*this*", and -- the half that matters more -- "gathering anything at all is
pointless here, so stop".

The objective is written over the candidate set rather than over entropy,
because that is what the stopping rule needs to read::

    o* = argmin_o  E[ |C after o| ]

Under a uniform prior on the survivors this ranks the same way an information
gain would, and it says something the entropy form does not: a query whose two
outcomes do not split ``C`` leaves ``E[|C|] = |C|`` exactly.  So "useless" is not
a threshold on a score, it is an equality, and a state where *every* available
query is useless is ``STRUCTURALLY_UNRESOLVABLE`` -- provable in one pass over the
pool, before a single question is asked.

**Asking is stronger than being told, and the difference is measurable.**  L8.31's
``reachable`` models the passive regime the earlier layers lived in: an
observation only arrives when the meaning satisfies it, so a meaning can never be
narrowed by the observations it fails.  A *query* is different -- it has two
outcomes and the negative one is evidence too.  So a word that is only
PARTIALLY_IDENTIFIABLE passively can be IDENTIFIABLE actively, and the hold-out
measures that gap rather than asserting it.  What does not change is the twins:
their signatures are equal, so both branches of every query behave alike and no
amount of asking separates them.  Active querying buys reach, never a miracle.

**Greedy is checked against optimal, not assumed to be it.**  Minimising the
expected size one step at a time is the standard move and is known not to be
optimal in general, so for small worlds the whole decision tree is enumerated and

    pi* = argmin_pi  E[ T_singleton ]

computed exactly by memoised recursion.  ``greedy_vs_optimal_regret`` is then a
measurement rather than a hope.  "We used information gain and it worked" is a
weaker claim than "we used it and it cost this much".

The invariant is ``false_resolution_rate``.  Both branches of a query keep the
true meaning -- the truth satisfies the query or it does not, and either way it
survives into the branch that actually happened -- so the candidate set is an
invariant containing the answer, and declaring a singleton can only ever declare
the right one.  That is the same structure as every safety argument since L8.18:
the mechanism cannot produce the bad outcome, rather than being unlikely to.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from .cross_context_identification_l829_experiment import (
    FieldFacts, Observation, available_probes, constrain,
)
from .semantic_identifiability_l831_experiment import effect, reachable
from .typed_operation_l819_experiment import ValueType

RESOLVED = "RESOLVED"
RESOLVING = "RESOLVING"
STRUCTURALLY_UNRESOLVABLE = "STRUCTURALLY_UNRESOLVABLE"


def branches(
    query: Observation, candidates: frozenset[str], universe: tuple[FieldFacts, ...]
) -> tuple[frozenset[str], frozenset[str]]:
    """What ``candidates`` becomes on each answer.

    Both branches are kept because both are evidence.  This is the whole
    difference from the passive regime, where only the ``yes`` branch can ever
    occur and a meaning cannot be narrowed by what it fails.
    """
    survives = constrain(query, universe)
    return candidates & survives, candidates - survives


def answer_as_observation(query: Observation, answer: bool) -> Observation:
    """The observation a yes/no answer actually asserts.

    A NO is not "nothing happened" -- it asserts the complement, and feeding it
    back as anything else is how an active loop manufactures a wrong meaning.
    That is not hypothetical: an earlier draft approximated the negative branch
    with a hand-picked CONTRAST and resolved 「フガ率」 to the wrong field, which is
    precisely the ``false_resolution_rate`` this layer claims is structurally
    impossible. It is impossible only because the negation is exact.
    """
    if answer:
        return query
    return Observation("NOT", query, f"not ({query.kind} {query.payload})")


def useful(
    query: Observation, candidates: frozenset[str], universe: tuple[FieldFacts, ...]
) -> bool:
    """Does asking change anything?  An equality, not a threshold."""
    yes, no = branches(query, candidates, universe)
    return bool(yes) and bool(no)


def expected_size(
    query: Observation, candidates: frozenset[str], universe: tuple[FieldFacts, ...]
) -> float:
    """E[|C after o|] under a uniform prior over the survivors."""
    if not candidates:
        return 0.0
    yes, no = branches(query, candidates, universe)
    total = len(candidates)
    return (len(yes) ** 2 + len(no) ** 2) / total


def plan(
    candidates: frozenset[str],
    universe: tuple[FieldFacts, ...],
    pool: tuple[Observation, ...],
) -> Observation | None:
    """The query that shrinks the candidate set most in expectation.

    Ties are broken by position in the pool so a run is reproducible, and
    ``None`` means no query in the pool splits ``candidates`` at all -- which the
    caller must read as a stopping condition and not as a failure to search
    harder.
    """
    scored = [
        (expected_size(query, candidates, universe), index, query)
        for index, query in enumerate(pool)
        if useful(query, candidates, universe)
    ]
    if not scored:
        return None
    return min(scored, key=lambda item: (item[0], item[1]))[2]


# --------------------------------------------------------------------------
# the exact optimum, for worlds small enough to enumerate
# --------------------------------------------------------------------------

def optimal_depth(
    candidates: frozenset[str],
    universe: tuple[FieldFacts, ...],
    pool: tuple[Observation, ...],
) -> float:
    """E[T] under the best possible policy, by memoised recursion over the tree.

    Exponential in the worst case, which is why the hold-out only runs it on
    small worlds.  ``inf`` means no policy resolves this set -- the same fact
    ``plan`` reports as ``None``, reached by a different route, which is what
    makes one a check on the other.
    """
    memo: dict[frozenset[str], float] = {}

    def solve(current: frozenset[str]) -> float:
        if len(current) <= 1:
            return 0.0
        if current in memo:
            return memo[current]
        memo[current] = float("inf")  # guards the cycle a useless query would make
        best = float("inf")
        for query in pool:
            yes, no = branches(query, current, universe)
            if not yes or not no:
                continue
            cost = 1.0 + (len(yes) * solve(yes) + len(no) * solve(no)) / len(current)
            best = min(best, cost)
        memo[current] = best
        return best

    return solve(candidates)


# --------------------------------------------------------------------------
# running the loop
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Step:
    query: Observation
    answer: bool
    before: frozenset[str]
    after: frozenset[str]
    predicted_size: float


@dataclass(frozen=True)
class ActiveRun:
    truth: str
    state: str
    meaning: str | None
    steps: tuple[Step, ...]
    #: Queries that did not shrink the candidate set.  Measured rather than
    #: assumed: ``useful`` is supposed to make this impossible, and a number that
    #: is only ever zero because nobody looked is not evidence of anything.
    wasted: int

    @property
    def resolved(self) -> bool:
        return self.state == RESOLVED

    @property
    def wrong(self) -> bool:
        return self.resolved and self.meaning != self.truth


def run_active(
    truth: str,
    universe: tuple[FieldFacts, ...],
    pool: tuple[Observation, ...],
    start: frozenset[str] | None = None,
    budget: int = 64,
) -> ActiveRun:
    """Ask, answer from the world, narrow, repeat -- or stop and say why.

    The planner never sees ``truth``; it only receives the answer to the question
    it chose, exactly as it would from a person or a log.
    """
    candidates = start if start is not None else frozenset(f.name for f in universe)
    steps: list[Step] = []
    while len(candidates) > 1 and len(steps) < budget:
        query = plan(candidates, universe, pool)
        if query is None:
            return ActiveRun(truth, STRUCTURALLY_UNRESOLVABLE, None, tuple(steps),
                             sum(1 for step in steps
                                 if len(step.after) == len(step.before)))
        predicted = expected_size(query, candidates, universe)
        answer = effect(truth, query, universe)
        yes, no = branches(query, candidates, universe)
        after = yes if answer else no
        steps.append(Step(query, answer, candidates, after, predicted))
        candidates = after

    wasted = sum(1 for step in steps if len(step.after) == len(step.before))
    if len(candidates) == 1:
        return ActiveRun(truth, RESOLVED, next(iter(candidates)), tuple(steps), wasted)
    return ActiveRun(truth, RESOLVING, None, tuple(steps), wasted)


def passive_reach(
    truth: str, universe: tuple[FieldFacts, ...], pool: tuple[Observation, ...]
) -> frozenset[str]:
    """What L8.31 says the passive regime can reach, for comparison."""
    return reachable(truth, universe, pool)


# --------------------------------------------------------------------------
# asking it out loud
# --------------------------------------------------------------------------

def render_query(query: Observation, surface: str) -> str:
    """The planned observation as a question a person could answer.

    Not decoration: the point of computing a minimal distinguishing set is that
    it can be *asked*, and a request for "more detail" is not the same act as a
    request for the one fact that would settle it.
    """
    if query.kind == "RANGE":
        return f"「{surface}」が {query.payload} 以上になった例はありますか？"
    if query.kind == "CONTRAST":
        # The constraint is "not X", so the yes-branch is the exclusion.
        return f"「{surface}」は {query.payload} とは別の指標ですか？"
    if query.kind == "TYPE":
        name = query.payload.value if isinstance(query.payload, ValueType) else query.payload
        return f"「{surface}」は {name} の値ですか？"
    return f"「{surface}」について {query.kind}({query.payload}) は成り立ちますか？"


def question_for(
    candidates: frozenset[str],
    universe: tuple[FieldFacts, ...],
    pool: tuple[Observation, ...],
    surface: str,
) -> tuple[str, Observation | None]:
    """What to ask next, in words -- or why there is nothing worth asking."""
    query = plan(candidates, universe, pool)
    if query is None:
        names = "、".join(sorted(candidates))
        return (f"「{surface}」は {names} のどれかですが、"
                f"今の観測語彙ではこれ以上絞れません。", None)
    return render_query(query, surface), query
