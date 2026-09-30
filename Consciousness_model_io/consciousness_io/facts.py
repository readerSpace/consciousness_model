"""Symbolic facts shared by observation, belief, prediction, goal, and question sets.

A :class:`Fact` is the single element type flowing through the set-theoretic
consciousness state ``C_t = (O_t, B_t, P_t, G_t, Q_t, A_t, W_t)`` described in the
design note.  A fact is a predicate applied to a tuple of arguments, e.g.::

    Fact("ON", ("cup", "table"))
    Fact("AT", ("cup", (1.2, 0.3, 0.8)))
    Fact("state", ("cup", "movable"))

Arguments must be hashable so that observation, belief, and goal *sets* behave
like real mathematical sets (membership, union, intersection).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Hashable, Iterable, Sequence, Tuple


@dataclass(frozen=True)
class Fact:
    """One predicate ``p(a_1, ..., a_n)`` with a scalar confidence in [0, 1]."""

    predicate: str
    args: Tuple[Hashable, ...] = ()
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if not self.predicate:
            raise ValueError("predicate must be non-empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")

    @property
    def key(self) -> Tuple[str, Hashable]:
        """Identity used when a newer observation should overwrite an older one.

        By default a fact is keyed by ``(predicate, first_argument)`` so that a
        fresh ``AT(cup, ...)`` replaces the previous position of ``cup`` instead
        of accumulating stale duplicates.  Facts with no arguments are keyed by
        the predicate alone.
        """

        return (self.predicate, self.args[0] if self.args else None)

    def to_relation(self) -> Tuple[str, str, str]:
        """Project the fact onto a ``(subject, verb, object)`` relation.

        This is the bridge to :class:`~consciousness_io.compressed_workspace.CompressedWorkspace`,
        whose MDL-style compression works on string triples.
        """

        if len(self.args) >= 2:
            return (str(self.args[0]), self.predicate, str(self.args[1]))
        if len(self.args) == 1:
            return (str(self.args[0]), "is", self.predicate)
        return (self.predicate, "is", "true")

    def tokens(self) -> Tuple[str, ...]:
        return (self.predicate,) + tuple(str(arg) for arg in self.args)

    def __str__(self) -> str:
        if not self.args:
            return self.predicate
        return f"{self.predicate}(" + ",".join(str(arg) for arg in self.args) + ")"


def fact(predicate: str, *args: Hashable, confidence: float = 1.0) -> Fact:
    """Terse constructor: ``fact("ON", "cup", "table")``."""

    return Fact(predicate, tuple(args), confidence)


KeyFn = Callable[[Fact], Hashable]


class BeliefStore:
    """A set of facts that merges by :pyattr:`Fact.key` (newest observation wins).

    Implements ``B_{t+1} = B_t ∪ f(O_t)`` with overwrite semantics so that the
    belief set stays finite even under a stream of observations of the same
    entity.  A custom ``key_fn`` lets an application decide what counts as "the
    same fact".
    """

    def __init__(self, key_fn: KeyFn | None = None) -> None:
        self._key_fn = key_fn or (lambda f: f.key)
        self._facts: dict[Hashable, Fact] = {}

    def update(self, observations: Iterable[Fact]) -> "BeliefStore":
        for observation in observations:
            self._facts[self._key_fn(observation)] = observation
        return self

    def discard(self, predicate: str, first_arg: Hashable | None = None) -> None:
        self._facts.pop((predicate, first_arg), None)

    def satisfies(self, goal: Fact) -> bool:
        """Whether ``goal`` is entailed: same key present with matching args."""

        held = self._facts.get(self._key_fn(goal))
        return held is not None and held.predicate == goal.predicate and held.args == goal.args

    def as_set(self) -> frozenset[Fact]:
        return frozenset(self._facts.values())

    def copy(self) -> "BeliefStore":
        clone = BeliefStore(self._key_fn)
        clone._facts = dict(self._facts)
        return clone

    def __iter__(self):
        return iter(self._facts.values())

    def __len__(self) -> int:
        return len(self._facts)

    def __contains__(self, item: Fact) -> bool:
        return self.satisfies(item)


def goal_distance(beliefs: BeliefStore | frozenset[Fact], goals: Sequence[Fact]) -> float:
    """``D(B, G)`` — number of goal facts not entailed, weighted by confidence.

    A satisfied goal contributes 0; an unmet goal contributes ``goal.confidence``
    (default 1).  This is the metric minimised by goal-directed action selection
    ``a* = argmin_a D(P(a, B_t), G_t)``.
    """

    if isinstance(beliefs, frozenset):
        store = BeliefStore().update(beliefs)
    else:
        store = beliefs
    return sum(goal.confidence for goal in goals if not store.satisfies(goal))
