"""L8.38: the allowed *combinations* of meanings depend on the context.

L8.35 conditioned one word's meaning on the episode.  L8.37 solved n words
together.  Putting them side by side gives "each word is polysemous, and then
solve the tuple", which is a sum of the two.  The thing that is not a sum is
this::

    B_c  subset of  M^n

where the **set of admissible assignments** is what the context selects.  A world
can then have, for two surfaces and meanings {a, b} x {x, y}::

    B_A = { (a,x), (b,y) }        B_B = { (a,y), (b,x) }

Every single-word marginal is *identical* across the two contexts -- the first
surface is {a, b} either way, the second is {x, y} either way -- and the sets are
different.  What the context changes is only the **correspondence**, and a
per-word account of polysemy is blind to it by construction: looking at either
word alone there is no context effect to find.

One consequence is worth stating because it constrains what can be claimed.
Identical unary marginals and ``|B_c| = 1`` are **incompatible**: a singleton's
marginals *are* its assignment, so two contexts with identical marginals and one
solution each are the same context.  The identity therefore holds at the level of
the belief sets, and what the layer has to show is that the difference between
them is *usable* -- that one further observation, the same one in both contexts,
lands on different meanings.  That is the shape of the hold-out's decisive
family, and it is a real constraint on the criterion rather than a convenience.

Everything else is inherited.  Execution is gated per context on ``|B_c| = 1``;
zero is a contradiction, two or more abstains.  No threshold is added, which is
the same claim L8.37 made about n and for the same reason.
"""
from __future__ import annotations

from dataclasses import dataclass

from .cross_context_identification_l829_experiment import FieldFacts, constrain
from .nway_semantics_l837_experiment import (
    AMBIGUOUS, CONTRADICTION, RESOLVED, verdict,
)

Assignment = tuple[str, ...]

#: What the context turns out to be doing, once the sets are compared.
NO_EFFECT = "NO_EFFECT"
UNARY_EFFECT = "UNARY_EFFECT"        # some word's own candidate set differs
RELATIONAL_EFFECT = "RELATIONAL_EFFECT"  # only the correspondence differs


@dataclass(frozen=True)
class ContextualJoint:
    """One belief set per context, over the same surfaces."""

    surfaces: tuple[str, ...]
    sets: tuple[tuple[str, frozenset[Assignment]], ...]

    @property
    def contexts(self) -> tuple[str, ...]:
        return tuple(label for label, _ in self.sets)

    def belief(self, context: str) -> frozenset[Assignment]:
        return dict(self.sets).get(context, frozenset())

    def verdict_in(self, context: str) -> str:
        return verdict(sorted(self.belief(context)))

    def meanings_in(self, context: str) -> dict[str, str] | None:
        live = self.belief(context)
        if len(live) != 1:
            return None
        return dict(zip(self.surfaces, next(iter(live))))

    # -- what the context is doing -------------------------------------
    def marginal(self, context: str, index: int) -> frozenset[str]:
        return frozenset(a[index] for a in self.belief(context))

    def marginals(self, context: str) -> tuple[frozenset[str], ...]:
        return tuple(self.marginal(context, i) for i in range(len(self.surfaces)))

    @property
    def marginals_agree(self) -> bool:
        """Do all single-word candidate sets match across every context?

        When they do, no account that looks at one word at a time can see the
        context at all -- which is exactly what makes the relational case
        irreducible to L8.35 applied n times.
        """
        found = {self.marginals(context) for context in self.contexts}
        return len(found) == 1

    @property
    def sets_agree(self) -> bool:
        return len({self.belief(context) for context in self.contexts}) == 1

    @property
    def effect(self) -> str:
        if self.sets_agree:
            return NO_EFFECT
        return RELATIONAL_EFFECT if self.marginals_agree else UNARY_EFFECT

    # -- using it ------------------------------------------------------
    def observe(self, index: int, observation, universe: tuple[FieldFacts, ...],
                context: str | None = None) -> "ContextualJoint":
        """Narrow one context, or all of them when the observation is general."""
        admitted = constrain(observation, universe)
        return ContextualJoint(self.surfaces, tuple(
            (label, frozenset(a for a in live if a[index] in admitted)
             if context in (None, label) else live)
            for label, live in self.sets))

    def erased(self) -> frozenset[Assignment]:
        """The same evidence with the context labels removed.

        The union rather than the intersection: without labels there is nothing
        to say which episode an assignment belonged to, so every one of them is
        still on the table.
        """
        found: set[Assignment] = set()
        for _, live in self.sets:
            found |= live
        return frozenset(found)


def erased_verdict(joint: ContextualJoint) -> str:
    return verdict(sorted(joint.erased()))


def per_word_view(joint: ContextualJoint) -> dict[str, dict[str, frozenset[str]]]:
    """What an account that looks at one word at a time would see.

    Kept as its own function because the claim is about what this view *cannot*
    contain, and a claim like that is worth being able to print.
    """
    return {
        surface: {context: joint.marginal(context, index)
                  for context in joint.contexts}
        for index, surface in enumerate(joint.surfaces)
    }


def per_word_sees_context(joint: ContextualJoint) -> bool:
    return any(len({tuple(sorted(sets[context])) for context in joint.contexts}) > 1
               for sets in per_word_view(joint).values())
