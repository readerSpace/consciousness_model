"""L8.30: a meaning that was learned has to be losable, and losing it safely is
not the same as changing your mind quickly.

L8.29 closed the loop -- an unknown word narrowed across contexts, acquired when
the candidate set became a singleton, and allowed to fill an argument the graph
had refused.  The dangerous half of that is what it did not do: once a belief
became a singleton it stayed true forever.  A word acquired from two contexts is
a conclusion drawn from two contexts, and the third can disagree.

The obvious repair is the wrong one.  Flipping the entry the moment a
contradiction appears trusts the newest observation most, and the newest
observation is exactly the one that has been corroborated least -- it may be a
misparse, an unusual phrasing, or a genuinely different word that happens to
share a substring.  So the rule this layer is built on is::

    contradiction != immediate remapping

A contradiction **suspends** rather than rewrites.  Authority to execute is
withdrawn in the same step it appears, and a new meaning is adopted only when
the evidence as a whole picks one, which is a different and later question.

The lexicon therefore becomes a state machine::

    UNKNOWN -> HYPOTHESIS -> ACQUIRED -> DISPUTED -> REVISED
                                              |  \\-> REVOKED
                                              \\----> (stays DISPUTED)

with only ACQUIRED and REVISED carrying execution authority.  DISPUTED and
REVOKED are both silent, so the fail-closed rule that has held since L8.18 now
holds over the *time evolution* of the lexicon and not only within a request.

**Revision without a threshold.**  When the observations cannot all be true at
once, something must be given up, and which thing is the whole question.  No
weights, no decay constant, no confidence number: the belief keeps every
observation with its provenance and asks which *maximal* subsets of them are
jointly satisfiable.  Among those, the ones discarding the fewest observations
are the cardinality-maximal repairs.

* one repair, one surviving candidate -> REVISED to it
* one repair, the old meaning ruled out but no unique replacement -> REVOKED
* several repairs that disagree -> DISPUTED, and the belief waits

That arithmetic gives the property the design was after without anyone
stipulating it.  A single contradicting observation against a single supporting
one is a tie -- two repairs of equal cost that disagree -- so it suspends and
cannot flip.  A second corroborating observation breaks the tie, because
discarding two costs more than discarding one.  **One disagreement suspends; two
agreeing disagreements revise.**  Nothing in the code counts to two.

One consequence worth stating plainly: a belief can be suspended forever.  Two
streams of evidence that contradict and never accumulate leave the word
DISPUTED, and that is the correct end state, not a failure to converge.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

from .cross_context_identification_l829_experiment import (
    FieldFacts, Observation, constrain,
)

UNKNOWN, HYPOTHESIS, ACQUIRED, DISPUTED, REVISED, REVOKED = (
    "UNKNOWN", "HYPOTHESIS", "ACQUIRED", "DISPUTED", "REVISED", "REVOKED")

#: The states in which the lexicon may be consulted by the executor.  Everything
#: else is silent, which is what makes suspension meaningful rather than
#: bookkeeping.
AUTHORITATIVE = frozenset({ACQUIRED, REVISED})

#: Enumerating repairs is exponential in the number of observations.  The cap is
#: a guard against a pathological stream, not a tuning knob: reaching it makes
#: the belief DISPUTED, which is the safe answer, never a guess.
MAX_OBSERVATIONS = 14


def candidates_of(
    observations: tuple[Observation, ...], universe: tuple[FieldFacts, ...]
) -> frozenset[str]:
    live = frozenset(facts.name for facts in universe)
    for observation in observations:
        live &= constrain(observation, universe)
    return live


def repairs(
    observations: tuple[Observation, ...], universe: tuple[FieldFacts, ...]
) -> tuple[tuple[tuple[int, ...], frozenset[str]], ...]:
    """Cardinality-maximal jointly-satisfiable subsets, with what each implies.

    Searched by increasing number of discarded observations, so the first level
    that yields anything consistent is the set of minimal repairs.  When the
    observations are already consistent this returns the whole set and nothing
    is discarded -- the ordinary case costs one intersection.
    """
    total = len(observations)
    if total > MAX_OBSERVATIONS:
        return ()
    for dropped in range(total + 1):
        found = []
        for kept in combinations(range(total), total - dropped):
            live = candidates_of(tuple(observations[index] for index in kept), universe)
            if live:
                found.append((kept, live))
        if found:
            return tuple(found)
    return ()


@dataclass
class LexicalBelief:
    """What a surface might mean, why, and whether it may be acted on.

    ``support`` is kept per candidate rather than as one set because a revision
    has to be able to say *which* evidence it is giving up.  A belief that only
    remembers its conclusion cannot be revised, only overwritten.
    """

    surface: str
    universe: tuple[FieldFacts, ...]
    observations: tuple[Observation, ...] = ()
    state: str = UNKNOWN
    meaning: str | None = None
    history: tuple[str, ...] = ()
    #: Which observations the current belief base actually holds.  A revision
    #: discards some, and they must *stay* discarded or the contradiction that
    #: produced them fires again on the very next agreeing observation and the
    #: belief oscillates REVISED -> DISPUTED forever.  ``None`` means all of them.
    #: The discard is provisional, not destructive: the full log is kept and the
    #: next contradiction reconsiders every observation, including these.
    retained: tuple[int, ...] | None = None
    #: The last meaning this surface carried authority for.  Kept separately from
    #: ``meaning`` because a DISPUTED belief has no current meaning and still has
    #: to know what it is disputing -- otherwise REVOKED cannot be told from
    #: "never knew anything".
    believed: str | None = None

    @property
    def base(self) -> tuple[Observation, ...]:
        """The observations the belief currently stands on."""
        if self.retained is None:
            return self.observations
        return tuple(self.observations[index] for index in self.retained)

    @property
    def candidates(self) -> frozenset[str]:
        return candidates_of(self.base, self.universe)

    @property
    def authoritative(self) -> bool:
        return self.state in AUTHORITATIVE

    @property
    def discarded(self) -> tuple[str, ...]:
        """What the current belief base set aside, so a revision can be read."""
        if self.retained is None:
            return ()
        kept = set(self.retained)
        return tuple(f"{o.kind}({o.payload})" for index, o in enumerate(self.observations)
                     if index not in kept)

    @property
    def repaired(self) -> frozenset[str]:
        """What the evidence still allows once the minimal repairs are applied.

        ``candidates`` is the raw intersection and is empty whenever a
        contradiction stands, which says nothing about what the word might mean.
        This is the set a report should show for a DISPUTED or REVOKED belief.
        """
        found = repairs(self.observations, self.universe)
        if not found:
            return frozenset()
        return frozenset().union(*(live for _, live in found))

    def support(self) -> dict[str, tuple[str, ...]]:
        """Provenance: which observations each still-possible candidate survives."""
        found: dict[str, list[str]] = {}
        for facts in self.universe:
            kept = [
                f"{o.kind}({o.payload})" for o in self.observations
                if facts.name in constrain(o, self.universe)
            ]
            if len(kept) == len(self.observations) and self.observations:
                found[facts.name] = tuple(kept)
        return {name: tuple(kept) for name, kept in found.items()}

    def _record(self, state: str, meaning: str | None, why: str,
                retained: tuple[int, ...] | None = None) -> "LexicalBelief":
        return LexicalBelief(
            self.surface, self.universe, self.observations, state, meaning,
            self.history + (f"{state}: {why}",), retained,
            meaning if meaning is not None else self.believed)

    def observe(self, observation: Observation) -> "LexicalBelief":
        """One more context. The state machine is entirely in this method."""
        index = len(self.observations)
        retained = None if self.retained is None else self.retained + (index,)
        moved = LexicalBelief(self.surface, self.universe,
                              self.observations + (observation,),
                              self.state, self.meaning, self.history, retained,
                              self.believed)
        live = moved.candidates

        if moved.state in AUTHORITATIVE and not live:
            # Suspend in the same step the disagreement arrives. Nothing is
            # rewritten here, and nothing may execute from here on.  The base is
            # reset to the whole log so the repair search that follows may
            # reconsider observations an earlier revision set aside.
            return moved._record(DISPUTED, None,
                                 f"contradicted by {observation.kind}({observation.payload})")

        if moved.state == DISPUTED:
            return moved._revise()

        if len(live) == 1:
            return moved._record(ACQUIRED, next(iter(live)), "a single candidate survives",
                                 moved.retained)
        if not live:
            return moved._record(DISPUTED, None, "no candidate survives")
        return moved._record(HYPOTHESIS, None, f"{len(live)} candidates survive",
                             moved.retained)

    def _revise(self) -> "LexicalBelief":
        """Adopt a new meaning only when the evidence as a whole picks one."""
        found = repairs(self.observations, self.universe)
        if not found:
            return self._record(DISPUTED, None, "no consistent reading of the evidence")

        implied = {subset_live for _, subset_live in found}
        if len(implied) == 1:
            kept = found[0][0]
            live = next(iter(implied))
            if len(live) == 1:
                return self._record(REVISED, next(iter(live)),
                                    "one repair, one surviving candidate", kept)
            previous = self._last_meaning()
            if previous is not None and previous not in live:
                return self._record(REVOKED, None,
                                    f"one repair rules out `{previous}` and names no replacement")
            return self._record(DISPUTED, None, "one repair, several candidates")

        shared = frozenset.intersection(*implied)
        if len(shared) == 1:
            agreed = next(iter(shared))
            kept = next(k for k, live in found if agreed in live)
            return self._record(REVISED, agreed,
                                "every repair agrees on one candidate", kept)
        previous = self._last_meaning()
        if previous is not None and all(previous not in live for live in implied):
            return self._record(REVOKED, None,
                                f"every repair rules out `{previous}`; none names a replacement")
        return self._record(DISPUTED, None, f"{len(implied)} repairs disagree")

    def _last_meaning(self) -> str | None:
        """The meaning this belief last held authority for, if any."""
        return self.believed


@dataclass
class RevisableLexicon:
    """Surfaces with beliefs attached. Only authoritative ones are visible.

    Duck-types L8.29's ``AcquiredLexicon`` so ``run_acquired_query`` can consult
    it unchanged: the executor never learns that revision exists, which is what
    keeps the safety argument in one place.
    """

    universe: tuple[FieldFacts, ...]
    beliefs: dict[str, LexicalBelief] = field(default_factory=dict)

    def belief(self, surface: str) -> LexicalBelief:
        if surface not in self.beliefs:
            self.beliefs[surface] = LexicalBelief(surface, self.universe)
        return self.beliefs[surface]

    def observe(self, surface: str, observation: Observation) -> LexicalBelief:
        updated = self.belief(surface).observe(observation)
        self.beliefs[surface] = updated
        return updated

    @property
    def entries(self) -> dict[str, str]:
        return {
            surface: belief.meaning
            for surface, belief in self.beliefs.items()
            if belief.authoritative and belief.meaning
        }

    def lookup(self, text: str) -> str | None:
        for surface, label in self.entries.items():
            if surface in text:
                return label
        return None

    def retract(self, surface: str) -> None:
        self.beliefs.pop(surface, None)
