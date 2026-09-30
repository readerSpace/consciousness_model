"""L8.35: find out whether a word needs a context before conditioning it on one.

Registering two meanings against one surface is easy and proves nothing -- it is
a dictionary, and any implementation that adds a sense whenever evidence
disagrees reaches perfect training accuracy by never being wrong about anything.
The question worth answering is the one before that: **does this word need to be
conditioned at all?**

A contradiction is not an answer to it.  L8.30 already showed that one
contradicted belief has at least three readings, and they look identical at the
moment the disagreement arrives:

* the word means one thing and an observation was wrong (**noise**);
* the word meant one thing and now means another (**revision** -- explained by
  *time*);
* the word means different things in different situations (**polysemy** --
  explained by *context*).

So the state machine refuses to jump::

    SINGLE --contradiction--> DISPUTED --a stable conditioning exists--> POLYSEMOUS
                                  ^                                         |
                                  +-------- no stable conditioning ---------+

**The partition is never assumed.**  Writing "in a simulation context it means A"
would put back exactly the hand-written table this project has spent twenty
layers removing.  Instead the evidence is asked whether *some* conditioning makes
the contradiction go away: the whole set is inconsistent, and yet every block of
some partition is internally consistent with a meaning of its own.  Which
variable does the partitioning -- time or context -- is what separates revision
from polysemy, and it is read off the data rather than chosen.

Two rules keep this from becoming the sense-splitting machine it could so easily
be.

**A block of one explains nothing.**  A partition whose block contains a single
observation says exactly what "that observation was noise" says, with an extra
parameter, so it is not admitted.  This is the whole content of
``false_sense_split_rate``, and it needs no threshold: one is not a pattern.

**An explanation that both variables fit is no explanation.**  If the time split
is also context-pure, and the context split is also time-contiguous, the data
cannot tell revision from polysemy and the honest report is that it cannot.  The
belief stays DISPUTED, which is where L8.30 leaves anything it cannot resolve.

And a split is refused outright when L8.31 says the two senses are in the same
observational equivalence class: there is nothing to split, and splitting would
invent a distinction the world does not support.

The context itself is not new.  It is the episode -- L8.13's boundary, introduced
twenty-two layers ago to stop old work leaking into new, arriving here as the
conditioning variable for word sense.  ``Episode -> Context -> Sense``.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

from .cross_context_identification_l829_experiment import (
    FieldFacts, Observation, constrain,
)
from .semantic_identifiability_l831_experiment import reachable

SINGLE, DISPUTED, POLYSEMOUS = "SINGLE", "DISPUTED", "POLYSEMOUS"

MONOSEMY, NOISE, REVISION, POLYSEMY = "MONOSEMY", "NOISE", "REVISION", "POLYSEMY"
AMBIGUOUS, UNIDENTIFIABLE, UNEXPLAINED = "AMBIGUOUS", "UNIDENTIFIABLE", "UNEXPLAINED"

#: A block this small says what "that observation was noise" says, with one more
#: parameter. Not a threshold to tune -- one is not a pattern.
MIN_BLOCK = 2


@dataclass(frozen=True)
class ContextualObservation:
    """An observation, plus where and when it was made.

    ``context`` is the episode it came from (L8.13). Nothing here knows what an
    episode *is*; it is an opaque label that either explains the disagreement or
    does not.
    """

    observation: Observation
    context: str
    time: int


def _live(evidence: tuple[ContextualObservation, ...],
          universe: tuple[FieldFacts, ...]) -> frozenset[str]:
    names = frozenset(facts.name for facts in universe)
    for item in evidence:
        names &= constrain(item.observation, universe)
    return names


@dataclass(frozen=True)
class Explanation:
    kind: str
    #: block label -> meaning. For MONOSEMY / NOISE the single label is "*".
    senses: dict[str, str]
    blocks: dict[str, tuple[int, ...]]
    discarded: tuple[int, ...] = ()
    note: str = ""

    @property
    def state(self) -> str:
        if self.kind in (MONOSEMY, NOISE):
            return SINGLE
        if self.kind in (REVISION, POLYSEMY):
            return POLYSEMOUS if self.kind == POLYSEMY else SINGLE
        return DISPUTED

    @property
    def conditioned(self) -> bool:
        return self.kind == POLYSEMY


# --------------------------------------------------------------------------
# the candidate explanations
# --------------------------------------------------------------------------

def _monosemy(evidence, universe) -> Explanation | None:
    live = _live(evidence, universe)
    if len(live) == 1:
        return Explanation(MONOSEMY, {"*": next(iter(live))},
                           {"*": tuple(range(len(evidence)))},
                           note="every observation holds of one meaning")
    return None


def _noise(evidence, universe) -> Explanation | None:
    """L8.30's repair, reused rather than reimplemented in spirit."""
    total = len(evidence)
    for dropped in range(1, total):
        best = None
        for kept in combinations(range(total), total - dropped):
            live = _live(tuple(evidence[i] for i in kept), universe)
            if len(live) == 1:
                if best is not None:
                    return None  # several equal-cost repairs disagree: not an explanation
                best = (kept, next(iter(live)))
            elif live:
                continue
        if best is not None:
            kept, meaning = best
            return Explanation(NOISE, {"*": meaning}, {"*": kept},
                               tuple(i for i in range(total) if i not in kept),
                               f"one meaning once {dropped} observation(s) are set aside")
    return None


def _partition_by(evidence, universe, key) -> tuple[dict[str, str], dict[str, tuple[int, ...]]] | None:
    """Blocks by some label, each internally consistent with a meaning of its own."""
    blocks: dict[str, list[int]] = {}
    for index, item in enumerate(evidence):
        blocks.setdefault(key(item), []).append(index)
    if len(blocks) < 2:
        return None
    senses: dict[str, str] = {}
    for label, indices in blocks.items():
        if len(indices) < MIN_BLOCK:
            return None  # a block of one is the noise hypothesis wearing a hat
        live = _live(tuple(evidence[i] for i in indices), universe)
        if len(live) != 1:
            return None
        senses[label] = next(iter(live))
    if len(set(senses.values())) < 2:
        return None  # the blocks agree, so the conditioning explains nothing
    return senses, {label: tuple(indices) for label, indices in blocks.items()}


def _revision(evidence, universe) -> Explanation | None:
    """A split in *time*: everything before t0 means one thing, after it another."""
    times = sorted({item.time for item in evidence})
    for cut in times[1:]:
        found = _partition_by(evidence, universe,
                              lambda item, cut=cut: "before" if item.time < cut else "after")
        if found:
            senses, blocks = found
            return Explanation(REVISION, senses, blocks,
                               note=f"the meaning changed at t={cut}")
    return None


def _polysemy(evidence, universe) -> Explanation | None:
    """A split by *context*: the label is opaque, and either it works or it does not."""
    found = _partition_by(evidence, universe, lambda item: item.context)
    if not found:
        return None
    senses, blocks = found
    return Explanation(POLYSEMY, senses, blocks,
                       note="each context is consistent with a sense of its own")


# --------------------------------------------------------------------------
# choosing between them
# --------------------------------------------------------------------------

def _context_pure(evidence, blocks) -> bool:
    """Would the context variable have produced the same blocks?"""
    return all(len({evidence[i].context for i in indices}) == 1
               for indices in blocks.values())


def _time_contiguous(evidence, blocks) -> bool:
    """Would a time split have produced the same blocks?"""
    spans = []
    for indices in blocks.values():
        times = [evidence[i].time for i in indices]
        spans.append((min(times), max(times)))
    spans.sort()
    return all(spans[i][1] < spans[i + 1][0] for i in range(len(spans) - 1))


def explain(evidence: tuple[ContextualObservation, ...],
            universe: tuple[FieldFacts, ...],
            probes: tuple[Observation, ...] = ()) -> Explanation:
    """Which account of the evidence the evidence itself supports.

    Ordered so that the cheapest account wins: no explanation at all beats an
    explanation. Only when the whole set is inconsistent is a conditioning
    considered, and only when exactly one conditioning variable fits is one
    adopted.
    """
    if not evidence:
        return Explanation(UNEXPLAINED, {}, {}, note="no evidence")

    settled = _monosemy(evidence, universe)
    if settled is not None:
        return settled

    revision, polysemy = _revision(evidence, universe), _polysemy(evidence, universe)

    if revision is not None and polysemy is not None:
        # Both variables fit. If each partition would also have been produced by
        # the other variable, the data cannot tell them apart and saying so is
        # the only honest move.
        if (_context_pure(evidence, revision.blocks)
                and _time_contiguous(evidence, polysemy.blocks)):
            return Explanation(AMBIGUOUS, {}, {},
                               note="time and context explain the same split")
    if polysemy is not None and revision is None:
        senses = set(polysemy.senses.values())
        if probes and len(senses) == 2:
            left, right = sorted(senses)
            if right in reachable(left, universe, probes):
                return Explanation(UNIDENTIFIABLE, {}, {},
                                   note=f"{left} and {right} are not separable here")
        return polysemy
    if revision is not None and polysemy is None:
        return revision
    if revision is not None and polysemy is not None:
        return polysemy if not _time_contiguous(evidence, polysemy.blocks) else revision

    noisy = _noise(evidence, universe)
    if noisy is not None:
        return noisy
    return Explanation(UNEXPLAINED, {}, {}, note="no account fits the evidence")


# --------------------------------------------------------------------------
# the lexicon that can hold a conditioned entry
# --------------------------------------------------------------------------

@dataclass
class ContextualLexicon:
    """Surface -> meaning, or (surface, context) -> meaning once conditioned.

    Duck-types the ``lookup`` L8.29's executor calls, so nothing downstream
    learns that senses exist; it asks in a context and gets a meaning or nothing.
    """

    universe: tuple[FieldFacts, ...]
    evidence: dict[str, list[ContextualObservation]] = None
    explanations: dict[str, Explanation] = None
    probes: tuple[Observation, ...] = ()

    def __post_init__(self):
        self.evidence = self.evidence or {}
        self.explanations = self.explanations or {}

    def observe(self, surface: str, observation: Observation,
                context: str, time: int) -> Explanation:
        self.evidence.setdefault(surface, []).append(
            ContextualObservation(observation, context, time))
        found = explain(tuple(self.evidence[surface]), self.universe, self.probes)
        self.explanations[surface] = found
        return found

    def meaning(self, surface: str, context: str | None) -> str | None:
        found = self.explanations.get(surface)
        if found is None:
            return None
        if found.kind in (MONOSEMY, NOISE):
            return found.senses["*"]
        if found.kind == REVISION:
            # The current meaning is the latest block's.
            latest = max(found.blocks, key=lambda label: max(found.blocks[label]))
            return found.senses[latest]
        if found.kind == POLYSEMY:
            return found.senses.get(context) if context is not None else None
        return None

    def lookup(self, text: str, context: str | None = None) -> str | None:
        for surface in self.evidence:
            if surface in text:
                return self.meaning(surface, context)
        return None
