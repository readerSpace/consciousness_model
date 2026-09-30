"""L8.31: decide, before learning anything, whether the word *could* be learned.

Three layers have now failed in the same way and called it three things.  L8.28
found that a third Number field collapsed decidability while recall stayed 1.0.
L8.29 found that in a ``nested`` world a field contained in every other field's
range cannot be separated by any literal, and built a twin control where no
amount of evidence resolves the pair.  L8.30 found that a ``banded`` world hides
noise: a stray observation consistent with the truth never contradicts, so
nothing can be revised, and the honest report was ``undetected_noise_rate``
rather than a revision failure.

Those are one phenomenon.  Two meanings are **observationally equivalent** when
the observation language cannot tell them apart::

    m_i ~_O m_j  <=>  for every o in O,  effect(m_i, o) = effect(m_j, o)

with ``effect(m, o)`` being simply whether ``m`` survives ``o``.  Everything the
three layers ran into is a non-singleton class of this relation, and this layer
computes those classes **before** any learning happens, from the schema and the
observation family alone.

The verdict is not a boolean.  ``IDENTIFIABLE`` means every class is a singleton;
``PARTIALLY_IDENTIFIABLE`` means some are and some are not; and
``STRUCTURALLY_UNIDENTIFIABLE`` means the word in question sits in a class of
size two or more, so no stream of evidence will ever name it.  The classes come
back with the verdict, because "cannot be decided" is far less useful than
"cannot be decided *between success_rate and energy_error*".

Two capabilities follow from having the classes rather than a flag.

**Prediction.**  Whether a particular observation is even capable of
contradicting a particular meaning is decidable in advance: ``o`` can be detected
as noise for ``m`` exactly when ``m`` is not in ``constrain(o)``.  So L8.30's
``undetected_noise_rate`` is predictable before L8.30 runs, and the hold-out
checks the two against each other.  Agreement is the evidence that these are not
three separate demonstrations but one structure measured three ways.

**Minimal distinguishing sets.**  A class that cannot be split by the current
observation language may be splittable by a small extension of it, and the
smallest such extension is searchable.  ``{success_rate, energy_error}`` under
TYPE and CONTRAST alone becomes ``{success_rate} | {energy_error}`` the moment a
RANGE literal at 0.5 is available.  That turns "I do not know" into "I do not
know, and here is what would tell me" -- which is the input the next layer needs.

The safety metric is ``false_identifiable_rate`` and it belongs to the same
family as every invariant since L8.18.  Declaring a word learnable when it is not
is the dangerous error: L8.29 would then acquire a meaning on partial evidence
and L8.30 would inherit a case it cannot correct.  Declaring a learnable word
unlearnable only costs coverage.  So the analysis is written to be conservative
in exactly that direction: a pair counts as distinguished only when some
observation actually separates them, never by default.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

from .cross_context_identification_l829_experiment import (
    Belief, FieldFacts, Observation, available_probes, constrain, truthful,
)

IDENTIFIABLE = "IDENTIFIABLE"
PARTIALLY_IDENTIFIABLE = "PARTIALLY_IDENTIFIABLE"
STRUCTURALLY_UNIDENTIFIABLE = "STRUCTURALLY_UNIDENTIFIABLE"


def effect(name: str, observation: Observation, universe: tuple[FieldFacts, ...]) -> bool:
    """Does this meaning survive this observation?  The whole relation rests here."""
    return name in constrain(observation, universe)


def signature(
    name: str, universe: tuple[FieldFacts, ...], observations: tuple[Observation, ...]
) -> tuple[bool, ...]:
    return tuple(effect(name, observation, universe) for observation in observations)


def equivalence_classes(
    universe: tuple[FieldFacts, ...], observations: tuple[Observation, ...]
) -> tuple[frozenset[str], ...]:
    """Partition the meanings by what the observation language can say about them.

    Two meanings land together exactly when no available observation treats them
    differently.  With an empty observation family everything is one class, which
    is the correct degenerate answer rather than a special case.

    This relation is **symmetric, and therefore optimistic**, so it describes the
    universe and never decides anything about safety.  Signatures differing at
    ``o`` means exactly one of the pair survives ``o`` -- but a world whose word
    means the one that *fails* ``o`` can never produce ``o``, so that separation
    is unavailable to it.  Ruling out is directional, and ``reachable`` below is
    the notion every verdict is actually built from.
    """
    grouped: dict[tuple[bool, ...], set[str]] = {}
    for facts in universe:
        grouped.setdefault(signature(facts.name, universe, observations), set()).add(facts.name)
    return tuple(sorted(
        (frozenset(names) for names in grouped.values()),
        key=lambda names: sorted(names)))


def reachable(
    name: str, universe: tuple[FieldFacts, ...], observations: tuple[Observation, ...]
) -> frozenset[str]:
    """What a truthful stream about ``name`` can never rule out.

    Only observations the meaning itself satisfies can occur in a world where it
    is the meaning, so only those may narrow anything.  This is exactly what
    L8.29's ``cross`` arm computes at the end of an exhaustive run, which makes
    the two independently checkable against each other.
    """
    live = frozenset(facts.name for facts in universe)
    for observation in observations:
        if effect(name, observation, universe):
            live &= constrain(observation, universe)
    return live


@dataclass(frozen=True)
class Analysis:
    verdict: str
    classes: tuple[frozenset[str], ...]
    #: The universe-wide picture from the symmetric relation, for reporting only.
    universe_verdict: str = ""
    #: The word whose meaning was asked about, when one was.
    word: str | None = None
    #: The class the word in question falls in, when one was named.
    of_interest: frozenset[str] | None = None
    #: A smallest set of extra observations that would split ``of_interest``.
    distinguishing: tuple[Observation, ...] = ()
    #: What ``of_interest`` becomes once those are available.
    refined: tuple[frozenset[str], ...] = ()

    @property
    def resolves(self) -> bool:
        """Whether the proposed extension actually isolates the word of interest.

        A refinement that turns ``{a, b, c}`` into ``{a} | {b, c}`` is progress
        and not an answer; reporting the two as the same thing is how an
        analyser starts promising more than it delivers.
        """
        if not self.refined or self.word is None:
            return False
        block = next((b for b in self.refined if self.word in b), None)
        return block is not None and len(block) == 1

    @property
    def learnable(self) -> bool:
        """Whether the meaning of interest can be pinned down at all."""
        if self.of_interest is None:
            return self.verdict == IDENTIFIABLE
        return len(self.of_interest) == 1

    def why(self) -> str:
        if self.learnable:
            return "the observation language separates it from everything else"
        names = ", ".join(sorted(self.of_interest or ()))
        if self.distinguishing:
            kinds = ", ".join(f"{o.kind}({o.payload})" for o in self.distinguishing)
            verb = "would pin it down" if self.resolves else "would narrow it, not settle it"
            return (f"not separable from {{{names}}} by this language; "
                    f"adding {kinds} {verb} -- more of the same evidence would not")
        return (f"not separable from {{{names}}} by any observation this world "
                f"can produce; no amount of evidence will name it")


def minimal_distinguishing_set(
    klass: frozenset[str],
    universe: tuple[FieldFacts, ...],
    pool: tuple[Observation, ...],
    limit: int = 3,
    target: str | None = None,
) -> tuple[tuple[Observation, ...], tuple[frozenset[str], ...]]:
    """The smallest extension of the language that splits ``klass``.

    Searched by increasing size, so the first hit is minimal.  Returning the
    resulting partition alongside matters: a set that splits ``{a, b, c}`` into
    ``{a} | {b, c}`` is progress and is not a solution, and the caller has to be
    able to tell.  When ``target`` is given, an extension that isolates it is
    preferred at every size over one that merely splits the class -- otherwise
    the search happily reports a refinement that leaves the word exactly as
    undetermined as it was.

    Extensions are evaluated directionally, as ``reachable`` is: an observation
    the target cannot satisfy could never arrive in a world where the target is
    the meaning, so it may not count towards isolating it.
    """
    if len(klass) < 2:
        return (), (klass,)

    def partition(extra):
        grouped: dict[tuple[bool, ...], set[str]] = {}
        for name in klass:
            live = reachable(name, universe, extra) if target is None else None
            key = signature(name, universe, extra)
            grouped.setdefault(key, set()).add(name)
        if target is not None:
            # Directional view for the target: only observations it satisfies
            # can narrow its own class.
            narrowed = klass & reachable(target, universe, extra)
            rest = klass - narrowed
            blocks = [b for b in (narrowed, rest) if b]
            if len(blocks) > 1:
                return tuple(sorted((frozenset(b) for b in blocks),
                                    key=lambda names: sorted(names)))
            return ()
        if len(grouped) > 1:
            return tuple(sorted((frozenset(names) for names in grouped.values()),
                                key=lambda names: sorted(names)))
        return ()

    fallback = None
    for size in range(1, limit + 1):
        for extra in combinations(pool, size):
            refined = partition(extra)
            if not refined:
                continue
            if target is None:
                return extra, refined
            block = next(b for b in refined if target in b)
            if len(block) == 1:
                return extra, refined
            if fallback is None:
                fallback = (extra, refined)
    return fallback if fallback is not None else ((), (klass,))


def analyse(
    universe: tuple[FieldFacts, ...],
    observations: tuple[Observation, ...],
    word_means: str | None = None,
    pool: tuple[Observation, ...] = (),
    limit: int = 3,
) -> Analysis:
    """Classes, a verdict, and -- when a class is stuck -- what would unstick it."""
    classes = equivalence_classes(universe, observations)
    sizes = [len(klass) for klass in classes]
    if all(size == 1 for size in sizes):
        verdict = IDENTIFIABLE
    elif any(size == 1 for size in sizes):
        verdict = PARTIALLY_IDENTIFIABLE
    else:
        verdict = STRUCTURALLY_UNIDENTIFIABLE

    if word_means is None:
        return Analysis(verdict, classes, verdict)


    # Directional, not symmetric: see ``reachable``.
    klass = reachable(word_means, universe, observations)
    if len(klass) == 1:
        return Analysis(IDENTIFIABLE, classes, verdict, word_means, klass)

    extra, refined = minimal_distinguishing_set(klass, universe, pool, limit, word_means)
    # The distinction the three earlier layers kept running into, made explicit:
    # a class the current language cannot split but an extension can is a gap in
    # the language, and more of the same evidence will never close it. A class no
    # extension splits is a wall.
    word_verdict = PARTIALLY_IDENTIFIABLE if extra else STRUCTURALLY_UNIDENTIFIABLE
    return Analysis(word_verdict, classes, verdict, word_means, klass, extra, refined)


# --------------------------------------------------------------------------
# prediction: what the later layers will be able to see
# --------------------------------------------------------------------------

def detectable_as_noise(
    truth: str, observation: Observation, universe: tuple[FieldFacts, ...]
) -> bool:
    """Can a world whose word means ``truth`` ever notice this observation is wrong?

    Only when the observation excludes the truth.  A stray observation the truth
    also satisfies is indistinguishable from a real one, so nothing downstream --
    no belief, no repair, no amount of corroboration -- can flag it.  This is
    L8.30's ``undetected_noise_rate`` computed before L8.30 runs.
    """
    return not effect(truth, observation, universe)


def predict_contradiction(
    stream: tuple[Observation, ...], universe: tuple[FieldFacts, ...]
) -> bool:
    """Will a belief fed this stream ever record a contradiction?

    Two conditions, and separating them matters.  The evidence has to be jointly
    unsatisfiable *and* the belief has to have committed to something before the
    conflict arrives -- a stream that never narrows to one candidate has no
    meaning to contradict, so it goes quiet rather than DISPUTED.  Detectability
    in principle is not the same as a contradiction being recorded.

    This is set arithmetic over prefixes and nothing else: no state machine, no
    repairs, no retained base.  That independence is what makes comparing it
    against L8.30's behaviour a check rather than a restatement.
    """
    live = frozenset(facts.name for facts in universe)
    committed = False
    for observation in stream:
        live &= constrain(observation, universe)
        if not live:
            return committed
        if len(live) == 1:
            committed = True
    return False


def saturate(
    truth: str, universe: tuple[FieldFacts, ...], count: int
) -> frozenset[str]:
    """What survives after ``count`` truthful observations drawn round-robin.

    The point is the cases where this stops shrinking. Intersection is monotone,
    so once the surviving set equals the truth's equivalence class, no further
    observation can change it however many arrive -- which is what separates
    "not enough data yet" from "the language cannot say it".
    """
    probes = tuple(p for p in available_probes(universe) if truthful(p, truth, universe))
    if not probes:
        return frozenset(facts.name for facts in universe)
    belief = Belief.prior("w", universe)
    for index in range(count):
        belief = Belief("w", belief.candidates & constrain(probes[index % len(probes)], universe))
    return belief.candidates
