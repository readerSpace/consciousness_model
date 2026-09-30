"""L8.29: narrow a word's meaning across episodes instead of guessing it in one.

L8.28.1 ended with a clean negative and, more usefully, with the exact shape of
it.  At two attested Number fields the generator was decidable because

    {runtime, success} - {runtime} = {success}

is elimination, not identification.  At three,

    {runtime, success, energy} - {runtime} = {success, energy}

and decidability went to zero while **recall stayed 1.0**: the truth was never
lost, it stopped being separable.  That distinction is what this layer is built
on.  A candidate set that still contains the answer is not a failure, it is an
unfinished measurement -- and measurements can be continued.

So meaning is treated as a latent variable narrowed by evidence over time:

    H_{t+1}(w) = H_t(w) ∩ C(e_t)

with the state read off the set and nothing else::

    |H| == 1  ->  DECIDABLE      a meaning, acquirable
    |H| == 0  ->  CONTRADICTION  the contexts cannot all be about one field
    |H| >= 2  ->  UNRESOLVED     not yet, and saying so is the correct answer

which keeps L8.18's fail-closed rule intact by construction rather than by a new
guard: nothing is acquired unless the set is a singleton, and a set that never
becomes one never acquires.  There is no threshold anywhere in this layer.

Four constraint sources, all label-free and all already present in the system:

``TYPE``
    The operator's projection hole (L8.21 over L8.19).  Separates Numbers from
    the rest and nothing within them -- which is precisely why one observation
    was not enough.

``CONTRAST``
    「Xではなく w」 removes one candidate.  Removing one at a time means K-1
    observations for K candidates: **linear**, and that is the baseline to beat.

``RANGE``
    A literal compared against w must lie inside the field's attested range.  A
    well-placed literal splits the candidate set rather than shaving it, so
    identification can be **logarithmic** instead of linear.  The store supplies
    the ranges; no vocabulary is consulted.

``DISTINCT``
    w occupies a role in the same request as a labelled field, and the algebra
    forbids one field in both, so w is not it.  Same arithmetic as CONTRAST,
    different provenance, counted separately.

Because a split carries more than a shave, *which context to seek next* becomes
a real decision, and the standard one applies:

    EIG(q) = H(M | D) - E[ H(M | D, q) ]

over a uniform prior on the surviving candidates.  This is the same
uncertainty-reducing intervention the earlier experiment planners used; the only
change is that the latent variable is a word's meaning rather than a state of
the world.

Honest note on regimes.  The three real fields have nearly disjoint attested
ranges (runtime 8.0--51.75, success_rate 0.40--0.75, energy_error 0.001--0.024),
so one RANGE observation almost identifies and the interesting question does not
arise.  The K-sweep therefore uses a synthetic universe whose neighbouring
ranges **overlap by half** on purpose: the harder regime, where a single literal
cannot finish and the difference between shaving and splitting is visible.  The
K=2 and K=3 universes are derived from the real store and cross-checked against
L8.28's numbers, so the synthetic extension is an extension of a measured
object, not a substitute for one.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from .evidence_graph_l827_experiment import build_graph, readings, run_graph_query
from .lexical_hypothesis_l828_experiment import (
    admitted_by_type, excluded_by_contrast,
)
from .semantic_proposer_l822_experiment import Proposer, ProposedResult
from .span_grounded_proposer_l823_experiment import SpanProposer
from .structural_sensor_l826_experiment import StructuralSensor
from .typed_operation_l819_experiment import FIELDS, TypedStore, ValueType

DECIDABLE, CONTRADICTION, UNRESOLVED = "DECIDABLE", "CONTRADICTION", "UNRESOLVED"

SOURCES = ("TYPE", "CONTRAST", "RANGE", "DISTINCT")


# --------------------------------------------------------------------------
# the universe a meaning is drawn from
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class FieldFacts:
    """What the system can know about a field without reading any text."""

    name: str
    value_type: ValueType
    low: float | None = None
    high: float | None = None
    #: Whether any request can name this field.  CONTRAST and DISTINCT need a
    #: *labelled* sibling, so an unnameable field can never be excluded by one --
    #: which is what makes the indistinguishable control genuinely indistinguishable
    #: rather than merely awkward.
    nameable: bool = True

    def admits(self, value: float) -> bool:
        if self.low is None or self.high is None:
            return False
        return self.low <= value <= self.high


def universe_from(store: TypedStore) -> tuple[FieldFacts, ...]:
    """Types from the schema, ranges from what episodes actually recorded."""
    seen: dict[str, list[float]] = {}
    for _, values in store.rows():
        for name, value in values.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                seen.setdefault(name, []).append(float(value))
    facts = []
    for spec in FIELDS:
        values = seen.get(spec.name)
        if not values:
            continue  # unattested: L8.28's attestation source already drops these
        facts.append(FieldFacts(spec.name, spec.value_type, min(values), max(values)))
    return tuple(facts)


#: How a synthetic world lays out its Number fields.  The geometry is not a
#: detail: it decides how much a single literal can ever say, and therefore what
#: the scaling law of identification actually is.
#:
#: ``banded``  fixed-width ranges sliding past each other.  Any value falls
#:             inside a constant number of them however many fields exist, so a
#:             literal localises to a constant-size block and EIG's cost stops
#:             growing with K.
#: ``nested``  ranges sharing a left edge and shrinking, so a literal splits the
#:             field set into a prefix and its complement and a well-placed one
#:             halves it.  This is the regime where identification is logarithmic.
#: ``disjoint`` ranges never overlap, so every field has values only it admits and
#: a false observation is always detectable.  It is the regime in which a claim
#: about *revision* can be made at all: noise you cannot detect is noise you
#: cannot revise, which is L8.29's identifiability result showing up again.
GEOMETRIES = ("banded", "nested")
ALL_GEOMETRIES = ("banded", "nested", "disjoint")


#: How much of a field's range its neighbour covers in the synthetic universe.
#: Half is a deliberate choice: disjoint ranges make one literal decisive and
#: total overlap makes every literal useless, so neither would measure anything.
SYNTHETIC_OVERLAP = 0.5


def synthetic_universe(
    count: int, base: tuple[FieldFacts, ...] = (), geometry: str = "banded"
) -> tuple[FieldFacts, ...]:
    """``count`` Number fields laid out by ``geometry`` (see ``GEOMETRIES``).

    The first fields are the real ones when ``base`` is given, so K=2 and K=3 are
    the measured system and larger K extends it under a stated rule.
    """
    facts = list(base[:count])
    width = 2.0
    step = width * (1.0 - SYNTHETIC_OVERLAP)
    for index in range(len(facts), count):
        if geometry == "banded":
            low = index * step
            facts.append(FieldFacts(f"metric_{index:02d}", ValueType.NUMBER, low, low + width))
        elif geometry == "nested":
            facts.append(FieldFacts(f"metric_{index:02d}", ValueType.NUMBER,
                                    0.0, round(1.0 / (index + 1), 9)))
        elif geometry == "disjoint":
            low = index * (width + 1.0)
            facts.append(FieldFacts(f"metric_{index:02d}", ValueType.NUMBER, low, low + width))
        elif geometry == "dominant":
            # Geometrically separated, so exactly one field can exceed the sum of
            # all the others at *any* n: 2^(n-1) + 1 > 2^0 + ... + 2^(n-2).
            # Added because the disjoint layout makes a sum constraint
            # unsatisfiable once n grows, which is a fact about that world rather
            # than about the solver reading it.
            low = float(2 ** index)
            facts.append(FieldFacts(f"metric_{index:02d}", ValueType.NUMBER, low, low + 1.0))
        else:
            raise ValueError(f"unknown geometry: {geometry}")
    return tuple(facts)


def indistinguishable_universe(count: int) -> tuple[FieldFacts, ...]:
    """A pair no observation in this vocabulary can separate.

    ``twin_a`` and ``twin_b`` share a type and an identical range and are never
    nameable, so CONTRAST and DISTINCT can never mention them and RANGE can never
    split them.  Staying UNRESOLVED forever is the correct behaviour, and a
    layer that ever resolves them is guessing.
    """
    facts = list(synthetic_universe(count))
    facts.append(FieldFacts("twin_a", ValueType.NUMBER, 0.0, 1.0, nameable=False))
    facts.append(FieldFacts("twin_b", ValueType.NUMBER, 0.0, 1.0, nameable=False))
    return tuple(facts)


# --------------------------------------------------------------------------
# observations and the belief they narrow
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Observation:
    kind: str  # TYPE | CONTRAST | RANGE | DISTINCT
    payload: object
    note: str = ""


def constrain(observation: Observation, universe: tuple[FieldFacts, ...]) -> frozenset[str]:
    """C(e): the candidates this one context leaves standing."""
    if observation.kind == "TYPE":
        return frozenset(f.name for f in universe if f.value_type is observation.payload)
    if observation.kind in ("CONTRAST", "DISTINCT"):
        return frozenset(f.name for f in universe if f.name != observation.payload)
    if observation.kind == "RANGE":
        return frozenset(f.name for f in universe if f.admits(float(observation.payload)))
    if observation.kind == "EQUALS":
        # A direct assertion: 「それじゃなくて成功率」 names the meaning outright.
        # It still enters as an observation rather than as a write to the lexicon,
        # so a later context can contradict it exactly like any other evidence.
        return frozenset(f.name for f in universe if f.name == observation.payload)
    if observation.kind == "NOT":
        # The complement of an inner observation.  Passive layers never build one
        # -- a world only states what holds -- but an *asked* question has two
        # answers and the negative one is evidence, so active querying needs the
        # negation to be a first-class observation rather than something a caller
        # approximates.  Added here so every layer that narrows through
        # ``constrain`` handles it without changing.
        return frozenset(f.name for f in universe) - constrain(observation.payload, universe)
    raise ValueError(f"unknown observation kind: {observation.kind}")


@dataclass
class Belief:
    """H(w), plus how it got that way."""

    word: str
    candidates: frozenset[str]
    history: list[Observation] = field(default_factory=list)

    @classmethod
    def prior(cls, word: str, universe: tuple[FieldFacts, ...]) -> "Belief":
        return cls(word, frozenset(f.name for f in universe))

    def observe(self, observation: Observation, universe) -> "Belief":
        return Belief(self.word,
                      self.candidates & constrain(observation, universe),
                      self.history + [observation])

    @property
    def state(self) -> str:
        if len(self.candidates) == 1:
            return DECIDABLE
        if not self.candidates:
            return CONTRADICTION
        return UNRESOLVED

    @property
    def meaning(self) -> str | None:
        return next(iter(self.candidates)) if self.state == DECIDABLE else None


# --------------------------------------------------------------------------
# which context to seek next
# --------------------------------------------------------------------------

def _entropy(size: int) -> float:
    return math.log2(size) if size > 0 else 0.0


def expected_information_gain(
    belief: Belief, probe: Observation, universe: tuple[FieldFacts, ...]
) -> float:
    """EIG under a uniform prior over the surviving candidates.

    The outcome of a probe is not known in advance, so the candidates are
    partitioned by what each outcome would imply and the expectation is taken
    over the blocks.  A RANGE probe splits; a CONTRAST probe shaves one off, and
    the arithmetic says so without being told.
    """
    current = belief.candidates
    if len(current) <= 1:
        return 0.0
    kept = current & constrain(probe, universe)
    blocks = [kept, current - kept]
    before = _entropy(len(current))
    after = sum(
        (len(block) / len(current)) * _entropy(len(block))
        for block in blocks if block
    )
    return before - after


def choose_probe(
    belief: Belief, probes: tuple[Observation, ...], universe: tuple[FieldFacts, ...]
) -> Observation | None:
    """Argmax EIG, ties broken deterministically so runs are reproducible."""
    scored = [
        (expected_information_gain(belief, probe, universe), index, probe)
        for index, probe in enumerate(probes)
    ]
    scored = [item for item in scored if item[0] > 0.0]
    if not scored:
        return None
    return max(scored, key=lambda item: (item[0], -item[1]))[2]


# --------------------------------------------------------------------------
# the context pool a world could offer
# --------------------------------------------------------------------------

def available_probes(universe: tuple[FieldFacts, ...]) -> tuple[Observation, ...]:
    """Every context this world could produce about an unknown Number word.

    RANGE literals are the midpoints of each field's attested range plus the
    boundaries between them, which is what a world talking about these fields
    would naturally mention.  No probe is tailored to any particular answer.
    """
    probes: list[Observation] = [Observation("TYPE", ValueType.NUMBER, "a Number-valued hole")]
    for facts in universe:
        if not facts.nameable:
            continue  # a sentence cannot say 「Xではなく」 about a field it cannot name
        probes.append(Observation("CONTRAST", facts.name, f"not {facts.name}"))
    numbers = [f for f in universe if f.low is not None]
    edges: set[float] = set()
    for facts in numbers:
        edges.add(round((facts.low + facts.high) / 2, 6))
        edges.add(round(facts.high, 6))
    for value in sorted(edges):
        probes.append(Observation("RANGE", value, f"a literal at {value}"))
    return tuple(probes)


def truthful(probe: Observation, truth: str, universe: tuple[FieldFacts, ...]) -> bool:
    """Could a world whose word means ``truth`` actually produce this context?"""
    return truth in constrain(probe, universe)


# --------------------------------------------------------------------------
# identification
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Identification:
    word: str
    truth: str
    state: str
    meaning: str | None
    observations: int
    history: tuple[str, ...]

    @property
    def acquired(self) -> bool:
        return self.state == DECIDABLE

    @property
    def wrong(self) -> bool:
        return self.acquired and self.meaning != self.truth


def identify(
    word: str,
    truth: str,
    universe: tuple[FieldFacts, ...],
    strategy: str = "eig",
    budget: int = 32,
) -> Identification:
    """Narrow ``word`` until the set settles, or the budget runs out.

    ``single``  one context only -- L8.28's regime, kept as the baseline.
    ``cross``   every truthful context in a fixed order, intersected.
    ``eig``     the next context chosen by expected information gain.

    Only contexts a truthful world could produce are admitted, so the sequence is
    a narrowing of the same latent variable and not a search with the answer in
    hand.  ``choose_probe`` never sees ``truth``.
    """
    probes = tuple(p for p in available_probes(universe) if truthful(p, truth, universe))
    belief = Belief.prior(word, universe)
    used: list[Observation] = []

    if strategy == "single":
        # The type hole is what a single request supplies on its own; L8.28 then
        # added at most one contrast from the same sentence.
        for probe in probes[:2]:
            belief = belief.observe(probe, universe)
            used.append(probe)
    elif strategy == "cross":
        for probe in probes:
            if belief.state != UNRESOLVED or len(used) >= budget:
                break
            belief = belief.observe(probe, universe)
            used.append(probe)
    elif strategy == "eig":
        remaining = list(probes)
        while belief.state == UNRESOLVED and remaining and len(used) < budget:
            probe = choose_probe(belief, tuple(remaining), universe)
            if probe is None:
                break
            remaining.remove(probe)
            belief = belief.observe(probe, universe)
            used.append(probe)
    else:
        raise ValueError(f"unknown strategy: {strategy}")

    return Identification(word, truth, belief.state, belief.meaning, len(used),
                          tuple(f"{p.kind}:{p.note}" for p in used))


# --------------------------------------------------------------------------
# acquisition, and using what was acquired
# --------------------------------------------------------------------------

@dataclass
class AcquiredLexicon:
    """Span text -> field, written only from a DECIDABLE belief.

    Separate from the classifiers on purpose: L8.28 kept generation away from
    promotion, and this keeps promotion reversible.  ``retract`` exists because
    an acquisition that turns out to break the safety invariant has to be
    removable without retraining anything.
    """

    entries: dict[str, str] = field(default_factory=dict)

    def acquire(self, identification: Identification) -> bool:
        if not identification.acquired:
            return False
        self.entries[identification.word] = identification.meaning
        return True

    def retract(self, word: str) -> None:
        self.entries.pop(word, None)

    def lookup(self, text: str) -> str | None:
        for word, label in self.entries.items():
            if word in text:
                return label
        return None


def run_acquired_query(
    store: TypedStore,
    request: str,
    lexicon: AcquiredLexicon,
    spans: SpanProposer | None = None,
    sentence: Proposer | None = None,
    structure: StructuralSensor | None = None,
) -> ProposedResult:
    """L8.27, with an acquired meaning allowed to fill an unnamed argument.

    This is the first point in the project where something the system learned
    about *language* can change what executes, so the opening is deliberately
    narrow: it applies only where L8.27 already refused for want of a name, only
    to the exact span the graph could not label, and only from a singleton
    belief.  Everything downstream is untouched -- L8.21 still compiles, L8.19
    still type-checks, uniqueness still decides.  A wrong entry costs a wrong
    program, which is why ``false_acquisition_rate`` is the metric that matters.
    """
    spans = spans or SpanProposer.trained()
    sentence = sentence or Proposer.trained()
    structure = structure or StructuralSensor()

    decision = run_graph_query(store, request, spans, sentence, structure)
    if decision.decision != "ABSTAIN" or "no sensor could name it" not in decision.reason:
        return decision

    graph = build_graph(request, store, spans, sentence, structure)
    unnamed = readings(graph)[1]
    if unnamed is None:
        return decision
    label = lexicon.lookup(unnamed.text)
    if label is None:
        return decision

    from .argument_binding_l821_experiment import compile_bound
    from .query_algebra_l820_experiment import (
        TypeError_, _Empty, canonical_render, evaluate, infer,
    )

    solutions = {}
    for node in graph.of_kind("OPERATOR"):
        compiled = compile_bound(request, store, (node.label, (label,)))
        if compiled.expr is None:
            continue
        try:
            infer(compiled.expr)
        except TypeError_:
            continue
        solutions[canonical_render(compiled.expr)] = compiled.expr
    if len(solutions) != 1:
        return decision
    expr = next(iter(solutions.values()))
    try:
        return ProposedResult("ANSWER", evaluate(expr, store), expr,
                              f"evaluated with the acquired meaning of 「{unnamed.text}」",
                              None, "acquired")
    except (_Empty, TypeError_, ValueError, ZeroDivisionError) as error:
        return ProposedResult("ABSTAIN", None, expr, str(error), "precondition", "acquired")


# --------------------------------------------------------------------------
# grounding: the same narrowing, read off a real request
# --------------------------------------------------------------------------

def observations_from_request(
    request: str,
    store: TypedStore,
    spans: SpanProposer,
    sentence: Proposer,
    structure: StructuralSensor,
) -> tuple[str, tuple[Observation, ...]]:
    """What one actual sentence contributes, via L8.27's graph and L8.28's sources.

    This is what keeps the K-sweep from floating free: at K=2 and K=3 the
    constraints below are the ones the real pipeline produces, not stipulated
    ones that happen to behave well.
    """
    graph = build_graph(request, store, spans, sentence, structure)
    unnamed = readings(graph)[1]
    if unnamed is None:
        return "", ()
    found: list[Observation] = []
    admitted = admitted_by_type(graph)
    for value_type in (ValueType.NUMBER, ValueType.CATEGORICAL, ValueType.TEXT):
        if {f.name for f in FIELDS if f.value_type is value_type} & admitted:
            found.append(Observation("TYPE", value_type, "from the projection hole"))
            break
    for name in sorted(excluded_by_contrast(request, graph, structure)):
        found.append(Observation("CONTRAST", name, f"「{name}ではなく」"))
    for value in numeric_literals(request, store):
        found.append(Observation("RANGE", value, f"compared against {value}"))
    return unnamed.text, tuple(found)


def common_word(spans: tuple[str, ...]) -> str:
    """The longest substring every unnamed span shares.

    Cross-context narrowing needs the *same* word across episodes, and the graph
    hands back a different phrase each time -- 「0.5以上のフガ率」 in one request and
    「超えたフガ率」 in another.  The recurring substring is what they are both about.
    Returning "" when nothing recurs is the honest answer: two spans with nothing
    in common are not evidence about one word, and the caller must not merge them.
    """
    if not spans:
        return ""
    shortest = min(spans, key=len)
    for size in range(len(shortest), 0, -1):
        for start in range(len(shortest) - size + 1):
            piece = shortest[start:start + size]
            if all(piece in span for span in spans):
                return piece
    return ""


#: A decimal that is not part of a categorical literal such as 64x64 or a seed.
_LITERAL = re.compile(r"(?<![0-9xX.])([0-9]+\.[0-9]+)(?![0-9xX])")


def numeric_literals(request: str, store: TypedStore) -> tuple[float, ...]:
    """Decimals a request compares the unknown word against.

    Deliberately narrow: only decimals, and only ones no categorical value in the
    store already claims.  A lattice size or a seed is a name, not a magnitude,
    and reading one as a magnitude would manufacture constraints out of nothing.
    """
    claimed = {
        str(value) for _, values in store.rows() for value in values.values()
        if isinstance(value, str)
    }
    found = []
    for match in _LITERAL.finditer(request):
        text = match.group(1)
        if text in claimed or any(text in name for name in claimed):
            continue
        found.append(float(text))
    return tuple(dict.fromkeys(found))
