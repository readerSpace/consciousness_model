"""L8.28: hypothesise what an unnamed span means, and promote nothing.

L8.27 ends with a node that is an admission of ignorance.  When the head governs
a phrase no lexical sensor can label, the graph holds an ``UNKNOWN`` and the
request is refused -- correctly, because answering with some other legible field
is the confidently-wrong failure that layer was built to remove.  On the L8.27.1
sweep that is 12.5% of the probes: **the argument was located, the refusal was
right, and the word is still unknown.**

Acquiring the word is two capabilities, not one, and this layer deliberately
implements only the first:

1. **hypothesis generation** -- from the span's position in the graph, produce a
   ranked set of labels it could denote, with the evidence for each.
2. **promotion** -- decide that one of them is now part of the lexicon.

Keeping them apart is the whole design.  Measured together, a good generator
with an unsafe promoter and a bad generator with a cautious one score the same,
and neither number says which half is working.  Measured apart, generation is
scored on recall and rank, promotion on what it would cost -- and this layer can
report that cost (``decidable_precision``: if a later stage promoted every
hypothesis this one finds unique, how often would it be wrong?) without paying
it.  The lexicon is not written to, the classifiers are not retrained, and
``run_hypothesis_query`` returns L8.27's decision **unchanged, by construction**:
it calls ``run_graph_query`` and attaches hypotheses beside the answer rather
than into it.  ``test_generation_cannot_change_a_single_decision`` pins that.

That separation buys something concrete and not just tidy.  A hypothesis may
rest on evidence far too weak to execute on, because it is not executing.  The
span sensor rejects 「成功の比率」 at coverage below ``SPAN_COVERAGE`` and must, since
acting on thin overlap is how L8.23 learned to dilute.  The same thin overlap is
perfectly good grounds for *suspecting* success_rate, once suspecting costs
nothing.  Four sources, each independently ablatable:

``type``
    The operator's projection hole admits certain value types (L8.21's
    ``_PROJECTION_TYPES`` over L8.19's ``FIELDS``).  MEAN's argument is a Number,
    so the span is not ``lattice_size`` whatever it looks like.  Costs no
    vocabulary at all -- this is the type system reading the graph.

``contrast``
    「Xではなく Y」 and 「Yを Xのかわりに」 assert that Y is *not* X.  When X is a span
    the sensors did label, one candidate is excluded by construction.  Also
    label-free: the particle says the relation, the known span supplies the term.

``attestation``
    ``energy_error`` is declared in the schema and never recorded by any
    generator.  A field with no values in the store is a poor hypothesis for a
    phrase someone is asking to average.  Evidence from the world, not language.

``subword``
    Character n-gram overlap with the trained forms, scored over every substring
    of the unknown phrase and with **no floor**.  This is the source that must be
    ablated to answer the question the next layer actually cares about: whether
    a word with *zero* overlap can be acquired from structure alone.
"""
from __future__ import annotations

from dataclasses import dataclass

from .argument_binding_l821_experiment import _PROJECTION_TYPES
from .evidence_graph_l827_experiment import (
    ACCUSATIVE, EvidenceGraph, Node, build_graph, phrases, readings, run_graph_query,
)
from .semantic_proposer_l822_experiment import Proposer, ProposedResult
from .span_grounded_proposer_l823_experiment import MAX_SPAN, MIN_SPAN, SpanProposer
from .structural_sensor_l826_experiment import StructuralSensor
from .typed_operation_l819_experiment import FIELDS, TypedStore, ValueType

_FIELDS_BY_NAME = {spec.name: spec for spec in FIELDS}

#: Particles that assert a difference rather than a role.  「Xではなく Y」 says the
#: argument is not X; 「Yを Xのかわりに」 says the same with the terms swapped.
CONTRASTIVE = ("ではなく", "のかわりに")

SOURCES = ("type", "contrast", "attestation", "subword")


@dataclass(frozen=True)
class Hypothesis:
    """What an unnamed span might denote, and why."""

    span: str
    label: str
    sources: frozenset[str]
    overlap: float  # best subword coverage found, 0.0 when none

    @property
    def rank_key(self) -> tuple:
        return (-len(self.sources), -self.overlap, self.label)

    def __str__(self) -> str:
        marks = "+".join(sorted(self.sources)) or "none"
        return f"「{self.span}」 ~ {self.label} [{marks} {self.overlap:.2f}]"


# --------------------------------------------------------------------------
# the four sources
# --------------------------------------------------------------------------

def admitted_by_type(graph: EvidenceGraph) -> frozenset[str]:
    """Fields whose value type can fill the projection hole of some operator here."""
    found: set[str] = set()
    for node in graph.of_kind("OPERATOR"):
        for value_type in _PROJECTION_TYPES.get(node.label, ()):
            found |= {spec.name for spec in FIELDS if spec.value_type is value_type}
    if not found:  # no operator with a typed hole; the type system says nothing
        return frozenset(spec.name for spec in FIELDS)
    return frozenset(found)


def excluded_by_contrast(
    request: str, graph: EvidenceGraph, structure: StructuralSensor
) -> frozenset[str]:
    """Labels a contrastive construction rules out.

    The construction is read off particles alone.  Which term it rules out comes
    from a span the lexical sensors *did* label, so nothing here knows what
    either phrase means -- only that the argument is not the other one.
    """
    labelled = [node for node in graph.of_kind("FIELD") if node.start >= 0]
    excluded: set[str] = set()
    for phrase in phrases(structure, request):
        if phrase.particle not in CONTRASTIVE:
            continue
        for node in labelled:
            if phrase.start <= node.start and node.end <= phrase.end:
                excluded.add(node.label)
    return frozenset(excluded)


def attested(store: TypedStore) -> frozenset[str]:
    """Fields any episode actually recorded a value for."""
    found: set[str] = set()
    for _, values in store.rows():
        found |= set(values)
    return frozenset(found)


def subword_overlap(text: str, spans: SpanProposer) -> dict[str, float]:
    """Best n-gram coverage per label over every substring of the unknown phrase.

    No floor.  The span sensor's ``SPAN_COVERAGE`` exists to stop thin overlap
    from *deciding* anything and stays exactly where it is; this reads the same
    quantity for a purpose where being wrong is free.
    """
    best: dict[str, float] = {}
    model = spans.field_model
    for start in range(len(text)):
        for size in range(MIN_SPAN, MAX_SPAN + 1):
            end = start + size
            if end > len(text):
                break
            piece = text[start:end]
            for label in model.counts:
                value = model.coverage(piece, label)
                if value > best.get(label, 0.0):
                    best[label] = value
    return {label: value for label, value in best.items() if value > 0.0}


# --------------------------------------------------------------------------
# generation
# --------------------------------------------------------------------------

def propose_hypotheses(
    request: str,
    store: TypedStore,
    spans: SpanProposer,
    sentence: Proposer,
    structure: StructuralSensor,
    disable: tuple[str, ...] = (),
) -> tuple[Hypothesis, ...]:
    """Ranked labels for the span the head governs but nothing could name.

    Returns empty whenever there is no unnamed argument -- a request the sensors
    read perfectly well generates no hypotheses, which is the control that keeps
    this from being a machine for inventing meanings.
    """
    graph = build_graph(request, store, spans, sentence, structure)
    unnamed = readings(graph)[1]
    if unnamed is None:
        return ()

    by_type = admitted_by_type(graph) if "type" not in disable else frozenset(
        spec.name for spec in FIELDS)
    excluded = excluded_by_contrast(request, graph, structure) if "contrast" not in disable \
        else frozenset()
    recorded = attested(store) if "attestation" not in disable else frozenset(
        spec.name for spec in FIELDS)
    overlaps = subword_overlap(unnamed.text, spans) if "subword" not in disable else {}

    found: list[Hypothesis] = []
    for spec in FIELDS:
        if spec.name in excluded:
            continue
        if spec.name not in by_type:
            continue
        if spec.name not in recorded:
            continue
        sources = {"type", "attestation"} - set(disable)
        if excluded and "contrast" not in disable:
            sources.add("contrast")
        overlap = overlaps.get(spec.name, 0.0)
        if overlap > 0.0:
            sources.add("subword")
        found.append(Hypothesis(unnamed.text, spec.name, frozenset(sources), overlap))

    # A phrase in the schema's own vocabulary is not the interesting case, but it
    # must not be dropped either: ranking is by evidence, not by novelty.
    return tuple(sorted(found, key=lambda item: item.rank_key))


@dataclass(frozen=True)
class HypothesisResult:
    """L8.27's answer, with hypotheses standing beside it and never inside it."""

    result: ProposedResult
    hypotheses: tuple[Hypothesis, ...]

    @property
    def decidable(self) -> bool:
        return len(self.hypotheses) == 1

    @property
    def top(self) -> Hypothesis | None:
        return self.hypotheses[0] if self.hypotheses else None


def run_hypothesis_query(
    store: TypedStore,
    request: str,
    spans: SpanProposer | None = None,
    sentence: Proposer | None = None,
    structure: StructuralSensor | None = None,
    disable: tuple[str, ...] = (),
) -> HypothesisResult:
    """Run L8.27 unchanged, then hypothesise about whatever it could not name.

    The decision is *literally* L8.27's -- this function has no branch that can
    alter it.  That is the separation the layer is for, expressed as control flow
    rather than as a promise in a docstring.
    """
    spans = spans or SpanProposer.trained()
    sentence = sentence or Proposer.trained()
    structure = structure or StructuralSensor()
    decision = run_graph_query(store, request, spans, sentence, structure)
    return HypothesisResult(
        decision,
        propose_hypotheses(request, store, spans, sentence, structure, disable),
    )
