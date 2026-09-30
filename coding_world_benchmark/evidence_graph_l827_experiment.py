"""L8.27: candidates stop being a set and become a graph with evidence on its edges.

L8.26 ends with three sensors and one flat answer: two sets, ``operators`` and
``fields``, each a bag of labels that cleared somebody's bar.  That shape throws
away the only thing the structural sensor actually knows.  Structure does not
say "success_rate is plausible"; it says *this position* is the argument the head
governs.  Collapsing that into a label set loses the position, and with it the
ability to notice that the governed position **has no label at all**.

That loss is not cosmetic.  Measured on held-out vocabulary::

    「64x64のケースでランタイムの値ではなく成功の比率をならして」

the span sensor cannot read 成功の比率 (the trained forms are 成功比率 / 成功割合,
and the inserted の breaks every n-gram), so the only field it labels is the
*excluded* one.  The whole-sentence sensor splits 0.50/0.50 and contributes
nothing.  L8.24 and L8.26 both answer ``MEAN(runtime_seconds)`` -- confidently,
with a well-typed program, and wrong.  The evidence to refuse was present the
whole time: structure reports the head governing 「成功の比率」 at strength 1.0 while
the labeled span sits at 0.37.  A set cannot hold that sentence.

So the representation changes::

    MEAN --bind--> success_rate --filter--> 64x64

Nodes are *grounded* candidates -- a label at a span, or a span with no label.
Edges are roles, and every edge carries which sensors witnessed it, from which
span, at what strength.  Four consequences follow directly from the shape:

**An unnamed argument is visible.**  When the bind edge lands on a span no
lexical sensor could label, the graph holds an ``UNKNOWN`` node.  If the operator
needs a field, that is a refusal: the request named its argument and this system
could not read it.  Answering with some *other* field that happens to be legible
is the failure above, and the graph makes it unrepresentable rather than
unlikely.

**Attachment is read off syntax, not proximity.**  Two structural facts, both
label-free, decide which span the bind edge lands on:

* の links a modifier to its head noun, so 「サクセスレートの値」 is one argument, not
  a far 「サクセスレート」 and a near 「値」.  Without this the light noun 値 wins every
  contest and every ``Xの値を`` request refuses.
* を marks the direct argument.  When some phrase carries it, the oblique
  phrases -- 「Yのかわりに」, 「Yではなく」, 「Yに対して」 -- are not candidates, however
  close to the head they sit.  L8.26 ranked by distance alone and therefore read
  「Aを Bのかわりに ならす」 as being about B.

**A plural argument is visible.**  と coordinates, so 「AとBを」 is one argument
naming two things and the algebra has no node for that.  The edge remembers it
was built across と even when only one conjunct was legible -- the case where a
label set sees a perfectly ordinary single-field request and answers it.

**The sensors can complete each other across one edge.**  Structure says where
without saying what; the sentence says what without saying where.  When the head
governs a span nothing could label *and* exactly one ungrounded whole-sentence
field exists, the two halves determine the edge between them, and the edge is
witnessed by both.  Uniqueness is the entire safety argument: with two fields in
play the sentence score splits and no ungrounded node clears ``WHOLE_STRONG``,
so the repair is unavailable in precisely the case where it would be guessing.

Neither fact is about meaning and neither introduces a new tuned constant; both
are the same particle inventory L8.26 already segments on, used for what it is
for.  ``STRUCTURE_STRONG`` is not consulted here at all -- the graph compares
edges against each other, as L8.26 compared witness sets against each other.

Authority is unchanged.  The graph proposes one binding; L8.21 still compiles
it, L8.19's type check still runs, and L8.21's uniqueness rule still decides
whether anything executes.  Every refusal below is a refusal the graph *adds*.

One defect found on the way and worth recording, because it was invisible for
three layers: both refusals above were first keyed on L8.19's ``OPERATIONS``
table, which predates MAX.  ``needs_field("MAX")`` therefore answered False by
default and neither guard applied to a MAX request -- six wrong answers on the
generated sweep, from a default that failed open.  The test is written around
the direction, not the operator.

What is left undone is stated in ``holdout_graph_l8271_experiment``: the graph
can locate an argument it cannot name, and 12.5% of the sweep is exactly that --
an argument correctly found, correctly refused, and still unnamed.  Naming it
from the structure it sits in is lexicon acquisition, which is the next layer.
"""
from __future__ import annotations

from dataclasses import dataclass

from .argument_binding_l821_experiment import collect_mentions, compile_bound, run_bound_query
from .evidence_arbitration_l824_experiment import WHOLE_STRONG, gather_evidence
from .query_algebra_l820_experiment import (
    Expr, TypeError_, _Empty, canonical_render, evaluate, infer,
)
from .semantic_proposer_l822_experiment import Proposer, ProposedResult, _unreadable
from .span_grounded_proposer_l823_experiment import SpanProposer
from .structural_sensor_l826_experiment import (
    _ACCUSATIVE_BONUS, _DISTANCE_DECAY, StructuralSensor,
)
from .typed_operation_l819_experiment import OPERATIONS, TypedStore

#: Only COUNT takes no field.  The membership test is written as an exclusion on
#: purpose: L8.19's registry predates MAX, so a table keyed by operator name
#: answers ``False`` for it by default and both refusals below would silently not
#: apply.  An operator this layer has never heard of must count as needing an
#: argument, not as needing none.
_NO_FIELD = frozenset(spec.name for spec in OPERATIONS if not spec.needs_field)


def needs_field(operator: str) -> bool:
    return operator not in _NO_FIELD

#: The direct-argument marker.  Everything else in L8.26's inventory is oblique.
ACCUSATIVE = "を"
#: Links, not case markers: a chunk followed by one of these is part of the next
#: chunk's noun phrase, so the pair is one argument.  の subordinates (「Xの値」);
#: と coordinates (「XとY」).  Both merge; what differs is what the merged phrase
#: then contains, which is what ``readings`` reads off it.
LINKS = ("の", "と")
#: The coordinating link.  「XとY」 is one argument naming two things, and the
#: algebra has no plural projection node, so a coordinated argument is a refusal
#: however legible its conjuncts are -- including when only one of them is.
COORDINATOR = "と"


# --------------------------------------------------------------------------
# phrases: L8.26's chunks with modifier chains closed up
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Phrase:
    start: int
    end: int
    text: str
    particle: str
    strength: float = 0.0
    #: The links consumed while merging.  A phrase remembers that it was built
    #: across と even when only one conjunct turned out to be readable, which is
    #: the case a label set cannot represent at all.
    links: tuple[str, ...] = ()


def phrases(structure: StructuralSensor, text: str) -> tuple[Phrase, ...]:
    """Segment, then merge の-chains, then score attachment to the head."""
    merged: list[Phrase] = []
    for start, end, piece, particle in structure.chunks(text):
        if merged and merged[-1].particle in LINKS:
            head = merged.pop()
            merged.append(Phrase(head.start, end, text[head.start:end], particle,
                                 links=head.links + (head.particle,)))
        else:
            merged.append(Phrase(start, end, piece, particle))
    if len(merged) < 2:
        return ()
    scored: list[Phrase] = []
    for distance, phrase in enumerate(reversed(merged[:-1])):
        strength = _DISTANCE_DECAY ** distance
        if phrase.particle == ACCUSATIVE:
            strength = min(1.0, strength + _ACCUSATIVE_BONUS)
        scored.append(Phrase(phrase.start, phrase.end, phrase.text, phrase.particle,
                             round(strength, 4), phrase.links))
    return tuple(scored)


# --------------------------------------------------------------------------
# the graph
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Node:
    kind: str  # OPERATOR | FIELD | VALUE | UNKNOWN
    label: str | None
    start: int
    end: int
    text: str
    witnesses: frozenset[str]

    def __str__(self) -> str:
        where = "" if self.start < 0 else f"@{self.start}:{self.end}"
        return f"{self.label or '?'}{where}"


@dataclass(frozen=True)
class Edge:
    role: str  # bind | filter
    head: Node
    tail: Node
    witnesses: frozenset[str]
    strength: float
    particle: str = ""

    def __str__(self) -> str:
        marks = "+".join(sorted(self.witnesses)) or "none"
        return f"{self.head} --{self.role}[{marks} {self.strength:.2f}]--> {self.tail}"


@dataclass(frozen=True)
class EvidenceGraph:
    nodes: tuple[Node, ...]
    edges: tuple[Edge, ...]
    request: str
    #: True when the governed argument was built across と.
    coordinated: bool = False

    def of_kind(self, kind: str) -> tuple[Node, ...]:
        return tuple(node for node in self.nodes if node.kind == kind)

    def binds(self) -> tuple[Edge, ...]:
        return tuple(edge for edge in self.edges if edge.role == "bind")

    def render(self) -> str:
        return " ; ".join(str(edge) for edge in self.edges) or "(no edges)"


def _lexical_nodes(request: str, spans: SpanProposer, sentence: Proposer) -> list[Node]:
    grounded: list[Node] = []
    for span in spans.detect(request):
        for label in span.labels:
            grounded.append(Node(span.kind, label, span.start, span.end, span.text,
                                 frozenset({"span"})))
    # A whole-sentence reading is ungrounded by construction: it scores the
    # request, not a position.  It reinforces a span that already exists and
    # otherwise enters as a node that no bind edge can reach -- which is the
    # honest shape, since the sentence cannot say *where* its evidence is.
    nodes: list[Node] = []
    for item in gather_evidence(request, spans, sentence):
        if item.whole_score < WHOLE_STRONG:
            continue
        matches = [n for n in grounded if n.kind == item.slot and n.label == item.candidate]
        if matches:
            for node in matches:
                grounded[grounded.index(node)] = Node(
                    node.kind, node.label, node.start, node.end, node.text,
                    node.witnesses | {"whole"})
        else:
            nodes.append(Node(item.slot, item.candidate, -1, -1, request,
                              frozenset({"whole"})))
    return grounded + nodes


def build_graph(
    request: str,
    store: TypedStore,
    spans: SpanProposer,
    sentence: Proposer,
    structure: StructuralSensor,
) -> EvidenceGraph:
    nodes = _lexical_nodes(request, spans, sentence)

    for mention in collect_mentions(request, store).values:
        start = request.find(mention.text)
        nodes.append(Node("VALUE", mention.text, start, start + len(mention.text),
                          mention.text, frozenset({"literal"})))

    operators = [node for node in nodes if node.kind == "OPERATOR"]
    fields = [node for node in nodes if node.kind == "FIELD" and node.start >= 0]
    values = [node for node in nodes if node.kind == "VALUE"]

    def covers(phrase: Phrase, node: Node) -> bool:
        return node.start >= 0 and phrase.start <= node.start and node.end <= phrase.end

    def overlaps(phrase: Phrase, node: Node) -> bool:
        # Containment is the wrong test for excluding the operator: a form like
        # 「ピークを」 is detected as a span that runs past the chunk boundary, so
        # the operator chunk is not *inside* its own node.  Any overlap means the
        # phrase is the predicate, not its argument.
        return node.start >= 0 and phrase.start < node.end and node.start < phrase.end

    # Which phrase does the head govern?  Accusative marking wins outright; with
    # none present, distance decides as it did in L8.26.
    segmented = phrases(structure, request)
    # Coordination is a fact about the clause, not about whichever phrase wins:
    # 「成功の割合と処理にかかった時間を」 splits at the clause-internal に, so the と
    # ends up in a phrase the accusative rule then discards.  Any と linking two
    # pre-head phrases means the argument list has more than one member.
    #
    # Scope, stated plainly: this treats every pre-head と as coordination.  This
    # vocabulary has no other use for it -- comparisons reach the symbolic path
    # through 比べて and never arrive here -- but a request using と for
    # accompaniment or quotation would refuse where it need not.
    coordinated = any(COORDINATOR in phrase.links for phrase in segmented)
    candidates = [p for p in segmented
                  if not any(overlaps(p, node) for node in operators)]
    accusative = [p for p in candidates if p.particle == ACCUSATIVE]
    governed = accusative or candidates

    edges: list[Edge] = []
    for phrase in sorted(governed, key=lambda p: -p.strength)[:1] if governed else []:
        inside = [node for node in fields if covers(phrase, node)]
        arguments: list[Node]
        if inside:
            # One phrase may name more than one field -- 「処理時間と成功比率を」 is a
            # single coordinated argument.  Both are kept as bind targets, which
            # is how a plural argument reaches the uniqueness rule as a conflict
            # rather than as an arbitrary pick.
            by_label: dict[str, Node] = {}
            for node in inside:
                previous = by_label.get(node.label)
                if previous is None or node.end - node.start > previous.end - previous.start:
                    by_label[node.label] = node
            arguments = list(by_label.values())
        else:
            unknown = Node("UNKNOWN", None, phrase.start, phrase.end, phrase.text,
                           frozenset({"structure"}))
            nodes.append(unknown)
            arguments = [unknown]
        for argument in arguments:
            for operator in operators:
                edges.append(Edge("bind", operator, argument,
                                  frozenset({"structure"}) | argument.witnesses,
                                  phrase.strength, phrase.particle))
            # A literal is never the projection, so every value mention is a
            # restrictor -- including one that sits inside the argument phrase.
            for value in values:
                edges.append(Edge("filter", argument, value, value.witnesses, 1.0))

    return EvidenceGraph(tuple(nodes), tuple(edges), request, coordinated)


# --------------------------------------------------------------------------
# traversal
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Reading:
    operator: str
    field: str | None
    witnesses: frozenset[str]


def readings(graph: EvidenceGraph) -> tuple[tuple[Reading, ...], Node | None]:
    """Walk bind edges into (operator, argument) pairs; report an unnamed argument.

    One repair is allowed, and only one: when the head governs a span nothing
    could label *and* the whole-sentence sensor named exactly one field without
    being able to say where it was, the two halves determine each other.  The
    edge is then jointly witnessed by ``structure`` and ``whole``.  Uniqueness is
    the whole safety argument -- with two fields in play the sentence score
    splits and no ungrounded node clears ``WHOLE_STRONG``, so the repair is
    unavailable exactly where it would be guessing.
    """
    floating = [node for node in graph.of_kind("FIELD") if node.start < 0]
    unnamed: Node | None = None
    found: list[Reading] = []
    for edge in graph.binds():
        if edge.tail.kind == "UNKNOWN":
            if len(floating) == 1:
                found.append(Reading(edge.head.label, floating[0].label,
                                     edge.witnesses | edge.head.witnesses
                                     | floating[0].witnesses))
                continue
            unnamed = edge.tail
            continue
        found.append(Reading(edge.head.label, edge.tail.label,
                             edge.witnesses | edge.head.witnesses))
    if not found:
        for node in graph.of_kind("OPERATOR"):
            found.append(Reading(node.label, None, node.witnesses))
    return tuple(dict.fromkeys(found)), unnamed


def run_graph_query(
    store: TypedStore,
    request: str,
    spans: SpanProposer | None = None,
    sentence: Proposer | None = None,
    structure: StructuralSensor | None = None,
) -> ProposedResult:
    base = run_bound_query(store, request)
    if not _unreadable(base):
        return ProposedResult(base.decision, base.value, base.expr, base.reason, base.stage)

    spans = spans or SpanProposer.trained()
    sentence = sentence or Proposer.trained()
    structure = structure or StructuralSensor()

    graph = build_graph(request, store, spans, sentence, structure)
    found, unnamed = readings(graph)
    if not found:
        return ProposedResult("NOT_AN_OPERATION", None, None,
                              "no operation; no sensor supported one strongly")

    witness = "+".join(sorted({w for reading in found for w in reading.witnesses})) or "none"

    if graph.coordinated and any(needs_field(r.operator) for r in found):
        return ProposedResult(
            "ABSTAIN", None, None,
            "the head governs a coordinated argument; the algebra has no plural projection",
            "grounding", witness)

    # The argument the head governs carries no label.  Executing would mean
    # answering about a field the request did not ask for.
    if unnamed is not None and any(needs_field(r.operator) for r in found):
        return ProposedResult(
            "ABSTAIN", None, None,
            f"the head governs 「{unnamed.text}」 and no sensor could name it",
            "grounding", witness)

    solutions: dict[str, Expr] = {}
    refusals: list[str] = []
    for reading in found:
        proposed = (reading.operator, (reading.field,) if reading.field else ())
        compiled = compile_bound(request, store, proposed)
        if compiled.expr is None:
            refusals.append(f"{reading.operator}: {compiled.reason}")
            continue
        try:
            infer(compiled.expr)
        except TypeError_:
            continue
        solutions[canonical_render(compiled.expr)] = compiled.expr

    if not solutions:
        detail = "; ".join(refusals) or "no reading compiled"
        return ProposedResult("ABSTAIN", None, None,
                              f"no program from the evidence graph ({detail})", "binding", witness)
    if len(solutions) > 1:
        return ProposedResult("ABSTAIN", None, None,
                              f"{len(solutions)} well-typed programs in the evidence graph; not unique",
                              "binding", witness)

    expr = next(iter(solutions.values()))
    try:
        return ProposedResult("ANSWER", evaluate(expr, store), expr,
                              "evaluated from the evidence graph", None, witness)
    except (_Empty, TypeError_, ValueError, ZeroDivisionError) as error:
        return ProposedResult("ABSTAIN", None, expr, str(error), "precondition", witness)
