"""L8.26: a third sensor that reads relations and never reads meaning.

L8.25 showed calibration sitting in the right place with nothing to do: the two
lexical sensors are near-bimodal, so the band where an estimated boundary would
differ from a constant is barely populated.  The bottleneck was never the
threshold.  It was that both sensors read the same thing -- which strings look
like which labels -- so they agree, or both fall silent, and their evidence
vector has one real dimension.

This sensor is built to be unable to do that.  **It is never told what a span
means.**  It segments on particles, takes the final chunk as the head (Japanese
is head-final) and reports only

    BIND(operator chunk, argument chunk, strength)

so it can say 「ならした」 attaches to 「成功比率」 without having any idea that one
is MEAN and the other is a success rate.  Labels stay entirely with the lexical
sensors; structure only says which of *their* candidates the head actually
governs.  That is what makes it heterogeneous rather than a second opinion.

Fusion changes shape accordingly.  L8.24 admitted every strongly-supported
candidate and let uniqueness decide, which is safe and, with two agreeing
sensors, often blind: two field spans in one request are both admitted and the
request is refused even when the sentence plainly attaches the operator to one
of them.  So a candidate is now also discarded when its supporting sensors are a
**strict subset** of a rival's.  A candidate that nothing supports which does
not also support a competitor is dominated, and dropping it is comparison, not
authority -- no sensor outranks another, and equal-sized disagreeing supports
are both kept, which is how conflict survives to be refused.

Strength is graded by distance and by the particle before the head, so unlike
the lexical scores it is genuinely continuous.  That matters beyond this layer:
it is the first evidence source whose reliability can vary inside the reachable
range, which is the condition L8.25 needs before calibration can do anything.
"""
from __future__ import annotations

from dataclasses import dataclass
import re

from .argument_binding_l821_experiment import compile_bound, run_bound_query
from .evidence_arbitration_l824_experiment import SlotEvidence, WHOLE_STRONG, gather_evidence
from .query_algebra_l820_experiment import (
    Expr, TypeError_, _Empty, canonical_render, evaluate, infer,
)
from .semantic_proposer_l822_experiment import Proposer, ProposedResult, _unreadable
from .span_grounded_proposer_l823_experiment import SpanProposer
from .typed_operation_l819_experiment import TypedStore

#: Case and topic markers.  Segmentation only -- no lexical knowledge is used.
_PARTICLES = ("のかわりに", "ではなく", "について", "に対して", "を", "の", "は", "で", "に", "と", "、")
_SPLIT = re.compile("|".join(re.escape(particle) for particle in _PARTICLES))

#: Attachment strength by how far the chunk sits from the head, plus a bonus when
#: the chunk is marked accusative.  Continuous by construction.
_DISTANCE_DECAY = 0.72
_ACCUSATIVE_BONUS = 0.25


@dataclass(frozen=True)
class Bind:
    start: int
    end: int
    text: str
    strength: float


@dataclass
class StructuralSensor:
    """Relations only.  This class contains no label vocabulary of any kind."""

    def chunks(self, text: str) -> list[tuple[int, int, str, str]]:
        found, position = [], 0
        for match in _SPLIT.finditer(text):
            piece = text[position:match.start()]
            if piece:
                found.append((position, match.start(), piece, match.group()))
            position = match.end()
        if position < len(text):
            found.append((position, len(text), text[position:], ""))
        return found

    def bindings(self, text: str) -> tuple[Bind, ...]:
        chunks = self.chunks(text)
        if len(chunks) < 2:
            return ()
        binds = []
        # The head is the final chunk; everything before it is a candidate
        # argument, weaker the further away it sits.
        for distance, (start, end, piece, particle) in enumerate(reversed(chunks[:-1])):
            strength = _DISTANCE_DECAY ** distance
            if particle == "を":
                strength = min(1.0, strength + _ACCUSATIVE_BONUS)
            binds.append(Bind(start, end, piece, round(strength, 4)))
        return tuple(sorted(binds, key=lambda bind: -bind.strength))

    def support(self, text: str, start: int, end: int) -> float:
        for bind in self.bindings(text):
            if bind.start <= start and end <= bind.end:
                return bind.strength
        return 0.0


STRUCTURE_STRONG = 0.80


@dataclass(frozen=True)
class FusedEvidence:
    slot: str
    candidate: str
    lexical: SlotEvidence
    structure_score: float

    @property
    def sources(self) -> frozenset[str]:
        found = set()
        if self.lexical.span_coverage > 0.0:
            found.add("span")
        if self.lexical.whole_score >= WHOLE_STRONG:
            found.add("whole")
        if self.structure_score >= STRUCTURE_STRONG:
            found.add("structure")
        return frozenset(found)


def fuse(
    request: str, spans: SpanProposer, sentence: Proposer, structure: StructuralSensor
) -> tuple[FusedEvidence, ...]:
    """Lexical evidence, plus how strongly the head governs each candidate's span."""
    detected = spans.detect(request)
    fused = []
    for item in gather_evidence(request, spans, sentence):
        best = 0.0
        for span in detected:
            if span.kind == item.slot and item.candidate in span.labels:
                best = max(best, structure.support(request, span.start, span.end))
        fused.append(FusedEvidence(item.slot, item.candidate, item, best))
    return tuple(fused)


def arbitrate_fused(evidence: tuple[FusedEvidence, ...]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Admit strongly-supported candidates, then drop the dominated ones."""
    admitted = [item for item in evidence if item.sources]
    kept = []
    for item in admitted:
        dominated = any(
            other is not item
            and other.slot == item.slot
            and item.sources < other.sources  # strict subset
            for other in admitted
        )
        if not dominated:
            kept.append(item)
    operators = tuple(dict.fromkeys(i.candidate for i in kept if i.slot == "OPERATOR"))
    fields = tuple(dict.fromkeys(i.candidate for i in kept if i.slot == "FIELD"))
    return operators, fields


def run_structural_query(
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

    evidence = fuse(request, spans, sentence, structure)
    operators, fields = arbitrate_fused(evidence)
    if not operators:
        return ProposedResult("NOT_AN_OPERATION", None, None,
                              "no operation; no sensor supported one strongly")

    witnesses = sorted({s for item in evidence if item.sources for s in item.sources})
    witness = "+".join(witnesses) or "none"

    solutions: dict[str, Expr] = {}
    refusals: list[str] = []
    for operator in operators:
        compiled = compile_bound(request, store, (operator, fields))
        if compiled.expr is None:
            refusals.append(f"{operator}: {compiled.reason}")
            continue
        try:
            infer(compiled.expr)
        except TypeError_:
            continue
        solutions[canonical_render(compiled.expr)] = compiled.expr

    if not solutions:
        detail = "; ".join(refusals) or "no candidate compiled"
        return ProposedResult("ABSTAIN", None, None,
                              f"no unique program from fused evidence ({detail})", "binding", witness)
    if len(solutions) > 1:
        return ProposedResult("ABSTAIN", None, None,
                              f"{len(solutions)} well-typed programs from fused evidence; not unique",
                              "binding", witness)

    expr = next(iter(solutions.values()))
    try:
        return ProposedResult("ANSWER", evaluate(expr, store), expr,
                              "evaluated from fused evidence", None, witness)
    except (_Empty, TypeError_, ValueError, ZeroDivisionError) as error:
        return ProposedResult("ABSTAIN", None, expr, str(error), "precondition", witness)
