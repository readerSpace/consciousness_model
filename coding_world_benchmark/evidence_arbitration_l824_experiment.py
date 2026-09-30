"""L8.24: keep the evidence per candidate, and let the type system arbitrate.

L8.23.1 measured the two sensors as a tie and showed why: they read the same
text at different resolutions and are wrong about different things.  Spans are
sharp where a phrase is covered and blind where it is not; the sentence is
robust to thin overlap and cannot separate two pieces of evidence that sit in
one string.  Neither is the better witness, so choosing one -- even per slot --
would be one more heuristic of the kind this project keeps removing.

So nothing is chosen.  Each candidate carries where its support came from:

    Evidence(slot, candidate) = (span_score, span_coverage, whole_score)

and a candidate is *admitted* when at least one sensor supports it **strongly**,
by that sensor's own standard: a span that cleared coverage and length, or a
whole-sentence score above ``WHOLE_STRONG``.  Admitted candidates go to L8.21
together and the uniqueness rule decides, exactly as before.

The strength requirement is what separates this from a union, and L8.23.1 shows
why a union fails.  In 「…ランタイムの値をアベレージ」 the sentence reading is
MAX 0.516 against MEAN 0.484 -- contention, not evidence.  Merging candidate
sets would put MAX back beside the span's MEAN and restore the ambiguity that
grounding had resolved.  Requiring strength drops both halves of a contended
reading while keeping a decisive one, so the sentence contributes where it
actually knows something and stays quiet where it is merely guessing.

What this buys is the complementary case, which neither sensor could do alone:
in 「…成功の比率をならした値は？」 the operator comes from a span and the field from
the sentence, assembled from different sources into one program.

Conflict is the case that proves it is arbitration rather than tidy merging: if
a span says MAX strongly and the sentence says MEAN strongly, both are admitted,
two well-typed programs survive and the request is refused.  ``arbitrate`` is
exercised on constructed evidence in the tests for exactly this reason -- with
two sensors reading the same string, natural conflicts are rare, and a rule that
is only ever tested on agreement has not been tested.
"""
from __future__ import annotations

from dataclasses import dataclass

from .argument_binding_l821_experiment import compile_bound, run_bound_query
from .query_algebra_l820_experiment import (
    Expr, TypeError_, _Empty, canonical_render, evaluate, infer,
)
from .semantic_proposer_l822_experiment import Proposer, ProposedResult, _unreadable
from .span_grounded_proposer_l823_experiment import SpanProposer
from .typed_operation_l819_experiment import TypedStore

#: A whole-sentence reading counts as evidence only when it is decisive.  Below
#: this the two labels are contending over one string, which is the state that
#: produced the dilution failure; admitting it would undo span grounding.
WHOLE_STRONG = 0.75


@dataclass(frozen=True)
class SlotEvidence:
    slot: str  # OPERATOR | FIELD
    candidate: str
    span_score: float
    span_coverage: float
    whole_score: float

    @property
    def sources(self) -> tuple[str, ...]:
        found = []
        if self.span_coverage > 0.0:
            found.append("span")
        if self.whole_score >= WHOLE_STRONG:
            found.append("whole")
        return tuple(found)

    @property
    def admitted(self) -> bool:
        return bool(self.sources)


def gather_evidence(
    request: str, spans: SpanProposer, sentence: Proposer
) -> tuple[SlotEvidence, ...]:
    """Collect support for every candidate from both sensors, choosing nothing."""
    span_operators, span_fields = spans.propose(request)
    span_detail = {span.text: span for span in spans.detect(request)}

    def best_span(slot: str, label: str) -> tuple[float, float]:
        best = (0.0, 0.0)
        for span in span_detail.values():
            if span.kind == slot and label in span.labels:
                best = max(best, (span.score, span.coverage))
        return best

    whole_operators = dict(sentence.operator_model.scores(request))
    whole_fields = dict(sentence.field_model.scores(request))

    evidence: list[SlotEvidence] = []
    for slot, proposed, whole in (
        ("OPERATOR", span_operators, whole_operators),
        ("FIELD", span_fields, whole_fields),
    ):
        for label in dict.fromkeys(tuple(proposed) + tuple(whole)):
            score, coverage = best_span(slot, label)
            evidence.append(SlotEvidence(slot, label, score, coverage, whole.get(label, 0.0)))
    return tuple(evidence)


def arbitrate(evidence: tuple[SlotEvidence, ...]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Admitted candidates per slot.  No ranking, no choosing between sensors."""
    operators = tuple(
        item.candidate for item in evidence if item.slot == "OPERATOR" and item.admitted
    )
    fields = tuple(item.candidate for item in evidence if item.slot == "FIELD" and item.admitted)
    return operators, fields


def run_arbitrated_query(
    store: TypedStore,
    request: str,
    spans: SpanProposer | None = None,
    sentence: Proposer | None = None,
) -> ProposedResult:
    base = run_bound_query(store, request)
    if not _unreadable(base):
        return ProposedResult(base.decision, base.value, base.expr, base.reason, base.stage)

    spans = spans or SpanProposer.trained()
    sentence = sentence or Proposer.trained()
    evidence = gather_evidence(request, spans, sentence)
    operators, fields = arbitrate(evidence)

    if not operators:
        return ProposedResult("NOT_AN_OPERATION", None, None,
                              "no operation; no sensor supported one strongly")

    sources = {item.candidate: item.sources for item in evidence if item.admitted}
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

    witness = "+".join(sorted({source for item in sources.values() for source in item})) or "none"

    if not solutions:
        detail = "; ".join(refusals) or "no candidate compiled"
        return ProposedResult("ABSTAIN", None, None,
                              f"no unique program from the admitted evidence ({detail})",
                              "binding", witness)
    if len(solutions) > 1:
        return ProposedResult("ABSTAIN", None, None,
                              f"{len(solutions)} well-typed programs from the admitted evidence; not unique",
                              "binding", witness)

    expr = next(iter(solutions.values()))
    try:
        return ProposedResult("ANSWER", evaluate(expr, store), expr,
                              "evaluated from arbitrated evidence", None, witness)
    except (_Empty, TypeError_, ValueError, ZeroDivisionError) as error:
        return ProposedResult("ABSTAIN", None, expr, str(error), "precondition", witness)
