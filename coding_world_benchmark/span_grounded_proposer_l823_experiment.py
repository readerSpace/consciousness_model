"""L8.23: score the span that carries the evidence, not the whole request.

L8.22.1 left one probe refused that a reader would answer.
「64x64で走らせた分のランタイムの値をアベレージ」 scored MAX 0.516 against MEAN 0.484 and
runtime 0.513 against success_rate 0.487, and neither margin cleared.  Nothing
was wrong with the evidence: ランタイム names a field and アベレージ names an
operator.  The classifier was handed the entire sentence, so both pieces of
evidence were averaged together with 「64x64で走らせた分の」 until each was noise to
the other.

The variable is isolated deliberately.  **The model and its training data are
unchanged from L8.22** -- the same character n-gram classifier over the same
(surface form, label) pairs.  Only inference changes: candidate spans are
enumerated, each is scored on its own, and overlaps are resolved by score.  Any
difference measured here is therefore attributable to grounding the evidence
rather than to a better model, which is the claim worth being able to make.

Span detection is given no more authority than the proposer was.  It says which
substrings look like evidence and what each might mean; a span that keeps two
labels within the margin passes both downstream, and L8.21's uniqueness rule
still decides whether anything runs.  So the failure mode remains a refusal.

The span labels are not hand-written.  ``render_with_spans`` walks a program and
emits the sentence together with the span each node produced, so the alignment
that supervises the classifier falls out of the renderer:

    MEAN(PROJECT(FILTER(src, lattice_size == 64x64), runtime_seconds))
      -> 「64x64で走らせた分のランタイムをならす」
      -> ("64x64", VALUE), ("ランタイム", FIELD:runtime_seconds), ("ならす", OPERATOR:MEAN)

which is the point of keeping programs as the ground truth: they can supervise
the parts as well as the whole.
"""
from __future__ import annotations

from dataclasses import dataclass

from .argument_binding_l821_experiment import compile_bound, run_bound_query
from .query_algebra_l820_experiment import (
    Expr, Filter, Mean, Project, Source, TypeError_, _Empty, canonical_render, evaluate, infer,
)
from .semantic_proposer_l822_experiment import (
    FIELD_FORMS, OPERATOR_FORMS, NgramClassifier, ProposedResult, Proposer, _unreadable,
)
from .typed_operation_l819_experiment import TypedStore

SPAN_FLOOR = 0.60
SPAN_MARGIN = 0.20
#: A span is evidence only when most of it is vocabulary the label knows.
SPAN_COVERAGE = 0.60
MAX_SPAN = 9
#: Two characters cannot carry evidence: 「の値」 is fully covered by MAX because
#: 頭打ちの値 is a training form, and 「レー」 by success_rate because of サクセスレート.
#: Coverage is trivially 1.0 at that length, so the floor is a length floor too.
#: The shortest trained form is three characters (ならす, 数えて), so nothing is lost.
MIN_SPAN = 3


@dataclass(frozen=True)
class Span:
    text: str
    start: int
    end: int
    kind: str  # OPERATOR | FIELD
    labels: tuple[str, ...]
    score: float
    coverage: float


# --------------------------------------------------------------------------
# alignment oracle: a program renders itself together with its span labels
# --------------------------------------------------------------------------

def render_with_spans(
    program: Expr, operator_form: str, field_form: str, value: str
) -> tuple[str, tuple[tuple[str, str, str], ...]]:
    """Render one program and report which substring came from which node."""
    if isinstance(program, Mean) and isinstance(program.source, Project):
        text = f"{value}で走らせた分の{field_form}を{operator_form}"
        return text, (
            (value, "VALUE", "lattice_size"),
            (field_form, "FIELD", program.source.field),
            (operator_form, "OPERATOR", "MEAN"),
        )
    raise TypeError_("renderer covers MEAN(PROJECT(FILTER(...))) only")


def build_aligned_training() -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Supervision for the span classifier, produced from programs.

    No span was annotated by hand: the renderer knows which substring it emitted
    for which node, so walking the training forms yields the pairs directly.
    """
    operators: list[tuple[str, str]] = []
    fields: list[tuple[str, str]] = []
    for operator_label, operator_split in OPERATOR_FORMS.items():
        for operator_form in operator_split["train"]:
            for field_label, field_split in FIELD_FORMS.items():
                for field_form in field_split["train"]:
                    program = Mean(Project(Filter(Source(), "lattice_size", "64x64"), field_label))
                    try:
                        _, spans = render_with_spans(program, operator_form, field_form, "64x64")
                    except TypeError_:  # pragma: no cover - renderer is total here
                        continue
                    for span_text, kind, label in spans:
                        if kind == "OPERATOR":
                            operators.append((span_text, operator_label))
                        elif kind == "FIELD":
                            fields.append((span_text, label))
    return operators, fields


@dataclass
class SpanProposer:
    operator_model: NgramClassifier
    field_model: NgramClassifier

    @classmethod
    def trained(cls) -> "SpanProposer":
        # Same classifier and the same forms L8.22 trained on, reached through
        # the alignment oracle instead of a hand-written list.  The variable
        # under test is inference, not the model.
        operators, fields = build_aligned_training()
        return cls(NgramClassifier.train(operators), NgramClassifier.train(fields))

    @staticmethod
    def _labels(scored: list[tuple[str, float]]) -> tuple[tuple[str, ...], float]:
        if not scored:
            return (), 0.0
        top = scored[0][1]
        return tuple(label for label, score in scored if top - score <= SPAN_MARGIN), top

    def _candidates(self, text: str, kind: str) -> list[Span]:
        model = self.operator_model if kind == "OPERATOR" else self.field_model
        found: list[Span] = []
        for start in range(len(text)):
            for size in range(MIN_SPAN, MAX_SPAN + 1):
                end = start + size
                if end > len(text):
                    break
                piece = text[start:end]
                labels, score = self._labels(model.scores(piece))
                if not labels or score < SPAN_FLOOR:
                    continue
                coverage = max(model.coverage(piece, label) for label in labels)
                if coverage < SPAN_COVERAGE:
                    continue
                found.append(Span(piece, start, end, kind, labels, score, coverage))
        return found

    @staticmethod
    def _resolve(candidates: list[Span]) -> list[Span]:
        """Greedy non-overlapping selection, strongest evidence first."""
        chosen: list[Span] = []
        # Strongest evidence first: coverage before relative score, then the
        # longer span, so a well-covered phrase wins over a fragment of it.
        for span in sorted(candidates, key=lambda s: (-s.coverage, -s.score, -(s.end - s.start), s.start)):
            if any(span.start < other.end and other.start < span.end for other in chosen):
                continue
            chosen.append(span)
        return sorted(chosen, key=lambda s: s.start)

    def detect(self, text: str) -> tuple[Span, ...]:
        spans = self._resolve(self._candidates(text, "OPERATOR")) + self._resolve(
            self._candidates(text, "FIELD")
        )
        return tuple(sorted(spans, key=lambda s: (s.start, s.kind)))

    def propose(self, text: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
        spans = self.detect(text)
        operators: list[str] = []
        fields: list[str] = []
        for span in spans:
            target = operators if span.kind == "OPERATOR" else fields
            for label in span.labels:
                if label not in target:
                    target.append(label)
        return tuple(operators), tuple(fields)


def run_span_query(
    store: TypedStore,
    request: str,
    proposer: SpanProposer | None = None,
    fallback: "Proposer | None" = None,
) -> ProposedResult:
    """Span evidence first, whole-sentence evidence as the fallback.

    Grounding cuts both ways, and the trade-off is worth stating.  Scoring a
    span sharpens the evidence -- 「アベレージ」 read on its own is unambiguous
    where the sentence was not -- but it also demands that the span be mostly
    known vocabulary, and 「数を教えて」 shares a single bigram with 「数えて」.
    L8.22 read it because the whole sentence still leaned that way; L8.23 alone
    would refuse it.

    So the two are used in precedence, not in union.  Union would be the obvious
    move and is wrong here: the whole-sentence reading of the dilution case
    keeps MAX alongside MEAN, so merging the candidate sets would restore the
    very ambiguity spans resolve.  Where a span carries evidence it is the
    better witness and decides the candidate set; where none does, the sentence
    is all there is.  Neither is authority -- the uniqueness rule still runs.
    """
    base = run_bound_query(store, request)
    if not _unreadable(base):
        return ProposedResult(base.decision, base.value, base.expr, base.reason, base.stage)

    proposer = proposer or SpanProposer.trained()
    operators, fields = proposer.propose(request)
    source = "span"
    if not operators:
        fallback = fallback or Proposer.trained()
        operators, fields = fallback.operators(request), fallback.fields(request)
        source = "sentence"
    if not operators:
        return ProposedResult("NOT_AN_OPERATION", None, None,
                              "no operation; neither a span nor the sentence carried evidence")

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
                              f"no unique program from the spans ({detail})", "binding", source)
    if len(solutions) > 1:
        return ProposedResult("ABSTAIN", None, None,
                              f"{len(solutions)} well-typed programs from the spans; not unique",
                              "binding", source)

    expr = next(iter(solutions.values()))
    try:
        return ProposedResult("ANSWER", evaluate(expr, store), expr, "evaluated from span evidence", None, source)
    except (_Empty, TypeError_, ValueError, ZeroDivisionError) as error:
        return ProposedResult("ABSTAIN", None, expr, str(error), "precondition", source)
