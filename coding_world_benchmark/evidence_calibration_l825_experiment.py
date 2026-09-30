"""L8.25: learn how far each sensor may be trusted, instead of choosing a number.

L8.24 admits a whole-sentence reading when its score clears ``WHOLE_STRONG =
0.75``.  That constant works, and it was read off one observed distribution:
decisive readings landed near 1.0 and contended ones near 0.5, so 0.75 sat in
the gap.  Nothing guarantees the gap is there in another vocabulary.  With
labels whose surface forms share substrings -- 実行時間 beside 実行回数 -- every
reading carries some evidence for its neighbour, the whole distribution slides
down, and 0.75 stops meaning "decisive" and starts meaning "rejected".

So the constant is replaced by an estimate:

    P(candidate is correct | sensor, slot, score band, coverage band)

measured from observations the system can generate for itself.  A program is
rendered, both sensors read the rendering, and each candidate they propose is
checked against the node the renderer actually emitted -- the alignment oracle
from L8.23 used for supervision of a different kind.  No reliability was
hand-assigned, and re-estimating on a new vocabulary produces a new table rather
than inheriting a threshold from the old one.

Calibration is admission only.  It decides which candidates reach L8.21, and
L8.21's uniqueness rule still decides whether anything runs, so a
mis-estimated reliability costs coverage and cannot cost correctness -- the same
boundary the learned proposer was given in L8.22.

Two properties are worth stating because a single global threshold cannot
express either:

* reliability is **per sensor**.  A span at coverage 0.8 and a sentence at score
  0.8 are not equally trustworthy, and the table says so in numbers.
* reliability is **per vocabulary**.  The band that is trustworthy in one domain
  need not be in another, which is exactly what a fixed constant assumes away.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from .argument_binding_l821_experiment import compile_bound, run_bound_query
from .evidence_arbitration_l824_experiment import SlotEvidence, gather_evidence
from .query_algebra_l820_experiment import (
    Expr, Filter, Mean, Project, Source, TypeError_, _Empty, canonical_render, evaluate, infer,
)
from .semantic_proposer_l822_experiment import NgramClassifier, Proposer, ProposedResult, _unreadable
from .span_grounded_proposer_l823_experiment import SpanProposer, render_with_spans
from .typed_operation_l819_experiment import TypedStore

#: A candidate is admitted when its estimated reliability clears this.  Unlike a
#: score threshold this is a statement about outcomes, so it carries across
#: vocabularies: "admit what is right at least four times in five".
RELIABILITY_FLOOR = 0.80
#: Bands with less support than this are not trusted in either direction.
MIN_SUPPORT = 3
BAND = 0.2


def band(value: float) -> float:
    return min(1.0, round(int(value / BAND) * BAND, 2))


@dataclass
class ReliabilityTable:
    """Estimated precision per (sensor, slot, score band, coverage band)."""

    correct: dict[tuple, int] = field(default_factory=lambda: defaultdict(int))
    total: dict[tuple, int] = field(default_factory=lambda: defaultdict(int))

    def observe(self, key: tuple, was_correct: bool) -> None:
        self.total[key] += 1
        if was_correct:
            self.correct[key] += 1

    def reliability(self, key: tuple) -> float | None:
        support = self.total.get(key, 0)
        if support < MIN_SUPPORT:
            return None
        return (self.correct.get(key, 0) + 0.5) / (support + 1.0)

    def admits(self, evidence: SlotEvidence) -> tuple[bool, tuple[str, ...]]:
        sources = []
        for sensor, score, coverage in (
            ("span", evidence.span_score, evidence.span_coverage),
            ("whole", evidence.whole_score, 0.0),
        ):
            if sensor == "span" and coverage <= 0.0:
                continue
            if sensor == "whole" and score <= 0.0:
                continue
            value = self.reliability((sensor, evidence.slot, band(score), band(coverage)))
            if value is not None and value >= RELIABILITY_FLOOR:
                sources.append(sensor)
        return bool(sources), tuple(sources)


@dataclass(frozen=True)
class FormsBank:
    """One vocabulary: the surface forms each label is rendered with."""

    name: str
    operators: dict[str, tuple[str, ...]]
    fields: dict[str, tuple[str, ...]]

    def training_pairs(self) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
        operators = [(form, label) for label, forms in self.operators.items() for form in forms]
        fields = [(form, label) for label, forms in self.fields.items() for form in forms]
        return operators, fields

    def sensors(self) -> tuple[SpanProposer, Proposer]:
        operators, fields = self.training_pairs()
        span = SpanProposer(NgramClassifier.train(operators), NgramClassifier.train(fields))
        whole = Proposer(NgramClassifier.train(operators), NgramClassifier.train(fields))
        return span, whole


#: The vocabulary L8.22-24 were built on: forms that barely overlap across labels.
SEPARATED = FormsBank(
    "separated",
    {
        "MEAN": ("ならす", "ならした", "ならす形で", "アベレージ値", "アベレージを出す"),
        "COUNT": ("数えて", "数える", "カウント数", "カウントする"),
        "MAX": ("ピーク値", "ピークを出す", "頭打ちの値"),
    },
    {
        "runtime_seconds": ("処理時間", "所要時間", "かかった時間", "ランタイム"),
        "success_rate": ("成功割合", "成功比率", "サクセスレート"),
    },
)

#: A vocabulary whose labels share substrings on purpose -- 実行時間 beside 実行回数,
#: 集計 in two operators.  Every reading now carries evidence for its neighbour,
#: so the score distribution slides down and a threshold tuned on SEPARATED
#: rejects readings that are in fact reliable here.
ENTANGLED = FormsBank(
    "entangled",
    {
        "MEAN": ("平坦化集計", "平坦化して集計", "平坦化"),
        "COUNT": ("個数集計", "個数を集計", "個数化"),
        "MAX": ("上限集計", "上限を集計", "上限化"),
    },
    {
        "runtime_seconds": ("実行時間", "実行所要", "実行の時間"),
        "success_rate": ("実行成功率", "実行成功割合", "実行の成功率"),
    },
)


def calibration_renderings(bank: FormsBank) -> list[tuple[str, str, str]]:
    """(text, operator label, field label) generated from programs, not written."""
    rows = []
    for operator_label, operator_forms in bank.operators.items():
        for operator_form in operator_forms:
            for field_label, field_forms in bank.fields.items():
                for field_form in field_forms:
                    program = Mean(Project(Filter(Source(), "lattice_size", "64x64"), field_label))
                    text, _ = render_with_spans(program, operator_form, field_form, "64x64")
                    rows.append((text, operator_label, field_label))
    return rows


def calibrate(bank: FormsBank, store: TypedStore) -> ReliabilityTable:
    """Estimate reliability by checking proposals against the rendered program."""
    span, whole = bank.sensors()
    table = ReliabilityTable()
    truth = {"OPERATOR": None, "FIELD": None}
    for text, operator_label, field_label in calibration_renderings(bank):
        truth["OPERATOR"], truth["FIELD"] = operator_label, field_label
        for evidence in gather_evidence(text, span, whole):
            was_correct = evidence.candidate == truth[evidence.slot]
            if evidence.span_coverage > 0.0:
                table.observe(
                    ("span", evidence.slot, band(evidence.span_score), band(evidence.span_coverage)),
                    was_correct,
                )
            if evidence.whole_score > 0.0:
                table.observe(
                    ("whole", evidence.slot, band(evidence.whole_score), band(0.0)), was_correct
                )
    return table


def run_calibrated_query(
    store: TypedStore,
    request: str,
    bank: FormsBank,
    table: ReliabilityTable,
    sensors: tuple[SpanProposer, Proposer] | None = None,
) -> ProposedResult:
    base = run_bound_query(store, request)
    if not _unreadable(base):
        return ProposedResult(base.decision, base.value, base.expr, base.reason, base.stage)

    span, whole = sensors or bank.sensors()
    evidence = gather_evidence(request, span, whole)

    operators, fields, witnesses = [], [], set()
    for item in evidence:
        admitted, sources = table.admits(item)
        if not admitted:
            continue
        witnesses.update(sources)
        (operators if item.slot == "OPERATOR" else fields).append(item.candidate)

    if not operators:
        return ProposedResult("NOT_AN_OPERATION", None, None,
                              "no operation; no sensor was reliable enough here")

    solutions: dict[str, Expr] = {}
    refusals: list[str] = []
    for operator in operators:
        compiled = compile_bound(request, store, (operator, tuple(fields)))
        if compiled.expr is None:
            refusals.append(f"{operator}: {compiled.reason}")
            continue
        try:
            infer(compiled.expr)
        except TypeError_:
            continue
        solutions[canonical_render(compiled.expr)] = compiled.expr

    witness = "+".join(sorted(witnesses)) or "none"
    if not solutions:
        detail = "; ".join(refusals) or "no candidate compiled"
        return ProposedResult("ABSTAIN", None, None,
                              f"no unique program from calibrated evidence ({detail})", "binding", witness)
    if len(solutions) > 1:
        return ProposedResult("ABSTAIN", None, None,
                              f"{len(solutions)} well-typed programs from calibrated evidence; not unique",
                              "binding", witness)

    expr = next(iter(solutions.values()))
    try:
        return ProposedResult("ANSWER", evaluate(expr, store), expr,
                              "evaluated from calibrated evidence", None, witness)
    except (_Empty, TypeError_, ValueError, ZeroDivisionError) as error:
        return ProposedResult("ABSTAIN", None, expr, str(error), "precondition", witness)
