"""L8.22: a learned proposer that may widen the candidate set and nothing else.

L8.18 through L8.21 built a pipeline whose every stage can only *narrow* what is
allowed to run: slots must be complete, operations must be well-typed, the tree
must parse, the binding must be unique.  The price is a closed world -- an
expression outside the hand-written cue tables is refused however obvious it is
to a reader.  「64x64で走らせたケースについて処理時間をならして」 is a mean, and the
compiler has never heard of 「ならして」.

The purpose of learning here is to convert some of those safe abstentions into
correct executions, and the constraint is that it must not create a single wrong
one.  So the learned component is given proposal rights, not authority:

    NL -> proposer -> candidate operators / fields  (scores)
                   -> symbolic compiler
                   -> type constraints
                   -> unique-binding solver
                   -> execution gate

A proposal of MEAN at 0.72 does not run MEAN.  It puts MEAN into the candidate
set; if the rest of the pipeline then yields exactly one well-typed program,
that program runs, and otherwise the request is still refused.  A confident
wrong proposal therefore costs coverage, never correctness.

Two design choices keep the claim honest:

* **The proposer only fires where the hand-written cues found nothing.** Every
  earlier benchmark is byte-identical with this layer installed, so any measured
  gain is coverage that did not exist before rather than a reshuffling of what
  did.
* **Candidates are taken by margin, not by argmax.** The top proposal carries
  the set only while it clears the runner-up; MEAN 0.72 against MODE 0.14 leaves
  one candidate, MEAN 0.45 against MODE 0.40 leaves two and the uniqueness rule
  then refuses.  This is the same margin discipline L8.15 introduced, applied to
  a learned score instead of a coverage score.

The model is a character n-gram classifier trained on machine-generated pairs of
(surface form, label), which is supervision this repository can actually produce
and check.  It cannot invent a meaning for a word with no subword overlap with
anything it has seen; what it can do is carry 「ならす」 to 「ならして」.  That is the
generalisation being measured, and calling it anything more would overstate it.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math

from .argument_binding_l821_experiment import compile_bound, run_bound_query
from .query_algebra_l820_experiment import (
    Expr, TypeError_, _AGGREGATES, _Empty, canonical_render, evaluate, infer,
)
from .typed_operation_l819_experiment import FIELDS, TypedStore

PROPOSAL_FLOOR = 0.25
PROPOSAL_MARGIN = 0.20

#: Surface forms per label, split so the held-out ones share subwords with the
#: trained ones but contain no hand-written cue.  ``assert_holdout_is_unseen``
#: checks the second half of that claim rather than trusting it.
OPERATOR_FORMS: dict[str, dict[str, tuple[str, ...]]] = {
    "MEAN": {
        "train": ("ならす", "ならした", "ならす形で", "アベレージ値", "アベレージを出す"),
        "holdout": ("ならして", "アベレージ", "ならした値", "アベレージで", "アベレージ的に", "ならしで"),
    },
    "COUNT": {
        "train": ("数えて", "数える", "カウント数", "カウントする"),
        "holdout": ("数を教えて", "カウントして", "カウントを出して", "数えた結果", "数えあげて"),
    },
    "MAX": {
        "train": ("ピーク値", "ピークを出す", "頭打ちの値"),
        "holdout": ("ピークは", "ピークを教えて", "ピークの値"),
    },
}

FIELD_FORMS: dict[str, dict[str, tuple[str, ...]]] = {
    "runtime_seconds": {
        "train": ("処理時間", "所要時間", "かかった時間", "ランタイム"),
        "holdout": ("処理にかかった時間", "ランタイムの値", "ランタイム値", "所要の時間", "ランタイム量"),
    },
    "success_rate": {
        "train": ("成功割合", "成功比率", "サクセスレート"),
        "holdout": ("成功の割合", "サクセスレートの値", "成功の比率", "サクセスレート値", "成功という割合"),
    },
    "lattice_size": {
        "train": ("格子の大きさ", "グリッドサイズ", "グリッド幅"),
        "holdout": ("グリッドの大きさ",),
    },
}

_CUES = tuple(cue for cue, _ in _AGGREGATES) + tuple(
    cue for spec in FIELDS for cue in spec.cues
)


def assert_holdout_is_unseen() -> None:
    """A held-out form that contains a hand-written cue would never reach the
    proposer, so the experiment would be measuring the cue table instead."""
    for bank in (OPERATOR_FORMS, FIELD_FORMS):
        for label, split in bank.items():
            for form in split["holdout"]:
                for cue in _CUES:
                    if cue in form:
                        raise AssertionError(f"held-out form {form!r} ({label}) contains cue {cue!r}")
                if form in split["train"]:
                    raise AssertionError(f"held-out form {form!r} is also in training")


def _ngrams(text: str, sizes: tuple[int, ...] = (2, 3)) -> list[str]:
    grams = []
    for size in sizes:
        grams.extend(text[index:index + size] for index in range(max(0, len(text) - size + 1)))
    return grams


@dataclass
class NgramClassifier:
    """Additive-smoothed character n-gram scorer.  Deterministic, no deps."""

    counts: dict[str, dict[str, int]]
    totals: dict[str, int]
    vocabulary: int
    alpha: float = 0.5

    @classmethod
    def train(cls, examples: list[tuple[str, str]]) -> "NgramClassifier":
        counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        totals: dict[str, int] = defaultdict(int)
        vocabulary: set[str] = set()
        for text, label in examples:
            for gram in _ngrams(text):
                counts[label][gram] += 1
                totals[label] += 1
                vocabulary.add(gram)
        return cls({k: dict(v) for k, v in counts.items()}, dict(totals), max(len(vocabulary), 1))

    def coverage(self, text: str, label: str) -> float:
        """Share of the text's n-grams this label has ever seen.

        ``scores`` is relative across labels and says nothing about how much
        evidence there is: with one label matching a single n-gram it returns
        1.0.  Whole-sentence use never noticed, because the sentence always had
        evidence somewhere; enumerating spans does, so absolute evidence has to
        be available separately.
        """
        grams = _ngrams(text)
        if not grams or label not in self.counts:
            return 0.0
        known = self.counts[label]
        return sum(1 for gram in grams if gram in known) / len(grams)

    def scores(self, text: str) -> list[tuple[str, float]]:
        grams = _ngrams(text)
        if not grams:
            return []
        logs = {}
        for label in self.counts:
            total = self.totals[label]
            value = 0.0
            hits = 0
            for gram in grams:
                count = self.counts[label].get(gram, 0)
                if count:
                    hits += 1
                value += math.log((count + self.alpha) / (total + self.alpha * self.vocabulary))
            if hits == 0:
                continue  # no evidence at all for this label
            logs[label] = value / len(grams)
        if not logs:
            return []
        best = max(logs.values())
        weights = {label: math.exp(value - best) for label, value in logs.items()}
        mass = sum(weights.values())
        return sorted(((label, weight / mass) for label, weight in weights.items()),
                      key=lambda item: (-item[1], item[0]))


def build_training_examples() -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    operators = [(form, label) for label, split in OPERATOR_FORMS.items() for form in split["train"]]
    fields = [(form, label) for label, split in FIELD_FORMS.items() for form in split["train"]]
    return operators, fields


@dataclass
class Proposer:
    operator_model: NgramClassifier
    field_model: NgramClassifier

    @classmethod
    def trained(cls) -> "Proposer":
        assert_holdout_is_unseen()
        operators, fields = build_training_examples()
        return cls(NgramClassifier.train(operators), NgramClassifier.train(fields))

    @staticmethod
    def _by_margin(scored: list[tuple[str, float]]) -> tuple[str, ...]:
        if not scored or scored[0][1] < PROPOSAL_FLOOR:
            return ()
        top = scored[0][1]
        return tuple(label for label, score in scored if top - score <= PROPOSAL_MARGIN)

    def operators(self, text: str) -> tuple[str, ...]:
        return self._by_margin(self.operator_model.scores(text))

    def fields(self, text: str) -> tuple[str, ...]:
        return self._by_margin(self.field_model.scores(text))


@dataclass(frozen=True)
class ProposedResult:
    decision: str
    value: object | None
    expr: Expr | None
    reason: str
    stage: str | None = None
    source: str = "symbolic"


_LEXICAL_MISSES = ("no operation in this request",)


def _unreadable(result) -> bool:
    """Requests the cue tables could not read as an operation at all."""
    if result.decision == "NOT_AN_OPERATION" and result.reason in _LEXICAL_MISSES:
        return True
    # A comparison whose operands name no aggregate: the shape parsed, the
    # operator did not, so COMPARE received two sets instead of two numbers.
    return result.decision == "ABSTAIN" and result.stage == "type" and "COMPARE takes two numbers" in result.reason


def run_proposed_query(store: TypedStore, request: str, proposer: Proposer | None = None) -> ProposedResult:
    """L8.21 first; the proposer only sees requests its cue tables could not read."""
    base = run_bound_query(store, request)
    if not _unreadable(base):
        return ProposedResult(base.decision, base.value, base.expr, base.reason, base.stage)

    proposer = proposer or Proposer.trained()
    operators = proposer.operators(request)
    if not operators:
        return ProposedResult("NOT_AN_OPERATION", None, None,
                              "no operation, and the proposer had no evidence either")

    fields = proposer.fields(request)

    # Each candidate operator is compiled by the ordinary symbolic pipeline.
    # The proposal only says which operators to try; types, binding uniqueness
    # and the gate decide whether any of them survives.
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
        # Distinguish "the proposal was untypeable" from "each candidate was
        # itself under-determined"; reporting the first for the second described
        # the wrong failure.
        detail = "; ".join(refusals) or "no candidate compiled"
        return ProposedResult("ABSTAIN", None, None,
                              f"no unique program from the proposal ({detail})", "binding", "proposed")
    if len(solutions) > 1:
        return ProposedResult("ABSTAIN", None, None,
                              f"{len(solutions)} well-typed programs from the proposal; not unique",
                              "binding", "proposed")

    expr = next(iter(solutions.values()))
    try:
        return ProposedResult("ANSWER", evaluate(expr, store), expr, "evaluated from a proposal", None, "proposed")
    except (_Empty, TypeError_, ValueError, ZeroDivisionError) as error:
        return ProposedResult("ABSTAIN", None, expr, str(error), "precondition", "proposed")
