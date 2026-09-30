"""L8.20: compile a request into a nested typed expression, then run it.

L8.19.1 found the limit of a flat parser.  Asked
「32x32と比べて64x64の平均実行時間は大きい？」 it recovered both operands and lost the
nesting: 「平均」 matched before 「と比べて」, so COMPARE(MEAN(A), MEAN(B)) was read as
MEAN with two contradictory filters.  First-match-wins has no notion of which
operator is on the outside.

So the query becomes a tree with precedence, not a slot record:

    Expr := Source
          | Filter(Expr, field, value)      Set[Episode] -> Set[Episode]
          | Project(Expr, field)            Set[Episode] -> Collection[t]
          | Count(Expr)                     Set[Episode] -> Integer
          | Mean(Expr) | Max(Expr)          Collection[Number] -> Number
          | Mode(Expr)                      Collection[t] -> t
          | Argmax(Expr, field)             Set[Episode] -> Episode
          | Compare(Expr, Expr)             Number x Number -> Comparison

Types are inferred bottom-up, so L8.19's signature checks become one pass over
the tree and report *which node* failed rather than which cue matched.  The
L8.18 gate is unchanged and still runs first: nothing here can make a partly
understood request execute.

Two things the flat version could not express, both of them ordinary Japanese:

* **precedence** -- COMPARE binds loosest, so it is taken off first and its
  operands are compiled recursively.  That single rule is what fixes L8.19.1.
* **ellipsis** -- 「32x32と比べて64x64の平均実行時間は」 states the aggregate and the
  field once, on the second operand only.  An operand that carries a filter but
  no operator inherits its sibling's, which is what makes the left side mean
  MEAN(PROJECT(FILTER(src, 32x32), runtime)) rather than a bare set.

Ellipsis inheritance is a real risk of over-reach, so it is deliberately narrow:
an operand inherits only when it resolved a filter and no aggregate of its own,
and inheritance never crosses a COMPARE boundary in the other direction.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import statistics

from .incomplete_operation_l818_experiment import SlotKind, parse_query_ir
from .typed_operation_l819_experiment import FIELDS, TypedStore, ValueType

_FIELDS_BY_NAME = {spec.name: spec for spec in FIELDS}

_COMPARE_MARKERS = ("と比べて", "と比較して", "に比べて", "より")
_AGGREGATES: tuple[tuple[str, str], ...] = (
    ("一番多く使った", "MODE"),
    ("最も多く使った", "MODE"),
    ("一番高い", "ARGMAX"),
    ("一番大きい", "ARGMAX"),
    ("一番長い", "ARGMAX"),
    ("最も高い", "ARGMAX"),
    ("最大", "MAX"),
    ("平均", "MEAN"),
    ("何件", "COUNT"),
    ("いくつ", "COUNT"),
    ("何個", "COUNT"),
    ("件数", "COUNT"),
)


class Expr:
    """Base node.  ``result_type`` is inferred, never declared by the parser."""


@dataclass(frozen=True)
class Source(Expr):
    pass


@dataclass(frozen=True)
class Filter(Expr):
    source: Expr
    field: str
    value: str


@dataclass(frozen=True)
class Project(Expr):
    source: Expr
    field: str


@dataclass(frozen=True)
class Count(Expr):
    source: Expr


@dataclass(frozen=True)
class Mean(Expr):
    source: Expr


@dataclass(frozen=True)
class Max(Expr):
    source: Expr


@dataclass(frozen=True)
class Mode(Expr):
    source: Expr


@dataclass(frozen=True)
class Argmax(Expr):
    source: Expr
    field: str


@dataclass(frozen=True)
class Compare(Expr):
    baseline: Expr
    subject: Expr


class Kind(Enum):
    EPISODE_SET = "Set[Episode]"
    NUMBERS = "Collection[Number]"
    CATEGORIES = "Collection[Categorical]"
    TEXTS = "Collection[Text]"
    NUMBER = "Number"
    INTEGER = "Integer"
    EPISODE = "Episode"
    CATEGORICAL = "Categorical"
    COMPARISON = "Comparison"


class TypeError_(Exception):
    pass


_COLLECTION_OF = {
    ValueType.NUMBER: Kind.NUMBERS,
    ValueType.CATEGORICAL: Kind.CATEGORIES,
    ValueType.TEXT: Kind.TEXTS,
}


def infer(node: Expr) -> Kind:
    """Bottom-up type inference.  Errors name the node, not the cue."""
    if isinstance(node, Source):
        return Kind.EPISODE_SET
    if isinstance(node, Filter):
        if infer(node.source) is not Kind.EPISODE_SET:
            raise TypeError_("FILTER takes Set[Episode]")
        return Kind.EPISODE_SET
    if isinstance(node, Project):
        if infer(node.source) is not Kind.EPISODE_SET:
            raise TypeError_("PROJECT takes Set[Episode]")
        return _COLLECTION_OF[_FIELDS_BY_NAME[node.field].value_type]
    if isinstance(node, Count):
        if infer(node.source) is not Kind.EPISODE_SET:
            raise TypeError_("COUNT takes Set[Episode]")
        return Kind.INTEGER
    if isinstance(node, (Mean, Max)):
        inner = infer(node.source)
        if inner is not Kind.NUMBERS:
            raise TypeError_(f"{type(node).__name__.upper()} takes Collection[Number], got {inner.value}")
        return Kind.NUMBER
    if isinstance(node, Mode):
        inner = infer(node.source)
        if inner not in (Kind.CATEGORIES, Kind.NUMBERS):
            raise TypeError_(f"MODE takes an equatable collection, got {inner.value}")
        return Kind.CATEGORICAL
    if isinstance(node, Argmax):
        if infer(node.source) is not Kind.EPISODE_SET:
            raise TypeError_("ARGMAX takes Set[Episode]")
        if _FIELDS_BY_NAME[node.field].value_type is not ValueType.NUMBER:
            raise TypeError_(f"ARGMAX orders by Number, `{node.field}` is not")
        return Kind.EPISODE
    if isinstance(node, Compare):
        left, right = infer(node.baseline), infer(node.subject)
        if left is not right or left not in (Kind.NUMBER, Kind.INTEGER):
            raise TypeError_(f"COMPARE takes two numbers, got {left.value} and {right.value}")
        return Kind.COMPARISON
    raise TypeError_(f"unknown node {node!r}")


def render(node: Expr) -> str:
    if isinstance(node, Source):
        return "episodes"
    if isinstance(node, Filter):
        return f"FILTER({render(node.source)}, {node.field} == {node.value})"
    if isinstance(node, Project):
        return f"PROJECT({render(node.source)}, {node.field})"
    if isinstance(node, Argmax):
        return f"ARGMAX({render(node.source)}, {node.field})"
    if isinstance(node, Compare):
        return f"COMPARE({render(node.baseline)}, {render(node.subject)})"
    return f"{type(node).__name__.upper()}({render(node.source)})"


def canonical_render(node: Expr) -> str:
    """``render`` with commutative filter chains put in a fixed order.

    Conjunctive filters commute, so FILTER(FILTER(s, seed), lattice) and
    FILTER(FILTER(s, lattice), seed) are the same set.  Comparing raw renderings
    counts that difference as a wrong tree, which measures emission order rather
    than meaning -- so equality is taken over the canonical form instead.
    """
    if isinstance(node, Filter):
        chain, inner = [], node
        while isinstance(inner, Filter):
            chain.append((inner.field, inner.value))
            inner = inner.source
        rendered = canonical_render(inner)
        for field, value in sorted(chain):
            rendered = f"FILTER({rendered}, {field} == {value})"
        return rendered
    if isinstance(node, Source):
        return "episodes"
    if isinstance(node, Project):
        return f"PROJECT({canonical_render(node.source)}, {node.field})"
    if isinstance(node, Argmax):
        return f"ARGMAX({canonical_render(node.source)}, {node.field})"
    if isinstance(node, Compare):
        return f"COMPARE({canonical_render(node.baseline)}, {canonical_render(node.subject)})"
    return f"{type(node).__name__.upper()}({canonical_render(node.source)})"


# --------------------------------------------------------------------------
# compilation
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class CompileResult:
    expr: Expr | None
    reason: str
    stage: str | None = None


def _find_filters(text: str, store: TypedStore) -> list[tuple[str, str]]:
    found = []
    for spec in FIELDS:
        if spec.value_type is not ValueType.CATEGORICAL:
            continue
        domain = {str(values[spec.name]) for _, values in store.rows() if spec.name in values}
        for value in sorted(domain, key=len, reverse=True):
            if value in text:
                found.append((spec.name, value))
                break
    return found


def _find_aggregate(text: str) -> str | None:
    return next((name for cue, name in _AGGREGATES if cue in text), None)


def _find_field(text: str) -> str | None:
    return next((spec.name for spec in FIELDS if any(cue in text for cue in spec.cues)), None)


def _build_operand(text: str, store: TypedStore, inherited: tuple[str | None, str | None]):
    """Compile one operand, inheriting an elided aggregate/field from a sibling."""
    aggregate = _find_aggregate(text) or inherited[0]
    field = _find_field(text) or inherited[1]
    node: Expr = Source()
    for name, value in _find_filters(text, store):
        node = Filter(node, name, value)

    if aggregate is None:
        return node, aggregate, field
    if aggregate == "COUNT":
        return Count(node), aggregate, field
    if aggregate == "ARGMAX":
        if field is None:
            raise TypeError_("ARGMAX needs a field and none was named")
        return Argmax(node, field), aggregate, field
    if field is None:
        raise TypeError_(f"{aggregate} needs a field and none was named")
    projected = Project(node, field)
    return {"MEAN": Mean, "MAX": Max, "MODE": Mode}[aggregate](projected), aggregate, field


def compile_query(request: str, store: TypedStore) -> CompileResult:
    # 1. L8.18's gate, first and unchanged.
    ir = parse_query_ir(store.episodic, request)
    blocking = tuple(kind for kind in ir.blocking if kind is not SlotKind.ANCHOR)
    if blocking and _find_aggregate(request) is None:
        return CompileResult(None, f"{blocking[0].value} unresolved", "gate")

    if _find_aggregate(request) is None and not any(
        marker in request for marker in _COMPARE_MARKERS
    ):
        return CompileResult(None, "no operation in this request", None)

    try:
        # 2. COMPARE binds loosest, so it comes off first and its operands are
        #    compiled recursively.  This is the whole fix for L8.19.1.
        for marker in _COMPARE_MARKERS:
            if marker in request:
                baseline_text, _, subject_text = request.partition(marker)
                subject, aggregate, field = _build_operand(subject_text, store, (None, None))
                baseline, _, _ = _build_operand(baseline_text, store, (aggregate, field))
                if aggregate is None:
                    return CompileResult(None, "COMPARE operands name no operation", "gate")
                return CompileResult(Compare(baseline, subject), "compiled")

        node, aggregate, _ = _build_operand(request, store, (None, None))
        if aggregate is None:
            return CompileResult(None, "no operation in this request", None)
        return CompileResult(node, "compiled")
    except TypeError_ as error:
        return CompileResult(None, str(error), "type")


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class EvalResult:
    decision: str  # ANSWER | ABSTAIN | NOT_AN_OPERATION
    value: object | None
    expr: Expr | None
    reason: str
    stage: str | None = None


class _Empty(Exception):
    pass


def _rows(node: Expr, store: TypedStore):
    if isinstance(node, Source):
        return store.rows()
    if isinstance(node, Filter):
        return [row for row in _rows(node.source, store) if str(row[1].get(node.field)) == node.value]
    raise TypeError_("expected a set-valued node")


def evaluate(node: Expr, store: TypedStore):
    if isinstance(node, Count):
        return len(_rows(node.source, store))
    if isinstance(node, Project):
        values = [row[1][node.field] for row in _rows(node.source, store) if node.field in row[1]]
        if not values:
            raise _Empty(f"no episode in that set records `{node.field}`")
        return values
    if isinstance(node, Mean):
        return round(statistics.fmean(evaluate(node.source, store)), 4)
    if isinstance(node, Max):
        return max(evaluate(node.source, store))
    if isinstance(node, Mode):
        values = evaluate(node.source, store)
        counts: dict[object, int] = {}
        for value in values:
            counts[value] = counts.get(value, 0) + 1
        best = max(counts.values())
        winners = sorted(str(key) for key, count in counts.items() if count == best)
        if len(winners) > 1:
            raise _Empty(f"tied most-common values: {', '.join(winners)}")
        return winners[0]
    if isinstance(node, Argmax):
        rows = [row for row in _rows(node.source, store) if node.field in row[1]]
        if not rows:
            raise _Empty(f"no episode in that set records `{node.field}`")
        return max(rows, key=lambda row: row[1][node.field])[0].episode_id
    if isinstance(node, Compare):
        baseline = evaluate(node.baseline, store)
        subject = evaluate(node.subject, store)
        return "greater" if subject > baseline else ("less" if subject < baseline else "equal")
    raise TypeError_(f"cannot evaluate {node!r}")


def run_query(store: TypedStore, request: str) -> EvalResult:
    compiled = compile_query(request, store)
    if compiled.expr is None:
        decision = "NOT_AN_OPERATION" if compiled.stage is None else "ABSTAIN"
        return EvalResult(decision, None, None, compiled.reason, compiled.stage)
    try:
        infer(compiled.expr)
    except TypeError_ as error:
        return EvalResult("ABSTAIN", None, compiled.expr, str(error), "type")
    try:
        return EvalResult("ANSWER", evaluate(compiled.expr, store), compiled.expr, "evaluated")
    except _Empty as error:
        return EvalResult("ABSTAIN", None, compiled.expr, str(error), "precondition")
    except (TypeError_, ValueError, ZeroDivisionError) as error:
        return EvalResult("ABSTAIN", None, compiled.expr, str(error), "precondition")
