"""L8.21: bind mentions to argument positions by solving, not by proximity.

L8.20 got the shape of the tree right and the arguments wrong.  In
「格子サイズ64x64の実行時間の平均を教えて」 every concept is recognised -- 格子サイズ,
64x64, 実行時間, 平均 -- and the roles are swapped: 格子サイズ belongs to the filter
and 実行時間 to the projection, but the compiler took the first field cue it saw
and projected a categorical, which MEAN then refused.  That is not missing
vocabulary.  It is missing *binding*.

The fix is not a better heuristic.  Replacing "first match" with "nearest match"
would swap one guess for another and would keep the property that a confident
misreading is possible.  Instead the holes in the tree state what they require,
every mention is a candidate for every hole it could type-check into, and all
assignments are enumerated:

    mentions  ->  type constraints  ->  search all bindings
                                             |
                                    exactly one well-typed?
                                       yes -> execute
                                       no  -> abstain

Uniqueness is the whole point.  Zero solutions and several solutions are both
refusals, so the layer can gain capability without gaining a way to be
confidently wrong -- the invariant L8.18 established and every layer since has
had to keep.

Two consequences worth stating, because they are what a proximity rule cannot do:

* a value can name its own field.  In 「64x64を使った実験の平均実行時間は？」 nothing
  says 格子サイズ, but 64x64 appears in exactly one field's domain, so the filter
  is forced by the value rather than by a nearby word.
* a request that genuinely does not determine its arguments now refuses.
  「実行時間と成功率の平均は？」 offers two Number fields for one projection hole;
  L8.20 silently took the first, and that is a wrong answer that looks right.

L8.20 is left untouched so the two can be measured side by side.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product

from .incomplete_operation_l818_experiment import SlotKind, parse_query_ir
from .query_algebra_l820_experiment import (
    Argmax,
    Compare,
    Count,
    Expr,
    Filter,
    Max,
    Mean,
    Mode,
    Project,
    Source,
    TypeError_,
    _AGGREGATES,
    _COMPARE_MARKERS,
    _find_aggregate,
    infer,
)
from .typed_operation_l819_experiment import FIELDS, TypedStore, ValueType

_FIELDS_BY_NAME = {spec.name: spec for spec in FIELDS}

#: Which argument types each aggregate's projection hole admits.
_PROJECTION_TYPES: dict[str, tuple[ValueType, ...]] = {
    "MEAN": (ValueType.NUMBER,),
    "MAX": (ValueType.NUMBER,),
    "MODE": (ValueType.CATEGORICAL, ValueType.NUMBER),
    "ARGMAX": (ValueType.NUMBER,),
}


@dataclass(frozen=True)
class ValueMention:
    text: str
    candidate_fields: tuple[str, ...]


@dataclass(frozen=True)
class Mentions:
    fields: tuple[str, ...]
    values: tuple[ValueMention, ...]
    aggregate: str | None


def collect_mentions(text: str, store: TypedStore) -> Mentions:
    """Every concept the text names, with no role assigned yet."""
    domains: dict[str, set[str]] = {}
    for spec in FIELDS:
        if spec.value_type is not ValueType.CATEGORICAL:
            continue
        domains[spec.name] = {
            str(values[spec.name]) for _, values in store.rows() if spec.name in values
        }

    seen: list[ValueMention] = []
    claimed: list[str] = []
    for value in sorted({v for values in domains.values() for v in values}, key=len, reverse=True):
        if value not in text:
            continue
        if any(value in other for other in claimed):
            continue  # already covered by a longer literal, e.g. 128 inside 128x128
        claimed.append(value)
        owners = tuple(sorted(name for name, domain in domains.items() if value in domain))
        seen.append(ValueMention(value, owners))

    fields = tuple(spec.name for spec in FIELDS if any(cue in text for cue in spec.cues))
    return Mentions(fields, tuple(seen), _find_aggregate(text))


@dataclass(frozen=True)
class Binding:
    filters: tuple[tuple[str, str], ...]
    projection: str | None


def candidate_bindings(mentions: Mentions, inherited_field: str | None) -> list[Binding]:
    """Every assignment of mentions to holes that could type-check."""
    filter_choices = [mention.candidate_fields for mention in mentions.values]
    if any(not choices for choices in filter_choices):
        return []

    aggregate = mentions.aggregate
    projection_choices: list[str | None]
    if aggregate is None or aggregate == "COUNT":
        projection_choices = [None]
    else:
        admitted = _PROJECTION_TYPES[aggregate]
        named = [
            name for name in mentions.fields
            if _FIELDS_BY_NAME[name].value_type in admitted
        ]
        if named:
            projection_choices = list(dict.fromkeys(named))
        elif inherited_field is not None:
            # Inheritance outranks the type error below.  In an elliptical
            # operand such as 「格子サイズ32x32と比べて…」 the named field fills the
            # *filter* hole and the projection comes from the sibling, so
            # reporting "MEAN cannot take lattice_size" would describe a hole
            # the field was never offered for.
            projection_choices = [inherited_field]
        elif mentions.fields:
            # A field was named and none of them can fill this hole.  That is a
            # type error about the field the user chose, not an ambiguity --
            # reporting it as "several bindings" would describe the wrong problem.
            raise TypeError_(
                f"{aggregate} cannot take "
                + ", ".join(f"`{name}` ({_FIELDS_BY_NAME[name].value_type.value})" for name in mentions.fields)
            )
        elif inherited_field is not None:
            projection_choices = [inherited_field]
        else:
            # Nothing named a field: every admissible field is a candidate, so a
            # schema with more than one Number field makes this ambiguous rather
            # than defaulting to whichever comes first.
            projection_choices = [
                spec.name for spec in FIELDS if spec.value_type in admitted
            ]

    bindings = []
    for filters in product(*filter_choices) if filter_choices else [()]:
        for projection in projection_choices:
            pairs = tuple(
                (field, mention.text) for field, mention in zip(filters, mentions.values)
            )
            bindings.append(Binding(pairs, projection))
    return bindings


def build_expr(binding: Binding, aggregate: str | None) -> Expr:
    node: Expr = Source()
    for field, value in binding.filters:
        node = Filter(node, field, value)
    if aggregate is None:
        return node
    if aggregate == "COUNT":
        return Count(node)
    if binding.projection is None:
        raise TypeError_(f"{aggregate} needs a field and none was named")
    if aggregate == "ARGMAX":
        return Argmax(node, binding.projection)
    return {"MEAN": Mean, "MAX": Max, "MODE": Mode}[aggregate](Project(node, binding.projection))


@dataclass(frozen=True)
class BindResult:
    expr: Expr | None
    reason: str
    stage: str | None = None
    solutions: int = 0


def _solve_operand(
    text: str,
    store: TypedStore,
    inherited: tuple[str | None, str | None],
    proposed: tuple[str | None, tuple[str, ...]] = (None, ()),
):
    """``proposed`` widens the mention sets (see L8.22); it never decides."""
    mentions = collect_mentions(text, store)
    aggregate = mentions.aggregate or inherited[0] or proposed[0]
    fields = tuple(dict.fromkeys(mentions.fields + proposed[1]))
    mentions = Mentions(fields, mentions.values, aggregate)

    try:
        bindings = candidate_bindings(mentions, inherited[1])
    except TypeError_ as error:
        return None, aggregate, None, str(error), 0

    well_typed = []
    for binding in bindings:
        try:
            expr = build_expr(binding, aggregate)
            infer(expr)
        except TypeError_:
            continue
        well_typed.append((expr, binding))

    unique = {}
    for expr, binding in well_typed:
        unique.setdefault(_key(expr), (expr, binding))
    solutions = list(unique.values())

    if not solutions:
        return None, aggregate, None, "no well-typed binding of the named concepts", 0
    if len(solutions) > 1:
        return None, aggregate, None, f"{len(solutions)} well-typed bindings; the request does not choose", len(solutions)
    expr, binding = solutions[0]
    return expr, aggregate, binding.projection, "unique well-typed binding", 1


def _key(expr: Expr) -> str:
    from .query_algebra_l820_experiment import render

    return render(expr)


def compile_bound(
    request: str,
    store: TypedStore,
    proposed: tuple[str | None, tuple[str, ...]] = (None, ()),
) -> BindResult:
    ir = parse_query_ir(store.episodic, request)
    blocking = tuple(kind for kind in ir.blocking if kind is not SlotKind.ANCHOR)
    if blocking and _find_aggregate(request) is None and proposed[0] is None:
        return BindResult(None, f"{blocking[0].value} unresolved", "gate")

    has_compare = any(marker in request for marker in _COMPARE_MARKERS)
    if _find_aggregate(request) is None and not has_compare and proposed[0] is None:
        return BindResult(None, "no operation in this request", None)

    for marker in _COMPARE_MARKERS:
        if marker in request:
            baseline_text, _, subject_text = request.partition(marker)
            subject, aggregate, field, reason, count = _solve_operand(subject_text, store, (None, None), proposed)
            if subject is None:
                return BindResult(None, reason, _stage_for(reason), count)
            baseline, _, _, reason, count = _solve_operand(baseline_text, store, (aggregate, field), proposed)
            if baseline is None:
                return BindResult(None, reason, _stage_for(reason), count)
            expr = Compare(baseline, subject)
            try:
                infer(expr)
            except TypeError_ as error:
                return BindResult(None, str(error), "type")
            return BindResult(expr, "compiled", None, 1)

    expr, aggregate, _, reason, count = _solve_operand(request, store, (None, None), proposed)
    if expr is None:
        return BindResult(None, reason, _stage_for(reason), count)
    if aggregate is None:
        return BindResult(None, "no operation in this request", None)
    return BindResult(expr, "compiled", None, 1)


def _stage_for(reason: str) -> str:
    return "type" if "cannot take" in reason else "binding"


@dataclass(frozen=True)
class BoundResult:
    decision: str
    value: object | None
    expr: Expr | None
    reason: str
    stage: str | None = None


def run_bound_query(store: TypedStore, request: str) -> BoundResult:
    from .query_algebra_l820_experiment import _Empty, evaluate

    compiled = compile_bound(request, store)
    if compiled.expr is None:
        decision = "NOT_AN_OPERATION" if compiled.stage is None else "ABSTAIN"
        return BoundResult(decision, None, None, compiled.reason, compiled.stage)
    try:
        infer(compiled.expr)
    except TypeError_ as error:
        return BoundResult("ABSTAIN", None, compiled.expr, str(error), "type")
    try:
        return BoundResult("ANSWER", evaluate(compiled.expr, store), compiled.expr, "evaluated")
    except _Empty as error:
        return BoundResult("ABSTAIN", None, compiled.expr, str(error), "precondition")
    except (TypeError_, ValueError, ZeroDivisionError) as error:
        return BoundResult("ABSTAIN", None, compiled.expr, str(error), "precondition")
