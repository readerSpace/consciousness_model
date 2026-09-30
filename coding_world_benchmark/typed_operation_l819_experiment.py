"""L8.19: give the OPERATION slot a type, and execute it.

L8.18 stopped the agent from executing a request it had only partly
understood, and left OPERATION cues such as 「平均」 or 「一番多く使った」 registered
with ``resolved=None`` so they would abstain rather than degrade into a lookup.
This layer resolves them, without touching that gate.  The gate answers "may
this run at all"; everything here answers "run it as what".

Operations carry signatures rather than being string handlers:

    COUNT   : Set[Episode]                        -> Integer
    MEAN    : Collection[Number]                  -> Number
    MODE    : Collection[Equatable]               -> Equatable
    ARGMAX  : Set[Episode] x (Episode -> Number)  -> Episode
    COMPARE : Number x Number                     -> Comparison

which makes a whole class of requests refusable before anything is read.
「実験名の平均は？」 resolves every slot it evokes -- the operation is MEAN, the
field is ``experiment_name``, both found -- and is still nonsense, because
MEAN wants Number and ``experiment_name`` is Text.  A string-matching
implementation would compute something; a typed one declines, and that
distinction is measured here as ``type_error_escape_rate``.

So the execution condition becomes three tests, in order, each of which can
only ever refuse:

    all evoked slots resolved      (L8.18, unchanged)
  ∧ well-typed(QueryIR)            (the signature admits these argument types)
  ∧ preconditions satisfied        (the data the signature needs is actually there)

The third is separate from the second on purpose.  ARGMAX over ``success_rate``
is well-typed whether or not any episode recorded a success rate; asking for
the maximum of an empty set is a different failure from asking for the mean of
a name, and collapsing them would hide which one occurred.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re
import statistics

from .episodic_memory_l813_experiment import Episode, EpisodicMemory
from .incomplete_operation_l818_experiment import SlotKind, parse_query_ir
from .structural_resolver_l815_experiment import query_tokens


class ValueType(Enum):
    NUMBER = "Number"
    CATEGORICAL = "Categorical"
    TEXT = "Text"
    EPISODE = "Episode"
    INTEGER = "Integer"
    COMPARISON = "Comparison"


@dataclass(frozen=True)
class FieldSpec:
    name: str
    value_type: ValueType
    pattern: str
    cues: tuple[str, ...]


FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec("lattice_size", ValueType.CATEGORICAL, r"格子サイズは\s*([0-9]+x[0-9]+)", ("格子サイズ",)),
    FieldSpec("seed", ValueType.CATEGORICAL, r"乱数種は\s*([0-9]+)", ("乱数種", "シード")),
    FieldSpec("runtime_seconds", ValueType.NUMBER, r"実行時間は\s*([0-9.]+)\s*秒", ("実行時間",)),
    FieldSpec("success_rate", ValueType.NUMBER, r"成功率は\s*([0-9.]+)", ("成功率",)),
    FieldSpec("experiment_name", ValueType.TEXT, r"", ("実験名", "名前")),
    # Declared but never emitted by the generators, so a request naming it is
    # well-typed and still unanswerable -- which is what separates the type
    # check from the precondition check.
    FieldSpec("energy_error", ValueType.NUMBER, r"エネルギー誤差は\s*([0-9.]+)", ("エネルギー誤差",)),
)

_FIELDS_BY_NAME = {spec.name: spec for spec in FIELDS}


@dataclass(frozen=True)
class OperationSpec:
    name: str
    argument_types: tuple[ValueType, ...]
    result_type: ValueType
    needs_field: bool
    cues: tuple[str, ...]


OPERATIONS: tuple[OperationSpec, ...] = (
    OperationSpec("COUNT", (ValueType.EPISODE,), ValueType.INTEGER, False,
                  ("何件", "いくつ", "何個", "件数")),
    OperationSpec("MEAN", (ValueType.NUMBER,), ValueType.NUMBER, True, ("平均",)),
    OperationSpec("MODE", (ValueType.CATEGORICAL, ValueType.NUMBER), ValueType.CATEGORICAL, True,
                  ("一番多く使った", "最も多く使った", "一番よく使った")),
    OperationSpec("ARGMAX", (ValueType.NUMBER,), ValueType.EPISODE, True,
                  ("一番高い", "最も高い", "一番長い", "最も長い", "最大", "一番大きい")),
    OperationSpec("COMPARE", (ValueType.NUMBER, ValueType.NUMBER), ValueType.COMPARISON, True,
                  ("と比べて", "より増えた", "より大きい", "より高い")),
)

_OPERATIONS_BY_NAME = {spec.name: spec for spec in OPERATIONS}


def extract_fields(episode: Episode, response: str) -> dict[str, object]:
    """Field extraction: the step between identifying an episode and computing."""
    values: dict[str, object] = {"experiment_name": episode.request}
    for spec in FIELDS:
        if not spec.pattern:
            continue
        match = re.search(spec.pattern, response)
        if not match:
            continue
        raw = match.group(1)
        values[spec.name] = float(raw) if spec.value_type is ValueType.NUMBER else raw
    return values


@dataclass
class TypedStore:
    episodic: EpisodicMemory
    fields: dict[str, dict[str, object]]

    def rows(self) -> list[tuple[Episode, dict[str, object]]]:
        return [(episode, self.fields.get(episode.episode_id, {})) for episode in self.episodic.episodes]


@dataclass(frozen=True)
class Filter:
    field: str
    value: object


@dataclass(frozen=True)
class TypedQuery:
    operation: str | None
    field: str | None
    filters: tuple[Filter, ...]
    operation_evoked: bool


def _detect_filters(request: str, store: TypedStore) -> tuple[Filter, ...]:
    filters = []
    for spec in FIELDS:
        if spec.value_type is not ValueType.CATEGORICAL:
            continue
        domain = {
            str(values[spec.name])
            for _, values in store.rows()
            if spec.name in values
        }
        for value in sorted(domain, key=len, reverse=True):
            if value in request:
                filters.append(Filter(spec.name, value))
    return tuple(filters)


def parse_typed_query(request: str, store: TypedStore) -> TypedQuery:
    operation = None
    evoked = False
    for spec in OPERATIONS:
        if any(cue in request for cue in spec.cues):
            evoked = True
            operation = spec.name
            break
    field = next(
        (spec.name for spec in FIELDS if any(cue in request for cue in spec.cues)), None
    )
    return TypedQuery(operation, field, _detect_filters(request, store), evoked)


@dataclass(frozen=True)
class OperationOutcome:
    decision: str  # ANSWER | ABSTAIN | NOT_AN_OPERATION
    value: object | None
    result_type: ValueType | None
    reason: str
    stage: str | None = None  # gate | type | precondition


def type_check(typed: TypedQuery) -> str | None:
    """Reject argument types the signature does not admit, before reading data."""
    if typed.operation is None:
        return None
    spec = _OPERATIONS_BY_NAME[typed.operation]
    if not spec.needs_field:
        return None
    if typed.field is None:
        return f"{spec.name} needs a field and none was named"
    field = _FIELDS_BY_NAME[typed.field]
    if field.value_type not in spec.argument_types:
        admitted = "/".join(item.value for item in spec.argument_types)
        return (
            f"{spec.name} takes {admitted} but `{field.name}` is {field.value_type.value}"
        )
    return None


def _group(store: TypedStore, item: Filter) -> list[tuple[Episode, dict[str, object]]]:
    return [row for row in store.rows() if str(row[1].get(item.field)) == str(item.value)]


def _selected(store: TypedStore, typed: TypedQuery) -> list[tuple[Episode, dict[str, object]]]:
    rows = store.rows()
    for item in typed.filters:
        rows = [row for row in rows if str(row[1].get(item.field)) == str(item.value)]
    return rows


def execute_typed(store: TypedStore, request: str) -> OperationOutcome:
    # 1. L8.18's gate, unchanged: a slot evoked but unresolved stops everything.
    ir = parse_query_ir(store.episodic, request)
    blocking = tuple(kind for kind in ir.blocking if kind is not SlotKind.ANCHOR)
    typed = parse_typed_query(request, store)
    if blocking and not (blocking == (SlotKind.OPERATION,) and typed.operation is not None):
        names = ", ".join(kind.value for kind in blocking)
        return OperationOutcome("ABSTAIN", None, None, f"{names} unresolved", "gate")

    if typed.operation is None:
        if typed.operation_evoked:
            return OperationOutcome("ABSTAIN", None, None, "operation evoked but unknown", "gate")
        return OperationOutcome("NOT_AN_OPERATION", None, None, "no operation in this request")

    # 2. Types.
    error = type_check(typed)
    if error is not None:
        return OperationOutcome("ABSTAIN", None, None, error, "type")

    spec = _OPERATIONS_BY_NAME[typed.operation]
    rows = _selected(store, typed)

    # 3. Preconditions: the signature is satisfiable only if the data is there.
    if not rows and spec.name != "COMPARE":
        fields = [item.field for item in typed.filters]
        if len(fields) != len(set(fields)):
            return OperationOutcome(
                "ABSTAIN", None, None,
                "two values were named for one field, which selects nothing", "precondition",
            )
        return OperationOutcome("ABSTAIN", None, None, "no episode matches the filter", "precondition")
    if spec.needs_field and spec.name != "COMPARE":
        # COMPARE checks each named group separately below; its filters select
        # two sets, not one intersection.
        present = [values[typed.field] for _, values in rows if typed.field in values]
        if not present:
            return OperationOutcome(
                "ABSTAIN", None, None, f"no episode records `{typed.field}`", "precondition"
            )
    else:
        present = []

    if spec.name == "COUNT":
        return OperationOutcome("ANSWER", len(rows), ValueType.INTEGER, "counted the selected episodes")
    if spec.name == "MEAN":
        return OperationOutcome(
            "ANSWER", round(statistics.fmean(present), 4), ValueType.NUMBER, "mean over the selected episodes"
        )
    if spec.name == "MODE":
        counts: dict[object, int] = {}
        for value in present:
            counts[value] = counts.get(value, 0) + 1
        best = max(counts.values())
        winners = sorted(str(key) for key, count in counts.items() if count == best)
        if len(winners) > 1:
            return OperationOutcome(
                "ABSTAIN", None, None, f"tied most-common values: {', '.join(winners)}", "precondition"
            )
        return OperationOutcome("ANSWER", winners[0], ValueType.CATEGORICAL, "most common value")
    if spec.name == "ARGMAX":
        best_row = max(
            (row for row in rows if typed.field in row[1]), key=lambda row: row[1][typed.field]
        )
        return OperationOutcome("ANSWER", best_row[0].episode_id, ValueType.EPISODE, "maximum over the field")

    # COMPARE is binary, so it needs two named groups.  Splitting one group in
    # half would answer *something* for a request that named only one operand,
    # which is the incomplete-operation mistake wearing a different hat.
    if len(typed.filters) != 2:
        return OperationOutcome(
            "ABSTAIN", None, None,
            f"COMPARE needs two named groups, {len(typed.filters)} given", "precondition",
        )
    operands = []
    for item in typed.filters:
        values = [row[1][typed.field] for row in _group(store, item) if typed.field in row[1]]
        if not values:
            return OperationOutcome(
                "ABSTAIN", None, None, f"`{item.value}` has no `{typed.field}` recorded", "precondition"
            )
        operands.append(statistics.fmean(values))
    verdict = "greater" if operands[1] > operands[0] else ("less" if operands[1] < operands[0] else "equal")
    return OperationOutcome("ANSWER", verdict, ValueType.COMPARISON, "compared the two named groups")


def build_store(episodic: EpisodicMemory, responses: dict[str, str]) -> TypedStore:
    return TypedStore(
        episodic,
        {
            episode.episode_id: extract_fields(episode, responses.get(episode.episode_id, ""))
            for episode in episodic.episodes
        },
    )
