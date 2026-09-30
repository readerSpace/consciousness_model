"""L8.18: never execute the part of a request that was understood.

L8.16 and L8.17 found the same defect twice.  In L8.16 the episode was
identified but what to return from it was not, and the layer answered with a
paraphrase.  In L8.17.1 the anchor was identified but the relation was not, and
the layer answered with the anchor itself.  Both are one mistake:

    a request was parsed in part, and the understood part was executed alone.

The operand survived, the operator was dropped, and the operand was returned as
if it were the result.  The stronger the evidence for the operand, the more
confident the wrong answer -- which is why 「Xの直後のを続けて」 failed precisely
*because* X was easy to ground.

So the rule is stated over slots rather than over Japanese phrases.  A request
is parsed into a typed IR, ``QueryIR``, whose slots are ANCHOR, RELATION,
PROJECTION and OPERATION, and execution is gated:

    a slot that is EVOKED but UNRESOLVED blocks execution, whatever else resolved

The word that matters is *evoked*.  The gate does not demand that every slot be
filled -- an ordinary reference evokes no RELATION, so nothing blocks and the
resolver behaves exactly as before.  It fires only when the request reached for
an operator the system cannot supply.  Conditioning on evocation rather than on
absence is what keeps the guard from eating ordinary references, and the cost of
that choice is that cue detection has to be accurate: 前処理 must not read as
「前」, 次元 must not read as 「次」.  Cue detection is therefore structural too --
a cue character buried inside a longer compound is not a cue.

Deliberately **not** conditioned on whether the anchor resolved.  That was the
tempting version, and it is the one that produces the paradox above.

### Pre-registered expectations

Written before L8.18.1 was generated or run, so the hold-out tests a stated
prediction rather than a summary of what happened:

1. known relation, unseen vocabulary -> resolved correctly (unchanged from L8.17)
2. unknown relation **with** a resolvable anchor -> ABSTAIN (this is the L8.17.1 defect)
3. unknown relation without an anchor -> ABSTAIN (already held in L8.17)
4. bare reference -> latest episode, unchanged
5. ordinary request containing 前処理 / 後処理 / 次元 -> **not** abstained, i.e. the
   guard costs no capability on non-relational text
6. unknown operation (「一番多く使ったのは」) -> ABSTAIN rather than a lookup result
7. wrong_reuse_rate 0.0 and unnecessary_abstention_rate 0.0, simultaneously
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re

from .episode_relation_l817_experiment import (
    RelationKind,
    build_relation_graph,
    parse_relation,
    traverse,
)
from .episodic_memory_l813_experiment import EpisodicMemory, ReferenceResolution
from .structural_resolver_l815_experiment import query_tokens, resolve_reference_structural


class SlotKind(Enum):
    ANCHOR = "ANCHOR"
    RELATION = "RELATION"
    PROJECTION = "PROJECTION"
    OPERATION = "OPERATION"


#: Cue characters that only count as relational when they stand on their own.
#: 前 in 前処理 is part of a word, not a step between episodes.
_BARE_CUES: tuple[str, ...] = ("前", "次", "後", "昔")

#: Compounds that *are* relational even though they are multi-character runs.
_COMPOUND_CUES: tuple[str, ...] = (
    "直前", "直後", "以前", "以後", "前回", "次回", "手前", "最初", "最後", "初回",
)

#: Compounds that merely contain a cue character and must never fire.  Listed so
#: the failure mode is visible; the structural rule below is what does the work.
_CUE_LOOKALIKES: tuple[str, ...] = ("前処理", "後処理", "次元", "事前", "事後", "前提", "背後")

_PROJECTION_CUES: tuple[str, ...] = ("一字一句", "そのまま", "原文", "逐語", "パラメータ", "設定値")
_KNOWN_PROJECTIONS: tuple[str, ...] = ("一字一句", "そのまま", "原文", "逐語", "パラメータ", "設定値")

_OPERATION_CUES: tuple[tuple[str, str | None], ...] = (
    ("何件", "COUNT"),
    ("いくつ", "COUNT"),
    ("何個", "COUNT"),
    ("件数", "COUNT"),
    # Evoked but unimplemented.  Listed so they ABSTAIN instead of falling
    # through to a plain lookup that returns one episode as if it were an
    # aggregate -- the same defect one level further out.  L8.19 resolves them.
    ("一番多い", None),
    ("一番多く", None),
    ("最も多く", None),
    ("平均", None),
    ("増えた", None),
    ("減った", None),
    ("より大きく", None),
    ("中央値", None),
    ("分散", None),
)

_KANJI_RUN = re.compile(r"[一-龥]{2,}")


def has_relation_cue(request: str) -> bool:
    """A cue is relational only when it is not buried inside a longer word."""
    if any(compound in request for compound in _COMPOUND_CUES):
        return True
    runs = _KANJI_RUN.findall(request)
    for cue in _BARE_CUES:
        if cue not in request:
            continue
        # Present, but if every occurrence sits inside a multi-kanji run it is
        # part of a word such as 前処理 rather than a step between episodes.
        buried = sum(run.count(cue) for run in runs)
        if request.count(cue) > buried:
            return True
    return False


@dataclass(frozen=True)
class Slot:
    kind: SlotKind
    evoked: bool
    resolved: object | None

    @property
    def blocks(self) -> bool:
        return self.evoked and self.resolved is None


@dataclass(frozen=True)
class QueryIR:
    request: str
    slots: tuple[Slot, ...]

    def slot(self, kind: SlotKind) -> Slot:
        return next(item for item in self.slots if item.kind is kind)

    @property
    def blocking(self) -> tuple[SlotKind, ...]:
        return tuple(item.kind for item in self.slots if item.blocks)


def parse_query_ir(episodic: EpisodicMemory, request: str) -> QueryIR:
    relation_kind, remainder = parse_relation(request)
    relation_evoked = relation_kind is not None or has_relation_cue(request)

    projection_evoked = any(cue in request for cue in _PROJECTION_CUES)
    projection = next((cue for cue in _KNOWN_PROJECTIONS if cue in request), None)

    operation_evoked = False
    operation: str | None = None
    for cue, resolved in _OPERATION_CUES:
        if cue in request:
            operation_evoked = True
            operation = resolved
            if resolved is not None:
                break

    anchor_text = remainder if relation_kind is not None else request
    anchor: str | None = None
    anchor_evoked = bool(query_tokens(anchor_text))
    if anchor_evoked:
        resolution = resolve_reference_structural(episodic, anchor_text)
        if resolution.decision == "CONTINUE_EPISODE":
            anchor = resolution.episode_id

    return QueryIR(
        request,
        (
            Slot(SlotKind.ANCHOR, anchor_evoked, anchor),
            Slot(SlotKind.RELATION, relation_evoked, relation_kind),
            Slot(SlotKind.PROJECTION, projection_evoked, projection),
            Slot(SlotKind.OPERATION, operation_evoked, operation),
        ),
    )


@dataclass(frozen=True)
class QueryOutcome:
    decision: str  # ANSWER | ABSTAIN | NEW_TASK
    episode_id: str | None
    value: object | None
    reason: str
    blocking: tuple[SlotKind, ...] = ()


def _count(episodic: EpisodicMemory, request: str) -> int:
    tokens = [
        token
        for token in query_tokens(request)
        if not any(cue in token for cue, _ in _OPERATION_CUES)
    ]
    if not tokens:
        return 0
    return sum(
        1
        for episode in episodic.episodes
        if all(token in episode.request for token in tokens)
    )


def execute(episodic: EpisodicMemory, request: str) -> QueryOutcome:
    """Parse, gate, then dispatch.  The gate runs before anything is answered."""
    ir = parse_query_ir(episodic, request)

    blocking = ir.blocking
    # ANCHOR alone never blocks: a request with no identifiable episode is an
    # ordinary miss that the resolver below reports in its own vocabulary.
    blocking = tuple(kind for kind in blocking if kind is not SlotKind.ANCHOR)
    if blocking:
        names = ", ".join(kind.value for kind in blocking)
        return QueryOutcome(
            "ABSTAIN", None, None,
            f"{names} was asked for but could not be resolved; "
            f"answering from the rest of the request would return an operand as a result",
            blocking,
        )

    operation = ir.slot(SlotKind.OPERATION)
    if operation.evoked and operation.resolved == "COUNT":
        return QueryOutcome("ANSWER", None, _count(episodic, request), "counted over the episode index")

    relation = ir.slot(SlotKind.RELATION)
    if relation.resolved is not None:
        graph = build_relation_graph(episodic)
        resolved = traverse(graph, relation.resolved, ir.slot(SlotKind.ANCHOR).resolved)
        if resolved.decision == "CONTINUE_EPISODE":
            return QueryOutcome("ANSWER", resolved.episode_id, None, resolved.reason)
        return QueryOutcome("ABSTAIN", None, None, resolved.reason, (SlotKind.RELATION,))

    fallback = resolve_reference_structural(episodic, request)
    if fallback.decision == "CONTINUE_EPISODE":
        return QueryOutcome("ANSWER", fallback.episode_id, None, fallback.reason)
    if fallback.decision == "NEW_EPISODE":
        return QueryOutcome("NEW_TASK", None, None, fallback.reason)
    return QueryOutcome("ABSTAIN", None, None, fallback.reason)


def resolve_reference_gated(episodic: EpisodicMemory, request: str) -> ReferenceResolution:
    """Drop-in resolver so the earlier benchmarks can measure this layer."""
    outcome = execute(episodic, request)
    if outcome.decision == "ANSWER" and outcome.episode_id is not None:
        return ReferenceResolution(
            "CONTINUE_EPISODE", outcome.episode_id, (outcome.episode_id,), outcome.reason
        )
    if outcome.decision == "NEW_TASK":
        return ReferenceResolution("NEW_EPISODE", None, (), outcome.reason)
    return ReferenceResolution("CLARIFY", None, (), outcome.reason)
