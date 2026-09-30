"""L8.33: an answer is not a request, and remembering which question it answers
is what keeps a conversation from drifting.

L8.32 can decide what to ask.  Asking it creates a problem the earlier layers
never had: the next thing the user says is 「いいえ」, and routing that as a fresh
request is exactly the failure this whole project started from -- a turn
interpreted without the state that gives it meaning.  So the question is not a
side effect of the answer.  It is a state.

    Goal -> Belief -> Identifiability -> Question -> Answer -> Observation
         -> Belief' -> Resume

Asking suspends the goal inside the episode.  The reply is interpreted **relative
to the pending question**, converted to an observation, folded into L8.30's
belief, and only then does the suspended goal run again.  Nothing about the
original request is re-derived: it was kept, because it was never finished.

**Only three kinds of reply commit anything.**  A plain yes, a plain no, and a
correction that names the field outright.  Everything else -- 「たぶん違う」,
「分からない」, 「質問の意味が分からない」 -- commits *no observation at all*.  That is
the whole safety content of the layer, and the case that matters most is
``UNKNOWN``: treating "I don't know" as a no is the obvious shortcut and it
silently manufactures evidence the user never gave.  A hedge is not an answer
either; 「たぶん違う」 is a person declining to commit, and recording it as a
commitment is the same mistake wearing a different word.

**An answer binds to the question that is pending, or to nothing.**  Each
question carries an id.  A reply while a newer question is open binds to the
newer one; a reply arriving after the goal has already resumed or been abandoned
binds to *nothing* and is treated as a fresh turn.  Without that, 「いいえ」 typed
one turn too late is silently applied to a question that was already answered --
the stale-binding accident, and the reason ``stale_question_answer_binding_rate``
is measured rather than assumed.

**Interruption ends the goal without evidence.**  「やっぱりこの作業やめて」 is not an
answer and not a new request; it abandons the suspended goal and commits nothing,
so a cancelled task leaves no trace in what the system believes words mean.

A question is only ever asked when the request cannot be answered without it and
L8.31 says the word is separable.  Asking about a structurally unidentifiable
word would be a question with no answer that helps, so the agent says that
instead -- which is the difference between "I need more information" and "no
information available here would settle this".
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

from .active_observation_l832_experiment import (
    STRUCTURALLY_UNRESOLVABLE, answer_as_observation, question_for,
)
from .cross_context_identification_l829_experiment import (
    FieldFacts, Observation, available_probes, observations_from_request,
    run_acquired_query, universe_from,
)
from .evidence_graph_l827_experiment import build_graph, readings, run_graph_query
from .lexical_revision_l830_experiment import RevisableLexicon
from .semantic_identifiability_l831_experiment import (
    STRUCTURALLY_UNIDENTIFIABLE, analyse,
)
from .semantic_proposer_l822_experiment import Proposer
from .span_grounded_proposer_l823_experiment import SpanProposer
from .structural_sensor_l826_experiment import StructuralSensor
from .typed_operation_l819_experiment import FIELDS, TypedStore, ValueType

IDLE, AWAITING_ANSWER = "IDLE", "AWAITING_ANSWER"

#: What a reply can be.  Only the first three carry evidence.
AFFIRM, DENY, CORRECTION = "AFFIRM", "DENY", "CORRECTION"
HEDGE, UNKNOWN, CONFUSED, INTERRUPT, OTHER = (
    "HEDGE", "UNKNOWN", "CONFUSED", "INTERRUPT", "OTHER")

COMMITTING = frozenset({AFFIRM, DENY, CORRECTION})

#: Ordered longest-first so 「たぶん違う」 is a hedge and not a denial. The order is
#: the rule: a hedge contains a denial as a substring, and matching the short
#: form first would turn every hesitation into a commitment.
_CUES: tuple[tuple[str, str], ...] = (
    ("質問の意味が分からない", CONFUSED),
    ("質問がわからない", CONFUSED),
    ("意味が分からない", CONFUSED),
    ("どういう意味", CONFUSED),
    ("たぶん違う", HEDGE),
    ("たぶんそう", HEDGE),
    ("かもしれません", HEDGE),
    ("かもしれない", HEDGE),
    ("だと思います", HEDGE),
    ("と思う", HEDGE),
    ("分からない", UNKNOWN),
    ("わからない", UNKNOWN),
    ("知らない", UNKNOWN),
    ("覚えていない", UNKNOWN),
    ("やっぱりやめて", INTERRUPT),
    ("やめて", INTERRUPT),
    ("中止", INTERRUPT),
    ("もういい", INTERRUPT),
    ("キャンセル", INTERRUPT),
    ("じゃなくて", CORRECTION),
    ("ではなくて", CORRECTION),
    ("いいえ", DENY),
    ("違います", DENY),
    ("違う", DENY),
    ("ちがう", DENY),
    ("はい", AFFIRM),
    ("そうです", AFFIRM),
    ("そう", AFFIRM),
    ("ええ", AFFIRM),
)


def classify(utterance: str) -> str:
    """What kind of reply this is, read left to right through the cue table.

    Deliberately a table and not a model: the layer under test is the dialogue
    state machine, and a learned classifier here would make every failure
    ambiguous between "the state machine is wrong" and "the classifier missed".
    """
    for cue, kind in _CUES:
        if cue in utterance:
            return kind
    return OTHER


def named_field(utterance: str) -> str | None:
    """Which field a correction names, via L8.19's own cue table."""
    for spec in FIELDS:
        for cue in spec.cues:
            if cue in utterance:
                return spec.name
    return None


# --------------------------------------------------------------------------
# the state
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class PendingQuestion:
    question_id: int
    query: Observation
    text: str
    surface: str


@dataclass(frozen=True)
class SuspendedGoal:
    """The unfinished request, kept rather than re-derived when the answer lands."""

    request: str
    surface: str
    candidates: frozenset[str]


@dataclass
class Dialogue:
    store: TypedStore
    sensors: tuple[SpanProposer, Proposer, StructuralSensor]
    universe: tuple[FieldFacts, ...]
    lexicon: RevisableLexicon
    state: str = IDLE
    pending: PendingQuestion | None = None
    suspended: SuspendedGoal | None = None
    asked: int = 0
    transcript: list["Turn"] = field(default_factory=list)

    @classmethod
    def over(cls, store: TypedStore) -> "Dialogue":
        universe = universe_from(store)
        return cls(store, (SpanProposer.trained(), Proposer.trained(), StructuralSensor()),
                   universe, RevisableLexicon(universe))


@dataclass(frozen=True)
class Turn:
    utterance: str
    kind: str
    #: Which question this reply was bound to, or None when it was bound to none.
    bound_to: int | None
    observation: Observation | None
    decision: str
    reply: str
    resumed: str | None = None


# --------------------------------------------------------------------------
# the driver
# --------------------------------------------------------------------------

def _unnamed_span(dialogue: Dialogue, request: str) -> str | None:
    graph = build_graph(request, dialogue.store, *dialogue.sensors)
    node = readings(graph)[1]
    return node.text if node is not None else None


def _attempt(dialogue: Dialogue, request: str):
    """Run the request through the pipeline the earlier layers already built."""
    return run_acquired_query(dialogue.store, request, dialogue.lexicon, *dialogue.sensors)


def _ask_about(dialogue: Dialogue, request: str, surface: str):
    """Plan a question, or explain why no question would help.

    The request is folded in first.  「処理時間ではなくフガ率を…」 already excludes one
    field, and planning from a blank belief would ask about it again -- a
    question whose answer the user has just given in the same sentence.
    """
    _, from_request = observations_from_request(
        request, dialogue.store, *dialogue.sensors)
    for observation in from_request:
        dialogue.lexicon.observe(surface, observation)
    belief = dialogue.lexicon.belief(surface)
    candidates = belief.candidates if belief.observations else frozenset(
        f.name for f in dialogue.universe)
    pool = available_probes(dialogue.universe)
    analysis = analyse(dialogue.universe, tuple(belief.observations),
                       next(iter(candidates)), pool)
    text, query = question_for(candidates, dialogue.universe, pool, surface)
    if query is None or analysis.verdict == STRUCTURALLY_UNIDENTIFIABLE:
        return None, (f"「{surface}」の意味を今の観測語彙では決められません"
                      f"（候補: {'、'.join(sorted(candidates))}）。")
    dialogue.asked += 1
    dialogue.pending = PendingQuestion(dialogue.asked, query, text, surface)
    dialogue.suspended = SuspendedGoal(request, surface, candidates)
    dialogue.state = AWAITING_ANSWER
    return query, text


def _observation_for(pending: PendingQuestion, kind: str, utterance: str) -> Observation | None:
    """The evidence a reply actually carries. Most replies carry none."""
    if kind == AFFIRM:
        return answer_as_observation(pending.query, True)
    if kind == DENY:
        return answer_as_observation(pending.query, False)
    if kind == CORRECTION:
        named = named_field(utterance)
        if named is None:
            return None  # a correction that names nothing is not evidence
        return Observation("EQUALS", named, f"「{utterance}」")
    return None


def handle(dialogue: Dialogue, utterance: str) -> Turn:
    """One turn. The reply's meaning depends on what the conversation is doing."""
    kind = classify(utterance)

    # ---- a reply while a question is open -------------------------------
    if dialogue.state == AWAITING_ANSWER and dialogue.pending is not None:
        pending, suspended = dialogue.pending, dialogue.suspended

        if kind == INTERRUPT:
            dialogue.state, dialogue.pending, dialogue.suspended = IDLE, None, None
            turn = Turn(utterance, kind, pending.question_id, None, "ABANDONED",
                        "中断しました。何も記録していません。")
            dialogue.transcript.append(turn)
            return turn

        observation = _observation_for(pending, kind, utterance)
        if observation is None:
            # No evidence. The goal stays suspended and the question stays open,
            # because dropping either would lose the only context that makes the
            # next utterance interpretable.
            reply = {
                UNKNOWN: "分からない、として扱います。この質問は取り下げます。",
                HEDGE: "確定的な回答ではないので記録しません。",
                CONFUSED: "質問を変えます。",
            }.get(kind, "回答として解釈できませんでした。")
            if kind in (UNKNOWN, CONFUSED):
                # Retire this question rather than re-asking it, and try another.
                remaining = dialogue.lexicon.belief(pending.surface).candidates
                pool = tuple(q for q in available_probes(dialogue.universe)
                             if q != pending.query)
                text, query = question_for(
                    remaining or suspended.candidates, dialogue.universe, pool,
                    pending.surface)
                if query is None:
                    dialogue.state, dialogue.pending, dialogue.suspended = IDLE, None, None
                    reply = (f"「{pending.surface}」について他に訊けることがありません。")
                else:
                    dialogue.asked += 1
                    dialogue.pending = PendingQuestion(
                        dialogue.asked, query, text, pending.surface)
                    reply = f"{reply} {text}"
            turn = Turn(utterance, kind, pending.question_id, None, "NO_COMMIT", reply)
            dialogue.transcript.append(turn)
            return turn

        dialogue.lexicon.observe(pending.surface, observation)
        dialogue.state, dialogue.pending = IDLE, None
        result = _attempt(dialogue, suspended.request)
        dialogue.suspended = None
        turn = Turn(utterance, kind, pending.question_id, observation,
                    result.decision,
                    f"「{suspended.request}」を再開しました。", suspended.request)
        dialogue.transcript.append(turn)
        return turn

    # ---- no question open ------------------------------------------------
    if kind in COMMITTING and dialogue.pending is None:
        # An answer with nothing to answer. Binding it to the last question would
        # be the stale-binding accident; it binds to nothing.
        turn = Turn(utterance, kind, None, None, "UNBOUND",
                    "どの質問への回答か分からないので、何も記録しません。")
        dialogue.transcript.append(turn)
        return turn

    if kind == INTERRUPT:
        turn = Turn(utterance, kind, None, None, "IDLE", "進行中の作業はありません。")
        dialogue.transcript.append(turn)
        return turn

    result = _attempt(dialogue, utterance)
    needs_name = (result.decision == "ABSTAIN"
                  and "no sensor could name it" in result.reason)
    if not needs_name:
        turn = Turn(utterance, "REQUEST", None, None, result.decision, "実行しました。")
        dialogue.transcript.append(turn)
        return turn

    surface = _unnamed_span(dialogue, utterance) or utterance
    query, text = _ask_about(dialogue, utterance, surface)
    turn = Turn(utterance, "REQUEST", dialogue.asked if query else None, None,
                "ASKED" if query else "UNANSWERABLE", text)
    dialogue.transcript.append(turn)
    return turn
