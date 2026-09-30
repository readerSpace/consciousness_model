"""The L8.27-L8.42 semantic layers, wired into the agent's session, UI and API.

Everything from the evidence graph onwards has so far lived in benchmarks.  This
module is the seam that puts it behind the product: one session object that owns
the interpretation state, a command surface the desktop app and the web UI both
call, and a JSON snapshot the inspector panel renders.

The composition is deliberately thin, because the layers already decide
everything that matters:

* **L8.27-L8.33** drive the conversation.  ``Dialogue`` parses the request into an
  evidence graph, notices the argument it cannot name, suspends the goal, asks,
  binds the reply to the question it answers, folds it into the belief, and
  resumes.  Nothing here re-implements any of that; the session holds the
  ``Dialogue`` and reports what it did.
* **L8.31** says whether the word is learnable at all, which is what separates
  "keep asking" from ``IMPOSSIBLE``.
* **L8.35** records what the surface meant *per episode* and decides whether the
  split is noise, a revision or genuine polysemy -- the session never decides
  that itself.
* **L8.36** supplies the question for the case L8.35 leaves open.  A session that
  visits episodes in order makes time and episode perfectly correlated, so both
  explain the split and ``AMBIGUOUS`` is the correct answer; the only thing that
  separates them is a question about the *model*, and the panel offers it.
* **L8.42** gates every question on whether the person can actually answer it.  A
  question about an episode they have not opened is not asked, and the session
  reports ``BLOCKED`` rather than ``IMPOSSIBLE`` -- the distinction that keeps a
  task one action away from done from being abandoned.

**What the audit can and cannot claim here.**  L8.40's benchmark computes
``wrong_answer`` against a known truth.  In a real session there is no oracle, so
that number cannot be produced honestly and this module does not produce it.
What it reports instead are the invariants that need no oracle -- a program
executed with no authority behind it, a lexicon exposing an entry whose belief is
disputed, evidence recorded from an utterance that committed to nothing, and a
program run in one episode with a meaning that belongs to another.  Each is a
real failure detectable from the session alone, each should stay at zero, and the
panel labels them as invariants rather than as accuracy.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .dialogue_script import help_text, looks_like_script, run as run_script
from .logic_reasoner import (
    LogicError, answer as answer_logic, is_logic_query, parse_binding,
)
from .math_expression import answer_query, is_query
from .physics_script import looks_like_physics, run as run_physics
from .theorem_service import (
    THEOREM_FEATURE_COMMANDS, handle_command as handle_theorem,
)

from .contextual_polysemy_l835_experiment import (
    AMBIGUOUS, POLYSEMY, REVISION, ContextualLexicon,
)
from .cost_aware_dialogue_l836_experiment import (
    ASK as COST_ASK, IMPOSSIBLE as COST_IMPOSSIBLE,
    MULTIPLE_OPTIMAL, STRUCTURE, choose, hypotheses,
)
from .cross_context_identification_l829_experiment import (
    Observation, available_probes, run_acquired_query, universe_from,
)
from .epistemic_dialogue_l833_experiment import (
    AWAITING_ANSWER, COMMITTING, Dialogue, handle,
)
from .holdout_binding_l8211_experiment import build_holdout
from .holdout_lexicon_l8281_experiment import widen_store
from .lexical_revision_l830_experiment import DISPUTED, REVOKED
from .query_algebra_l820_experiment import canonical_render
from .question_accessibility_l842_experiment import Access
from .semantic_identifiability_l831_experiment import (
    STRUCTURALLY_UNIDENTIFIABLE, analyse,
)

ANSWER = "ANSWER"
ASKED = "ASKED"
ABSTAIN = "ABSTAIN"
BLOCKED = "BLOCKED"
IMPOSSIBLE = "IMPOSSIBLE"
CONTRADICTION = "CONTRADICTION"
RESOLVED = "RESOLVED"
UNRESOLVED = "UNRESOLVED"
NO_COMMIT = "NO_COMMIT"

DEFAULT_EPISODE = "current"

SEMANTIC_FEATURE_COMMANDS: tuple[tuple[str, str], ...] = (
    ("/py <code>", "対話を Python 風の呼び出しで書く（```python ブロックも可）"),
    ("/physics <script>", "場と式を宣言して軌跡を解き、図にする（```physics ブロックも可）"),
    ("<式>=?", "算術をその場で計算する（例: 1+1=?）"),
    ("<関係>ならば？", "集合・順序の関係から導けることを出す（例: A in B and B in A ならば？）"),
    ("in := subset", "曖昧な記号の読みを固定する（subset ／ member ／ proper）"),
) + THEOREM_FEATURE_COMMANDS + (
    ("/ask <request>", "L8.27-L8.42 要求を解釈にかける（未知語なら質問して再開する）"),
    ("/semantics", "L8.27-L8.42 解釈状態（候補・質問・語義・不変条件）"),
    ("/semantics episodes", "episode ごとの参照可否と語義"),
    ("/semantics open <episode>", "L8.42 その episode を開いて質問可能にする"),
    ("/semantics episode <name>", "現在の episode を切り替える"),
    ("/semantics model <label>", "L8.36 構造質問に答える（時期 / 場面）"),
    ("/semantics reset", "解釈状態を捨てて最初からやり直す"),
    ("/answer <reply>", "保留中の質問に答える（はい／いいえ／分からない／訂正）"),
)


@dataclass(frozen=True)
class Slot:
    """One editable part of a template."""

    name: str
    label: str
    default: str
    options: tuple[str, ...] = ()


@dataclass(frozen=True)
class Template:
    """A request or reply form whose behaviour is *measured*, not asserted.

    The forms here are the ones the frozen dialogue hold-out (L8.33.1) drives
    the system with, and ``expect`` records what that hold-out established each
    one does.  ``agent_contract_verification`` replays every template through the
    running bridge and checks it still does that, so "verified" in the UI is a
    property that was observed this run rather than a label someone typed.
    """

    id: str
    kind: str  # REQUEST | ANSWER | COMMAND
    label: str
    form: str
    slots: tuple[Slot, ...] = ()
    note: str = ""
    #: What replaying it should produce. ``decision`` is the dialogue's own
    #: verdict and ``commits`` whether the utterance recorded any evidence.
    expect: tuple[tuple[str, object], ...] = ()
    verified_by: str = ""
    #: The same move written as a call, for the script surface. Same slots.
    code: str = ""


REQUEST_FORM = "{lattice}のケースで{excluded}ではなく{surface}をならして"
ANSWERABLE_FORM = "{lattice}を使った実験の平均実行時間は？"

TEMPLATES: tuple[Template, ...] = (
    Template(
        "unknown_argument", "REQUEST", "未知語を含む依頼", REQUEST_FORM,
        (Slot("lattice", "条件", "64x64"),
         Slot("excluded", "除外する指標", "処理時間"),
         Slot("surface", "未知語", "フガ率")),
        "名前の付かない引数が残るので、実行せずに質問する",
        (("decision", "ASKED"), ("commits", False)), "L8.33.1 / 契約検証",
        'ask("' + REQUEST_FORM + '")'),
    Template(
        "answerable", "REQUEST", "未知語のない依頼", ANSWERABLE_FORM,
        (Slot("lattice", "条件", "64x64"),),
        "対照: 答えられる依頼で質問を出してはいけない",
        (("decision", "ANSWER"), ("commits", False)), "L8.33.1 NO_QUESTION",
        'ask("' + ANSWERABLE_FORM + '")'),
    Template(
        "affirm", "ANSWER", "はい（肯定）", "はい", (),
        "肯定も証拠。もう一方の候補が残る",
        (("decision", "ANSWER"), ("commits", True)), "L8.33.1 AFFIRM", "yes()"),
    Template(
        "deny", "ANSWER", "いいえ（否定）", "いいえ", (),
        "否定は証拠の不在ではなく、逆向きの証拠",
        (("decision", "ANSWER"), ("commits", True)), "L8.33.1 DENY", "no()"),
    Template(
        "correction", "ANSWER", "訂正（指標を名指す）", "それじゃなくて{field}",
        (Slot("field", "正しい指標", "成功率",
              ("成功率", "実行時間", "エネルギー誤差")),),
        "訊かれた以上のことに答えている発話",
        (("decision", "ANSWER"), ("commits", True)), "L8.33.1 CORRECTION",
        'correct("{field}")'),
    Template(
        "unknown", "ANSWER", "分からない", "分からない", (),
        "否定として扱わない。質問を取り下げて別を訊く",
        (("decision", "NO_COMMIT"), ("commits", False)), "L8.33.1 UNKNOWN",
        "dont_know()"),
    Template(
        "hedge", "ANSWER", "たぶん違う（確言しない）", "たぶん違う", (),
        "確定的でない回答は記録しない",
        (("decision", "NO_COMMIT"), ("commits", False)), "L8.33.1 HEDGE", "hedge()"),
    Template(
        "confused", "ANSWER", "質問の意味が分からない", "質問の意味が分からない", (),
        "訊き方を変える。何も記録しない",
        (("decision", "NO_COMMIT"), ("commits", False)), "L8.33.1 CONFUSED",
        "confused()"),
    Template(
        "interrupt", "ANSWER", "作業をやめる", "やっぱりこの作業やめて", (),
        "中断は語の意味に痕跡を残さない",
        (("decision", "ABANDONED"), ("commits", False)), "L8.33.1 INTERRUPT",
        "interrupt()"),
    Template(
        "ask_command", "COMMAND", "解釈にかける", "/ask " + REQUEST_FORM,
        (Slot("lattice", "条件", "64x64"),
         Slot("excluded", "除外する指標", "処理時間"),
         Slot("surface", "未知語", "フガ率")),
        "平文チャットは素通りするので、明示的にパイプラインへ入れる",
        (), "契約検証 /ask", 'ask("' + REQUEST_FORM + '")'),
    Template(
        "open_episode", "COMMAND", "episode を開く", "/semantics open {episode}",
        (Slot("episode", "episode 名", "lab"),),
        "BLOCKED の質問が訊けるようになる（L8.42）",
        (), "契約検証 /api/semantic/open",
        'open_episode("{episode}")'),
    Template(
        "switch_episode", "COMMAND", "episode を切り替える",
        "/semantics episode {episode}", (Slot("episode", "episode 名", "lab"),),
        "以後の質問と実行はこの episode のものになる",
        (), "契約検証 /api/semantic/episode",
        'episode("{episode}")'),
    Template(
        "model", "COMMAND", "モデルを答える", "/semantics model {model}",
        (Slot("model", "どちらか", "場面", ("場面", "時期")),),
        "時間と場面が両方説明してしまうとき、どちらかを名指す（L8.36）",
        (), "契約検証 /api/semantic/model", 'model("{model}")'),
    Template(
        "logic_query", "COMMAND", "関係から導けることを出す",
        "({left} {rel} {right}) and ({right} {rel} {left}) ならば？",
        (Slot("left", "左の集合", "A"), Slot("right", "右の集合", "B"),
         Slot("rel", "関係", "in", ("in", "⊆", "∈", "⊊"))),
        "`in` のように読みが複数ある記号は、決めずに全ての読みで答える",
        (), "契約検証 logic",
        "({left} {rel} {right}) and ({right} {rel} {left}) ならば？"),
    Template(
        "math_query", "COMMAND", "その場で計算する", "{left}{op}{right}=?",
        (Slot("left", "左", "1"), Slot("op", "演算", "+", ("+", "-", "*", "/", "^")),
         Slot("right", "右", "1")),
        "式として読めたときだけ答える。散文は素通りする",
        (), "契約検証 math", "{left}{op}{right}=?"),
    Template(
        "lorentz_trajectory", "COMMAND", "ローレンツ力の軌跡を描く",
        "Bz = {bz}\nm = {m}\nvector: B = (0, 0, Bz)\nlorentz: F exists\n"
        "calc(m d^2 x / dt^2 = lorentz(B))\ndraw(x)",
        (Slot("bz", "磁場 Bz", "1.0"), Slot("m", "質量 m", "1.0")),
        "値のない記号があれば実行せずに挙げる（Bz や m を消すと分かる）",
        (), "契約検証 physics",
        "Bz = {bz}\nm = {m}\nvector: B = (0, 0, Bz)\nlorentz: F exists\n"
        "calc(m d^2 x / dt^2 = lorentz(B))\ndraw(x)"),
    Template(
        "state", "COMMAND", "解釈状態を出す", "/semantics", (),
        "候補・質問・語義・不変条件をチャットに書き出す",
        (), "契約検証 /semantics", "state()"),
)


def render_template(template: Template, values: dict | None = None) -> str:
    """Fill the slots, falling back to each slot's default."""
    filled = {slot.name: (values or {}).get(slot.name) or slot.default
              for slot in template.slots}
    return template.form.format(**filled)


def render_code(template: Template, values: dict | None = None) -> str:
    """The same move as a call, with the same slots filled in."""
    if not template.code:
        return ""
    filled = {slot.name: (values or {}).get(slot.name) or slot.default
              for slot in template.slots}
    return template.code.format(**filled)


def template_by_id(identifier: str) -> Template | None:
    return next((item for item in TEMPLATES if item.id == identifier), None)


def templates_payload() -> list[dict]:
    return [{
        "id": item.id, "kind": item.kind, "label": item.label,
        "form": item.form, "note": item.note, "verified_by": item.verified_by,
        "expect": dict(item.expect),
        "preview": render_template(item),
        "code": render_code(item),
        "slots": [{"name": slot.name, "label": slot.label,
                   "default": slot.default, "options": list(slot.options)}
                  for slot in item.slots],
    } for item in TEMPLATES]


def default_store():
    """The frozen hold-out episodes the semantic layers were measured on.

    The agent's *workspace* is a folder of code; the semantic layers need a store
    of recorded episodes, which a folder does not provide.  Rather than invent
    one, the session runs on the same frozen store the benchmarks use and the
    panel says so, so what the UI shows is the real machinery on real data.
    """
    return widen_store(build_holdout(24)[0])


@dataclass(frozen=True)
class TurnRecord:
    kind: str
    text: str
    status: str
    reply: str
    bound_to: int | None
    committed: bool
    invariants: dict


@dataclass
class SemanticSession:
    """One person's interpretation state, across episodes and across turns."""

    store: object = None
    episode: str = DEFAULT_EPISODE
    episodes: tuple[str, ...] = (DEFAULT_EPISODE,)
    dialogue: Dialogue = None
    senses: ContextualLexicon = None
    access: Access = None
    turns: list[TurnRecord] = field(default_factory=list)
    decided: dict = field(default_factory=dict)
    #: Readings the person has fixed for an ambiguous operator (`in := subset`).
    #: The same move as answering a question about a word, so it lives with the
    #: rest of the interpretation state rather than in the reasoner.
    operators: dict = field(default_factory=dict)
    clock: int = 0
    #: The last request the person made, kept so the panel can offer to run it
    #: again once the word turns out to mean different things in different
    #: episodes. Without it the comparison would have to be retyped.
    last_request: str | None = None

    def __post_init__(self):
        self.store = self.store if self.store is not None else default_store()
        if self.dialogue is None:
            self.reset()

    # -- lifecycle -----------------------------------------------------
    def reset(self) -> None:
        self.dialogue = Dialogue.over(self.store)
        universe = universe_from(self.store)
        self.senses = ContextualLexicon(universe, probes=available_probes(universe))
        self.access = Access.of(known=self.episodes, detail=(self.episode,))
        self.turns = []
        self.decided = {}
        self.operators = {}
        self.clock = 0
        self.last_request = None

    @property
    def universe(self):
        return self.dialogue.universe

    # -- accessibility (L8.42) -----------------------------------------
    def open_episode(self, name: str) -> None:
        if name not in self.episodes:
            self.episodes = self.episodes + (name,)
        self.access = self.access.opening(name)

    def use_episode(self, name: str) -> None:
        if name not in self.episodes:
            self.episodes = self.episodes + (name,)
            self.access = Access.of(known=self.episodes, detail=self.access.detail)
        self.episode = name

    @property
    def recallable(self) -> bool:
        return self.episode in self.access.detail

    @property
    def blocked(self) -> bool:
        """A question is waiting and its episode is not open.

        L8.42's distinction, in the product: the question exists and is out of
        reach, which is not the same as there being nothing to ask.
        """
        return self.dialogue.pending is not None and not self.recallable

    # -- the conversation ----------------------------------------------
    def ask(self, message: str) -> dict:
        if self.dialogue.state == AWAITING_ANSWER:
            return self.answer(message)
        self.last_request = message
        turn = handle(self.dialogue, message)
        return self._record("REQUEST", message, turn)

    def answer(self, text: str) -> dict:
        if self.blocked:
            # Never fold in a reply to a question the person could not see the
            # basis for. The goal stays suspended and the question stays open.
            return self._snapshot(BLOCKED,
                                  "この質問は今の episode を開かないと答えられません。")
        turn = handle(self.dialogue, text)
        return self._record("ANSWER", text, turn)

    def _record(self, kind: str, text: str, turn) -> dict:
        self.clock += 1
        invariants = self._invariants(turn)
        self._learn_sense(turn)
        status = BLOCKED if self.blocked else turn.decision
        self.turns.append(TurnRecord(kind, text, status, turn.reply,
                                     turn.bound_to, turn.observation is not None,
                                     invariants))
        return self._snapshot(status, turn.reply, turn)

    def _learn_sense(self, turn) -> None:
        """Record what each authoritative surface meant, once per *use*.

        One observation per program that actually ran, not one per distinct
        meaning: L8.35 will not explain a block of size one, so a session that
        deduplicated would make polysemy undetectable by construction rather than
        by evidence.  A goal resuming and executing is a genuine separate use of
        the word in that episode, which is what the evidence should be.
        """
        if turn.decision != ANSWER:
            return
        for surface, belief in self.dialogue.lexicon.beliefs.items():
            if belief.authoritative and belief.meaning:
                self.senses.observe(surface, Observation("EQUALS", belief.meaning),
                                    self.episode, self.clock)

    # -- the invariants that need no oracle ----------------------------
    def _invariants(self, turn) -> dict:
        lexicon = self.dialogue.lexicon
        entries = lexicon.entries
        executed = turn.decision == ANSWER

        disputed = any(belief.state in (DISPUTED, REVOKED) and surface in entries
                       for surface, belief in lexicon.beliefs.items())
        # A leak is a meaning belonging to *another* episode, which is strictly
        # narrower than "not what this episode last recorded": revising a word
        # inside one episode changes that episode's own meaning and is not a
        # leak. Counting it as one would make the invariant fire on correct
        # behaviour, which is worse than not having it.
        stale = False
        if executed:
            for surface, meaning in entries.items():
                here = self.sense_for(surface, self.episode)
                elsewhere = {self.sense_for(surface, episode)
                             for episode in self.episodes if episode != self.episode}
                if here is not None and here != meaning and meaning in elsewhere:
                    stale = True
        return {
            "executed_without_authority": int(executed and not entries),
            "authority_while_disputed": int(disputed),
            "committed_without_commitment": int(
                turn.observation is not None and turn.kind not in COMMITTING),
            "stale_episode_leak": int(stale),
        }

    def totals(self) -> dict:
        keys = ("executed_without_authority", "authority_while_disputed",
                "committed_without_commitment", "stale_episode_leak")
        return {key: sum(t.invariants.get(key, 0) for t in self.turns)
                for key in keys}

    # -- what the panel shows ------------------------------------------
    def surface(self) -> str | None:
        if self.dialogue.pending is not None:
            return self.dialogue.pending.surface
        if self.dialogue.suspended is not None:
            return self.dialogue.suspended.surface
        live = [s for s, b in self.dialogue.lexicon.beliefs.items() if b.observations]
        return live[-1] if live else None

    def candidates(self) -> list[str]:
        surface = self.surface()
        if surface is None:
            return []
        belief = self.dialogue.lexicon.belief(surface)
        if not belief.observations:
            return sorted(facts.name for facts in self.universe)
        return sorted(belief.candidates or belief.repaired)

    def diagnosis(self) -> str:
        """RESOLVED / ASKED / BLOCKED / IMPOSSIBLE / CONTRADICTION / UNRESOLVED.

        ``IMPOSSIBLE`` comes from L8.31 -- the observation language cannot
        separate the candidates, so no amount of asking helps -- and ``BLOCKED``
        from L8.42.  Keeping them apart is the point of both layers.
        """
        surface = self.surface()
        if surface is None:
            return UNRESOLVED
        belief = self.dialogue.lexicon.belief(surface)
        if belief.authoritative and belief.meaning:
            return RESOLVED
        candidates = self.candidates()
        if not candidates:
            return CONTRADICTION
        if len(candidates) == 1:
            return RESOLVED
        analysis = analyse(self.universe, tuple(belief.observations),
                           candidates[0], available_probes(self.universe))
        if analysis.verdict == STRUCTURALLY_UNIDENTIFIABLE:
            return IMPOSSIBLE
        if self.blocked:
            return BLOCKED
        if self.dialogue.pending is not None:
            return ASKED
        return UNRESOLVED

    # -- the model question (L8.35 leaves it open, L8.36 asks it) -------
    def _evidence(self, surface: str) -> tuple:
        return tuple(self.senses.evidence.get(surface, ()))

    def _structural_choice(self, surface: str):
        """The model question L8.36 would ask, and the space it is asking about.

        Among the tied structural questions the one whose branches *are* the
        models is preferred: a yes/no about one of them costs the same and
        answers less, and the person is being asked which account is true, not
        whether one particular account is.
        """
        evidence = self._evidence(surface)
        if not evidence:
            return None
        space = hypotheses(evidence, self.universe)
        if len(space) < 2:
            return None
        contexts = tuple(sorted({item.context for item in evidence}))
        meanings = tuple(sorted(facts.name for facts in self.universe))
        choice = choose(space, contexts, meanings)
        if choice.state not in (COST_ASK, MULTIPLE_OPTIMAL):
            return None
        structural = tuple(q for q in choice.questions if q.kind == STRUCTURE)
        if not structural:
            return None
        # L8.36 reports MULTIPLE_OPTIMAL rather than picking when several tie, and
        # that is kept rather than collapsed: the branches carry the models, so
        # the person's answer selects a question *and* a branch together.
        return structural, space

    def structural_question(self, surface: str | None = None) -> dict | None:
        """The question that separates "the meaning changed" from "two senses".

        Offered only where L8.35 actually reports ``AMBIGUOUS``; where the
        evidence already decides, asking would be the avoidable interaction
        L8.36 measures and refuses.
        """
        surface = surface or self.surface()
        if surface is None or surface in self.decided:
            return None
        explanation = self.senses.explanations.get(surface)
        if explanation is None or explanation.kind != AMBIGUOUS:
            return None
        found = self._structural_choice(surface)
        if found is None:
            return None
        questions, space = found
        return {
            "surface": surface,
            "models": sorted({h.model for h in space}),
            "questions": [{
                "text": question.text,
                "cost": list(question.cost.vector),
                "branches": [{"label": name,
                              "models": sorted({h.model for h in block}),
                              "count": len(block)}
                             for name, block in question.outcomes if block],
            } for question in questions],
            "text": questions[0].text,
            "options": sorted({model for question in questions
                               for name, block in question.outcomes if block
                               for model in {h.model for h in block}}),
        }

    def answer_structural(self, label: str, surface: str | None = None) -> dict:
        """Fold the person's answer into the hypothesis space, L8.36's way."""
        surface = surface or self.surface()
        question = self.structural_question(surface)
        if question is None:
            return self._snapshot(self.diagnosis(),
                                  "いま構造質問は保留されていません。")
        questions, _ = self._structural_choice(surface)
        wanted = _model_of(label)
        block = None
        for asked in questions:
            for _name, candidates in asked.outcomes:
                if candidates and {h.model for h in candidates} == {wanted}:
                    block = candidates
                    break
            if block is not None:
                break
        if block is None:
            options = "／".join(question["options"])
            return self._snapshot(
                self.diagnosis(),
                f"答えを解釈できませんでした。どのモデルか名前で答えてください（{options}）。")
        if len(block) == 1:
            self.decided[surface] = next(iter(block))
            return self._snapshot(self.diagnosis(),
                                  f"「{surface}」のモデルを {self.decided[surface].model} "
                                  f"として確定しました。")
        return self._snapshot(self.diagnosis(),
                              f"候補が {len(block)} 件残りました。もう一度質問します。")

    def sense_for(self, surface: str, episode: str) -> str | None:
        """What the surface means in an episode, after any decided model."""
        chosen = self.decided.get(surface)
        if chosen is not None:
            return chosen.meaning_in(episode)
        return self.senses.meaning(surface, episode)

    def lexicon_rows(self) -> list[dict]:
        rows = []
        for surface, belief in sorted(self.dialogue.lexicon.beliefs.items()):
            rows.append({
                "surface": surface, "state": belief.state,
                "meaning": belief.meaning, "believed": belief.believed,
                "authoritative": belief.authoritative,
                "candidates": sorted(belief.candidates or belief.repaired),
                "observations": [f"{o.kind}({o.payload})" for o in belief.observations],
                "discarded": list(belief.discarded),
            })
        return rows

    def sense_rows(self) -> list[dict]:
        rows = []
        for surface, explanation in sorted(self.senses.explanations.items()):
            chosen = self.decided.get(surface)
            model = chosen.model if chosen is not None else explanation.kind
            rows.append({
                "surface": surface, "model": model,
                "decided": chosen is not None,
                "conditioned": model in (POLYSEMY, REVISION),
                "by_episode": {episode: self.sense_for(surface, episode)
                               for episode in self.episodes},
            })
        return rows

    def episode_rows(self) -> list[dict]:
        return [{"name": episode,
                 "referable": episode in self.access.known,
                 "recallable": episode in self.access.detail,
                 "current": episode == self.episode}
                for episode in self.episodes]

    def _snapshot(self, status: str, reply: str = "", turn=None) -> dict:
        pending = self.dialogue.pending
        return {
            "status": status,
            "reply": reply,
            "diagnosis": self.diagnosis(),
            "episode": self.episode,
            "episodes": self.episode_rows(),
            "blocked": self.blocked,
            "surface": self.surface(),
            "candidates": self.candidates(),
            "question": None if pending is None else {
                "id": pending.question_id, "text": pending.text,
                "surface": pending.surface, "askable": self.recallable,
            },
            "structural_question": self.structural_question(),
            "suspended": None if self.dialogue.suspended is None
                         else self.dialogue.suspended.request,
            "last_request": self.last_request,
            "lexicon": self.lexicon_rows(),
            "senses": self.sense_rows(),
            "invariants": self.totals(),
            "turns": [{"kind": t.kind, "text": t.text, "status": t.status,
                       "bound_to": t.bound_to, "committed": t.committed}
                      for t in self.turns[-12:]],
            "store": "frozen hold-out episodes (L8.21.1 + L8.28.1)",
            "bound_to": None if turn is None else turn.bound_to,
            "resumed": None if turn is None else turn.resumed,
        }

    def state(self) -> dict:
        return self._snapshot(self.diagnosis())

    # -- re-running a resolved goal ------------------------------------
    def rerun(self, request: str) -> dict:
        """Run a request again per episode, through whatever each one means.

        This is where a word that turned out to have two senses shows itself: the
        same request, the same surface, two programs.
        """
        found = {}
        for episode in self.episodes:
            view = _EpisodeView(self, episode)
            result = run_acquired_query(self.store, request, view,
                                        *self.dialogue.sensors)
            found[episode] = (canonical_render(result.expr)
                              if result.decision == ANSWER and result.expr is not None
                              else None)
        return found


@dataclass(frozen=True)
class _EpisodeView:
    """The session's lexicon fixed to one episode, in the executor's shape.

    Falls back to the single acquired meaning when no sense has been recorded for
    that episode, so an ordinary monosemous word keeps working unchanged and the
    executor never learns that senses exist.
    """

    session: SemanticSession
    episode: str

    def lookup(self, text: str) -> str | None:
        for surface in self.session.dialogue.lexicon.beliefs:
            if surface in text:
                found = self.session.sense_for(surface, self.episode)
                if found is not None:
                    return found
        return self.session.dialogue.lexicon.lookup(text)


#: The person answers in their own words; the branches are model names.
_MODEL_WORDS = {
    "時期": REVISION, "時間": REVISION, "変わった": REVISION, "revision": REVISION,
    "場面": POLYSEMY, "文脈": POLYSEMY, "多義": POLYSEMY, "polysemy": POLYSEMY,
    "monosemy": "MONOSEMY", "noise": "NOISE",
}


def _model_of(label: str) -> str | None:
    """Read the person's words onto the model they name, not onto a yes/no.

    The tied structural questions are yes/no about *different* models, so a bare
    "yes" would be ambiguous between them; naming the model picks the question
    and the branch at once.
    """
    text = label.strip()
    lowered = text.lower()
    if text.upper() in (REVISION, POLYSEMY, "MONOSEMY", "NOISE"):
        return text.upper()
    for cue, model in _MODEL_WORDS.items():
        if cue in lowered:
            return model
    return None


# --------------------------------------------------------------------------
# the command surface both UIs call
# --------------------------------------------------------------------------

def _markdown_state(state: dict) -> str:
    lines = [f"**解釈状態: {state['status']} / diagnosis {state['diagnosis']}**",
             f"- episode: `{state['episode']}`",
             f"- 対象語: {state['surface'] or '-'}",
             f"- 候補: {'、'.join(state['candidates']) or '-'}"]
    if state["suspended"]:
        lines.append(f"- 保留中の目標: 「{state['suspended']}」")
    if state["question"]:
        mark = "" if state["question"]["askable"] else \
            "（今は答えられません: `/semantics open <episode>` で開いてください）"
        lines.append(f"- 質問 #{state['question']['id']}: "
                     f"{state['question']['text']}{mark}")
    if state["structural_question"]:
        found = state["structural_question"]
        lines.append(f"- 構造質問: {found['text']}"
                     f"（`/semantics model {'／'.join(found['options'])}`）")
    if state["lexicon"]:
        lines += ["", "| 語 | 状態 | 意味 | 候補 |", "| --- | --- | --- | --- |"]
        for row in state["lexicon"]:
            lines.append(f"| {row['surface']} | {row['state']} "
                         f"| {row['meaning'] or '-'} "
                         f"| {'、'.join(row['candidates']) or '-'} |")
    if state["senses"]:
        lines += ["", "| 語 | モデル | episode ごとの語義 |", "| --- | --- | --- |"]
        for row in state["senses"]:
            per = "、".join(f"{k}→{v or '-'}" for k, v in row["by_episode"].items())
            mark = "（確定）" if row["decided"] else ""
            lines.append(f"| {row['surface']} | {row['model']}{mark} | {per} |")
    lines += ["", "**セッション不変条件（真値を必要としないものだけ）**",
              "| 不変条件 | 件数 |", "| --- | --- |"]
    for key, value in state["invariants"].items():
        lines.append(f"| {key} | {value} |")
    lines += ["", f"（episodes: {state['store']}）"]
    return "\n".join(lines)


def _markdown_episodes(state: dict) -> str:
    lines = ["**episode ごとのアクセスと語義（L8.42 / L8.35）**", "",
             "| episode | 参照可 | 想起可 | 現在 |", "| --- | --- | --- | --- |"]
    for row in state["episodes"]:
        lines.append(f"| {row['name']} | {row['referable']} | {row['recallable']} "
                     f"| {'●' if row['current'] else ''} |")
    if state["senses"]:
        lines += ["", "| 語 | モデル | episode ごとの語義 |", "| --- | --- | --- |"]
        for row in state["senses"]:
            per = "、".join(f"{k}→{v or '-'}" for k, v in row["by_episode"].items())
            lines.append(f"| {row['surface']} | {row['model']} | {per} |")
    return "\n".join(lines)


PHYSICS_USAGE = """使い方: 場と式を宣言して、解いて、描きます。

```physics
Bz = 1.0
m = 1.0
vector: B = (0, 0, Bz)
lorentz: F exists
calc(m d^2 x / dt^2 = lorentz(B))
draw(x)
```

値を書いていない記号があると**実行せずに**それを挙げます。"""


USAGE = ("使い方: /semantics ／ /semantics episodes ／ "
         "/semantics open <episode> ／ /semantics episode <name> ／ "
         "/semantics model <時期|場面> ／ /semantics reset")


def handle_command(session: SemanticSession, message: str) -> str | None:
    """Return markdown for a semantic command, or ``None`` if it is not one.

    Shared by the desktop app and the HTTP bridge so both surfaces show the same
    state; neither owns a second copy of the logic.
    """
    text = message.strip()
    # The theorem bundle has its own command prefix and its own process; it is
    # asked first because nothing else claims `/theorem`.
    spoken = handle_theorem(text)
    if spoken is not None:
        return spoken
    # A question about a number is answered as one. `is_query` only says yes to
    # something ending in `=?` or made of arithmetic alone, so prose with a
    # number in it is untouched.
    if is_query(text):
        return answer_query(text)
    # `in := subset` fixes a reading for the rest of the session, which is the
    # same move as answering a question about an unknown word.
    try:
        binding = parse_binding(text)
    except LogicError as error:
        return str(error)
    if binding is not None:
        symbol, reading = binding
        session.operators[symbol] = reading
        return (f"`{symbol}` をこの読みに固定しました。"
                f"以後の論理の質問はこの読みだけで答えます。")
    if is_logic_query(text):
        return answer_logic(text, session.operators)["output"]
    if text.startswith("/physics"):
        source = text[len("/physics"):].strip()
        if not source:
            return PHYSICS_USAGE
        return run_physics(source)["output"]
    if looks_like_physics(text):
        return run_physics(text)["output"]
    # The script surface comes first: a message that parses completely into
    # known calls was written as code and means it. Anything else -- prose, a
    # slash command, a reply -- fails to parse and falls straight through.
    if text.startswith("/py"):
        source = text[len("/py"):].strip()
        if not source:
            return help_text()
        return run_script(session, source)["output"]
    if looks_like_script(text):
        return run_script(session, text)["output"]
    if text.startswith("/ask "):
        # The explicit way in. Plain chat is left alone on purpose: routing every
        # message through the evidence graph would change what the workspace
        # agent does with messages that have nothing to do with an unknown word.
        state = session.ask(text[len("/ask "):].strip())
        return f"{state['reply']}\n\n{_markdown_state(state)}"
    if text == "/ask":
        return "使い方: /ask <要求文>（例: /ask 64x64のケースで処理時間ではなくフガ率をならして）"
    if text.startswith("/answer "):
        state = session.answer(text[len("/answer "):].strip())
        return f"{state['reply']}\n\n{_markdown_state(state)}"
    if text == "/answer":
        return "使い方: /answer はい ／ /answer いいえ ／ /answer 分からない"
    if not text.startswith("/semantics"):
        return None
    rest = text[len("/semantics"):].strip()
    if not rest:
        return _markdown_state(session.state())
    if rest == "episodes":
        return _markdown_episodes(session.state())
    if rest == "reset":
        session.reset()
        return "解釈状態を初期化しました。\n\n" + _markdown_state(session.state())
    if rest.startswith("open "):
        session.open_episode(rest[len("open "):].strip())
        return _markdown_state(session.state())
    if rest.startswith("episode "):
        session.use_episode(rest[len("episode "):].strip())
        return _markdown_state(session.state())
    if rest.startswith("model "):
        state = session.answer_structural(rest[len("model "):].strip())
        return f"{state['reply']}\n\n{_markdown_state(state)}"
    return USAGE


def render_state(state: dict) -> str:
    return _markdown_state(state)
