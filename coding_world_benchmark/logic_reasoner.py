"""Set and order relations, closed under a named handful of rules.

``(A in B) and (B in A) ならば？`` has two answers, and which one is right depends
on what ``in`` means:

* read as ``⊆``, antisymmetry gives **A = B**;
* read as ``∈``, regularity gives **a contradiction** -- no two sets in ZF
  contain each other as members.

So this is the project's own problem again, in a third costume.  ``in`` is a
symbol the person used whose meaning the system does not have, exactly like
「フガ率」 and like ``Bz``, and the rule the dialogue layers arrived at applies
without change: **do not answer as though the ambiguity were not there.**  What
the reasoner does instead is carry every reading at once -- it derives the
consequences under each, labels them, and says how to fix the reading if the
person wants one answer.  Fixing it (``in := subset``) binds the symbol for the
session in the same way answering a question binds a word.

``⊂`` is ambiguous for a different reason and gets the same treatment: some
authors write it for ``⊆`` and others for ``⊊``, and a reasoner that silently
picks one is telling half its readers something false.

**What is inside the fence.**  The rules below are the whole engine; there is no
search, no arithmetic, and no set construction.  Premises are conjunctions of
atoms over named sets, and the derivation is a forward closure that stops at a
fixed point.  Anything else -- disjunctive premises, quantifiers, negated
subset claims, concrete sets -- is refused by name rather than approximated,
which is the fail-closed default the rest of the project runs on.

**Non-consequences are reported too.**  A reasoner that only prints what follows
lets the reader supply the rest, and the usual mistakes here are about what does
*not* follow: ``∈`` is not transitive, and ``⊆`` never gives ``∈``.  Those are
listed beside the derivation when the premises invite them.
"""
from __future__ import annotations

import itertools
import re
from dataclasses import dataclass

SUBSET, PROPER, MEMBER, EQUAL, NOT_EQUAL = "⊆", "⊊", "∈", "=", "≠"
RELATIONS = (SUBSET, PROPER, MEMBER, EQUAL, NOT_EQUAL)

READABLE = {
    SUBSET: "部分集合 ⊆", PROPER: "真部分集合 ⊊", MEMBER: "要素 ∈",
    EQUAL: "等しい =", NOT_EQUAL: "等しくない ≠",
}

#: Spellings that mean exactly one relation. ``⊇`` and friends are the mirror of
#: another relation, so they are stored as (relation, swap).
DIRECT = {
    "⊆": (SUBSET, False), "⊑": (SUBSET, False), "<=": (SUBSET, False),
    "subseteq": (SUBSET, False), "subset of": (SUBSET, False),
    "⊇": (SUBSET, True), ">=": (SUBSET, True), "supseteq": (SUBSET, True),
    "⊊": (PROPER, False), "⊈": (PROPER, False), "<": (PROPER, False),
    "⊋": (PROPER, True), ">": (PROPER, True),
    "∈": (MEMBER, False), "elem": (MEMBER, False), "member of": (MEMBER, False),
    "∋": (MEMBER, True),
    "=": (EQUAL, False), "==": (EQUAL, False), "≡": (EQUAL, False),
    "≠": (NOT_EQUAL, False), "!=": (NOT_EQUAL, False), "/=": (NOT_EQUAL, False),
}

#: Spellings that mean more than one thing. The reasoner carries every reading
#: rather than choosing, which is the point of the module.
AMBIGUOUS = {
    "in": ((SUBSET, False), (MEMBER, False)),
    "⊂": ((SUBSET, False), (PROPER, False)),
    "⊃": ((SUBSET, True), (PROPER, True)),
}

READING_NAMES = {
    (SUBSET, False): "subset", (MEMBER, False): "member",
    (PROPER, False): "proper", (SUBSET, True): "supset",
    (PROPER, True): "proper_supset",
}
BY_NAME = {
    "subset": (SUBSET, False), "⊆": (SUBSET, False), "部分集合": (SUBSET, False),
    "member": (MEMBER, False), "∈": (MEMBER, False), "要素": (MEMBER, False),
    "proper": (PROPER, False), "⊊": (PROPER, False), "真部分集合": (PROPER, False),
}

#: Four labelled readings is already a lot to read; beyond that the honest move
#: is to ask for one to be fixed rather than to print a wall of cases. The bound
#: is reachable with the table above (three ambiguous symbols make eight), so it
#: is a live rail rather than a comment.
MAX_READINGS = 4


class LogicError(ValueError):
    """A question that will not be answered, and why."""


@dataclass(frozen=True)
class Fact:
    relation: str
    left: str
    right: str

    def __str__(self) -> str:
        return f"{self.left} {self.relation} {self.right}"


@dataclass(frozen=True)
class Step:
    fact: Fact
    rule: str
    because: tuple


CONTRADICTION = "CONTRADICTION"

#: ``=`` and ``≠`` are symmetric, so the two orientations are one fact. Storing
#: them in a fixed order keeps a derivation from listing the same thing twice
#: and makes the output stable.
SYMMETRIC = (EQUAL, NOT_EQUAL)


def fact(relation: str, left: str, right: str) -> Fact:
    if relation in SYMMETRIC and right < left:
        left, right = right, left
    return Fact(relation, left, right)


# --------------------------------------------------------------------------
# the rules, each named so a derivation can be read
# --------------------------------------------------------------------------

RULE_NAMES = {
    "antisymmetry": "反対称律（外延性）: X ⊆ Y かつ Y ⊆ X ならば X = Y",
    "subset_transitive": "⊆ の推移律: X ⊆ Y かつ Y ⊆ Z ならば X ⊆ Z",
    "equal_subset": "= の展開: X = Y ならば X ⊆ Y かつ Y ⊆ X",
    "proper_subset": "⊊ の展開: X ⊊ Y ならば X ⊆ Y かつ X ≠ Y",
    "proper_asymmetry": "⊊ の非対称性: X ⊊ Y かつ Y ⊊ X は成り立たない",
    "regularity": "正則性公理: X ∈ Y かつ Y ∈ X となる集合は存在しない",
    "self_membership": "正則性公理: X ∈ X となる集合は存在しない",
    "member_subset": "∈ と ⊆ の合成: X ∈ Y かつ Y ⊆ Z ならば X ∈ Z",
    "subset_reflexive": "⊆ の反射律: X ⊆ X",
    "equal_not_equal": "矛盾: X = Y と X ≠ Y は両立しない",
    "narrow_to_proper": "X ⊆ Y かつ X ≠ Y ならば X ⊊ Y",
}

NON_CONSEQUENCES = (
    (MEMBER, "∈ は推移的ではありません: X ∈ Y かつ Y ∈ Z でも X ∈ Z とは限らない"),
    (SUBSET, "⊆ からは ∈ は出ません: X ⊆ Y は X ∈ Y を含意しない"),
)


def close(premises) -> tuple[set, list, str | None]:
    """Forward-chain to a fixed point, keeping why each fact arrived."""
    facts = set(premises)
    steps: list[Step] = []
    contradiction: str | None = None

    def add(fact: Fact, rule: str, because) -> bool:
        if fact in facts:
            return False
        facts.add(fact)
        steps.append(Step(fact, rule, tuple(because)))
        return True

    for name in {fact.left for fact in premises} | {fact.right for fact in premises}:
        add(fact(SUBSET, name, name), "subset_reflexive", ())

    changed = True
    while changed and contradiction is None:
        changed = False
        current = list(facts)
        for one in current:
            if one.relation == MEMBER and one.left == one.right:
                contradiction = "self_membership"
                break
            if one.relation == EQUAL and one.left != one.right:
                changed |= add(fact(SUBSET, one.left, one.right), "equal_subset", (one,))
                changed |= add(fact(SUBSET, one.right, one.left), "equal_subset", (one,))
            if one.relation == PROPER:
                changed |= add(fact(SUBSET, one.left, one.right), "proper_subset", (one,))
                changed |= add(fact(NOT_EQUAL, one.left, one.right), "proper_subset", (one,))
            for other in current:
                if one.relation == SUBSET and other.relation == SUBSET:
                    if one.left == other.right and one.right == other.left \
                            and one.left != one.right:
                        changed |= add(fact(EQUAL, one.left, one.right),
                                       "antisymmetry", (one, other))
                    if one.right == other.left and one.left != other.right:
                        changed |= add(fact(SUBSET, one.left, other.right),
                                       "subset_transitive", (one, other))
                if one.relation == MEMBER and other.relation == MEMBER \
                        and one.left == other.right and one.right == other.left:
                    contradiction = "regularity"
                    break
                if one.relation == PROPER and other.relation == PROPER \
                        and one.left == other.right and one.right == other.left:
                    contradiction = "proper_asymmetry"
                    break
                if one.relation == MEMBER and other.relation == SUBSET \
                        and one.right == other.left and one.right != other.right:
                    changed |= add(fact(MEMBER, one.left, other.right),
                                   "member_subset", (one, other))
                if one.relation == EQUAL and other.relation == NOT_EQUAL \
                        and {one.left, one.right} == {other.left, other.right}:
                    contradiction = "equal_not_equal"
                    break
                if one.relation == SUBSET and other.relation == NOT_EQUAL \
                        and (one.left, one.right) in ((other.left, other.right),
                                                      (other.right, other.left)):
                    changed |= add(fact(PROPER, one.left, one.right),
                                   "narrow_to_proper", (one, other))
            if contradiction is not None:
                break
    return facts, steps, contradiction


# --------------------------------------------------------------------------
# reading the question
# --------------------------------------------------------------------------

_SPELLINGS = sorted(set(DIRECT) | set(AMBIGUOUS), key=len, reverse=True)
_NAME = r"[A-Za-z_Α-ω][A-Za-z_0-9']*"
_ATOM = re.compile(
    r"^[\s(（]*(?P<left>" + _NAME + r")\s*"
    r"(?P<relation>" + "|".join(re.escape(item) for item in _SPELLINGS) + r")\s*"
    r"(?P<right>" + _NAME + r")[\s)）]*$", re.IGNORECASE)

_CONJUNCTION = re.compile(r"\s+and\s+|\s*&&\s*|\s*∧\s*|\s*,\s*|かつ|、", re.IGNORECASE)
_DISJUNCTION = re.compile(r"\s+or\s+|\s*\|\|\s*|\s*∨\s*|または", re.IGNORECASE)
_THEN = re.compile(r"ならば|=>|⇒|⊢|therefore|なら", re.IGNORECASE)
_ASKS = re.compile(r"[?？]\s*$|ならば\s*[?？]?\s*$|何が言える|どうなる")
_BINDING = re.compile(r"^\s*(?P<symbol>in|⊂|⊃)\s*:=\s*(?P<reading>\S+)\s*$",
                      re.IGNORECASE)


@dataclass(frozen=True)
class Question:
    premises: tuple            # tuple of (left, spelling, right)
    conclusion: tuple | None   # the same shape, when one was proposed
    source: str


def _atom(text: str) -> tuple:
    match = _ATOM.match(text.strip())
    if match is None:
        raise LogicError(f"`{text.strip()}` を関係として読めません")
    return (match.group("left"), match.group("relation").lower(),
            match.group("right"))


def parse(question: str) -> Question:
    text = question.strip()
    text = re.sub(r"[?？]\s*$", "", text).strip()
    left, _, right = "", "", ""
    split = _THEN.split(text, maxsplit=1)
    left = split[0]
    right = split[1] if len(split) > 1 else ""
    if _DISJUNCTION.search(left):
        raise LogicError("前提の「または」はまだ扱えません（「かつ」だけです）")
    premises = tuple(_atom(part) for part in _CONJUNCTION.split(left) if part.strip())
    if not premises:
        raise LogicError("前提が読めません")
    conclusion = _atom(right) if right.strip() else None
    return Question(premises, conclusion, question.strip())


def ambiguous_symbols(question: Question) -> list[str]:
    used = {spelling for _, spelling, _ in question.premises}
    if question.conclusion:
        used.add(question.conclusion[1])
    return sorted(symbol for symbol in used if symbol in AMBIGUOUS)


def _resolve(atom, choices: dict) -> Fact:
    left, spelling, right = atom
    relation, swap = choices.get(spelling) or DIRECT[spelling]
    return fact(relation, right, left) if swap else fact(relation, left, right)


def readings(question: Question, bound: dict | None = None):
    """Every way of reading the ambiguous symbols, with the bound ones fixed."""
    bound = {key.lower(): value for key, value in (bound or {}).items()}
    open_symbols = [symbol for symbol in ambiguous_symbols(question)
                    if symbol not in bound]
    fixed = {symbol: bound[symbol] for symbol in ambiguous_symbols(question)
             if symbol in bound}
    if not open_symbols:
        return [dict(fixed)]
    combinations = list(itertools.product(
        *[AMBIGUOUS[symbol] for symbol in open_symbols]))
    if len(combinations) > MAX_READINGS:
        raise LogicError("曖昧な記号が多すぎます。`in := subset` のように決めてください")
    return [dict(fixed, **dict(zip(open_symbols, choice)))
            for choice in combinations]


def parse_binding(message: str) -> tuple[str, tuple] | None:
    """``in := subset`` -- the same move as answering a question about a word."""
    match = _BINDING.match(message)
    if match is None:
        return None
    reading = BY_NAME.get(match.group("reading").lower())
    if reading is None:
        raise LogicError(
            f"`{match.group('reading')}` という読みは知りません"
            f"（{'、'.join(sorted(set(BY_NAME) - set('⊆∈⊊')))}）")
    return match.group("symbol").lower(), reading


# --------------------------------------------------------------------------
# answering
# --------------------------------------------------------------------------

def _label(choices: dict) -> str:
    if not choices:
        return "唯一の読み"
    return "、".join(f"`{symbol}` = {READABLE[relation]}"
                     + ("（左右入れ替え）" if swap else "")
                     for symbol, (relation, swap) in sorted(choices.items()))


def analyse(question: Question, choices: dict) -> dict:
    premises = [_resolve(atom, choices) for atom in question.premises]
    facts, steps, contradiction = close(premises)
    given = set(premises)
    # ``A = B`` and ``B = A`` are the same fact written twice; keeping both
    # makes a derivation look longer than it is.
    seen: set = set()
    derived = []
    for step in steps:
        if step.fact in given or step.rule == "subset_reflexive":
            continue
        key = (step.fact.relation, frozenset((step.fact.left, step.fact.right))) \
            if step.fact.relation in (EQUAL, NOT_EQUAL) else step.fact
        if key in seen:
            continue
        seen.add(key)
        derived.append(step)
    verdict, answer = "DERIVED", None
    if contradiction is not None:
        verdict = CONTRADICTION
        answer = RULE_NAMES[contradiction]
    elif question.conclusion is not None:
        wanted = _resolve(question.conclusion, choices)
        verdict = "FOLLOWS" if wanted in facts else "DOES_NOT_FOLLOW"
        answer = str(wanted)
    elif derived:
        answer = "、".join(str(step.fact) for step in derived)
    else:
        verdict = "NOTHING_NEW"
    return {"choices": choices, "label": _label(choices),
            "premises": [str(fact) for fact in premises],
            "derived": [{"fact": str(step.fact), "rule": step.rule,
                         "why": RULE_NAMES[step.rule],
                         "because": [str(item) for item in step.because]}
                        for step in derived],
            "verdict": verdict, "answer": answer,
            "contradiction": contradiction}


def _non_consequences(question: Question, choices: dict) -> list[str]:
    used = {_resolve(atom, choices).relation for atom in question.premises}
    return [text for relation, text in NON_CONSEQUENCES if relation in used]


def answer(question: str, bound: dict | None = None) -> dict:
    """Every reading's consequences, labelled, with nothing chosen for you."""
    try:
        parsed = parse(question)
        choices = readings(parsed, bound)
    except LogicError as error:
        return {"ok": False, "error": str(error), "readings": [],
                "ambiguous": [], "output": f"**答えませんでした**: {error}"}

    results = [analyse(parsed, choice) for choice in choices]
    open_symbols = [symbol for symbol in ambiguous_symbols(parsed)
                    if symbol not in {key.lower() for key in (bound or {})}]
    agree = len({(item["verdict"], item["answer"]) for item in results}) == 1

    lines: list[str] = []
    if open_symbols and not agree:
        lines += [f"**`{'`、`'.join(open_symbols)}` の読みによって答えが変わります。**"
                  "どちらか決めずに片方だけ答えるのは、決まっていないことを"
                  "決まったことにするので、両方出します。", ""]
    elif open_symbols:
        lines += [f"（`{'`、`'.join(open_symbols)}` は複数の読みがありますが、"
                  "どの読みでも同じ結論になります。）", ""]

    for item in results:
        if len(results) > 1:
            lines.append(f"### 読み: {item['label']}")
        lines.append("前提: " + "、".join(f"`{fact}`" for fact in item["premises"]))
        if item["verdict"] == CONTRADICTION:
            lines.append(f"→ **矛盾**。{item['answer']}")
        elif item["verdict"] == "FOLLOWS":
            lines.append(f"→ **成り立ちます**: `{item['answer']}`")
        elif item["verdict"] == "DOES_NOT_FOLLOW":
            lines.append(f"→ **出てきません**: `{item['answer']}` "
                         "はこの前提からは導けません")
        elif item["verdict"] == "NOTHING_NEW":
            lines.append("→ 前提以上のことは出てきません")
        else:
            lines.append(f"→ **{item['answer']}**")
        if item["derived"]:
            lines += ["", "| 導かれた事実 | 根拠 | 使った前提 |",
                      "| --- | --- | --- |"]
            for step in item["derived"]:
                lines.append(f"| `{step['fact']}` | {step['why']} "
                             f"| {'、'.join(f'`{x}`' for x in step['because']) or '-'} |")
        lines.append("")

    notes = sorted({note for item in results
                    for note in _non_consequences(parsed, item["choices"])})
    if notes:
        lines += ["**よくある取り違え（成り立たないもの）**", ""]
        lines += [f"- {note}" for note in notes]
        lines.append("")
    if open_symbols:
        options = "／".join(
            f"`{symbol} := {READING_NAMES[reading]}`"
            for symbol in open_symbols for reading in AMBIGUOUS[symbol])
        lines.append(f"読みを決めるなら: {options}")
    return {"ok": True, "error": None, "readings": results,
            "ambiguous": open_symbols, "agree": agree,
            "output": "\n".join(lines).strip()}


def is_logic_query(message: str) -> bool:
    """Is this a question about relations between named sets?

    Conservative on purpose: it has to ask something, and it has to parse
    completely into relations over names. ``1+1=?`` is arithmetic and is handled
    before this; a sentence with a stray ``in`` in it does not parse.
    """
    text = message.strip()
    if not _ASKS.search(text) and not _THEN.search(text):
        return False
    if not any(spelling in text.lower() for spelling in _SPELLINGS):
        return False
    try:
        parse(text)
    except LogicError:
        return False
    return True
