"""Controlled Japanese text-reasoning test for compressed belief revision.

The input language intentionally has a small, explicit grammar.  This tests
multi-hop inference and later contradictory evidence, not broad language-model
ability.  Derived propositions are mirrored into the compressed workspace;
B3 freezes its first answer while B4 re-evaluates it after each new sentence.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import re
from typing import Dict, Iterable, Mapping, Sequence, Tuple

from Consciousness_model import CompressedWorkspace


Fact = Tuple[str, str, bool]
Rule = Tuple[str, str]


def _negative_predicate(stem: str) -> str:
    """Normalize the limited negative forms used by this controlled grammar."""
    if stem.endswith("てい"):
        return stem + "る"
    if stem.endswith("く"):
        return stem[:-1] + "い"
    return stem


def _universal_source(noun_phrase: str) -> str:
    return noun_phrase[:-2] if noun_phrase.endswith("もの") else noun_phrase


@dataclass(frozen=True)
class TextCase:
    name: str
    sentences: Tuple[str, ...]
    query: Tuple[str, str]
    expected: str
    commit_after: int


CASES = (
    TextCase(
        "多段含意",
        ("アオイはキツネである。", "すべてのキツネは速い。", "すべての速いものは警戒している。"),
        ("アオイ", "警戒している"), "真", 3,
    ),
    TextCase(
        "情報不足",
        ("アオイはキツネである。", "すべてのキツネは速い。"),
        ("アオイ", "警戒している"), "不明", 2,
    ),
    TextCase(
        "後続矛盾",
        ("アオイはキツネである。", "すべてのキツネは速い。", "すべての速いものは警戒している。", "アオイは警戒していない。"),
        ("アオイ", "警戒している"), "不明", 3,
    ),
    TextCase(
        "否定含意",
        ("レンは金属である。", "すべての金属は重い。", "レンは軽くない。"),
        ("レン", "軽い"), "偽", 3,
    ),
)


class TextBeliefState:
    def __init__(self) -> None:
        self.facts: set[Fact] = set()
        self.rules: set[Rule] = set()
        self.workspace = CompressedWorkspace(capacity=16)

    def add(self, sentence: str) -> None:
        sentence = sentence.strip()
        universal = re.fullmatch(r"すべての(.+)は(.+)。", sentence)
        negative = re.fullmatch(r"(.+)は(.+)ない。", sentence)
        positive = re.fullmatch(r"(.+)は(.+)である。", sentence)
        if universal:
            self.rules.add((_universal_source(universal.group(1)), universal.group(2)))
        elif negative:
            self.facts.add((negative.group(1), _negative_predicate(negative.group(2)), False))
        elif positive:
            self.facts.add((positive.group(1), positive.group(2), True))
        else:
            raise ValueError(f"unsupported sentence: {sentence}")
        self._derive()

    def _derive(self) -> None:
        changed = True
        while changed:
            changed = False
            for subject, predicate, truth in tuple(self.facts):
                if truth:
                    for source, target in self.rules:
                        if predicate == source and (subject, target, True) not in self.facts:
                            self.facts.add((subject, target, True))
                            changed = True
        self.workspace.ingest((subject, "is" if truth else "is-not", predicate) for subject, predicate, truth in self.facts)

    def answer(self, subject: str, predicate: str) -> str:
        positive = (subject, predicate, True) in self.facts
        negative = (subject, predicate, False) in self.facts
        if positive and negative:
            return "不明"  # contradiction: revoke a single committed truth value
        if positive:
            return "真"
        if negative:
            return "偽"
        return "不明"


def _evaluate(case: TextCase, adaptive: bool) -> tuple[float, float]:
    state = TextBeliefState()
    committed = None
    revisions = 0
    for index, sentence in enumerate(case.sentences, start=1):
        state.add(sentence)
        if index == case.commit_after:
            committed = state.answer(*case.query)
        elif adaptive and committed is not None:
            revised = state.answer(*case.query)
            if revised != committed:
                revisions += 1
                committed = revised
    answer = committed if committed is not None else state.answer(*case.query)
    return float(answer == case.expected), float(revisions)


def run(cases: Iterable[TextCase] = CASES) -> Mapping[str, Mapping[str, float]]:
    rows = tuple(cases)
    result: Dict[str, Mapping[str, float]] = {}
    for condition, adaptive in (("B3", False), ("B4", True)):
        scores = [_evaluate(case, adaptive) for case in rows]
        result[condition] = {
            "accuracy": sum(score for score, _ in scores) / len(scores),
            "mean_revision_count": sum(revision for _, revision in scores) / len(scores),
        }
    return result


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2, sort_keys=True))
