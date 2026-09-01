"""Lightweight bilingual lexical similarity for reproducible Phase 1 clustering."""

from __future__ import annotations

import re

from .models import AtomicClaim


STOP_WORDS = {"が", "は", "を", "に", "の", "で", "と", "へ", "や", "する", "した", "ある", "the", "and", "for", "with"}


def tokens(text: str) -> set[str]:
    latin = re.findall(r"[a-zA-Z][a-zA-Z0-9_-]*", text.lower())
    japanese = re.findall(r"[一-龯ぁ-んァ-ンー]{2,}", text)
    return {token for token in latin + japanese if token not in STOP_WORDS}


def similarity(first: AtomicClaim, second: AtomicClaim) -> float:
    first_tokens, second_tokens = tokens(first.text), tokens(second.text)
    if not first_tokens or not second_tokens:
        return 0.0
    lexical = len(first_tokens & second_tokens) / len(first_tokens | second_tokens)
    subject_bonus = 0.2 if first.subject and first.subject == second.subject else 0.0
    return min(1.0, lexical + subject_bonus)


def matching_pairs(claims: list[AtomicClaim], threshold: float = 0.3) -> list[tuple[str, str]]:
    return [(first.id, second.id) for position, first in enumerate(claims)
            for second in claims[position + 1:] if similarity(first, second) >= threshold]