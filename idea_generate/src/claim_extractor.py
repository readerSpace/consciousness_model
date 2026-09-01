"""A transparent baseline claim extractor; replaceable by a schema-bound LLM."""

from __future__ import annotations

import re

from coding_world_benchmark.portable_japanese_dialogue import JapaneseDialogueEngine, SemanticFrame

from .models import AtomicClaim, RawNote


TYPE_PATTERNS = (
    ("question", r"[?？]|未検証|不明|疑問|どう|か$"),
    ("experiment", r"実験|比較|評価|測定|テスト|試す|\btest\b|\bmeasure\b|\bcompare\b"),
    ("result", r"結果|成功|失敗|改善|低下|できた|確認"),
    ("goal", r"したい|目標|目指す"),
    ("hypothesis", r"かもしれない|仮説|予測|なら|有効|改善する|\bmay\b|\bmight\b"),
)


def _claim_type(sentence: str, frame: SemanticFrame) -> str:
    if frame.speech_act == "QUESTION":
        return "question"
    if frame.speech_act == "REQUEST":
        return "experiment"
    for claim_type, pattern in TYPE_PATTERNS:
        if re.search(pattern, sentence, re.IGNORECASE):
            return claim_type
    return "observation"


def _confidence(claim_type: str) -> float:
    return {"result": 0.8, "observation": 0.7, "hypothesis": 0.55,
            "experiment": 0.5, "question": 0.35, "goal": 0.45}.get(claim_type, 0.5)


def extract_claims(notes: list[RawNote], dialogue: JapaneseDialogueEngine | None = None) -> list[AtomicClaim]:
    engine = dialogue or JapaneseDialogueEngine()
    claims: list[AtomicClaim] = []
    for note in notes:
        sentences = [part.strip() for part in re.split(r"[。.!！?？\n]+", note.text) if part.strip()]
        for index, sentence in enumerate(sentences, start=1):
            frame = engine.analyze(sentence, speaker_id=note.id)
            claim_type = _claim_type(sentence, frame)
            claims.append(AtomicClaim(
                id=f"{note.id}-c{index}", text=sentence, claim_type=claim_type,
                subject=frame.topics[0] if frame.topics else next(iter(note.tags), None),
                relation=frame.speech_act.lower(), confidence=_confidence(claim_type),
                source_note_ids=[note.id],
            ))
    return claims