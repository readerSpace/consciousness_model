"""A small, portable Japanese dialogue component with no project dependencies."""

from __future__ import annotations

from dataclasses import dataclass
from math import log2
from typing import Iterable


TOPIC_KEYWORDS: dict[str, tuple[str, ...]] = {
    "food": ("食", "料理", "カレー", "ラーメン", "ご飯", "腹", "お腹"),
    "game": ("ゲーム", "勝負", "負け", "勝ち", "プレイ", "ランク", "Minecraft"),
    "weather": ("天気", "雨", "暑", "寒", "晴れ"),
    "music": ("歌", "音楽", "曲", "ライブ"),
    "work": ("仕事", "学校", "授業", "テスト", "課題"),
    "travel": ("旅行", "旅", "観光", "ホテル"),
}


@dataclass(frozen=True)
class SemanticFrame:
    """Inspectable result of analyzing one Japanese utterance."""

    text: str
    speaker_id: str
    speech_act: str
    topics: tuple[str, ...]
    events: tuple[str, ...]
    temporal_references: tuple[str, ...]
    discourse_references: tuple[str, ...]
    affect: tuple[str, ...]


@dataclass(frozen=True)
class DialogueReply:
    """Reply text plus the intermediate decisions used to produce it."""

    text: str
    response_act: str
    frame: SemanticFrame
    attended_frames: tuple[SemanticFrame, ...]
    memory_signal: float
    attention_signal: float


@dataclass(frozen=True)
class _WorkspaceItem:
    score: float
    sequence: int
    frame: SemanticFrame


class JapaneseDialogueEngine:
    """Analyze, select bounded context, infer a response act, and realize Japanese.

    This is intentionally a transparent baseline, not a general Japanese language
    model. It accepts text and optional caller-owned history/memories, so it can
    be embedded in a CLI, web service, game, or agent without database or LLM
    dependencies.
    """

    def __init__(self, workspace_capacity: int = 8) -> None:
        if not 1 <= workspace_capacity <= 64:
            raise ValueError("workspace_capacity must be between 1 and 64")
        self.workspace_capacity = workspace_capacity

    def analyze(self, text: str, speaker_id: str = "user") -> SemanticFrame:
        """Convert one Japanese utterance into a small symbolic semantic frame."""
        if not text or not text.strip():
            raise ValueError("text must not be empty")
        normalized = text.strip()
        topics = tuple(
            topic for topic, keywords in TOPIC_KEYWORDS.items()
            if any(keyword.lower() in normalized.lower() for keyword in keywords)
        )
        temporal = tuple(word for word in ("昨日", "今日", "明日", "さっき", "また") if word in normalized)
        discourse = tuple(word for word in ("それ", "あれ", "これ", "昨日も", "また") if word in normalized)
        affect = tuple(name for name, markers in {
            "joy": ("嬉", "楽しい", "最高"),
            "anger": ("怒", "むかつ"),
            "sadness": ("悲", "つら"),
            "surprise": ("!", "！", "やばい"),
            "difficulty": ("難しい", "負け", "無理"),
            "amusement": ("w", "笑"),
        }.items() if any(marker in normalized for marker in markers))
        events = [f"mentions:{topic}" for topic in topics]
        if "買" in normalized and "game" in topics:
            events.append("bought:game")
        if "負け" in normalized:
            events.append("lost:game")
        return SemanticFrame(
            text=normalized,
            speaker_id=speaker_id,
            speech_act=self._speech_act(normalized),
            topics=topics,
            events=tuple(events),
            temporal_references=temporal,
            discourse_references=discourse,
            affect=affect,
        )

    def reply(
        self,
        text: str,
        display_name: str = "あなた",
        speaker_id: str = "user",
        history: Iterable[str | SemanticFrame] = (),
        memories: Iterable[str] = (),
    ) -> DialogueReply:
        """Return a reply using only the capacity-limited attended context.

        ``history`` may contain strings or prior ``SemanticFrame`` values. The
        caller owns storage and decides which long-term memories to pass in.
        """
        current = self.analyze(text, speaker_id)
        prior = tuple(
            entry if isinstance(entry, SemanticFrame) else self.analyze(entry, "history")
            for entry in history
        )
        attended = self._attend((*prior, current), current)
        memory_signal, attention_signal = self._readout(attended)
        response_act = self._infer_response(current, attended, tuple(memories), attention_signal)
        return DialogueReply(
            text=self._realize(response_act, current, display_name, tuple(memories)),
            response_act=response_act,
            frame=current,
            attended_frames=attended,
            memory_signal=memory_signal,
            attention_signal=attention_signal,
        )

    def _attend(self, frames: tuple[SemanticFrame, ...], current: SemanticFrame) -> tuple[SemanticFrame, ...]:
        candidates = [
            _WorkspaceItem(self._salience(frame, frame == current, frames[:index]), index, frame)
            for index, frame in enumerate(frames)
        ]
        selected = sorted(candidates, key=lambda item: (-item.score, -item.sequence))[: self.workspace_capacity]
        return tuple(item.frame for item in selected)

    @staticmethod
    def _salience(frame: SemanticFrame, is_current: bool, history: tuple[SemanticFrame, ...]) -> float:
        known_topics = {topic for item in history for topic in item.topics}
        novelty = any(topic not in known_topics for topic in frame.topics)
        pragmatic = frame.speech_act in {"QUESTION", "REQUEST", "TEASE"}
        return (
            (0.75 if is_current else 0.0)
            + (0.18 if novelty else 0.0)
            + 0.18 * len(frame.events)
            + 0.10 * len(frame.topics)
            + (0.32 if pragmatic else 0.0)
            + 0.16 * (len(frame.temporal_references) + len(frame.discourse_references))
            + 0.20 * len(frame.affect)
        )

    @staticmethod
    def _readout(frames: tuple[SemanticFrame, ...]) -> tuple[float, float]:
        if not frames:
            return 0.0, 0.0
        event_density = sum(len(frame.events) for frame in frames) / len(frames)
        reference_density = sum(len(frame.temporal_references) + len(frame.discourse_references) for frame in frames) / len(frames)
        act_counts: dict[str, int] = {}
        for frame in frames:
            act_counts[frame.speech_act] = act_counts.get(frame.speech_act, 0) + 1
        probabilities = [count / len(frames) for count in act_counts.values()]
        entropy = -sum(probability * log2(probability) for probability in probabilities)
        memory_signal = min(1.0, (event_density + reference_density) / 4.0)
        attention_signal = min(1.0, (len(frames) / 8.0) + (entropy / 4.0))
        return round(memory_signal, 3), round(attention_signal, 3)

    @staticmethod
    def _infer_response(
        current: SemanticFrame,
        attended: tuple[SemanticFrame, ...],
        memories: tuple[str, ...],
        attention_signal: float,
    ) -> str:
        if current.speech_act == "GREETING":
            return "GREETING"
        if current.speech_act == "TEASE":
            return "COUNTER_TEASE"
        if memories and (current.discourse_references or current.topics):
            return "MEMORY_REFERENCE"
        if current.speech_act == "QUESTION" and attention_signal >= 0.125:
            return "REACTION"
        if current.speech_act == "AGREEMENT":
            return "AGREEMENT"
        if any(frame.speaker_id == current.speaker_id and frame.topics == current.topics for frame in attended[1:]):
            return "FOLLOW_UP"
        return "FOLLOW_UP"

    @staticmethod
    def _realize(response_act: str, frame: SemanticFrame, display_name: str, memories: tuple[str, ...]) -> str:
        if response_act == "GREETING":
            return f"こんにちは、{display_name}。今日は何の話をする？"
        if response_act == "COUNTER_TEASE":
            return "うるさい、次は勝つから見てろw"
        if response_act == "MEMORY_REFERENCE" and memories:
            return f"そういえば{memories[0]}って話してたよね。今回はどうだった？"[:120]
        if response_act == "REACTION":
            topic = {"weather": "天気と予定", "game": "ゲームのこと", "food": "料理のこと"}.get(
                frame.topics[0] if frame.topics else "", "そのこと"
            )
            return f"{display_name}、{topic}は気になるね。もう少し詳しく教えて？"
        if response_act == "AGREEMENT":
            return f"わかる、それはいいね。{display_name}はどう思う？"
        topic = {"food": "その料理", "game": "そのゲーム", "weather": "その天気"}.get(
            frame.topics[0] if frame.topics else "", "その話"
        )
        return f"なるほど、{display_name}。{topic}、そのあとどうなったの？"

    @staticmethod
    def _speech_act(text: str) -> str:
        if any(word in text for word in ("こんにちは", "こんばんは", "おはよう")):
            return "GREETING"
        if "？" in text or "?" in text or any(word in text for word in ("教えて", "何", "どう", "どこ", "いつ")):
            return "QUESTION"
        if any(word in text for word in ("して", "ください", "やって", "お願い")):
            return "REQUEST"
        if any(word in text for word in ("うるさい", "弱そう", "また負け", "w", "笑")):
            return "TEASE"
        if any(word in text for word in ("そう", "わかる", "同意", "いいね")):
            return "AGREEMENT"
        if any(word in text for word in ("！", "やばい", "すご", "いいな")):
            return "REACTION"
        return "INFORM"