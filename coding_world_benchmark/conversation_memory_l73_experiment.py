"""L7.3: compressed, reusable conversation knowledge memory."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import gzip
import json
from pathlib import Path
import re


@dataclass(frozen=True)
class MemoryFact:
    key: str
    value: str
    source: str
    uses: int = 1


def _compact_lines(text: str) -> list[str]:
    lines = []
    for raw in text.splitlines():
        line = re.sub(r"\s+", " ", raw).strip()
        if not line or line.startswith("##") or line.startswith("###"):
            continue
        if any(marker in line for marker in ("$$", "source:", "confidence", "conclusion", "結論", "判定", "equation", "status:", "entanglement", "distance", "geometry")):
            lines.append(line[:600])
    return lines


class ConversationKnowledgeMemory:
    def __init__(self, path: Path, max_facts: int = 240):
        self.path = path
        self.max_facts = max_facts
        self.facts: list[MemoryFact] = []
        self.load()

    def load(self) -> None:
        if not self.path.is_file():
            return
        try:
            with gzip.open(self.path, "rt", encoding="utf-8") as stream:
                payload = json.load(stream)
            self.facts = [MemoryFact(**item) for item in payload.get("facts", []) if isinstance(item, dict)]
        except (OSError, EOFError, json.JSONDecodeError, TypeError, KeyError):
            self.facts = []

    def learn(self, user_text: str, assistant_text: str) -> None:
        source = self._topic_key(user_text)
        for value in _compact_lines(assistant_text):
            key = self._fact_key(source, value)
            existing = next((fact for fact in self.facts if fact.key == key), None)
            if existing:
                index = self.facts.index(existing)
                self.facts[index] = MemoryFact(existing.key, existing.value, existing.source, existing.uses + 1)
            else:
                self.facts.append(MemoryFact(key, value, source))
        self.facts = self.facts[-self.max_facts:]

    def recall(self, query: str, limit: int = 6, policy: str = "all") -> tuple[MemoryFact, ...]:
        tokens = {token.lower() for token in re.findall(r"[a-z][a-z0-9_-]{2,}|[一-龥ぁ-んァ-ヶ]{2,}", query)}
        scored = []
        for fact in self.facts:
            haystack = f"{fact.source} {fact.value}".lower()
            if policy in {"repair_related_only", "task_contract_only"} and any(word in haystack for word in ("gauge", "su2", "su3", "wilson", "quantum", "entanglement", "時空", "量子")):
                continue
            if policy == "repair_related_only" and not any(word in haystack for word in ("repair", "patch", "router", "fallback", "修正", "変更", "taskspec", "expectedbehavior", "observedbehavior", "behavior monitor")):
                continue
            if policy == "task_contract_only" and not any(word in haystack for word in ("taskspec", "task spec", "expectedbehavior", "observedbehavior", "behavior monitor", "contract", "契約", "不一致", "file_modification")):
                continue
            if policy == "scientific_modeling_only":
                if any(word in haystack for word in ("gauge", "su2", "su3", "wilson", "router", "patch", "fallback")):
                    continue
                if not any(word in haystack for word in ("lorentz", "magnetic", "particle", "ode", "rk4", "simulation", "equation", "ローレンツ", "磁場", "粒子", "方程式")):
                    continue
            if policy == "path_resolution_only" and not any(word in f"{fact.source} {fact.value}".lower() for word in ("path", "directory", "folder", "フォルダ", "移動", "削除")):
                continue
            score = sum(token in haystack for token in tokens) + min(fact.uses, 3) * 0.1
            if policy == "task_contract_only":
                score += 3.0
            elif policy == "scientific_modeling_only":
                score += 2.0
            elif policy == "repair_related_only":
                score += 1.0
            if score > 0:
                scored.append((score, fact))
        scored.sort(key=lambda item: (-item[0], item[1].key))
        return tuple(fact for _, fact in scored[:limit])

    def context(self, query: str, limit: int = 6, policy: str = "all") -> str:
        return "\n".join(f"- {fact.value} (memory:{fact.source}, uses:{fact.uses})" for fact in self.recall(query, limit, policy))

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": 1, "facts": [asdict(fact) for fact in self.facts]}
        with gzip.open(self.path, "wt", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False)

    @staticmethod
    def _topic_key(text: str) -> str:
        lower = text.lower()
        if any(word in text for word in ("量子", "時空", "entanglement", "quantum")):
            return "quantum_geometry"
        if any(word in text for word in ("放射", "荷電粒子", "radiation", "particle")):
            return "particle_radiation"
        if any(word in text for word in ("波動方程式", "wave equation", "pde")):
            return "wave_equation"
        return re.sub(r"[^a-z0-9一-龥ぁ-んァ-ヶ]+", "_", lower)[:80] or "general"

    @staticmethod
    def _fact_key(source: str, value: str) -> str:
        normalized = re.sub(r"\W+", " ", value.lower()).strip()
        return f"{source}:{normalized[:160]}"
