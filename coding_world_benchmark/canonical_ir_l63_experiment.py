"""L6.3: normalize mixed-language coding requests into English canonical IR.

The IR is intentionally small and deterministic. It validates the design
choice that Japanese UI text can be translated into stable English enum names
and labels before repository analysis, without forcing the user-facing report
itself to become English.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import re


class CanonicalConcept(Enum):
    INITIAL_STATE = "initial_state"
    MEMORY_KERNEL_ESTIMATOR = "memory_kernel_estimator"
    CALL_GRAPH = "call_graph"
    DATA_FLOW = "data_flow"
    EVIDENCE_GRAPH = "evidence_graph"
    EXPERIMENT_PLAN = "experiment_plan"
    REPOSITORY = "repository"


class CanonicalRelation(Enum):
    CALLS = "calls"
    PRODUCES = "produces"
    DEPENDS_ON = "depends_on"
    FLOWS_TO = "flows_to"
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    DERIVED_FROM = "derived_from"
    MEASURED_BY = "measured_by"
    IMPLEMENTED_BY = "implemented_by"
    VERIFIED_BY = "verified_by"


_CONCEPT_LABELS = {
    CanonicalConcept.INITIAL_STATE: "initial state",
    CanonicalConcept.MEMORY_KERNEL_ESTIMATOR: "memory kernel estimator",
    CanonicalConcept.CALL_GRAPH: "call graph",
    CanonicalConcept.DATA_FLOW: "data flow",
    CanonicalConcept.EVIDENCE_GRAPH: "evidence graph",
    CanonicalConcept.EXPERIMENT_PLAN: "experiment plan",
    CanonicalConcept.REPOSITORY: "current repository",
}


_CONCEPT_PATTERNS = (
    (CanonicalConcept.INITIAL_STATE, ("初期状態", "初期条件", "initial state", "initial condition", "starting state")),
    (CanonicalConcept.MEMORY_KERNEL_ESTIMATOR, ("memory kernel estimator", "記憶カーネル", "メモリカーネル")),
    (CanonicalConcept.CALL_GRAPH, ("call graph", "呼び出し関係", "呼出関係", "呼び出しグラフ")),
    (CanonicalConcept.DATA_FLOW, ("data flow", "dataflow", "データフロー", "流れ")),
    (CanonicalConcept.EVIDENCE_GRAPH, ("evidence graph", "根拠グラフ", "証拠グラフ", "根拠")),
    (CanonicalConcept.EXPERIMENT_PLAN, ("experiment plan", "実験計画", "検証計画")),
    (CanonicalConcept.REPOSITORY, ("current repository", "repository", "repo", "リポジトリ", "このモデル")),
)


_RELATION_PATTERNS = (
    (CanonicalRelation.CALLS, ("calls", "call chain", "呼び出")),
    (CanonicalRelation.PRODUCES, ("produces", "生成", "作る", "出力")),
    (CanonicalRelation.DEPENDS_ON, ("depends on", "depends_on", "依存")),
    (CanonicalRelation.FLOWS_TO, ("flows to", "flows_to", "data flow", "dataflow", "流れる", "データフロー")),
    (CanonicalRelation.SUPPORTS, ("supports", "支持")),
    (CanonicalRelation.CONTRADICTS, ("contradicts", "矛盾")),
    (CanonicalRelation.DERIVED_FROM, ("derived from", "derived_from", "由来", "導出")),
    (CanonicalRelation.MEASURED_BY, ("measured by", "measured_by", "測定")),
    (CanonicalRelation.IMPLEMENTED_BY, ("implemented by", "implemented_by", "実装")),
    (CanonicalRelation.VERIFIED_BY, ("verified by", "verified_by", "検証", "テスト")),
)


_REQUEST_ACTIONS = {
    "summarize": "summarize",
    "explain": "explain",
    "inspect": "inspect",
    "execute": "execute",
    "test": "test",
    "verify": "verify",
    "verify_existing": "verify",
    "design_experiment": "design_experiment",
    "design_and_verify": "design_and_verify",
    "open_research_task": "open_research_task",
    "scientific_research": "scientific_research",
    "math_derivation": "math_derivation",
    "delete_directory": "delete_directory",
    "explain_as_math": "explain_as_math",
    "implement": "implement",
    "modify": "modify",
}


@dataclass(frozen=True)
class CanonicalTerm:
    concept: CanonicalConcept
    canonical_label: str
    mentions: tuple[str, ...]


@dataclass(frozen=True)
class CanonicalEdge:
    source: str
    relation: CanonicalRelation
    target: str


@dataclass(frozen=True)
class CanonicalTask:
    action: str
    target: str
    scope: str
    source_language: str
    concepts: tuple[CanonicalTerm, ...]
    relations: tuple[CanonicalRelation, ...]
    edges: tuple[CanonicalEdge, ...]
    original_text: str


@dataclass(frozen=True)
class _RequestIntent:
    request_type: str
    target: str | None


def _classify_request(text: str) -> _RequestIntent:
    lower = text.lower()
    if any(word in text for word in ("削除して", "消して", "消去して")) or any(word in lower for word in ("delete", "remove")):
        kind = "delete_directory"
    elif any(word in text for word in ("論文", "研究", "文献", "量子情報", "時空が創発")) or any(
        word in lower for word in ("paper", "literature", "emergent spacetime", "quantum information")
    ):
        kind = "scientific_research"
    elif any(word in text for word in ("導出", "解を求め", "波動方程式", "偏微分方程式")) or any(
        word in lower for word in ("derive", "wave equation", "pde")
    ):
        kind = "math_derivation"
    elif (
        ".py" not in lower and "/" not in text and "\\" not in text
        and (any(word in text for word in ("荷電粒子", "電磁波", "放射", "加速度"))
             or any(word in lower for word in ("charged particle", "radiation", "electromagnetic")))
        and (any(word in text for word in ("検証", "作成実行", "シミュレーション", "結果を要約"))
             or any(word in lower for word in ("verify", "simulate", "validate")))
    ):
        kind = "open_research_task"
    elif any(word in text for word in ("実装", "追加", "作成", "generate", "implement")):
        kind = "implement"
    elif any(word in text for word in ("修正", "変更", "直して", "modify", "fix")):
        kind = "modify"
    elif "テスト" in text or re.search(r"\btests?\b", lower):
        kind = "test"
    elif any(word in text for word in ("実行", "動か", "走らせ", "試して")) or any(word in lower for word in ("run", "execute")):
        kind = "execute"
    elif (file_match := re.search(r"([A-Za-z0-9_.-]+\.py)", text)) is None and (
        any(word in text for word in ("荷電粒子", "電磁波", "放射", "加速度"))
        or any(word in lower for word in ("charged particle", "radiation", "electromagnetic"))
    ) and (
        any(word in text for word in ("検証", "作成実行", "シミュレーション", "結果を要約"))
        or any(word in lower for word in ("verify", "simulate", "validate"))
    ):
        kind = "open_research_task"
    elif (file_match := re.search(r"([A-Za-z0-9_.-]+\.py)", text)) is None and (
        any(word in text for word in ("検証して", "確かめて", "正しいか", "シミュレーションして"))
        or any(word in lower for word in ("verify", "validate", "simulate"))
    ) and any(word in text for word in ("相対論", "相対論的", "電車", "列車", "車庫", "同時性", "同時刻")):
        kind = "design_and_verify"
    elif any(word in text for word in ("検証", "verify")):
        kind = "verify_existing" if file_match else "verify"
    elif (
        any(word in text for word in ("数式", "方程式", "式で", "latex", "数学"))
        or any(word in lower for word in ("equation", "formula", "mathematical"))
    ) and any(word in text for word in ("シミュレーション", "アルゴリズム", "モデル", "計算")):
        kind = "explain_as_math"
    elif any(word in text for word in ("説明", "解説")) or any(word in lower for word in ("explain", "how")):
        kind = "explain"
    elif any(word in text for word in ("一覧", "構成")) or any(word in lower for word in ("inspect", "files")):
        kind = "inspect"
    else:
        kind = "summarize"
    file_match = re.search(r"([A-Za-z0-9_.-]+\.py)", text)
    return _RequestIntent(kind, file_match.group(1) if file_match else None)


def _source_language(text: str) -> str:
    has_japanese = bool(re.search(r"[一-龥ぁ-んァ-ヶ]", text))
    has_latin = bool(re.search(r"[A-Za-z]", text))
    if has_japanese and has_latin:
        return "mixed"
    if has_japanese:
        return "ja"
    return "en"


def _find_patterns(text: str, patterns: tuple[tuple[Enum, tuple[str, ...]], ...]) -> dict[Enum, tuple[str, ...]]:
    lower = text.lower()
    found: dict[Enum, tuple[str, ...]] = {}
    for item, aliases in patterns:
        mentions = tuple(alias for alias in aliases if alias.lower() in lower)
        if mentions:
            found[item] = mentions
    return found


def canonicalize_request(text: str) -> CanonicalTask:
    intent = _classify_request(text)
    concepts_by_kind = _find_patterns(text, _CONCEPT_PATTERNS)
    if not concepts_by_kind:
        concepts_by_kind[CanonicalConcept.REPOSITORY] = ("current repository",)
    concepts = tuple(
        CanonicalTerm(concept, _CONCEPT_LABELS[concept], mentions)
        for concept, mentions in concepts_by_kind.items()
    )
    relation_hits = tuple(_find_patterns(text, _RELATION_PATTERNS).keys())
    target = intent.target or next((term.canonical_label for term in concepts), "current repository")
    if intent.target:
        target = Path(intent.target).name
    scope = "current repository" if CanonicalConcept.REPOSITORY in concepts_by_kind or not intent.target else "target file"
    edges = _infer_edges(intent.request_type, target, concepts, relation_hits)
    return CanonicalTask(
        action=_REQUEST_ACTIONS[intent.request_type],
        target=target,
        scope=scope,
        source_language=_source_language(text),
        concepts=concepts,
        relations=relation_hits,
        edges=edges,
        original_text=text,
    )


def _infer_edges(
    request_type: str,
    target: str,
    concepts: tuple[CanonicalTerm, ...],
    relation_hits: tuple[CanonicalRelation, ...],
) -> tuple[CanonicalEdge, ...]:
    concept_labels = {term.concept: term.canonical_label for term in concepts}
    edges: list[CanonicalEdge] = []
    if request_type in {"execute", "test", "verify", "verify_existing", "design_and_verify"}:
        edges.append(CanonicalEdge(target, CanonicalRelation.VERIFIED_BY, "experiment or test evidence"))
    if request_type in {"implement", "modify"}:
        edges.append(CanonicalEdge(target, CanonicalRelation.IMPLEMENTED_BY, "coding agent"))
    if CanonicalConcept.INITIAL_STATE in concept_labels:
        edges.append(CanonicalEdge("initial-state producer", CanonicalRelation.PRODUCES, concept_labels[CanonicalConcept.INITIAL_STATE]))
    if CanonicalConcept.DATA_FLOW in concept_labels:
        edges.append(CanonicalEdge(target, CanonicalRelation.FLOWS_TO, concept_labels[CanonicalConcept.DATA_FLOW]))
    if CanonicalConcept.CALL_GRAPH in concept_labels:
        edges.append(CanonicalEdge(target, CanonicalRelation.CALLS, concept_labels[CanonicalConcept.CALL_GRAPH]))
    for relation in relation_hits:
        if all(edge.relation is not relation for edge in edges):
            edges.append(CanonicalEdge(target, relation, "requested relation"))
    return tuple(edges)


def format_canonical_task(task: CanonicalTask) -> str:
    lines = [
        "【内部IR】",
        f"- action: `{task.action}`",
        f"- target: `{task.target}`",
        f"- scope: `{task.scope}`",
        f"- source_language: `{task.source_language}`",
    ]
    if task.concepts:
        lines.append("- concepts: " + ", ".join(
            f"{term.concept.name}=`{term.canonical_label}`" for term in task.concepts
        ))
    if task.relations:
        lines.append("- relations: " + ", ".join(f"`{relation.value}`" for relation in task.relations))
    if task.edges:
        lines.append("- edges:")
        lines.extend(
            f"  - `{edge.source}` -[{edge.relation.value}]-> `{edge.target}`"
            for edge in task.edges
        )
    return "\n".join(lines)
