"""L6.2: classify coding requests and render compact structured repo reports."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import re

from .canonical_ir_l63_experiment import CanonicalTask, canonicalize_request, format_canonical_task
from .repository_semantic_l58_experiment import RepositoryGraph, index_repository


class RequestType(Enum):
    SUMMARIZE = "summarize"
    EXPLAIN = "explain"
    INSPECT = "inspect"
    EXECUTE = "execute"
    TEST = "test"
    VERIFY = "verify"
    VERIFY_EXISTING = "verify_existing"
    DESIGN_EXPERIMENT = "design_experiment"
    DESIGN_AND_VERIFY = "design_and_verify"
    OPEN_RESEARCH_TASK = "open_research_task"
    SCIENTIFIC_RESEARCH = "scientific_research"
    MATH_DERIVATION = "math_derivation"
    DELETE_DIRECTORY = "delete_directory"
    MOVE_DIRECTORIES = "move_directories"
    EXPLAIN_AS_MATH = "explain_as_math"
    IMPLEMENT = "implement"
    MODIFY = "modify"


@dataclass(frozen=True)
class RequestIntent:
    request_type: RequestType
    target: str | None
    question: str


@dataclass(frozen=True)
class StructuredReport:
    intent: RequestIntent
    canonical_task: CanonicalTask
    target_file: str | None
    role: str
    key_symbols: tuple[str, ...]
    experiments: tuple[str, ...]
    call_paths: tuple[str, ...]
    dataflow: tuple[str, ...]
    evidence: tuple[str, ...]


def classify_request(text: str) -> RequestIntent:
    lower = text.lower()
    if any(word in text for word in ("移動させて", "移動して", "まとめて")) or any(word in lower for word in ("move", "relocate")):
        kind = RequestType.MOVE_DIRECTORIES
    elif any(word in text for word in ("削除して", "消して", "消去して")) or any(word in lower for word in ("delete", "remove")):
        kind = RequestType.DELETE_DIRECTORY
    elif any(word in text for word in ("論文", "研究", "文献", "量子情報", "時空が創発")) or any(
        word in lower for word in ("paper", "literature", "emergent spacetime", "quantum information")
    ):
        kind = RequestType.SCIENTIFIC_RESEARCH
    elif any(word in text for word in ("導出", "解を求め", "波動方程式", "偏微分方程式")) or any(
        word in lower for word in ("derive", "wave equation", "pde")
    ):
        kind = RequestType.MATH_DERIVATION
    elif (
        ".py" not in lower and "/" not in text and "\\" not in text
        and (any(word in text for word in ("荷電粒子", "電磁波", "放射", "加速度"))
             or any(word in lower for word in ("charged particle", "radiation", "electromagnetic")))
        and (any(word in text for word in ("検証", "作成実行", "シミュレーション", "結果を要約"))
             or any(word in lower for word in ("verify", "simulate", "validate")))
    ):
        kind = RequestType.OPEN_RESEARCH_TASK
    elif any(word in text for word in ("実装", "追加", "作成", "generate", "implement")):
        kind = RequestType.IMPLEMENT
    elif any(word in text for word in ("修正", "変更", "直して", "modify", "fix")):
        kind = RequestType.MODIFY
    elif "テスト" in text or re.search(r"\btests?\b", lower):
        kind = RequestType.TEST
    elif any(word in text for word in ("実行", "動か", "走らせ", "試して")) or any(word in lower for word in ("run", "execute")):
        kind = RequestType.EXECUTE
    elif (file_match := re.search(r"([A-Za-z0-9_.-]+\.py)", text)) is None and (
        any(word in text for word in ("荷電粒子", "電磁波", "放射", "加速度"))
        or any(word in lower for word in ("charged particle", "radiation", "electromagnetic"))
    ) and (
        any(word in text for word in ("検証", "作成実行", "シミュレーション", "結果を要約"))
        or any(word in lower for word in ("verify", "simulate", "validate"))
    ):
        kind = RequestType.OPEN_RESEARCH_TASK
    elif (file_match := re.search(r"([A-Za-z0-9_.-]+\.py)", text)) is None and (
        any(word in text for word in ("検証して", "確かめて", "正しいか", "シミュレーションして"))
        or any(word in lower for word in ("verify", "validate", "simulate"))
    ) and any(word in text for word in ("相対論", "相対論的", "電車", "列車", "車庫", "同時性", "同時刻")):
        kind = RequestType.DESIGN_AND_VERIFY
    elif any(word in text for word in ("検証", "verify")):
        kind = RequestType.VERIFY_EXISTING if file_match else RequestType.VERIFY
    elif (
        any(word in text for word in ("数式", "方程式", "式で", "latex", "数学"))
        or any(word in lower for word in ("equation", "formula", "mathematical"))
    ) and any(word in text for word in ("シミュレーション", "アルゴリズム", "モデル", "計算")):
        kind = RequestType.EXPLAIN_AS_MATH
    elif any(word in text for word in ("説明", "解説", "explain", "how")):
        kind = RequestType.EXPLAIN
    elif any(word in text for word in ("一覧", "構成", "inspect", "files")):
        kind = RequestType.INSPECT
    else:
        kind = RequestType.SUMMARIZE
    # Do not require a word boundary: Japanese often follows the filename
    # directly, e.g. ``model.pyを要約して``.
    file_match = re.search(r"([A-Za-z0-9_.-]+\.py)", text)
    return RequestIntent(kind, file_match.group(1) if file_match else None, text)


def build_report(root: Path, request: str) -> StructuredReport:
    intent = classify_request(request)
    canonical_task = canonicalize_request(request)
    graph = index_repository(root)
    target_symbols = tuple(symbol for symbol in graph.symbols
                            if intent.target and Path(symbol.file).name.lower() == intent.target.lower())
    if not target_symbols:
        target_symbols = graph.symbols[:0]
    names = {symbol.name for symbol in target_symbols}
    experiments = tuple(symbol.name for symbol in target_symbols if symbol.kind == "experiment")[:12]
    key_symbols = tuple(symbol.name for symbol in target_symbols if symbol.kind in {"function", "class", "experiment"})[:16]
    edges = tuple(edge for edge in graph.edges if edge.source in names or edge.target in names)
    call_paths = tuple(f"{edge.source} -> {edge.target}" for edge in edges if edge.relation == "calls")[:12]
    dataflow = tuple(f"{edge.source} -[{edge.relation}]-> {edge.target}" for edge in edges
                     if edge.relation in {"flows_to", "writes", "reads", "returns"})[:12]
    evidence = tuple(f"{symbol.file}:{symbol.line} ({symbol.name})" for symbol in target_symbols[:16])
    role = (f"{intent.target}内の関数・実験エントリポイントを解析するモジュール。"
            if intent.target else "質問に対応する対象ファイルを特定できませんでした。")
    return StructuredReport(intent, canonical_task, intent.target, role, key_symbols, experiments, call_paths, dataflow, evidence)


def format_report(report: StructuredReport) -> str:
    lines = ["## Structured Repository Report", "", f"【依頼】{report.intent.question}",
             f"【種別】{report.intent.request_type.value}", f"【対象】{report.target_file or '未特定'}",
             "", format_canonical_task(report.canonical_task), "", f"【役割】{report.role}"]
    if report.key_symbols:
        lines.extend(["", "【主要シンボル】", *[f"- `{item}`" for item in report.key_symbols]])
    if report.experiments:
        lines.extend(["", "【主要な実験】", *[f"- `{item}`" for item in report.experiments]])
    if report.call_paths:
        lines.extend(["", "【呼び出し関係】", *[f"- `{item}`" for item in report.call_paths]])
    if report.dataflow:
        lines.extend(["", "【データフロー】", *[f"- `{item}`" for item in report.dataflow]])
    lines.extend(["", "【根拠】"] + ([f"- `{item}`" for item in report.evidence] if report.evidence else ["- 対応するコード根拠なし"]))
    return "\n".join(lines)


def structured_repository_report(root: Path, request: str) -> str:
    return format_report(build_report(root, request))
