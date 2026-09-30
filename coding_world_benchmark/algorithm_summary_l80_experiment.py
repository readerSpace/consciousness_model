"""L8.0: structure-based algorithm summaries for local modules."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from .repository_semantic_l58_experiment import CodeEdge, RepositoryGraph, Symbol, index_repository


@dataclass(frozen=True)
class AlgorithmStage:
    order: int
    label: str
    symbols: tuple[str, ...]
    evidence: tuple[str, ...]


@dataclass(frozen=True)
class AlgorithmSummary:
    target: str
    purpose: str
    symbols: tuple[str, ...]
    stages: tuple[AlgorithmStage, ...]
    states: tuple[str, ...]
    decisions: tuple[str, ...]
    dataflow: tuple[str, ...]
    evidence: tuple[str, ...]


def is_algorithm_summary_request(text: str) -> bool:
    lower = text.lower()
    has_algorithm = any(word in text for word in ("アルゴリズム", "処理の流れ", "処理手順")) or any(word in lower for word in ("algorithm", "workflow", "processing flow"))
    has_summary = any(word in text for word in ("要約", "まとめ", "概要")) or any(word in lower for word in ("summarize", "summary", "summarise"))
    return has_algorithm and has_summary


def _target_tokens(request: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(token.lower() for token in re.findall(r"[A-Za-z][A-Za-z0-9_]{2,}", request)))


def _target_symbols(graph: RepositoryGraph, request: str) -> tuple[Symbol, ...]:
    tokens = _target_tokens(request)
    scored: list[tuple[int, Symbol]] = []
    for symbol in graph.symbols:
        haystack = f"{symbol.name} {symbol.file}".lower()
        score = sum(token in haystack for token in tokens)
        if score:
            scored.append((score, symbol))
    if scored:
        best = max(score for score, _ in scored)
        selected = tuple(symbol for score, symbol in scored if score == best and score > 0)
        files = {symbol.file for symbol in selected}
        return tuple(symbol for symbol in graph.symbols if symbol.file in files)
    return ()


def _stage_label(name: str) -> str:
    lower = name.lower()
    if any(word in lower for word in ("init", "initialize", "setup", "create")):
        return "初期化"
    if any(word in lower for word in ("observe", "measure", "sense", "read")):
        return "観測"
    if any(word in lower for word in ("candidate", "generate", "propose", "discover")):
        return "候補生成"
    if any(word in lower for word in ("score", "rank", "select", "choose", "evaluate")):
        return "評価・選択"
    if any(word in lower for word in ("execute", "run", "apply", "act", "step")):
        return "実行・状態遷移"
    if any(word in lower for word in ("memory", "update", "learn", "store", "record")):
        return "記憶・更新"
    return "処理"


def _qualified_label(symbol: Symbol) -> str:
    return symbol.name.rsplit(".", 1)[-1]


class AlgorithmSummarizer:
    def summarize(self, graph: RepositoryGraph, request: str) -> AlgorithmSummary:
        selected = _target_symbols(graph, request)
        if not selected:
            return AlgorithmSummary(request, "対象モジュールを特定できませんでした。", (), (), (), (), (), ())
        selected_files = {symbol.file for symbol in selected}
        selected_names = {symbol.name for symbol in selected}
        functions = [symbol for symbol in selected if symbol.kind in {"function", "experiment"}]
        selected_edges = tuple(edge for edge in graph.edges if edge.file in selected_files and (edge.source in selected_names or edge.target in selected_names))
        call_targets = {edge.target for edge in selected_edges if edge.relation == "calls"}
        entrypoints = [symbol for symbol in functions if symbol.name in {edge.source for edge in selected_edges if edge.relation == "calls"} or any(word in symbol.name.lower() for word in ("main", "run", "step", "process"))]
        if not entrypoints:
            entrypoints = functions[:]
        ordered = list(dict.fromkeys(entrypoints + [symbol for symbol in functions if symbol.name in call_targets]))

        grouped: dict[str, list[Symbol]] = {}
        for symbol in ordered:
            grouped.setdefault(_stage_label(_qualified_label(symbol)), []).append(symbol)
        stages = tuple(
            AlgorithmStage(index, label, tuple(symbol.name for symbol in symbols), tuple(f"{symbol.file}:{symbol.line}" for symbol in symbols))
            for index, (label, symbols) in enumerate(grouped.items(), 1)
        )
        states = tuple(dict.fromkeys(edge.target for edge in selected_edges if edge.relation == "writes"))
        decisions = tuple(symbol.name for symbol in functions if any(word in symbol.name.lower() for word in ("select", "choose", "score", "evaluate", "filter")))
        dataflow = tuple(f"{edge.source} -[{edge.relation}]-> {edge.target}" for edge in selected_edges if edge.relation in {"flows_to", "writes", "returns"})[:24]
        evidence = tuple(dict.fromkeys(f"{edge.file}:{edge.line} {edge.source} -[{edge.relation}]-> {edge.target}" for edge in selected_edges))
        symbol_names = tuple(symbol.name for symbol in selected if symbol.kind in {"class", "function", "experiment"})[:24]
        target = str(Path(selected[0].file).with_suffix(""))
        purpose = f"{target} の中心処理を、呼び出し関係と状態更新の順序として要約します。"
        return AlgorithmSummary(target, purpose, symbol_names, stages, states, decisions, dataflow, evidence[:32])


def summarize_algorithm(root: Path, request: str) -> AlgorithmSummary:
    return AlgorithmSummarizer().summarize(index_repository(root), request)


def format_algorithm_summary(summary: AlgorithmSummary) -> str:
    lines = ["## Algorithm Summary", "", "【対象】", f"`{summary.target}`", "", "【目的】", summary.purpose, "", "【主要構成】"]
    lines.extend(f"- `{symbol}`" for symbol in summary.symbols) if summary.symbols else lines.append("- 対象symbolを特定できませんでした")
    lines.extend(["", "【アルゴリズム】"])
    if summary.stages:
        for stage in summary.stages:
            lines.append(f"{stage.order}. {stage.label}: " + ", ".join(f"`{symbol}`" for symbol in stage.symbols))
    else:
        lines.append("- 処理段階を抽出できませんでした")
    lines.extend(["", "【主要state】"])
    lines.extend(f"- `{state}`" for state in summary.states) if summary.states else lines.append("- なし")
    lines.extend(["", "【判断規則】"])
    lines.extend(f"- `{decision}`" for decision in summary.decisions) if summary.decisions else lines.append("- 明示的な選択関数なし")
    lines.extend(["", "【Data Flow】"])
    lines.extend(f"- `{flow}`" for flow in summary.dataflow) if summary.dataflow else lines.append("- なし")
    lines.extend(["", "【コード根拠】"])
    lines.extend(f"- `{item}`" for item in summary.evidence) if summary.evidence else lines.append("- なし")
    return "\n".join(lines)
