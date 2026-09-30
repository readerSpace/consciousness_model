"""L5.9: grounded natural-language explanations from the repository graph."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import argparse

from .repository_semantic_l58_experiment import context_for, index_repository


class InferenceType(Enum):
    DIRECT = "direct"
    CALL_GRAPH_INFERENCE = "call_graph_inference"
    DATAFLOW_INFERENCE = "dataflow_inference"
    AMBIGUOUS = "ambiguous"
    RUNTIME_UNVERIFIED = "runtime_unverified"


@dataclass(frozen=True)
class ExplanationClaim:
    text: str
    evidence_ids: tuple[str, ...]
    confidence: float
    inference_type: InferenceType


@dataclass(frozen=True)
class GroundedExplanation:
    question: str
    claims: tuple[ExplanationClaim, ...]
    ambiguous: bool
    candidates: tuple[str, ...]


def _evidence_id(file: str, line: int, symbol: str) -> str:
    return f"{file}:{line}:{symbol}"


def explain(root: Path, question: str) -> GroundedExplanation:
    graph = index_repository(root)
    context = context_for(graph, question)
    claims: list[ExplanationClaim] = []
    candidates = tuple(item["target"] for item in context["interpretations"])
    if context["ambiguous"]:
        ids = tuple(_evidence_id(item["file"], item["line"], item["target"]) for item in context["interpretations"])
        claims.append(ExplanationClaim(
            f"この質問には複数の候補symbolがあります: {', '.join(candidates)}。",
            ids, 0.55, InferenceType.AMBIGUOUS,
        ))
    for item in context["interpretations"][:6]:
        claims.append(ExplanationClaim(
            f"{item['target']} は {item['file']}:{item['line']} で定義されています（静的解析上の根拠）。",
            (_evidence_id(item["file"], item["line"], item["target"]),), 0.9,
            InferenceType.DIRECT,
        ))
    for edge in context["call_graph"][:12]:
        claims.append(ExplanationClaim(
            f"{edge['source']} は {edge['target']} を呼び出します（実行時の分岐は未検証）。",
            (_evidence_id(edge["file"], edge["line"], edge["source"]),), 0.8,
            InferenceType.CALL_GRAPH_INFERENCE,
        ))
    for edge in context["dataflow"][:16]:
        if edge["relation"] == "flows_to":
            text = f"{edge['source']} から {edge['target']} へ値が流れる代入経路があります。"
        else:
            text = f"{edge['source']} は {edge['target']} を {edge['relation']} します。"
        claims.append(ExplanationClaim(
            text + "（静的解析上の経路で、実行時到達は未検証）。",
            (_evidence_id(edge["file"], edge["line"], edge["source"]),), 0.7,
            InferenceType.DATAFLOW_INFERENCE if edge["relation"] == "flows_to" else InferenceType.RUNTIME_UNVERIFIED,
        ))
    return GroundedExplanation(question, tuple(claims), bool(context["ambiguous"]), candidates)


def format_explanation(result: GroundedExplanation, *, max_claims: int | None = None) -> str:
    if not result.claims:
        return "Code Context Graphから質問に対応するsymbolを特定できませんでした。"
    lines = ["根拠付きRepository説明", "", f"質問: {result.question}"]
    if result.ambiguous:
        lines.extend(["", "曖昧性: 複数の解釈候補があります。対象実験を指定すると絞り込めます。"])
    lines.append("")
    claims = result.claims[:max_claims] if max_claims is not None else result.claims
    for index, claim in enumerate(claims, 1):
        evidence = ", ".join(claim.evidence_ids) or "なし"
        lines.append(f"{index}. {claim.text}")
        lines.append(f"   根拠: {evidence} | 種別: {claim.inference_type.value} | confidence={claim.confidence:.2f}")
    return "\n".join(lines)


def explain_repository_question(root: Path, question: str, *, max_claims: int | None = None) -> str:
    return format_explanation(explain(root, question), max_claims=max_claims)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a line-grounded repository explanation.")
    parser.add_argument("root", nargs="?", default=".", help="repository root")
    parser.add_argument("question", nargs="*", default=["initial state"], help="natural-language question")
    args = parser.parse_args()
    print(explain_repository_question(Path(args.root), " ".join(args.question)))


if __name__ == "__main__":
    main()
