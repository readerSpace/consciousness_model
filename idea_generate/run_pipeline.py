"""Run the complete deterministic Phase 1 research workspace pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from idea_generate.src.claim_extractor import extract_claims
from idea_generate.src.compressor import compress_claims
from idea_generate.src.experiment_generator import generate_experiments
from idea_generate.src.experiment_ranker import rank_experiments
from idea_generate.src.gap_detector import detect_gaps
from idea_generate.src.note_parser import load_markdown_note, load_notes, write_json
from idea_generate.src.reporter import render_digest
from idea_generate.src.research_graph import build_graph, classify_state
from idea_generate.src.workspace import select_workspace


def run(
    input_path: Path,
    output_directory: Path,
    workspace_size: int = 8,
    update_path: Path | None = None,
) -> str:
    notes = load_notes(input_path)
    if update_path is not None:
        notes.append(load_markdown_note(update_path, note_id=f"update-{update_path.stem}"))
    claims = extract_claims(notes)
    concepts = compress_claims(claims)
    graph = build_graph(claims, concepts)
    state = classify_state(claims)
    gaps = detect_gaps(claims, concepts)
    proposals = rank_experiments(
        gaps, generate_experiments(gaps, context_claims=claims), workspace_capacity=workspace_size
    )
    selected = select_workspace(proposals, workspace_size)
    output_directory.mkdir(parents=True, exist_ok=True)
    write_json(output_directory / "claims.json", claims)
    write_json(output_directory / "concepts.json", concepts)
    write_json(output_directory / "gaps.json", gaps)
    write_json(output_directory / "experiments.json", selected)
    (output_directory / "research_graph.json").write_text(
        json.dumps({"nodes": graph.nodes, "edges": graph.edges}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    digest = render_digest(concepts, state, selected)
    (output_directory / "digest.txt").write_text(digest, encoding="utf-8")
    return digest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Path to a RawNote JSONL file.")
    parser.add_argument("--output", type=Path, default=Path("idea_generate/data/output"))
    parser.add_argument("--workspace-size", type=int, default=8)
    parser.add_argument("--update", type=Path, help="Optional Markdown note to add to this analysis run.")
    arguments = parser.parse_args()
    print(run(arguments.input, arguments.output, arguments.workspace_size, arguments.update))