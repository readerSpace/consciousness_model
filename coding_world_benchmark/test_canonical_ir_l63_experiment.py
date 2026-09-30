from pathlib import Path

from coding_world_benchmark.canonical_ir_l63_experiment import (
    CanonicalConcept,
    CanonicalRelation,
    canonicalize_request,
    format_canonical_task,
)
from coding_world_benchmark.structured_repository_report_l62_experiment import structured_repository_report


def test_japanese_and_english_initial_state_collapse_to_same_concept():
    japanese = canonicalize_request("このモデルの初期状態を要約して")
    english = canonicalize_request("summarize the initial condition in the current repository")

    assert japanese.action == "summarize"
    assert english.action == "summarize"
    assert japanese.source_language == "ja"
    assert english.source_language == "en"
    assert japanese.concepts[0].concept is CanonicalConcept.INITIAL_STATE
    assert english.concepts[0].concept is CanonicalConcept.INITIAL_STATE
    assert japanese.concepts[0].canonical_label == "initial state"
    assert english.concepts[0].canonical_label == "initial state"


def test_mixed_language_request_keeps_file_target_and_english_relations():
    task = canonicalize_request("model.pyの初期状態 data flow を検証して")

    assert task.action == "verify"
    assert task.target == "model.py"
    assert task.source_language == "mixed"
    assert CanonicalRelation.FLOWS_TO in task.relations
    assert any(edge.relation is CanonicalRelation.VERIFIED_BY for edge in task.edges)
    assert any(edge.relation is CanonicalRelation.PRODUCES for edge in task.edges)


def test_formatted_ir_uses_english_canonical_names_with_japanese_heading():
    rendered = format_canonical_task(canonicalize_request("このモデルの呼び出し関係と根拠を説明して"))

    assert "【内部IR】" in rendered
    assert "action: `explain`" in rendered
    assert "CALL_GRAPH=`call graph`" in rendered
    assert "EVIDENCE_GRAPH=`evidence graph`" in rendered


def test_structured_report_embeds_canonical_ir_without_losing_japanese_output(tmp_path: Path):
    (tmp_path / "model.py").write_text(
        "def make_initial_state():\n"
        "    rho = 0.45\n"
        "    return rho\n\n"
        "def evolve():\n"
        "    return make_initial_state()\n",
        encoding="utf-8",
    )

    report = structured_repository_report(tmp_path, "model.pyの初期状態を要約して")

    assert "【内部IR】" in report
    assert "INITIAL_STATE=`initial state`" in report
    assert "【役割】" in report
    assert "【根拠】" in report
