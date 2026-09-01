from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from idea_generate.run_pipeline import run
from idea_generate.src.claim_extractor import extract_claims
from idea_generate.src.compressor import compress_claims
from idea_generate.src.consciousness_reasoner import infer_next_experiments
from idea_generate.src.experiment_generator import generate_experiments
from idea_generate.src.keep_importer import import_keep_notes
from idea_generate.src.models import AtomicClaim, RawNote, ResearchGap


class PipelineTests(unittest.TestCase):
    def test_extractor_splits_a_note_into_atomic_claims(self) -> None:
        claims = extract_claims([RawNote("n1", "A hypothesis may improve. Test it.")])
        self.assertEqual(2, len(claims))
        self.assertEqual("hypothesis", claims[0].claim_type)
        self.assertEqual("experiment", claims[1].claim_type)

    def test_question_and_measurement_are_not_results(self) -> None:
        claims = extract_claims([RawNote("n1", "性能が改善するか？成功率を測定する。")])
        self.assertEqual(["question", "experiment"], [claim.claim_type for claim in claims])

    def test_dialogue_topics_enrich_claim_subject(self) -> None:
        claims = extract_claims([RawNote("n1", "Minecraftで操作を評価する。", tags=["fallback"])])
        self.assertEqual("game", claims[0].subject)
        self.assertEqual("inform", claims[0].relation)

    def test_keep_importer_filters_and_limits_notes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            (directory / "included.json").write_text(
                json.dumps({"title": "意識の仮説", "textContent": "意識モデルを実験する", "userEditedTimestampUsec": 2}, ensure_ascii=False),
                encoding="utf-8",
            )
            (directory / "ignored.json").write_text(
                json.dumps({"title": "買い物", "textContent": "牛乳", "userEditedTimestampUsec": 3}, ensure_ascii=False),
                encoding="utf-8",
            )
            notes = import_keep_notes(directory, limit=1)
            self.assertEqual(1, len(notes))
            self.assertIn("意識モデル", notes[0].text)

    def test_consciousness_model_selects_stronger_gap(self) -> None:
        gaps = [
            ResearchGap("G01", "weak", "weak hypothesis", None, 0.2, 0.2, 0.2, 0.2, 0.2, 0.8),
            ResearchGap("G02", "strong", "strong hypothesis", None, 0.9, 0.9, 0.9, 0.9, 0.9, 0.1),
        ]
        inference = infer_next_experiments(gaps, capacity=1)
        self.assertEqual("G02", inference.ranked_gap_ids[0])
        self.assertEqual("ACT", inference.action.kind)

    def test_generator_formulates_question_and_uses_theoretical_design(self) -> None:
        gap = ResearchGap("G01", "未検証", "スペクトル縮退とは何の物理的意味があるのか？", None, 0.8, 0.8, 0.8, 0.8, 0.8, 0.4)
        proposal = generate_experiments([gap])[0]
        self.assertEqual("theoretical_analysis", proposal.experiment_type)
        self.assertIn("対称性", proposal.target_hypothesis)
        self.assertIn("level splitting", proposal.metrics)
        self.assertTrue(proposal.falsification_condition)

    def test_theory_update_refines_theoretical_design(self) -> None:
        gap = ResearchGap("G01", "未検証", "スペクトル縮退とは何の物理的意味があるのか？", None, 0.8, 0.8, 0.8, 0.8, 0.8, 0.4)
        update = AtomicClaim("U01", "偶然の縮退と隠れた対称性を区別する必要がある。", "observation")
        proposal = generate_experiments([gap], context_claims=[update])[0]
        self.assertIn("偶然の縮退", proposal.expected_result)
        self.assertTrue(proposal.risks)

    def test_generator_uses_benchmark_design_for_minecraft(self) -> None:
        gap = ResearchGap("G01", "未検証", "マイクラで自力でダイヤ装備を作成できるか", None, 0.8, 0.8, 0.8, 0.8, 0.8, 0.4)
        proposal = generate_experiments([gap])[0]
        self.assertEqual("benchmark", proposal.experiment_type)
        self.assertIn("goal completion rate", proposal.metrics)
        self.assertIsNotNone(proposal.control_condition)

    def test_generator_leaves_controls_empty_for_market_search(self) -> None:
        gap = ResearchGap("G01", "未検証", "クラウドファンディングで開発費を賄えるか", None, 0.8, 0.8, 0.8, 0.8, 0.8, 0.4)
        proposal = generate_experiments([gap])[0]
        self.assertEqual("search_experiment", proposal.experiment_type)
        self.assertIsNone(proposal.intervention)
        self.assertIsNone(proposal.control_condition)

    def test_compression_groups_related_claims(self) -> None:
        claims = extract_claims([
            RawNote("n1", "Affordance representation improves transfer.", tags=["affordance"]),
            RawNote("n2", "Affordance representation may improve unknown transfer.", tags=["affordance"]),
        ])
        concepts = compress_claims(claims)
        self.assertEqual(1, len(concepts))
        self.assertEqual(2, concepts[0].support_count)

    def test_full_pipeline_generates_falsifiable_experiments(self) -> None:
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory)
            digest = run(root / "data" / "sample_notes.jsonl", output, workspace_size=3)
            experiments = json.loads((output / "experiments.json").read_text(encoding="utf-8"))
            self.assertTrue(experiments)
            self.assertLessEqual(len(experiments), 3)
            self.assertTrue(all(item["falsification_condition"] for item in experiments))
            self.assertTrue(all(item["rationale"] for item in experiments))
            self.assertIn("NEXT EXPERIMENTS", digest)
            self.assertTrue(json.loads((output / "research_graph.json").read_text(encoding="utf-8")))

    def test_markdown_update_changes_the_analyzed_claim_set(self) -> None:
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory)
            update = output / "update.md"
            update.write_text("新しい仮説は、隠れた対称性が縮退を保護することである。", encoding="utf-8")
            run(root / "data" / "sample_notes.jsonl", output, update_path=update)
            claims = json.loads((output / "claims.json").read_text(encoding="utf-8"))
            self.assertTrue(any("隠れた対称性" in claim["text"] for claim in claims))


if __name__ == "__main__":
    unittest.main()