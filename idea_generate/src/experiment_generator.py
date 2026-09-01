"""Generate falsifiable experiment templates from detected gaps."""

from __future__ import annotations

from coding_world_benchmark.portable_japanese_dialogue import JapaneseDialogueEngine

from .experiment_critic import review_experiment
from .experiment_type_classifier import classify_hypothesis
from .hypothesis_formulator import formulate_hypothesis
from .models import AtomicClaim, ExperimentProposal, ResearchGap


def _design(experiment_type: str, context: str = "") -> dict[str, object]:
    if experiment_type == "theoretical_analysis":
        design = {"procedure": ["縮退を持つ既知のハミルトニアンを3種類選ぶ。", "各系の対称性群を特定する。", "対称性を破る微小摂動を導入する。", "縮退準位の分裂を計算して比較する。"], "intervention": "対称性を破る微小摂動を導入する。", "control": "摂動を導入しない元のハミルトニアン。", "expected": "対称性に由来する縮退なら、対応する対称性を破ることで準位分裂が生じる。", "falsification": "対象とした縮退が対称性破れ後も維持され、別の保存量でも説明できない。", "metrics": ["degeneracy count", "level splitting", "symmetry representation dimension"], "resources": ["数式処理環境", "対象系の文献"]}
        if "偶然の縮退" in context or "隠れた対称性" in context:
            design["procedure"].insert(2, "偶然の縮退と隠れた対称性の候補を独立に分類する。")
            design["expected"] = "対称性保護型は対応する対称性破れで分裂し、偶然の縮退・隠れた対称性の候補は別に分類できる。"
            design["falsification"] = "分類後も対称性破れへの応答が系統的に区別できず、偶然の縮退または隠れた対称性としても説明できない。"
        return design
    if experiment_type == "benchmark":
        return {"procedure": ["初期条件と最終目標を固定した評価環境を作る。", "介入条件と対照条件を各20 trial実行する。", "同一seedごとの達成度を記録して比較する。"], "intervention": "ConsciousnessControllerによる状況更新・再計画を有効化する。", "control": "固定計画または再計画なしのエージェント。", "expected": "介入条件で目標達成率または中間目標達成率が高くなる。", "falsification": "介入条件が対照より改善せず、再計画の寄与も観測されない。", "metrics": ["goal completion rate", "time to completion", "subgoal completion rate", "replanning count", "failure recovery rate"], "resources": ["評価環境", "20以上のランダムseed", "実行ログ"]}
    if experiment_type == "search_experiment":
        return {"procedure": ["必要金額と対象顧客を事前登録する。", "説明資料を2種類作成し小規模に提示する。", "閲覧数、支援意向、転換率を測定する。", "必要転換率と比較する。"], "intervention": None, "control": None, "expected": "観測転換率が目標金額に必要な転換率以上になる。", "falsification": "信頼区間上限でも必要転換率を下回る。", "metrics": ["visitor-to-support conversion", "pledge intent", "cost per interested visitor"], "resources": ["説明資料", "募集ページ", "分析ツール"]}
    if experiment_type == "ablation":
        return {"procedure": ["対象機構を含む完全モデルを評価する。", "対象機構のみを除いたモデルを同一条件で評価する。", "複数seedの差を比較する。"], "intervention": "対象機構を有効化する。", "control": "対象機構のみを除いたアブレーション条件。", "expected": "完全モデルがアブレーション条件を事前登録した主要指標で上回る。", "falsification": "差が事前登録した最小効果量に達しない。", "metrics": ["primary benchmark score", "effect size", "confidence interval"], "resources": ["再現可能な実行環境", "複数random seed"]}
    if experiment_type == "observational_analysis":
        return {"procedure": ["観測単位と収集基準を定義する。", "データを収集して交絡候補を記録する。", "事前登録した分析で傾向を推定する。"], "intervention": None, "control": None, "expected": "事前定義した効果方向と大きさが観測される。", "falsification": "信頼区間が事前定義した最小効果を含まない。", "metrics": ["estimated effect", "confidence interval", "sample coverage"], "resources": ["データセット", "分析ノートブック"]}
    return {"procedure": ["対象と比較基準を事前登録する。", "同一入力条件で両条件を評価する。", "仮説に対応する指標を集計する。"], "intervention": "提案手法または対象条件を適用する。", "control": "既存の基準手法または対象条件を適用しない。", "expected": "提案条件が事前登録した主要指標で基準を上回る。", "falsification": "主要指標の差が事前登録した最小効果量に達しない。", "metrics": ["domain-specific primary score", "confidence interval", "inference cost"], "resources": ["評価データ", "再現可能な実行環境"]}


def generate_experiments(
    gaps: list[ResearchGap], dialogue: JapaneseDialogueEngine | None = None, context_claims: list[AtomicClaim] | None = None,
) -> list[ExperimentProposal]:
    engine = dialogue or JapaneseDialogueEngine()
    context = "\n".join(claim.text for claim in context_claims or [])
    proposals: list[ExperimentProposal] = []
    history: list[str] = []
    for index, gap in enumerate(gaps, start=1):
        hypothesis = formulate_hypothesis(gap.target_hypothesis)
        experiment_type = classify_hypothesis(hypothesis)
        design = _design(experiment_type, context)
        reply = engine.reply(
            hypothesis,
            display_name="研究者",
            speaker_id=gap.id,
            history=history,
            memories=(gap.description,),
        )
        proposal = ExperimentProposal(
            id=f"E{index:02d}", title=f"{experiment_type}: {hypothesis[:36]}", target_hypothesis=hypothesis,
            experiment_type=experiment_type, procedure=design["procedure"], intervention=design["intervention"],
            control_condition=design["control"], expected_result=design["expected"],
            falsification_condition=design["falsification"], metrics=design["metrics"],
            required_resources=design["resources"], estimated_cost=gap.estimated_cost,
            assumptions=["評価条件と成功基準を実行前に固定できる。"],
            risks=["偶然の縮退と隠れた対称性を区別する必要がある。"] if experiment_type == "theoretical_analysis" and ("偶然の縮退" in context or "隠れた対称性" in context) else [],
            rationale=reply.text,
        )
        proposal.risks.extend(review_experiment(proposal).problems)
        proposals.append(proposal)
        history.append(hypothesis)
    return proposals