"""Classify the lowest-cost discriminating investigation for a hypothesis."""

from __future__ import annotations


EXPERIMENT_TYPES = {
    "controlled_experiment", "simulation", "benchmark", "ablation",
    "theoretical_analysis", "literature_check", "observational_analysis", "search_experiment",
}


def classify_hypothesis(hypothesis: str) -> str:
    normalized = hypothesis.lower()
    if any(term in normalized for term in ("対称性", "物理的意味", "ハミルトニアン", "縮退")):
        return "theoretical_analysis"
    if any(term in normalized for term in ("minecraft", "マイクラ", "攻略", "装備", "ベンチマーク")):
        return "benchmark"
    if any(term in normalized for term in ("原因", "寄与", "機構", "ablation")):
        return "ablation"
    if any(term in normalized for term in ("実データ", "傾向", "観察", "相関")):
        return "observational_analysis"
    if any(term in normalized for term in ("文献", "論文", "先行研究")):
        return "literature_check"
    if any(term in normalized for term in ("探索", "検索", "市場", "クラウドファンディング")):
        return "search_experiment"
    if any(term in normalized for term in ("性能", "llm", "モデル", "比較", "できる")):
        return "controlled_experiment"
    return "simulation"