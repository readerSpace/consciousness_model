"""L7.8/L7.9: validate local paper extraction and integrate it as evidence."""
from __future__ import annotations

from dataclasses import dataclass
import re
from pathlib import Path

from .local_paper_l77_experiment import EquationEvidence, PaperKnowledge
from .scientific_knowledge_l70_experiment import ScientificEvidence


@dataclass(frozen=True)
class EquationQuality:
    page: int
    latex: str
    normalized_latex: str
    balanced_delimiters: bool
    symbol_consistent: bool
    confidence: float
    status: str


@dataclass(frozen=True)
class PaperQualityReport:
    source_file: str
    text_recovery_accuracy: float
    equation_recovery_accuracy: float
    symbol_accuracy: float
    section_structure_accuracy: float
    figure_caption_accuracy: float
    page_provenance_accuracy: float
    unresolved_page_rate: float
    equations: tuple[EquationQuality, ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class KnowledgeEquivalence:
    equation_overlap: float
    section_overlap: float
    figure_overlap: float
    overall: float
    equivalent: bool
    missing_equations: tuple[str, ...]


_OCR_REPLACEMENTS = str.maketrans({
    "ﬁ": "fi",
    "ﬂ": "fl",
    "“": '"',
    "”": '"',
    "−": "-",
    "–": "-",
})


def normalize_ocr_text(text: str) -> str:
    """Normalize layout noise without silently changing scientific symbols."""
    text = text.translate(_OCR_REPLACEMENTS)
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def normalize_latex(latex: str) -> str:
    value = normalize_ocr_text(latex)
    value = re.sub(r"\\left|\\right", "", value)
    value = re.sub(r"\s+", " ", value).strip()
    return re.sub(r"\s*([=,])\s*", r"\1", value)


def _balanced(value: str) -> bool:
    pairs = {"{": "}", "(": ")", "[": "]"}
    stack: list[str] = []
    for char in value:
        if char in pairs:
            stack.append(pairs[char])
        elif char in pairs.values():
            if not stack or stack.pop() != char:
                return False
    return not stack


def _symbols(latex: str) -> set[str]:
    commands = set(re.findall(r"\\([A-Za-z]+)", latex))
    identifiers = set(re.findall(r"(?<![A-Za-z])[A-Za-z](?![A-Za-z])", latex))
    return commands | identifiers


def _equation_quality(equation: EquationEvidence) -> EquationQuality:
    normalized = normalize_latex(equation.latex)
    balanced = _balanced(normalized)
    symbols = _symbols(normalized)
    context = normalize_ocr_text(equation.surrounding_text)
    context_symbols = _symbols(context)
    symbol_consistent = not symbols or not context_symbols or bool(symbols & context_symbols)
    checks = sum((balanced, symbol_consistent))
    confidence = round(min(0.99, equation.confidence * (0.75 + 0.125 * checks)), 3)
    status = "validated" if balanced and symbol_consistent else "needs_review"
    return EquationQuality(equation.page, equation.latex, normalized, balanced, symbol_consistent, confidence, status)


def assess_paper_quality(knowledge: PaperKnowledge) -> PaperQualityReport:
    equations = tuple(_equation_quality(item) for item in knowledge.equations)
    page_count = max(knowledge.page_count, 1)
    pages_with_text = {item.page for item in knowledge.sections}
    unresolved_rate = len(knowledge.unresolved_pages) / page_count
    text_recovery = len(pages_with_text) / page_count
    equation_accuracy = sum(item.status == "validated" for item in equations) / max(len(equations), 1)
    symbol_accuracy = sum(item.symbol_consistent for item in equations) / max(len(equations), 1)
    structure = len(knowledge.sections) / page_count
    structure = min(1.0, structure)
    figure_quality = 1.0 if knowledge.figures else 0.0
    provenance = 1.0 if all(item.page > 0 and item.page <= page_count for item in knowledge.sections + knowledge.equations + knowledge.figures) else 0.0
    warnings = list(knowledge.warnings)
    if knowledge.mode.value == "scanned_pdf" and not knowledge.sections:
        warnings.append("No OCR text was recovered from the scanned paper.")
    return PaperQualityReport(
        knowledge.source_file,
        round(text_recovery, 3),
        round(equation_accuracy, 3),
        round(symbol_accuracy, 3),
        round(structure, 3),
        round(figure_quality, 3),
        round(provenance, 3),
        round(unresolved_rate, 3),
        equations,
        tuple(dict.fromkeys(warnings)),
    )


def paper_to_scientific_evidence(knowledge: PaperKnowledge) -> tuple[ScientificEvidence, ...]:
    """Promote only page-grounded equations/claims into the common evidence IR."""
    quality = assess_paper_quality(knowledge)
    source = Path(knowledge.source_file).name
    evidence: list[ScientificEvidence] = []
    for equation in quality.equations:
        context = next((item.surrounding_text for item in knowledge.equations if item.page == equation.page and normalize_latex(item.latex) == equation.normalized_latex), "")
        claim = next((line for line in normalize_ocr_text(context).splitlines() if line and "=" not in line), f"Equation extracted from page {equation.page}.")
        assumptions = (f"local PDF page {equation.page}", f"extraction status: {equation.status}")
        evidence.append(ScientificEvidence(claim, "local_pdf", f"{source}#page={equation.page}", (equation.normalized_latex,), assumptions, equation.confidence))
    return tuple(evidence)


def compare_paper_knowledge(reference: PaperKnowledge, candidate: PaperKnowledge, threshold: float = 0.8) -> KnowledgeEquivalence:
    reference_equations = {normalize_latex(item.latex) for item in reference.equations}
    candidate_equations = {normalize_latex(item.latex) for item in candidate.equations}
    reference_sections = {normalize_ocr_text(item.title).lower() for item in reference.sections}
    candidate_sections = {normalize_ocr_text(item.title).lower() for item in candidate.sections}
    reference_figures = {normalize_ocr_text(item.caption).lower() for item in reference.figures}
    candidate_figures = {normalize_ocr_text(item.caption).lower() for item in candidate.figures}

    def overlap(left: set[str], right: set[str]) -> float:
        return len(left & right) / max(len(left), 1)

    equation_overlap = overlap(reference_equations, candidate_equations)
    section_overlap = overlap(reference_sections, candidate_sections)
    figure_overlap = overlap(reference_figures, candidate_figures) if reference_figures else 1.0
    overall = round(0.6 * equation_overlap + 0.25 * section_overlap + 0.15 * figure_overlap, 3)
    return KnowledgeEquivalence(round(equation_overlap, 3), round(section_overlap, 3), round(figure_overlap, 3), overall, overall >= threshold, tuple(sorted(reference_equations - candidate_equations)))


def format_paper_quality_report(report: PaperQualityReport) -> str:
    lines = ["## Paper OCR Quality", "", f"【対象】`{report.source_file}`", "【品質指標】"]
    metrics = (
        ("text_recovery_accuracy", report.text_recovery_accuracy),
        ("equation_recovery_accuracy", report.equation_recovery_accuracy),
        ("symbol_accuracy", report.symbol_accuracy),
        ("section_structure_accuracy", report.section_structure_accuracy),
        ("figure_caption_accuracy", report.figure_caption_accuracy),
        ("page_provenance_accuracy", report.page_provenance_accuracy),
        ("unresolved_page_rate", report.unresolved_page_rate),
    )
    lines.extend(f"- `{name}` = `{value:.3f}`" for name, value in metrics)
    lines.extend(["", "【数式検査】"])
    lines.extend(f"- page {item.page}: `{item.status}` confidence={item.confidence:.3f} / `{item.normalized_latex}`" for item in report.equations)
    if not report.equations:
        lines.append("- 抽出なし")
    warnings = [f"- {warning}" for warning in report.warnings] or ["- なし"]
    lines.extend(["", "【警告】", *warnings])
    return "\n".join(lines)


def format_integrated_evidence(evidence: tuple[ScientificEvidence, ...]) -> str:
    lines = ["## Integrated Scientific Evidence", ""]
    if not evidence:
        return "\n".join(lines + ["- 抽出できる根拠はありません。"])
    for item in evidence:
        lines.extend([
            f"- {item.claim}",
            f"  - source: `{item.source}`",
            f"  - confidence: `{item.confidence:.3f}`",
        ])
        for equation in item.equations:
            lines.extend(["  - equation:", "    $$", f"    {equation}", "    $$"])
    return "\n".join(lines)
