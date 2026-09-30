from coding_world_benchmark.local_paper_l77_experiment import (
    EquationEvidence,
    PDFMode,
    PaperKnowledge,
    PaperSection,
)
from coding_world_benchmark.paper_knowledge_quality_l78_experiment import (
    assess_paper_quality,
    compare_paper_knowledge,
    normalize_ocr_text,
    paper_to_scientific_evidence,
)


def _paper(equation: str, title: str = "Methods") -> PaperKnowledge:
    return PaperKnowledge(
        "paper.pdf",
        PDFMode.TEXT_PDF,
        1,
        (PaperSection(title, f"{title}\n{equation}", 1, 0.96),),
        (EquationEvidence(equation, 1, f"{title}\n{equation}", 0.98, "text_extracted"),),
        (),
        (),
        (),
    )


def test_ocr_normalization_keeps_scientific_content_and_cleans_noise():
    assert normalize_ocr_text("S_A = 1 − 2\n\n") == "S_A = 1 - 2"


def test_quality_report_validates_balanced_equation_and_promotes_evidence():
    knowledge = _paper(r"S_A = \frac{Area(\gamma_A)}{4G_N}")
    report = assess_paper_quality(knowledge)
    evidence = paper_to_scientific_evidence(knowledge)

    assert report.equation_recovery_accuracy == 1.0
    assert report.page_provenance_accuracy == 1.0
    assert evidence[0].source == "paper.pdf#page=1"
    assert evidence[0].equations == (r"S_A=\frac{Area(\gamma_A)}{4G_N}",)


def test_knowledge_equivalence_compares_semantics_not_raw_ocr_spacing():
    reference = _paper(r"S_A = \frac{Area(\gamma_A)}{4G_N}")
    scanned = _paper(r"S_A=\frac{Area(\gamma_A)}{4G_N}")
    equivalence = compare_paper_knowledge(reference, scanned)

    assert equivalence.equation_overlap == 1.0
    assert equivalence.equivalent
    assert not equivalence.missing_equations


def test_quality_marks_unbalanced_equation_for_review():
    report = assess_paper_quality(_paper(r"S_A = \frac{Area(\gamma_A)}{4G_N"))

    assert report.equations[0].status == "needs_review"
    assert report.equation_recovery_accuracy == 0.0
