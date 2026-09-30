from pathlib import Path

from coding_world_benchmark.local_paper_l77_experiment import (
    PDFMode,
    classify_pdf_text,
    extract_local_paper,
    format_paper_knowledge,
)


class FakeOCR:
    def extract(self, image_path: Path, page: int) -> str:
        return "Figure 1: entanglement geometry\n$$ S_A = Area(gamma_A) / (4 G_N) $$"


def test_pdf_mode_detector_distinguishes_text_and_scanned_pages():
    assert classify_pdf_text(("A" * 100,)) is PDFMode.TEXT_PDF
    assert classify_pdf_text(("", "")) is PDFMode.SCANNED_PDF


def test_text_pdf_extraction_produces_sections_equations_and_figures(tmp_path, monkeypatch):
    class FakePage:
        def extract_text(self):
            return "Introduction\n$$ S_A = Area(gamma_A) / (4 G_N) $$\nFigure 1: tensor network geometry"

    class FakeReader:
        pages = [FakePage()]

    monkeypatch.setattr("coding_world_benchmark.local_paper_l77_experiment.PdfReader", lambda path: FakeReader())
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"placeholder")
    knowledge = extract_local_paper(pdf)

    assert knowledge.mode is PDFMode.TEXT_PDF
    assert knowledge.sections
    assert knowledge.equations
    assert knowledge.figures
    assert "Local Paper Knowledge" in format_paper_knowledge(knowledge)


def test_scanned_pdf_uses_injected_ocr_and_preserves_page_provenance(tmp_path, monkeypatch):
    class FakePage:
        def extract_text(self):
            return ""

    class FakeReader:
        pages = [FakePage()]

    monkeypatch.setattr("coding_world_benchmark.local_paper_l77_experiment.PdfReader", lambda path: FakeReader())
    monkeypatch.setattr("coding_world_benchmark.local_paper_l77_experiment._render_page", lambda pdf, page, output: output / "page-1.png")
    pdf = tmp_path / "scanned.pdf"
    pdf.write_bytes(b"placeholder")
    knowledge = extract_local_paper(pdf, FakeOCR())

    assert knowledge.mode is PDFMode.SCANNED_PDF
    assert knowledge.equations[0].page == 1
    assert knowledge.figures[0].detected_concepts == ("entanglement", "geometry")
    assert not knowledge.unresolved_pages
