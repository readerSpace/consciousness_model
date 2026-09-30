"""L7.7: convert local text/scanned PDFs into paper knowledge IR."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Protocol

from pypdf import PdfReader


class PDFMode(Enum):
    TEXT_PDF = "text_pdf"
    SCANNED_PDF = "scanned_pdf"


@dataclass(frozen=True)
class PaperSection:
    title: str
    text: str
    page: int
    confidence: float


@dataclass(frozen=True)
class EquationEvidence:
    latex: str
    page: int
    surrounding_text: str
    confidence: float
    parse_status: str


@dataclass(frozen=True)
class FigureEvidence:
    caption: str
    page: int
    detected_concepts: tuple[str, ...]
    referenced_in_sections: tuple[str, ...]


@dataclass(frozen=True)
class PaperKnowledge:
    source_file: str
    mode: PDFMode
    page_count: int
    sections: tuple[PaperSection, ...]
    equations: tuple[EquationEvidence, ...]
    figures: tuple[FigureEvidence, ...]
    unresolved_pages: tuple[int, ...]
    warnings: tuple[str, ...]


class OCRBackend(Protocol):
    def extract(self, image_path: Path, page: int) -> str: ...


def classify_pdf_text(page_texts: tuple[str, ...], threshold: int = 80) -> PDFMode:
    return PDFMode.TEXT_PDF if sum(len(text.strip()) for text in page_texts) >= threshold else PDFMode.SCANNED_PDF


def _extract_equations(text: str, page: int) -> tuple[EquationEvidence, ...]:
    blocks = re.findall(r"\$\$(.*?)\$\$|\\\[(.*?)\\\]|\\begin\{equation\*?\}(.*?)\\end\{equation\*?\}", text, re.S)
    equations: list[EquationEvidence] = []
    for block in blocks:
        latex = next((item.strip() for item in block if item and item.strip()), "")
        if latex:
            equations.append(EquationEvidence(latex, page, text[:320], 0.98, "text_extracted"))
    for line in text.splitlines():
        line = line.strip()
        if "=" in line and len(line) < 240 and any(symbol in line for symbol in ("∂", "gamma", "rho", "S_", "Area", "u(", "x(", "t")):
            equations.append(EquationEvidence(line, page, text[:320], 0.72, "heuristic_text_equation"))
    return tuple(equations)


def _extract_figures(text: str, page: int) -> tuple[FigureEvidence, ...]:
    figures = []
    for match in re.finditer(r"(?:Figure|Fig\.)\s*([0-9A-Za-z]+)\s*[:.-]?\s*(.+)", text, re.I):
        caption = match.group(0).strip()
        concepts = tuple(word for word in ("entanglement", "tensor", "geometry", "network", "scaling") if word in caption.lower())
        figures.append(FigureEvidence(caption, page, concepts, ()))
    return tuple(figures)


class TesseractOCR:
    def __init__(self, executable: str = "tesseract"):
        self.executable = executable

    def extract(self, image_path: Path, page: int) -> str:
        if shutil.which(self.executable) is None:
            return ""
        completed = subprocess.run([self.executable, str(image_path), "stdout", "-l", "eng"], capture_output=True, text=True, timeout=60)
        return completed.stdout if completed.returncode == 0 else ""


def _render_page(pdf_path: Path, page: int, output_dir: Path) -> Path | None:
    prefix = output_dir / f"page_{page:04d}"
    command = ["pdftoppm", "-f", str(page), "-l", str(page), "-png", "-r", "180", str(pdf_path), str(prefix)]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    images = sorted(output_dir.glob(f"{prefix.name}-*.png"))
    return images[0] if images else None


def extract_local_paper(pdf_path: Path, ocr: OCRBackend | None = None) -> PaperKnowledge:
    pdf_path = pdf_path.resolve()
    reader = PdfReader(str(pdf_path))
    page_texts = tuple(page.extract_text() or "" for page in reader.pages)
    mode = classify_pdf_text(page_texts)
    warnings: list[str] = []
    unresolved: list[int] = []
    if mode is PDFMode.SCANNED_PDF:
        ocr = ocr or TesseractOCR()
        with tempfile.TemporaryDirectory(prefix="paper-render-") as directory:
            output_dir = Path(directory)
            scanned_text: list[str] = []
            for page_number in range(1, len(reader.pages) + 1):
                image = _render_page(pdf_path, page_number, output_dir)
                text = ocr.extract(image, page_number) if image else ""
                scanned_text.append(text)
                if not text.strip():
                    unresolved.append(page_number)
            page_texts = tuple(scanned_text)
        if unresolved:
            warnings.append("OCR unavailable or produced no text for one or more scanned pages.")
    sections: list[PaperSection] = []
    equations: list[EquationEvidence] = []
    figures: list[FigureEvidence] = []
    for page, text in enumerate(page_texts, 1):
        if text.strip():
            title = next((line.strip() for line in text.splitlines() if line.strip()), f"page {page}")
            confidence = 0.96 if mode is PDFMode.TEXT_PDF else 0.68
            sections.append(PaperSection(title[:160], text[:12000], page, confidence))
            equations.extend(_extract_equations(text, page))
            figures.extend(_extract_figures(text, page))
    if mode is PDFMode.SCANNED_PDF and not ocr:
        warnings.append("Provide an OCRBackend or install Tesseract to recover scanned-page text.")
    return PaperKnowledge(str(pdf_path), mode, len(reader.pages), tuple(sections), tuple(equations), tuple(figures), tuple(unresolved), tuple(warnings))


def format_paper_knowledge(knowledge: PaperKnowledge) -> str:
    lines = ["## Local Paper Knowledge", "", f"【対象】`{knowledge.source_file}`", f"【モード】`{knowledge.mode.value}`", f"【ページ数】`{knowledge.page_count}`"]
    lines.extend(["", "【セクション】", *[f"- page {item.page}: {item.title} (confidence={item.confidence:.2f})" for item in knowledge.sections] or ["- 抽出なし"]])
    lines.append("\n【数式Evidence】")
    for equation in knowledge.equations:
        lines.extend([f"- page {equation.page} / `{equation.parse_status}` / confidence={equation.confidence:.2f}", "$$", equation.latex, "$$"])
    if not knowledge.equations:
        lines.append("- 抽出なし")
    lines.extend(["\n【図表Evidence】", *[f"- page {item.page}: {item.caption}" for item in knowledge.figures] or ["- 抽出なし"], "\n【未解決】", *[f"- scanned page {page}" for page in knowledge.unresolved_pages] or ["- なし"], "\n【注意】", *[f"- {warning}" for warning in knowledge.warnings] or ["- なし"]])
    return "\n".join(lines)
