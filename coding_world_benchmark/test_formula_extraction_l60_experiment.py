from pathlib import Path

from coding_world_benchmark.formula_extraction_l60_experiment import extract_formulas, formula_evidence


def test_extracts_python_formula_with_latex_and_symbols(tmp_path: Path):
    (tmp_path / "model.py").write_text(
        "rho_total = rho_birth + rho_matter + rho_late\n"
        "kinetic = v**2 / (2*m)\n", encoding="utf-8"
    )
    formulas = extract_formulas(tmp_path)
    total = next(item for item in formulas if item.raw_text.startswith("rho_total"))
    assert total.source_file == "model.py"
    assert total.source_line == 1
    assert {"rho_birth", "rho_matter", "rho_late"}.issubset(total.symbols)
    assert total.latex.startswith("rho_{birth}") or "rho" in total.latex
    assert formula_evidence(total)["source"] == "model.py:1"


def test_extracts_tex_and_markdown_blocks_with_line_evidence(tmp_path: Path):
    (tmp_path / "notes.md").write_text("# Model\n\n$$\\rho_{total}=\\rho_b+\\rho_m$$\n", encoding="utf-8")
    (tmp_path / "derivation.tex").write_text("text\n\\begin{equation}\nE=mc^2\n\\end{equation}\n", encoding="utf-8")
    formulas = extract_formulas(tmp_path)
    assert len(formulas) == 2
    assert {item.role for item in formulas} == {"documentation"}
    assert any("rho" in item.raw_text for item in formulas)
    assert any(item.source_file == "derivation.tex" and item.source_line == 2 for item in formulas)


def test_ignores_build_directories(tmp_path: Path):
    ignored = tmp_path / "__pycache__"
    ignored.mkdir()
    (ignored / "generated.py").write_text("x = a + b\n", encoding="utf-8")
    assert extract_formulas(tmp_path) == ()
