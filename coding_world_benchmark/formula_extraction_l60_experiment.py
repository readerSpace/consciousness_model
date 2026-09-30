"""L6.0: extract formulas from source files with line-grounded evidence.

This is intentionally an extraction boundary, not a theorem prover. Python
assignments are parsed from the AST and normalized with SymPy when available;
Markdown/TeX formulas are retained as source-grounded LaTeX nodes.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
from typing import Any

try:
    import sympy as sp
except ImportError:  # pragma: no cover - the fallback keeps extraction usable
    sp = None


@dataclass(frozen=True)
class FormulaNode:
    formula_id: str
    raw_text: str
    latex: str
    normalized_ast: str
    symbols: tuple[str, ...]
    source_file: str
    source_line: int
    role: str


def _formula_id(source_file: str, line: int, raw_text: str) -> str:
    digest = hashlib.sha1(f"{source_file}:{line}:{raw_text}".encode()).hexdigest()[:12]
    return f"formula-{digest}"


def _python_formula(raw: str) -> tuple[str, str, tuple[str, ...]]:
    if sp is None:
        return raw, raw, tuple(sorted(set(re.findall(r"[A-Za-z_]\w*", raw))))
    try:
        expression = sp.sympify(raw, locals={name: sp.Symbol(name) for name in re.findall(r"[A-Za-z_]\w*", raw)})
        if not isinstance(expression, sp.Basic):
            raise TypeError("not a symbolic expression")
        return str(expression), sp.latex(expression), tuple(sorted(str(item) for item in expression.free_symbols))
    except Exception:
        return raw, raw, tuple(sorted(set(re.findall(r"[A-Za-z_]\w*", raw))))


def _from_python(path: Path, root: Path) -> list[FormulaNode]:
    relative = str(path.relative_to(root))
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source, filename=relative)
    except (OSError, SyntaxError):
        return []
    nodes: list[FormulaNode] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not node.targets:
            continue
        if not isinstance(node.value, (ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Constant)):
            continue
        target = ast.unparse(node.targets[0])
        value = ast.unparse(node.value)
        normalized, latex, symbols = _python_formula(value)
        nodes.append(FormulaNode(_formula_id(relative, node.lineno, value), f"{target} = {value}",
                                 f"{target} = {latex}", normalized, symbols, relative, node.lineno, "assignment"))
    return nodes


_TEX_BLOCK = re.compile(r"(?:\\\((.*?)\\\)|\\\[(.*?)\\\]|\$\$(.*?)\$\$|\\begin\{equation\*?\}(.*?)\\end\{equation\*?\})", re.S)


def _from_markup(path: Path, root: Path) -> list[FormulaNode]:
    relative = str(path.relative_to(root))
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    nodes: list[FormulaNode] = []
    for match in _TEX_BLOCK.finditer(source):
        raw = next((group for group in match.groups() if group is not None), "").strip()
        if not raw:
            continue
        line = source.count("\n", 0, match.start()) + 1
        symbols = tuple(sorted(set(re.findall(r"\\?([A-Za-z]+)(?:_|\^)?", raw))))
        nodes.append(FormulaNode(_formula_id(relative, line, raw), raw, raw, raw, symbols, relative, line, "documentation"))
    return nodes


def extract_formulas(root: Path) -> tuple[FormulaNode, ...]:
    root = root.resolve()
    excluded = {".git", ".venv", "__pycache__", ".pytest_cache", "node_modules"}
    formulas: list[FormulaNode] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or excluded.intersection(path.parts):
            continue
        if path.suffix.lower() == ".py":
            formulas.extend(_from_python(path, root))
        elif path.suffix.lower() in {".md", ".tex", ".txt"}:
            formulas.extend(_from_markup(path, root))
    return tuple(formulas)


def formula_evidence(formula: FormulaNode) -> dict[str, Any]:
    return {
        "formula_id": formula.formula_id,
        "source": f"{formula.source_file}:{formula.source_line}",
        "claim": formula.latex,
        "symbols": formula.symbols,
        "role": formula.role,
    }


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Extract source-grounded formulas from a repository.")
    parser.add_argument("root", nargs="?", default=".")
    args = parser.parse_args()
    for formula in extract_formulas(Path(args.root)):
        print(f"{formula.source_file}:{formula.source_line} [{formula.formula_id}] {formula.latex}")


if __name__ == "__main__":
    main()
