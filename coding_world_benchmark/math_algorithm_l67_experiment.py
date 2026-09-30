"""L6.7: explain simulation algorithms as Math IR and LaTeX.

This module connects repository/experiment understanding to mathematical
semantic understanding. It extracts important equations from the target
simulation script, classifies them by role, renders LaTeX, and reports line
evidence.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
import re

from .experiment_semantic_l66_experiment import ExperimentSpec, extract_experiment_spec, resolve_experiment_file

try:
    import sympy as sp
except ImportError:  # pragma: no cover
    sp = None


@dataclass(frozen=True)
class Equation:
    role: str
    target: str
    expression: str
    latex: str
    evidence_id: str


@dataclass(frozen=True)
class EquationIR:
    """An abstract equation reconstructed from one or more code equations."""

    lhs: str
    rhs: str
    equation_type: str
    semantic_role: str
    latex: str
    derived_from: tuple[str, ...]
    confidence: float
    status: str = "inferred"


@dataclass(frozen=True)
class MathematicalModel:
    target_script: str
    states: tuple[str, ...]
    parameters: tuple[str, ...]
    equations: tuple[Equation, ...]
    update_rules: tuple[Equation, ...]
    constraints: tuple[Equation, ...]
    observables: tuple[Equation, ...]
    initial_conditions: tuple[Equation, ...]
    governing_equations: tuple[EquationIR, ...]
    algorithm_order: tuple[str, ...]
    evidence_ids: tuple[str, ...]


def is_math_explanation_request(text: str) -> bool:
    lower = text.lower()
    asks_math = any(term in text for term in ("数式", "方程式", "式で", "数学")) or any(
        term in lower for term in ("latex", "equation", "formula", "math")
    )
    has_algorithm = any(term in text for term in ("アルゴリズム", "シミュレーション", "実験")) or any(
        term in lower for term in ("algorithm", "simulation", "experiment")
    )
    return asks_math and has_algorithm


def _normalize_symbol(text: str) -> str:
    text = re.sub(r"(\w+)\[['\"]([^'\"]+)['\"]\]", r"\2", text)
    text = re.sub(r"args\.([A-Za-z_]\w*)", r"\1", text)
    text = re.sub(r"math\.sqrt", "sqrt", text)
    return text


def _latex_expr(expr: str) -> str:
    normalized = _normalize_symbol(expr)
    if sp is not None:
        try:
            names = {name: sp.Symbol(name) for name in re.findall(r"[A-Za-z_]\w*", normalized) if name != "sqrt"}
            parsed = sp.sympify(normalized, locals={**names, "sqrt": sp.sqrt})
            return sp.latex(parsed)
        except Exception:
            pass
    return normalized.replace("**", "^").replace("*", r" \cdot ")


def _latex_assignment(target: str, expr: str) -> str:
    return f"{_latex_name(target)} = {_latex_expr(expr)}"


def _latex_name(name: str) -> str:
    greek = {"beta": r"\beta", "gamma": r"\gamma", "rho": r"\rho", "dt": r"\Delta t"}
    if name in greek:
        return greek[name]
    if name.endswith("0") and len(name) > 1:
        return f"{name[:-1]}_0"
    return name.replace("_", r"\_")


def _update_latex(target: str, expr: str) -> str:
    normalized_expr = _normalize_symbol(expr)
    normalized_expr = re.sub(rf"\b{re.escape(target)}\b", f"{target}_n", normalized_expr)
    rhs = _latex_expr(normalized_expr)
    return f"{_latex_name(target)}_{{n+1}} = {rhs}"


def _continuous_equation(update: Equation) -> EquationIR | None:
    """Lift an Euler-like update to its corresponding first-order ODE."""
    target = _normalize_symbol(update.target)
    expression = _normalize_symbol(update.expression).replace(" ", "")
    target_pattern = re.escape(target)
    match = re.fullmatch(
        rf"{target_pattern}\+(.+?)\*(?:dt|delta_t|time_step|h)", expression
    )
    sign = "+"
    if not match:
        match = re.fullmatch(
            rf"{target_pattern}-(.+?)\*(?:dt|delta_t|time_step|h)", expression
        )
        sign = "-"
    if not match:
        match = re.fullmatch(
            rf"{target_pattern}\+(.+?)\*(?:dt|delta_t|time_step|h)", expression
        )
    if not match:
        return None
    rhs = match.group(1)
    if sign == "-":
        rhs = f"-({rhs})"
    lhs_latex = rf"\frac{{d {_latex_name(target)}}}{{dt}}"
    return EquationIR(
        lhs=f"d {target} / dt",
        rhs=rhs,
        equation_type="ODE",
        semantic_role="kinematics" if target in {"x", "position"} else "dynamics",
        latex=f"{lhs_latex} = {_latex_expr(rhs)}",
        derived_from=(update.evidence_id,),
        confidence=0.98,
    )


def _target_name(target: ast.AST) -> str | None:
    if isinstance(target, ast.Name):
        return target.id
    if isinstance(target, ast.Attribute):
        return target.attr
    if isinstance(target, ast.Subscript):
        text = ast.unparse(target)
        return _normalize_symbol(text)
    return None


def _role_for_equation(target: str, expr: str, in_loop: bool) -> str:
    lower = target.lower()
    expr_lower = expr.lower()
    if in_loop or re.search(rf"\b{re.escape(target)}\b", _normalize_symbol(expr)):
        return "STATE_UPDATE"
    if lower in {"gamma", "beta"} or "sqrt" in expr_lower or "**" in expr:
        return "PHYSICAL_EQUATION"
    if any(term in lower for term in ("observable", "metric", "measurement", "trajectory")):
        return "OBSERVABLE"
    if any(term in lower for term in ("error", "threshold", "limit")):
        return "CONSTRAINT"
    return "EQUATION"


def _parents(tree: ast.AST) -> dict[int, ast.AST]:
    parents: dict[int, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[id(child)] = node
    return parents


def _inside_loop(node: ast.AST, parents: dict[int, ast.AST]) -> bool:
    current = node
    while id(current) in parents:
        current = parents[id(current)]
        if isinstance(current, (ast.For, ast.While)):
            return True
    return False


def _extract_equations(path: Path, root: Path) -> tuple[Equation, ...]:
    relative = path.relative_to(root).as_posix()
    source = path.read_text(encoding="utf-8", errors="replace")
    tree = ast.parse(source, filename=relative)
    parents = _parents(tree)
    equations: list[Equation] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target_node in node.targets:
                target = _target_name(target_node)
                if not target:
                    continue
                expr = ast.unparse(node.value)
                role = _role_for_equation(target, expr, _inside_loop(node, parents))
                latex = _update_latex(target, expr) if role == "STATE_UPDATE" else _latex_assignment(target, expr)
                equations.append(Equation(role, target, _normalize_symbol(expr), latex, f"{relative}:{node.lineno}"))
        elif isinstance(node, ast.AugAssign):
            target = _target_name(node.target)
            if not target:
                continue
            op = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/"}.get(type(node.op), "+")
            expr = f"{target} {op} {ast.unparse(node.value)}"
            equations.append(Equation("STATE_UPDATE", target, _normalize_symbol(expr), _update_latex(target, expr), f"{relative}:{node.lineno}"))
    return tuple(equations)


def extract_mathematical_model(root: Path, request: str) -> MathematicalModel:
    root = root.resolve()
    spec = extract_experiment_spec(root, request)
    path = resolve_experiment_file(root, request)
    equations = _extract_equations(path, root)
    state_names = {state.name for state in spec.initial_conditions}
    for equation in equations:
        if equation.role == "STATE_UPDATE":
            state_names.add(equation.target)
    parameters = tuple(dict.fromkeys(parameter.name for parameter in spec.parameters))
    updates = tuple(equation for equation in equations if equation.role == "STATE_UPDATE")
    constraints = tuple(equation for equation in equations if equation.role == "CONSTRAINT")
    observables = tuple(equation for equation in equations if equation.role == "OBSERVABLE")
    initial_conditions = tuple(
        Equation("INITIAL_CONDITION", state.name, str(state.value), _latex_assignment(state.name, str(state.value)), state.evidence_id)
        for state in spec.initial_conditions
    )
    governing_equations = tuple(
        abstracted
        for update in updates
        for abstracted in (_continuous_equation(update),)
        if abstracted is not None
    )
    evidence = tuple(dict.fromkeys(
        [equation.evidence_id for equation in equations]
        + [equation.evidence_id for equation in initial_conditions]
        + list(spec.evidence_ids)
    ))
    return MathematicalModel(
        spec.entry_point,
        tuple(sorted(state_names)),
        parameters,
        equations,
        updates,
        constraints,
        observables,
        initial_conditions,
        governing_equations,
        tuple(step.role for step in spec.execution_steps),
        evidence,
    )


def format_math_report(model: MathematicalModel) -> str:
    physical = tuple(equation for equation in model.equations if equation.role == "PHYSICAL_EQUATION")
    other_equations = tuple(equation for equation in model.equations if equation.role == "EQUATION")
    lines = ["## Mathematical Algorithm Report", "", "【対象】", f"`{model.target_script}`"]
    lines.extend(["", "【状態変数】", *([f"- `{item}`" for item in model.states] or ["- 抽出なし"])])
    lines.extend(["", "【主要パラメータ】", *([f"- `{item}`" for item in model.parameters] or ["- 抽出なし"])])
    lines.extend(["", "【支配方程式】"])
    if model.governing_equations:
        lines.append("コードのEuler型更新から連続極限として推定した方程式です。")
        for equation in model.governing_equations:
            lines.extend([
                f"- 種別: `{equation.equation_type}` / 役割: `{equation.semantic_role}` / 信頼度: `{equation.confidence:.2f}`",
                "$$",
                equation.latex,
                "$$",
                f"- 導出元: `{', '.join(equation.derived_from)}`",
            ])
    else:
        lines.append("- 連続極限を一意に推定できる更新則はありません。離散更新則を参照してください。")
    lines.extend(["", "【基本式】"])
    for equation in physical + other_equations:
        lines.extend([f"- 根拠: `{equation.evidence_id}`", "$$", equation.latex, "$$"])
    if not physical and not other_equations:
        lines.append("- 抽出なし")
    lines.extend(["", "【更新則】"])
    for equation in model.update_rules:
        lines.extend([f"- 根拠: `{equation.evidence_id}`", "$$", equation.latex, "$$"])
    if not model.update_rules:
        lines.append("- 抽出なし")
    lines.extend(["", "【初期条件】"])
    for equation in model.initial_conditions:
        lines.extend([f"- 根拠: `{equation.evidence_id}`", "$$", equation.latex, "$$"])
    if not model.initial_conditions:
        lines.append("- 抽出なし")
    lines.extend(["", "【観測量】"])
    for equation in model.observables:
        lines.extend([f"- 根拠: `{equation.evidence_id}`", "$$", equation.latex, "$$"])
    if not model.observables:
        lines.append("- 明示的な観測量式は抽出できませんでした。")
    lines.extend(["", "【計算順序】"])
    lines.append(" → ".join(model.algorithm_order) if model.algorithm_order else "抽出なし")
    lines.extend(["", "【根拠】"])
    lines.extend([f"- `{item}`" for item in model.evidence_ids] or ["- 行根拠なし"])
    return "\n".join(lines)
