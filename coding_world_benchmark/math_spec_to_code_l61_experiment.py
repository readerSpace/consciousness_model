"""L6.1: generate executable code from a small mathematical specification IR."""
from __future__ import annotations

import ast
from dataclasses import dataclass
import re
from typing import Any

import sympy as sp


@dataclass(frozen=True)
class EquationSpec:
    derivative: str
    rhs: str


@dataclass(frozen=True)
class SpecificationIR:
    states: tuple[str, ...]
    parameters: tuple[str, ...]
    equations: tuple[EquationSpec, ...]
    solver: str
    outputs: tuple[str, ...] = ("trajectory",)
    invariants: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProgramArtifact:
    source: str
    tree: ast.Module
    semantic_formula: str
    solver: str
    states: tuple[str, ...] = ()
    parameters: tuple[str, ...] = ()


def specification_from_text(text: str) -> SpecificationIR:
    normalized = text.replace("−", "-").replace("·", "*").replace("′", "'")
    matches = list(re.finditer(r"d\s*([A-Za-z_]\w*)\s*/\s*d\s*t\s*=\s*([^,;\n]+)", normalized))
    if not matches:
        matches = list(re.finditer(r"([A-Za-z_]\w*)_dot\s*=\s*([^,;\n]+)", normalized))
    if not matches:
        raise ValueError("微分方程式を指定してください（例: dx/dt = -k*x）")
    equations = []
    states = []
    all_names = set()
    for match in matches:
        state = match.group(1)
        rhs = re.split(r"\s+(?:with|using|and|check|solve)\b", match.group(2), maxsplit=1, flags=re.I)[0].strip().rstrip(".")
        states.append(state)
        equations.append(EquationSpec(f"d{state}/dt", rhs))
        all_names.update(re.findall(r"[A-Za-z_]\w*", rhs))
    parameters = tuple(sorted(all_names - set(states) - {"sin", "cos", "exp"}))
    solver = "RK4" if re.search(r"RK\s*4|Runge[- ]?Kutta", normalized, re.I) else "Euler"
    invariants = ("energy conservation",) if re.search(r"energy|保存則|conserv", normalized, re.I) else ()
    return SpecificationIR(tuple(states), parameters, tuple(equations), solver,
                           ("trajectory",), invariants)


def _validate_expression(expression: str, names: set[str]) -> sp.Expr:
    parsed = sp.sympify(expression, locals={name: sp.Symbol(name) for name in names})
    if parsed.free_symbols - {sp.Symbol(name) for name in names}:
        raise ValueError("equation contains undeclared symbols")
    return parsed


def generate_program(spec: SpecificationIR) -> ProgramArtifact:
    if len(spec.states) != len(spec.equations) or not spec.states:
        raise ValueError("statesとequationsの数が一致していません")
    if len(spec.states) > 1:
        return _generate_multistate_program(spec)
    state = spec.states[0]
    equation = spec.equations[0]
    expression = _validate_expression(equation.rhs, set(spec.parameters) | {state})
    rhs = sp.pycode(expression)
    parameter_args = ", ".join(spec.parameters)
    state_args = f"{state}, {parameter_args}" if parameter_args else state
    if spec.solver.upper() == "RK4":
        source = f'''def rhs({state_args}):\n    return {rhs}\n\ndef rk4_step({state_args}, dt):\n    k1 = rhs({state_args})\n    k2 = rhs({state} + 0.5 * dt * k1, {parameter_args})\n    k3 = rhs({state} + 0.5 * dt * k2, {parameter_args})\n    k4 = rhs({state} + dt * k3, {parameter_args})\n    return {state} + dt * (k1 + 2*k2 + 2*k3 + k4) / 6\n\ndef simulate({state}, times, {parameter_args}):\n    values = [{state}]\n    for left, right in zip(times, times[1:]):\n        {state} = rk4_step({state}, {parameter_args}, right - left)\n        values.append({state})\n    return values\n'''
    else:
        source = f'''def rhs({state_args}):\n    return {rhs}\n\ndef simulate({state}, times, {parameter_args}):\n    values = [{state}]\n    for left, right in zip(times, times[1:]):\n        {state} = {state} + (right - left) * rhs({state}, {parameter_args})\n        values.append({state})\n    return values\n'''
    tree = ast.parse(source)
    return ProgramArtifact(source, tree, sp.sstr(expression), spec.solver, spec.states, spec.parameters)


def _generate_multistate_program(spec: SpecificationIR) -> ProgramArtifact:
    state_names = spec.states
    expressions = [_validate_expression(item.rhs, set(state_names) | set(spec.parameters)) for item in spec.equations]
    state_index = {name: index for index, name in enumerate(state_names)}
    def vector_expression(expression: sp.Expr) -> str:
        rendered = sp.pycode(expression)
        for name, index in sorted(state_index.items(), key=lambda item: -len(item[0])):
            rendered = re.sub(rf"\b{re.escape(name)}\b", f"state[{index}]", rendered)
        return rendered
    rhs_lines = ", ".join(vector_expression(expression) for expression in expressions)
    parameter_args = ", ".join(spec.parameters)
    call_params = f", {parameter_args}" if parameter_args else ""
    if spec.solver.upper() == "RK4":
        source = f'''def rhs(state{call_params}):\n    return ({rhs_lines},)\n\ndef rk4_step(state, dt{call_params}):\n    k1 = rhs(state{call_params})\n    k2 = rhs([value + 0.5 * dt * slope for value, slope in zip(state, k1)]{call_params})\n    k3 = rhs([value + 0.5 * dt * slope for value, slope in zip(state, k2)]{call_params})\n    k4 = rhs([value + dt * slope for value, slope in zip(state, k3)]{call_params})\n    return [value + dt * (a + 2*b + 2*c + d) / 6 for value, a, b, c, d in zip(state, k1, k2, k3, k4)]\n\ndef simulate(state, times{call_params}):\n    values = [list(state)]\n    for left, right in zip(times, times[1:]):\n        state = rk4_step(state, right - left{call_params})\n        values.append(list(state))\n    return values\n'''
    else:
        source = f'''def rhs(state{call_params}):\n    return ({rhs_lines},)\n\ndef simulate(state, times{call_params}):\n    values = [list(state)]\n    for left, right in zip(times, times[1:]):\n        slope = rhs(state{call_params})\n        state = [value + (right - left) * derivative for value, derivative in zip(state, slope)]\n        values.append(list(state))\n    return values\n'''
    tree = ast.parse(source)
    return ProgramArtifact(source, tree, "; ".join(sp.sstr(expression) for expression in expressions), spec.solver, spec.states, spec.parameters)


def semantic_check(artifact: ProgramArtifact, *, samples: int = 8) -> dict[str, Any]:
    namespace: dict[str, Any] = {}
    exec(compile(artifact.tree, "<generated>", "exec"), namespace)
    if "; " in artifact.semantic_formula:
        formulas = [sp.sympify(item) for item in artifact.semantic_formula.split("; ")]
        state_names = list(artifact.states)
        parameter_names = list(artifact.parameters)
        expected = [sp.lambdify(tuple(sp.Symbol(name) for name in state_names + parameter_names), formula, "math") for formula in formulas]
        observed = namespace["rhs"]
        errors = []
        for index in range(samples):
            state = [0.3 + index / 7, -0.2 + index / 11][:len(formulas)]
            values = state + [0.7 + index / 10 for _ in parameter_names]
            actual = observed(state, *values[len(state):])
            errors.extend(abs(float(function(*values)) - float(value)) for function, value in zip(expected, actual))
        result = {"passed": max(errors, default=0.0) < 1e-12, "max_error": max(errors, default=0.0),
                  "samples": samples, "solver": artifact.solver, "invariant_passed": None}
        if len(formulas) == 2:
            trajectory = namespace["simulate"]([1.0, 0.0], [i / 10 for i in range(31)], *([1.0] * len(parameter_names)))
            energies = [sum(value * value for value in state) for state in trajectory]
            drift = max(energies) - min(energies)
            result["invariant_passed"] = drift < 1e-6
            result["invariant_drift"] = drift
            result["passed"] = result["passed"] and result["invariant_passed"]
        return result
    symbol_names = sorted(str(symbol) for symbol in sp.sympify(artifact.semantic_formula).free_symbols)
    if len(symbol_names) < 2:
        raise ValueError("semantic check requires state and parameter")
    state_name, parameter_name = symbol_names[0], symbol_names[1]
    state = sp.Symbol(state_name)
    parameter = sp.Symbol(parameter_name)
    expected = sp.lambdify((state, parameter), sp.sympify(artifact.semantic_formula), "math")
    observed = namespace["rhs"]
    errors = [abs(float(expected(x, k)) - float(observed(x, k)))
              for x, k in ((i / 3, 0.2 + i / 10) for i in range(samples))]
    return {"passed": max(errors, default=0.0) < 1e-12, "max_error": max(errors, default=0.0),
            "samples": samples, "solver": artifact.solver}


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Generate and semantically check an ODE program.")
    parser.add_argument("spec", nargs="+", help="natural-language mathematical specification")
    args = parser.parse_args()
    spec = specification_from_text(" ".join(args.spec))
    artifact = generate_program(spec)
    print(artifact.source)
    print(semantic_check(artifact))


if __name__ == "__main__":
    main()
