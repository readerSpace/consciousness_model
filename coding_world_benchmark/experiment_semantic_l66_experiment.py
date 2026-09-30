"""L6.6: extract experiment semantics from simulation Python files.

This layer lifts repository facts from "matching files" into research-oriented
meaning: experiment entry point, parameters, initial conditions, execution
logic, observables, outputs, success conditions, and line evidence.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

from .execution_router_l65_experiment import discover_execution_candidates


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    value: object | None
    default: object | None
    source: str
    role: str | None
    evidence_id: str


@dataclass(frozen=True)
class StateSpec:
    name: str
    value: object | None
    role: str | None
    evidence_id: str


@dataclass(frozen=True)
class ProcessStep:
    order: int
    role: str
    symbols: tuple[str, ...]
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    evidence_ids: tuple[str, ...]


@dataclass(frozen=True)
class ExperimentSpec:
    entry_point: str
    purpose: str | None
    parameters: tuple[ParameterSpec, ...]
    initial_conditions: tuple[StateSpec, ...]
    execution_steps: tuple[ProcessStep, ...]
    observables: tuple[str, ...]
    outputs: tuple[str, ...]
    success_conditions: tuple[str, ...]
    evidence_ids: tuple[str, ...]


def is_experiment_configuration_request(text: str) -> bool:
    lower = text.lower()
    has_experiment = any(term in text for term in ("シミュレーション", "実験", "検証")) or any(
        term in lower for term in ("simulation", "experiment")
    )
    asks_config = any(term in text for term in ("設定", "条件", "パラメータ", "構成", "どのような")) or any(
        term in lower for term in ("configuration", "setting", "parameter", "condition")
    )
    return has_experiment and asks_config


def _literal(node: ast.AST) -> object | None:
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError):
        return None


def _source_segment(source: str, node: ast.AST) -> str:
    return ast.get_source_segment(source, node) or ast.unparse(node)


def _name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    if isinstance(node, ast.Subscript):
        return _name(node.value)
    return None


def _assigned_names(target: ast.AST) -> tuple[str, ...]:
    if isinstance(target, ast.Name):
        return (target.id,)
    if isinstance(target, (ast.Tuple, ast.List)):
        return tuple(name for item in target.elts for name in _assigned_names(item))
    return ()


def _role_for_name(name: str) -> str | None:
    lower = name.lower()
    if lower in {"beta", "gamma", "velocity", "v", "speed"} or "velocity" in lower:
        return "velocity/control parameter"
    if lower in {"dt", "time_step", "timestep"}:
        return "time step"
    if lower in {"steps", "n_steps", "num_steps", "iterations"} or "step" in lower:
        return "step count"
    if lower in {"x0", "initial_x", "position0"} or "initial" in lower and ("x" in lower or "position" in lower):
        return "initial position"
    if lower in {"v0", "initial_v", "velocity0"} or "initial" in lower and "vel" in lower:
        return "initial velocity"
    if "seed" in lower:
        return "random seed"
    if "output" in lower or "path" in lower:
        return "output path"
    return None


def _is_parameter_name(name: str) -> bool:
    lower = name.lower()
    return bool(_role_for_name(name)) or lower in {
        "beta", "dt", "steps", "n_steps", "num_steps", "duration", "sample_rate", "sweep_min", "sweep_max"
    }


def _is_initial_condition_name(name: str) -> bool:
    lower = name.lower()
    return (
        lower.startswith(("initial_", "init_"))
        or lower.endswith("0")
        or lower in {"x0", "v0", "state0", "initial_state", "initial_position", "initial_velocity"}
    )


def _call_name(node: ast.Call) -> str:
    return _name(node.func) or ""


def _extract_argparse_parameters(tree: ast.AST, relative: str) -> list[ParameterSpec]:
    parameters: list[ParameterSpec] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not _call_name(node).endswith("add_argument"):
            continue
        option = next((arg.value for arg in node.args if isinstance(arg, ast.Constant) and isinstance(arg.value, str)), None)
        if not option:
            continue
        name = option.lstrip("-").replace("-", "_")
        default = None
        for keyword in node.keywords:
            if keyword.arg == "default":
                default = _literal(keyword.value)
        parameters.append(ParameterSpec(name, None, default, "CLI", _role_for_name(name), f"{relative}:{node.lineno}"))
    return parameters


def _extract_assignment_specs(tree: ast.AST, source: str, relative: str) -> tuple[list[ParameterSpec], list[StateSpec]]:
    parameters: list[ParameterSpec] = []
    states: list[StateSpec] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        value_node = node.value
        if value_node is None:
            continue
        value = _literal(value_node)
        rendered = _source_segment(source, value_node)
        for target in targets:
            for name in _assigned_names(target):
                evidence = f"{relative}:{node.lineno}"
                if _is_initial_condition_name(name):
                    states.append(StateSpec(name, value if value is not None else rendered, _role_for_name(name), evidence))
                elif _is_parameter_name(name) and value is not None:
                    parameters.append(ParameterSpec(name, value, None, "constant", _role_for_name(name), evidence))
    return parameters, states


def _step_role(name: str) -> str | None:
    lower = name.lower()
    if any(term in lower for term in ("init", "initial", "setup", "make_state")):
        return "初期化"
    if any(term in lower for term in ("simulate", "evolve", "update", "step", "integrate")):
        return "時間発展"
    if any(term in lower for term in ("measure", "observe", "metric", "evaluate")):
        return "観測量計算"
    if any(term in lower for term in ("save", "plot", "write", "export")):
        return "結果保存"
    if lower in {"main", "run"}:
        return "実行制御"
    return None


def _extract_process_steps(tree: ast.AST, relative: str) -> tuple[ProcessStep, ...]:
    functions = {node.name: node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    entry = functions.get("main") or functions.get("run")
    ordered_calls: list[tuple[str, int]] = []
    if entry:
        for node in ast.walk(entry):
            if isinstance(node, ast.Call):
                call = _call_name(node).split(".")[-1]
                if call in functions:
                    ordered_calls.append((call, node.lineno))
    if not ordered_calls:
        ordered_calls = [(name, node.lineno) for name, node in functions.items()]
    steps: list[ProcessStep] = []
    used_roles: set[str] = set()
    for name, line in ordered_calls:
        role = _step_role(name)
        if not role or role in used_roles:
            continue
        used_roles.add(role)
        steps.append(ProcessStep(len(steps) + 1, role, (name,), (), (), (f"{relative}:{line}",)))
    return tuple(steps)


def _extract_outputs_and_observables(tree: ast.AST, relative: str) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    outputs: list[str] = []
    observables: list[str] = []
    success: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            call = _call_name(node)
            evidence = f"{relative}:{node.lineno}"
            if any(term in call for term in ("save", "savefig", "to_csv", "write", "dump", "print")):
                outputs.append(f"{call} ({evidence})")
            if any(term in call.lower() for term in ("measure", "mean", "max", "min", "norm", "evaluate")):
                observables.append(f"{call} ({evidence})")
        if isinstance(node, ast.Assert):
            success.append(f"assertion ({relative}:{node.lineno})")
        if isinstance(node, ast.Compare):
            text = ast.unparse(node)
            if any(op in text for op in ("<", ">", "==")) and any(term in text.lower() for term in ("error", "threshold", "success", "stable")):
                success.append(f"{text} ({relative}:{node.lineno})")
    return tuple(dict.fromkeys(observables)), tuple(dict.fromkeys(outputs)), tuple(dict.fromkeys(success))


def resolve_experiment_file(root: Path, request: str) -> Path:
    root = root.resolve()
    explicit = re.search(r"([A-Za-z0-9_.-]+\.py)", request)
    if explicit and (root / explicit.group(1)).is_file():
        return (root / explicit.group(1)).resolve()
    candidates = discover_execution_candidates(root, request)
    if candidates:
        return (root / candidates[0].script).resolve()
    py_files = [
        path for path in sorted(root.rglob("*.py"))
        if not {".git", ".venv", "__pycache__", ".pytest_cache", "node_modules"}.intersection(path.parts)
    ]
    scored = []
    for path in py_files:
        haystack = path.name.lower()
        score = 0
        if "シミュレーション" in request and "simulation" in haystack:
            score += 5
        if any(term in haystack for term in ("experiment", "simulation", "sim")):
            score += 2
        scored.append((score, path))
    if scored:
        return max(scored, key=lambda item: item[0])[1].resolve()
    raise ValueError("実験/シミュレーションのPythonファイルを特定できませんでした")


def extract_experiment_spec(root: Path, request: str) -> ExperimentSpec:
    root = root.resolve()
    path = resolve_experiment_file(root, request)
    relative = path.relative_to(root).as_posix()
    source = path.read_text(encoding="utf-8", errors="replace")
    tree = ast.parse(source, filename=relative)
    cli_parameters = _extract_argparse_parameters(tree, relative)
    constant_parameters, initial_conditions = _extract_assignment_specs(tree, source, relative)
    observables, outputs, success_conditions = _extract_outputs_and_observables(tree, relative)
    steps = _extract_process_steps(tree, relative)
    evidence = tuple(dict.fromkeys(
        [item.evidence_id for item in cli_parameters]
        + [item.evidence_id for item in constant_parameters]
        + [item.evidence_id for item in initial_conditions]
        + [evidence for step in steps for evidence in step.evidence_ids]
    ))
    purpose = None
    module_doc = ast.get_docstring(tree)
    if module_doc:
        purpose = module_doc.splitlines()[0]
    elif "simulation" in path.name.lower():
        purpose = "Python simulation"
    return ExperimentSpec(
        relative,
        purpose,
        tuple(cli_parameters + constant_parameters),
        tuple(initial_conditions),
        steps,
        observables,
        outputs,
        success_conditions,
        evidence,
    )


def format_experiment_report(spec: ExperimentSpec) -> str:
    lines = ["## Experiment Semantic Report", "", "【対象実験】", f"`{spec.entry_point}`"]
    lines.extend(["", "【目的】", spec.purpose or "コード構造から明示目的は特定できませんでした。"])
    lines.append("")
    lines.append("【初期条件】")
    if spec.initial_conditions:
        lines.extend(
            f"- `{item.name}` = `{item.value}` ({item.role or 'initial condition'}, 根拠: `{item.evidence_id}`)"
            for item in spec.initial_conditions
        )
    else:
        lines.append("- 明示的な初期条件は抽出できませんでした。")
    lines.extend(["", "【主要パラメータ】"])
    if spec.parameters:
        for item in spec.parameters:
            value = item.default if item.default is not None else item.value
            value_label = "default" if item.default is not None else "value"
            lines.append(f"- `{item.name}` {value_label}=`{value}` source={item.source} role={item.role or 'parameter'} 根拠: `{item.evidence_id}`")
    else:
        lines.append("- 明示的なパラメータは抽出できませんでした。")
    lines.extend(["", "【処理の流れ】"])
    if spec.execution_steps:
        lines.extend(f"{step.order}. {step.role}: {', '.join(f'`{symbol}`' for symbol in step.symbols)} 根拠: {', '.join(f'`{e}`' for e in step.evidence_ids)}" for step in spec.execution_steps)
    else:
        lines.append("- 処理段階は関数名から圧縮できませんでした。")
    lines.extend(["", "【観測量】"])
    lines.extend([f"- `{item}`" for item in spec.observables] or ["- 明示的な観測量計算は抽出できませんでした。"])
    lines.extend(["", "【出力】"])
    lines.extend([f"- `{item}`" for item in spec.outputs] or ["- 明示的な出力処理は抽出できませんでした。"])
    lines.extend(["", "【検証条件】"])
    lines.extend([f"- `{item}`" for item in spec.success_conditions] or ["- 明示的な成功条件/assertionは抽出できませんでした。"])
    lines.extend(["", "【根拠】"])
    lines.extend([f"- `{item}`" for item in spec.evidence_ids] or ["- 行根拠なし"])
    return "\n".join(lines)
