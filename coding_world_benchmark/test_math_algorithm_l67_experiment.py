from pathlib import Path

from coding_world_benchmark.math_algorithm_l67_experiment import (
    extract_mathematical_model,
    format_math_report,
    is_math_explanation_request,
)
from coding_world_benchmark.canonical_ir_l63_experiment import canonicalize_request
from coding_world_benchmark.structured_repository_report_l62_experiment import RequestType, classify_request


def _repo(tmp_path: Path) -> Path:
    (tmp_path / "README.md").write_text(
        "Quick start:\npython .\\relativistic_tube_simulation.py --beta 0.9\n",
        encoding="utf-8",
    )
    (tmp_path / "relativistic_tube_simulation.py").write_text(
        "import argparse\n"
        "import math\n\n"
        "dt = 0.01\n"
        "initial_x = 0.0\n"
        "initial_v = 0.0\n\n"
        "def initialize_state(beta):\n"
        "    gamma = 1.0 / math.sqrt(1.0 - beta**2)\n"
        "    state0 = {'x': initial_x, 'v': initial_v, 'beta': beta, 'gamma': gamma}\n"
        "    return state0\n\n"
        "def simulate(state, steps):\n"
        "    for _ in range(steps):\n"
        "        state['x'] = state['x'] + state['beta'] * dt\n"
        "        state['v'] += state['gamma'] * dt\n"
        "    return state\n\n"
        "def measure_observable(state):\n"
        "    observable = state['x'] / state['gamma']\n"
        "    return observable\n\n"
        "def main():\n"
        "    parser = argparse.ArgumentParser()\n"
        "    parser.add_argument('--beta', type=float, default=0.9)\n"
        "    parser.add_argument('--steps', type=int, default=100)\n"
        "    args = parser.parse_args()\n"
        "    state = initialize_state(args.beta)\n"
        "    final_state = simulate(state, args.steps)\n"
        "    print(measure_observable(final_state))\n",
        encoding="utf-8",
    )
    return tmp_path


def test_math_explanation_request_is_detected():
    request = "pythonシミュレーションのアルゴリズムを数式で表して"
    assert is_math_explanation_request(request)
    assert classify_request(request).request_type is RequestType.EXPLAIN_AS_MATH
    assert canonicalize_request(request).action == "explain_as_math"
    assert not is_math_explanation_request("pythonシミュレーションの設定を調査して")


def test_extracts_math_model_with_latex_equations_updates_and_evidence(tmp_path: Path):
    model = extract_mathematical_model(_repo(tmp_path), "pythonシミュレーションのアルゴリズムを数式で表して")

    assert model.target_script == "relativistic_tube_simulation.py"
    assert "beta" in model.parameters
    assert "steps" in model.parameters
    assert {"initial_x", "initial_v", "state0", "x", "v"}.intersection(model.states)
    assert any(equation.target == "gamma" and r"\gamma" in equation.latex and r"\sqrt" in equation.latex for equation in model.equations)
    assert any(equation.target == "x" and "x_{n+1}" in equation.latex for equation in model.update_rules)
    assert any(equation.target == "v" and "v_{n+1}" in equation.latex for equation in model.update_rules)
    assert any(equation.lhs == "d x / dt" and r"\frac{d x}{dt}" in equation.latex for equation in model.governing_equations)
    assert any(equation.lhs == "d v / dt" and r"\frac{d v}{dt}" in equation.latex for equation in model.governing_equations)
    assert model.initial_conditions
    assert model.evidence_ids


def test_math_report_uses_latex_blocks_and_required_sections(tmp_path: Path):
    report = format_math_report(extract_mathematical_model(_repo(tmp_path), "アルゴリズムを数式で表して"))

    assert "Mathematical Algorithm Report" in report
    assert "【状態変数】" in report
    assert "【主要パラメータ】" in report
    assert "【支配方程式】" in report
    assert "連続極限として推定" in report
    assert "【基本式】" in report
    assert "【更新則】" in report
    assert "【初期条件】" in report
    assert "【計算順序】" in report
    assert "$$" in report
    assert "候補ファイル" not in report
