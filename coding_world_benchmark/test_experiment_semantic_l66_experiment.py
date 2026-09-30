from pathlib import Path

from coding_world_benchmark.experiment_semantic_l66_experiment import (
    extract_experiment_spec,
    format_experiment_report,
    is_experiment_configuration_request,
)


def _repo(tmp_path: Path) -> Path:
    (tmp_path / "README.md").write_text(
        "# Relativistic tube\n\n"
        "Quick start:\n"
        "python .\\relativistic_tube_simulation.py --beta 0.9\n",
        encoding="utf-8",
    )
    (tmp_path / "relativistic_tube_simulation.py").write_text(
        '"""Relativistic tube simulation for beta-dependent trajectory checks."""\n'
        "import argparse\n\n"
        "dt = 0.01\n"
        "steps = 10000\n"
        "initial_position = 0.0\n"
        "initial_velocity = 0.0\n\n"
        "def initialize_state(beta):\n"
        "    state0 = {'x': initial_position, 'v': initial_velocity, 'beta': beta}\n"
        "    return state0\n\n"
        "def simulate(state, dt, steps):\n"
        "    trajectory = []\n"
        "    for _ in range(steps):\n"
        "        state['x'] = state['x'] + state['beta'] * dt\n"
        "        trajectory.append(state['x'])\n"
        "    return trajectory\n\n"
        "def measure_trajectory(trajectory):\n"
        "    return max(trajectory)\n\n"
        "def save_results(observable):\n"
        "    print('observable', observable)\n\n"
        "def main():\n"
        "    parser = argparse.ArgumentParser()\n"
        "    parser.add_argument('--beta', type=float, default=0.9)\n"
        "    args = parser.parse_args()\n"
        "    state = initialize_state(args.beta)\n"
        "    trajectory = simulate(state, dt, steps)\n"
        "    observable = measure_trajectory(trajectory)\n"
        "    save_results(observable)\n\n"
        "if __name__ == '__main__':\n"
        "    main()\n",
        encoding="utf-8",
    )
    return tmp_path


def test_experiment_configuration_request_is_detected():
    assert is_experiment_configuration_request("pythonでのシミュレーションがどのような設定で行われているか調査して")
    assert not is_experiment_configuration_request("READMEを要約して")


def test_extracts_parameters_initial_conditions_logic_outputs_and_evidence(tmp_path: Path):
    spec = extract_experiment_spec(
        _repo(tmp_path),
        "pythonでのシミュレーションがどのような設定で行われているか調査して",
    )

    parameter_names = {item.name for item in spec.parameters}
    initial_names = {item.name for item in spec.initial_conditions}
    step_roles = [item.role for item in spec.execution_steps]

    assert spec.entry_point == "relativistic_tube_simulation.py"
    assert "beta" in parameter_names
    assert "dt" in parameter_names
    assert "steps" in parameter_names
    assert "initial_position" in initial_names
    assert "initial_velocity" in initial_names
    assert step_roles == ["初期化", "時間発展", "観測量計算", "結果保存"]
    assert any("measure_trajectory" in item for item in spec.observables)
    assert any("print" in item for item in spec.outputs)
    assert spec.evidence_ids


def test_experiment_report_has_required_sections_not_file_list(tmp_path: Path):
    spec = extract_experiment_spec(_repo(tmp_path), "シミュレーションの設定を調査して")
    report = format_experiment_report(spec)

    assert "【対象実験】" in report
    assert "【初期条件】" in report
    assert "【主要パラメータ】" in report
    assert "【処理の流れ】" in report
    assert "【出力】" in report
    assert "【根拠】" in report
    assert "候補ファイル" not in report
