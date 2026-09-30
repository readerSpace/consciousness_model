from pathlib import Path

from coding_world_benchmark.scientific_simulation_l88_experiment import (
    build_charged_particle_task,
    format_simulation_implementation_report,
    implement_scientific_simulation,
    is_scientific_simulation_implementation_request,
)


REQUEST = "一様磁場中で荷電粒子が運動するシミュレーションを作成して"


def test_uniform_magnetic_particle_request_is_implementation_task():
    assert is_scientific_simulation_implementation_request(REQUEST)
    task = build_charged_particle_task(REQUEST)

    assert task.action == "IMPLEMENT_SCIENTIFIC_SIMULATION"
    assert task.domain == "physics"
    assert task.requires_code_change
    assert not task.requires_execution
    assert "charged particle" in task.system


def test_implements_python_simulation_and_tests(tmp_path: Path):
    result = implement_scientific_simulation(tmp_path, REQUEST)

    simulation = tmp_path / "charged_particle_uniform_B.py"
    tests = tmp_path / "test_charged_particle_uniform_B.py"
    assert simulation.is_file()
    assert tests.is_file()
    assert "def rk4_step" in simulation.read_text(encoding="utf-8")
    assert "test_speed_is_conserved" in tests.read_text(encoding="utf-8")

    report = format_simulation_implementation_report(result)
    assert "Scientific Simulation Implementation" in report
    assert "IMPLEMENT_SCIENTIFIC_SIMULATION" in report
    assert "検索" not in report


def test_execute_variant_runs_generated_unittest(tmp_path: Path):
    result = implement_scientific_simulation(tmp_path, REQUEST + " 実行して")

    assert result.verification_command == ("python", "-m", "unittest", "test_charged_particle_uniform_B.py")
    assert result.return_code == 0
    assert "OK" in result.output
