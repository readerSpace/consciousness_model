from coding_world_benchmark.coding_agent_app import LocalWorkspaceAgent
from coding_world_benchmark.open_scientific_compiler_l74_experiment import (
    compile_pairwise_distance_experiment,
    is_pairwise_geometry_request,
    run_compiled_experiment,
)
import pytest


REQUEST = "次の実験を検証して。Estimate pairwise distance from mutual information or entanglement entropy."


def test_pairwise_request_is_not_repo_search():
    assert is_pairwise_geometry_request(REQUEST)


def test_research_plan_compiles_to_candidate_model_experiment():
    plan = compile_pairwise_distance_experiment()
    assert "mutual information" in plan.hypothesis
    assert len(plan.candidate_models) == 3
    assert "shuffled correlations" in plan.controls
    assert "def selected" not in plan.source
    assert "models =" in plan.source


def test_compiled_experiment_executes_and_compares_models():
    result = run_compiled_experiment(compile_pairwise_distance_experiment())
    assert result.return_code == 0
    assert result.selected_model == "A_neg_log"
    assert result.reconstruction_error == pytest.approx(0.0, abs=1e-18)
    assert result.triangle_violation_rate == 0.0
    assert len(result.model_errors) == 3


def test_app_returns_compiled_report_instead_of_unrelated_repo_file(tmp_path):
    result = LocalWorkspaceAgent(tmp_path).handle(REQUEST)
    assert "Compiled Scientific Experiment Report" in result.text
    assert "selected model: `A_neg_log`" in result.text
    assert "relativistic_tube_simulation.py" not in result.text
