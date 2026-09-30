from coding_world_benchmark.canonical_ir_l63_experiment import canonicalize_request
from coding_world_benchmark.coding_agent_app import LocalWorkspaceAgent
from coding_world_benchmark.scientific_experiment_l68_experiment import (
    design_relativistic_garage_experiment,
    format_verification_report,
    is_design_and_verify_request,
    run_experiment,
)
from coding_world_benchmark.structured_repository_report_l62_experiment import RequestType, classify_request


REQUEST = "電車が相対論的な速さで車庫に入るとき、同時刻で入れるか検証して"


def test_natural_language_request_routes_to_design_and_verify():
    assert is_design_and_verify_request(REQUEST)
    assert classify_request(REQUEST).request_type is RequestType.DESIGN_AND_VERIFY
    assert canonicalize_request(REQUEST).action == "design_and_verify"


def test_relativistic_plan_contains_hypotheses_equations_and_strategy():
    plan = design_relativistic_garage_experiment(REQUEST)

    assert plan.domain == "special_relativity"
    assert len(plan.hypotheses) == 2
    assert any("\\gamma" in equation for equation in plan.equations)
    assert "analytic_lorentz_transform" == plan.strategy.primary
    assert "numerical_parameter_sweep" == plan.strategy.secondary
    assert "def lorentz_time" in plan.source


def test_generated_experiment_executes_and_evaluates_both_claims():
    plan = design_relativistic_garage_experiment(REQUEST)
    result = run_experiment(plan)

    assert result.return_code == 0
    assert result.fits_at_test_speed is True
    assert result.contracted_length < 60.0
    assert result.beta_threshold >= 0.8
    assert result.simultaneity_delta_train < 0.0
    assert result.sweep_fit_count > 0


def test_app_returns_scientific_report_without_workspace_or_file_search(tmp_path):
    result = LocalWorkspaceAgent(tmp_path).handle(REQUEST)

    assert result.provider == "local"
    assert "Scientific Experiment Verification Report" in result.text
    assert "【検証仮説】" in result.text
    assert "【理論式】" in result.text
    assert "【実行結果】" in result.text
    assert "【判定】" in result.text
    assert "候補ファイル" not in result.text
    assert "見つかりません" not in result.text
