from coding_world_benchmark.canonical_ir_l63_experiment import canonicalize_request
from coding_world_benchmark.coding_agent_app import LocalWorkspaceAgent
from coding_world_benchmark.open_research_l69_experiment import (
    design_open_research_plan,
    format_open_research_report,
    is_open_research_request,
    parse_research_question,
    run_open_research_plan,
)
from coding_world_benchmark.structured_repository_report_l62_experiment import RequestType, classify_request


REQUEST = "相対論的な運動をしている荷電粒子が加速度を受けた時に電磁波を出力するシミュレーションを作成実行して。結果を要約して"


def test_open_question_becomes_research_ir_and_open_research_route():
    assert is_open_research_request(REQUEST)
    question = parse_research_question(REQUEST)
    assert question.domain == "electrodynamics"
    assert question.task == "simulate_and_verify"
    assert classify_request(REQUEST).request_type is RequestType.OPEN_RESEARCH_TASK
    assert canonicalize_request(REQUEST).action == "open_research_task"


def test_open_plan_has_knowledge_gap_measurements_controls_and_equations():
    plan = design_open_research_plan(REQUEST)

    assert plan.question.requested_effect == "electromagnetic radiation"
    assert plan.knowledge_gaps[0].blocking is False
    assert "zero acceleration" in plan.controls
    assert "radiated_power" in plan.dependent_variables
    assert any("gamma" in equation for equation in plan.equations)
    assert "def radiated_power" in plan.source


def test_generated_open_experiment_runs_and_supports_predictions():
    result = run_open_research_plan(design_open_research_plan(REQUEST))

    assert result.return_code == 0
    assert result.zero_acceleration_power == 0.0
    assert result.positive_acceleration_power > 0.0
    assert result.power_ratio_at_high_beta > 1.0
    assert result.monotonic_with_beta is True
    assert result.diagnosis == "implementation_ok_and_prediction_supported"


def test_app_uses_open_research_report_without_repo_target(tmp_path):
    result = LocalWorkspaceAgent(tmp_path).handle(REQUEST)
    assert result.provider == "local"
    assert "Open Research Verification Report" in result.text
    assert "ResearchQuestionIR" in result.text
    assert "【Knowledge Gap】" in result.text
    assert "【測定設計】" in result.text
    assert "【結論】" in result.text
    assert "候補ファイル" not in result.text
