import pytest

from paranormal_intervention_experiment import (
    CausalBeliefAgent,
    ParanormalCaseEnv,
    cause_models,
    evaluate_case,
    generate_case,
    run,
)


def test_agent_view_hides_truth_and_action_outcomes():
    case = generate_case("case_test", "window_reflection", seed=12)
    view = ParanormalCaseEnv(case).observe()

    assert "ground_truth" not in view
    assert "action_outcomes" not in view
    assert case.ground_truth["cause"] not in repr(view)


def test_surface_observation_is_causally_ambiguous():
    models = cause_models()
    causes = [
        cause for cause, model in models.items()
        if cause != "unknown" and model.observation_likelihoods.get("visual:white_human_like_shape", 0.0) >= 0.15
    ]
    window_forms = [
        token for token in models["window_reflection"].observation_likelihoods
        if token.startswith("visual:")
    ]

    assert len(causes) >= 5
    assert len(window_forms) >= 4


def test_eig_prefers_identifying_window_reflection_intervention():
    case = generate_case("case_test", "window_reflection", seed=21, forced_visual="white_human_like_shape")
    env = ParanormalCaseEnv(case)
    agent = CausalBeliefAgent(cause_models())
    agent.observe(env.observe())

    selected = agent.choose_action(env.available_actions())

    assert selected in case.evaluation["minimum_identifying_actions"]


def test_interventions_identify_holdout_surface_form():
    case = generate_case("case_test", "window_reflection", seed=22, forced_visual="face_like_image")
    result = evaluate_case(case, max_actions=3)

    assert result.final_prediction == "window_reflection"
    assert result.minimum_action_hit


def test_unknown_cases_remain_unexplained():
    results = [evaluate_case(generate_case(f"unknown_{index}", "unknown", seed=100 + index), max_actions=3) for index in range(12)]

    assert sum(row.final_prediction == "UNEXPLAINED" for row in results) / len(results) >= 0.75


def test_run_reports_phase1_success_metrics():
    report = run()

    assert report.ground_truth_hidden_from_agent
    assert report.known_cause_count == 10
    assert report.unknown_case_rate == pytest.approx(1 / 11)
    assert report.three_step_known_accuracy > report.one_step_known_accuracy
    assert report.three_step_known_accuracy >= 0.70
    assert report.unknown_overconfidence_rate <= 0.25
    assert report.minimum_identifying_action_rate >= 0.70
    assert report.holdout_window_reflection_accuracy >= 0.80
