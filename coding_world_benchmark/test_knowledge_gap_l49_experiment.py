from coding_world_benchmark.knowledge_gap_l49_experiment import (
    CandidateAction,
    cases,
    detect_knowledge_gaps,
    run_knowledge_gap_experiment,
    run_policy,
    select_workspace_action,
)


REPORT = run_knowledge_gap_experiment()


def test_l49_uses_twelve_cases_with_no_search_and_blocking_search_examples():
    all_cases = cases()
    assert len(all_cases) == 12
    assert sum(not case.retrieval_needed for case in all_cases) == 3
    assert sum(case.retrieval_needed for case in all_cases) == 9
    assert sum(case.expected_action is CandidateAction.ABANDON_HYPOTHESIS for case in all_cases) >= 2


def test_gap_detector_emits_explicit_structured_unknowns_only_when_needed():
    no_search = next(case for case in cases() if case.name == "traceback_names_local")
    assert detect_knowledge_gaps(no_search) == ()

    low_power = next(case for case in cases() if case.name == "small_n_power")
    gaps = detect_knowledge_gaps(low_power)
    assert len(gaps) == 1
    assert gaps[0].category == "power_analysis"
    assert gaps[0].blocking
    assert gaps[0].confidence < 0.5
    assert gaps[0].expected_value > 0.8


def test_finite_workspace_selects_search_only_for_decision_relevant_gaps():
    no_search = next(case for case in cases() if case.name == "positive_replication_local")
    assert select_workspace_action(no_search, detect_knowledge_gaps(no_search)) is CandidateAction.RUN_EXPERIMENT

    confounded = next(case for case in cases() if case.name == "control_moves_with_treatment")
    assert select_workspace_action(confounded, detect_knowledge_gaps(confounded)) is CandidateAction.SEARCH


def test_knowledge_gap_driven_policy_has_perfect_gap_precision_and_recall():
    metrics = REPORT["policies"]["knowledge_gap_driven"]["metrics"]
    assert metrics["knowledge_gap_precision"] == 1.0
    assert metrics["knowledge_gap_recall"] == 1.0
    assert metrics["unnecessary_search_rate"] == 0.0
    assert metrics["blocking_gap_resolution_rate"] == 1.0
    assert metrics["post_retrieval_action_accuracy"] == 1.0


def test_gap_driven_retrieval_beats_no_retrieval_always_search_and_keyword_search():
    policies = REPORT["policies"]
    gap_driven = policies["knowledge_gap_driven"]["metrics"]
    assert gap_driven["post_retrieval_action_accuracy"] > policies["no_retrieval"]["metrics"]["post_retrieval_action_accuracy"]
    assert gap_driven["unnecessary_search_rate"] < policies["always_search"]["metrics"]["unnecessary_search_rate"]
    assert gap_driven["knowledge_gap_recall"] > policies["keyword_search"]["metrics"]["knowledge_gap_recall"]
    assert gap_driven["experiment_improvement_rate"] > policies["always_search"]["metrics"]["experiment_improvement_rate"]


def test_retrieval_can_stop_the_agent_instead_of_extending_the_experiment():
    terminal = next(case for case in cases() if case.name == "unsafe_to_continue")
    row = run_policy(terminal, "knowledge_gap_driven")
    assert "terminal_falsification" in row["searched_categories"]
    assert row["action"] == "abandon_hypothesis"
    assert row["action_correct"]
