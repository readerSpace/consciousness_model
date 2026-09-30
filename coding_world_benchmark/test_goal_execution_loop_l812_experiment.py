from coding_world_benchmark.behavior_monitor_l87_experiment import BehaviorEvent, ExecutionTrace
from coding_world_benchmark.goal_execution_loop_l812_experiment import (
    RemediationAction,
    format_experiment,
    format_loop_outcome,
    remediation_for,
    run_experiment,
    run_goal_loop,
    trace_with,
    unsupported_gaps,
)
from coding_world_benchmark.goal_semantics_l89_experiment import GoalSatisfaction, extract_goal


CREATE_REQUEST = "一様磁場中で荷電粒子が運動するシミュレーションを作成して"
SEARCH_ONLY = "ローカル要約\n一致ファイル: solver.py:12: def step\n一致ファイル: main.py:3: import numpy"


def _has(directives, action):
    return any(item.action == action for item in directives)


def _search_only_executor(request, directives):
    return SEARCH_ONLY, trace_with(BehaviorEvent.REPOSITORY_INDEXED)


def _writes_when_told(request, directives):
    if _has(directives, RemediationAction.WRITE_ARTIFACT):
        return "simulation.py を作成しました。", trace_with(BehaviorEvent.FILE_WRITTEN)
    return _search_only_executor(request, directives)


def test_create_goal_recovers_after_write_artifact_directive():
    outcome = run_goal_loop(CREATE_REQUEST, _writes_when_told)

    assert outcome.status == "GOAL_ACHIEVED"
    assert outcome.iteration_count == 2
    assert outcome.remediation_rounds == 1
    assert outcome.iterations[0].directives == ()
    assert not outcome.iterations[0].satisfaction.achieved
    assert "FILE_WRITTEN >= 1" in outcome.iterations[0].satisfaction.missing_observations
    assert _has(outcome.iterations[1].directives, RemediationAction.WRITE_ARTIFACT)


def test_satisfied_goal_does_not_trigger_a_second_round():
    def executor(request, directives):
        return "simulation.py を作成しました。", trace_with(BehaviorEvent.FILE_WRITTEN)

    outcome = run_goal_loop(CREATE_REQUEST, executor)

    assert outcome.status == "GOAL_ACHIEVED"
    assert outcome.iteration_count == 1
    assert outcome.remediation_rounds == 0


def test_executor_ignoring_directives_escalates_instead_of_spinning():
    outcome = run_goal_loop(CREATE_REQUEST, _search_only_executor, max_iterations=6)

    assert outcome.status == "ESCALATED_NO_PROGRESS"
    assert outcome.iteration_count == 2
    assert "FILE_WRITTEN >= 1" in outcome.unresolved_gaps


def test_information_goal_is_not_forced_to_write_a_file():
    def executor(request, directives):
        return "ローレンツ力は F = qv × B で与えられます。", ExecutionTrace()

    outcome = run_goal_loop("ローレンツ力について説明して", executor)

    assert outcome.status == "GOAL_ACHIEVED"
    assert outcome.iteration_count == 1
    assert outcome.plan.contract.required_observations == (
        "response satisfies requested information need",
    )


def test_execute_goal_recovers_return_code_without_rerunning_everything():
    def executor(request, directives):
        if _has(directives, RemediationAction.CAPTURE_RETURN_CODE):
            return "テストを実行しました。終了コード: 0", trace_with(BehaviorEvent.SUBPROCESS_STARTED)
        return "テストを実行しました。", trace_with(BehaviorEvent.SUBPROCESS_STARTED)

    outcome = run_goal_loop("テストを実行して", executor)

    assert outcome.status == "GOAL_ACHIEVED"
    assert outcome.iteration_count == 2
    assert outcome.iterations[0].satisfaction.missing_observations == ("return_code present",)
    assert _has(outcome.iterations[1].directives, RemediationAction.CAPTURE_RETURN_CODE)


def test_changing_gap_between_rounds_is_treated_as_progress():
    steps = iter(
        (
            (SEARCH_ONLY, trace_with(BehaviorEvent.REPOSITORY_INDEXED)),
            ("テストを実行しました。", trace_with(BehaviorEvent.SUBPROCESS_STARTED)),
            ("simulation.py を作成しました。", trace_with(BehaviorEvent.FILE_WRITTEN)),
        )
    )

    def executor(request, directives):
        return next(steps)

    outcome = run_goal_loop(CREATE_REQUEST, executor)

    assert outcome.status == "GOAL_ACHIEVED"
    assert outcome.iteration_count == 3


def test_budget_exhaustion_is_distinct_from_stalling():
    steps = iter(
        (
            (SEARCH_ONLY, trace_with(BehaviorEvent.REPOSITORY_INDEXED)),
            ("テストを実行しました。", trace_with(BehaviorEvent.SUBPROCESS_STARTED)),
        )
    )

    def executor(request, directives):
        return next(steps)

    outcome = run_goal_loop(CREATE_REQUEST, executor, max_iterations=2)

    assert outcome.status == "ESCALATED_BUDGET_EXHAUSTED"
    assert outcome.iteration_count == 2


def test_remediation_vocabulary_is_closed_and_abstains_on_unknown_gaps():
    goal = extract_goal(CREATE_REQUEST)
    unknown = GoalSatisfaction(goal, False, ("QUANTUM_COHERENCE >= 1",), None, "stub")

    assert remediation_for(unknown) == ()
    assert unsupported_gaps(unknown) == ("QUANTUM_COHERENCE >= 1",)


def test_each_directive_names_the_gap_it_came_from():
    goal = extract_goal(CREATE_REQUEST)
    satisfaction = GoalSatisfaction(
        goal, False, ("FILE_WRITTEN >= 1",), "search_results_only", "stub"
    )

    directives = remediation_for(satisfaction)
    actions = {item.action for item in directives}

    assert actions == {RemediationAction.WRITE_ARTIFACT, RemediationAction.ADVANCE_BEYOND_SEARCH}
    assert {item.gap for item in directives} == {"FILE_WRITTEN >= 1", "search_results_only"}
    assert all(item.instruction for item in directives)


def test_format_loop_outcome_reports_every_iteration():
    rendered = format_loop_outcome(run_goal_loop(CREATE_REQUEST, _writes_when_told))

    assert "Goal Execution Loop (L8.12)" in rendered
    assert "GOAL_ACHIEVED" in rendered
    assert "WRITE_ARTIFACT" in rendered
    assert "missing `FILE_WRITTEN >= 1`" in rendered


def test_experiment_fixtures_all_meet_their_expected_outcome():
    report = run_experiment()

    assert all(item["status_ok"] for item in report["results"])
    assert all(item["iterations_ok"] for item in report["results"])
    assert report["metrics"]["goal_completion_rate"] == 1.0
    assert report["metrics"]["escalation_accuracy"] == 1.0
    assert report["metrics"]["recovery_rate"] == 1.0
    assert "Goal Execution Loop Experiment (L8.12)" in format_experiment(report)
