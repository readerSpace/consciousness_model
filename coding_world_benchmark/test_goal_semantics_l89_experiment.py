from coding_world_benchmark.goal_semantics_l89_experiment import (
    GoalOperator,
    contract_from_goal,
    compile_task_graph,
    evaluate_goal_contract,
    extract_goal,
    format_goal_analysis,
    observe_goal_behavior,
)


def test_create_simulation_goal_requires_artifact_not_search_result():
    goal = extract_goal("一様磁場中で荷電粒子が運動するシミュレーションを作成して")
    graph = compile_task_graph(goal)
    contract = contract_from_goal(goal)

    assert GoalOperator.CREATE in goal.operators
    assert goal.desired_state.startswith("working artifact exists")
    assert "uniform magnetic field" in goal.desired_properties
    assert "charged particle" in goal.desired_properties
    assert "artifact_created" in goal.completion_conditions
    assert "search_results_only" in goal.prohibited_early_exits
    assert "formulate_model" in graph.required_actions
    assert "write_file" in graph.required_actions
    assert "FILE_WRITTEN >= 1" in contract.required_observations


def test_composite_goal_preserves_ordered_actions():
    goal = extract_goal("調べてから実装してテストして")
    graph = compile_task_graph(goal)

    assert goal.operators[:3] == (GoalOperator.KNOW, GoalOperator.CREATE, GoalOperator.EXECUTE)
    assert "resolve_knowledge_gap" in graph.required_actions
    assert "write_file" in graph.required_actions
    assert "run_command" in graph.required_actions
    assert "return_code_recorded" in goal.completion_conditions


def test_diagnose_change_verify_goal():
    goal = extract_goal("この結果がおかしい原因を調べて直して検証して")
    graph = compile_task_graph(goal)
    rendered = format_goal_analysis(goal, graph, contract_from_goal(goal))

    assert GoalOperator.DIAGNOSE in goal.operators
    assert GoalOperator.CHANGE in goal.operators
    assert GoalOperator.VERIFY in goal.operators
    assert "diagnose_cause" in graph.required_actions
    assert "modify_file" in graph.required_actions
    assert "compare_expected_observed" in graph.required_actions
    assert "Goal Semantics" in rendered


def test_goal_contract_rejects_search_only_create_response():
    goal = extract_goal("一様磁場中で荷電粒子が運動するシミュレーションを作成して")

    class _Trace:
        events = []

    observation = observe_goal_behavior(_Trace(), "ローカル要約（検索語一致なし・ファイル概観）\n候補ファイル: 0件")
    satisfaction = evaluate_goal_contract(contract_from_goal(goal), observation)

    assert not satisfaction.achieved
    assert "FILE_WRITTEN >= 1" in satisfaction.missing_observations
    assert satisfaction.forbidden_final_response == "file_overview_only"


def test_goal_contract_accepts_create_when_file_is_written():
    goal = extract_goal("一様磁場中で荷電粒子が運動するシミュレーションを作成して")

    class _Event:
        value = "FILE_WRITTEN"

    class _Trace:
        events = [_Event()]

    satisfaction = evaluate_goal_contract(contract_from_goal(goal), observe_goal_behavior(_Trace(), "生成ファイル: charged_particle_uniform_B.py"))

    assert satisfaction.achieved


def test_move_directory_goal_requires_workspace_diff():
    goal = extract_goal("SU2とSU3の検証フォルダをまとめて移動して")
    contract = contract_from_goal(goal)

    assert GoalOperator.CHANGE in goal.operators
    assert "WORKSPACE_DIFF >= 1" in contract.required_observations


def test_move_directory_goal_accepts_path_move_event():
    goal = extract_goal("SU2とSU3の検証フォルダをまとめて移動して")

    class _Event:
        value = "PATH_MOVED"

    class _Trace:
        events = [_Event()]

    satisfaction = evaluate_goal_contract(contract_from_goal(goal), observe_goal_behavior(_Trace(), "移動しました: su2_su3"))

    assert satisfaction.achieved
