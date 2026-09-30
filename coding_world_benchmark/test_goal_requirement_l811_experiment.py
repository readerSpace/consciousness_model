from coding_world_benchmark.goal_requirement_l811_experiment import (
    compile_goal_driven_plan,
    format_goal_driven_plan,
)


def _actions(plan):
    return tuple(slot.action for slot in plan.requirement_graph.requirements)


def _constraints(plan):
    return set().union(*(set(slot.constraints) for slot in plan.requirement_graph.requirements))


def test_create_scientific_simulation_request_gets_requirement_graph():
    plan = compile_goal_driven_plan("一様磁場中で荷電粒子が運動するシミュレーションを作成して")

    assert _actions(plan) == ("formulate_physics_model", "generate_simulation_code", "write_file")
    assert "uniform_magnetic_field" in _constraints(plan)
    assert "charged_particle" in _constraints(plan)
    assert "simulation" in _constraints(plan)
    assert "artifact_created" in _constraints(plan)
    assert plan.contract.required_observations == ("FILE_WRITTEN >= 1",)
    assert any(edge.relation == "feeds" for edge in plan.requirement_graph.edges)


def test_cross_language_create_request_keeps_canonical_requirements():
    japanese = compile_goal_driven_plan("一様磁場中で荷電粒子が運動するシミュレーションを作成して")
    english = compile_goal_driven_plan("Create a charged particle simulation in a uniform magnetic field")

    assert _actions(japanese) == _actions(english)
    assert {"uniform_magnetic_field", "charged_particle", "simulation"} <= _constraints(english)
    assert japanese.contract.required_observations == english.contract.required_observations


def test_composite_goal_orders_create_execute_verify_requirements():
    plan = compile_goal_driven_plan("シミュレーションを作成して実行して検証して")

    actions = _actions(plan)
    assert actions.index("write_file") < actions.index("resolve_execution_target")
    assert actions.index("run_command") < actions.index("capture_result")
    assert "compare_expected_observed" in actions
    assert ("FILE_WRITTEN >= 1", "SUBPROCESS_STARTED >= 1", "return_code present", "verification verdict present") == plan.contract.required_observations


def test_format_goal_driven_plan_contains_goal_and_requirement_graph():
    rendered = format_goal_driven_plan(compile_goal_driven_plan("READMEを要約して"))

    assert "Goal Semantics" in rendered
    assert "Requirement Graph" in rendered
    assert "produce_structured_explanation" in rendered
