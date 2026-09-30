from pathlib import Path
import subprocess

from coding_world_benchmark.external_repair_l81_experiment import (
    NoOpPatchExecutor,
    ProposalSentenceRole,
    ProposalStatus,
    RegressionVerifier,
    analyze_repository_impact,
    build_requirement_graph,
    classify_proposal_sentences,
    extract_requirement_slots,
    format_repair_workflow,
    detect_repair_intent,
    extract_implementation_items,
    extract_implementation_requirements,
    format_recognized_task_spec,
    format_requirement_graph,
    parse_external_repair_request,
    parse_repair_proposal,
    plan_repair,
    recognized_repair_task_spec,
    run_repair_workflow,
)


PROPOSAL = """Problem: algorithm summary returns search results only
Target files: algorithm_summary_l80_experiment.py
Target symbols: AlgorithmSummarizer
Suggested changes:
- route ALGORITHM_SUMMARY before generic search
- compress calls and dataflow into stages
Constraints:
- keep local and API-free
Regression tests:
- test_algorithm_summary_l80_experiment.py
"""


def test_repair_proposal_is_structured_and_resolved_against_repo(tmp_path: Path):
    (tmp_path / "algorithm_summary_l80_experiment.py").write_text("class AlgorithmSummarizer:\n    pass\n", encoding="utf-8")
    instruction = parse_repair_proposal(PROPOSAL)
    plan = plan_repair(tmp_path, instruction)

    assert instruction.problem.startswith("algorithm summary")
    assert instruction.target_files == ("algorithm_summary_l80_experiment.py",)
    assert instruction.target_symbols == ("AlgorithmSummarizer",)
    assert plan.status is ProposalStatus.APPLICABLE
    assert "algorithm_summary_l80_experiment.py" in plan.resolved_files
    assert any(item.endswith(".AlgorithmSummarizer") for item in plan.resolved_symbols)


def test_unresolved_proposal_is_not_applied_or_verified(tmp_path: Path):
    result = run_repair_workflow(tmp_path, PROPOSAL)

    assert result.plan.status is ProposalStatus.PARTIAL
    assert not result.patch.applied
    assert result.verification is None
    assert "RepairPlan" in format_repair_workflow(result)


def test_regression_verifier_uses_injected_runner(tmp_path: Path):
    seen = {}

    def runner(command, cwd):
        seen["command"] = command
        seen["cwd"] = cwd
        return subprocess.CompletedProcess(command, 0, "2 passed", "")

    result = RegressionVerifier(runner).verify(tmp_path, ("test_algorithm_summary_l80_experiment.py",))

    assert result.passed
    assert "pytest" in result.command
    assert seen["cwd"] == tmp_path


def test_structured_chatgpt_proposal_has_high_repair_intent_score():
    evidence = detect_repair_intent("""L8.3: Semantic Repair Decomposition
PatchIntentを追加し、routerのfallbackを変更する。
Regression tests: add multi-step routing repair
""")

    assert evidence.intent == "EXTERNAL_REPAIR_PROPOSAL"
    assert evidence.score >= 0.72
    assert evidence.reasons


def test_implementation_wrapper_extracts_proposal_body_and_items():
    request = parse_external_repair_request("""以下の提案から実装項目をリストアップし実装して
L8.6: Self-Modification Benchmark
1. 既知欠陥fixtureを追加
2. failure detectorを追加
3. regression testを自動実行
""")

    assert request.instruction == "implement"
    assert "Self-Modification Benchmark" in request.proposal_body
    assert len(request.implementation_items) >= 4
    assert any("fixture" in item for item in request.implementation_items)
    assert extract_implementation_items(request.proposal_body) == request.implementation_items


def test_task_contract_proposal_is_recognized_as_repair_task_spec():
    text = """以下の提案から実装項目をリストアップし実装して
TaskSpecを画面に表示する
Expected: FILE_MODIFICATION >= 1
Observed: FILE_MODIFICATION = 0
Behavior Monitorで不一致なら異常として扱う
Regression tests:
- TaskSpec preview
"""
    evidence = detect_repair_intent(text)
    request = parse_external_repair_request(text)
    instruction = parse_repair_proposal(request.proposal_body)
    spec = recognized_repair_task_spec(request, instruction)
    rendered = format_recognized_task_spec(spec)

    assert evidence.score >= 0.72
    assert spec.intent == "IMPLEMENT_PROPOSAL"
    assert "task contract visualization" in spec.proposal_topics
    assert "expected behavior generation" in spec.proposal_topics
    assert spec.memory_policy == "task_contract_only"
    assert "Recognized TaskSpec" in rendered


def test_change_requirement_extractor_filters_descriptions_from_goal_loop_proposal():
    proposal = """L8.10: Goal-Directed Continuation
Goalを満たすまで継続する手段になります。
1. Goal未達時に missing contract を抽出する
2. missing contract から next action を決める
3. TaskGraphを途中状態から再計画する
4. Goal達成まで execute → observe → evaluate を反復する
5. max steps / budget / loop detection を追加する
6. premature termination をテストする
"""

    classified = classify_proposal_sentences(proposal)
    requirements = extract_implementation_requirements(proposal)
    items = extract_implementation_items(proposal)

    assert any(sentence.text == "Goalを満たすまで継続する手段になります。" and sentence.role is ProposalSentenceRole.DESCRIPTION for sentence in classified)
    assert "Goalを満たすまで継続する手段になります。" not in requirements
    assert any("missing contract" in item for item in requirements)
    assert any("next action" in item for item in requirements)
    assert any("TaskGraph" in item for item in requirements)
    assert any("premature termination" in item for item in items)


def test_repository_impact_analysis_runs_before_operation_selection(tmp_path: Path):
    (tmp_path / "goal_semantics_l89_experiment.py").write_text(
        "class GoalContract:\n    pass\n\n"
        "def evaluate_goal_contract(contract, observation):\n    return None\n",
        encoding="utf-8",
    )
    (tmp_path / "behavior_monitor_l87_experiment.py").write_text(
        "class BehaviorAudit:\n    pass\n\n"
        "def audit_request(root, request, handler):\n    return None\n",
        encoding="utf-8",
    )
    requirements = (
        "Goal未達時に missing contract を抽出する",
        "missing contract から next action を決める",
        "ExpectedBehavior と ObservedBehavior を比較する",
    )

    impacts = analyze_repository_impact(tmp_path, requirements)

    assert impacts[0].capability == "GOAL_DIRECTED_REPLANNING"
    assert any("goal_semantics" in file for file in impacts[0].likely_files)
    assert any("GoalContract" in symbol or "evaluate_goal_contract" in symbol for symbol in impacts[0].likely_symbols)
    assert impacts[2].capability == "BEHAVIOR_MONITORING"
    assert "before selecting operation_type" in impacts[0].strategy


def test_requirement_slots_normalize_paraphrases_to_the_same_requirement():
    variants = (
        "Goal未達時に missing contract を抽出する",
        "Goalが満たされていなければ欠けたpostconditionを特定する",
        "完了条件の未充足部分を抽出する",
    )

    slots = extract_requirement_slots(variants)

    assert {slot.action for slot in slots} == {"derive_missing_contract"}
    assert {slot.object for slot in slots} == {"GoalContract"}
    assert len({slot.requirement_id for slot in slots}) <= 2
    assert slots[0].confidence >= 0.9


def test_requirement_graph_binds_goal_replanning_sequence():
    proposal = """L8.10: Requirement Graph
1. Goal未達時に missing contract を抽出する
2. missing contract から next action を決める
3. TaskGraphを途中状態から再計画する
4. Goal達成まで execute → observe → evaluate を反復する
5. max steps / budget / loop detection を追加する
6. premature termination をテストする
"""

    graph = build_requirement_graph(proposal)
    rendered = format_requirement_graph(graph)

    actions = {slot.action for slot in graph.requirements}
    assert {"derive_missing_contract", "plan_next_action", "replan_taskgraph", "iterate_until_goal_achieved", "add_loop_guard", "add_regression_test"} <= actions
    assert any(edge.relation == "enables" for edge in graph.edges)
    assert any(edge.relation == "bounds" for edge in graph.edges)
    assert "Requirement Graph" in rendered
    assert "bounded_iterations" in rendered


def test_repository_impact_uses_semantic_slots_for_paraphrased_goal_contract(tmp_path: Path):
    (tmp_path / "goal_semantics_l89_experiment.py").write_text(
        "class GoalContract:\n    pass\n\n"
        "def evaluate_goal_contract(contract, observation):\n    return None\n",
        encoding="utf-8",
    )
    requirements = ("Goalが満たされていなければ欠けたpostconditionを特定する",)

    impacts = analyze_repository_impact(tmp_path, requirements)

    assert impacts[0].capability == "GOAL_DIRECTED_REPLANNING"
    assert any(item.startswith("slot:") for item in impacts[0].evidence)
    assert any("goal_semantics" in file for file in impacts[0].likely_files)
