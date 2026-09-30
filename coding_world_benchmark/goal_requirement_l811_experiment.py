"""L8.11: compile ordinary user goals into requirement graphs.

L8.10 made external repair proposals inspectable as semantic slots.  This
module lifts the same representation above the ordinary task router: a normal
user request is first parsed as ``GoalIR``, then converted into a
``RequirementGraph`` that can feed ``TaskGraph`` and ``GoalContract``.
"""
from __future__ import annotations

from dataclasses import dataclass

from .external_repair_l81_experiment import RequirementEdge, RequirementGraph, RequirementSlot, format_requirement_graph
from .goal_semantics_l89_experiment import (
    GoalContract,
    GoalIR,
    GoalOperator,
    TaskGraph,
    compile_task_graph,
    contract_from_goal,
    extract_goal,
    format_goal_analysis,
)


@dataclass(frozen=True)
class GoalDrivenPlan:
    request: str
    goal: GoalIR
    requirement_graph: RequirementGraph
    task_graph: TaskGraph
    contract: GoalContract


def compile_goal_driven_plan(request: str) -> GoalDrivenPlan:
    goal = extract_goal(request)
    requirement_graph = requirement_graph_from_goal(goal)
    task_graph = compile_task_graph(goal)
    contract = contract_from_goal(goal)
    return GoalDrivenPlan(request, goal, requirement_graph, task_graph, contract)


def requirement_graph_from_goal(goal: GoalIR) -> RequirementGraph:
    slots = _slots_from_goal(goal)
    by_action = {slot.action: slot for slot in slots}
    edges: list[RequirementEdge] = []
    for source, relation, target in _goal_edges(goal):
        if source in by_action and target in by_action:
            edges.append(RequirementEdge(by_action[source].requirement_id, relation, by_action[target].requirement_id))
    return RequirementGraph(slots, tuple(edges))


def _slots_from_goal(goal: GoalIR) -> tuple[RequirementSlot, ...]:
    slots: list[RequirementSlot] = []
    ops = set(goal.operators)
    constraints = _goal_constraints(goal)

    if GoalOperator.KNOW in ops or GoalOperator.DISCOVER in ops:
        slots.append(_slot(goal, "resolve_knowledge_gap", "KnowledgeGap", "knowledge_answered", constraints))
    if GoalOperator.DIAGNOSE in ops:
        slots.extend((
            _slot(goal, "observe_failure", "ObservedBehavior", "failure_visible", constraints),
            _slot(goal, "diagnose_cause", "FailureDiagnosis", "repairable_cause_identified", constraints),
        ))
    if GoalOperator.CREATE in ops:
        if "simulation" in goal.desired_properties:
            slots.extend((
                _slot(goal, "formulate_physics_model", "MathematicalModel", "simulation_specified", constraints),
                _slot(goal, "generate_simulation_code", "SimulationSource", "code_generated", constraints),
                _slot(goal, "write_file", "WorkspaceArtifact", "artifact_created", constraints),
            ))
        else:
            slots.extend((
                _slot(goal, "design_artifact", "ArtifactSpec", "artifact_specified", constraints),
                _slot(goal, "write_file", "WorkspaceArtifact", "artifact_created", constraints),
            ))
    if GoalOperator.CHANGE in ops:
        slots.extend((
            _slot(goal, "inspect_target", "WorkspaceTarget", "target_understood", constraints),
            _slot(goal, "apply_change", "WorkspaceDiff", "change_matches_request", constraints),
        ))
    if GoalOperator.EXECUTE in ops:
        slots.extend((
            _slot(goal, "resolve_execution_target", "ExecutionTarget", "command_planned", constraints),
            _slot(goal, "run_command", "ProcessExecution", "process_started", constraints),
            _slot(goal, "capture_result", "ExecutionResult", "return_code_recorded", constraints),
        ))
    if GoalOperator.VERIFY in ops:
        slots.extend((
            _slot(goal, "run_verification", "VerificationProcedure", "verification_performed", constraints),
            _slot(goal, "compare_expected_observed", "BehaviorContract", "verification_result_reported", constraints),
        ))
    if GoalOperator.EXPLAIN in ops and not slots:
        slots.append(_slot(goal, "produce_structured_explanation", "StructuredAnswer", "information_need_satisfied", constraints))

    return tuple(dict.fromkeys(slots))


def _goal_constraints(goal: GoalIR) -> tuple[str, ...]:
    constraints: list[str] = []
    for item in goal.desired_properties:
        normalized = item.strip().lower().replace(" ", "_").replace("-", "_")
        if normalized:
            constraints.append(normalized)
    for item in goal.completion_conditions:
        constraints.append(item)
    return tuple(dict.fromkeys(constraints))


def _slot(goal: GoalIR, action: str, obj: str, effect: str, constraints: tuple[str, ...]) -> RequirementSlot:
    trigger = _trigger_for_goal(goal)
    requirement_id = f"{trigger}:{action}:{obj}".lower()
    return RequirementSlot(
        requirement_id=requirement_id,
        source_text=goal.desired_state,
        trigger=trigger,
        action=action,
        object=obj,
        effect=effect,
        constraints=constraints,
        metrics=(),
        confidence=goal.confidence,
    )


def _trigger_for_goal(goal: GoalIR) -> str:
    if GoalOperator.CREATE in goal.operators:
        return "user_requested_create"
    if GoalOperator.CHANGE in goal.operators:
        return "user_requested_change"
    if GoalOperator.EXECUTE in goal.operators:
        return "user_requested_execute"
    if GoalOperator.VERIFY in goal.operators:
        return "user_requested_verify"
    if GoalOperator.EXPLAIN in goal.operators:
        return "user_requested_explain"
    return "user_requested_knowledge"


def _goal_edges(goal: GoalIR) -> tuple[tuple[str, str, str], ...]:
    ops = set(goal.operators)
    edges: list[tuple[str, str, str]] = []
    if GoalOperator.KNOW in ops and GoalOperator.CREATE in ops:
        edges.append(("resolve_knowledge_gap", "feeds", "formulate_physics_model" if "simulation" in goal.desired_properties else "design_artifact"))
    if GoalOperator.DIAGNOSE in ops:
        edges.append(("observe_failure", "feeds", "diagnose_cause"))
    if GoalOperator.CREATE in ops:
        if "simulation" in goal.desired_properties:
            edges.extend((
                ("formulate_physics_model", "feeds", "generate_simulation_code"),
                ("generate_simulation_code", "feeds", "write_file"),
            ))
        else:
            edges.append(("design_artifact", "feeds", "write_file"))
    if GoalOperator.CHANGE in ops:
        edges.append(("inspect_target", "feeds", "apply_change"))
    if GoalOperator.EXECUTE in ops:
        edges.extend((
            ("resolve_execution_target", "feeds", "run_command"),
            ("run_command", "feeds", "capture_result"),
        ))
    if GoalOperator.VERIFY in ops:
        edges.append(("run_verification", "feeds", "compare_expected_observed"))
    if GoalOperator.CREATE in ops and GoalOperator.EXECUTE in ops:
        edges.append(("write_file", "feeds", "resolve_execution_target"))
    if GoalOperator.EXECUTE in ops and GoalOperator.VERIFY in ops:
        edges.append(("capture_result", "feeds", "compare_expected_observed"))
    return tuple(edges)


def format_goal_driven_plan(plan: GoalDrivenPlan) -> str:
    return "\n\n".join((
        format_goal_analysis(plan.goal, plan.task_graph, plan.contract),
        format_requirement_graph(plan.requirement_graph),
    ))
