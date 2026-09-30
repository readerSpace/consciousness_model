"""L8.9: extract goal semantics before choosing a concrete route.

The layer is intentionally small: it does not replace specialized routers.
It states what final world state the user asked for, then compiles that goal
into required actions and completion conditions.  Search can appear in the
task graph, but for CREATE/CHANGE/EXECUTE/VERIFY goals it is never sufficient
as the final behaviour.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re
from typing import Any


class GoalOperator(Enum):
    KNOW = "KNOW"
    EXPLAIN = "EXPLAIN"
    CREATE = "CREATE"
    CHANGE = "CHANGE"
    EXECUTE = "EXECUTE"
    VERIFY = "VERIFY"
    DISCOVER = "DISCOVER"
    DIAGNOSE = "DIAGNOSE"


@dataclass(frozen=True)
class GoalIR:
    operators: tuple[GoalOperator, ...]
    desired_state: str
    target: str
    desired_properties: tuple[str, ...]
    completion_conditions: tuple[str, ...]
    prohibited_early_exits: tuple[str, ...]
    confidence: float


@dataclass(frozen=True)
class TaskGraph:
    goal: GoalIR
    required_actions: tuple[str, ...]
    optional_actions: tuple[str, ...]
    terminal_conditions: tuple[str, ...]


@dataclass(frozen=True)
class GoalContract:
    goal: GoalIR
    required_observations: tuple[str, ...]
    forbidden_final_responses: tuple[str, ...]
    statement: str


@dataclass(frozen=True)
class GoalObservation:
    file_written_count: int
    subprocess_started_count: int
    path_moved_count: int
    repository_indexed_count: int
    return_code_recorded: bool
    verification_result_recorded: bool
    response_text: str


@dataclass(frozen=True)
class GoalSatisfaction:
    goal: GoalIR
    achieved: bool
    missing_observations: tuple[str, ...]
    forbidden_final_response: str | None
    statement: str


def _unique(items: list[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(item for item in items if item))


def extract_goal(text: str) -> GoalIR:
    lower = text.lower()
    operators: list[GoalOperator] = []
    if any(word in text for word in ("原因", "おかしい", "異常")) or any(word in lower for word in ("diagnose", "root cause", "why broken")):
        operators.append(GoalOperator.DIAGNOSE)
    if any(word in text for word in ("調べて", "検索して", "論文", "文献")) or any(word in lower for word in ("research", "search", "paper")):
        operators.append(GoalOperator.KNOW)
    if any(word in text for word in ("説明", "解説", "要約", "数式で")) or any(word in lower for word in ("explain", "summarize")):
        operators.append(GoalOperator.EXPLAIN)
    if any(word in text for word in ("作成して", "作って", "生成して", "実装して", "追加して")) or any(word in lower for word in ("create", "build", "generate", "implement", "add")):
        operators.append(GoalOperator.CREATE)
    if any(word in text for word in ("修正", "変更", "直して", "移動して", "移動させて", "削除して", "消して", "消去して")) or any(word in lower for word in ("fix", "modify", "change", "move", "delete", "remove", "relocate")):
        operators.append(GoalOperator.CHANGE)
    if any(word in text for word in ("実行して", "動かして", "走らせて", "テストして")) or any(word in lower for word in ("run", "execute", "test")):
        operators.append(GoalOperator.EXECUTE)
    if any(word in text for word in ("検証して", "検証する", "確かめて", "正しいか")) or any(word in lower for word in ("verify", "validate")):
        operators.append(GoalOperator.VERIFY)
    if any(word in text for word in ("発見", "未知", "探索")) or any(word in lower for word in ("discover", "explore")):
        operators.append(GoalOperator.DISCOVER)
    if not operators:
        operators.append(GoalOperator.KNOW)

    properties: list[str] = []
    if any(word in text for word in ("一様磁場", "磁場")) or "magnetic field" in lower:
        properties.append("uniform magnetic field" if "一様" in text or "uniform" in lower else "magnetic field")
    if any(word in text for word in ("荷電粒子", "粒子")) or "particle" in lower:
        properties.append("charged particle" if "荷電" in text or "charged" in lower else "particle")
    if any(word in text for word in ("シミュレーション",)) or "simulation" in lower:
        properties.append("simulation")
    if any(word in text for word in ("提案", "修正案")) or "proposal" in lower:
        properties.append("external proposal")
    if "taskspec" in lower or "expectedbehavior" in lower or "observedbehavior" in lower:
        properties.append("task contract")
    if any(word in text for word in ("フォルダ", "ディレクトリ")) or any(word in lower for word in ("folder", "directory")):
        properties.append("directory")

    target = _infer_target(text, tuple(properties))
    desired_state = _desired_state(operators, target)
    completion = _completion_conditions(operators, tuple(properties))
    prohibited = _prohibited_early_exits(operators)
    confidence = 0.95 if any(op in operators for op in (GoalOperator.CREATE, GoalOperator.CHANGE, GoalOperator.EXECUTE, GoalOperator.VERIFY)) else 0.75
    return GoalIR(_unique(operators), desired_state, target, _unique(properties), completion, prohibited, confidence)


def _infer_target(text: str, properties: tuple[str, ...]) -> str:
    lower = text.lower()
    if "simulation" in properties or "シミュレーション" in text:
        if "charged particle" in properties and any("magnetic" in item for item in properties):
            return "charged-particle uniform-magnetic-field simulation"
        return "simulation"
    file_match = re.search(r"([A-Za-z0-9_.-]+\.[A-Za-z0-9]+)", text)
    if file_match:
        return file_match.group(1)
    if "external proposal" in properties:
        return "external repair proposal"
    if "directory" in properties:
        return "directory target"
    if "readme" in lower:
        return "README"
    return "request target"


def _desired_state(operators: list[GoalOperator], target: str) -> str:
    if GoalOperator.CREATE in operators:
        return f"working artifact exists: {target}"
    if GoalOperator.CHANGE in operators:
        return f"target is changed to requested state: {target}"
    if GoalOperator.EXECUTE in operators:
        return f"execution result exists: {target}"
    if GoalOperator.VERIFY in operators:
        return f"verification verdict exists: {target}"
    if GoalOperator.EXPLAIN in operators:
        return f"understandable explanation exists: {target}"
    return f"knowledge answer exists: {target}"


def _completion_conditions(operators: list[GoalOperator], properties: tuple[str, ...]) -> tuple[str, ...]:
    conditions: list[str] = []
    if GoalOperator.CREATE in operators:
        conditions.extend(("artifact_created", "artifact_matches_requested_properties"))
        if "simulation" in properties:
            conditions.append("simulation_code_created")
    if GoalOperator.CHANGE in operators:
        conditions.extend(("file_modified", "change_matches_request"))
    if GoalOperator.EXECUTE in operators:
        conditions.extend(("subprocess_started", "return_code_recorded"))
    if GoalOperator.VERIFY in operators:
        conditions.append("verification_result_reported")
    if GoalOperator.EXPLAIN in operators:
        conditions.append("structured_explanation_returned")
    if not conditions:
        conditions.append("answer_returned")
    return _unique(conditions)


def _prohibited_early_exits(operators: list[GoalOperator]) -> tuple[str, ...]:
    if any(op in operators for op in (GoalOperator.CREATE, GoalOperator.CHANGE, GoalOperator.EXECUTE, GoalOperator.VERIFY)):
        return ("search_results_only", "file_overview_only", "memory_dump_only")
    if GoalOperator.EXPLAIN in operators:
        return ("file_overview_only",)
    return ()


def compile_task_graph(goal: GoalIR) -> TaskGraph:
    actions: list[str] = []
    optional: list[str] = []
    ops = set(goal.operators)
    if GoalOperator.KNOW in ops or GoalOperator.DISCOVER in ops:
        actions.append("resolve_knowledge_gap")
        optional.append("search_or_read_sources")
    if GoalOperator.DIAGNOSE in ops:
        actions.extend(("observe_failure", "diagnose_cause"))
    if GoalOperator.CREATE in ops:
        if "simulation" in goal.desired_properties:
            actions.extend(("formulate_model", "generate_code", "write_file"))
        else:
            actions.extend(("design_artifact", "write_file"))
    if GoalOperator.CHANGE in ops:
        actions.extend(("inspect_target", "modify_file"))
    if GoalOperator.EXECUTE in ops:
        actions.extend(("run_command", "capture_result"))
    if GoalOperator.VERIFY in ops:
        actions.extend(("run_or_check_verification", "compare_expected_observed"))
    if GoalOperator.EXPLAIN in ops and not actions:
        actions.append("produce_structured_explanation")
    if "artifact_created" in goal.completion_conditions and "verify_artifact" not in actions:
        optional.append("verify_artifact")
    return TaskGraph(goal, _unique(actions), _unique(optional), goal.completion_conditions)


def contract_from_goal(goal: GoalIR) -> GoalContract:
    required: list[str] = []
    if "artifact_created" in goal.completion_conditions:
        required.append("FILE_WRITTEN >= 1")
    if "file_modified" in goal.completion_conditions or "change_matches_request" in goal.completion_conditions:
        required.append("WORKSPACE_DIFF >= 1")
    if "subprocess_started" in goal.completion_conditions:
        required.append("SUBPROCESS_STARTED >= 1")
    if "return_code_recorded" in goal.completion_conditions:
        required.append("return_code present")
    if "verification_result_reported" in goal.completion_conditions:
        required.append("verification verdict present")
    if not required:
        required.append("response satisfies requested information need")
    statement = f"Goal `{goal.desired_state}` requires " + ", ".join(required)
    return GoalContract(goal, tuple(required), goal.prohibited_early_exits, statement)


def observe_goal_behavior(trace: Any, response_text: str) -> GoalObservation:
    events = tuple(getattr(trace, "events", ()))

    def count(name: str) -> int:
        return sum(1 for event in events if getattr(event, "value", str(event)) == name)

    return GoalObservation(
        file_written_count=count("FILE_WRITTEN"),
        subprocess_started_count=count("SUBPROCESS_STARTED"),
        path_moved_count=count("PATH_MOVED"),
        repository_indexed_count=count("REPOSITORY_INDEXED"),
        return_code_recorded=bool(re.search(r"return code\s*[`:]?\s*-?\d+|終了コード\s*:\s*-?\d+", response_text, re.I)),
        verification_result_recorded=bool(re.search(r"検証|verification|passed|failed|OK|FAILED", response_text, re.I)),
        response_text=response_text,
    )


def evaluate_goal_contract(contract: GoalContract, observation: GoalObservation) -> GoalSatisfaction:
    missing: list[str] = []
    for requirement in contract.required_observations:
        if requirement == "FILE_WRITTEN >= 1" and observation.file_written_count < 1:
            missing.append(requirement)
        elif requirement == "WORKSPACE_DIFF >= 1" and observation.file_written_count + observation.path_moved_count < 1:
            missing.append(requirement)
        elif requirement == "SUBPROCESS_STARTED >= 1" and observation.subprocess_started_count < 1:
            missing.append(requirement)
        elif requirement == "return_code present" and not observation.return_code_recorded:
            missing.append(requirement)
        elif requirement == "verification verdict present" and not observation.verification_result_recorded:
            missing.append(requirement)
    forbidden = _forbidden_response(contract.forbidden_final_responses, observation)
    achieved = not missing and forbidden is None
    if achieved:
        statement = f"GOAL_ACHIEVED: {contract.goal.desired_state}"
    else:
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if forbidden:
            details.append(f"forbidden final response `{forbidden}`")
        statement = f"GOAL_NOT_ACHIEVED: {contract.goal.desired_state} (" + "; ".join(details) + ")"
    return GoalSatisfaction(contract.goal, achieved, tuple(missing), forbidden, statement)


def _forbidden_response(forbidden: tuple[str, ...], observation: GoalObservation) -> str | None:
    text = observation.response_text
    if "file_overview_only" in forbidden and ("候補ファイル" in text or "ファイル概観" in text or "一致ファイル" in text):
        return "file_overview_only"
    if "memory_dump_only" in forbidden and text.strip().startswith("【再利用された知識】"):
        return "memory_dump_only"
    if "search_results_only" in forbidden and _looks_like_search_only(text):
        return "search_results_only"
    for item in forbidden:
        if item in {"search_results_only", "file_overview_only", "memory_dump_only"}:
            continue
    return None


def _looks_like_search_only(text: str) -> bool:
    if "ローカル要約" in text and ("一致ファイル" in text or "候補ファイル" in text):
        return True
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    raw_hits = [line for line in lines if re.search(r"[\w./\\-]+\.[A-Za-z]{1,5}:\d+:", line)]
    return len(lines) > 0 and len(raw_hits) / len(lines) >= 0.5


def format_goal_satisfaction(result: GoalSatisfaction) -> str:
    lines = ["## Goal Satisfaction", "", f"- achieved: `{result.achieved}`", f"- {result.statement}"]
    if result.missing_observations:
        lines.extend(["", "【Missing Observations】", *[f"- `{item}`" for item in result.missing_observations]])
    if result.forbidden_final_response:
        lines.extend(["", "【Forbidden Final Response】", f"- `{result.forbidden_final_response}`"])
    return "\n".join(lines)


def format_goal_analysis(goal: GoalIR, graph: TaskGraph | None = None, contract: GoalContract | None = None) -> str:
    graph = graph or compile_task_graph(goal)
    contract = contract or contract_from_goal(goal)
    properties = [f"  - {item}" for item in goal.desired_properties] or ["  - none"]
    prohibited = [f"  - `{item}`" for item in goal.prohibited_early_exits] or ["  - none"]
    lines = [
        "## Goal Semantics",
        "",
        "【GoalIR】",
        f"- operators: `{ ' -> '.join(operator.value for operator in goal.operators) }`",
        f"- desired_state: `{goal.desired_state}`",
        f"- target: `{goal.target}`",
        "- desired_properties:",
        *properties,
        "- completion_conditions:",
        *[f"  - `{item}`" for item in goal.completion_conditions],
        "- prohibited_early_exits:",
        *prohibited,
        "",
        "【TaskGraph】",
        *[f"{index}. `{action}`" for index, action in enumerate(graph.required_actions, 1)],
        "",
        "【GoalContract】",
        f"- {contract.statement}",
        *[f"- forbidden final response: `{item}`" for item in contract.forbidden_final_responses],
    ]
    return "\n".join(lines)
