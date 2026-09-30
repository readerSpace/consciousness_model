from pathlib import Path
import subprocess

from coding_world_benchmark.autonomous_patch_l82_experiment import run_autonomous_patch_workflow
from coding_world_benchmark.autonomous_repair_l84_experiment import FailureType, run_autonomous_repair_loop
from coding_world_benchmark.process_execution import combined_output, run_text
from coding_world_benchmark.external_repair_l81_experiment import (
    RegressionVerifier,
    parse_repair_proposal,
    plan_repair,
)
from coding_world_benchmark.semantic_change_compiler_l85_experiment import (
    ChangeOperation,
    SemanticPatchSynthesizer,
    compile_semantic_changes,
    extract_semantic_requirements,
    format_change_compilation,
    to_patch_intents,
)
from coding_world_benchmark.semantic_repair_l83_experiment import decompose_repair


def _verifier(passed: bool = True) -> RegressionVerifier:
    def run(command, cwd):
        return subprocess.CompletedProcess(command, 0 if passed else 1, "1 passed" if passed else "1 failed", "")

    return RegressionVerifier(run)


def _router_repo(tmp_path: Path) -> Path:
    (tmp_path / "coding_agent_app.py").write_text(
        "\n".join([
            '"""app"""',
            "from __future__ import annotations",
            "",
            "",
            "def legacy_intent_router(message):",
            "    return message.upper()",
            "",
            "",
            "def handle_request(message):",
            "    return legacy_intent_router(message)",
            "",
        ]),
        encoding="utf-8",
    )
    return tmp_path


# --------------------------------------------------------------------------
# stage 1: proposal sentences become semantic requirements
# --------------------------------------------------------------------------

def test_requirements_carry_operation_artifact_anchor_and_members():
    requirements = extract_semantic_requirements((
        "TaskIR型を追加",
        "action",
        "requires_execution",
        "LanguageToTaskParserを追加",
        "既存legacy_intent_routerの前にSemanticParserを配置",
        "add regression test test_language_task_parser.py",
    ))

    assert [item.operation for item in requirements] == [
        ChangeOperation.ADD_DATACLASS,
        ChangeOperation.ADD_CLASS,
        ChangeOperation.INSERT_ROUTE,
        ChangeOperation.ADD_TEST,
    ]
    assert requirements[0].artifact == "TaskIR"
    assert requirements[0].members == ("action", "requires_execution")
    assert requirements[1].artifact == "LanguageToTaskParser"
    assert requirements[2].anchor == "legacy_intent_router"
    assert requirements[2].artifact == "SemanticParser"


# --------------------------------------------------------------------------
# stage 2: requirements become repository-grounded CodeChangeIR
# --------------------------------------------------------------------------

def test_semantic_requirements_compile_to_code_change_ir(tmp_path: Path):
    root = _router_repo(tmp_path)
    instruction = parse_repair_proposal(
        """Problem: 自由文理解層がない
Target files: coding_agent_app.py
Suggested changes:
- TaskIR型を追加 (action, target, requires_execution)
- language_task_parser.py に LanguageToTaskParser を追加
- 既存legacy_intent_routerの前にSemanticParserを配置
Regression tests:
- test_coding_agent_app.py
"""
    )
    plan = plan_repair(root, instruction)

    compilation = compile_semantic_changes(root, instruction, plan)
    by_operation = {item.operation: item for item in compilation.changes}

    dataclass_ir = by_operation[ChangeOperation.ADD_DATACLASS]
    assert dataclass_ir.file == "coding_agent_app.py"
    assert dataclass_ir.symbol == "TaskIR"
    assert dataclass_ir.members == ("action", "target", "requires_execution")

    new_module = by_operation[ChangeOperation.ADD_CLASS]
    assert new_module.file == "language_task_parser.py"
    assert new_module.creates_file

    route = by_operation[ChangeOperation.INSERT_ROUTE]
    assert (route.file, route.symbol, route.anchor) == ("coding_agent_app.py", "handle_request", "legacy_intent_router")

    assert compilation.unresolved == ()
    assert {intent.operation for intent in to_patch_intents(compilation)} == {
        "ADD_DATACLASS", "ADD_CLASS", "INSERT_ROUTE",
    }
    assert "Code Change IR" in format_change_compilation(compilation)


def test_ungrounded_anchor_is_reported_instead_of_guessed(tmp_path: Path):
    root = _router_repo(tmp_path)
    instruction = parse_repair_proposal(
        """Problem: routing
Target files: coding_agent_app.py
Suggested changes:
- 既存unknown_routerの前にSemanticParserを配置
Regression tests:
- test_coding_agent_app.py
"""
    )

    compilation = compile_semantic_changes(root, instruction, plan_repair(root, instruction))

    assert compilation.compiled == ()
    assert "anchor" in compilation.unresolved[0].reason


# --------------------------------------------------------------------------
# stage 3: CodeChangeIR becomes a verified patch
# --------------------------------------------------------------------------

def test_semantic_patch_synthesizer_applies_dataclass_route_and_new_module(tmp_path: Path):
    root = _router_repo(tmp_path)
    instruction = parse_repair_proposal(
        """Problem: 自由文理解層がない
Target files: coding_agent_app.py
Suggested changes:
- TaskIR型を追加 (action, target, requires_execution)
- language_task_parser.py に LanguageToTaskParser を追加
- 既存legacy_intent_routerの前にSemanticParserを配置
Regression tests:
- test_coding_agent_app.py
"""
    )
    plan = plan_repair(root, instruction)

    result = run_autonomous_patch_workflow(root, instruction, plan, synthesizer=SemanticPatchSynthesizer(), verifier=_verifier(True))

    assert result.status == "accepted"
    app_source = (root / "coding_agent_app.py").read_text(encoding="utf-8")
    assert "class TaskIR" in app_source
    assert "requires_execution: str" in app_source
    assert "def semantic_parser_route(message):" in app_source
    route_line = app_source.index("_l85_routed = semantic_parser_route(message)")
    assert route_line < app_source.index("return legacy_intent_router(message)")
    assert "class LanguageToTaskParser" in (root / "language_task_parser.py").read_text(encoding="utf-8")


def test_generated_route_preserves_existing_behaviour(tmp_path: Path):
    root = _router_repo(tmp_path)
    instruction = parse_repair_proposal(
        """Problem: routing order
Target files: coding_agent_app.py
Suggested changes:
- 既存legacy_intent_routerの前にSemanticParserを配置
Regression tests:
- test_coding_agent_app.py
"""
    )
    plan = plan_repair(root, instruction)
    run_autonomous_patch_workflow(root, instruction, plan, synthesizer=SemanticPatchSynthesizer(), verifier=_verifier(True))

    namespace: dict[str, object] = {}
    exec(compile((root / "coding_agent_app.py").read_text(encoding="utf-8"), "app", "exec"), namespace)

    assert namespace["handle_request"]("ok") == "OK"


def test_enum_member_is_inserted_into_existing_enum(tmp_path: Path):
    (tmp_path / "failures.py").write_text(
        "from enum import Enum\n\n\nclass FailureType(Enum):\n    WRONG_TARGET = \"WRONG_TARGET\"\n",
        encoding="utf-8",
    )
    instruction = parse_repair_proposal(
        """Problem: 診断種別が足りない
Target files: failures.py
Suggested changes:
- FailureType enum に member SEMANTIC_GAP を追加
Regression tests:
- test_failures.py
"""
    )
    plan = plan_repair(tmp_path, instruction)

    result = run_autonomous_patch_workflow(tmp_path, instruction, plan, synthesizer=SemanticPatchSynthesizer(), verifier=_verifier(True))

    assert result.status == "accepted"
    source = (tmp_path / "failures.py").read_text(encoding="utf-8")
    assert 'SEMANTIC_GAP = "SEMANTIC_GAP"' in source
    assert source.index("WRONG_TARGET") < source.index("SEMANTIC_GAP")


def test_generated_regression_test_guards_the_new_symbol(tmp_path: Path):
    root = _router_repo(tmp_path)
    instruction = parse_repair_proposal(
        """Problem: 回帰が無い
Target files: coding_agent_app.py
Suggested changes:
- TaskIR型を追加 (action, target)
- add regression test test_task_ir.py
Regression tests:
- test_task_ir.py
"""
    )
    plan = plan_repair(root, instruction)

    result = run_autonomous_patch_workflow(root, instruction, plan, synthesizer=SemanticPatchSynthesizer(), verifier=_verifier(True))

    assert result.status == "accepted"
    generated = root / "test_task_ir.py"
    assert generated.is_file()
    completed = run_text(["python", "-m", "pytest", "-q", str(generated)], cwd=root)
    assert completed.returncode == 0, combined_output(completed)


def test_regression_failure_rolls_back_created_files(tmp_path: Path):
    root = _router_repo(tmp_path)
    instruction = parse_repair_proposal(
        """Problem: 新規モジュール
Target files: coding_agent_app.py
Suggested changes:
- language_task_parser.py に LanguageToTaskParser を追加
Regression tests:
- test_coding_agent_app.py
"""
    )
    plan = plan_repair(root, instruction)

    result = run_autonomous_patch_workflow(root, instruction, plan, synthesizer=SemanticPatchSynthesizer(), verifier=_verifier(False))

    assert result.status == "regression_failed"
    assert not (root / "language_task_parser.py").exists()


# --------------------------------------------------------------------------
# integration: L8.3 steps and the L8.4 loop
# --------------------------------------------------------------------------

def test_decomposition_steps_are_code_level(tmp_path: Path):
    root = _router_repo(tmp_path)
    instruction = parse_repair_proposal(
        """Problem: 自由文理解層がない
Target files: coding_agent_app.py
Suggested changes:
- TaskIR型を追加 (action, target)
- language_task_parser.py に LanguageToTaskParser を追加
- 既存legacy_intent_routerの前にSemanticParserを配置
- add regression test test_language_task_parser.py
Regression tests:
- test_language_task_parser.py
"""
    )
    plan = plan_repair(root, instruction)

    decomposition = decompose_repair(instruction, plan, root)
    steps = decomposition.steps

    assert [step.operation_type for step in steps] == ["ADD_DATACLASS", "ADD_CLASS", "INSERT_ROUTE", "ADD_TEST"]
    assert steps[0].target_files == ("coding_agent_app.py",)
    assert steps[1].target_files == ("language_task_parser.py",)
    assert steps[2].target_symbols == ("handle_request",)
    assert steps[3].depends_on == ("step_1", "step_2", "step_3")
    assert steps[3].test_requirements == ("test_language_task_parser.py",)
    assert all(step.grounded for step in steps)
    assert all(step.patch_intents for step in steps)


def test_repair_loop_accepts_semantic_change_without_synthesis_unavailable(tmp_path: Path):
    root = _router_repo(tmp_path)
    proposal = """Problem: 自由文理解層がない
Target files: coding_agent_app.py
Suggested changes:
- TaskIR型を追加 (action, target, requires_execution)
- 既存legacy_intent_routerの前にSemanticParserを配置
Regression tests:
- test_coding_agent_app.py
"""

    result = run_autonomous_repair_loop(root, proposal, verifier=_verifier(True))

    assert result.outcome == "accepted"
    assert not any(item.failure_type is FailureType.SYNTHESIS_UNAVAILABLE for item in result.diagnoses)


def test_loop_reports_why_grounding_failed(tmp_path: Path):
    root = _router_repo(tmp_path)
    proposal = """Problem: routing
Target files: coding_agent_app.py
Suggested changes:
- 既存unknown_routerの前にSemanticParserを配置
Regression tests:
- test_coding_agent_app.py
"""
    synthesizer = SemanticPatchSynthesizer()

    result = run_autonomous_repair_loop(root, proposal, verifier=_verifier(True), synthesizer=synthesizer)

    assert result.outcome == "blocked_after_diagnosis"
    assert result.diagnoses[0].failure_type is FailureType.SYNTHESIS_UNAVAILABLE
    assert any("anchor" in item for item in result.diagnoses[0].evidence)
