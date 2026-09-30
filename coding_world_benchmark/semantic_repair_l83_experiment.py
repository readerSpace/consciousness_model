"""L8.3: decompose a repair proposal into dependency-ordered, code-level steps.

A step is only useful to L8.2 if it names the file, the symbol and the
formalized operation to perform.  The decomposition therefore runs each
requested change through the L8.5 semantic compiler and keeps the grounded
result on the step itself.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .autonomous_patch_l82_experiment import PatchIntent
from .external_repair_l81_experiment import RepairInstruction, RepairPlan
from .semantic_change_compiler_l85_experiment import (
    ChangeOperation,
    CodeChangeIR,
    compile_semantic_changes,
    extract_semantic_requirements,
)


@dataclass(frozen=True)
class RepairStep:
    id: str
    objective: str
    depends_on: tuple[str, ...]
    target_files: tuple[str, ...]
    target_symbols: tuple[str, ...]
    operation_type: str
    expected_behavior: str
    test_requirements: tuple[str, ...]
    patch_intents: tuple[PatchIntent, ...]
    verification: tuple[str, ...]
    grounded: bool = False
    unresolved_reason: str = ""


@dataclass(frozen=True)
class RepairDecomposition:
    steps: tuple[RepairStep, ...]
    cycle_free: bool


def _ungrounded_changes(instruction: RepairInstruction, plan: RepairPlan) -> tuple[CodeChangeIR, ...]:
    """Requirement-level view used when no repository root is available."""
    fallback = plan.resolved_files[0] if plan.resolved_files else ""
    changes = []
    for requirement in extract_semantic_requirements(instruction.requested_changes):
        changes.append(CodeChangeIR(
            requirement.target_file_hint or fallback, requirement.artifact or None, requirement.operation,
            requirement.anchor or None, requirement.expected_behavior, requirement.members, False,
            "unresolved", "repository rootが指定されていないため接地していません。", (), 0.0, requirement,
        ))
    return tuple(changes)


def decompose_repair(
    instruction: RepairInstruction,
    plan: RepairPlan,
    root: Path | None = None,
) -> RepairDecomposition:
    changes = (
        compile_semantic_changes(root, instruction, plan).changes
        if root is not None
        else _ungrounded_changes(instruction, plan)
    )
    steps: list[RepairStep] = []
    previous: str | None = None
    for index, change in enumerate(changes, 1):
        step_id = f"step_{index}"
        is_test = change.operation is ChangeOperation.ADD_TEST
        dependencies = tuple(item.id for item in steps) if is_test else ((previous,) if previous else ())
        grounded = change.status == "compiled"
        intents = (
            (PatchIntent(change.file, change.symbol, change.operation.value, change.behavior,
                         instruction.constraints, change.evidence),)
            if grounded and change.file
            else ()
        )
        test_requirements = instruction.acceptance_tests if is_test else ()
        steps.append(RepairStep(
            id=step_id,
            objective=change.requirement.text,
            depends_on=dependencies,
            target_files=(change.file,) if change.file else (),
            target_symbols=(change.symbol,) if change.symbol else plan.resolved_symbols,
            operation_type=change.operation.value,
            expected_behavior=change.behavior,
            test_requirements=test_requirements,
            patch_intents=intents,
            verification=("compile", *instruction.acceptance_tests) if is_test else ("compile",),
            grounded=grounded,
            unresolved_reason=change.reason,
        ))
        previous = step_id
    return RepairDecomposition(tuple(steps), True)


def format_repair_decomposition(decomposition: RepairDecomposition) -> str:
    lines = ["## Semantic Repair Decomposition", "", f"【DAG】cycle_free=`{decomposition.cycle_free}`", ""]
    for step in decomposition.steps:
        depends = ", ".join(step.depends_on) if step.depends_on else "なし"
        lines.extend([
            f"### {step.id}",
            f"- objective: {step.objective}",
            f"- operation_type: `{step.operation_type}`",
            f"- target_files: `{', '.join(step.target_files) or '-'}`",
            f"- target_symbols: `{', '.join(step.target_symbols) or '-'}`",
            f"- expected_behavior: {step.expected_behavior}",
            f"- test_requirements: {', '.join(step.test_requirements) or 'なし'}",
            f"- depends_on: `{depends}`",
            f"- grounded: `{step.grounded}`" + (f" ({step.unresolved_reason})" if step.unresolved_reason else ""),
            "- verification: " + ", ".join(step.verification),
        ])
    if not decomposition.steps:
        lines.append("- 分解可能な変更案がありません。")
    return "\n".join(lines)
