"""L8.4: bounded repair retries with diagnosis and loop prevention."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Protocol

from .autonomous_patch_l82_experiment import AutonomousPatchResult, PatchSynthesizer, run_autonomous_patch_workflow
from .external_repair_l81_experiment import RepairInstruction, RepairPlan, RegressionVerifier, parse_repair_proposal, plan_repair


class FailureType(Enum):
    WRONG_TARGET = "WRONG_TARGET"
    INCOMPLETE_EDIT = "INCOMPLETE_EDIT"
    IMPORT_BREAKAGE = "IMPORT_BREAKAGE"
    ROUTING_REGRESSION = "ROUTING_REGRESSION"
    BEHAVIORAL_REGRESSION = "BEHAVIORAL_REGRESSION"
    TEST_EXPECTATION_MISMATCH = "TEST_EXPECTATION_MISMATCH"
    SYNTHESIS_UNAVAILABLE = "SYNTHESIS_UNAVAILABLE"


@dataclass(frozen=True)
class FailureDiagnosis:
    failure_type: FailureType
    evidence: tuple[str, ...]
    failed_tests: tuple[str, ...]
    confidence: float


@dataclass(frozen=True)
class RepairAttempt:
    attempt_id: int
    instruction: RepairInstruction
    failure_type: str | None
    failed_tests: tuple[str, ...]
    changed_symbols: tuple[str, ...]
    outcome: str


@dataclass(frozen=True)
class RepairHypothesis:
    statement: str
    next_action: str
    based_on: FailureType


@dataclass(frozen=True)
class AutonomousRepairResult:
    outcome: str
    attempts: tuple[RepairAttempt, ...]
    diagnoses: tuple[FailureDiagnosis, ...]
    hypotheses: tuple[RepairHypothesis, ...]
    final_patch: AutonomousPatchResult | None


class RepairStrategy(Protocol):
    def propose(self, instruction: RepairInstruction, plan: RepairPlan, diagnosis: FailureDiagnosis, attempts: tuple[RepairAttempt, ...]) -> RepairInstruction | None: ...


def diagnose_failure(result: AutonomousPatchResult) -> FailureDiagnosis | None:
    if result.status == "synthesis_unavailable":
        return FailureDiagnosis(FailureType.SYNTHESIS_UNAVAILABLE, (result.message,), (), .95)
    if result.status == "static_validation_failed":
        return FailureDiagnosis(FailureType.INCOMPLETE_EDIT, (result.message,), (), .92)
    if result.status in {"regression_failed", "temporary_apply_failed"}:
        output = result.regression.output if result.regression else result.message
        lower = output.lower()
        failure = FailureType.IMPORT_BREAKAGE if "importerror" in lower or "modulenotfounderror" in lower else FailureType.BEHAVIORAL_REGRESSION if "assert" in lower or "failed" in lower else FailureType.TEST_EXPECTATION_MISMATCH
        failed = tuple(line.strip() for line in output.splitlines() if "failed" in line.lower() or "error" in line.lower())[:8]
        return FailureDiagnosis(failure, (output[-1000:],), failed, .78)
    if result.status == "apply_failed":
        return FailureDiagnosis(FailureType.WRONG_TARGET, (result.message,), (), .9)
    return None


class NoRetryStrategy:
    def propose(self, instruction, plan, diagnosis, attempts):
        return None


def _signature(instruction: RepairInstruction, diagnosis: FailureDiagnosis) -> tuple[str, str, str]:
    return ("|".join(instruction.target_files + instruction.target_symbols), "|".join(instruction.requested_changes), diagnosis.failure_type.value)


def _synthesis_evidence(synthesizer: PatchSynthesizer | None) -> tuple[str, ...]:
    """Surface why the L8.5 compiler could not ground a requirement."""
    compilation = getattr(synthesizer, "last_compilation", None)
    if compilation is None:
        return ()
    return tuple(f"{item.operation.value} {item.file or '-'}: {item.reason}" for item in compilation.unresolved)


def run_autonomous_repair_loop(root: Path, proposal: str | RepairInstruction, strategy: RepairStrategy | None = None, verifier: RegressionVerifier | None = None, max_attempts: int = 3, synthesizer: PatchSynthesizer | None = None) -> AutonomousRepairResult:
    instruction = parse_repair_proposal(proposal) if isinstance(proposal, str) else proposal
    strategy = strategy or NoRetryStrategy()
    verifier = verifier or RegressionVerifier()
    attempts: list[RepairAttempt] = []
    diagnoses: list[FailureDiagnosis] = []
    hypotheses: list[RepairHypothesis] = []
    seen: set[tuple[str, str, str]] = set()
    final_patch = None
    for attempt_id in range(1, max_attempts + 1):
        plan = plan_repair(root, instruction)
        final_patch = run_autonomous_patch_workflow(root, instruction, plan, synthesizer=synthesizer, verifier=verifier)
        changed = tuple(intent.symbol or "" for intent in final_patch.intents)
        if final_patch.status == "accepted":
            attempts.append(RepairAttempt(attempt_id, instruction, None, (), changed, "accepted"))
            return AutonomousRepairResult("accepted", tuple(attempts), tuple(diagnoses), tuple(hypotheses), final_patch)
        diagnosis = diagnose_failure(final_patch)
        if diagnosis is None:
            break
        if diagnosis.failure_type is FailureType.SYNTHESIS_UNAVAILABLE:
            evidence = _synthesis_evidence(synthesizer)
            if evidence:
                diagnosis = FailureDiagnosis(diagnosis.failure_type, diagnosis.evidence + evidence, diagnosis.failed_tests, diagnosis.confidence)
        diagnoses.append(diagnosis)
        attempts.append(RepairAttempt(attempt_id, instruction, diagnosis.failure_type.value, diagnosis.failed_tests, changed, final_patch.status))
        signature = _signature(instruction, diagnosis)
        if signature in seen:
            return AutonomousRepairResult("repeated_failure", tuple(attempts), tuple(diagnoses), tuple(hypotheses), final_patch)
        seen.add(signature)
        hypotheses.append(RepairHypothesis(f"{diagnosis.failure_type.value} が発生したため、同じ差分は再試行しない。", "strategy_propose_next_instruction", diagnosis.failure_type))
        next_instruction = strategy.propose(instruction, plan, diagnosis, tuple(attempts))
        if next_instruction is None:
            return AutonomousRepairResult("blocked_after_diagnosis", tuple(attempts), tuple(diagnoses), tuple(hypotheses), final_patch)
        instruction = next_instruction
    return AutonomousRepairResult("max_attempts_reached", tuple(attempts), tuple(diagnoses), tuple(hypotheses), final_patch)


def format_autonomous_repair(result: AutonomousRepairResult) -> str:
    lines = ["## Autonomous Repair Loop", "", f"【結果】`{result.outcome}`", f"【試行回数】`{len(result.attempts)}`", "", "【試行履歴】"]
    lines.extend(f"- attempt {item.attempt_id}: `{item.outcome}` failure=`{item.failure_type or '-'}`" for item in result.attempts)
    lines.extend(["", "【診断】"])
    diagnoses = [f"- `{item.failure_type.value}` confidence={item.confidence:.2f}: {'; '.join(item.evidence)[:500]}" for item in result.diagnoses] or ["- なし"]
    lines.extend(diagnoses)
    return "\n".join(lines)
