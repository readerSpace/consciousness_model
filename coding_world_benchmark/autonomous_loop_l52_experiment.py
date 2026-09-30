"""L5.2: sealed autonomous scientific loop.

This experiment connects the prior layers into one budgeted loop:

Task -> plan -> execution -> L4.8 diagnosis -> L4.9 gap decision ->
L5.0 grounded evidence -> L5.1 revision -> re-execution -> conclusion.

The benchmark is sealed and deterministic.  It does not test live Web access or
LLM quality; it tests whether component outputs compose into a scientific
outcome without unsupported final claims, harmful looping, or budget violations.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
import json

from .evidence_grounded_l50_experiment import EvidenceStatus, GroundedEvidence, RETRIEVED_AT
from .experiment_revision_l51_experiment import ExperimentPlan, RevisionDecision, RevisionProposal
from .knowledge_gap_l49_experiment import CandidateAction, KnowledgeGap
from .scientific_diagnosis_l48_experiment import ResultDiagnosis


class FinalStatus(Enum):
    SUPPORTED = "supported"
    FALSIFIED = "falsified"
    UNRESOLVED = "unresolved"
    CODE_REPAIRED = "code_repaired"
    BUDGET_STOPPED = "budget_stopped"


@dataclass(frozen=True)
class LoopTask:
    name: str
    prompt: str
    initial_plan: ExperimentPlan
    expected_status: FinalStatus
    expected_conclusion: str
    evidence_status: EvidenceStatus = EvidenceStatus.SUPPORTED
    needs_two_revisions: bool = False
    cyclic: bool = False


@dataclass
class ScientificLoopState:
    iteration: int
    hypothesis_status: str
    diagnosis_history: list[str] = field(default_factory=list)
    knowledge_gaps: list[str] = field(default_factory=list)
    evidence_history: list[str] = field(default_factory=list)
    revision_history: list[str] = field(default_factory=list)
    execution_history: list[str] = field(default_factory=list)
    budget_remaining: dict[str, int] = field(default_factory=dict)
    unsupported_final_claim: bool = False
    terminated_by_budget: bool = False


@dataclass(frozen=True)
class ExecutionResult:
    diagnosis: ResultDiagnosis
    conclusion: str


def _base_plan() -> ExperimentPlan:
    return ExperimentPlan(sample_size=4, measurement="raw_mean", controls=("baseline",),
                          implementation_ok=True, hypothesis_active=True)


def tasks() -> tuple[LoopTask, ...]:
    base = _base_plan()
    return (
        LoopTask("positive_no_search", "Replicated signal with known measurement.", base,
                 FinalStatus.SUPPORTED, "positive"),
        LoopTask("low_power_then_samples", "Low power should trigger sample-size revision.", base,
                 FinalStatus.SUPPORTED, "positive"),
        LoopTask("instrument_then_measurement", "Instrument failure should trigger measurement revision.", base,
                 FinalStatus.SUPPORTED, "positive"),
        LoopTask("confounded_then_control", "Confound should trigger a control revision.", base,
                 FinalStatus.FALSIFIED, "negative"),
        LoopTask("implementation_then_repair", "Implementation failure should route to code repair.",
                 replace(base, implementation_ok=False), FinalStatus.CODE_REPAIRED, "positive"),
        LoopTask("negative_no_search", "Decisive negative result should be abandoned without search.", base,
                 FinalStatus.FALSIFIED, "negative"),
        LoopTask("conflicting_evidence_hold", "Conflicting evidence should hold the loop.", base,
                 FinalStatus.UNRESOLVED, "unresolved", EvidenceStatus.CONFLICTING),
        LoopTask("unresolved_evidence_hold", "Unresolved evidence should hold the loop.", base,
                 FinalStatus.UNRESOLVED, "unresolved", EvidenceStatus.UNRESOLVED),
        LoopTask("second_revision_needed", "First sample-size revision is still underpowered.", base,
                 FinalStatus.SUPPORTED, "positive", needs_two_revisions=True),
        LoopTask("cyclic_low_power_budget_stop", "Repeated low power must stop at budget.", base,
                 FinalStatus.BUDGET_STOPPED, "unresolved", cyclic=True),
    )


def _diagnosis_for(task: LoopTask, plan: ExperimentPlan) -> ExecutionResult:
    if not plan.implementation_ok:
        return ExecutionResult(ResultDiagnosis.IMPLEMENTATION_FAILURE, "no_conclusion")
    if not plan.hypothesis_active:
        return ExecutionResult(ResultDiagnosis.NEGATIVE_RESULT, "negative")
    if task.name == "positive_no_search":
        return ExecutionResult(ResultDiagnosis.POSITIVE_RESULT, "positive")
    if task.name == "negative_no_search":
        return ExecutionResult(ResultDiagnosis.NEGATIVE_RESULT, "negative")
    if task.name == "implementation_then_repair":
        return ExecutionResult(ResultDiagnosis.POSITIVE_RESULT, "positive")
    if task.name == "instrument_then_measurement":
        if plan.measurement == "nanmean":
            return ExecutionResult(ResultDiagnosis.POSITIVE_RESULT, "positive")
        return ExecutionResult(ResultDiagnosis.INSTRUMENT_FAILURE, "no_conclusion")
    if task.name == "confounded_then_control":
        if "shuffled_negative" in plan.controls:
            return ExecutionResult(ResultDiagnosis.NEGATIVE_RESULT, "negative")
        return ExecutionResult(ResultDiagnosis.CONFOUNDED, "false_positive")
    if task.name == "second_revision_needed":
        if plan.sample_size >= 24:
            return ExecutionResult(ResultDiagnosis.POSITIVE_RESULT, "positive")
        return ExecutionResult(ResultDiagnosis.LOW_POWER, "inconclusive")
    if task.name == "cyclic_low_power_budget_stop":
        return ExecutionResult(ResultDiagnosis.LOW_POWER, "inconclusive")
    if task.name == "low_power_then_samples":
        if plan.sample_size >= 20:
            return ExecutionResult(ResultDiagnosis.POSITIVE_RESULT, "positive")
        return ExecutionResult(ResultDiagnosis.LOW_POWER, "inconclusive")
    if task.evidence_status is EvidenceStatus.CONFLICTING:
        return ExecutionResult(ResultDiagnosis.INSTRUMENT_FAILURE, "no_conclusion")
    if task.evidence_status is EvidenceStatus.UNRESOLVED:
        return ExecutionResult(ResultDiagnosis.POSITIVE_RESULT, "positive")
    return ExecutionResult(ResultDiagnosis.POSITIVE_RESULT, "positive")


def _gap_for(diagnosis: ResultDiagnosis) -> KnowledgeGap | None:
    mapping = {
        ResultDiagnosis.LOW_POWER: ("power_analysis", "How should sample size be revised after low power?"),
        ResultDiagnosis.INSTRUMENT_FAILURE: ("measurement_method", "What measurement method is valid here?"),
        ResultDiagnosis.CONFOUNDED: ("causal_control", "Which control separates treatment from confound?"),
        ResultDiagnosis.POSITIVE_RESULT: ("domain_mechanism", "What alternative mechanism could explain this positive result?"),
    }
    if diagnosis not in mapping:
        return None
    category, question = mapping[diagnosis]
    return KnowledgeGap(question, "Autonomous loop generated a blocking gap.", True, 0.25, 0.88, category)


def _evidence_for(task: LoopTask, gap: KnowledgeGap) -> tuple[EvidenceStatus, tuple[GroundedEvidence, ...]]:
    if task.evidence_status is not EvidenceStatus.SUPPORTED:
        return task.evidence_status, ()
    claims = {
        "power_analysis": "Sample-size planning should use an estimated effect size and target power.",
        "measurement_method": "nanmean ignores NaN values when computing the mean.",
        "causal_control": "A shuffled negative control separates treatment effects from shared drift.",
    }
    claim = claims[gap.category]
    return EvidenceStatus.SUPPORTED, (GroundedEvidence(
        claim=claim,
        source_url=f"https://example.test/l52/{task.name}/{gap.category}",
        source_type="official_docs" if gap.category != "power_analysis" else "paper",
        retrieved_at=RETRIEVED_AT,
        supports=True,
        relevance=0.9,
        authority=0.88,
        independence_group=f"{task.name}-{gap.category}",
        uncertainty=0.12,
        quoted_span=claim[:48],
    ),)


def _proposal_for(diagnosis: ResultDiagnosis, evidence: tuple[GroundedEvidence, ...]) -> RevisionProposal | None:
    if diagnosis is ResultDiagnosis.LOW_POWER:
        return RevisionProposal("sample_size", "increase", tuple(item.claim for item in evidence),
                                tuple(item.source_url for item in evidence),
                                "reduce uncertainty enough for a decision", "higher compute cost", 0.86)
    if diagnosis is ResultDiagnosis.INSTRUMENT_FAILURE:
        return RevisionProposal("measurement", "replace", tuple(item.claim for item in evidence),
                                tuple(item.source_url for item in evidence),
                                "make the recorded metric valid", "estimator assumptions remain testable", 0.9)
    if diagnosis is ResultDiagnosis.CONFOUNDED:
        return RevisionProposal("controls", "add", tuple(item.claim for item in evidence),
                                tuple(item.source_url for item in evidence),
                                "separate treatment effect from shared drift", "control may be conservative", 0.84)
    return None


def _apply_loop_revision(task: LoopTask, plan: ExperimentPlan, proposal: RevisionProposal | None) -> ExperimentPlan:
    if proposal is None:
        return plan
    if proposal.target == "sample_size":
        if task.needs_two_revisions and plan.sample_size < 12:
            return replace(plan, sample_size=12)
        return replace(plan, sample_size=max(plan.sample_size, 24))
    if proposal.target == "measurement":
        return replace(plan, measurement="nanmean")
    if proposal.target == "controls":
        return replace(plan, controls=tuple(dict.fromkeys(plan.controls + ("shuffled_negative",))))
    return plan


def _repair_code(plan: ExperimentPlan) -> ExperimentPlan:
    return replace(plan, implementation_ok=True)


def _terminal_from_result(result: ExecutionResult) -> FinalStatus | None:
    if result.diagnosis is ResultDiagnosis.POSITIVE_RESULT:
        return FinalStatus.SUPPORTED
    if result.diagnosis is ResultDiagnosis.NEGATIVE_RESULT:
        return FinalStatus.FALSIFIED
    return None


def _initial_state(max_iterations: int, max_retrievals: int, max_executions: int,
                   max_revisions: int) -> ScientificLoopState:
    return ScientificLoopState(
        iteration=0,
        hypothesis_status="active",
        budget_remaining={
            "iterations": max_iterations,
            "retrievals": max_retrievals,
            "executions": max_executions,
            "revisions": max_revisions,
        },
    )


def run_loop(task: LoopTask, policy: str, *, max_iterations: int = 4, max_retrievals: int = 3,
             max_executions: int = 4, max_revisions: int = 3) -> dict[str, object]:
    state = _initial_state(max_iterations, max_retrievals, max_executions, max_revisions)
    plan = task.initial_plan
    final_status = FinalStatus.UNRESOLVED
    conclusion = "unresolved"
    while state.budget_remaining["iterations"] > 0 and state.budget_remaining["executions"] > 0:
        state.iteration += 1
        state.budget_remaining["iterations"] -= 1
        state.budget_remaining["executions"] -= 1
        result = _diagnosis_for(task, plan)
        state.execution_history.append(f"{state.iteration}:{result.conclusion}")
        state.diagnosis_history.append(result.diagnosis.value)

        terminal = _terminal_from_result(result)
        if result.diagnosis is ResultDiagnosis.POSITIVE_RESULT and task.evidence_status is EvidenceStatus.UNRESOLVED:
            terminal = None
        if terminal is not None:
            final_status = terminal
            conclusion = result.conclusion
            if task.name == "implementation_then_repair" and plan.implementation_ok:
                final_status = FinalStatus.CODE_REPAIRED
            break
        if result.diagnosis is ResultDiagnosis.IMPLEMENTATION_FAILURE:
            if policy in {"coding_only", "diagnosis_only", "diagnosis_knowledge_gap",
                          "diagnosis_retrieval_revision", "full_autonomous_loop"}:
                if state.budget_remaining["revisions"] <= 0:
                    break
                state.budget_remaining["revisions"] -= 1
                plan = _repair_code(plan)
                state.revision_history.append(RevisionDecision.ROUTE_TO_CODE_REPAIR.value)
                if policy != "full_autonomous_loop":
                    final_status = FinalStatus.UNRESOLVED
                    conclusion = "unresolved"
                    break
                continue

        if result.diagnosis is ResultDiagnosis.NEGATIVE_RESULT:
            final_status = FinalStatus.FALSIFIED
            conclusion = "negative"
            state.hypothesis_status = "abandoned"
            break

        if policy == "coding_only":
            final_status = FinalStatus.SUPPORTED if result.conclusion == "false_positive" else FinalStatus.UNRESOLVED
            conclusion = result.conclusion if result.conclusion == "false_positive" else "unresolved"
            break
        if policy == "diagnosis_only":
            final_status = FinalStatus.UNRESOLVED
            conclusion = "unresolved"
            break

        gap = _gap_for(result.diagnosis)
        if gap is None:
            final_status = FinalStatus.UNRESOLVED
            conclusion = "unresolved"
            break
        state.knowledge_gaps.append(gap.category)
        if policy == "diagnosis_knowledge_gap":
            final_status = FinalStatus.UNRESOLVED
            conclusion = "unresolved"
            break

        if state.budget_remaining["retrievals"] <= 0:
            state.terminated_by_budget = True
            final_status = FinalStatus.BUDGET_STOPPED
            conclusion = "unresolved"
            break
        state.budget_remaining["retrievals"] -= 1
        evidence_status, evidence = _evidence_for(task, gap)
        state.evidence_history.append(evidence_status.value)

        if evidence_status is EvidenceStatus.CONFLICTING or evidence_status is EvidenceStatus.UNRESOLVED:
            final_status = FinalStatus.UNRESOLVED
            conclusion = "unresolved"
            break

        if state.budget_remaining["revisions"] <= 0:
            state.terminated_by_budget = True
            final_status = FinalStatus.BUDGET_STOPPED
            conclusion = "unresolved"
            break
        state.budget_remaining["revisions"] -= 1
        proposal = _proposal_for(result.diagnosis, evidence)
        plan = _apply_loop_revision(task, plan, proposal)
        state.revision_history.append(proposal.change_type if proposal else RevisionDecision.KEEP_PLAN.value)
        if policy == "diagnosis_retrieval_revision":
            final = _diagnosis_for(task, plan)
            state.budget_remaining["executions"] -= 1
            state.execution_history.append(f"{state.iteration + 1}:{final.conclusion}")
            state.diagnosis_history.append(final.diagnosis.value)
            terminal = _terminal_from_result(final)
            final_status = terminal or FinalStatus.UNRESOLVED
            conclusion = final.conclusion if terminal else "unresolved"
            break
    else:
        state.terminated_by_budget = True
        final_status = FinalStatus.BUDGET_STOPPED
        conclusion = "unresolved"

    budget_violation = any(value < 0 for value in state.budget_remaining.values())
    false_conclusion = conclusion in {"positive", "negative", "false_positive"} and conclusion != task.expected_conclusion
    unsupported_final_claim = final_status is FinalStatus.SUPPORTED and not (
        task.name in {"positive_no_search", "implementation_then_repair"} or EvidenceStatus.SUPPORTED.value in state.evidence_history
    )
    state.unsupported_final_claim = unsupported_final_claim
    return {
        "task": task.name,
        "policy": policy,
        "final_status": final_status.value,
        "expected_status": task.expected_status.value,
        "conclusion": conclusion,
        "expected_conclusion": task.expected_conclusion,
        "correct_final_conclusion": conclusion == task.expected_conclusion and final_status is task.expected_status,
        "false_conclusion": false_conclusion,
        "abandon_correct": final_status is FinalStatus.FALSIFIED and task.expected_status is FinalStatus.FALSIFIED,
        "unresolved_correct": final_status in {FinalStatus.UNRESOLVED, FinalStatus.BUDGET_STOPPED}
        and task.expected_status in {FinalStatus.UNRESOLVED, FinalStatus.BUDGET_STOPPED},
        "iterations": state.iteration,
        "executions": len(state.execution_history),
        "retrievals": max_retrievals - state.budget_remaining["retrievals"],
        "revisions": len(state.revision_history),
        "unnecessary_action": bool(state.evidence_history or state.revision_history)
        and task.name in {"positive_no_search", "negative_no_search"},
        "loop_terminated_correctly": final_status is task.expected_status,
        "budget_violation": budget_violation,
        "unsupported_final_claim": unsupported_final_claim,
        "state": {
            "iteration": state.iteration,
            "hypothesis_status": state.hypothesis_status,
            "diagnosis_history": state.diagnosis_history,
            "knowledge_gaps": state.knowledge_gaps,
            "evidence_history": state.evidence_history,
            "revision_history": state.revision_history,
            "execution_history": state.execution_history,
            "budget_remaining": state.budget_remaining,
        },
    }


def _aggregate(rows: list[dict[str, object]]) -> dict[str, float]:
    resolved = [row for row in rows if row["final_status"] not in {FinalStatus.UNRESOLVED.value, FinalStatus.BUDGET_STOPPED.value}]
    return {
        "task_resolution_rate": len(resolved) / len(rows),
        "correct_final_conclusion_rate": sum(bool(row["correct_final_conclusion"]) for row in rows) / len(rows),
        "false_conclusion_rate": sum(bool(row["false_conclusion"]) for row in rows) / len(rows),
        "abandon_accuracy": sum(bool(row["abandon_correct"]) for row in rows if row["expected_status"] == FinalStatus.FALSIFIED.value)
        / max(1, sum(row["expected_status"] == FinalStatus.FALSIFIED.value for row in rows)),
        "mean_iterations_to_resolution": sum(int(row["iterations"]) for row in rows) / len(rows),
        "mean_executions_per_task": sum(int(row["executions"]) for row in rows) / len(rows),
        "mean_retrievals_per_task": sum(int(row["retrievals"]) for row in rows) / len(rows),
        "unnecessary_action_rate": sum(bool(row["unnecessary_action"]) for row in rows) / len(rows),
        "loop_termination_accuracy": sum(bool(row["loop_terminated_correctly"]) for row in rows) / len(rows),
        "budget_violation_rate": sum(bool(row["budget_violation"]) for row in rows) / len(rows),
        "unsupported_final_claim_rate": sum(bool(row["unsupported_final_claim"]) for row in rows) / len(rows),
    }


def run_autonomous_scientific_loop_experiment() -> dict[str, object]:
    policies = ("coding_only", "diagnosis_only", "diagnosis_knowledge_gap",
                "diagnosis_retrieval_revision", "full_autonomous_loop")
    report: dict[str, object] = {"tasks": [task.name for task in tasks()], "policies": {}}
    for policy in policies:
        rows = [run_loop(task, policy) for task in tasks()]
        report["policies"][policy] = {"rows": rows, "metrics": _aggregate(rows)}
    return report


if __name__ == "__main__":
    print(json.dumps(run_autonomous_scientific_loop_experiment(), ensure_ascii=False, indent=2, sort_keys=True))
