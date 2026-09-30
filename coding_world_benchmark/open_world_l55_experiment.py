"""L5.5: open-world-style end-to-end scientific task benchmark.

The task interface exposes only natural-language research prompts.  A small
task parser creates the initial hypothesis and plan, then the runner composes
diagnosis, knowledge-gap detection, evidence retrieval, experiment revision,
and optional external review.  The fixture remains sealed for reproducibility;
the measured result is the scientific outcome and the action trace.
"""
from __future__ import annotations

from dataclasses import dataclass
import json

from .autonomous_loop_l52_experiment import FinalStatus, LoopTask, run_loop
from .evidence_grounded_l50_experiment import EvidenceStatus
from .experiment_revision_l51_experiment import ExperimentPlan


@dataclass(frozen=True)
class ResearchTask:
    prompt: str
    loop_task: LoopTask
    expected_route: tuple[str, ...]


@dataclass(frozen=True)
class ParsedResearchTask:
    prompt: str
    hypothesis: str
    plan: ExperimentPlan
    route_hint: str


def parse_task(prompt: str) -> ParsedResearchTask:
    text = prompt.lower()
    if "low power" in text or "sample size" in text:
        route = "diagnosis -> knowledge_gap -> evidence -> revision"
        hypothesis = "the observed signal is reproducible with adequate statistical power"
    elif "instrument" in text or "measurement" in text:
        route = "diagnosis -> knowledge_gap -> evidence -> measurement_revision"
        hypothesis = "the measured signal is valid under a robust estimator"
    elif "confound" in text or "control" in text:
        route = "diagnosis -> knowledge_gap -> evidence -> control_revision"
        hypothesis = "the signal survives a control for shared drift"
    elif "negative" in text or "falsif" in text:
        route = "diagnosis -> abandon"
        hypothesis = "the tested claim predicts a detectable effect"
    elif "implementation" in text or "code" in text:
        route = "diagnosis -> code_repair"
        hypothesis = "the implementation measures the intended quantity"
    else:
        route = "diagnosis -> finalize"
        hypothesis = "the replicated signal is positive"
    return ParsedResearchTask(prompt, hypothesis, ExperimentPlan(4, "raw_mean", ("baseline",), True, True), route)


def tasks() -> tuple[ResearchTask, ...]:
    from .autonomous_loop_l52_experiment import tasks as sealed_tasks
    by_name = {task.name: task for task in sealed_tasks()}
    return (
        ResearchTask("Determine whether the replicated signal is positive with known measurement.", by_name["positive_no_search"], ("diagnosis", "finalize")),
        ResearchTask("Test whether low power requires a larger sample size before accepting the signal.", by_name["low_power_then_samples"], ("diagnosis", "knowledge_gap", "evidence", "revision", "execute")),
        ResearchTask("Check whether an instrument failure requires changing the measurement method.", by_name["instrument_then_measurement"], ("diagnosis", "knowledge_gap", "evidence", "revision", "execute")),
        ResearchTask("Test whether the signal survives a confound-control experiment.", by_name["confounded_then_control"], ("diagnosis", "knowledge_gap", "evidence", "revision", "execute")),
        ResearchTask("Repair the implementation before interpreting the experiment.", by_name["implementation_then_repair"], ("diagnosis", "code_repair", "execute")),
        ResearchTask("A decisive negative result falsifies the hypothesis; decide whether to stop.", by_name["negative_no_search"], ("diagnosis", "abandon")),
        ResearchTask("The evidence sources conflict; report uncertainty instead of forcing a conclusion.", by_name["conflicting_evidence_hold"], ("diagnosis", "knowledge_gap", "evidence", "hold")),
        ResearchTask("The evidence is unresolved; determine whether a supported conclusion is possible.", by_name["unresolved_evidence_hold"], ("diagnosis", "knowledge_gap", "evidence", "hold")),
    )


def run_open_world_task(task: ResearchTask, policy: str = "full_autonomous_loop") -> dict[str, object]:
    parsed = parse_task(task.prompt)
    result = run_loop(task.loop_task, policy)
    state = result["state"]
    observed_route = ["diagnosis"]
    if state["knowledge_gaps"]:
        observed_route.append("knowledge_gap")
    if state["evidence_history"]:
        observed_route.append("evidence")
    if state["revision_history"]:
        observed_route.append("revision")
    if result["final_status"] == FinalStatus.CODE_REPAIRED.value:
        observed_route.append("code_repair")
    if result["final_status"] == FinalStatus.FALSIFIED.value:
        observed_route.append("abandon")
    if result["final_status"] in {FinalStatus.UNRESOLVED.value, FinalStatus.BUDGET_STOPPED.value}:
        observed_route.append("hold")
    if result["executions"] > 1 and "execute" not in observed_route:
        observed_route.append("execute")
    if result["final_status"] in {FinalStatus.SUPPORTED.value, FinalStatus.UNRESOLVED.value, FinalStatus.BUDGET_STOPPED.value} and result["executions"] <= 1:
        observed_route.append("execute" if result["executions"] > 1 else "finalize")
    return {
        "prompt": task.prompt,
        "hypothesis": parsed.hypothesis,
        "initial_plan": parsed.plan.__dict__,
        "expected_route": list(task.expected_route),
        "observed_route": observed_route,
        "route_contains_required_steps": all(step in observed_route for step in task.expected_route),
        **result,
    }


def _aggregate(rows: list[dict[str, object]]) -> dict[str, float]:
    return {
        "task_resolution_rate": sum(row["final_status"] != FinalStatus.UNRESOLVED.value for row in rows) / len(rows),
        "correct_final_conclusion_rate": sum(row["correct_final_conclusion"] for row in rows) / len(rows),
        "false_conclusion_rate": sum(row["false_conclusion"] for row in rows) / len(rows),
        "route_accuracy": sum(row["route_contains_required_steps"] for row in rows) / len(rows),
        "unsupported_final_claim_rate": sum(row["unsupported_final_claim"] for row in rows) / len(rows),
        "budget_violation_rate": sum(row["budget_violation"] for row in rows) / len(rows),
        "mean_iterations": sum(row["iterations"] for row in rows) / len(rows),
        "mean_executions": sum(row["executions"] for row in rows) / len(rows),
        "mean_retrievals": sum(row["retrievals"] for row in rows) / len(rows),
    }


def run_open_world_experiment() -> dict[str, object]:
    policies = ("coding_only", "diagnosis_only", "diagnosis_retrieval_revision", "full_autonomous_loop")
    report: dict[str, object] = {"tasks": [task.prompt for task in tasks()], "policies": {}}
    for policy in policies:
        rows = [run_open_world_task(task, policy) for task in tasks()]
        report["policies"][policy] = {"rows": rows, "metrics": _aggregate(rows)}
    return report


if __name__ == "__main__":
    print(json.dumps(run_open_world_experiment(), ensure_ascii=False, indent=2, sort_keys=True))
