"""L8.12: drive a goal-driven plan until its contract is satisfied.

L8.9 states what a goal requires and judges one finished response.  L8.11
compiles a request into a requirement graph, a task graph and a contract.
Neither of them continues: a run whose contract is unsatisfied simply reports
``GOAL_NOT_ACHIEVED`` and stops, which is exactly the shape of the failure
where "作成して" is answered with a file listing.

This module closes that loop.  Every unmet observation is translated into a
remediation directive drawn from a *closed* vocabulary, the executor is
invoked again carrying those directives, and the contract is re-evaluated.
The loop stops when the goal is achieved, when another attempt would repeat
the previous failure, or when a gap has no directive -- it never invents a
remediation it cannot name.

Three termination rules matter more than the happy path:

* ``ESCALATED_NO_PROGRESS`` -- the same gap survived a remediation attempt.
  Without this the loop spins on an executor that ignores directives.
* ``ESCALATED_UNSUPPORTED_GAP`` -- the gap is real but outside the vocabulary.
  Abstaining is the correct behaviour, as in L8.6's ABSTAIN fixtures.
* ``ESCALATED_BUDGET_EXHAUSTED`` -- progress was made every round but the
  budget ran out.  Distinct from stalling, because the diagnosis differs.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Protocol

from .behavior_monitor_l87_experiment import BehaviorEvent, ExecutionTrace
from .goal_requirement_l811_experiment import GoalDrivenPlan, compile_goal_driven_plan
from .goal_semantics_l89_experiment import (
    GoalObservation,
    GoalSatisfaction,
    evaluate_goal_contract,
    observe_goal_behavior,
)


class RemediationAction(Enum):
    """Closed vocabulary of next actions.

    Kept deliberately small and tied to the observations L8.9 actually checks.
    A gap that cannot be expressed here escalates rather than being guessed at.
    """

    WRITE_ARTIFACT = "WRITE_ARTIFACT"
    APPLY_WORKSPACE_CHANGE = "APPLY_WORKSPACE_CHANGE"
    RUN_COMMAND = "RUN_COMMAND"
    CAPTURE_RETURN_CODE = "CAPTURE_RETURN_CODE"
    REPORT_VERIFICATION = "REPORT_VERIFICATION"
    ADVANCE_BEYOND_SEARCH = "ADVANCE_BEYOND_SEARCH"


#: Requirement strings are the ones ``contract_from_goal`` emits verbatim.
_REQUIREMENT_ACTIONS: dict[str, tuple[RemediationAction, str]] = {
    "FILE_WRITTEN >= 1": (
        RemediationAction.WRITE_ARTIFACT,
        "Write the requested artifact to a file; a description of it does not satisfy the goal.",
    ),
    "WORKSPACE_DIFF >= 1": (
        RemediationAction.APPLY_WORKSPACE_CHANGE,
        "Apply the change to the workspace; report the touched path after writing or moving it.",
    ),
    "SUBPROCESS_STARTED >= 1": (
        RemediationAction.RUN_COMMAND,
        "Actually start the command; quoting how to run it does not satisfy the goal.",
    ),
    "return_code present": (
        RemediationAction.CAPTURE_RETURN_CODE,
        "Report the process return code in the response.",
    ),
    "verification verdict present": (
        RemediationAction.REPORT_VERIFICATION,
        "State the verification verdict (passed or failed) explicitly in the response.",
    ),
}

_FORBIDDEN_ACTIONS: dict[str, tuple[RemediationAction, str]] = {
    "search_results_only": (
        RemediationAction.ADVANCE_BEYOND_SEARCH,
        "Search output is an intermediate step; continue to the action the goal asked for.",
    ),
    "file_overview_only": (
        RemediationAction.ADVANCE_BEYOND_SEARCH,
        "A file overview is an intermediate step; continue to the action the goal asked for.",
    ),
    "memory_dump_only": (
        RemediationAction.ADVANCE_BEYOND_SEARCH,
        "Reused knowledge is an intermediate step; continue to the action the goal asked for.",
    ),
}


@dataclass(frozen=True)
class RemediationDirective:
    action: RemediationAction
    gap: str
    instruction: str


@dataclass(frozen=True)
class LoopIteration:
    index: int
    directives: tuple[RemediationDirective, ...]
    response_text: str
    observation: GoalObservation
    satisfaction: GoalSatisfaction


@dataclass(frozen=True)
class LoopOutcome:
    request: str
    plan: GoalDrivenPlan
    iterations: tuple[LoopIteration, ...]
    status: str
    statement: str
    unresolved_gaps: tuple[str, ...]

    @property
    def achieved(self) -> bool:
        return self.status == "GOAL_ACHIEVED"

    @property
    def iteration_count(self) -> int:
        return len(self.iterations)

    @property
    def remediation_rounds(self) -> int:
        return sum(1 for iteration in self.iterations if iteration.directives)


class GoalExecutor(Protocol):
    """One attempt at the request.

    ``directives`` is empty on the first attempt and carries the remediation
    for every later one, so an executor that reads it can recover while one
    that ignores it stalls -- which is what makes the loop measurable.
    """

    def __call__(
        self, request: str, directives: tuple[RemediationDirective, ...]
    ) -> tuple[str, ExecutionTrace]:
        ...


def remediation_for(satisfaction: GoalSatisfaction) -> tuple[RemediationDirective, ...]:
    """Translate an unsatisfied contract into directives, or nothing if unnameable."""
    directives: list[RemediationDirective] = []
    for requirement in satisfaction.missing_observations:
        entry = _REQUIREMENT_ACTIONS.get(requirement)
        if entry is None:
            continue
        action, instruction = entry
        directives.append(RemediationDirective(action, requirement, instruction))
    forbidden = satisfaction.forbidden_final_response
    if forbidden is not None:
        entry = _FORBIDDEN_ACTIONS.get(forbidden)
        if entry is not None:
            action, instruction = entry
            if not any(item.action == action for item in directives):
                directives.append(RemediationDirective(action, forbidden, instruction))
    return tuple(directives)


def unsupported_gaps(satisfaction: GoalSatisfaction) -> tuple[str, ...]:
    """Gaps the vocabulary cannot express.  Non-empty means abstain, not guess."""
    gaps = [item for item in satisfaction.missing_observations if item not in _REQUIREMENT_ACTIONS]
    forbidden = satisfaction.forbidden_final_response
    if forbidden is not None and forbidden not in _FORBIDDEN_ACTIONS:
        gaps.append(forbidden)
    return tuple(gaps)


def _gap_signature(satisfaction: GoalSatisfaction) -> frozenset[str]:
    items = set(satisfaction.missing_observations)
    if satisfaction.forbidden_final_response is not None:
        items.add(f"forbidden:{satisfaction.forbidden_final_response}")
    return frozenset(items)


def run_goal_loop(
    request: str,
    executor: GoalExecutor,
    *,
    max_iterations: int = 4,
) -> LoopOutcome:
    """Execute, observe and re-plan until the goal contract is satisfied."""
    if max_iterations < 1:
        raise ValueError("max_iterations must be at least 1")

    plan = compile_goal_driven_plan(request)
    iterations: list[LoopIteration] = []
    directives: tuple[RemediationDirective, ...] = ()
    previous_gap: frozenset[str] | None = None

    for index in range(1, max_iterations + 1):
        response_text, trace = executor(request, directives)
        observation = observe_goal_behavior(trace, response_text)
        satisfaction = evaluate_goal_contract(plan.contract, observation)
        iterations.append(
            LoopIteration(index, directives, response_text, observation, satisfaction)
        )

        if satisfaction.achieved:
            return _outcome(request, plan, iterations, "GOAL_ACHIEVED", ())

        unsupported = unsupported_gaps(satisfaction)
        if unsupported:
            return _outcome(request, plan, iterations, "ESCALATED_UNSUPPORTED_GAP", unsupported)

        gap = _gap_signature(satisfaction)
        if previous_gap is not None and gap == previous_gap:
            return _outcome(request, plan, iterations, "ESCALATED_NO_PROGRESS", tuple(sorted(gap)))

        next_directives = remediation_for(satisfaction)
        if not next_directives:
            return _outcome(request, plan, iterations, "ESCALATED_UNSUPPORTED_GAP", tuple(sorted(gap)))

        previous_gap = gap
        directives = next_directives

    final_gap = _gap_signature(iterations[-1].satisfaction)
    return _outcome(request, plan, iterations, "ESCALATED_BUDGET_EXHAUSTED", tuple(sorted(final_gap)))


def _outcome(
    request: str,
    plan: GoalDrivenPlan,
    iterations: list[LoopIteration],
    status: str,
    unresolved: tuple[str, ...],
) -> LoopOutcome:
    goal_state = plan.goal.desired_state
    count = len(iterations)
    if status == "GOAL_ACHIEVED":
        statement = f"GOAL_ACHIEVED after {count} iteration(s): {goal_state}"
    else:
        detail = ", ".join(unresolved) if unresolved else "no gap recorded"
        statement = f"{status} after {count} iteration(s): {goal_state} (unresolved: {detail})"
    return LoopOutcome(request, plan, tuple(iterations), status, statement, unresolved)


def format_loop_outcome(outcome: LoopOutcome, title: str = "Goal Execution Loop (L8.12)") -> str:
    lines = [
        f"## {title}",
        "",
        f"- request: `{outcome.request}`",
        f"- goal: `{outcome.plan.goal.desired_state}`",
        f"- status: `{outcome.status}`",
        f"- iterations: `{outcome.iteration_count}`",
        f"- remediation rounds: `{outcome.remediation_rounds}`",
        "",
        "【Iterations】",
    ]
    for iteration in outcome.iterations:
        actions = ", ".join(item.action.value for item in iteration.directives) or "none"
        verdict = "achieved" if iteration.satisfaction.achieved else "not achieved"
        lines.append(f"- {iteration.index}. directives=[{actions}] -> {verdict}")
        if not iteration.satisfaction.achieved:
            for gap in iteration.satisfaction.missing_observations:
                lines.append(f"    - missing `{gap}`")
            if iteration.satisfaction.forbidden_final_response:
                lines.append(
                    f"    - forbidden `{iteration.satisfaction.forbidden_final_response}`"
                )
    if outcome.unresolved_gaps:
        lines.extend(["", "【Unresolved】", *[f"- `{item}`" for item in outcome.unresolved_gaps]])
    lines.extend(["", f"- {outcome.statement}"])
    return "\n".join(lines)


# --------------------------------------------------------------------------
# experiment fixtures
# --------------------------------------------------------------------------

def trace_with(*events: BehaviorEvent) -> ExecutionTrace:
    trace = ExecutionTrace()
    for event in events:
        trace.record(event, "fixture")
    return trace


StepFn = Callable[[tuple[RemediationDirective, ...]], tuple[str, ExecutionTrace]]


@dataclass(frozen=True)
class LoopFixture:
    name: str
    request: str
    steps: tuple[StepFn, ...]
    expected_status: str
    expected_iterations: int | None
    note: str


def _scripted(steps: tuple[StepFn, ...]) -> GoalExecutor:
    calls = {"n": 0}

    def executor(
        request: str, directives: tuple[RemediationDirective, ...]
    ) -> tuple[str, ExecutionTrace]:
        index = min(calls["n"], len(steps) - 1)
        calls["n"] += 1
        return steps[index](directives)

    return executor


def _has(directives: tuple[RemediationDirective, ...], action: RemediationAction) -> bool:
    return any(item.action == action for item in directives)


def _search_only(_: tuple[RemediationDirective, ...]) -> tuple[str, ExecutionTrace]:
    return (
        "ローカル要約\n一致ファイル: simulation.py:12: def step\n一致ファイル: solver.py:4: import numpy",
        trace_with(BehaviorEvent.REPOSITORY_INDEXED),
    )


def _write_if_told(directives: tuple[RemediationDirective, ...]) -> tuple[str, ExecutionTrace]:
    if _has(directives, RemediationAction.WRITE_ARTIFACT):
        return (
            "lorentz_simulation.py を作成しました。",
            trace_with(BehaviorEvent.FILE_WRITTEN),
        )
    return _search_only(directives)


def _write_immediately(_: tuple[RemediationDirective, ...]) -> tuple[str, ExecutionTrace]:
    return ("lorentz_simulation.py を作成しました。", trace_with(BehaviorEvent.FILE_WRITTEN))


def _never_writes(_: tuple[RemediationDirective, ...]) -> tuple[str, ExecutionTrace]:
    return _search_only(())


def _run_without_return_code(_: tuple[RemediationDirective, ...]) -> tuple[str, ExecutionTrace]:
    return ("テストを実行しました。", trace_with(BehaviorEvent.SUBPROCESS_STARTED))


def _run_with_return_code(directives: tuple[RemediationDirective, ...]) -> tuple[str, ExecutionTrace]:
    if _has(directives, RemediationAction.CAPTURE_RETURN_CODE):
        return (
            "テストを実行しました。終了コード: 0",
            trace_with(BehaviorEvent.SUBPROCESS_STARTED),
        )
    return _run_without_return_code(directives)


def _explains_only(_: tuple[RemediationDirective, ...]) -> tuple[str, ExecutionTrace]:
    return (
        "ローレンツ力は F = qv × B で与えられ、一様磁場中では旋回運動になります。",
        trace_with(),
    )


def fixtures() -> tuple[LoopFixture, ...]:
    create_request = "一様磁場中で荷電粒子が運動するシミュレーションを作成して"
    return (
        LoopFixture(
            "S1-first-attempt",
            create_request,
            (_write_immediately,),
            "GOAL_ACHIEVED",
            1,
            "control: an already-satisfied goal must not trigger a second round",
        ),
        LoopFixture(
            "S2-recovers-after-directive",
            create_request,
            (_write_if_told,),
            "GOAL_ACHIEVED",
            2,
            "headline case: search-only first answer, artifact after WRITE_ARTIFACT",
        ),
        LoopFixture(
            "S3-return-code-recovered",
            "テストを実行して",
            (_run_with_return_code,),
            "GOAL_ACHIEVED",
            2,
            "partial progress: subprocess ran, return code added after directive",
        ),
        LoopFixture(
            "S4-deaf-executor-escalates",
            create_request,
            (_never_writes,),
            "ESCALATED_NO_PROGRESS",
            2,
            "control: ignoring directives must escalate, not spin to the budget",
        ),
        LoopFixture(
            "S5-information-goal-untouched",
            "ローレンツ力について説明して",
            (_explains_only,),
            "GOAL_ACHIEVED",
            1,
            "control: an information goal must not be forced to write a file",
        ),
        LoopFixture(
            "S6-stalls-then-recovers",
            create_request,
            (_search_only, _run_without_return_code, _write_immediately),
            "GOAL_ACHIEVED",
            3,
            "progress changes the gap each round, so the loop is allowed to continue",
        ),
    )


def run_experiment(max_iterations: int = 4) -> dict[str, object]:
    results = []
    for fixture in fixtures():
        outcome = run_goal_loop(
            fixture.request, _scripted(fixture.steps), max_iterations=max_iterations
        )
        status_ok = outcome.status == fixture.expected_status
        iterations_ok = (
            fixture.expected_iterations is None
            or outcome.iteration_count == fixture.expected_iterations
        )
        results.append(
            {
                "name": fixture.name,
                "note": fixture.note,
                "status": outcome.status,
                "expected_status": fixture.expected_status,
                "iterations": outcome.iteration_count,
                "expected_iterations": fixture.expected_iterations,
                "remediation_rounds": outcome.remediation_rounds,
                "status_ok": status_ok,
                "iterations_ok": iterations_ok,
                "outcome": outcome,
            }
        )

    achievable = [item for item in results if item["expected_status"] == "GOAL_ACHIEVED"]
    escalating = [item for item in results if item["expected_status"] != "GOAL_ACHIEVED"]
    achieved = [item for item in achievable if item["status"] == "GOAL_ACHIEVED"]
    first_try = [item for item in achieved if item["remediation_rounds"] == 0]
    recovered = [item for item in achieved if item["remediation_rounds"] > 0]
    needed_recovery = [item for item in achievable if item["expected_iterations"] != 1]

    metrics = {
        "goal_completion_rate": _ratio(len(achieved), len(achievable)),
        "first_attempt_success_rate": _ratio(len(first_try), len(achievable)),
        "recovery_rate": _ratio(len(recovered), len(needed_recovery)),
        "escalation_accuracy": _ratio(
            sum(1 for item in escalating if item["status_ok"]), len(escalating)
        ),
        "iteration_budget_accuracy": _ratio(
            sum(1 for item in results if item["iterations_ok"]), len(results)
        ),
        "mean_iterations_to_goal": _mean([item["iterations"] for item in achieved]),
    }
    return {"results": results, "metrics": metrics}


def _ratio(numerator: int, denominator: int) -> float | str:
    if denominator == 0:
        return "n/a"
    return round(numerator / denominator, 3)


def _mean(values: list[int]) -> float | str:
    if not values:
        return "n/a"
    return round(sum(values) / len(values), 3)


def format_experiment(report: dict[str, object]) -> str:
    results = report["results"]
    metrics = report["metrics"]
    lines = ["## Goal Execution Loop Experiment (L8.12)", "", "【Fixtures】"]
    for item in results:
        mark = "OK" if item["status_ok"] and item["iterations_ok"] else "NG"
        lines.append(
            f"- [{mark}] {item['name']}: {item['status']} "
            f"in {item['iterations']} iteration(s) "
            f"(expected {item['expected_status']}/{item['expected_iterations']})"
        )
        lines.append(f"    - {item['note']}")
    lines.extend(["", "【Metrics】"])
    for key, value in metrics.items():
        lines.append(f"- {key}: `{value}`")
    return "\n".join(lines)


def main() -> None:  # pragma: no cover - manual entry point
    report = run_experiment()
    print(format_experiment(report))
    print()
    demo = run_goal_loop(
        "一様磁場中で荷電粒子が運動するシミュレーションを作成して",
        _scripted((_write_if_told,)),
    )
    print(format_loop_outcome(demo))


if __name__ == "__main__":  # pragma: no cover
    main()
