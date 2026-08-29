"""Generic application bridge and verification helpers for the model.

The bridge intentionally has no dependency on a particular application.  An
application supplies numeric features and ranked candidates, then consumes the
bounded workspace and readout signals through ordinary Python callables.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Callable, Iterable, Mapping, Sequence

from .consciousness import CognitiveReadout, FiniteWorkspace, PhenomenalState, WorkspaceItem, readout


@dataclass(frozen=True)
class ConnectionInput:
    """Application-neutral input for one control cycle."""

    sensory: Sequence[float]
    context: Sequence[float]
    history: Sequence[float] = ()
    # A sequence keeps a test case reusable across repeated verification runs.
    candidates: Sequence[WorkspaceItem] = ()
    report: bool = True


@dataclass(frozen=True)
class ConnectionResult:
    """The only values made available to connected application consumers."""

    workspace: tuple[WorkspaceItem, ...]
    state: PhenomenalState
    signals: CognitiveReadout

    @property
    def attention(self) -> float:
        return self.signals.values["attention"]

    @property
    def memory(self) -> float:
        return self.signals.values["memory"]


Consumer = Callable[[ConnectionResult], None]


class ConsciousnessConnection:
    """Connect application feature encoders and consumers to core primitives.

    ``process`` is side-effect free except for updating the finite candidate
    store.  Use ``dispatch`` when consumers such as a planner or memory writer
    should receive the result.  This split makes connection tests deterministic.
    """

    def __init__(self, capacity: int = 8, consumers: Mapping[str, Consumer] | None = None) -> None:
        self.workspace = FiniteWorkspace(capacity=capacity)
        self._consumers: dict[str, Consumer] = dict(consumers or {})

    def register(self, name: str, consumer: Consumer) -> None:
        if not name:
            raise ValueError("consumer name must be non-empty")
        if not callable(consumer):
            raise TypeError("consumer must be callable")
        self._consumers[name] = consumer

    def process(self, input: ConnectionInput) -> ConnectionResult:
        self._validate_input(input)
        workspace = self.workspace.ingest(input.candidates)
        state = PhenomenalState.integrate(input.sensory, input.context, input.history)
        return ConnectionResult(workspace, state, readout(state, report=input.report))

    def dispatch(self, result: ConnectionResult, consumers: Sequence[str] | None = None) -> None:
        """Deliver a processed result to selected consumers in caller order."""
        names = tuple(self._consumers) if consumers is None else tuple(consumers)
        for name in names:
            try:
                consumer = self._consumers[name]
            except KeyError as error:
                raise ValueError(f"unknown consumer: {name}") from error
            consumer(result)

    def run(self, input: ConnectionInput, consumers: Sequence[str] | None = None) -> ConnectionResult:
        """Process one cycle and explicitly dispatch it."""
        result = self.process(input)
        self.dispatch(result, consumers)
        return result

    @staticmethod
    def _validate_input(input: ConnectionInput) -> None:
        values = tuple(input.sensory) + tuple(input.context) + tuple(input.history)
        if not values:
            raise ValueError("connection input requires at least one feature")
        try:
            invalid = next(value for value in values if not isfinite(float(value)))
        except TypeError as error:
            raise TypeError("features must be numeric") from error
        except StopIteration:
            invalid = None
        if invalid is not None:
            raise ValueError("features must be finite")


@dataclass(frozen=True)
class ConnectionCheck:
    """One portable, declarative assertion for an integration cycle."""

    name: str
    input: ConnectionInput
    expected_workspace: Sequence[str] = ()
    expect_report: bool | None = None
    memory_range: tuple[float, float] | None = None
    attention_range: tuple[float, float] | None = None


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    failures: tuple[str, ...] = ()


@dataclass(frozen=True)
class VerificationReport:
    results: tuple[CheckResult, ...]

    @property
    def passed(self) -> bool:
        return all(result.passed for result in self.results)

    @property
    def failures(self) -> tuple[str, ...]:
        return tuple(failure for result in self.results for failure in result.failures)


def verify_connection(connection: ConsciousnessConnection, checks: Iterable[ConnectionCheck]) -> VerificationReport:
    """Run adapter-independent checks without dispatching application effects."""
    results: list[CheckResult] = []
    for check in checks:
        failures: list[str] = []
        try:
            result = connection.process(check.input)
        except (TypeError, ValueError) as error:
            results.append(CheckResult(check.name, False, (str(error),)))
            continue
        actual = tuple(item.identifier for item in result.workspace)
        if check.expected_workspace and actual != tuple(check.expected_workspace):
            failures.append(f"workspace={actual!r}, expected={tuple(check.expected_workspace)!r}")
        if check.expect_report is not None and (result.signals.report is not None) != check.expect_report:
            failures.append("report availability did not match expectation")
        _check_range("memory", result.memory, check.memory_range, failures)
        _check_range("attention", result.attention, check.attention_range, failures)
        results.append(CheckResult(check.name, not failures, tuple(failures)))
    return VerificationReport(tuple(results))


def _check_range(name: str, value: float, expected: tuple[float, float] | None, failures: list[str]) -> None:
    if expected is None:
        return
    lower, upper = expected
    if lower > upper:
        failures.append(f"{name} range is invalid")
    elif not lower <= value <= upper:
        failures.append(f"{name}={value}, expected in [{lower}, {upper}]")
