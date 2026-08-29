import pytest

from Consciousness_model import (
    ConnectionCheck,
    ConnectionInput,
    ConsciousnessConnection,
    WorkspaceItem,
    verify_connection,
)


def test_connection_exposes_bounded_workspace_and_signals_to_consumer():
    received = []
    connection = ConsciousnessConnection(capacity=2)
    connection.register("planner", received.append)
    cycle = ConnectionInput(
        sensory=(0.8, 0.2), context=(0.4,),
        candidates=(WorkspaceItem("low", 0.1), WorkspaceItem("high", 0.9), WorkspaceItem("middle", 0.5)),
    )

    result = connection.run(cycle, consumers=("planner",))

    assert tuple(item.identifier for item in result.workspace) == ("high", "middle")
    assert result.memory == pytest.approx((0.8 + 0.2 + 0.4) / 3)
    assert result.attention > 0
    assert received == [result]


def test_process_does_not_dispatch_and_verification_has_no_application_effects():
    received = []
    connection = ConsciousnessConnection(consumers={"memory": received.append})
    cycle = ConnectionInput(sensory=(1.0,), context=(), candidates=(WorkspaceItem("fact", 1.0),), report=False)

    connection.process(cycle)
    report = verify_connection(connection, [
        ConnectionCheck("contract", cycle, expected_workspace=("fact",), expect_report=False, memory_range=(1.0, 1.0)),
    ])

    assert report.passed
    assert received == []


def test_verification_reports_contract_failures_and_invalid_features():
    connection = ConsciousnessConnection()
    report = verify_connection(connection, [
        ConnectionCheck("wrong", ConnectionInput(sensory=(float("nan"),), context=())),
    ])

    assert not report.passed
    assert "features must be finite" in report.failures
