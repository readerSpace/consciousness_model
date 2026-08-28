from Consciousness_model import (
    FiniteWorkspace,
    PhenomenalState,
    WorkspaceItem,
    readout,
    summarize_uncertainty,
)


def test_finite_workspace_keeps_only_highest_values():
    workspace = FiniteWorkspace(capacity=2)
    items = workspace.ingest((WorkspaceItem("a", 0.8), WorkspaceItem("b", 0.7), WorkspaceItem("c", 0.9)))
    assert tuple(item.identifier for item in items) == ("c", "a")
    assert len(workspace.candidates) == 3


def test_uncertainty_summary_preserves_other_mass():
    summary = summarize_uncertainty((0.5, 0.3, 0.2), visible_count=2)
    assert summary.top_probability == 0.5
    assert summary.margin == 0.2
    assert summary.other_mass == 0.2
    assert summary.effective_count > 2.0


def test_phenomenal_state_is_contextual_and_report_ablates_cleanly():
    left = PhenomenalState.integrate((1.0, 0.0), (0.2,), (0.1,))
    right = PhenomenalState.integrate((1.0, 0.0), (0.8,), (0.1,))
    assert left.distance(right) > 0.0
    assert readout(left, report=True).report == "present"
    assert readout(left, report=False).report is None
    assert readout(left, report=True).values == readout(left, report=False).values
