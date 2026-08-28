"""Public API for the functional consciousness model and its experiments."""

from .core import (
    DiscoveryConfig,
    KNNModel,
    Law,
    Observation,
    RegimeLaw,
    SurrogatePrediction,
    discover_power_law,
    discover_regime,
    evaluate_ood,
    generate_noisy_observations,
    integrate_rk4,
)
from .consciousness import (
    CognitiveReadout,
    FiniteWorkspace,
    PhenomenalState,
    UncertaintySummary,
    WorkspaceItem,
    readout,
    summarize_uncertainty,
)
from .consciousness_model import (
    ConceptualAction,
    ConsciousnessController,
    IntegratedState,
    SelfObservation,
)
from .compressed_workspace import (
    CompressedWorkspace,
    Concept,
    Relation,
)

__all__ = [
    "DiscoveryConfig",
    "KNNModel",
    "Law",
    "Observation",
    "RegimeLaw",
    "SurrogatePrediction",
    "discover_power_law",
    "discover_regime",
    "evaluate_ood",
    "generate_noisy_observations",
    "integrate_rk4",
    "CognitiveReadout",
    "FiniteWorkspace",
    "PhenomenalState",
    "UncertaintySummary",
    "WorkspaceItem",
    "readout",
    "summarize_uncertainty",
    "ConceptualAction",
    "ConsciousnessController",
    "IntegratedState",
    "SelfObservation",
    "CompressedWorkspace",
    "Concept",
    "Relation",
]
