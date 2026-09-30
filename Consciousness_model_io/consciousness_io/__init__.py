"""consciousness_io — the functional consciousness model with configurable I/O.

This package keeps the original, dependency-free ``consciousness_model``
primitives (finite/compressed workspace, phenomenal state, application bridge,
physical law discovery) and adds a **set-theoretic agent whose input and output
are user-configured**:

    C_t = (O_t, B_t, P_t, G_t, Q_t, A_t, W_t),   C_{t+1} = F(C_t, O_{t+1}, a_t)

Quick start::

    from consciousness_io import (
        ConsciousAgent, AgentConfig, SensorChannel, ActionSchema,
        PredictedEffect, Fact, fact,
    )

    camera = SensorChannel("camera", encode=my_vision_to_facts)
    grasp = ActionSchema("GRASP", effect=my_grasp_effect)
    agent = ConsciousAgent([camera], [grasp], goals=[fact("HAVE", "robot", "cup")])
    state = agent.step({"camera": frame})
    print(state.summary())

The package provides functional mechanisms only.  It does not establish
subjective experience.
"""

# --- new configurable-I/O agent -------------------------------------------
from .facts import BeliefStore, Fact, fact, goal_distance
from .io_config import (
    ActionBinding,
    ActionSchema,
    AgentConfig,
    PredictedEffect,
    SensorChannel,
)
from .agent import (
    ConsciousAgent,
    ConsciousState,
    Decision,
    Prediction,
    Question,
)

# --- original primitives (unchanged) --------------------------------------
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
from .integration import (
    CheckResult,
    ConnectionCheck,
    ConnectionInput,
    ConnectionResult,
    ConsciousnessConnection,
    VerificationReport,
    verify_connection,
)

__all__ = [
    # configurable-I/O agent
    "ConsciousAgent",
    "ConsciousState",
    "Decision",
    "Prediction",
    "Question",
    "AgentConfig",
    "SensorChannel",
    "ActionSchema",
    "ActionBinding",
    "PredictedEffect",
    "Fact",
    "fact",
    "BeliefStore",
    "goal_distance",
    # physical primitives
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
    # consciousness primitives
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
    "CheckResult",
    "ConnectionCheck",
    "ConnectionInput",
    "ConnectionResult",
    "ConsciousnessConnection",
    "VerificationReport",
    "verify_connection",
]
