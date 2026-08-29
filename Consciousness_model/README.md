# Consciousness_model

Reusable, dependency-free primitives for the project.

## Consciousness API

```python
from Consciousness_model import (
    FiniteWorkspace,
    PhenomenalState,
    WorkspaceItem,
    summarize_uncertainty,
)
```

`FiniteWorkspace` keeps a bounded broadcast set while retaining all candidates.
Use `summarize_uncertainty` on the full probability distribution; pass
`visible_count` to add an `OTHER` tail without renormalizing Top-K.

`PhenomenalState.integrate` creates a private integrated state from sensory,
context, and history values. `readout` exposes downstream values and can ablate
only the report channel.

The package provides functional mechanisms only. It does not establish
subjective experience.

## Connecting another project

`ConsciousnessConnection` is the application-neutral boundary.  Convert your
application state into numeric features and `WorkspaceItem` candidates, then
give the result to named consumers.  Processing and dispatching are separate so
connection tests do not accidentally write memory or execute actions.

```python
from Consciousness_model import (
    ConsciousnessConnection, ConnectionCheck, ConnectionInput, WorkspaceItem,
    verify_connection,
)

connection = ConsciousnessConnection(capacity=2)
connection.register("planner", lambda result: planner.plan(
    result.workspace, memory=result.memory, attention=result.attention
))
cycle = ConnectionInput(
    sensory=(0.8, 0.1), context=(0.4,),
    candidates=(WorkspaceItem("hazard", 0.9), WorkspaceItem("goal", 0.7)),
)
result = connection.run(cycle, consumers=("planner",))

report = verify_connection(connection, [
    ConnectionCheck("hazard is selected", cycle, expected_workspace=("hazard", "goal")),
])
assert report.passed, report.failures
```

Use `process` rather than `run` when a caller wants the result without invoking
registered consumers.  `verify_connection` never dispatches consumers; it
checks capacity selection, report availability, and optional signal ranges.

## Physical API

The law-discovery and numerical integration API remains available from
`Consciousness_model`, including `Observation`, `Law`, `discover_power_law`, and
`integrate_rk4`.
