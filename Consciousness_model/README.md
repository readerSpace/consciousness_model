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

## Physical API

The law-discovery and numerical integration API remains available from
`Consciousness_model`, including `Observation`, `Law`, `discover_power_law`, and
`integrate_rk4`.
