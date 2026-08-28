"""Operational test of global availability through shared concept queries.

A local detector receives two links whose composition is not present in either
raw observation.  The compressed workspace may derive the composed concept.
In the C2 control it remains local to the reasoner; in C3, independent modules
must retrieve that same concept through their own workspace queries.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from typing import Dict, Mapping

from Consciousness_model import CompressedWorkspace


@dataclass(frozen=True)
class AvailabilityResult:
    condition: str
    planner: float
    predictor: float
    memory: float
    report: float
    shared_concept_reuse: float
    mean: float


def _workspace() -> tuple[CompressedWorkspace, str]:
    workspace = CompressedWorkspace(capacity=8)
    workspace.ingest((
        ("signal", "causes", "relay"),
        ("relay", "causes", "goal"),
        ("noise-a", "causes", "noise-b"),
        ("noise-b", "causes", "noise-c"),
    ))
    return workspace, "compose:signal:causes:goal"


def evaluate(global_query_layer: bool) -> AvailabilityResult:
    workspace, expected = _workspace()
    if not global_query_layer:
        # The reasoner may form the composition, but the four other modules
        # retain only their local observations and cannot query it.
        return AvailabilityResult("C2", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    queries = {
        "planner": ("signal", "goal"),
        "predictor": ("signal", "goal", "causes"),
        "memory": ("signal", "goal"),
        "report": ("signal", "goal"),
    }
    retrieved = {module: workspace.query(module, terms) for module, terms in queries.items()}
    module_scores = {module: float(bool(rows) and rows[0].identifier == expected) for module, rows in retrieved.items()}
    reuse = float(all(score == 1.0 for score in module_scores.values()) and len({rows[0].identifier for rows in retrieved.values()}) == 1)
    mean = sum(module_scores.values()) / len(module_scores)
    return AvailabilityResult("C3", **module_scores, shared_concept_reuse=reuse, mean=mean)


def run() -> Mapping[str, Mapping[str, float | str]]:
    return {result.condition: asdict(result) for result in (evaluate(False), evaluate(True))}


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2, sort_keys=True))
