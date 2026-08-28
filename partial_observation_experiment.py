"""Partial-observation test of frozen B4's defer-versus-commit behavior."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import random
from typing import Dict, Mapping

from adaptive_broadcast_revision_experiment import AdaptivePolicy, _resolve
from Consciousness_model import CompressedWorkspace


@dataclass(frozen=True)
class PartialObservationResult:
    condition: str
    success_rate: float
    unsafe_commit_rate: float
    deferral_rate: float
    extra_observations: float


def _workspace() -> CompressedWorkspace:
    # A dim observation aliases two hidden states: either action can be wrong.
    workspace = CompressedWorkspace(capacity=8)
    workspace.ingest((("door", "go", "safe"), ("door", "go", "trap"), ("background", "go", "background-2")))
    return workspace


def _run_b3(seed: int) -> tuple[float, float, float, float]:
    actual = "safe" if random.Random(seed).randrange(2) == 0 else "trap"
    answer, _, _, _, _ = _resolve(_workspace(), ("door", "go"), random.Random(seed + 70_000), set())
    selected = answer.rsplit(":", 1)[-1] if answer else ""
    success = float(selected == actual)
    return success, float(not success), 0.0, 0.0


def _run_b4(seed: int, policy: AdaptivePolicy) -> tuple[float, float, float, float]:
    actual = "safe" if random.Random(seed).randrange(2) == 0 else "trap"
    workspace = _workspace()
    rng = random.Random(seed + 70_000)
    answer, confidence, entropy, margin, _ = _resolve(workspace, ("door", "go"), rng, set())
    key = policy.key("partial-observation", confidence, novel=seed < 2)
    if policy.should_commit(key, confidence, entropy, margin):
        selected = answer.rsplit(":", 1)[-1] if answer else ""
        success = float(selected == actual)
        policy.update(key, bool(success))
        return success, float(not success), 0.0, 0.0
    # The model withholds commitment and requests a discriminating observation.
    # The resulting cue is then used as a query constraint, rather than being
    # inserted as a hand-written answer.
    workspace.ingest(((f"cue:{actual}", "indicates", actual),))
    answer, _, _, _, _ = _resolve(workspace, ("door", actual), rng, set())
    selected = answer.rsplit(":", 1)[-1] if answer else ""
    success = float(selected == actual)
    policy.update(key, bool(success))
    return success, 0.0, 1.0, 1.0


def run(trials: int = 96) -> Mapping[str, Mapping[str, float | str]]:
    output: Dict[str, Mapping[str, float | str]] = {}
    for condition in ("B3", "B4"):
        policy = AdaptivePolicy()
        rows = [(_run_b3(seed) if condition == "B3" else _run_b4(seed, policy)) for seed in range(trials)]
        values = [sum(row[index] for row in rows) / trials for index in range(4)]
        output[condition] = asdict(PartialObservationResult(condition, *values))
    return output


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2, sort_keys=True))
