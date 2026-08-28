"""A small, reproducible C0--C3 functional-consciousness ablation.

This is intentionally a *functional* benchmark.  It tests whether finite
working memory, compression, and a broadcast interface make different
cognitive functions available; it makes no claim about subjective experience.

The local perceptual component uses :class:`physics_model.KNNModel`, so the
benchmark can be run against the model that currently exists in this project.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from typing import Dict, Iterable, List, Mapping, Tuple

from Consciousness_model import KNNModel


@dataclass(frozen=True)
class Condition:
    name: str
    finite_working_memory: bool
    compression: bool
    global_broadcast: bool


CONDITIONS = (
    Condition("C0", False, False, False),
    Condition("C1", True, False, False),
    Condition("C2", True, True, False),
    Condition("C3", True, True, True),
)


@dataclass(frozen=True)
class Score:
    global_availability: float
    novel_problem_solving: float
    limited_capacity_selection: float
    flexible_planning: float
    metacognition: float

    @property
    def mean(self) -> float:
        return sum(asdict(self).values()) / 5.0


class Workspace:
    """The smallest architecture needed to express the C0--C3 distinction."""

    def __init__(self, condition: Condition, capacity: int = 3) -> None:
        self.condition = condition
        self.capacity = capacity
        self.raw: List[Tuple[str, str]] = []
        self.concepts: Dict[str, set[str]] = {}

    def observe(self, subject: str, predicate: str, important: bool = False) -> None:
        if not self.condition.finite_working_memory:
            return
        if self.condition.compression:
            # A concept is a lossier, reusable representation: duplicate facts
            # collapse and task-relevant facts survive competing observations.
            if important or len(self.concepts) < self.capacity or subject in self.concepts:
                self.concepts.setdefault(subject, set()).add(predicate)
            return
        self.raw.append((subject, predicate))
        self.raw = self.raw[-self.capacity :]

    def has(self, subject: str, predicate: str, module: str) -> bool:
        if module != "concept" and not self.condition.global_broadcast:
            return False
        if self.condition.compression:
            return predicate in self.concepts.get(subject, set())
        return (subject, predicate) in self.raw

    def compose(self, left: str, middle: str, right: str) -> bool:
        """Infer left -> right from left -> middle and middle -> right."""
        if not self.condition.compression:
            return False
        return middle in self.concepts.get(left, set()) and right in self.concepts.get(middle, set())


def _local_perception() -> float:
    """Exercise the existing local model, separately from the workspace."""
    model = KNNModel(neighbors=1).fit(((0.0,), (1.0,)), (0.0, 1.0))
    return float(model.predict((1.0,)).value == 1.0)


def evaluate(condition: Condition) -> Score:
    # Information first detected by a local module; later consumers need broadcast.
    workspace = Workspace(condition)
    workspace.observe("signal", "red")
    global_availability = float(_local_perception() and workspace.has("signal", "red", "action"))

    # Unseen composition: striped -> triangle and triangle -> metal entails striped -> metal.
    workspace = Workspace(condition)
    workspace.observe("striped", "triangle")
    workspace.observe("triangle", "metal")
    novel_problem_solving = float(workspace.compose("striped", "triangle", "metal"))

    # Four items exceed raw WM capacity.  The goal item comes first, so C1 loses it;
    # C2/C3 retain it through task-relevant compression.
    workspace = Workspace(condition)
    workspace.observe("goal", "green", important=True)
    for item in ("noise-a", "noise-b"):
        workspace.observe(item, "irrelevant")
    short_delay = workspace.has("goal", "green", "concept")
    workspace.observe("noise-c", "irrelevant")
    workspace.observe("noise-d", "irrelevant")
    long_delay = workspace.has("goal", "green", "concept")
    limited_capacity_selection = (float(short_delay) + float(long_delay)) / 2.0

    # Planning is deliberately an action-module readout of a compressed state.
    workspace = Workspace(condition)
    workspace.observe("at-key", "door-open")
    workspace.observe("door-open", "reach-goal")
    flexible_planning = float(
        workspace.compose("at-key", "door-open", "reach-goal")
        and condition.global_broadcast
    )

    # A calibrated report is only possible when the compressed state is globally
    # available to the report module.  Score both a known and an unknown probe.
    workspace = Workspace(condition)
    workspace.observe("target", "present")
    reports: Iterable[Tuple[bool, bool]] = (
        (workspace.has("target", "present", "report"), True),
        (workspace.has("absent", "present", "report"), False),
    )
    metacognition = sum(float(answer == truth) for answer, truth in reports) / 2.0
    return Score(global_availability, novel_problem_solving, limited_capacity_selection, flexible_planning, metacognition)


def run() -> Mapping[str, Mapping[str, float]]:
    result: Dict[str, Mapping[str, float]] = {}
    for condition in CONDITIONS:
        score = evaluate(condition)
        result[condition.name] = {**asdict(score), "mean": score.mean}
    return result


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2, sort_keys=True))
