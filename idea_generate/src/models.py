"""Core data contracts for the idea research workspace."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class RawNote:
    id: str
    text: str
    created_at: str = ""
    tags: list[str] = field(default_factory=list)


@dataclass
class AtomicClaim:
    id: str
    text: str
    claim_type: str
    subject: str | None = None
    relation: str | None = None
    object: str | None = None
    confidence: float = 0.5
    source_note_ids: list[str] = field(default_factory=list)


@dataclass
class Concept:
    id: str
    name: str
    description: str
    member_claim_ids: list[str]
    abstraction_level: int
    confidence: float
    support_count: int
    predictions: list[str] = field(default_factory=list)
    unresolved_questions: list[str] = field(default_factory=list)


@dataclass
class ResearchGap:
    id: str
    description: str
    target_hypothesis: str
    concept_id: str | None
    uncertainty: float
    importance: float
    information_gain: float
    novelty: float
    dependency_impact: float
    estimated_cost: float


@dataclass
class ExperimentProposal:
    id: str
    title: str
    target_hypothesis: str
    experiment_type: str
    procedure: list[str]
    intervention: str | None
    control_condition: str | None
    expected_result: str
    falsification_condition: str
    metrics: list[str]
    required_resources: list[str]
    estimated_cost: float
    assumptions: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    score: float = 0.0
    rationale: str = ""


@dataclass(frozen=True)
class ExperimentReview:
    testable: bool
    actually_tests_hypothesis: bool
    falsifiable: bool
    measurable: bool
    has_valid_control: bool | None
    duplicates_existing_test: bool
    problems: list[str] = field(default_factory=list)


@dataclass
class ResearchState:
    established: list[str] = field(default_factory=list)
    hypotheses: list[str] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)
    contradictions: list[str] = field(default_factory=list)


def to_dict(value: object) -> dict:
    """Serialize a workspace record without leaking dataclass details."""
    return asdict(value)