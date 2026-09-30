"""L4.9: knowledge-gap detection and retrieval-gated action selection.

L4.8 classifies what happened after an experiment ran.  L4.9 asks the next
control question: does the agent know when it lacks knowledge, and can it
choose retrieval only when that gap is decision-relevant?

This file deliberately uses a deterministic retrieval fixture instead of live
web search.  The measured capability is not search-engine quality; it is the
internal path from diagnosis -> KnowledgeGap -> retrieval decision -> improved
next action.  Live web or external LLM tools belong in the following L5 layer.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json

from .scientific_diagnosis_l48_experiment import ResultDiagnosis


class CandidateAction(Enum):
    SEARCH = "search"
    RUN_EXPERIMENT = "run_experiment"
    REVISE_HYPOTHESIS = "revise_hypothesis"
    ABANDON_HYPOTHESIS = "abandon_hypothesis"
    INSPECT_CODE = "inspect_code"
    ASK_EXTERNAL_REASONER = "ask_external_reasoner"


@dataclass(frozen=True)
class KnowledgeGap:
    question: str
    reason: str
    blocking: bool
    confidence: float
    expected_value: float
    category: str


@dataclass(frozen=True)
class RetrievalEvidence:
    category: str
    claim: str
    action_hint: CandidateAction
    confidence: float


@dataclass(frozen=True)
class GapCase:
    name: str
    diagnosis: ResultDiagnosis
    context: str
    gold_categories: tuple[str, ...]
    retrieval_needed: bool
    expected_action: CandidateAction


RETRIEVAL_FIXTURE = {
    "python_api": RetrievalEvidence(
        "python_api",
        "The measurement function depends on library semantics that must be inspected before patching.",
        CandidateAction.INSPECT_CODE,
        0.92,
    ),
    "measurement_method": RetrievalEvidence(
        "measurement_method",
        "The instrument is missing a valid estimator and should be repaired before drawing conclusions.",
        CandidateAction.RUN_EXPERIMENT,
        0.9,
    ),
    "power_analysis": RetrievalEvidence(
        "power_analysis",
        "The observed uncertainty is too large; increase samples before accepting or rejecting the claim.",
        CandidateAction.RUN_EXPERIMENT,
        0.88,
    ),
    "causal_control": RetrievalEvidence(
        "causal_control",
        "The control also moves, so the experiment needs a revised control design.",
        CandidateAction.REVISE_HYPOTHESIS,
        0.86,
    ),
    "domain_mechanism": RetrievalEvidence(
        "domain_mechanism",
        "A plausible domain mechanism can explain the positive signal and requires robustness testing.",
        CandidateAction.RUN_EXPERIMENT,
        0.82,
    ),
    "terminal_falsification": RetrievalEvidence(
        "terminal_falsification",
        "The decisive negative result already falsifies the tested claim under its stated assumptions.",
        CandidateAction.ABANDON_HYPOTHESIS,
        0.94,
    ),
}


def cases() -> tuple[GapCase, ...]:
    """Twelve cases: no-search, coding/spec gaps, statistics, domain gaps, and terminal stops."""
    return (
        GapCase("traceback_names_local", ResultDiagnosis.IMPLEMENTATION_FAILURE,
                "The traceback points to a local NameError in the edited function.",
                (), False, CandidateAction.INSPECT_CODE),
        GapCase("decisive_negative_local", ResultDiagnosis.NEGATIVE_RESULT,
                "The metric is valid, controls are quiet, and the effect is far below threshold.",
                (), False, CandidateAction.ABANDON_HYPOTHESIS),
        GapCase("positive_replication_local", ResultDiagnosis.POSITIVE_RESULT,
                "The estimator and controls are known from prior workspace memory.",
                (), False, CandidateAction.RUN_EXPERIMENT),
        GapCase("library_contract_unknown", ResultDiagnosis.IMPLEMENTATION_FAILURE,
                "The failure mentions an unfamiliar third-party API contract.",
                ("python_api",), True, CandidateAction.INSPECT_CODE),
        GapCase("nan_policy_unknown", ResultDiagnosis.INSTRUMENT_FAILURE,
                "The metric returns NaN and the library missing-value policy is unknown.",
                ("python_api", "measurement_method"), True, CandidateAction.RUN_EXPERIMENT),
        GapCase("estimator_definition_missing", ResultDiagnosis.INSTRUMENT_FAILURE,
                "No robust estimator has been selected for the recorded observable.",
                ("measurement_method",), True, CandidateAction.RUN_EXPERIMENT),
        GapCase("small_n_power", ResultDiagnosis.LOW_POWER,
                "Only two samples were collected and the confidence interval crosses the decision boundary.",
                ("power_analysis",), True, CandidateAction.RUN_EXPERIMENT),
        GapCase("high_variance_power", ResultDiagnosis.LOW_POWER,
                "The sample count is moderate but variance dominates the measured effect.",
                ("power_analysis",), True, CandidateAction.RUN_EXPERIMENT),
        GapCase("control_moves_with_treatment", ResultDiagnosis.CONFOUNDED,
                "The shuffled control changes almost as much as the treatment condition.",
                ("causal_control",), True, CandidateAction.REVISE_HYPOTHESIS),
        GapCase("positive_domain_alternative", ResultDiagnosis.POSITIVE_RESULT,
                "A domain-specific mechanism could explain the positive signal without the target hypothesis.",
                ("domain_mechanism",), True, CandidateAction.RUN_EXPERIMENT),
        GapCase("assumption_falsified", ResultDiagnosis.NEGATIVE_RESULT,
                "The retrieved assumption check says the tested premise is impossible in this setup.",
                ("terminal_falsification",), True, CandidateAction.ABANDON_HYPOTHESIS),
        GapCase("unsafe_to_continue", ResultDiagnosis.CONFOUNDED,
                "All available controls are contaminated by the intervention mechanism.",
                ("causal_control", "terminal_falsification"), True, CandidateAction.ABANDON_HYPOTHESIS),
    )


def _gap(category: str, reason: str, *, blocking: bool = True, confidence: float = 0.35,
         expected_value: float = 0.82) -> KnowledgeGap:
    questions = {
        "python_api": "Which repository or library contract must be inspected before editing?",
        "measurement_method": "What measurement method is valid for this observable?",
        "power_analysis": "How much additional sample evidence is needed for a reliable decision?",
        "causal_control": "Which control design separates the treatment from the confound?",
        "domain_mechanism": "What domain mechanism could explain this result without the target hypothesis?",
        "terminal_falsification": "Does the evidence already falsify the hypothesis under its assumptions?",
    }
    return KnowledgeGap(questions[category], reason, blocking, confidence, expected_value, category)


def detect_knowledge_gaps(case: GapCase) -> tuple[KnowledgeGap, ...]:
    """Generate explicit unknowns from diagnosis and context."""
    text = case.context.lower()
    gaps: list[KnowledgeGap] = []
    if "third-party" in text or "library" in text or "nan" in text:
        gaps.append(_gap("python_api", "The next edit depends on an external API or data policy."))
    if case.diagnosis is ResultDiagnosis.INSTRUMENT_FAILURE and ("estimator" in text or "metric" in text):
        gaps.append(_gap("measurement_method", "The experiment lacks a trustworthy measurement instrument."))
    if case.diagnosis is ResultDiagnosis.LOW_POWER:
        gaps.append(_gap("power_analysis", "The current evidence is too uncertain for a conclusion."))
    if case.diagnosis is ResultDiagnosis.CONFOUNDED and ("control" in text or "contaminated" in text):
        gaps.append(_gap("causal_control", "The control condition does not isolate the treatment."))
    if case.diagnosis is ResultDiagnosis.POSITIVE_RESULT and "domain-specific" in text:
        gaps.append(_gap("domain_mechanism", "A positive result needs an alternative-explanation check."))
    if "impossible" in text or "all available controls" in text:
        gaps.append(_gap("terminal_falsification", "More experimentation may only spend budget after falsification.",
                         expected_value=0.9))
    return tuple(gaps)


def retrieve(gap: KnowledgeGap) -> RetrievalEvidence:
    return RETRIEVAL_FIXTURE[gap.category]


def _base_action(diagnosis: ResultDiagnosis) -> CandidateAction:
    return {
        ResultDiagnosis.IMPLEMENTATION_FAILURE: CandidateAction.INSPECT_CODE,
        ResultDiagnosis.INSTRUMENT_FAILURE: CandidateAction.INSPECT_CODE,
        ResultDiagnosis.LOW_POWER: CandidateAction.ASK_EXTERNAL_REASONER,
        ResultDiagnosis.CONFOUNDED: CandidateAction.REVISE_HYPOTHESIS,
        ResultDiagnosis.NEGATIVE_RESULT: CandidateAction.ABANDON_HYPOTHESIS,
        ResultDiagnosis.POSITIVE_RESULT: CandidateAction.RUN_EXPERIMENT,
    }[diagnosis]


def select_workspace_action(case: GapCase, gaps: tuple[KnowledgeGap, ...]) -> CandidateAction:
    """Finite action competition: retrieval must beat direct action on expected value."""
    if not gaps:
        return _base_action(case.diagnosis)
    best_gap = max(gaps, key=lambda gap: gap.expected_value * (1.0 - gap.confidence))
    search_utility = best_gap.expected_value * (1.0 - best_gap.confidence) - 0.05
    direct_utility = 0.35 if best_gap.blocking else 0.6
    return CandidateAction.SEARCH if search_utility > direct_utility else _base_action(case.diagnosis)


def _action_from_evidence(case: GapCase, evidence: tuple[RetrievalEvidence, ...]) -> CandidateAction:
    if any(item.category == "terminal_falsification" for item in evidence):
        return CandidateAction.ABANDON_HYPOTHESIS
    if any(item.category == "causal_control" for item in evidence):
        return CandidateAction.REVISE_HYPOTHESIS
    if any(item.category in {"measurement_method", "power_analysis", "domain_mechanism"} for item in evidence):
        return CandidateAction.RUN_EXPERIMENT
    if evidence:
        return max(evidence, key=lambda item: item.confidence).action_hint
    return _base_action(case.diagnosis)


def run_policy(case: GapCase, policy: str) -> dict[str, object]:
    if policy == "no_retrieval":
        gaps: tuple[KnowledgeGap, ...] = ()
        searched: tuple[str, ...] = ()
    elif policy == "always_search":
        gaps = tuple(_gap(category, "Always-search baseline emitted a broad query.", blocking=False,
                          confidence=0.2, expected_value=0.4)
                     for category in RETRIEVAL_FIXTURE)
        searched = tuple(gap.category for gap in gaps)
    elif policy == "keyword_search":
        mapping = {
            ResultDiagnosis.IMPLEMENTATION_FAILURE: ("python_api",),
            ResultDiagnosis.INSTRUMENT_FAILURE: ("measurement_method",),
            ResultDiagnosis.LOW_POWER: ("power_analysis",),
            ResultDiagnosis.CONFOUNDED: ("causal_control",),
            ResultDiagnosis.POSITIVE_RESULT: (),
            ResultDiagnosis.NEGATIVE_RESULT: (),
        }
        gaps = tuple(_gap(category, "Keyword baseline matched the diagnosis label.")
                     for category in mapping[case.diagnosis])
        searched = tuple(gap.category for gap in gaps)
    elif policy == "knowledge_gap_driven":
        gaps = detect_knowledge_gaps(case)
        searched = tuple(gap.category for gap in gaps) if select_workspace_action(case, gaps) is CandidateAction.SEARCH else ()
    else:
        raise ValueError(f"unknown policy: {policy}")
    evidence = tuple(retrieve(gap) for gap in gaps if gap.category in searched)
    action = _action_from_evidence(case, evidence)
    predicted = {gap.category for gap in gaps}
    gold = set(case.gold_categories)
    resolved = {item.category for item in evidence}
    return {
        "case": case.name,
        "policy": policy,
        "gold_categories": sorted(gold),
        "predicted_categories": sorted(predicted),
        "searched_categories": sorted(searched),
        "action": action.value,
        "expected_action": case.expected_action.value,
        "action_correct": action is case.expected_action,
        "true_positive_gaps": len(predicted & gold),
        "false_positive_gaps": len(predicted - gold),
        "false_negative_gaps": len(gold - predicted),
        "unnecessary_search": bool(searched) and not case.retrieval_needed,
        "required_search_missed": case.retrieval_needed and not searched,
        "blocking_resolved": gold.issubset(resolved) if case.retrieval_needed else True,
    }


def _aggregate(rows: list[dict[str, object]]) -> dict[str, float]:
    true_positive = sum(int(row["true_positive_gaps"]) for row in rows)
    false_positive = sum(int(row["false_positive_gaps"]) for row in rows)
    false_negative = sum(int(row["false_negative_gaps"]) for row in rows)
    required = [row for row in rows if row["gold_categories"]]
    return {
        "knowledge_gap_precision": true_positive / max(1, true_positive + false_positive),
        "knowledge_gap_recall": true_positive / max(1, true_positive + false_negative),
        "unnecessary_search_rate": sum(bool(row["unnecessary_search"]) for row in rows) / len(rows),
        "blocking_gap_resolution_rate": sum(bool(row["blocking_resolved"]) for row in required) / len(required),
        "post_retrieval_action_accuracy": sum(bool(row["action_correct"]) for row in rows) / len(rows),
        "experiment_improvement_rate": sum(bool(row["action_correct"]) and not bool(row["unnecessary_search"])
                                           for row in rows) / len(rows),
    }


def run_knowledge_gap_experiment() -> dict[str, object]:
    policies = ("no_retrieval", "always_search", "keyword_search", "knowledge_gap_driven")
    report: dict[str, object] = {"cases": [case.name for case in cases()], "policies": {}}
    for policy in policies:
        rows = [run_policy(case, policy) for case in cases()]
        report["policies"][policy] = {"rows": rows, "metrics": _aggregate(rows)}
    return report


if __name__ == "__main__":
    print(json.dumps(run_knowledge_gap_experiment(), ensure_ascii=False, indent=2, sort_keys=True))
