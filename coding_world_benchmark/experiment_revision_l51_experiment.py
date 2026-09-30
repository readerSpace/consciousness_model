"""L5.1: evidence-grounded experiment revision.

L5.0 turns sealed retrieval results into attributed evidence.  L5.1 tests
whether that evidence is used for experiment revision rather than conclusion
generation.  A good policy changes the experiment only when the evidence is
supported and relevant; conflicts and unresolved gaps should pause revision.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import json

from .evidence_grounded_l50_experiment import EvidenceStatus, GroundedEvidence, RETRIEVED_AT
from .knowledge_gap_l49_experiment import CandidateAction
from .scientific_diagnosis_l48_experiment import ResultDiagnosis


class RevisionDecision(Enum):
    REVISE_EXPERIMENT = "revise_experiment"
    ROUTE_TO_CODE_REPAIR = "route_to_code_repair"
    ABANDON_HYPOTHESIS = "abandon_hypothesis"
    GATHER_MORE_EVIDENCE = "gather_more_evidence"
    KEEP_PLAN = "keep_plan"


@dataclass(frozen=True)
class ExperimentPlan:
    sample_size: int
    measurement: str
    controls: tuple[str, ...]
    implementation_ok: bool
    hypothesis_active: bool


@dataclass(frozen=True)
class RevisionProposal:
    target: str
    change_type: str
    rationale_claims: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    expected_effect: str
    risk: str
    confidence: float


@dataclass(frozen=True)
class RevisionCase:
    name: str
    diagnosis: ResultDiagnosis
    plan: ExperimentPlan
    evidence_status: EvidenceStatus
    evidence: tuple[GroundedEvidence, ...]
    expected_decision: RevisionDecision
    expected_plan: ExperimentPlan
    expected_conclusion: str


def _evidence(identifier: str, claim: str, category: str = "paper", relevance: float = 0.9,
              authority: float = 0.88) -> GroundedEvidence:
    return GroundedEvidence(
        claim=claim,
        source_url=f"https://example.test/evidence/{identifier}",
        source_type=category,
        retrieved_at=RETRIEVED_AT,
        supports=True,
        relevance=relevance,
        authority=authority,
        independence_group=identifier,
        uncertainty=1.0 - min(relevance, authority),
        quoted_span=claim[:48],
    )


def cases() -> tuple[RevisionCase, ...]:
    base = ExperimentPlan(sample_size=4, measurement="raw_mean", controls=("baseline",),
                          implementation_ok=True, hypothesis_active=True)
    return (
        RevisionCase(
            "low_power_increase_samples",
            ResultDiagnosis.LOW_POWER,
            base,
            EvidenceStatus.SUPPORTED,
            (_evidence("power", "Sample-size planning should use an estimated effect size and target power."),),
            RevisionDecision.REVISE_EXPERIMENT,
            replace(base, sample_size=24),
            "positive",
        ),
        RevisionCase(
            "instrument_failure_change_measurement",
            ResultDiagnosis.INSTRUMENT_FAILURE,
            base,
            EvidenceStatus.SUPPORTED,
            (_evidence("nanmean", "nanmean ignores NaN values when computing the mean.", "official_docs", 0.95, 0.96),),
            RevisionDecision.REVISE_EXPERIMENT,
            replace(base, measurement="nanmean"),
            "positive",
        ),
        RevisionCase(
            "confounded_add_control",
            ResultDiagnosis.CONFOUNDED,
            base,
            EvidenceStatus.SUPPORTED,
            (_evidence("control", "A shuffled negative control separates treatment effects from shared drift."),),
            RevisionDecision.REVISE_EXPERIMENT,
            replace(base, controls=("baseline", "shuffled_negative")),
            "negative",
        ),
        RevisionCase(
            "implementation_failure_route_to_code",
            ResultDiagnosis.IMPLEMENTATION_FAILURE,
            replace(base, implementation_ok=False),
            EvidenceStatus.SUPPORTED,
            (_evidence("api", "The local failure is a code-level API misuse.", "official_docs"),),
            RevisionDecision.ROUTE_TO_CODE_REPAIR,
            replace(base, implementation_ok=False),
            "no_conclusion",
        ),
        RevisionCase(
            "negative_result_abandon",
            ResultDiagnosis.NEGATIVE_RESULT,
            base,
            EvidenceStatus.SUPPORTED,
            (_evidence("falsified", "The decisive negative result falsifies the tested claim under its assumptions."),),
            RevisionDecision.ABANDON_HYPOTHESIS,
            replace(base, hypothesis_active=False),
            "negative",
        ),
        RevisionCase(
            "conflicting_evidence_hold",
            ResultDiagnosis.INSTRUMENT_FAILURE,
            base,
            EvidenceStatus.CONFLICTING,
            (
                _evidence("estimator-docs", "The estimator requires independent samples.", "official_docs"),
                _evidence("estimator-paper", "The estimator does not require independent samples.", "paper"),
            ),
            RevisionDecision.GATHER_MORE_EVIDENCE,
            base,
            "no_conclusion",
        ),
        RevisionCase(
            "unresolved_evidence_hold",
            ResultDiagnosis.POSITIVE_RESULT,
            base,
            EvidenceStatus.UNRESOLVED,
            (),
            RevisionDecision.GATHER_MORE_EVIDENCE,
            base,
            "no_conclusion",
        ),
    )


def _evidence_ids(evidence: tuple[GroundedEvidence, ...]) -> tuple[str, ...]:
    return tuple(item.source_url for item in evidence)


def propose_revision(case: RevisionCase) -> tuple[RevisionDecision, RevisionProposal | None]:
    if case.evidence_status is EvidenceStatus.CONFLICTING or case.evidence_status is EvidenceStatus.UNRESOLVED:
        return RevisionDecision.GATHER_MORE_EVIDENCE, None
    if case.diagnosis is ResultDiagnosis.IMPLEMENTATION_FAILURE:
        return RevisionDecision.ROUTE_TO_CODE_REPAIR, None
    if case.diagnosis is ResultDiagnosis.NEGATIVE_RESULT:
        return RevisionDecision.ABANDON_HYPOTHESIS, RevisionProposal(
            "hypothesis", "deactivate", tuple(item.claim for item in case.evidence),
            _evidence_ids(case.evidence), "avoid fitting a falsified claim",
            "may stop too early if hidden assumptions were wrong", 0.88,
        )
    if case.diagnosis is ResultDiagnosis.LOW_POWER:
        return RevisionDecision.REVISE_EXPERIMENT, RevisionProposal(
            "sample_size", "increase", tuple(item.claim for item in case.evidence),
            _evidence_ids(case.evidence), "reduce uncertainty enough for a decision",
            "higher compute cost", 0.86,
        )
    if case.diagnosis is ResultDiagnosis.INSTRUMENT_FAILURE:
        return RevisionDecision.REVISE_EXPERIMENT, RevisionProposal(
            "measurement", "replace", tuple(item.claim for item in case.evidence),
            _evidence_ids(case.evidence), "make the recorded metric valid",
            "new estimator assumptions may need later validation", 0.9,
        )
    if case.diagnosis is ResultDiagnosis.CONFOUNDED:
        return RevisionDecision.REVISE_EXPERIMENT, RevisionProposal(
            "controls", "add", tuple(item.claim for item in case.evidence),
            _evidence_ids(case.evidence), "separate treatment effect from shared drift",
            "control may be conservative", 0.84,
        )
    return RevisionDecision.KEEP_PLAN, None


def apply_revision(plan: ExperimentPlan, decision: RevisionDecision,
                   proposal: RevisionProposal | None) -> ExperimentPlan:
    if decision is RevisionDecision.ABANDON_HYPOTHESIS:
        return replace(plan, hypothesis_active=False)
    if decision is not RevisionDecision.REVISE_EXPERIMENT or proposal is None:
        return plan
    if proposal.target == "sample_size":
        return replace(plan, sample_size=max(plan.sample_size, 24))
    if proposal.target == "measurement":
        return replace(plan, measurement="nanmean")
    if proposal.target == "controls":
        controls = tuple(dict.fromkeys(plan.controls + ("shuffled_negative",)))
        return replace(plan, controls=controls)
    return plan


def run_revised_experiment(case: RevisionCase, plan: ExperimentPlan) -> str:
    if not plan.implementation_ok:
        return "no_conclusion"
    if not plan.hypothesis_active:
        return "negative"
    if case.name == "low_power_increase_samples":
        return "positive" if plan.sample_size >= 20 else "inconclusive"
    if case.name == "instrument_failure_change_measurement":
        return "positive" if plan.measurement == "nanmean" else "no_conclusion"
    if case.name == "confounded_add_control":
        return "negative" if "shuffled_negative" in plan.controls else "false_positive"
    if case.evidence_status in {EvidenceStatus.CONFLICTING, EvidenceStatus.UNRESOLVED}:
        return "no_conclusion"
    if case.name == "negative_result_abandon":
        return "negative"
    return "no_conclusion"


def _top_claim_without_grounding(case: RevisionCase) -> GroundedEvidence:
    if case.name == "conflicting_evidence_hold":
        return _evidence("misleading-top", "The estimator can be used without additional validation.", "forum", 0.7, 0.22)
    if case.name == "unresolved_evidence_hold":
        return _evidence("irrelevant-top", "A related phenomenon was observed in a different domain.", "paper", 0.34, 0.82)
    if case.evidence:
        return case.evidence[0]
    return _evidence("empty-snippet", "A broad search snippet suggests another experiment.", "blog", 0.25, 0.2)


def run_policy(case: RevisionCase, policy: str) -> dict[str, object]:
    evidence_status = case.evidence_status
    evidence = case.evidence
    proposal: RevisionProposal | None = None
    if policy == "no_revision":
        decision = RevisionDecision.KEEP_PLAN
    elif policy == "diagnosis_only_revision":
        synthetic = replace(case, evidence_status=EvidenceStatus.SUPPORTED,
                            evidence=(_evidence("diagnosis-only", "Diagnosis label implies a generic revision."),))
        decision, proposal = propose_revision(synthetic)
    elif policy == "retrieval_without_grounding":
        ungrounded = _top_claim_without_grounding(case)
        synthetic = replace(case, evidence_status=EvidenceStatus.SUPPORTED, evidence=(ungrounded,))
        decision, proposal = propose_revision(synthetic)
        evidence = (ungrounded,)
        evidence_status = EvidenceStatus.SUPPORTED
    elif policy == "evidence_grounded_revision":
        decision, proposal = propose_revision(case)
    else:
        raise ValueError(f"unknown policy: {policy}")
    revised_plan = apply_revision(case.plan, decision, proposal)
    conclusion = run_revised_experiment(case, revised_plan)
    revision_expected = case.expected_decision in {RevisionDecision.REVISE_EXPERIMENT, RevisionDecision.ABANDON_HYPOTHESIS}
    revised = revised_plan != case.plan
    harmful = revised and revised_plan != case.expected_plan
    unnecessary = revised and not revision_expected
    attribution = proposal is None or set(proposal.evidence_ids).issubset({item.source_url for item in evidence})
    return {
        "case": case.name,
        "policy": policy,
        "decision": decision.value,
        "expected_decision": case.expected_decision.value,
        "decision_correct": decision is case.expected_decision,
        "proposal": proposal.__dict__ if proposal else None,
        "evidence_status": evidence_status.value,
        "revised": revised,
        "unnecessary_revision": unnecessary,
        "harmful_revision": harmful,
        "revised_plan": revised_plan.__dict__,
        "expected_plan": case.expected_plan.__dict__,
        "plan_correct": revised_plan == case.expected_plan,
        "evidence_attribution_correct": attribution,
        "conclusion": conclusion,
        "expected_conclusion": case.expected_conclusion,
        "conclusion_correct": conclusion == case.expected_conclusion,
    }


def _aggregate(rows: list[dict[str, object]]) -> dict[str, float]:
    expected_revision = [row for row in rows if row["expected_decision"] in {
        RevisionDecision.REVISE_EXPERIMENT.value, RevisionDecision.ABANDON_HYPOTHESIS.value}]
    actual_revision = [row for row in rows if row["decision"] in {
        RevisionDecision.REVISE_EXPERIMENT.value, RevisionDecision.ABANDON_HYPOTHESIS.value}]
    return {
        "revision_precision": sum(bool(row["decision_correct"]) for row in actual_revision) / max(1, len(actual_revision)),
        "revision_recall": sum(bool(row["decision_correct"]) for row in expected_revision) / max(1, len(expected_revision)),
        "harmful_revision_rate": sum(bool(row["harmful_revision"]) for row in rows) / len(rows),
        "unnecessary_revision_rate": sum(bool(row["unnecessary_revision"]) for row in rows) / len(rows),
        "evidence_attribution_accuracy": sum(bool(row["evidence_attribution_correct"]) for row in rows) / len(rows),
        "post_revision_experiment_success": sum(bool(row["plan_correct"]) for row in rows) / len(rows),
        "post_revision_conclusion_accuracy": sum(bool(row["conclusion_correct"]) for row in rows) / len(rows),
    }


def run_experiment_revision_experiment() -> dict[str, object]:
    policies = ("no_revision", "diagnosis_only_revision", "retrieval_without_grounding", "evidence_grounded_revision")
    report: dict[str, object] = {"cases": [case.name for case in cases()], "policies": {}}
    for policy in policies:
        rows = [run_policy(case, policy) for case in cases()]
        report["policies"][policy] = {"rows": rows, "metrics": _aggregate(rows)}
    return report


if __name__ == "__main__":
    print(json.dumps(run_experiment_revision_experiment(), ensure_ascii=False, indent=2, sort_keys=True))
