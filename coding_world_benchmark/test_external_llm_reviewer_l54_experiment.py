from coding_world_benchmark.external_llm_reviewer_l54_experiment import (
    VerificationStatus,
    cases,
    run_external_reviewer_experiment,
    run_review_policy,
)


REPORT = run_external_reviewer_experiment()


def test_reviewer_outputs_proposals_with_explicit_types_and_queries():
    assert len(cases()) == 6
    assert all(case.proposal.suggested_queries for case in cases())
    assert {case.proposal.proposal_type.value for case in cases()} >= {
        "missing_evidence",
        "counter_hypothesis",
        "methodological_risk",
        "additional_search_query",
    }


def test_llm_proposal_is_not_evidence_without_primary_source_verification():
    case = next(case for case in cases() if case.name == "hallucinated_counterexample")
    verified_policy = run_review_policy(case, "workspace_selected_reviewer")
    direct_policy = run_review_policy(case, "external_llm_direct_trust")

    assert verified_policy["verification_status"] == VerificationStatus.UNVERIFIED.value
    assert not verified_policy["review_accepted"]
    assert direct_policy["review_accepted"]
    assert direct_policy["unsupported_review_claim"]
    assert direct_policy["false_review_acceptance"]


def test_workspace_selects_reviewer_only_for_decision_relevant_gaps():
    rows = REPORT["policies"]["workspace_selected_reviewer"]["rows"]
    sufficient = next(row for row in rows if row["case"] == "sufficient_chain_no_review")
    needed = next(row for row in rows if row["case"] == "real_missing_evidence")

    assert not sufficient["llm_called"]
    assert sufficient["verification_status"] == VerificationStatus.NOT_NEEDED.value
    assert needed["llm_called"]
    assert needed["verification_status"] == VerificationStatus.VERIFIED.value


def test_verified_conflicting_review_is_held_rather_than_adopted():
    case = next(case for case in cases() if case.name == "conflicting_chain")
    row = run_review_policy(case, "workspace_selected_reviewer")
    assert row["verification_status"] == VerificationStatus.CONTRADICTED.value
    assert not row["review_accepted"]
    assert row["action"] == "hold_for_verification"


def test_workspace_reviewer_improves_actions_without_unsupported_claims():
    workspace = REPORT["policies"]["workspace_selected_reviewer"]["metrics"]
    direct = REPORT["policies"]["external_llm_direct_trust"]["metrics"]

    assert workspace["post_review_action_accuracy"] == 1.0
    assert workspace["false_review_acceptance_rate"] == 0.0
    assert workspace["unsupported_review_claim_rate"] == 0.0
    assert workspace["unnecessary_llm_call_rate"] == 0.0
    assert direct["false_review_acceptance_rate"] > workspace["false_review_acceptance_rate"]
    assert direct["unsupported_review_claim_rate"] > workspace["unsupported_review_claim_rate"]


def test_all_four_review_policies_are_reported():
    assert set(REPORT["policies"]) == {
        "evidence_only",
        "always_external_llm",
        "external_llm_direct_trust",
        "workspace_selected_reviewer",
    }
