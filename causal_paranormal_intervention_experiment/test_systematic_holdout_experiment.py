import pytest

from systematic_holdout_experiment import (
    CAUSES,
    LearnedCausalAgent,
    build_splits,
    evaluate,
    run,
)


def test_splits_hold_out_the_intended_compositional_axes():
    train, tests = build_splits()
    train_appearances = {(case.cause, case.appearance) for case in train}
    train_contexts = {(case.cause, case.context) for case in train}
    train_factor_combos = {(case.cause, case.factors) for case in train}

    assert len(CAUSES) == 10
    assert all((case.cause, case.appearance) not in train_appearances for case in tests["appearance_holdout"])
    assert all((case.cause, case.context) not in train_contexts for case in tests["context_holdout"])
    assert all((case.cause, case.factors) not in train_factor_combos for case in tests["factor_combo_holdout"] if case.cause != "unknown")


def test_shared_concepts_beat_no_transfer_on_factor_combo_holdout():
    train, tests = build_splits()
    shared = evaluate(LearnedCausalAgent(train, reusable_factors=True), tests["factor_combo_holdout"], "factor_combo_holdout")
    blocked = evaluate(LearnedCausalAgent(train, reusable_factors=False), tests["factor_combo_holdout"], "factor_combo_holdout")

    assert shared.known_cause_accuracy - blocked.known_cause_accuracy >= 0.20
    assert shared.minimal_action_hit_rate >= blocked.minimal_action_hit_rate


def test_anonymized_observation_names_preserve_transfer_gain():
    train, tests = build_splits()
    shared = evaluate(LearnedCausalAgent(train, reusable_factors=True), tests["factor_combo_holdout"], "factor_combo_holdout", anonymized=True)
    blocked = evaluate(LearnedCausalAgent(train, reusable_factors=False), tests["factor_combo_holdout"], "factor_combo_holdout", anonymized=True)

    assert shared.known_cause_accuracy - blocked.known_cause_accuracy >= 0.15


def test_report_contains_systematic_holdout_metrics():
    report = run()
    concept = report.accuracy_by_holdout_level["factor_combo_holdout"]["causal_eig_concept"]
    no_transfer = report.accuracy_by_holdout_level["factor_combo_holdout"]["eig_no_transfer"]
    nearest = report.accuracy_by_holdout_level["factor_combo_holdout"]["nearest"]

    assert report.train_cases > 0
    assert set(report.test_cases_by_split) == {"random", "appearance_holdout", "context_holdout", "factor_combo_holdout"}
    assert report.accuracy_by_holdout_level["random"]["causal_eig_concept"] >= 0.85
    assert concept >= 0.70
    assert concept > no_transfer
    assert concept >= nearest
    assert report.transfer_gain_by_split["factor_combo_holdout"] == pytest.approx(concept - no_transfer)
    assert report.anonymized_token_transfer_gain >= 0.15
