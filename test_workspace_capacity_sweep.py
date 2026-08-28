from Consciousness_model import CompressedWorkspace
from attentional_blink_experiment import run as run_blink
from interference_capacity_sweep import run as run_interference
from ignition_threshold_experiment import global_access_probability, run as run_ignition
from global_availability_experiment import run as run_availability
from global_availability_robustness import run as run_availability_robustness
from broadcast_cache_experiment import run as run_cache, unsafe_gate_stress_test
from adaptive_broadcast_revision_experiment import run as run_adaptive_broadcast
from cross_domain_revision_experiment import run as run_cross_domain_revision
from partial_observation_experiment import run as run_partial_observation
from text_reasoning_experiment import TextBeliefState, run as run_text_reasoning
from domain_invariant_meta_control_experiment import run as run_meta_control
from integrated_b5_experiment import run as run_integrated_b5
from uncertainty_encoder_experiment import run as run_uncertainty_encoder
from uncertainty_control_robustness import run as run_uncertainty_robustness
from compressed_uncertainty_representation_experiment import (
    run as run_compressed_uncertainty,
    run_active_intervention_loop,
    run_continuous_domain_holdout,
    run_domain_holdout,
    run_intervention,
    run_operator_search,
    run_cost_intervention,
    run_invariant_mdl,
)
from active_intervention_discovery_experiment import (
    run as run_active_discovery,
    run_cost_learning,
    run_operation_value_search,
    run_operator_formula_search,
    run_world_transfer,
)
from random_world_science_discovery_experiment import run as run_random_world_discovery
from random_world_science_discovery_experiment import (
    run_decisive_experiment,
    run_decision_identification,
    run_adaptive_selector,
    run_continuous_adaptive_selector,
    run_selection_agreement,
    run_theory_discrimination,
    run_weighted_theory_discrimination,
    run_selector_comparison,
)
from independent_world_generalization_experiment import run as run_independent_worlds
from independent_world_generalization_experiment import (
    run_meta_controller_recompression,
    run_pareto_frontier,
    run_recursive_compression_ablation,
    run_robust_mdl,
    run_failure_family_selection,
    run_diagnostic_environment_selection,
    run_value_aware_comparison,
)
from metacognitive_summary_experiment import run as run_metacognitive_summary
from metacognition_experiment import _single_trial, run as run_metacognition
from world_model_planning_experiment import _trial, run as run_planning
from workspace_capacity_sweep import run
from functional_phenomenal_state_experiment import run as run_phenomenal_state
from qualia_theory_competition_experiment import run as run_qualia_theory_competition


def test_composed_concept_is_reused_by_distinct_modules():
    workspace = CompressedWorkspace(capacity=8)
    workspace.ingest((("a", "leads-to", "b"), ("b", "leads-to", "c")))
    planner = workspace.query("planner", ("a", "c"))
    predictor = workspace.query("predictor", ("a", "c"))
    assert planner and predictor
    assert planner[0].identifier == predictor[0].identifier
    assert planner[0].kind == "compose"


def test_capacity_sweep_is_reproducible_and_not_assumed_inverse_u():
    result = run(capacities=(2, 8, None), trials=16)
    assert set(result) == {"2", "8", "infinity"}
    assert all(0.0 <= row["mean"] <= 1.0 for row in result.values())


def test_attention_interference_dilutes_a_present_target():
    workspace = CompressedWorkspace(capacity=None)
    workspace.ingest((("start", "leads-to", "middle"), ("middle", "leads-to", "goal"), ("noise", "leads-to", "goal")))
    distribution = dict(workspace.attention_distribution(("start", "goal"), focus=0.8))
    correct = workspace.long_term["compose:start:leads-to:goal"]
    assert correct in distribution
    assert distribution[correct] < 1.0
    result = run_interference(capacities=(2, 16, None), trials=8, noise_per_goal=4)
    assert set(result) == {"2", "16", "infinity"}


def test_candidate_competition_creates_an_empirical_access_threshold():
    weak = global_access_probability(0.1)
    strong = global_access_probability(1.0)
    result = run_ignition()
    assert weak.global_access_probability < 0.5 < strong.global_access_probability
    assert result["access_threshold_at_p_ge_0_5"] is not None


def test_t2_access_recovers_with_lag_after_t1_selection():
    points = run_blink()["points"]
    assert points[0]["t2_access_after_t1"] < points[0]["t2_access_alone"]
    assert points[0]["t2_access_after_t1"] < points[-1]["t2_access_after_t1"]
    assert points[-1]["blink_cost"] == 0.0


def test_compressed_world_model_can_simulate_an_unobserved_multistep_route():
    success, depth = _trial(seed=3, capacity=None)
    assert success == 1.0
    assert depth == 4.0
    result = run_planning(capacities=(2, 16, None), trials=16)
    assert result["infinity"]["success_rate"] == 1.0


def test_workspace_evidence_supports_calibrated_metacognitive_reports():
    assert _single_trial("known", 0).report == "known"
    assert _single_trial("ambiguous", 0).report == "need_more_evidence"
    assert _single_trial("unknown", 0).report == "insufficient_evidence"
    result = run_metacognition(trials_per_condition=20)
    assert result["metareport_accuracy"] == 1.0
    assert result["brier_score"] < 0.15


def test_same_compressed_concept_is_retrieved_by_multiple_modules():
    result = run_availability()
    assert result["C2"]["mean"] == 0.0
    assert result["C3"]["mean"] == 1.0
    assert result["C3"]["shared_concept_reuse"] == 1.0


def test_global_availability_is_measured_under_capacity_and_interference():
    result = run_availability_robustness(capacities=(2, 16, None), trials=8, noise_per_goal=4)
    assert set(result) == {"2", "16", "infinity"}
    assert all(0.0 <= row["all_module_access"] <= 1.0 for row in result.values())


def test_semantic_broadcast_cache_reduces_searches_but_exposes_gate_tradeoff():
    result = run_cache(trials=12)
    assert result["B3"]["workspace_searches"] < result["B0"]["workspace_searches"]
    assert result["B3"]["consistency"] >= result["B0"]["consistency"]
    unsafe = unsafe_gate_stress_test(trials=12)
    assert unsafe["B3"]["error_propagation"] >= result["B3"]["error_propagation"]


def test_adaptive_broadcast_revision_reduces_final_shared_error():
    result = run_adaptive_broadcast(trials=24)
    assert result["B4"]["revocation_rate"] > 0.0
    assert result["B4"]["final_error_propagation"] < result["B3"]["final_error_propagation"]


def test_frozen_b4_generalizes_and_reports_revision_quality():
    result = run_cross_domain_revision(trials=24, feedback_error_rate=0.05)
    for domain in ("maze", "physics"):
        assert result[domain]["B4"]["recovery_rate"] > 0.0
        assert 0.0 <= result[domain]["B4"]["false_revoke_rate"] <= 1.0


def test_b4_defers_for_a_disambiguating_observation_under_partial_observation():
    result = run_partial_observation(trials=24)
    assert result["B4"]["deferral_rate"] > result["B3"]["deferral_rate"]
    assert result["B4"]["unsafe_commit_rate"] < result["B3"]["unsafe_commit_rate"]


def test_b4_revises_a_committed_textual_inference_after_later_contradiction():
    state = TextBeliefState()
    for sentence in ("アオイはキツネである。", "すべてのキツネは速い。", "すべての速いものは警戒している。"):
        state.add(sentence)
    assert state.answer("アオイ", "警戒している") == "真"
    state.add("アオイは警戒していない。")
    assert state.answer("アオイ", "警戒している") == "不明"
    result = run_text_reasoning()
    assert result["B4"]["accuracy"] > result["B3"]["accuracy"]


def test_domain_invariant_b5_uses_all_three_actions_on_held_out_task_labels():
    result = run_meta_control(train_trials=180, test_trials=96)
    assert result["B5"]["success_rate"] > result["B4"]["success_rate"]
    assert result["B5"]["observe_rate"] > 0.0


def test_b5_uses_actual_workspace_and_text_state_features_on_held_out_domains():
    result = run_integrated_b5(train_trials=120, test_trials=60)
    assert set(result) == {"B4", "B5"}
    assert all(0.0 <= row["success_rate"] <= 1.0 for row in result.values())


def test_uncertainty_encoder_selects_observation_when_information_gain_is_high():
    result = run_uncertainty_encoder(trials=24)
    assert result["partial"]["observe_rate"] == 1.0
    assert result["partial"]["mean_entropy_drop_after_observe"] > 0.0
    assert result["physics"]["observe_rate"] == 0.0


def test_uncertainty_control_has_stable_actions_and_cost_sensitive_boundaries():
    result = run_uncertainty_robustness()
    assert all(row["partial"] == "observe" for row in result["coefficient_grid"])
    assert [row["action"] for row in result["cost_sweep"]] == ["observe", "re-search", "commit"]


def test_compressed_uncertainty_representation_preserves_meta_actions():
    result = run_compressed_uncertainty()
    assert result["best"]["test_error"] <= result["full"]["test_error"]
    assert result["best"]["representation_cost"] < result["full"]["representation_cost"]
    assert result["compression_ratio"] < 1.0


def test_margin_only_representation_fails_counterfactual_eig_intervention():
    result = run_intervention()
    assert result["distinct_actions"] == ("commit", "observe")
    assert result["margin_only_error"] == 0.5
    assert result["full_representation_error"] == 0.0


def test_domain_holdout_exposes_lookup_representation_gap():
    result = run_domain_holdout()
    assert result["train_domain"] == "physics"
    assert result["test_domain"] == "partial"
    assert result["margin_only_error"] == 1.0


def test_operator_search_discovers_intervened_eig_signal():
    result = run_operator_search()
    assert result["best_operator"] == "eig_observe"
    assert result["errors"]["eig_observe"] == 0.0


def test_cost_intervention_requires_cost_in_meta_representation():
    result = run_cost_intervention()
    assert result["distinct_actions"] == ("commit", "observe")
    assert result["eig_only_error"] == 0.5
    assert result["full_representation_error"] == 0.0


def test_invariant_mdl_selects_eig_and_cost_across_environments():
    result = run_invariant_mdl()
    assert result["best"]["test_error"] == 0.0
    assert "eig_observe" in result["best"]["features"]
    assert "cost_observe" in result["best"]["features"]


def test_continuous_controller_generalizes_beyond_lookup_keys():
    result = run_continuous_domain_holdout()
    assert result["lookup_error"] == 1.0
    assert result["continuous_error"] == 0.0


def test_active_intervention_loop_selects_discriminating_experiments():
    result = run_active_intervention_loop()
    assert result["initial_representation"] == ("margin",)
    assert result["rounds"][0]["intervention"]["selected"]["feature"] == "eig_observe"
    assert result["rounds"][0]["recompressed"]["best"]["test_error"] == 0.0
    assert "eig_observe" in result["rounds"][0]["recompressed"]["best"]["features"]
    assert result["rounds"][1]["intervention"]["selected"]["feature"] in {"cost_observe", "cost_search"}


def test_hidden_law_active_discovery_reaches_zero_error_formula():
    result = run_active_discovery()
    selected = [round_result["selected"]["intervention"]["target"] for round_result in result["rounds"]]
    assert selected == ["eig_observe", "cost_observe", "cost_search"]
    assert result["final_formula"]["error"] == 0.0
    assert set(result["final_formula"]["terms"]) == set(result["true_terms"] + ("eig_research", "cost_search"))


def test_operator_dsl_recovers_eig_minus_cost_structure():
    result = run_operator_formula_search()
    assert result["formula"]["error"] == 0.0
    assert result["formula"]["name"] == result["true_formula"]


def test_operation_value_and_measured_cost_search_are_automatic():
    value_result = run_operation_value_search()
    cost_result = run_cost_learning()
    assert value_result["selected_value"] == 0.9
    assert cost_result["selected_lowest_cost"] == "short_observation"


def test_active_intervention_beats_random_across_hidden_worlds():
    result = run_world_transfer()
    assert result["passive_formula_error"] > 0.0
    assert result["active_beats_random"]


def test_distribution_shift_benchmark_reports_active_efficiency():
    result = run_world_transfer(worlds=5)
    assert result["passive_formula_error"] == 0.26666666666666666
    assert result["active_intervention_steps"] < result["random_intervention_steps"]
    assert result["active_beats_random"]


def test_operator_formula_search_recovers_semantically_correct_law():
    result = run_operator_formula_search()
    assert result["formula"]["error"] == 0.0
    assert "eig_observe-cost_observe" in result["formula"]["terms"]
    assert "eig_research-cost_search" in result["formula"]["terms"]


def test_random_world_benchmark_compares_discovery_baselines():
    result = run_random_world_discovery(worlds=100)
    strategies = result["strategies"]
    assert strategies["passive"]["recovery_rate"] == 0.67
    assert strategies["active"]["recovery_rate"] == 0.9
    assert strategies["active"]["recovery_rate"] > strategies["random"]["recovery_rate"]
    assert strategies["active"]["mean_cost"] < strategies["random"]["mean_cost"]
    assert strategies["eig"]["recovery_rate"] == strategies["active"]["recovery_rate"]


def test_eig_and_active_selection_agreement_explains_equal_recovery():
    result = run_selection_agreement(worlds=100)
    assert result["agreement"] == 0.9


def test_decisive_experiment_separates_eig_from_falsification():
    result = run_decisive_experiment()
    assert result["eig_choice"] == "high-eig-low-discrimination"
    assert result["active_choice"] == "low-eig-decisive"
    assert not result["eig_beats_active"]


def test_competing_theory_discrimination_finds_decisive_probe():
    result = run_theory_discrimination()
    assert result["hypotheses"] == 2
    assert result["discrimination"] == 1.0
    assert result["selected"]


def test_weighted_theory_discrimination_respects_posterior_weights():
    result = run_weighted_theory_discrimination()
    assert result["top_pair"] == ("H1", "H2")
    assert result["discrimination"] == 0.1762


def test_afm_selects_different_interventions_than_single_hypothesis_af1():
    result = run_selector_comparison(worlds=100)
    assert result["agreement_af1_afm"] == 0.18
    assert result["agreement_eig_afm"] == 0.14


def test_afm_reduces_identification_steps_in_decisive_world():
    result = run_decision_identification()
    assert result["identify_steps"] == {"eig": 2, "af1": 2, "afm": 1}
    assert result["afm_beats_eig"]


def test_adaptive_selector_switches_between_information_and_theory_search():
    result = run_adaptive_selector()
    assert result["test_error"] == 0.0
    assert result["strategy_counts"] == {"EIG": 12, "AF-1": 6, "AF-M": 12}
    assert result["features"] == ("entropy", "top_pair_mass", "top_gap", "effective_count")


def test_continuous_adaptive_selector_generalizes_without_regime_labels():
    result = run_continuous_adaptive_selector(train_worlds=80, test_worlds=20)
    assert result["adaptive_error"] == 0.0
    assert result["adaptive_recovery"] > result["fixed_recovery"]["EIG"]
    assert result["features"] == ("entropy", "top_pair_mass", "top_gap", "effective_count")


def test_independent_worlds_expose_adaptive_lookup_generalization_gap():
    result = run_independent_worlds(train_worlds=80, test_worlds=20)
    assert result["mean_theory_count"] == 4.65
    assert result["adaptive_train_recovery"] > 0.9
    assert result["adaptive_recovery"] == 0.35
    assert result["fixed_recovery"]["AF-1"] == 0.85
    assert result["adaptive_recovery"] < result["fixed_recovery"]["AF-1"]


def test_action_value_aware_representation_recovers_independent_world_strategy():
    result = run_value_aware_comparison(train_worlds=80, test_worlds=20)
    assert result["value_aware_recovery"] == 1.0
    assert result["af1_oracle_worlds"] == 17
    assert result["af1_mean_top_pair_mass"] > 0.7


def test_meta_controller_recompression_finds_minimal_action_value_features():
    result = run_meta_controller_recompression(train_worlds=80, test_worlds=20)
    assert result["selected_features"] == ("max_eig", "max_af1")
    assert result["train_error"] == 0.0
    assert result["test_error"] == 0.0
    assert result["representation_cost"] < result["full_cost"]


def test_recursive_compression_ablation_exposes_ood_overfitting():
    result = run_recursive_compression_ablation(train_worlds=80, test_worlds=20)
    levels = result["levels"]
    assert result["afm_world_fraction"] == 0.25
    assert levels["R0_object_only"]["ood_error"] == 0.8
    assert levels["R1_uncertainty"]["ood_error"] == 0.55
    assert levels["R2_theory"]["ood_error"] == 0.3
    assert levels["R3_recompressed"]["train_error"] == 0.0
    assert levels["R3_recompressed"]["ood_error"] == 0.6
    assert levels["R3_recompressed"]["ood_error"] > levels["R2_theory"]["ood_error"]


def test_robust_mdl_improves_final_ood_without_using_final_holdout():
    result = run_robust_mdl(train_worlds=60, validation_worlds=20, test_worlds=20)
    assert result["selected_features"] == ("max_af1", "max_afm")
    assert result["test_mean_error"] == 0.4
    assert result["naive_test_error"] == 0.6
    assert result["test_mean_error"] < result["naive_test_error"]
    assert result["validation_worst_error"] == 0.4


def test_failure_family_selection_exposes_representative_set_bias():
    result = run_failure_family_selection(train_worlds=60, validation_worlds=20, test_worlds=20)
    assert result["failure_families"] == 4
    assert result["selected_environment_count"] == 4
    assert result["family_test_error"] == 0.8
    assert result["random_test_error"] == 0.2
    assert result["family_test_error"] > result["random_test_error"]


def test_pareto_frontier_exposes_cost_robustness_tradeoff():
    result = run_pareto_frontier(train_worlds=60, validation_worlds=20, test_worlds=20)
    assert result["frontier_size"] == 8
    assert result["selected_features"] == ("max_af1", "max_afm")
    assert result["selected_test_error"] == 0.4


def test_diagnostic_environment_selection_finds_representation_disagreement():
    result = run_diagnostic_environment_selection(train_worlds=60, validation_worlds=20, test_worlds=20)
    assert result["diagnostic_disagreement"] == 1
    assert result["test_error"] == 0.4


def test_preselection_summary_reports_tail_mass_without_expanding_workspace():
    result = run_metacognitive_summary(noise_values=(0, 10), perturbations=16)
    assert result["sweep"][1]["tail_mass"] > 0.0
    assert 0.0 <= result["brier_corrected"] <= 1.0
    assert result["brier_corrected"] < result["brier_naive"]
    assert result["flip_rate_corrected"] < result["flip_rate_naive"]
    assert result["sweep"][1]["effective_count"] > 1.0


def test_functional_phenomenal_state_candidate_passes_six_conditions():
    result = run_phenomenal_state()
    assert result["internal_privacy"]
    assert result["causal_efficacy"]["changed_outputs"] == 5
    assert result["report_ablation"]["memory_same"]
    assert result["report_ablation"]["attention_same"]
    assert result["report_ablation"]["planning_same"]
    assert result["report_ablation"]["report_changed"]
    assert result["context_dependence"]["different"]
    assert result["metamerism"]["approximately_same"]
    assert result["phenomenal_geometry"]["stable_order"]
    assert result["access_dissociation"]["q_same"]
    assert result["access_dissociation"]["report_different"]


def test_q_ablation_and_unity_tests_separate_integrated_q_from_controls():
    result = run_phenomenal_state()
    conditions = result["q_ablation"]["conditions"]
    assert result["q_ablation"]["same_budget"]
    assert conditions["Q4_integrated_q"]["mean_error"] == 0.0
    assert conditions["Q4_integrated_q"]["crossmodal"]
    assert conditions["Q0_no_q"]["crossmodal"] is False
    assert conditions["Q3_split_latent"]["mean_error"] > conditions["Q4_integrated_q"]["mean_error"]
    assert result["split_brain"]["separated_disagreement"]
    assert result["split_brain"]["recombined_consistency"]


def test_q_has_temporal_integration_and_competing_hypothesis_signal():
    result = run_phenomenal_state()
    assert result["temporal_integration"]["smooth"]
    assert result["temporal_integration"]["history_effective"]
    assert result["hypothesis_discrimination"]["best_supported"] == "integrated_private"


def test_integrated_q_generalizes_to_frozen_unseen_cognitive_tasks():
    result = run_phenomenal_state()["novel_cognitive_tasks"]
    conditions = result["conditions"]
    assert result["evaluation_frozen"]
    assert result["q4_best"] == "Q4_integrated_q"
    assert conditions["Q4_integrated_q"]["mean_error"] == 0.0
    assert conditions["Q4_integrated_q"]["mean_error"] < conditions["Q0_no_q"]["mean_error"]
    assert conditions["Q4_integrated_q"]["mean_error"] < conditions["Q3_split_latent"]["mean_error"]


def test_hidden_q_split_interval_is_detected_and_reintegrated():
    result = run_phenomenal_state()["split_interval_detection"]
    assert result["detected"]
    assert result["detected_interval"] == (3, 4)
    assert result["reintegrated"]


def test_blind_ab_protocol_equalizes_generation_and_declares_human_gap():
    result = run_phenomenal_state()["blind_ab_protocol"]
    assert result["generator_config_equal"]
    assert result["labels_blinded"]
    assert len(result["rating_dimensions"]) == 5
    assert result["human_data_collected"] is False


def test_q_competing_theories_are_separated_by_targeted_interventions():
    result = run_qualia_theory_competition()
    assert result["paired_predictions"]["complete_separation"]
    assert result["paired_predictions"]["distinguishable_pairs"] == 6
    assert result["blind_machine_evaluation"]["q4_best"] == "Q4_integrated"
    assert result["blind_machine_evaluation"]["conditions"]["Q4_integrated"] == 1.0
