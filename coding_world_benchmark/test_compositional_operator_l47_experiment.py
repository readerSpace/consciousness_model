from coding_world_benchmark.compositional_operator_l47_experiment import (
    DISTRACTOR_PRIMITIVES,
    run_compositional_operator_experiment,
)


REPORT = run_compositional_operator_experiment()


def test_one_shared_composer_acquires_a_distinct_operator_for_every_contract_family():
    assert REPORT["all_families_acquired"]
    assert REPORT["operators_acquired"] == 4
    assert REPORT["distinct_operator_signatures"] == 4
    assert REPORT["search_is_family_agnostic"]


def test_every_acquired_operator_is_verified_by_a_real_unittest_process():
    assert all(family["verified_by_unittest_subprocess"] for family in REPORT["families"].values())


def test_distractor_primitives_are_offered_but_never_enter_an_operator():
    assert all(primitive in REPORT["primitives"] for primitive in DISTRACTOR_PRIMITIVES)
    assert REPORT["distractors_unused"]


def test_reuse_is_cheaper_than_acquisition_on_renamed_holdouts():
    for family in REPORT["families"].values():
        assert family["holdout_success"]
        assert family["acquisition_cost_exceeds_reuse_cost"]


def test_parameterized_compression_beats_exact_memorisation_and_repeated_search():
    ablation = REPORT["ablation"]
    assert ablation["memorize_exact"]["success_rate"] == 0.0
    assert ablation["parameterized_operator"]["success_rate"] == 1.0
    assert ablation["full"]["success_rate"] == 1.0
    assert ablation["no_compression"]["success_rate"] == 1.0
    assert ablation["full"]["mean_trials"] < ablation["no_compression"]["mean_trials"]


def test_operators_are_promoted_only_when_the_description_length_gain_is_positive():
    for family in REPORT["families"].values():
        assert family["promoted_by_mdl"] == (family["delta_description_length"] > 0)
    assert REPORT["mdl_promoted_operators"] == 4


def test_rollback_operator_transfers_to_a_restructured_multi_file_repository():
    structural = REPORT["structural_holdout"]
    assert structural["success"]
    assert structural["verified_by_unittest_subprocess"]
    assert structural["state_expression"] == "inventory.counts"
