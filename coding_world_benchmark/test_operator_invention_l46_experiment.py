from coding_world_benchmark.operator_invention_l46_experiment import run_operator_invention_experiment


def test_composed_rollback_operator_transfers_to_a_different_holdout_representation():
    report = run_operator_invention_experiment()

    assert report["acquisition_success"]
    assert report["holdout_success"]
    assert not report["no_operator_holdout_success"]
    assert report["concept_acquisition_cost_exceeds_reuse_cost"]
    assert report["operator"]["name"] == "ROLLBACK"
    assert report["operator"]["reuse_count"] == 1