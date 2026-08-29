from coding_world_benchmark.operation_discovery_l4_experiment import run_benchmark


def test_l4_closed_loop_outperforms_rule_baseline_and_needs_recovery_controls():
    report = run_benchmark()

    assert report["full"]["task_success"] == 1.0
    assert report["full"]["failure_recovery_rate"] == 1.0
    assert report["rule_baseline"]["task_success"] < report["full"]["task_success"]
    assert report["no_failure_memory"]["task_success"] < report["full"]["task_success"]
    assert report["no_reobservation"]["task_success"] < report["full"]["task_success"]
    assert report["no_candidate_suppression"]["task_success"] < report["full"]["task_success"]