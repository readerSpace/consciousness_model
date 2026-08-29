from pathlib import Path

from coding_world_benchmark.coding_world_benchmark import (
    CONDITIONS,
    generate_tasks,
    run_benchmark,
    run_trial,
)


def test_full_controller_recovers_from_an_initially_misleading_repair(tmp_path: Path):
    task = generate_tasks(1)[0]
    result = run_trial(task, "C_full", tmp_path)

    assert result.task_success
    assert result.first_patch_failed
    assert result.recovered_after_first_failure
    assert result.hidden_tests_passed


def test_benchmark_reports_each_requested_ablation():
    report = run_benchmark(tasks=4)

    assert set(report) == set(CONDITIONS)
    assert report["C_full"]["task_success"] == 1.0
    assert report["F_no_reobservation"]["task_success"] < report["C_full"]["task_success"]
    assert report["G_no_candidate_suppression"]["mean_repeated_failed_hypotheses"] > 0