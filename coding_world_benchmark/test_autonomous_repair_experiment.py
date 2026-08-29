from pathlib import Path

from coding_world_benchmark.autonomous_repair_experiment import (
    holdout_tasks,
    run_autonomous_trial,
    run_holdout_benchmark,
)


def test_l2_generates_a_file_specific_hypothesis_and_real_edit(tmp_path: Path):
    task = next(task for task in holdout_tasks() if task.domain == "cache")

    result = run_autonomous_trial(task, "L2", tmp_path)

    assert result.task_success
    assert result.provided_hypotheses == 0
    assert result.hypotheses_generated >= 1
    assert result.edits == 1
    assert "cache_service.py" in result.report


def test_l2_transfers_to_holdout_domains_without_preprovided_candidates():
    report = run_holdout_benchmark(repetitions=1, levels=("L2", "L5"))

    assert report["L2"]["tasks"] == 6
    assert report["L2"]["task_success"] == 1.0
    assert report["L2"]["provided_hypotheses"] == 0
    assert report["L5"]["task_success"] == 1.0