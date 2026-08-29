from pathlib import Path

from coding_world_benchmark.heterogeneous_l3_experiment import l3_tasks, run_l3_benchmark, run_l3_trial


def test_state_transition_agent_repairs_each_heterogeneous_state_shape(tmp_path: Path):
    results = [run_l3_trial(task, "state_transition", tmp_path) for task in l3_tasks()]

    assert all(result.task_success for result in results)
    assert all(result.hidden_test_pass_rate == 1.0 for result in results)
    assert all(result.irrelevant_files_read == 1 for result in results)


def test_l3_separates_state_transition_reasoning_from_get_pop_pattern_matching():
    report = run_l3_benchmark()

    assert report["state_transition"]["task_success"] == 1.0
    assert report["pattern_baseline"]["task_success"] < report["state_transition"]["task_success"]
    assert report["pattern_baseline"]["task_success"] == 1 / 6