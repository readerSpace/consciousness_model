from coding_world_benchmark.l45_holdout_experiment import evaluate_l4_grammar, generate_adversarial_tasks


def test_l45_generator_is_reproducible_and_covers_unseen_contracts():
    tasks = generate_adversarial_tasks(count=100, seed=41)
    report = evaluate_l4_grammar(tasks)

    assert len(tasks) == 100
    assert {task.kind for task in tasks} == {"rollback", "coherence", "idempotency", "cleanup"}
    assert report["candidate_coverage"] == 0.0
    assert report["task_success"] == 0.0