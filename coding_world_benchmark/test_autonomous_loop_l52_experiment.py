from coding_world_benchmark.autonomous_loop_l52_experiment import (
    FinalStatus,
    ScientificLoopState,
    run_autonomous_scientific_loop_experiment,
    run_loop,
    tasks,
)


REPORT = run_autonomous_scientific_loop_experiment()


def test_l52_covers_terminal_revision_hold_second_pass_and_budget_cases():
    names = [task.name for task in tasks()]
    assert names == [
        "positive_no_search",
        "low_power_then_samples",
        "instrument_then_measurement",
        "confounded_then_control",
        "implementation_then_repair",
        "negative_no_search",
        "conflicting_evidence_hold",
        "unresolved_evidence_hold",
        "second_revision_needed",
        "cyclic_low_power_budget_stop",
    ]
    assert any(task.needs_two_revisions for task in tasks())
    assert any(task.cyclic for task in tasks())
    assert any(task.expected_status is FinalStatus.UNRESOLVED for task in tasks())
    assert any(task.expected_status is FinalStatus.BUDGET_STOPPED for task in tasks())


def test_scientific_loop_state_keeps_the_required_histories():
    state = ScientificLoopState(
        iteration=1,
        hypothesis_status="active",
        diagnosis_history=["low_power"],
        knowledge_gaps=["power_analysis"],
        evidence_history=["supported"],
        revision_history=["increase"],
        execution_history=["1:inconclusive"],
        budget_remaining={"iterations": 3, "retrievals": 2, "executions": 3, "revisions": 2},
    )
    assert state.diagnosis_history
    assert state.knowledge_gaps
    assert state.evidence_history
    assert state.revision_history
    assert state.execution_history
    assert set(state.budget_remaining) == {"iterations", "retrievals", "executions", "revisions"}


def test_full_loop_finishes_easy_positive_and_negative_without_unnecessary_actions():
    positive = next(task for task in tasks() if task.name == "positive_no_search")
    positive_row = run_loop(positive, "full_autonomous_loop")
    assert positive_row["final_status"] == "supported"
    assert positive_row["conclusion"] == "positive"
    assert positive_row["retrievals"] == 0
    assert positive_row["revisions"] == 0
    assert not positive_row["unnecessary_action"]

    negative = next(task for task in tasks() if task.name == "negative_no_search")
    negative_row = run_loop(negative, "full_autonomous_loop")
    assert negative_row["final_status"] == "falsified"
    assert negative_row["conclusion"] == "negative"
    assert negative_row["retrievals"] == 0
    assert negative_row["revisions"] == 0
    assert not negative_row["unnecessary_action"]


def test_full_loop_connects_gap_evidence_revision_and_reexecution():
    low_power = next(task for task in tasks() if task.name == "low_power_then_samples")
    row = run_loop(low_power, "full_autonomous_loop")
    assert row["final_status"] == "supported"
    assert row["conclusion"] == "positive"
    assert row["state"]["diagnosis_history"] == ["low_power", "positive_result"]
    assert row["state"]["knowledge_gaps"] == ["power_analysis"]
    assert row["state"]["evidence_history"] == ["supported"]
    assert row["state"]["revision_history"] == ["increase"]


def test_code_repair_and_confounded_revision_reach_correct_outcomes():
    code = next(task for task in tasks() if task.name == "implementation_then_repair")
    code_row = run_loop(code, "full_autonomous_loop")
    assert code_row["final_status"] == "code_repaired"
    assert code_row["state"]["revision_history"] == ["route_to_code_repair"]

    confounded = next(task for task in tasks() if task.name == "confounded_then_control")
    confounded_row = run_loop(confounded, "full_autonomous_loop")
    assert confounded_row["final_status"] == "falsified"
    assert confounded_row["conclusion"] == "negative"
    assert confounded_row["state"]["knowledge_gaps"] == ["causal_control"]


def test_conflicting_and_unresolved_evidence_stop_without_false_conclusions():
    for name in ("conflicting_evidence_hold", "unresolved_evidence_hold"):
        row = run_loop(next(task for task in tasks() if task.name == name), "full_autonomous_loop")
        assert row["final_status"] == "unresolved"
        assert row["conclusion"] == "unresolved"
        assert not row["false_conclusion"]
        assert row["retrievals"] == 1
        assert row["revisions"] == 0


def test_second_revision_and_cyclic_budget_behaviour_are_explicit():
    second = next(task for task in tasks() if task.name == "second_revision_needed")
    second_row = run_loop(second, "full_autonomous_loop")
    assert second_row["final_status"] == "supported"
    assert second_row["state"]["diagnosis_history"] == ["low_power", "low_power", "positive_result"]
    assert second_row["revisions"] == 2

    cyclic = next(task for task in tasks() if task.name == "cyclic_low_power_budget_stop")
    cyclic_row = run_loop(cyclic, "full_autonomous_loop")
    assert cyclic_row["final_status"] == "budget_stopped"
    assert cyclic_row["conclusion"] == "unresolved"
    assert cyclic_row["loop_terminated_correctly"]
    assert not cyclic_row["budget_violation"]


def test_full_loop_outcome_metrics_are_stronger_than_partial_policies():
    policies = REPORT["policies"]
    full = policies["full_autonomous_loop"]["metrics"]
    assert full["correct_final_conclusion_rate"] == 1.0
    assert full["false_conclusion_rate"] == 0.0
    assert full["budget_violation_rate"] == 0.0
    assert full["unsupported_final_claim_rate"] == 0.0
    assert full["loop_termination_accuracy"] == 1.0
    assert full["correct_final_conclusion_rate"] > policies["diagnosis_only"]["metrics"]["correct_final_conclusion_rate"]
    assert full["correct_final_conclusion_rate"] > policies["diagnosis_retrieval_revision"]["metrics"]["correct_final_conclusion_rate"]
    assert full["false_conclusion_rate"] < policies["coding_only"]["metrics"]["false_conclusion_rate"]
