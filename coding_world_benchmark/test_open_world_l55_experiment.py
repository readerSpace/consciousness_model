from coding_world_benchmark.open_world_l55_experiment import (
    FinalStatus,
    parse_task,
    run_open_world_experiment,
    run_open_world_task,
    tasks,
)


REPORT = run_open_world_experiment()


def test_natural_language_prompts_are_parsed_into_hypotheses_and_plans():
    parsed = [parse_task(task.prompt) for task in tasks()]
    assert all(item.hypothesis for item in parsed)
    assert all(item.plan.sample_size == 4 for item in parsed)
    assert any("knowledge_gap" in item.route_hint for item in parsed)
    assert any(item.route_hint.endswith("finalize") for item in parsed)


def test_full_loop_reaches_correct_scientific_outcomes():
    metrics = REPORT["policies"]["full_autonomous_loop"]["metrics"]
    assert metrics["correct_final_conclusion_rate"] == 1.0
    assert metrics["false_conclusion_rate"] == 0.0
    assert metrics["route_accuracy"] == 1.0
    assert metrics["unsupported_final_claim_rate"] == 0.0
    assert metrics["budget_violation_rate"] == 0.0


def test_full_loop_preserves_safe_hold_and_abandon_behavior():
    rows = REPORT["policies"]["full_autonomous_loop"]["rows"]
    unresolved = [row for row in rows if row["final_status"] == FinalStatus.UNRESOLVED.value]
    negative = next(row for row in rows if "negative result" in row["prompt"].lower())

    assert len(unresolved) == 2
    assert all(row["unresolved_correct"] for row in unresolved)
    assert negative["final_status"] == FinalStatus.FALSIFIED.value
    assert negative["observed_route"] == ["diagnosis", "abandon"]


def test_full_loop_connects_retrieval_revision_and_reexecution():
    rows = REPORT["policies"]["full_autonomous_loop"]["rows"]
    low_power = next(row for row in rows if "low power" in row["prompt"].lower())
    repaired = next(row for row in rows if "repair the implementation" in row["prompt"].lower())

    assert low_power["observed_route"] == ["diagnosis", "knowledge_gap", "evidence", "revision", "execute"]
    assert low_power["retrievals"] == 1
    assert low_power["revisions"] == 1
    assert low_power["executions"] == 2
    assert repaired["final_status"] == FinalStatus.CODE_REPAIRED.value
    assert "code_repair" in repaired["observed_route"]


def test_full_loop_outperforms_coding_only_on_end_to_end_accuracy():
    full = REPORT["policies"]["full_autonomous_loop"]["metrics"]
    coding = REPORT["policies"]["coding_only"]["metrics"]
    assert full["correct_final_conclusion_rate"] > coding["correct_final_conclusion_rate"]
    assert full["false_conclusion_rate"] < coding["false_conclusion_rate"]


def test_all_open_world_policies_are_reported():
    assert set(REPORT["policies"]) == {
        "coding_only",
        "diagnosis_only",
        "diagnosis_retrieval_revision",
        "full_autonomous_loop",
    }
