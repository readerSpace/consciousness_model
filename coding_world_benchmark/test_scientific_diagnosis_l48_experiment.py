from coding_world_benchmark.scientific_diagnosis_l48_experiment import (
    ResultDiagnosis,
    collect_evidence,
    cases,
    diagnose,
    naive_success_only_action,
    next_action,
    run_scientific_diagnosis_experiment,
)


REPORT = run_scientific_diagnosis_experiment()


def test_critic_recognizes_all_six_result_diagnoses():
    assert set(REPORT["diagnosis_labels"]) == {entry.value for entry in ResultDiagnosis}
    assert REPORT["diagnostic_accuracy"] == 1.0
    assert REPORT["separates_scientific_failure_from_code_failure"]


def test_diagnosis_drives_different_next_actions():
    actions = {row["case"]: row["next_action"] for row in REPORT["cases"]}
    assert actions == {
        "implementation_failure": "repair_code",
        "instrument_failure": "repair_measurement",
        "low_power": "increase_samples",
        "confounded": "revise_controls",
        "negative_result": "record_falsification",
        "positive_result": "record_support",
    }


def test_naive_success_only_agent_conflates_non_positive_results_with_code_bugs():
    naive = {row["case"]: row["naive_action"] for row in REPORT["cases"]}
    assert naive["implementation_failure"] == "repair_code"
    assert naive["instrument_failure"] == "repair_code"
    assert naive["low_power"] == "record_support"
    assert naive["negative_result"] == "repair_code"
    assert REPORT["naive_action_agreement"] < 0.5


def test_repository_execution_is_part_of_the_evidence(tmp_path):
    implementation_case = next(case for case in cases() if case.name == "implementation_failure")
    evidence = collect_evidence(implementation_case, tmp_path / "repo")
    assert not evidence.implementation_ok
    diagnosis = diagnose(evidence)
    assert diagnosis is ResultDiagnosis.IMPLEMENTATION_FAILURE
    assert next_action(diagnosis) == "repair_code"
    assert naive_success_only_action(evidence) == "repair_code"
