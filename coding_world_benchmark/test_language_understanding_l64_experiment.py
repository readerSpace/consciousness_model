from coding_world_benchmark.language_understanding_l64_experiment import (
    LanguageContext,
    SemanticIntent,
    TargetType,
    compare_language_policies,
    default_language_cases,
    evaluate_cases,
    parse_semantic_task,
    summarize_language_results,
)


def test_composite_instruction_is_decomposed_into_work_sequence():
    task = parse_semantic_task("この仮説を検証する実験を実装して結果をまとめて")

    assert task.intent is SemanticIntent.IMPLEMENT
    assert task.requires_modification
    assert task.requires_execution
    assert task.requires_repository_context
    assert task.operation_sequence == (
        "hypothesis_extraction",
        "experiment_design",
        "implementation",
        "execution",
        "evaluation",
        "report",
    )


def test_reference_and_ambiguity_are_scored_separately():
    reference = parse_semantic_task("さっき追加した実験をもう一度実行して", LanguageContext(previous_experiment="exp465"))
    ambiguous = parse_semantic_task(
        "初期状態を説明して",
        LanguageContext(repository_targets=("exp445.initial_state", "exp464.initial_state")),
    )

    assert reference.target_type is TargetType.EXPERIMENT
    assert reference.target == "exp465"
    assert reference.requires_reference_resolution
    assert ambiguous.ambiguous
    assert ambiguous.requires_repository_context


def test_language_benchmark_reports_dashboard_metrics():
    cases = default_language_cases()
    results = evaluate_cases(cases)
    summary = summarize_language_results(cases, results)

    assert summary["cases"] == float(len(cases))
    assert summary["intent_accuracy"] == 1.0
    assert summary["target_resolution"] == 1.0
    assert summary["reference_resolution"] == 1.0
    assert summary["ambiguity_detection"] == 1.0
    assert summary["cross_language_consistency"] == 1.0


def test_canonical_context_policy_beats_keyword_baseline_on_structured_fields():
    comparison = compare_language_policies(default_language_cases())

    assert comparison["canonical_ir_context"]["semantic_task_accuracy"] > comparison["keyword_parser"]["semantic_task_accuracy"]
    assert comparison["canonical_ir_context"]["repository_context_accuracy"] > comparison["keyword_parser"]["repository_context_accuracy"]
    assert comparison["canonical_ir_context"]["operation_sequence_accuracy"] > comparison["keyword_parser"]["operation_sequence_accuracy"]
