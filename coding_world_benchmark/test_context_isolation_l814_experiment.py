import inspect

import pytest

from coding_world_benchmark import episodic_memory_l813_experiment as l813
from coding_world_benchmark.context_isolation_l814_experiment import (
    MAX_EPISODES,
    build_memory,
    build_queries,
    format_experiment,
    generate_episodes,
    run_experiment,
    run_size,
)


@pytest.fixture(scope="module")
def large_run():
    return run_size(100)


def test_generated_corpus_has_no_duplicate_ground_truth():
    episodes = generate_episodes(MAX_EPISODES)

    assert len({item.phrase for item in episodes}) == MAX_EPISODES
    assert len({item.artifact for item in episodes}) == MAX_EPISODES
    assert len({item.model_id for item in episodes}) == MAX_EPISODES


def test_generating_more_than_the_distinct_supply_is_refused():
    with pytest.raises(ValueError):
        generate_episodes(MAX_EPISODES + 1)


def test_resolver_is_given_the_query_text_only():
    # The resolver takes a string; it has no parameter through which the
    # generator's expected episode could reach it.
    parameters = list(inspect.signature(l813.EpisodicMemory.resolve_reference).parameters)

    assert parameters == ["self", "request"]


def test_phrase_probes_are_only_emitted_when_truth_is_unambiguous():
    episodes = generate_episodes(40)
    episodic = build_memory(episodes)

    for query in build_queries(episodes, episodic):
        if query.kind != "ENTITY_PHRASE":
            continue
        target = episodic.episode(query.expected_episode_id)
        phrase = next(
            item.phrase for item in episodes if f"ep-{item.index + 1:03d}" == query.expected_episode_id
        )
        tokens = set(l813._named_tokens(phrase))
        owners = [
            episode.episode_id for episode in episodic.episodes if tokens <= set(episode.names)
        ]
        assert target is not None
        assert owners == [query.expected_episode_id]


def test_isolation_still_holds_at_one_hundred_episodes(large_run):
    metrics = large_run["metrics"]

    assert metrics["stale_intent_leak_rate"] == 0.0
    assert metrics["new_task_contamination_rate"] == 0.0


def test_specific_names_do_not_degrade_with_corpus_size(large_run):
    per_kind = large_run["per_kind_accuracy"]

    assert per_kind["ARTIFACT"] == 1.0
    assert per_kind["ENTITY_SPECIFIC"] == 1.0
    assert per_kind["BARE_RECENT"] == 1.0
    assert per_kind["UNKNOWN_ENTITY"] == 1.0


def test_component_phrases_collapse_once_components_are_shared(large_run):
    per_kind = large_run["per_kind_accuracy"]

    assert per_kind["ENTITY_PHRASE"] == 0.0


def test_the_failure_mode_is_refusal_not_corruption(large_run):
    # Every phrase probe that fails does so by asking, so no episode is ever
    # silently substituted for another.  This is what makes the degradation
    # safe to ship while the resolver is redesigned.
    metrics = large_run["metrics"]
    failures = [
        item
        for item in large_run["records"]
        if not item["correct"] and item["expected_decision"] == "CONTINUE_EPISODE"
    ]

    assert failures
    assert all(item["decision"] == "CLARIFY" for item in failures)
    assert metrics["wrong_episode_reuse_rate"] == 0.0


def test_scaling_degradation_is_measured_and_non_negative():
    report = run_experiment(sizes=(2, 10, 100))
    degradation = report["episode_scaling_degradation"]

    assert degradation[2] == 0.0
    assert all(value >= 0 for value in degradation.values())
    assert degradation[100] > 0
    assert "episode_scaling_degradation" in format_experiment(report)


def test_wrong_episode_reuse_stays_zero_at_every_size():
    report = run_experiment(sizes=(2, 10, 50))

    for run in report["runs"]:
        assert run["metrics"]["wrong_episode_reuse_rate"] == 0.0
        assert run["metrics"]["clarification_accuracy"] in (1.0, "n/a")


def test_the_benchmark_does_not_touch_the_frozen_l813_resolver():
    before = l813._REFERENCE_STOPWORDS
    run_size(20)

    assert l813._REFERENCE_STOPWORDS is before


def test_results_are_deterministic_for_a_seed():
    assert run_size(20, seed=8140)["metrics"] == run_size(20, seed=8140)["metrics"]
