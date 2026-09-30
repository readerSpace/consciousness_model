import pytest

from coding_world_benchmark.context_isolation_l814_experiment import default_resolver
from coding_world_benchmark.holdout_resolution_l8151_experiment import (
    MAX_EPISODES,
    generate_episodes,
    run_holdout,
)
from coding_world_benchmark.structural_resolver_l815_experiment import resolve_reference_structural


@pytest.fixture(scope="module")
def cold():
    return run_holdout(50)


@pytest.fixture(scope="module")
def cold_baseline():
    return run_holdout(50, resolver=default_resolver)


def test_holdout_vocabulary_shares_nothing_with_the_development_benchmark():
    from coding_world_benchmark import context_isolation_l814_experiment as l814

    development = set(l814._MODELS) | set(l814._ASPECTS) | set(l814._DIMENSIONS)
    holdout = " ".join(item.phrase for item in generate_episodes(MAX_EPISODES))

    assert not any(word in holdout for word in development)


def test_named_references_generalize_to_unseen_vocabulary(cold):
    # The weights were chosen on lattice-model phrases; they transfer unchanged
    # to signal-processing ones, so coverage is doing the work, not the tuning.
    assert cold["per_kind_accuracy"]["PHRASE_NEW_VOCAB"] == 1.0
    assert cold["per_kind_accuracy"]["ARTIFACT"] == 1.0
    assert cold["per_kind_accuracy"]["UNKNOWN_ENTITY"] == 1.0
    assert cold["per_kind_accuracy"]["AMBIGUOUS"] == 1.0


def test_isolation_generalizes_but_resolution_only_partly(cold, cold_baseline):
    assert cold["metrics"]["stale_intent_leak_rate"] == 0.0
    assert cold["metrics"]["new_task_contamination_rate"] == 0.0
    assert cold["metrics"]["reference_resolution_accuracy"] > cold_baseline["metrics"][
        "reference_resolution_accuracy"
    ]


def test_temporal_and_relational_references_are_not_handled(cold):
    for kind in ("TEMPORAL_FIRST", "TEMPORAL_PREVIOUS", "RELATION_AFTER"):
        assert cold["per_kind_accuracy"][kind] == 0.0


def test_relational_references_produce_confidently_wrong_episodes(cold):
    """The defect this hold-out exists to find; asserted so it cannot be lost.

    L8.14 reported wrong_episode_reuse_rate 0.0 and that looked like a property
    of the resolver.  It was a property of the query distribution.  With
    relational references present, the resolver answers with a wrong episode
    rather than asking: 「その前の実験」 falls through to the bare-reference rule
    and returns the latest, and 「Xの後にやった方」 returns X itself.  Both are
    off-by-one and neither is a clarification, so the safe-failure guarantee
    does not hold outside L8.14.

    Deliberately not fixed here: repairing a defect found on a cold hold-out
    turns the hold-out into development data.  This assertion must be updated
    by whoever adds the relation graph (L8.16), on a fresh hold-out.
    """
    assert cold["metrics"]["wrong_episode_reuse_rate"] > 0.0

    wrong = [
        item
        for item in cold["records"]
        if item["episode"] is not None and item["episode"] != item["expected_episode"]
    ]
    assert {item["kind"] for item in wrong} == {"TEMPORAL_PREVIOUS", "RELATION_AFTER"}


def test_an_unmarked_paraphrase_is_read_as_new_work(cold):
    # 「前に作った…を直して」 carries no marker from L8.13's list and no specific
    # name, so the structural gate calls it new work.  Continuity is lost rather
    # than corrupted, but it is a miss the marker list cannot close by itself.
    missed = [
        item
        for item in cold["records"]
        if item["kind"] == "PARAPHRASE" and item["decision"] == "NEW_EPISODE"
    ]

    assert missed
    assert all("前に作った" in item["text"] for item in missed)


def test_holdout_is_deterministic():
    assert run_holdout(20)["metrics"] == run_holdout(20)["metrics"]
