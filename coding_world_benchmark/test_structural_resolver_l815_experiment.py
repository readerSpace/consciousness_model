import pytest

from coding_world_benchmark.context_isolation_l814_experiment import default_resolver, run_size
from coding_world_benchmark.structural_resolver_l815_experiment import (
    THRESHOLD_ABSOLUTE,
    THRESHOLD_MARGIN,
    WEIGHT_GOAL,
    WEIGHT_RECENCY,
    ambiguity_cases,
    build_ambiguity_memory,
    canonical,
    query_tokens,
    resolve_reference_structural,
    run_ambiguity_controls,
    score_episodes,
)


@pytest.fixture(scope="module")
def scored_large():
    return run_size(100, resolver=resolve_reference_structural)


@pytest.fixture(scope="module")
def controls():
    return run_ambiguity_controls()


def test_circumstantial_features_cannot_reach_the_margin():
    # Goal agreement and recency may order candidates that evidence already
    # separated; they must never be able to license a CONTINUE by themselves.
    assert WEIGHT_GOAL + WEIGHT_RECENCY < THRESHOLD_MARGIN


def test_coverage_recovers_phrase_references_at_one_hundred_episodes(scored_large):
    assert scored_large["per_kind_accuracy"]["ENTITY_PHRASE"] == 1.0
    assert scored_large["metrics"]["reference_resolution_accuracy"] == 1.0


def test_the_zero_guarantees_are_kept_under_scoring(scored_large):
    metrics = scored_large["metrics"]

    assert metrics["wrong_episode_reuse_rate"] == 0.0
    assert metrics["stale_intent_leak_rate"] == 0.0
    assert metrics["new_task_contamination_rate"] == 0.0


def test_scoring_removes_the_unnecessary_clarifications(scored_large):
    baseline = run_size(100, resolver=default_resolver)

    assert baseline["metrics"]["unnecessary_clarification_rate"] > 0.0
    assert scored_large["metrics"]["unnecessary_clarification_rate"] == 0.0


def test_genuinely_tied_candidates_still_clarify(controls):
    rows = {
        (row["resolver"], row["text"]): row for row in controls["rows"]
    }
    tie = rows[("L8.15", "SU2の検証を続けて")]

    assert tie["decision"] == "CLARIFY"
    assert tie["episode"] is None


def test_a_distinguishing_component_decides_in_both_directions(controls):
    rows = {(row["resolver"], row["text"]): row for row in controls["rows"]}

    assert rows[("L8.15", "3次元SU2の方を続けて")]["episode"] == "ep-003"
    assert rows[("L8.15", "二次元SU2の方を続けて")]["episode"] == "ep-001"


def test_recency_is_not_what_decides_between_the_two(controls):
    # ep-003 is the most recent, so a recency-led resolver would answer it for
    # both queries.  The older episode must win when the query names it.
    rows = {(row["resolver"], row["text"]): row for row in controls["rows"]}
    older = rows[("L8.15", "二次元SU2の方を続けて")]["episode"]

    assert older == "ep-001"
    assert older != controls["memory"].episodes[-1].episode_id


def test_the_controls_are_ones_the_old_resolver_fails(controls):
    failures = [
        row for row in controls["rows"] if row["resolver"] == "L8.13" and not row["correct"]
    ]

    assert {row["text"] for row in failures} == {"3次元SU2の方を続けて", "二次元SU2の方を続けて"}
    assert all(row["correct"] for row in controls["rows"] if row["resolver"] == "L8.15")


def test_kanji_and_ascii_numerals_fold_together():
    assert canonical("三次元") == "3次元"
    assert canonical("3次元") == "3次元"
    assert "3次元" in query_tokens("3次元SU2の方を続けて")
    assert "3次元" in query_tokens("三次元SU2の方を続けて")


def test_scores_expose_the_coverage_that_decided(controls):
    scores = score_episodes(controls["memory"], "3次元SU2の方を続けて")

    assert scores[0].episode_id == "ep-003"
    assert scores[0].coverage == 1.0
    assert scores[0].score >= THRESHOLD_ABSOLUTE
    assert scores[0].score - scores[1].score >= THRESHOLD_MARGIN


def test_every_ambiguity_control_has_a_stated_reason():
    assert all(case.note for case in ambiguity_cases())
    assert build_ambiguity_memory().episodes
