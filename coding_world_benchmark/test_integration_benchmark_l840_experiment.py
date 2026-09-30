"""Tests for L8.40: the twelve-stage integration run and its audit."""
from __future__ import annotations

import pytest

from coding_world_benchmark.contextual_polysemy_l835_experiment import POLYSEMY
from coding_world_benchmark.integration_benchmark_l840_experiment import (
    ENERGY, LAB, RATE, REQUEST, SIM, SURFACE, Audit, run_experiment,
)
from coding_world_benchmark.lexical_revision_l830_experiment import (
    ACQUIRED, DISPUTED, REVISED,
)


@pytest.fixture(scope="module")
def report():
    return run_experiment()


def _stage(report, name):
    return next(s for s in report["stages"] if s["name"] == name)


# -- the invariants --------------------------------------------------------

def test_the_three_invariants_hold_at_every_stage(report):
    found = report["metrics"]
    assert found["wrong_answer"] == 0
    assert found["false_acquisition"] == 0
    assert found["stale_episode_leak"] == 0


def test_every_stage_is_audited_and_meets_its_expectation(report):
    assert report["metrics"]["audited_after_every_stage"]
    assert report["metrics"]["all_stages_ok"]
    for stage in report["stages"]:
        assert set(stage["audit"]) >= {
            "wrong_answer", "false_acquisition", "stale_episode_leak"}


def test_the_audit_can_fail(report):
    """The control: the wrong episode's view is handed in on purpose."""
    control = report["control"]
    assert control["detected"]
    assert ENERGY in control["executed"]
    assert control["audit"]["wrong_answer"]
    assert control["audit"]["stale_episode_leak"]


def test_a_field_never_believed_is_an_error_but_not_a_leak():
    audit = Audit()
    audit.record({SURFACE: RATE})
    never = audit.check(f"MEAN(...{'runtime_seconds'}...)", ENERGY, {})
    assert never["wrong_answer"] and not never["stale_episode_leak"]
    leaked = audit.check(f"MEAN(...{RATE}...)", ENERGY, {})
    assert leaked["wrong_answer"] and leaked["stale_episode_leak"]


# -- the arc ---------------------------------------------------------------

def test_the_unknown_word_stops_execution_before_anything_is_learned(report):
    assert _stage(report, "NEW_EPISODE")["detail"]["decision"] == "ABSTAIN"
    assert _stage(report, "NEW_EPISODE")["executed"] is None


def test_hypotheses_do_not_change_the_decision(report):
    detail = _stage(report, "HYPOTHESES")["detail"]
    assert detail["count"] > 0
    assert not detail["decidable"]
    assert detail["decision_unchanged"]


def test_the_marginals_are_not_mistaken_for_an_answer(report):
    detail = _stage(report, "JOINT_AMBIGUITY")["detail"]
    assert detail["state"] == "MARGINAL_ONLY"
    assert all(len(m) == 2 for m in detail["marginals"])


def test_the_context_effect_is_invisible_to_a_per_word_view(report):
    detail = _stage(report, "CONTEXT_DISCOVERED")["detail"]
    assert detail["marginals_agree"]
    assert not detail["per_word_sees_context"]


def test_a_question_is_planned_over_the_worlds(report):
    detail = _stage(report, "QUESTION_PLANNED")["detail"]
    assert detail["worlds"] > 1
    assert detail["text"]


def test_the_audit_runs_between_turns_not_after_the_conversation(report):
    """The window the invariants are about closes when the dialogue ends."""
    for name in ("GOAL_SUSPENDED", "UNKNOWN_ANSWER"):
        assert _stage(report, name)["authority"] == {}


def test_unknown_commits_nothing_and_is_not_read_as_a_denial(report):
    detail = _stage(report, "UNKNOWN_ANSWER")["detail"]
    assert not detail["committed"]
    assert detail["decision"] == "NO_COMMIT"
    assert _stage(report, "UNKNOWN_ANSWER")["authority"] == {}


def test_the_original_goal_resumes_with_the_acquired_meaning(report):
    stage = _stage(report, "ACQUIRED_AND_RESUMED")
    assert stage["detail"]["resumed"] == REQUEST
    assert stage["detail"]["state"] == ACQUIRED
    assert stage["detail"]["meaning"] == RATE
    assert RATE in stage["executed"]


def test_one_disagreement_suspends_authority_without_remapping(report):
    stage = _stage(report, "DISPUTED")
    assert stage["detail"]["state"] == DISPUTED
    assert stage["authority"] == {}
    assert stage["executed"] is None


def test_the_revision_executes_and_the_discarded_meaning_does_not(report):
    stage = _stage(report, "REVISED_AND_RERUN")
    assert stage["detail"]["state"] == REVISED
    assert stage["detail"]["meaning"] == ENERGY
    assert ENERGY in stage["executed"] and RATE not in stage["executed"]


def test_each_episode_executes_with_its_own_sense(report):
    for context, expected, other in ((SIM, RATE, ENERGY), (LAB, ENERGY, RATE)):
        stage = _stage(report, f"SENSE_EXECUTED[{context}]")
        assert stage["detail"]["model"] == POLYSEMY
        assert stage["detail"]["sense"] == expected
        assert expected in stage["executed"] and other not in stage["executed"]
    assert report["per_context"][SIM] != report["per_context"][LAB]


def test_the_run_covers_the_layers_it_claims(report):
    claimed = " ".join(stage["layers"] for stage in report["stages"])
    for layer in ("L8.27", "L8.28", "L8.29", "L8.30", "L8.31", "L8.33",
                  "L8.34", "L8.35", "L8.36", "L8.37", "L8.38", "L8.39"):
        assert layer in claimed
