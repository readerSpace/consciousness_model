import pytest

from coding_world_benchmark.epistemic_dialogue_l833_experiment import (
    AFFIRM, AWAITING_ANSWER, COMMITTING, CONFUSED, CORRECTION, DENY, HEDGE, IDLE,
    INTERRUPT, OTHER, UNKNOWN, Dialogue, classify, handle, named_field,
)
from coding_world_benchmark.holdout_binding_l8211_experiment import build_holdout
from coding_world_benchmark.holdout_dialogue_l8331_experiment import (
    ANSWERABLE, REQUEST, build_scenarios, run_experiment,
)
from coding_world_benchmark.holdout_lexicon_l8281_experiment import widen_store


@pytest.fixture()
def dialogue():
    return Dialogue.over(widen_store(build_holdout(24)[0]))


@pytest.fixture(scope="module")
def report():
    return run_experiment()


# --------------------------------------------------------------------------
# what a reply is
# --------------------------------------------------------------------------

def test_a_hedge_is_not_a_denial():
    """The cue table is ordered longest-first, and the order is the rule.

    「たぶん違う」 contains 「違う」, so matching the short form first would turn every
    hesitation into a commitment.
    """
    assert classify("たぶん違う") == HEDGE
    assert classify("違う") == DENY
    assert classify("違うと思う") == HEDGE


def test_not_knowing_is_not_saying_no():
    """The shortcut that silently manufactures evidence the user never gave."""
    assert classify("分からない") == UNKNOWN
    assert classify("分からない") != DENY


def test_the_reply_kinds_are_separated():
    assert classify("はい") == AFFIRM
    assert classify("質問の意味が分からない") == CONFUSED
    assert classify("それじゃなくて成功率") == CORRECTION
    assert classify("やっぱりこの作業やめて") == INTERRUPT
    assert classify("64x64のケースの処理時間をならして") == OTHER
    assert named_field("それじゃなくて成功率") == "success_rate"
    assert named_field("それじゃなくて") is None


# --------------------------------------------------------------------------
# asking suspends; answering resumes
# --------------------------------------------------------------------------

def test_a_question_suspends_the_goal_rather_than_discarding_it(dialogue):
    turn = handle(dialogue, REQUEST)
    assert turn.decision == "ASKED"
    assert dialogue.state == AWAITING_ANSWER
    assert dialogue.suspended.request == REQUEST
    assert dialogue.pending.question_id == 1


def test_the_question_uses_what_the_request_already_said(dialogue):
    """「処理時間ではなく…」 excludes one field, so asking about it again would be
    asking for something the user just supplied in the same sentence."""
    turn = handle(dialogue, REQUEST)
    assert "runtime_seconds" not in turn.reply
    assert dialogue.suspended.candidates == {"success_rate", "energy_error"}


def test_an_answer_resumes_the_original_request(dialogue):
    handle(dialogue, REQUEST)
    turn = handle(dialogue, "いいえ")
    assert turn.kind == DENY
    assert turn.bound_to == 1
    assert turn.resumed == REQUEST
    assert turn.decision == "ANSWER"
    assert dialogue.state == IDLE


def test_a_correction_answers_more_than_was_asked(dialogue):
    handle(dialogue, REQUEST)
    turn = handle(dialogue, "それじゃなくて成功率")
    assert turn.observation is not None and turn.observation.kind == "EQUALS"
    assert dialogue.lexicon.belief("フガ率").meaning == "success_rate"


# --------------------------------------------------------------------------
# what must not be committed
# --------------------------------------------------------------------------

@pytest.mark.parametrize("reply", ["分からない", "たぶん違う", "質問の意味が分からない"])
def test_a_non_answer_commits_no_observation(dialogue, reply):
    handle(dialogue, REQUEST)
    before = len(dialogue.lexicon.belief("フガ率").observations)
    turn = handle(dialogue, reply)
    assert turn.observation is None
    assert turn.decision == "NO_COMMIT"
    assert len(dialogue.lexicon.belief("フガ率").observations) == before


def test_unknown_retires_the_question_instead_of_answering_it(dialogue):
    handle(dialogue, REQUEST)
    turn = handle(dialogue, "分からない")
    assert dialogue.pending is not None
    assert dialogue.pending.question_id == 2
    assert dialogue.pending.query != dialogue.transcript[0].observation
    assert "取り下げます" in turn.reply


def test_an_interruption_leaves_no_trace(dialogue):
    handle(dialogue, REQUEST)
    before = len(dialogue.lexicon.belief("フガ率").observations)
    turn = handle(dialogue, "やっぱりこの作業やめて")
    assert turn.decision == "ABANDONED"
    assert turn.observation is None
    assert dialogue.state == IDLE and dialogue.suspended is None
    assert len(dialogue.lexicon.belief("フガ率").observations) == before


# --------------------------------------------------------------------------
# the stale-binding accident
# --------------------------------------------------------------------------

def test_a_late_answer_binds_to_nothing(dialogue):
    handle(dialogue, REQUEST)
    handle(dialogue, "はい")
    before = len(dialogue.lexicon.belief("フガ率").observations)
    turn = handle(dialogue, "いいえ")
    assert turn.bound_to is None
    assert turn.decision == "UNBOUND"
    assert turn.observation is None
    assert len(dialogue.lexicon.belief("フガ率").observations) == before


def test_an_answer_binds_to_the_newest_question(dialogue):
    handle(dialogue, REQUEST)
    handle(dialogue, "分からない")  # retires Q1, opens Q2
    turn = handle(dialogue, "はい")
    assert turn.bound_to == 2


# --------------------------------------------------------------------------
# not asking when there is nothing to ask about
# --------------------------------------------------------------------------

def test_an_answerable_request_produces_no_question(dialogue):
    turn = handle(dialogue, ANSWERABLE)
    assert turn.decision == "ANSWER"
    assert dialogue.state == IDLE and dialogue.pending is None


# --------------------------------------------------------------------------
# the hold-out
# --------------------------------------------------------------------------

def test_the_invariants(report):
    found = report["metrics"]
    assert found["wrong_observation_commit_rate"] == 0.0
    assert found["false_resolution_rate"] == 0.0
    assert found["stale_question_answer_binding_rate"] == 0.0
    assert found["unnecessary_question_rate"] == 0.0


def test_binding_and_resume_are_accurate(report):
    found = report["metrics"]
    assert found["answer_to_question_binding_accuracy"] == 1.0
    assert found["suspended_goal_resume_accuracy"] == 1.0
    assert found["commit_exactness"] == 1.0


def test_every_reply_kind_is_exercised(report):
    kinds = {turn["kind"] for run in report["runs"] for turn in run["turns"]}
    assert {AFFIRM, DENY, CORRECTION, UNKNOWN, HEDGE, CONFUSED, INTERRUPT} <= kinds


def test_the_two_directions_of_commitment_both_resolve(report):
    runs = {run["name"]: run for run in report["runs"]}
    assert runs["AFFIRM"]["resolved_field"] == "energy_error"
    assert runs["DENY"]["resolved_field"] == "success_rate"
    assert runs["UNKNOWN"]["resolved_field"] is None


def test_the_report_formats(report):
    from coding_world_benchmark.holdout_dialogue_l8331_experiment import format_experiment

    text = format_experiment(report)
    assert "記録なし" in text and "stale_question_answer_binding_rate" in text
