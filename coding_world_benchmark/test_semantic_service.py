"""Tests for the semantic session behind the agent's UI and API.

These run without tkinter on purpose: everything the integration adds lives in
``semantic_service``, so the behaviour is testable on any machine even though the
desktop app itself is not.
"""
from __future__ import annotations

import re

import pytest

from coding_world_benchmark.contextual_polysemy_l835_experiment import (
    AMBIGUOUS, POLYSEMY,
)
from coding_world_benchmark.cross_context_identification_l829_experiment import (
    Observation,
)
from coding_world_benchmark.lexical_revision_l830_experiment import (
    ACQUIRED, REVISED,
)
from coding_world_benchmark.holdout_dialogue_l8331_experiment import (
    ANSWERABLE, REQUEST as HOLDOUT_REQUEST,
)
from coding_world_benchmark.semantic_service import (
    ANSWER, ASKED, BLOCKED, NO_COMMIT, RESOLVED, SEMANTIC_FEATURE_COMMANDS,
    TEMPLATES, SemanticSession, handle_command, render_state, render_template,
    template_by_id, templates_payload,
)

REQUEST = "64x64のケースで処理時間ではなくフガ率をならして"
SURFACE = "フガ率"
RATE, ENERGY = "success_rate", "energy_error"


@pytest.fixture
def session():
    return SemanticSession()


def _resolve(session, answer="はい"):
    session.ask(REQUEST)
    return session.ask(answer)


# -- the conversation the layers already knew how to have ------------------

def test_an_unknown_word_produces_a_question_not_an_answer(session):
    state = session.ask(REQUEST)
    assert state["status"] == ASKED
    assert state["question"]["text"]
    assert state["suspended"] == REQUEST
    assert not state["lexicon"] or not state["lexicon"][0]["authoritative"]


def test_i_dont_know_commits_nothing_and_retires_the_question(session):
    session.ask(REQUEST)
    state = session.ask("分からない")
    assert state["status"] == NO_COMMIT
    assert not state["turns"][-1]["committed"]
    assert state["question"]["id"] != 1


def test_an_answer_resumes_the_goal_and_executes(session):
    state = _resolve(session, "いいえ")
    assert state["status"] == ANSWER
    assert state["resumed"] == REQUEST
    assert state["diagnosis"] == RESOLVED
    assert state["lexicon"][0]["state"] == ACQUIRED


def test_a_contradiction_takes_authority_away_before_anything_else(session):
    # 「いいえ」 lands on success_rate, so energy_error is a real disagreement.
    _resolve(session, "いいえ")
    session.dialogue.lexicon.observe(SURFACE, Observation("EQUALS", ENERGY))
    state = session.state()
    assert state["lexicon"][0]["state"] == "DISPUTED"
    assert not state["lexicon"][0]["authoritative"]
    assert state["invariants"]["authority_while_disputed"] == 0


def test_a_second_disagreement_is_a_revision(session):
    _resolve(session, "いいえ")
    for _ in range(2):
        session.dialogue.lexicon.observe(SURFACE, Observation("EQUALS", ENERGY))
    state = session.state()
    assert state["lexicon"][0]["state"] == REVISED
    assert state["lexicon"][0]["meaning"] == ENERGY


# -- accessibility (L8.42) -------------------------------------------------

def test_a_question_about_an_unopened_episode_is_not_asked():
    session = SemanticSession(episodes=("sim", "lab"), episode="sim")
    session.use_episode("lab")
    state = session.ask(REQUEST)
    assert state["status"] == BLOCKED
    assert state["blocked"] and state["question"]["askable"] is False


def test_answering_while_blocked_commits_nothing():
    session = SemanticSession(episodes=("sim", "lab"), episode="sim")
    session.use_episode("lab")
    session.ask(REQUEST)
    state = session.answer("はい")
    assert state["status"] == BLOCKED
    assert all(not turn["committed"] for turn in state["turns"])


def test_opening_the_episode_unblocks_the_same_question():
    session = SemanticSession(episodes=("sim", "lab"), episode="sim")
    session.use_episode("lab")
    session.ask(REQUEST)
    pending = session.dialogue.pending.question_id
    session.open_episode("lab")
    state = session.state()
    assert not state["blocked"] and state["diagnosis"] == ASKED
    assert state["question"]["id"] == pending
    assert session.ask("はい")["status"] == ANSWER


# -- the model question (L8.35 leaves it open, L8.36 asks it) --------------

def _two_episode_session():
    session = SemanticSession(episodes=("sim", "lab"), episode="sim")
    for _ in range(2):
        session.ask(REQUEST)
        session.ask("いいえ")
    session.open_episode("lab")
    session.use_episode("lab")
    for _ in range(2):
        session.dialogue.lexicon.observe(SURFACE, Observation("EQUALS", ENERGY))
    for _ in range(2):
        session.ask(REQUEST)
    return session


def test_a_sequential_session_leaves_time_and_episode_tied(session):
    found = _two_episode_session()
    assert found.senses.explanations[SURFACE].kind == AMBIGUOUS
    assert found.state()["senses"][0]["by_episode"] == {"sim": None, "lab": None}


def test_the_structural_question_is_offered_exactly_there(session):
    found = _two_episode_session()
    question = found.state()["structural_question"]
    assert question is not None
    assert sorted(question["models"]) == ["POLYSEMY", "REVISION"]
    # L8.36 reports the tie rather than picking, and every branch names a model.
    assert len(question["questions"]) >= 2
    for item in question["questions"]:
        for branch in item["branches"]:
            assert branch["models"]


def test_naming_the_model_decides_it_and_the_senses_separate():
    found = _two_episode_session()
    state = found.answer_structural("場面")
    assert state["senses"][0]["model"] == POLYSEMY
    assert state["senses"][0]["decided"]
    assert state["senses"][0]["by_episode"] == {"sim": RATE, "lab": ENERGY}


def test_the_same_request_then_runs_differently_per_episode():
    found = _two_episode_session()
    found.answer_structural("場面")
    programs = found.rerun(REQUEST)
    assert RATE in programs["sim"] and ENERGY not in programs["sim"]
    assert ENERGY in programs["lab"] and RATE not in programs["lab"]


def test_an_answer_that_names_no_model_changes_nothing():
    found = _two_episode_session()
    before = found.state()["senses"]
    state = found.answer_structural("よくわからない")
    assert state["senses"] == before
    assert not found.decided


def test_no_structural_question_where_the_evidence_already_decides(session):
    _resolve(session)
    assert session.state()["structural_question"] is None


# -- the invariants that need no oracle ------------------------------------

def test_the_session_reports_only_invariants_it_can_check(session):
    state = _resolve(session)
    assert set(state["invariants"]) == {
        "executed_without_authority", "authority_while_disputed",
        "committed_without_commitment", "stale_episode_leak"}
    assert "wrong_answer" not in state["invariants"]


def test_the_invariants_stay_at_zero_through_the_whole_arc():
    found = _two_episode_session()
    found.answer_structural("場面")
    assert all(value == 0 for value in found.state()["invariants"].values())


def test_a_leak_is_narrower_than_a_disagreement(session):
    """Revising inside one episode changes that episode's own meaning."""
    _resolve(session, "いいえ")
    for _ in range(2):
        session.dialogue.lexicon.observe(SURFACE, Observation("EQUALS", ENERGY))
    session.ask(REQUEST)
    assert session.state()["invariants"]["stale_episode_leak"] == 0


# -- the command surface both UIs share ------------------------------------

def test_the_ask_command_is_the_explicit_way_into_the_pipeline(session):
    """Plain chat is left alone; the interpretation pipeline is opted into."""
    state = handle_command(session, f"/ask {REQUEST}")
    assert state is not None and "ASKED" in state
    assert session.dialogue.pending is not None
    assert handle_command(session, "/ask") is not None


def test_commands_are_recognised_and_others_are_not(session):
    assert handle_command(session, "/semantics") is not None
    assert handle_command(session, "/semantics episodes") is not None
    assert handle_command(session, "こんにちは") is None
    assert handle_command(session, "/inspect") is None


def test_open_and_episode_commands_change_the_access_state(session):
    handle_command(session, "/semantics episode lab")
    assert session.episode == "lab" and not session.recallable
    handle_command(session, "/semantics open lab")
    assert session.recallable


def test_reset_discards_the_interpretation_state(session):
    _resolve(session)
    handle_command(session, "/semantics reset")
    state = session.state()
    assert state["lexicon"] == [] and state["turns"] == []


def test_answer_command_routes_to_the_pending_question(session):
    session.ask(REQUEST)
    spoken = handle_command(session, "/answer はい")
    assert spoken and "ANSWER" in spoken
    assert session.state()["diagnosis"] == RESOLVED


def test_the_rendered_state_names_the_store_it_runs_on(session):
    text = render_state(session.state())
    assert "hold-out" in text
    assert "不変条件" in text


#: Some advertised entries are shapes of input rather than literal commands, so
#: the check needs a real example of each shape.
COMMAND_SAMPLES = {
    "/answer": "/answer はい",
    "/ask": f"/ask {REQUEST}",
    "<式>=?": "1+1=?",
    "<関係>ならば？": "A ⊆ B and B ⊆ A ならば？",
}


def test_every_advertised_command_is_handled(session):
    for command, _ in SEMANTIC_FEATURE_COMMANDS:
        stem = command.split(" ")[0]
        sample = COMMAND_SAMPLES.get(command) or COMMAND_SAMPLES.get(
            stem, command.replace("<episode>", "lab")
                         .replace("<name>", "lab")
                         .replace("<label>", "場面")
                         .replace("<reply>", "はい"))
        assert handle_command(session, sample) is not None

# -- the editable templates ------------------------------------------------

def test_the_request_template_is_the_form_the_holdout_verified():
    """Not a similar sentence: the same one, rebuilt from its slots."""
    assert render_template(template_by_id("unknown_argument")) == HOLDOUT_REQUEST
    assert render_template(template_by_id("answerable")) == ANSWERABLE


def test_every_reply_template_is_one_the_holdout_drives_the_system_with():
    replies = {render_template(item) for item in TEMPLATES if item.kind == "ANSWER"}
    assert {"はい", "いいえ", "分からない", "たぶん違う",
            "質問の意味が分からない", "やっぱりこの作業やめて"} <= replies
    assert render_template(template_by_id("correction")) == "それじゃなくて成功率"


def test_every_placeholder_has_a_slot_and_every_slot_a_placeholder():
    for item in TEMPLATES:
        placeholders = set(re.findall(r"\{(\w+)\}", item.form))
        assert placeholders == {slot.name for slot in item.slots}, item.id
        render_template(item)  # would raise on a mismatch


def test_slots_can_be_overridden_one_at_a_time():
    template = template_by_id("unknown_argument")
    filled = render_template(template, {"surface": "ホゲ尺"})
    assert "ホゲ尺" in filled and "64x64" in filled and "フガ率" not in filled


def test_template_ids_are_unique_and_kinds_are_known():
    ids = [item.id for item in TEMPLATES]
    assert len(ids) == len(set(ids))
    assert {item.kind for item in TEMPLATES} == {"REQUEST", "ANSWER", "COMMAND"}


def test_only_request_and_reply_templates_declare_what_they_do():
    """A command's behaviour is its endpoint's; only utterances carry expect."""
    for item in TEMPLATES:
        if item.expect:
            assert item.kind in ("REQUEST", "ANSWER"), item.id
            assert dict(item.expect).keys() == {"decision", "commits"}


def test_the_payload_carries_what_the_panel_renders():
    payload = templates_payload()
    assert len(payload) == len(TEMPLATES)
    for item in payload:
        assert set(item) >= {"id", "kind", "label", "form", "slots", "preview",
                             "note", "verified_by", "expect"}
        assert item["preview"]
        assert item["verified_by"]


def test_every_template_has_a_call_form_with_the_same_slots():
    import re as _re
    for item in TEMPLATES:
        if not item.code:
            continue
        placeholders = set(_re.findall(r"\{(\w+)\}", item.code))
        assert placeholders <= {slot.name for slot in item.slots}, item.id


def test_the_payload_carries_the_call_form():
    codes = {item["id"]: item["code"] for item in templates_payload()}
    assert codes["affirm"] == "yes()"
    assert codes["correction"] == 'correct("成功率")'
    assert codes["unknown_argument"] == f'ask("{HOLDOUT_REQUEST}")'


def test_a_script_typed_into_chat_is_recognised(session):
    spoken = handle_command(session, 'ask("%s")' % HOLDOUT_REQUEST)
    assert spoken is not None and "スクリプトを実行しました" in spoken
    assert session.dialogue.pending is not None


def test_prose_is_still_prose(session):
    assert handle_command(session, "64x64の平均(runtime)を出して") is None
    assert handle_command(session, "こんにちは") is None


def test_arithmetic_is_answered_in_chat(session):
    assert handle_command(session, "1+1=?") == "`1+1` = **2**"
    assert handle_command(session, "2^10=?").endswith("**1024**")


def test_a_physics_script_is_routed_to_the_physics_language(session):
    written = ("vector: B=(0, 0, Bz)\nlorentz: F exists\n"
               "calc(m d^2 x / dt^2 = lorentz(B))\ndraw(x)")
    spoken = handle_command(session, written)
    assert spoken is not None and "値が決まっていない" in spoken
    # It must not have been read as a dialogue script or as a request.
    assert session.dialogue.pending is None and session.turns == []


def test_a_relation_question_is_answered_under_every_reading(session):
    spoken = handle_command(session, "(A in B) and (B in A)ならば？")
    assert spoken is not None
    assert "A = B" in spoken and "矛盾" in spoken


def test_fixing_a_reading_lives_in_the_session(session):
    handle_command(session, "in := subset")
    assert session.operators == {"in": ("⊆", False)}
    spoken = handle_command(session, "(A in B) and (B in A)ならば？")
    assert "矛盾" not in spoken and "A = B" in spoken
    session.reset()
    assert session.operators == {}


def test_the_theorem_bundle_is_reachable_from_chat(session):
    spoken = handle_command(session, "/theorem search assoc")
    assert spoken is not None and "dA_add_assoc" in spoken


def test_the_theorem_commands_are_advertised():
    commands = {command for command, _ in SEMANTIC_FEATURE_COMMANDS}
    assert "/theorem" in commands
    assert any(command.startswith("/theorem search") for command in commands)


def test_the_new_templates_cover_both_languages():
    ids = {item["id"] for item in templates_payload()}
    assert {"math_query", "lorentz_trajectory", "logic_query"} <= ids


def test_a_command_template_is_a_command_the_handler_knows(session):
    for item in TEMPLATES:
        if item.kind == "COMMAND":
            assert handle_command(session, render_template(item)) is not None
