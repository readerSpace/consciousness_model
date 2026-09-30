"""Tests for the Python-shaped dialogue surface.

Two properties carry the design: nothing outside a fixed table is accepted, and
a script that is not wholly readable is wholly not run.
"""
from __future__ import annotations

import pytest

from coding_world_benchmark.dialogue_script import (
    BY_NAME, SIGNATURES, ScriptError, help_text, looks_like_script, parse, run,
)
from coding_world_benchmark.holdout_dialogue_l8331_experiment import (
    REQUEST as HOLDOUT_REQUEST,
)
from coding_world_benchmark.semantic_service import SemanticSession

ASK = f'ask("{HOLDOUT_REQUEST}")'


@pytest.fixture
def session():
    return SemanticSession()


# -- what is accepted ------------------------------------------------------

def test_a_sequence_of_calls_parses_in_order():
    calls = parse('open_episode("lab")\nepisode("lab")\n' + ASK + "\nno()")
    assert [call.name for call in calls] == ["open_episode", "episode", "ask", "no"]
    assert calls[0].args == ("lab",)
    assert calls[-1].line == 4


def test_comments_and_blank_lines_are_ignored():
    calls = parse("# まず開く\n\nopen_episode('lab')\n\n# そして訊く\nstate()")
    assert [call.name for call in calls] == ["open_episode", "state"]


def test_keywords_are_accepted_where_the_signature_has_them():
    call = parse('rerun(request="平均を出して")')[0]
    assert call.kwargs == {"request": "平均を出して"}


def test_a_fenced_block_is_unwrapped():
    calls = parse("```python\nstate()\n```")
    assert [call.name for call in calls] == ["state"]


# -- what is refused -------------------------------------------------------

@pytest.mark.parametrize("source, reason", [
    ("import os", "import"),
    ("x = ask('a')", "assignment"),
    ("foo()", "unknown call"),
    ("os.system('ls')", "attribute call"),
    ("ask(name)", "a name instead of a literal"),
    ("ask(f'{x}')", "an f-string"),
    ("ask(state())", "a nested call"),
    ("ask(**{'request': 'a'})", "**kwargs"),
    ("ask('a', 'b')", "too many arguments"),
    ("ask()", "a missing required argument"),
    ("episode('a', name='b')", "a doubled argument"),
    ("ask('a') if True else no()", "an expression"),
    ("for i in range(3): no()", "control flow"),
    ("", "nothing to run"),
])
def test_everything_outside_the_table_is_refused(source, reason):
    with pytest.raises(ScriptError):
        parse(source)


def test_an_unknown_call_says_what_is_available():
    with pytest.raises(ScriptError) as error:
        parse("explode()")
    assert "explode" in str(error.value) and "ask" in str(error.value)


def test_the_refusal_names_the_line():
    with pytest.raises(ScriptError) as error:
        parse(ASK + "\nno()\nimport os")
    assert error.value.line == 3


# -- the whole-script guarantee --------------------------------------------

def test_a_script_with_one_bad_line_runs_none_of_it(session):
    found = run(session, ASK + "\nimport os")
    assert not found["ok"] and found["error"]
    assert session.dialogue.pending is None
    assert session.turns == []
    assert found["state"]["lexicon"] == []


def test_a_good_script_runs_every_line(session):
    found = run(session, 'open_episode("lab")\n' + ASK + "\nno()")
    assert found["ok"]
    assert [item["call"] for item in found["results"]] == ["open_episode", "ask", "no"]
    assert found["results"][-1]["status"] == "ANSWER"
    assert session.state()["diagnosis"] == "RESOLVED"


# -- the calls are the session's own actions -------------------------------

def test_the_shorthands_are_the_replies_the_holdout_fixed(session):
    run(session, ASK)
    spoken = run(session, "dont_know()")
    assert spoken["results"][0]["status"] == "NO_COMMIT"
    assert not session.turns[-1].committed


def test_yes_is_the_same_move_as_answering_in_words():
    by_code, by_words = SemanticSession(), SemanticSession()
    run(by_code, ASK)
    run(by_code, "yes()")
    by_words.ask(HOLDOUT_REQUEST)
    by_words.answer("はい")
    assert by_code.state()["lexicon"] == by_words.state()["lexicon"]
    assert by_code.state()["diagnosis"] == by_words.state()["diagnosis"]


def test_rerun_without_an_argument_uses_the_last_request(session):
    run(session, 'open_episode("lab")\n' + ASK + "\nno()")
    found = run(session, "rerun()")
    assert found["ok"]
    assert "MEAN(" in found["results"][0]["detail"]


def test_reset_clears_the_session(session):
    run(session, ASK + "\nno()")
    run(session, "reset()")
    assert session.state()["lexicon"] == [] and session.turns == []


def test_the_theorem_calls_reach_the_bundle(session):
    found = run(session, 'theorem_search("assoc")')
    assert found["ok"] and "dA_add_assoc" in found["results"][0]["detail"]


def test_a_theorem_run_from_a_script_keeps_the_caveat(session):
    found = run(session, "theorem_run(8)")
    assert found["ok"]
    assert "「証明された」ではありません" in found["results"][0]["detail"]


def test_help_lists_every_call():
    text = help_text()
    for signature in SIGNATURES:
        assert signature.example in text
        assert signature.doc in text


# -- detection -------------------------------------------------------------

@pytest.mark.parametrize("message", [
    "state()", "no()", ASK, "```python\nstate()\n```",
])
def test_a_message_that_parses_completely_is_a_script(message):
    assert looks_like_script(message)


@pytest.mark.parametrize("message", [
    "こんにちは", "はい", "/inspect", "/semantics open lab",
    "64x64(の平均)を出して", "平均(runtime)", "これは ask( で終わる",
    "os.system('ls')", "print('hi')",
])
def test_anything_else_falls_through_untouched(message):
    assert not looks_like_script(message)


def test_the_table_and_the_signatures_agree():
    assert set(BY_NAME) == {signature.name for signature in SIGNATURES}
    for signature in SIGNATURES:
        assert signature.required <= len(signature.params)
        assert signature.example.startswith(signature.name + "(")
