"""Tests for the relation reasoner.

The property that matters is not that it derives things. It is that an ambiguous
operator produces every reading rather than one.
"""
from __future__ import annotations

import pytest

from coding_world_benchmark.logic_reasoner import (
    CONTRADICTION, EQUAL, LogicError, MEMBER, PROPER, SUBSET, Fact, analyse,
    answer, close, fact, is_logic_query, parse, parse_binding, readings,
)

BOTH_WAYS = "(A in B) and (B in A)ならば？"


def _verdicts(question, bound=None):
    found = answer(question, bound)
    assert found["ok"], found["error"]
    return {item["label"]: (item["verdict"], item["answer"])
            for item in found["readings"]}


# -- the ambiguity is carried, not resolved --------------------------------

def test_the_two_readings_of_in_give_different_answers():
    found = answer(BOTH_WAYS)
    assert found["ambiguous"] == ["in"]
    assert not found["agree"]
    verdicts = [(item["verdict"], item["answer"]) for item in found["readings"]]
    assert ("DERIVED", "A = B") in verdicts
    assert any(verdict == CONTRADICTION for verdict, _ in verdicts)


def test_both_readings_are_labelled_in_the_answer():
    output = answer(BOTH_WAYS)["output"]
    assert "部分集合 ⊆" in output and "要素 ∈" in output
    assert "A = B" in output and "正則性" in output


def test_the_answer_says_how_to_fix_the_reading():
    output = answer(BOTH_WAYS)["output"]
    assert "in := subset" in output and "in := member" in output


def test_a_fixed_reading_leaves_one_answer():
    symbol, reading = parse_binding("in := subset")
    found = answer(BOTH_WAYS, {symbol: reading})
    assert found["ambiguous"] == []
    assert len(found["readings"]) == 1
    assert found["readings"][0]["answer"] == "A = B"


def test_subset_sign_is_ambiguous_too():
    """Some authors write ⊂ for ⊆ and others for ⊊."""
    found = answer("A ⊂ B and B ⊂ A ならば？")
    assert found["ambiguous"] == ["⊂"]
    verdicts = {item["verdict"] for item in found["readings"]}
    assert verdicts == {"DERIVED", CONTRADICTION}


def test_an_unknown_reading_name_is_refused():
    with pytest.raises(LogicError):
        parse_binding("in := frobnicate")


# -- the rules -------------------------------------------------------------

@pytest.mark.parametrize("question, wanted", [
    ("A ⊆ B かつ B ⊆ C ならば？", "A ⊆ C"),
    ("A ⊆ B and B ⊆ A ならば？", "A = B"),
    ("A ∈ B and B ⊆ C ならば？", "A ∈ C"),
])
def test_what_follows_follows(question, wanted):
    found = answer(question)
    assert found["readings"][0]["verdict"] == "DERIVED"
    assert wanted in found["readings"][0]["answer"]


@pytest.mark.parametrize("question", [
    "A ∈ B and B ∈ A ならば？",
    "A ⊊ B and B ⊊ A ならば？",
    "A ∈ A ならば？",
    "A = B and A ≠ B ならば？",
])
def test_what_cannot_be_is_reported_as_a_contradiction(question):
    found = answer(question)
    assert any(item["verdict"] == CONTRADICTION for item in found["readings"])


def test_a_proposed_conclusion_is_checked():
    assert answer("A ⊆ B and B ⊆ A => A = B ?")["readings"][0]["verdict"] == "FOLLOWS"


def test_a_proposed_conclusion_that_does_not_follow_is_denied():
    found = answer("A ∈ B and B ∈ C => A ∈ C ?")
    assert found["readings"][0]["verdict"] == "DOES_NOT_FOLLOW"
    assert "出てきません" in found["output"]


def test_membership_is_not_transitive():
    facts, _, contradiction = close([fact(MEMBER, "A", "B"), fact(MEMBER, "B", "C")])
    assert contradiction is None
    assert Fact(MEMBER, "A", "C") not in facts


def test_a_subset_is_not_a_member():
    facts, _, _ = close([fact(SUBSET, "A", "B")])
    assert Fact(MEMBER, "A", "B") not in facts


def test_the_usual_mistakes_are_listed():
    output = answer("A ∈ B and B ∈ C ならば？")["output"]
    assert "∈ は推移的ではありません" in output


def test_every_derived_fact_carries_the_rule_that_made_it():
    for step in answer("A ⊆ B かつ B ⊆ C ならば？")["readings"][0]["derived"]:
        assert step["rule"] and step["why"] and step["because"]


def test_equality_is_one_fact_not_two():
    derived = answer("A ⊆ B and B ⊆ A ならば？")["readings"][0]["derived"]
    equalities = [step for step in derived if " = " in step["fact"]]
    assert len(equalities) == 1 and equalities[0]["fact"] == "A = B"


def test_nothing_new_is_said_when_nothing_follows():
    assert answer("A ⊆ B ならば？")["readings"][0]["verdict"] == "NOTHING_NEW"


# -- what it refuses -------------------------------------------------------

@pytest.mark.parametrize("question", [
    "A or B ならば？", "A ⊆ B または B ⊆ A ならば？",
    "∀x (x ∈ A) ならば？", "A ⊆ ならば？", "ならば？",
    "A ⊆ {1,2} ならば？",
])
def test_anything_outside_the_fence_is_refused(question):
    found = answer(question)
    assert not found["ok"] and found["error"]


def test_too_many_ambiguous_symbols_asks_for_a_decision():
    with pytest.raises(LogicError):
        readings(parse("A in B and C ⊂ D and E ⊃ F"))


def test_two_ambiguous_symbols_are_still_shown_in_full():
    assert len(readings(parse("A in B and C ⊂ D"))) == 4


# -- detection -------------------------------------------------------------

@pytest.mark.parametrize("message", [
    BOTH_WAYS, "A ⊆ B かつ B ⊆ C ならば？", "A ∈ B => A ∈ B ?",
])
def test_a_relation_question_is_recognised(message):
    assert is_logic_query(message)


@pytest.mark.parametrize("message", [
    "1+1=?", "こんにちは", "state()", "はい", "/semantics",
    "64x64の平均は？", "これは in の話ではありません", 'ask("a")',
])
def test_everything_else_is_left_alone(message):
    assert not is_logic_query(message)


def test_symmetric_facts_have_one_orientation():
    assert fact(EQUAL, "B", "A") == fact(EQUAL, "A", "B")
    assert fact(SUBSET, "B", "A") != fact(SUBSET, "A", "B")


def test_the_written_question_parses_despite_its_stray_paren():
    """The question as it was actually typed, unbalanced bracket and all."""
    parsed = parse("A in B) and (B in A)ならば？")
    assert parsed.premises == (("A", "in", "B"), ("B", "in", "A"))


def test_proper_subset_unpacks_into_subset_and_inequality():
    derived = {step["fact"] for step in
               answer("A ⊊ B ならば？")["readings"][0]["derived"]}
    assert {"A ⊆ B", "A ≠ B"} <= derived


def test_analyse_reports_the_reading_it_used():
    found = analyse(parse(BOTH_WAYS), {"in": (MEMBER, False)})
    assert found["verdict"] == CONTRADICTION
    assert "要素 ∈" in found["label"]


def test_proper_is_narrowed_from_subset_and_inequality():
    derived = {step["fact"] for step in
               answer("A ⊆ B and A ≠ B ならば？")["readings"][0]["derived"]}
    assert "A ⊊ B" in derived
    assert PROPER in "".join(derived)
