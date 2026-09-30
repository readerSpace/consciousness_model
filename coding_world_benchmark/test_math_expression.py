"""Tests for the calculator: it answers, and it refuses everything else."""
from __future__ import annotations

import math

import pytest

from coding_world_benchmark.math_expression import (
    MathError, answer_query, evaluate, format_number, free_names, is_query,
)


# -- what it answers -------------------------------------------------------

@pytest.mark.parametrize("source, value", [
    ("1+1", 2), ("2*3+4", 10), ("(2+3)*4", 20), ("7/2", 3.5),
    ("7//2", 3), ("7%2", 1), ("-3+1", -2), ("2**10", 1024),
    ("2^10", 1024), ("3^2^2", 81),
])
def test_arithmetic(source, value):
    assert evaluate(source) == value


def test_caret_is_exponentiation_not_xor():
    """Python would say 8; in a mathematical input language that is wrong."""
    assert evaluate("2^10") == 1024


@pytest.mark.parametrize("source, value", [
    ("sqrt(2)", math.sqrt(2)), ("sin(pi/4)", math.sin(math.pi / 4)),
    ("log(e)", 1.0), ("max(1, 5, 3)", 5), ("abs(-2)", 2),
    ("hypot(3, 4)", 5.0), ("factorial(5)", 120),
])
def test_functions_and_constants(source, value):
    assert evaluate(source) == pytest.approx(value)


def test_bound_names_are_usable():
    assert evaluate("2*a + b", {"a": 3, "b": 1}) == 7


# -- what it refuses -------------------------------------------------------

@pytest.mark.parametrize("source", [
    '__import__("os")', "os.system('ls')", "open('x')", "eval('1')",
    "(lambda: 1)()", "[i for i in range(3)]", "'abc'", "True",
    "a", "1 if True else 2", "print(1)", "{1: 2}", "x.y",
])
def test_anything_outside_the_table_is_refused(source):
    with pytest.raises(MathError):
        evaluate(source)


def test_an_unknown_function_is_named():
    with pytest.raises(MathError) as error:
        evaluate("frobnicate(2)")
    assert "frobnicate" in str(error.value)


def test_division_by_zero_and_huge_powers_are_refused():
    with pytest.raises(MathError):
        evaluate("1/0")
    with pytest.raises(MathError):
        evaluate("9**9**9")


# -- when it speaks up -----------------------------------------------------

@pytest.mark.parametrize("message", ["1+1=?", "2^10 = ?", "sqrt(2)=？", "1+2*3"])
def test_a_question_about_a_number_is_recognised(message):
    assert is_query(message)


@pytest.mark.parametrize("message", [
    "こんにちは", "はい", "/inspect", "これは1+1みたいな話です",
    "64x64の平均を出して", "2026", "ask(\"a\")",
])
def test_everything_else_falls_through(message):
    assert not is_query(message)
    assert answer_query(message) is None


def test_the_answer_shows_the_question_and_the_value():
    assert answer_query("1+1=?") == "`1+1` = **2**"


def test_an_unanswerable_question_says_why_rather_than_guessing():
    spoken = answer_query("1/0=?")
    assert spoken and "0 では割れません" in spoken


# -- helpers ---------------------------------------------------------------

def test_free_names_skips_functions_and_constants():
    assert free_names("sqrt(2)*a + pi") == {"a"}
    assert free_names("1+1") == set()


@pytest.mark.parametrize("value, shown", [
    (2, "2"), (3.5, "3.5"), (2.0, "2"), (1 / 3, "0.3333333333"),
])
def test_numbers_are_shown_plainly(value, shown):
    assert format_number(value) == shown
