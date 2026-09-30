"""Tests for the physics language.

The behaviour that matters is not the integrator -- it is what happens to a
symbol nobody bound.
"""
from __future__ import annotations

import math

import pytest

from coding_world_benchmark.physics_script import (
    PhysicsError, build, looks_like_physics, missing, parse, resolve, run,
    solve,
)

WRITTEN = """vector: B=(0, 0, Bz)
lorentz: F exists
calc(m d^2 x / dt^2 = lorentz(B))
draw(x)"""

BOUND = "Bz = 1.0\nm = 1.0\n" + WRITTEN


def _model(source):
    return build(parse(source))


# -- reading the script ----------------------------------------------------

def test_the_written_script_parses_as_four_statements():
    kinds = [statement.kind for statement in parse(WRITTEN)]
    assert kinds == ["vector", "law", "calc", "draw"]


def test_the_equation_is_recognised_with_or_without_spaces():
    for form in ("m d^2 x / dt^2 = lorentz(B)",
                 "md^2 x / dt^2 = lorentz(B)",
                 "m * d^2x/dt^2 = lorentz(B)"):
        equation = _model(f"vector: B=(0,0,1)\ncalc({form})").equation
        assert equation.mass == "m" and equation.variable == "x"
        assert equation.law == "lorentz" and equation.arguments == ("B",)


def test_comments_and_blank_lines_are_ignored():
    assert len(parse("# 磁場\n\nvector: B=(0,0,1)\n")) == 1


# -- refusing rather than guessing -----------------------------------------

def test_a_misspelled_law_offers_the_nearest_one():
    with pytest.raises(PhysicsError) as error:
        parse("lorentz: F exitst")
    assert "exists" in str(error.value)


def test_a_misspelled_force_offers_the_nearest_one():
    with pytest.raises(PhysicsError) as error:
        build(parse("vector: B=(0,0,1)\ncalc(m d^2 x / dt^2 = lotentz(B))"))
    assert "lorentz" in str(error.value)


@pytest.mark.parametrize("source", [
    "vektor: B=(0,0,1)", "vector: B=(0,0)", "vector: B=0,0,1",
    "calc(dx/dt = lorentz(B))", "calc(m d^2 x / dt^2 = B)",
    "plot(x)", "これは文章です", "",
])
def test_anything_unreadable_is_refused(source):
    with pytest.raises((PhysicsError, Exception)):
        build(parse(source))


def test_a_script_with_one_bad_line_produces_nothing():
    found = run(BOUND + "\nplot(x)")
    assert not found["ok"] and not found["figures"] and not found["results"]


# -- the symbol nobody bound ----------------------------------------------

def test_the_symbols_the_person_wrote_are_required():
    assert missing(_model(WRITTEN)) == ["Bz", "m"]


def test_the_symbols_the_system_brought_are_not():
    """q, x0, v0 and t are the law's and the drawing's, not the person's."""
    wanted = missing(_model(WRITTEN))
    assert not {"q", "x0", "v0", "t"} & set(wanted)


def test_an_unbound_symbol_stops_the_run_and_is_named():
    found = run(WRITTEN)
    assert not found["ok"] and found["needs"] == ["Bz", "m"]
    assert not found["figures"]
    assert "Bz" in found["output"] and "m" in found["output"]


def test_a_field_that_depends_on_position_is_not_silently_uniform():
    """``B = (0, 0, x)`` needs ``x``, so it stops rather than freezing a value."""
    found = run("m = 1.0\nvector: B=(0, 0, x)\nlorentz: F exists\n"
                "calc(m d^2 x / dt^2 = lorentz(B))")
    assert not found["ok"] and "x" in found["needs"]


def test_scalars_may_refer_to_each_other():
    assert missing(_model("a = 2\nBz = 2*a\nm = 1\n" + WRITTEN)) == []


# -- solving ---------------------------------------------------------------

def test_the_written_script_runs_once_its_symbols_are_bound():
    found = run(BOUND)
    assert found["ok"]
    assert len(found["figures"]) == 1
    assert found["figures"][0]["data_uri"].startswith("data:image/png;base64,")


def test_the_supplied_values_are_all_disclosed():
    found = run(BOUND)
    assert set(found["supplied"]) == {"q", "t", "x0", "v0"}
    for name in found["supplied"]:
        assert f"`{name}`" in found["output"]


def test_the_integrator_agrees_with_the_closed_form():
    solution = solve(resolve(_model(BOUND)))
    assert solution.omega == pytest.approx(1.0)
    assert solution.analytic_error < 1e-6


def test_the_motion_is_a_helix_of_the_right_radius_and_pitch():
    solution = solve(resolve(_model(BOUND)))
    radius = [math.hypot(x, y - (-1.0)) for x, y, _ in solution.positions]
    # v0 = (1, 0, 0.5), omega = 1 -> centre (0, -1), radius 1, pitch 0.5 per unit t
    assert max(radius) == pytest.approx(1.0, abs=1e-6)
    assert solution.positions[-1][2] == pytest.approx(0.5 * solution.times[-1],
                                                      abs=1e-6)


def test_an_electric_field_accelerates_uniformly():
    source = ("m = 2.0\nvector: B = (0, 0, 0)\nvector: E = (1, 0, 0)\n"
              "vector: v0 = (0, 0, 0)\ntime: t = (0, 4, 400)\n"
              "lorentz: F exists\ncalc(m d^2 x / dt^2 = lorentz(B, E))")
    solution = solve(resolve(_model(source)))
    # x = 1/2 (q/m) E t^2 with q supplied as 1
    assert solution.positions[-1][0] == pytest.approx(0.5 * 0.5 * 16, rel=1e-6)


@pytest.mark.parametrize("source, reason", [
    ("m = 0\nBz = 1\n" + WRITTEN, "zero mass"),
    ("m = 1\nBz = 1\ntime: t = (0, 1, 1)\n" + WRITTEN, "too few steps"),
    ("m = 1\nBz = 1\ntime: t = (5, 1, 100)\n" + WRITTEN, "time runs backwards"),
])
def test_a_world_that_cannot_be_solved_is_reported(source, reason):
    found = run(source)
    assert not found["ok"] and not found["figures"]


def test_drawing_before_solving_is_refused():
    found = run("draw(x)")
    assert not found["ok"] and not found["figures"]


def test_drawing_something_that_was_not_solved_is_refused():
    found = run(BOUND.replace("draw(x)", "draw(y)"))
    assert not found["ok"] and "y" in found["output"]


# -- detection -------------------------------------------------------------

@pytest.mark.parametrize("source", [
    WRITTEN, BOUND, "vector: B=(0,0,1)", "calc(1+1)", "lorentz: F exitst",
])
def test_a_physics_script_is_recognised(source):
    assert looks_like_physics(source)


@pytest.mark.parametrize("source", [
    "こんにちは", 'ask("a")', "state()", "1+1=?", "/semantics", "はい",
])
def test_everything_else_is_left_alone(source):
    assert not looks_like_physics(source)


def test_calc_of_a_plain_expression_is_arithmetic():
    found = run("calc(1+1)")
    assert found["ok"] and "2" in found["results"][0]["detail"]
