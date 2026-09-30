from coding_world_benchmark.math_spec_to_code_l61_experiment import (
    generate_program,
    semantic_check,
    specification_from_text,
)


def test_specification_ir_extracts_ode_and_solver():
    spec = specification_from_text("Solve dx/dt = -k*x with RK4 and check the solution")
    assert spec.states == ("x",)
    assert spec.parameters == ("k",)
    assert spec.solver == "RK4"
    assert spec.equations[0].rhs == "-k*x"


def test_generated_program_is_ast_valid_and_matches_equation():
    spec = specification_from_text("dx/dt = -k*x, use RK4")
    artifact = generate_program(spec)
    assert artifact.tree.body
    assert "def rk4_step" in artifact.source
    result = semantic_check(artifact)
    assert result["passed"]
    assert result["max_error"] == 0.0


def test_euler_generation_is_available_as_explicit_baseline():
    spec = specification_from_text("dx/dt = -k*x")
    artifact = generate_program(spec)
    assert artifact.solver == "Euler"
    assert "def simulate" in artifact.source
    assert semantic_check(artifact)["passed"]


def test_multistate_rk4_checks_energy_like_invariant():
    spec = specification_from_text("dx/dt = -y; dy/dt = x with RK4 and energy conservation")
    assert spec.states == ("x", "y")
    assert spec.invariants == ("energy conservation",)
    artifact = generate_program(spec)
    assert "zip(state, k1" in artifact.source
    result = semantic_check(artifact)
    assert result["passed"]
    assert result["invariant_passed"]
    assert result["invariant_drift"] < 1e-6
