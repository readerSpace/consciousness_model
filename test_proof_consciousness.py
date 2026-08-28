from proof_consciousness import And, Atom, Implies, ProofKernel, ProofPlanner, ProofStep


def test_workspace_controlled_planner_proves_conjunction_commutation():
    left, right = Atom("A"), Atom("B")
    planner = ProofPlanner(capacity=2)

    result = planner.prove((And(left, right),), And(right, left))

    assert result.control == "commit"
    assert result.proof is not None
    assert result.state.workspace.concepts
    assert planner.kernel.check(result.proof, (And(left, right),))
    assert planner.strategy_confidence("commit") == 1.0


def test_planner_uses_implication_elimination():
    premise, conclusion = Atom("P"), Atom("Q")
    result = ProofPlanner().prove((premise, Implies(premise, conclusion)), conclusion)

    assert result.proof is not None
    assert result.proof.rule == "imp_elim"


def test_kernel_rejects_a_forged_assumption_proof():
    premise, unrelated = Atom("P"), Atom("Q")
    forged = ProofStep("assumption", unrelated)

    assert not ProofKernel().check(forged, (premise,))


def test_failed_search_retrieves_and_updates_reflective_risk():
    known, unknown = Atom("P"), Atom("Q")
    planner = ProofPlanner()
    initial_risk = planner.controller.risk

    result = planner.prove((known,), unknown)

    assert result.proof is None
    assert result.control == "retrieve"
    assert planner.strategy_confidence("retrieve") == 0.0
    assert planner.controller.risk > initial_risk