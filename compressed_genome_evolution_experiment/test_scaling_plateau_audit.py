"""Tests for exp605 -- Scaling / Plateau Audit."""

from __future__ import annotations

import cumulative_instruction_bootstrapping as B
import scaling_plateau_audit as S


def test_scalar_semantic_solver_matches_witness_dp_at_exp604_scale():
    ecology = S.Ecology(S.AuditConfig(steps=30, capacity=64, max_arity=8))
    for retained in ((), (2,), (2, 4)):
        for arity in (2, 4, 8):
            assert ecology.semantic_cost(arity, retained) == B.semantic_complexity(
                arity, ecology.history(retained))[0]


def test_basis_gating_controls_reachable_task_catalog():
    result = S.run_audit("adaptive_language", S.AuditConfig(30, 64, 8), 1)
    for record in result["records"]:
        if record["arity"] == 4:
            assert 2 in record["basis_before"]
        if record["arity"] == 8:
            assert 4 in record["basis_before"]


def test_late_births_are_reinvention_not_late_semantic_novelty():
    result = S.run_audit("adaptive_language", S.AuditConfig(30, 64, 8), 1)
    summary = result["summary"]
    assert summary["n_semantic_novelty"] == 3
    assert summary["late_births"] > 0
    assert summary["late_semantic_novelty"] == 0
    assert summary["n_reinvent"] > 0
    assert summary["catalog_exhausted"]


def test_larger_capacity_unlocks_more_of_the_same_catalog():
    low = S.run_audit("adaptive_language", S.AuditConfig(30, 64, 64), 1)["summary"]
    high = S.run_audit("adaptive_language", S.AuditConfig(30, 512, 64), 1)["summary"]
    assert high["max_arity"] > low["max_arity"]
    assert high["n_semantic_novelty"] > low["n_semantic_novelty"]
    assert high["capacity_fraction"] < 0.95


def test_semantic_distance_is_zero_only_for_identical_typed_operations():
    assert S.Ecology.semantic_distance(8, 8) == 0.0
    assert S.Ecology.semantic_distance(8, 16) == 1.0
