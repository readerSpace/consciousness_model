"""Tests for exp603 -- Abstraction Retention Audit."""

from __future__ import annotations

import abstraction_retention_audit as A


def test_maintenance_costs_are_dependency_aware():
    costs = A.operation_accounting()
    assert costs == {"O1": 14, "O2_given_O1": 18, "O2_compiled_without_O1": 36}
    assert A.FULL_HISTORY.language_bits() == 32
    assert A.KEEP_O2.language_bits() == 36


def test_delete_o1_requires_self_contained_o2_definition():
    full_k, _ = A.task_cost(8, A.FULL_HISTORY)
    compiled_k, _ = A.task_cost(8, A.KEEP_O2)
    assert full_k == 30
    assert compiled_k == 36
    assert A.KEEP_O2.language_bits() > A.FULL_HISTORY.language_bits()


def test_single_o3_prunes_o2_but_retains_useful_o1():
    result = A.audit_workload((8,))
    assert result["free_choice"]["state"] == "delete_O2"
    assert result["free_choice"]["retains"] == ["O1"]
    assert result["conditions"]["delete_O2"]["L_total"] == 52
    assert result["conditions"]["full_history"]["L_total"] == 62


def test_future_reuse_retains_both_intermediate_abstractions():
    result = A.audit_workload((8, 8, 8))
    assert result["free_choice"]["state"] == "full_history"
    assert result["free_choice"]["retains"] == ["O1", "O2"]
    assert result["conditions"]["full_history"]["L_total"] == 122
    assert result["conditions"]["delete_O2"]["L_total"] == 128


def test_free_choice_dominates_forced_delete_and_reset_controls():
    for workload in ((8,), (8, 8, 8)):
        result = A.audit_workload(workload)
        chosen = result["free_choice"]["L_total"]
        assert chosen <= result["conditions"]["reset"]["L_total"]
        assert chosen <= result["conditions"]["delete_O1_compiled_O2"]["L_total"]
