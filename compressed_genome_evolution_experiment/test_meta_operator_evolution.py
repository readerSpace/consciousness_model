"""Focused tests for exp609 -- Meta-Operator Evolution."""

from __future__ import annotations

import meta_operator_evolution as M


def test_meta_rule_class_is_behavioral_not_identifier():
    alias = M.MetaRule(99, M.R1.program)
    assert M.R1.probe_signature() == alias.probe_signature()
    assert M.meta_rule_distance(M.R1, alias) == 0.0
    assert M.meta_rule_distance(M.R1, M.R2) > 0.0


def test_adaptive_meta_rule_birth_is_late_reused_and_load_bearing():
    result = M.run_meta_evolution(M.ADAPTIVE_META, M.Config(100), 1)
    summary, lineage = result["summary"], result["meta_lineage"]
    assert summary["late_functional_meta_rules"] == 1
    assert lineage[2]["birth"] >= 50
    assert lineage[2]["reuse"] >= 2
    assert lineage[2]["knockout_delta"] > 0
    assert lineage[2]["functional"]


def test_fixed_meta_rule_grows_less_than_adaptive_without_meta_birth():
    config = M.Config(100)
    fixed = M.run_meta_evolution(M.FIXED_META, config, 1)["summary"]
    adaptive = M.run_meta_evolution(M.ADAPTIVE_META, config, 1)["summary"]
    assert fixed["n_meta_rule_classes"] == 1
    assert fixed["n_functional_meta_rules"] == 0
    assert fixed["late_functional_meta_rules"] == 0
    assert adaptive["n_operator_classes"] > fixed["n_operator_classes"]
    assert adaptive["frontier_operator"] > fixed["frontier_operator"]


def test_operator_lineage_records_which_meta_rule_generated_each_child():
    result = M.run_meta_evolution(M.ADAPTIVE_META, M.Config(30), 1)
    lineage = result["operator_lineage"]
    assert lineage[0]["rule"] == 1
    assert any(item["rule"] == 2 for item in lineage)
    assert all(len(item["parents"]) == 2 for item in lineage)