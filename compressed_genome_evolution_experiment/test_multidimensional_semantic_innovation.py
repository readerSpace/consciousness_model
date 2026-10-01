"""Tests for exp607 -- Multidimensional Semantic Innovation."""

from __future__ import annotations

import multidimensional_semantic_innovation as M


def test_probe_semantics_cluster_by_behavior_not_operator_name():
    assert M.operator_semantic_class("REVERSE") != M.operator_semantic_class("MAP_FLIP")
    assert M.semantic_distance(M.Task(2, ("ITERATE",)), M.Task(2, ("ITERATE",))) == 0.0
    assert M.structural_distance(M.Task(2, ("ITERATE",)), M.Task(2, ("ITERATE", "REVERSE"))) == 1.0


def test_parametric_generator_has_no_structural_novelty():
    summary = M.run_innovation(M.PARAMETRIC, "adaptive_language", M.Config(100), 1)["summary"]
    assert summary["late_parametric_novelty"] > 0
    assert summary["late_structural_novelty"] == 0
    assert summary["operator_classes"] == 1


def test_compositional_generator_continues_structural_novelty_and_classes():
    summary = M.run_innovation(M.COMPOSITIONAL, "adaptive_language", M.Config(300), 1)["summary"]
    assert summary["late_structural_novelty"] > 0
    assert summary["structural_novelty_rate_late"] > 0
    assert summary["operator_classes"] > 1
    assert summary["n_retained_final"] > 1
    assert summary["total_reuses"] > 0


def test_fixed_and_no_retention_controls_plateau_early():
    config = M.Config(300)
    adaptive = M.run_innovation(M.COMPOSITIONAL, "adaptive_language", config, 1)["summary"]
    fixed = M.run_innovation(M.COMPOSITIONAL, "fixed_language", config, 1)["summary"]
    no_retention = M.run_innovation(M.COMPOSITIONAL, "no_retention", config, 1)["summary"]
    assert fixed["late_structural_novelty"] == no_retention["late_structural_novelty"] == 0
    assert adaptive["frontier_depth"] > fixed["frontier_depth"]
    assert adaptive["frontier_depth"] > no_retention["frontier_depth"]


def test_keep_all_preserves_all_invented_operator_classes():
    summary = M.run_innovation(M.COMPOSITIONAL, "keep_all", M.Config(100), 1)["summary"]
    assert summary["n_retained_final"] == summary["operator_classes"] == 4
