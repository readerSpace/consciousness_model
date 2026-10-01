"""Tests for exp606 -- Generative Semantic Space."""

from __future__ import annotations

import generative_semantic_space as G


def test_probe_distance_is_semantic_not_tree_identity():
    first = G.Task(3, reverse=True, flip=False)
    assert G.probe_distance(first, G.Task(3, True, False)) == 0.0
    assert G.probe_distance(first, G.Task(4, True, False)) > 0.5
    assert G.probe_distance(first, G.Task(3, False, False)) > 0.5


def test_variable_length_tasks_remain_comparable_without_expansion():
    assert G.probe_distance(G.Task(3000, True, True), G.Task(2999, True, True)) == 1.0
    assert G.raw_cost(G.Task(3000, False, False)) == 30004


def test_adaptive_invents_retains_and_reuses_a_compressing_instruction():
    summary = G.run_space("adaptive_language", G.Config(100, 64), 1)["summary"]
    assert summary["n_birth"] == summary["n_retained"] == 1
    assert summary["reuses"] > 0
    assert summary["max_D_language"] < 64


def test_adaptive_has_continuing_late_novelty_but_fixed_language_plateaus():
    config = G.Config(300, 64)
    adaptive = G.run_space("adaptive_language", config, 1)["summary"]
    fixed = G.run_space("fixed_language", config, 1)["summary"]
    assert adaptive["late_semantic_novelty"] > 0
    assert adaptive["frontier_depth"] > fixed["frontier_depth"]
    assert fixed["late_semantic_novelty"] == 0


def test_capacity_below_definition_cost_blocks_the_adaptive_frontier():
    low = G.run_space("adaptive_language", G.Config(100, 16), 1)["summary"]
    high = G.run_space("adaptive_language", G.Config(100, 32), 1)["summary"]
    assert low["n_retained"] == 0
    assert high["n_retained"] == 1
    assert high["frontier_depth"] > low["frontier_depth"]
