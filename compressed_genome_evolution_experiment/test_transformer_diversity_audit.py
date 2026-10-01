"""Focused tests for exp611 -- Transformer Diversity / Self-Modification Audit."""

from __future__ import annotations

import transformer_diversity_audit as E


def test_transformer_classes_are_program_probe_behavior_not_raw_program_identity():
    first = E.construct_transformer((1, 1))
    alias = E.U.crossover(E.U.ARG0, E.U.ARG1)
    different = E.construct_transformer((1, 2))
    assert E.transformer_signature(first) == E.transformer_signature(alias)
    assert E.transformer_signature(first) != E.transformer_signature(different)


def test_generic_crossover_and_mutation_construct_first_two_demands():
    assert E.construct_transformer((1, 1)) == E.U.crossover(E.U.ARG0, E.U.ARG1)
    assert E.construct_transformer((1, 2)) == E.U.mutate(E.construct_transformer((1, 1)))


def test_functional_transformer_diversity_increases_with_horizon():
    counts = [E.run_audit(E.UNIFIED_ADAPTIVE, E.Config(steps), 1)["summary"]
              ["n_functional_transformer_classes"] for steps in (100, 300, 1000)]
    assert counts[0] < counts[1] < counts[2]
    result = E.run_audit(E.UNIFIED_ADAPTIVE, E.Config(100), 1)
    assert result["summary"]["late_functional_transformer_classes"] > 0


def test_functional_transformers_have_mdl_fitness_reuse_and_knockout_evidence():
    result = E.run_audit(E.UNIFIED_ADAPTIVE, E.Config(100), 1)
    functional = [item for item in result["transformers"].values() if item["functional"]]
    assert functional
    assert all(item["reuse"] >= 2 and item["knockout_delta"] > 0 for item in functional)
    assert all(item["mean_child_delta_L"] > 0 and item["mean_child_delta_F"] > 0 for item in functional)


def test_self_application_retention_fixed_and_random_controls_do_not_count_transformers():
    config = E.Config(100)
    adaptive = E.run_audit(E.UNIFIED_ADAPTIVE, config, 1)["summary"]
    controls = {condition: E.run_audit(condition, config, 1)["summary"] for condition in E.CONDITIONS[1:]}
    assert adaptive["n_functional_transformer_classes"] > 1
    assert all(summary["n_functional_transformer_classes"] == 0 for summary in controls.values())
    assert controls[E.RANDOM_REWRITE]["n_raw_programs"] == adaptive["n_raw_programs"]
    assert controls[E.RANDOM_REWRITE]["max_ancestry_depth"] == 0