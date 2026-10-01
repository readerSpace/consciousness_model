"""Focused tests for exp610 -- Unified Program Ecology."""

from __future__ import annotations

import unified_program_ecology as U


def test_same_program_representation_supports_data_and_program_application():
    child = U.transform(U.R1, U.Q0, U.Q1)
    assert child == U.crossover(U.Q0, U.Q1)
    assert U.semantic_signature(child) == (-5, -2, 1, 4, 7)
    assert U.apply_data(child, 2) == 7


def test_generic_mutation_derives_second_transformer_without_meta_type():
    assert U.R1 == U.crossover(U.ARG0, U.ARG1)
    assert U.R2 == U.mutate(U.R1)
    assert U.transformer_signature(U.R1) != U.transformer_signature(U.R2)


def test_unified_reproduces_semantic_growth_and_births_late_transformer():
    result = U.run_ecology(U.UNIFIED, U.Config(100), 1)
    summary = result["summary"]
    assert summary["n_functional_semantic_classes"] == 100
    assert summary["late_functional_semantic_classes"] > 0
    assert summary["late_functional_transformer_classes"] == 1
    assert summary["max_ancestry_depth"] == 100
    assert any(item["functional"] and item["birth"] >= 50 for item in result["transformers"].values())


def test_transformer_knockout_and_specialized_control():
    config = U.Config(100)
    specialized = U.run_ecology(U.SPECIALIZED, config, 1)["summary"]
    unified = U.run_ecology(U.UNIFIED, config, 1)
    assert specialized["n_functional_semantic_classes"] == unified["summary"]["n_functional_semantic_classes"]
    assert specialized["n_functional_transformer_classes"] == 0
    assert all(item["knockout_delta"] > 0 for item in unified["transformers"].values())