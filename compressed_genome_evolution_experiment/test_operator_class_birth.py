"""Focused tests for exp608 -- Operator-Class Birth."""

from __future__ import annotations

import operator_class_birth as O


def test_probe_class_is_behavioral_not_identifier():
    first = O.Operator(4, 3, 2)
    alias = O.Operator(99, 3, 2)
    different = O.Operator(4, 3, 3)
    assert first.probe_signature() == alias.probe_signature()
    assert O.operator_distance(first, alias) == 0.0
    assert O.operator_distance(first, different) > 0.0


def test_generative_adaptive_has_late_confirmed_functional_births():
    summary = O.run_birth(O.GENERATIVE_ENV, O.ADAPTIVE_LANGUAGE, O.Config(100), 1)["summary"]
    assert summary["late_functional_classes"] > 0
    assert summary["n_functional_classes"] > 1
    assert summary["capacity_fraction"] < 0.95


def test_fixed_environment_and_fixed_language_do_not_create_functional_classes():
    config = O.Config(100)
    fixed_environment = O.run_birth(O.FIXED_ENV, O.ADAPTIVE_LANGUAGE, config, 1)["summary"]
    fixed_language = O.run_birth(O.GENERATIVE_ENV, O.FIXED_LANGUAGE, config, 1)["summary"]
    assert fixed_environment["late_functional_classes"] == 0
    assert fixed_language["n_functional_classes"] == 0
    assert fixed_language["frontier_operator"] < 10


def test_confirmed_classes_have_recursive_lineage_reuse_and_knockout_effect():
    result = O.run_birth(O.GENERATIVE_ENV, O.ADAPTIVE_LANGUAGE, O.Config(30), 1)
    lineage = result["lineage"]
    assert lineage[3]["parents"] == [1, 2]
    assert lineage[4]["parents"] == [3, 1]
    assert lineage[3]["reuse"] >= 2
    assert lineage[3]["load_bearing"]
    assert lineage[3]["functional"]