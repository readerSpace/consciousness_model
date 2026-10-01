"""Tests for exp604 -- Endogenous Abstraction Ecology."""

from __future__ import annotations

import endogenous_abstraction_ecology as E


def _runs():
    return E.run_conditions(steps=30, seed=1)


def test_ecology_is_deterministic_and_records_dynamic_task_selection():
    first = E.run_ecology("adaptive_language", 30, 1)
    second = E.run_ecology("adaptive_language", 30, 1)
    assert first["records"] == second["records"]
    assert {record["arity"] for record in first["records"]} == {2, 4, 8}
    assert any(record["offers_birth"] for record in first["records"])
    for record in first["records"]:
        if record["arity"] == 4:
            assert 2 in record["basis_before"]
        if record["arity"] == 8:
            assert 4 in record["basis_before"]


def test_adaptive_language_has_late_birth_reuse_retention_and_pruning():
    summary = _runs()["adaptive_language"]["summary"]
    assert summary["late_births"] > 0
    assert summary["late_reuse"] > 0
    assert summary["n_pruned"] > 0
    assert summary["n_retained_final"] > 0
    assert summary["mean_A_late"] > 0
    assert summary["max_D_language"] < E.CAPACITY


def test_fixed_and_no_retention_controls_break_cumulative_advantage():
    runs = _runs()
    adaptive = runs["adaptive_language"]["summary"]
    fixed = runs["fixed_language"]["summary"]
    no_retention = runs["no_retention"]["summary"]
    assert fixed["n_birth"] == fixed["late_reuse"] == 0
    assert fixed["mean_A_late"] == 0
    assert no_retention["n_retained_final"] == 0
    assert adaptive["mean_A_late"] > no_retention["mean_A_late"] > fixed["mean_A_late"]


def test_keep_all_accumulates_bloat_that_adaptive_forgetting_avoids():
    runs = _runs()
    adaptive = runs["adaptive_language"]["summary"]
    keep_all = runs["keep_all"]["summary"]
    assert keep_all["n_pruned"] == 0
    assert keep_all["final_D_language"] > adaptive["final_D_language"]
    assert adaptive["mean_description"] < keep_all["mean_description"]


def test_life_history_contains_birth_use_reuse_and_prune_events():
    lives = _runs()["adaptive_language"]["life_history"]
    assert any(life["birth"] for life in lives.values())
    assert any(life["reuse"] > 0 for life in lives.values())
    assert any(life["prune"] for life in lives.values())
