"""Tests for exp602 -- Cumulative Instruction Bootstrapping."""

from __future__ import annotations

import cumulative_instruction_bootstrapping as B


N_TUPLES = [1, 2, 3, 4, 6, 8, 12]


def test_o1_is_recovered_from_the_exp601_distant_basis():
    cost, tokens = B.discover_o1()
    assert cost == 14
    assert tokens == ("REF0", "BOX", "REF1", "PAIR", "FLATTEN")


def test_useful_history_lowers_four_argument_semantic_complexity():
    base_cost, base = B.semantic_complexity(4, B.BASE)
    useful_cost, useful = B.semantic_complexity(4, B.USEFUL_O1)
    irrelevant_cost, irrelevant = B.semantic_complexity(4, B.IRRELEVANT)
    args = ([0], [1], [1, 0], [0, 1])
    expected = [0, 1, 1, 0, 0, 1]
    assert base_cost == irrelevant_cost == 36
    assert useful_cost == 18 < base_cost
    assert B.evaluate(base, args) == B.evaluate(useful, args) == B.evaluate(irrelevant, args) == expected


def test_irrelevant_history_has_equal_cost_but_cannot_help_concatenation():
    assert B.USEFUL_O1.retained_bits == B.IRRELEVANT.retained_bits == 14
    duplicate = B.Expr("O_IRRELEVANT_DUP", (B.Expr("REF", index=0),))
    assert B.evaluate(duplicate, ([1, 0], [0, 1])) == [1, 0, 1, 0]


def test_useful_history_accelerates_o2_birth_against_both_controls():
    base = B.birth_threshold(4, B.BASE, N_TUPLES)
    useful = B.birth_threshold(4, B.USEFUL_O1, N_TUPLES)
    irrelevant = B.birth_threshold(4, B.IRRELEVANT, N_TUPLES)
    assert useful == 2
    assert base == irrelevant == 3


def test_cumulative_history_lowers_later_abstraction_costs():
    trace = B.cumulative_trace()
    assert [row["K_P0"] for row in trace] == [14, 36, 80]
    assert [row["K_history"] for row in trace] == [14, 18, 30]
    assert [row["ratio"] for row in trace] == [1.0, 0.5, 0.375]
    assert B.birth_threshold(8, B.BASE, N_TUPLES) == 6
    assert B.birth_threshold(8, B.USEFUL_O1_O2, N_TUPLES) == 2


def test_o2_genome_is_selected_only_when_total_description_drops():
    useful = B.audit_birth(4, B.USEFUL_O1, 2)
    base = B.audit_birth(4, B.BASE, 2)
    assert useful["born"] and useful["total"] < useful["fixed"]
    assert not base["born"] and base["total"] == base["fixed"]
