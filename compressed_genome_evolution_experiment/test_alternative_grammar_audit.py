"""Tests for exp600 -- Instruction Necessity / Alternative-Grammar Audit."""

from __future__ import annotations

import numpy as np

import alternative_grammar_audit as A
import codec_evo as C


def _world(n_comp: int = 8):
    return C._world_for_ncomp(2, n_comp, 3, 4, np.random.default_rng(1))


def test_all_grammars_enumerate_valid_programs():
    for grammar in A.GRAMMARS:
        assert A.definitions(grammar)


def test_three_independent_grammars_converge_on_concatenation_semantics():
    results = A.convergence_audit(_world(), (
        "postfix_ref_concat", "copy_append", "stack_merge"))
    for result in results.values():
        assert result["born"]
        assert result["total"] < result["fixed"]
        assert result["analysis"].concatenate_equivalent


def test_copy_append_grammar_can_execute_slice_but_does_not_need_it_for_convergence():
    sliced = A.Definition("copy_append", (A.Token("COPY0"), A.Token("SLICE1")))
    assert A.evaluate(sliced, ([1, 0], [0, 1])) == [1]
    result = A.audit_grammar(_world(), "copy_append")
    assert tuple(token.name for token in result["definition"].tokens) == (
        "COPY0", "COPY1", "APPEND")


def test_composition_incapable_grammar_does_not_birth_opcode():
    result = A.audit_grammar(_world(), "copy_repeat_only")
    assert not result["born"]
    assert result["total"] == result["fixed"]
    assert A.threshold("copy_repeat_only", [1, 2, 4, 8]) is None


def test_capable_grammars_share_mdl_birth_threshold():
    for grammar in ("postfix_ref_concat", "copy_append", "stack_merge"):
        assert A.threshold(grammar, [1, 2, 4, 6]) == 2
