"""Tests for exp601 -- Primitive Basis Ablation / Computational Necessity."""

from __future__ import annotations

import numpy as np

import codec_evo as C
import primitive_basis_ablation as P


N_COMPS = [1, 2, 3, 4, 6, 8]


def _world(n_comp: int = 8):
    return C._world_for_ncomp(2, n_comp, 3, 4, np.random.default_rng(1))


def test_semantic_complexity_identifies_full_and_distant_witnesses():
    full_cost, full = P.semantic_complexity("full")
    distant_cost, distant = P.semantic_complexity("distant_box_pair_flatten")
    assert full_cost == 8
    assert distant_cost == 14
    assert P.is_concatenation(full) and P.is_concatenation(distant)
    assert len(distant.tokens) > len(full.tokens)


def test_ref_order_and_merge_ablations_make_concatenation_inexpressible():
    for basis in ("minus_ref", "minus_order", "minus_merge"):
        cost, witness = P.semantic_complexity(basis)
        assert cost is None and witness is None
        result = P.audit_basis(_world(), basis)
        assert not result["expressible"] and not result["born"]


def test_no_order_basis_can_only_merge_in_the_wrong_order():
    reverse = P.Definition("minus_order", ("ARG0_THEN_ARG1", "REVERSE_MERGE"))
    assert P.evaluate(reverse, ([1, 0], [0, 1])) == [0, 1, 1, 0]
    assert not P.is_concatenation(reverse)


def test_reuse_is_required_even_when_concatenation_is_expressible():
    result = P.audit_basis(_world(), "minus_reuse")
    assert result["expressible"]
    assert result["K_concat"] == 8
    assert not result["born"]
    assert result["total"] == result["fixed"]


def test_larger_semantic_complexity_requires_later_birth():
    assert P.birth_threshold("full", N_COMPS) == 2
    assert P.birth_threshold("distant_box_pair_flatten", N_COMPS) == 3


def test_no_instruction_birth_in_flat_and_random_controls():
    rng = np.random.default_rng(9)
    for world in (C.flat_world(2, 96, 2, rng), C.random_world(2, 96, rng)):
        for basis in ("full", "distant_box_pair_flatten"):
            assert not P.audit_basis(world, basis)["born"]
