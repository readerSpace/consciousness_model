"""Tests for exp599a -- Codec Evolution (codec_evo.py).

Validates the codec (expand + language-aware cost), the COMPOSE saving, the
three-world selection, and that evolution's language adoption tracks the exact
MDL crossover.
Run:  python -m pytest test_codec_evo.py -q
"""

from __future__ import annotations

import numpy as np

import codec_evo as C
from codec_evo import Ins, Prog, OP_LIT, OP_CALL, OP_REP, OP_COMPOSE


def test_compose_expands_like_two_calls():
    g_calls = Prog([[Ins(OP_LIT, 0)], [Ins(OP_LIT, 1)]],
                   [Ins(OP_CALL, 0), Ins(OP_CALL, 1)], 2, 2)
    g_comp = Prog([[Ins(OP_LIT, 0)], [Ins(OP_LIT, 1)]],
                  [Ins(OP_COMPOSE, 0, 1)], 2, 2)
    assert C.expand(g_calls) == C.expand(g_comp) == [0, 1]


def test_compose_saves_one_opcode():
    g_calls = Prog([[Ins(OP_LIT, 0)], [Ins(OP_LIT, 1)]],
                   [Ins(OP_CALL, 0), Ins(OP_CALL, 1)], 2, 2)
    g_comp = Prog([[Ins(OP_LIT, 0)], [Ins(OP_LIT, 1)]],
                  [Ins(OP_COMPOSE, 0, 1)], 2, 2)
    # program section: COMPOSE saves exactly OPCODE_BITS vs two CALLs
    b_calls = C._list_bits(g_calls.program, 2)
    b_comp = C._list_bits(g_comp.program, 2)
    assert b_calls - b_comp == C.OPCODE_BITS


def test_D0_cannot_use_compose():
    g = Prog([[Ins(OP_LIT, 0)], [Ins(OP_LIT, 1)]], [Ins(OP_COMPOSE, 0, 1)], 2, 2)
    assert C.description_length(g, C.LANG_D0, 48) >= 10 ** 9     # illegal under D0
    assert C.description_length(g, C.LANG_D1, 48) < 10 ** 9


def test_language_cost_charges_D1():
    g = Prog([], [Ins(OP_LIT, 0)], 2, 1)
    assert (C.description_length(g, C.LANG_D1, 48)
            - C.description_length(g, C.LANG_D0, 48)) == 48


def test_handbuilt_compositional_genomes_are_F1():
    rng = np.random.default_rng(1)
    w = C._world_for_ncomp(2, 12, 3, 4, rng)
    for g in (C.g_D0_inline(w), C.g_D0_pairmacros(w), C.g_flat_pairmacros(w),
              C.g_D1_compose_inline(w), C.g_D1_compose_macros(w)):
        assert C.expand(g) == w.target


def test_three_world_selection():
    rng = np.random.default_rng(3)
    flat = C.flat_world(2, 96, 2, rng)
    comp = C._world_for_ncomp(2, 20, 3, 4, rng)
    rand = C.random_world(2, 96, rng)
    assert C.selected_language(flat, 48)["selected"] == C.LANG_D0
    assert C.selected_language(comp, 48)["selected"] == C.LANG_D1
    assert C.selected_language(rand, 48)["selected"] == C.LANG_D0


def test_mdl_crossover_exists_and_monotone_in_opcode_price():
    ncs = [2, 4, 6, 8, 12, 16, 20, 30]
    cheap = C.mdl_crossover(2, ncs, 3, 4, 24, seed=1)["n_comp_star_MDL"]
    dear = C.mdl_crossover(2, ncs, 3, 4, 72, seed=1)["n_comp_star_MDL"]
    assert cheap is not None and dear is not None
    assert cheap <= dear                            # costlier opcode -> later adoption


def test_evolution_tracks_mdl_crossover():
    ncs = [2, 4, 6, 8, 12, 16, 20, 30]
    nstar = C.mdl_crossover(2, ncs, 3, 4, 48, seed=1)["n_comp_star_MDL"]
    nevo = C.k_evo_sweep(2, ncs, 3, 4, 48, [1, 2, 3])["n_comp_evo"]
    assert nevo is not None
    assert abs(nevo - nstar) <= 4                    # evolution adopts near the MDL point


def test_evolution_preserves_phenotype():
    rng = np.random.default_rng(5)
    w = C._world_for_ncomp(2, 20, 3, 4, rng)
    res = C.evolve_language(w, 48, seed=11)
    assert res["champion_lang"] in (C.LANG_D0, C.LANG_D1)


if __name__ == "__main__":
    import traceback
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        try:
            fn(); print(f"PASS {fn.__name__}"); passed += 1
        except Exception:
            print(f"FAIL {fn.__name__}"); traceback.print_exc()
    print(f"\n{passed}/{len(fns)} passed")
