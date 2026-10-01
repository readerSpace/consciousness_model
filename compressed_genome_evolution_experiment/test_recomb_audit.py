"""Tests for exp598 -- Recombination Bottleneck Audit (recomb_audit.py).

Validates the decisive instruments: hand-built genomes are exact and F=1, the
compose delta has the right sign structure, factorisers are phenotype-preserving,
and n_recomb counts macros with >=2 distinct children.
Run:  python -m pytest test_recomb_audit.py -q
"""

from __future__ import annotations

import numpy as np

from genome import expand, description_length, Instr, OP_CALL
import recomb_audit as R
from hierarchical import HConfig


def test_handbuilt_genomes_are_exact_F1():
    env, cfg = R.clean_setup(module_reuse=3, seed=1)
    tgt = env.target()
    assert expand(R.genome_flat(env)) == tgt
    assert expand(R.genome_recomb(env)) == tgt
    assert expand(R.genome_literal(tgt, cfg.A)) == tgt


def test_recomb_genome_has_recombination_flat_does_not():
    env, _ = R.clean_setup(module_reuse=3, seed=2)
    assert R.n_recomb(R.genome_recomb(env)) >= 1
    assert R.n_recomb(R.genome_flat(env)) == 0


def test_dl_compose_direct_sign_structure():
    # low reuse: composing costs bits; extreme reuse: it pays
    assert R.dl_compose_direct(2, 3)["dL"] < 0
    assert R.dl_compose_direct(30, 3)["dL"] > 0
    # independent of motif length (composing CALLs, not literals)
    assert R.dl_compose_direct(6, 3)["dL"] == R.dl_compose_direct(6, 8)["dL"]


def test_dl_compose_direct_is_monotone_in_reuse():
    vals = [R.dl_compose_direct(k, 3)["dL"] for k in (2, 6, 10, 20, 30)]
    assert vals == sorted(vals)                     # more reuse -> less negative


def test_factorizers_preserve_phenotype():
    env, cfg = R.clean_setup(module_reuse=4, seed=3)
    tgt = env.target()
    gl = R.genome_literal(tgt, cfg.A)
    assert expand(R.factorize_full(gl, cfg)) == tgt
    assert expand(R.optimal_factorize(gl, cfg)) == tgt


def test_optimal_factorize_never_worse():
    env, cfg = R.clean_setup(module_reuse=6, seed=4)
    gl = R.genome_literal(env.target(), cfg.A)
    assert description_length(R.optimal_factorize(gl, cfg)) <= description_length(gl)


def test_compose_pass_keeps_only_reductions():
    # a genome whose program has a repeated distinct-CALL pair that does NOT pay
    env, cfg = R.clean_setup(module_reuse=2, seed=5)
    gl = R.genome_literal(env.target(), cfg.A)
    before = description_length(R.mdl_factorize(gl, cfg))
    after = description_length(R.optimal_factorize(gl, cfg))
    assert after <= before                          # compose pass never lengthens


def test_apply_pair_macro_creates_recombination():
    env, cfg = R.clean_setup(module_reuse=3, seed=6)
    # a program of [CALL0, CALL1] repeated -> compose into a 2-child macro
    from genome import Genome, OP_LIT
    g = Genome(macros=[[Instr(OP_LIT, 0)], [Instr(OP_LIT, 1)]],
               program=[Instr(OP_CALL, 0), Instr(OP_CALL, 1)] * 3,
               A=2, L=6, nmax=8)
    h = R._apply_pair_macro(g, Instr(OP_CALL, 0), Instr(OP_CALL, 1))
    assert R.n_recomb(h) >= 1
    assert expand(h) == expand(g)


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
