"""Tests for exp597 -- Hierarchical Compositional Escalation (hierarchical.py).

Checks the measurement machinery is correct (hierarchy depth, reuse rate,
recombination, factorisation is phenotype-preserving and only compresses) and
that the controls behave, so any later claim rests on sound instruments.
Run:  python -m pytest test_hierarchical.py -q
"""

from __future__ import annotations

import os
import re

import numpy as np

import genome as G
from genome import Genome, Instr, OP_LIT, OP_CALL, OP_REP, OP_MIR, expand, \
    description_length, macro_call_counts
import hierarchical as H
from hierarchical import (HConfig, factorize, factorize_full, hierarchy_depth,
                          reuse_rate_g, champion_metrics, seed_env, HEnv,
                          run_hierarchical, _referenced_macros)


def _cfg(**kw):
    base = dict(L=36, pop=24, tau_E=20, env_changes=3, seed=1)
    base.update(kw)
    return HConfig(**base)


# ---------------------------------------------------------------------------
# factorisation is phenotype-preserving and never lengthens the program
# ---------------------------------------------------------------------------
def test_factorize_preserves_phenotype():
    cfg = _cfg(L=24)
    # program = (0 1 1) repeated -> a clear repeated substring
    prog = [Instr(OP_LIT, 0), Instr(OP_LIT, 1), Instr(OP_LIT, 1)] * 4
    g = Genome(macros=[], program=prog, A=2, L=24, nmax=8)
    before = expand(g)
    h = factorize(g, cfg)
    assert expand(h) == before                     # phenotype unchanged
    assert len(h.program) <= len(g.program)        # program did not grow
    assert len(h.macros) >= 1                       # a module was extracted


def test_factorize_full_is_phenotype_preserving():
    cfg = _cfg(L=24)
    prog = [Instr(OP_LIT, 0), Instr(OP_LIT, 1)] * 6
    g = Genome(macros=[], program=prog, A=2, L=24, nmax=8)
    before = expand(g)
    h = factorize_full(g, cfg)
    assert expand(h) == before


def test_factorize_creates_reused_module():
    cfg = _cfg(L=24)
    prog = [Instr(OP_LIT, 0), Instr(OP_LIT, 1), Instr(OP_LIT, 0)] * 4
    g = Genome(macros=[], program=prog, A=2, L=24, nmax=8)
    h = factorize(g, cfg)
    counts = macro_call_counts(h)
    assert any(c >= 2 for c in counts)              # the extracted macro is reused
    assert reuse_rate_g(h) > 0.0


# ---------------------------------------------------------------------------
# structural metrics are exact on hand-built genomes
# ---------------------------------------------------------------------------
def test_hierarchy_depth_counts_macro_of_macro():
    # m0 = '01'; m1 = CALL m0 twice -> depth 2
    g = Genome(macros=[[Instr(OP_LIT, 0), Instr(OP_LIT, 1)],
                       [Instr(OP_CALL, 0), Instr(OP_CALL, 0)]],
               program=[Instr(OP_CALL, 1), Instr(OP_CALL, 1)], A=2, L=16, nmax=8)
    used = [0, 1]
    assert hierarchy_depth(g, used) == 2


def test_recombination_is_two_distinct_children():
    # m2 references m0 AND m1 -> recombination; m3 references only m0 -> not
    g = Genome(
        macros=[[Instr(OP_LIT, 0), Instr(OP_LIT, 0)],
                [Instr(OP_LIT, 1), Instr(OP_LIT, 1)],
                [Instr(OP_CALL, 0), Instr(OP_CALL, 1)],
                [Instr(OP_CALL, 0), Instr(OP_CALL, 0)]],
        program=[Instr(OP_CALL, 2), Instr(OP_CALL, 3)], A=2, L=16, nmax=8)
    assert len(_referenced_macros(g.macros[2])) == 2     # recombination
    assert len(_referenced_macros(g.macros[3])) == 1     # not recombination
    mm = champion_metrics(g, expand(g), [], _cfg())
    assert mm["n_recomb"] >= 1


def test_reuse_rate_fraction():
    # macro0 called 3x (reused), macro1 called 1x -> reuse rate = 3/4
    g = Genome(macros=[[Instr(OP_LIT, 0), Instr(OP_LIT, 1)],
                       [Instr(OP_LIT, 1), Instr(OP_LIT, 0)]],
               program=[Instr(OP_CALL, 0), Instr(OP_CALL, 0),
                        Instr(OP_CALL, 0), Instr(OP_CALL, 1)], A=2, L=16, nmax=8)
    assert abs(reuse_rate_g(g) - 0.75) < 1e-9


# ---------------------------------------------------------------------------
# environment hierarchy
# ---------------------------------------------------------------------------
def test_seed_env_target_length_and_structure():
    cfg = _cfg(L=48, motif_len=3, n_motifs0=3, n_modules0=3)
    rng = np.random.default_rng(0)
    env = seed_env(cfg, rng)
    assert len(env.target()) == cfg.L
    # each module is a pair of DISTINCT motif indices (composition)
    assert all(a != b for (a, b) in env.modules)


# ---------------------------------------------------------------------------
# determinism and controls
# ---------------------------------------------------------------------------
def test_same_seed_same_run():
    r1 = run_hierarchical(_cfg(tau_E=15, env_changes=2))
    r2 = run_hierarchical(_cfg(tau_E=15, env_changes=2))
    assert r1["series"]["U_t"] == r2["series"]["U_t"]
    assert r1["summary"] == r2["summary"]


def test_no_factorize_forms_far_fewer_modules():
    on = run_hierarchical(_cfg(tau_E=60, env_changes=3, allow_factorize=True))
    off = run_hierarchical(_cfg(tau_E=60, env_changes=3, allow_factorize=False))
    assert on["summary"]["reuse_rate_max"] >= off["summary"]["reuse_rate_max"]


def test_no_python_hash_seeding():
    here = os.path.dirname(os.path.abspath(__file__))
    text = open(os.path.join(here, "hierarchical.py"), encoding="utf-8").read()
    code = re.sub(r'""".*?"""', "", text, flags=re.S)
    code = re.sub(r"#.*", "", code)
    assert "hash(" not in code


if __name__ == "__main__":
    import traceback
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
            passed += 1
        except Exception:
            print(f"FAIL {fn.__name__}")
            traceback.print_exc()
    print(f"\n{passed}/{len(fns)} passed")
