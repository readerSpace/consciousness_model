"""Tests for exp596 -- Endogenous Complexity Escalation (escalation.py).

Discipline mirrors the rest of the project:
  * determinism (same seed -> same result; no Python hash() seeding)
  * the module/novelty/reuse/recombination accounting is exact on hand-built
    genomes (not just "looks plausible")
  * C_t = L*(F>=f|E) is a real bound: never above the environment's own genome
    length, and constant when the environment is frozen
  * the control flags actually change behaviour (frozen freezes; no-recomb never
    produces a recombined module)
Run:  python -m pytest test_escalation.py -q
      (or: python test_escalation.py)
"""

from __future__ import annotations

import os
import re

import numpy as np

import genome as G
from genome import (Genome, Instr, OP_LIT, OP_CALL, OP_REP, OP_MIR,
                    expand, description_length)
import escalation as E
from escalation import (CoevoConfig, run_coevolution, module_metrics, cstar,
                        _match, _is_recombined, _knockout_fitness, _seed_env)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _small_cfg(**kw):
    base = dict(A=2, L=16, pop=24, inner_gens=4, macro_steps=8,
                n_env_cands=3, probe=3, seed=1)
    base.update(kw)
    return CoevoConfig(**base)


# ---------------------------------------------------------------------------
# determinism
# ---------------------------------------------------------------------------
def test_same_seed_same_run():
    r1 = run_coevolution(_small_cfg(seed=7))
    r2 = run_coevolution(_small_cfg(seed=7))
    assert r1["series"]["C_t"] == r2["series"]["C_t"]
    assert r1["series"]["F"] == r2["series"]["F"]
    assert r1["summary"] == r2["summary"]


def test_no_python_hash_seeding():
    here = os.path.dirname(os.path.abspath(__file__))
    text = open(os.path.join(here, "escalation.py"), encoding="utf-8").read()
    code = re.sub(r'""".*?"""', "", text, flags=re.S)
    code = re.sub(r"#.*", "", code)
    assert "hash(" not in code


# ---------------------------------------------------------------------------
# C_t = L*(F>=f | E) is a real, well-behaved bound
# ---------------------------------------------------------------------------
def test_cstar_not_above_env_genome_length():
    """The environment's OWN genome solves it (F=1), so the minimal description
    achieving F>=f can never exceed L(E)."""
    cfg = _small_cfg()
    rng = np.random.default_rng(0)
    from evolution import random_structured_program
    from escalation import _gacfg
    gacfg = _gacfg(cfg)
    for _ in range(20):
        env = random_structured_program(rng, gacfg)
        target = expand(env)
        assert cstar(target, cfg) <= description_length(env)


def test_cstar_constant_when_target_constant():
    cfg = _small_cfg()
    target = [0, 1] * (cfg.L // 2)
    a = cstar(target, cfg)
    b = cstar(list(target), cfg)
    assert a == b
    # C_t is a BIT length; a period-2 tiling compresses well below a literal
    # genome (gamma(1)+gamma(L+1)+L*(2+1) bits for L=16 = 58 bits)
    assert a < 58


def test_cstar_noise_costs_more_than_structure():
    cfg = _small_cfg()
    struct = [0, 1] * (cfg.L // 2)
    rng = np.random.default_rng(3)
    noise = [int(b) for b in rng.integers(0, 2, size=cfg.L)]
    assert cstar(struct, cfg) <= cstar(noise, cfg)


# ---------------------------------------------------------------------------
# module accounting is exact on hand-built genomes
# ---------------------------------------------------------------------------
def test_reuse_counts_macro_called_twice():
    # macro0 = '01'; program calls it twice + pads -> reused module
    g = Genome(macros=[[Instr(OP_LIT, 0), Instr(OP_LIT, 1)]],
               program=[Instr(OP_CALL, 0), Instr(OP_CALL, 0)],
               A=2, L=16, nmax=8)
    cfg = _small_cfg()
    target = expand(g)
    mm = module_metrics(g, target, [], cfg)
    assert mm["N_modules"] == 1
    assert mm["N_reused"] == 1
    assert mm["N_recombined"] == 0


def test_recombined_module_detected():
    # macro2 references macro0 AND macro1 -> recombination
    g = Genome(
        macros=[[Instr(OP_LIT, 0), Instr(OP_LIT, 0)],
                [Instr(OP_LIT, 1), Instr(OP_LIT, 1)],
                [Instr(OP_CALL, 0), Instr(OP_CALL, 1)]],
        program=[Instr(OP_CALL, 2), Instr(OP_CALL, 2)],
        A=2, L=16, nmax=8)
    assert _is_recombined(g, 2)
    assert not _is_recombined(g, 0)
    cfg = _small_cfg()
    mm = module_metrics(g, expand(g), [], cfg)
    assert mm["N_recombined"] >= 1


def test_knockout_drops_fitness_for_loadbearing_module():
    # macro0 supplies the '01' pattern; target is that pattern -> knockout hurts
    g = Genome(macros=[[Instr(OP_LIT, 0), Instr(OP_LIT, 1)]],
               program=[Instr(OP_REP, 8, 0)], A=2, L=16, nmax=8)
    target = expand(g)
    base = _match(expand(g), target)
    ko = _knockout_fitness(g, 0, target)
    assert base == 1.0
    assert ko < base


def test_novelty_requires_distance_and_knockout():
    cfg = _small_cfg(novelty_eps=0.25, knockout_drop=0.02)
    g = Genome(macros=[[Instr(OP_LIT, 0), Instr(OP_LIT, 1)]],
               program=[Instr(OP_REP, 8, 0)], A=2, L=16, nmax=8)
    target = expand(g)
    # empty history -> the load-bearing module is novel
    mm0 = module_metrics(g, target, [], cfg)
    assert mm0["N_novel"] == 1
    # if the SAME module (its '01' expansion) is already in history, not novel
    sig = (0, 1)
    mm1 = module_metrics(g, target, [sig], cfg)
    assert mm1["N_novel"] == 0


# ---------------------------------------------------------------------------
# control flags actually change behaviour
# ---------------------------------------------------------------------------
def test_frozen_environment_keeps_C_constant():
    r = run_coevolution(_small_cfg(freeze_env=True, macro_steps=10))
    C = r["series"]["C_t"]
    assert all(c == C[0] for c in C)          # target never changes -> C constant


def test_no_recomb_never_produces_recombined_module():
    r = run_coevolution(_small_cfg(allow_recomb=False, macro_steps=12))
    assert all(x == 0 for x in r["series"]["N_recombined"])


def test_capacity_penalty_makes_over_budget_lethal():
    # an individual whose description exceeds K_max must score J well below F
    from escalation import _eval_ind, Individual
    cfg = _small_cfg(K_max=8)
    big = Genome(macros=[],
                 program=[Instr(OP_LIT, 0)] * cfg.L, A=2, L=cfg.L, nmax=8)
    ind = Individual(g=big)
    target = [0] * cfg.L
    _eval_ind(ind, target, cfg)
    assert description_length(big) > cfg.K_max
    assert ind.J < ind.F                      # capacity penalty applied


def test_seed_env_is_valid_and_length_L():
    cfg = _small_cfg()
    env = _seed_env(cfg)
    G.validate(env)
    assert len(expand(env)) == cfg.L


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
