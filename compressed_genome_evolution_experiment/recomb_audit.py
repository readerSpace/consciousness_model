"""exp598 -- Recombination Bottleneck Audit (engine).

exp597 reached 5/6 strict compositional criteria; the SOLE failure was
recombination (a macro composing two DISTINCT existing modules, M_AB=(M_A,M_B)).
exp598 isolates WHY M_A+M_B -> M_AB is not selected, WITHOUT adding new world
mechanics, and WITHOUT the tau_E sweep (exp597 found no tau_E phase transition).

The decisive quantity is the direct MDL change of composing two modules:

    dL_compose = L(G_without_module_macros) - L(G_with_module_macros)

computed EXACTLY in the exp588 codec on hand-built genomes that both reproduce
the same target at F=1.  dL_compose > 0 means recombination pays bits (so any
failure is a SEARCH problem); dL_compose <= 0 means the current codec gives no
selective benefit to recombination at all (an EXPRESSIVITY/MDL problem).

Four conditions cross match-quality (F=1?) with the factoriser (greedy vs a
composition-aware one):

    evolved-imperfect   normal exp597 GA            (F~0.74, greedy)
    oracle-phenotype    F=1 literal genome, greedy  (is match quality enough?)
    oracle-factorization evolved phenotype, optimal  (is greedy the blocker?)
    oracle-both         F=1 + optimal factoriser    (achievability ceiling)

decision table:
    F=1 succeeds & optimal succeeds -> match quality was the bottleneck
    F=1 fails    & optimal succeeds -> the greedy factoriser was the bottleneck
    both fail (dL_compose<=0)       -> the codec/MDL does not reward recombination

Plus a propose/survive/fix decomposition of recombination events during normal
evolution (can't generate / dies in MDL / transiently wins but never fixes).

All randomness flows through explicit numpy Generators.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from statistics import mean
from typing import Dict, List, Optional, Tuple

import numpy as np

import genome as G
from genome import (Genome, Instr, OP_LIT, OP_CALL, OP_REP, OP_MIR,
                    expand, validate, description_length, macro_call_counts)
from hierarchical import (HConfig, HEnv, _gacfg, _one_generation, _reeval_pop,
                          _eval, _mutate, factorize_full, champion_metrics,
                          reuse_rate_g, _referenced_macros, Individual,
                          random_structured_program, _match)


# ---------------------------------------------------------------------------
# a CLEAN hierarchical environment: each module repeats `module_reuse` times,
# target = one exact tile (no padding) so hand-built genomes hit F=1 exactly
# ---------------------------------------------------------------------------
def make_clean_env(cfg: HConfig, rng: np.random.Generator,
                   module_reuse: int = 2) -> HEnv:
    motifs = [rng.integers(0, cfg.A, size=cfg.motif_len).tolist()
              for _ in range(cfg.n_motifs0)]
    modules = [(i % cfg.n_motifs0, (i + 2) % cfg.n_motifs0)
               for i in range(cfg.n_modules0)]
    order: List[int] = []
    for mi in range(cfg.n_modules0):
        order += [mi] * module_reuse                # each module repeated k times
    base_len = len(order) * 2 * cfg.motif_len
    env = HEnv(motifs, modules, order, cfg.A, base_len)   # L = one exact tile
    return env


def _distinct_modules(env: HEnv) -> List[int]:
    seen = []
    for mi in env.order:
        if mi not in seen:
            seen.append(mi)
    return seen


def clean_setup(module_reuse: int, seed: int = 0, motif_len: int = 3,
                n_motifs0: int = 3, n_modules0: int = 3, pop: int = 80,
                **kw) -> Tuple[HEnv, HConfig]:
    """A clean env (one exact tile) with an HConfig whose L matches the tile, so
    hand-built genomes and the GA all operate at the same phenotype length."""
    cfg0 = HConfig(motif_len=motif_len, n_motifs0=n_motifs0,
                   n_modules0=n_modules0, pop=pop, seed=seed, **kw)
    rng = np.random.default_rng(seed)
    env = make_clean_env(cfg0, rng, module_reuse=module_reuse)
    cfg = HConfig(motif_len=motif_len, n_motifs0=n_motifs0,
                  n_modules0=n_modules0, pop=pop, seed=seed, L=env.L, **kw)
    return env, cfg


# ---------------------------------------------------------------------------
# hand-built genomes that both reproduce the target at F=1
# ---------------------------------------------------------------------------
def genome_flat(env: HEnv) -> Genome:
    """Motif macros + a program that inlines motif-calls per module.  Reuse of
    motifs, but NO module macro -> NO recombination (depth 1)."""
    motif_macros = [[Instr(OP_LIT, s) for s in m] for m in env.motifs]
    program: List[Instr] = []
    for mi in env.order:
        a, b = env.modules[mi]
        program += [Instr(OP_CALL, a), Instr(OP_CALL, b)]
    g = Genome(macros=[list(m) for m in motif_macros], program=program,
               A=env.A, L=env.L, nmax=8)
    validate(g)
    return g


def genome_recomb(env: HEnv) -> Genome:
    """Motif macros + one module macro per DISTINCT module (M_AB=(CALL a,CALL b),
    referencing two distinct motif macros = recombination) + a program of
    module-calls.  Same phenotype, depth 2."""
    M = len(env.motifs)
    motif_macros = [[Instr(OP_LIT, s) for s in m] for m in env.motifs]
    distinct = _distinct_modules(env)
    module_macro_index = {}
    module_macros = []
    for k, mi in enumerate(distinct):
        a, b = env.modules[mi]
        module_macros.append([Instr(OP_CALL, a), Instr(OP_CALL, b)])
        module_macro_index[mi] = M + k
    program = [Instr(OP_CALL, module_macro_index[mi]) for mi in env.order]
    g = Genome(macros=[list(m) for m in motif_macros] + module_macros,
               program=program, A=env.A, L=env.L, nmax=8)
    validate(g)
    return g


# ---------------------------------------------------------------------------
# the decisive quantity: dL_compose (exact, same phenotype)
# ---------------------------------------------------------------------------
def compose_delta(env: HEnv) -> dict:
    gf = genome_flat(env)
    gr = genome_recomb(env)
    tgt = env.target()
    assert expand(gf) == tgt and expand(gr) == tgt          # both F=1
    Lf = description_length(gf)
    Lr = description_length(gr)
    n_r = sum(1 for i in range(len(gr.macros))
              if len(_referenced_macros(gr.macros[i])) >= 2)
    return {
        "L_flat": Lf,
        "L_recomb": Lr,
        "dL_compose": Lf - Lr,                               # >0 => recomb pays
        "recomb_favoured": bool(Lf - Lr > 0),
        "n_recomb_in_recomb_genome": n_r,
        "module_reuse": len(env.order) // max(1, len(_distinct_modules(env))),
    }


def dl_compose_direct(k: int, motif_len: int, A: int = 2, seed: int = 0) -> dict:
    """The PUREST per-composition delta: two motif macros A, B already exist and
    the pair [CALL A, CALL B] occurs k times (scattered among a filler call);
    compose them into M_AB=[CALL A, CALL B].  dL = L(before) - L(after) > 0 iff
    recombination pays.  Isolates the codec's CALL-indirection cost from matching
    and from motif content (dL is independent of motif_len)."""
    rng = np.random.default_rng(seed)
    mA = [Instr(OP_LIT, int(x)) for x in rng.integers(0, A, motif_len)]
    mB = [Instr(OP_LIT, int(x)) for x in rng.integers(0, A, motif_len)]
    mC = [Instr(OP_LIT, int(x)) for x in rng.integers(0, A, motif_len)]
    prog: List[Instr] = []
    for _ in range(k):
        prog += [Instr(OP_CALL, 0), Instr(OP_CALL, 1), Instr(OP_CALL, 2)]
    probe = Genome(macros=[mA, mB, mC], program=prog, A=A, L=10_000, nmax=8)
    L = len(expand(probe))
    g = Genome(macros=[mA, mB, mC], program=prog, A=A, L=L, nmax=8)
    validate(g)
    before = description_length(g)
    h = _apply_pair_macro(g, Instr(OP_CALL, 0), Instr(OP_CALL, 1))
    after = description_length(h)
    return {"k": k, "dL": before - after, "favoured": bool(before - after > 0),
            "n_recomb_after": n_recomb(h)}


# ---------------------------------------------------------------------------
# composition-aware "optimal" factoriser: greedy motif Re-Pair, THEN explicitly
# compose repeated adjacent pairs of DISTINCT macro-calls into module macros
# whenever that lowers the description length
# ---------------------------------------------------------------------------
def _apply_pair_macro(g: Genome, a: Instr, b: Instr) -> Genome:
    h = g.copy()
    new_idx = len(h.macros)
    h.macros.append([a, b])
    prog = h.program
    newprog: List[Instr] = []
    i = 0
    while i < len(prog):
        if i < len(prog) - 1 and prog[i] == a and prog[i + 1] == b:
            newprog.append(Instr(OP_CALL, new_idx))
            i += 2
        else:
            newprog.append(prog[i])
            i += 1
    h.program = newprog
    try:
        validate(h)
    except Exception:
        return g
    return h


def _apply_substring_macro(g: Genome, key: tuple) -> Genome:
    h = g.copy()
    new_idx = len(h.macros)
    h.macros.append(list(key))
    prog = h.program
    klen = len(key)
    newprog: List[Instr] = []
    i = 0
    while i < len(prog):
        if tuple(prog[i:i + klen]) == key:
            newprog.append(Instr(OP_CALL, new_idx))
            i += klen
        else:
            newprog.append(prog[i])
            i += 1
    h.program = newprog
    try:
        validate(h)
    except Exception:
        return g
    return h


def mdl_factorize(g: Genome, cfg: HConfig) -> Genome:
    """MDL-OPTIMAL greedy: at each step extract the repeated substring that gives
    the greatest decrease in description length; stop when none decreases it."""
    h = g
    while len(h.macros) < cfg.max_macros:
        prog = h.program
        n = len(prog)
        best = None
        best_L = description_length(h)
        maxlen = min(cfg.max_macro_len, n // 2)
        for length in range(2, maxlen + 1):
            occ: Dict[tuple, List[int]] = {}
            for s in range(0, n - length + 1):
                occ.setdefault(tuple(prog[s:s + length]), []).append(s)
            for key, positions in occ.items():
                cnt = 0
                last_end = -1
                for p in positions:
                    if p >= last_end:
                        cnt += 1
                        last_end = p + length
                if cnt >= 2:
                    cand = _apply_substring_macro(h, key)
                    Lc = description_length(cand)
                    if Lc < best_L:
                        best_L = Lc
                        best = cand
        if best is None:
            break
        h = best
    return h


def _compose_pass(g: Genome, cfg: HConfig) -> Genome:
    h = g
    while len(h.macros) < cfg.max_macros:
        prog = h.program
        cnt: Counter = Counter()
        for i in range(len(prog) - 1):
            a, b = prog[i], prog[i + 1]
            if a.op == OP_CALL and b.op == OP_CALL and a.a != b.a:
                cnt[(a, b)] += 1
        best = None
        best_L = description_length(h)
        for (a, b), c in cnt.items():
            if c >= 2:
                cand = _apply_pair_macro(h, a, b)
                Lc = description_length(cand)
                if Lc < best_L:
                    best_L = Lc
                    best = cand
        if best is None:
            break
        h = best
    return h


def optimal_factorize(g: Genome, cfg: HConfig) -> Genome:
    """Composition-aware MDL factoriser: extract motif macros (MDL-optimal greedy),
    then compose repeated distinct-macro pairs into module macros wherever that
    lowers L.  Phenotype-preserving and never worse than the input."""
    h = mdl_factorize(g, cfg)
    h = _compose_pass(h, cfg)
    return h if description_length(h) <= description_length(g) else g


def n_recomb(g: Genome) -> int:
    counts = macro_call_counts(g)
    return sum(1 for i in range(len(g.macros))
              if counts[i] >= 1 and len(_referenced_macros(g.macros[i])) >= 2)


# ---------------------------------------------------------------------------
# F=1 literal genome of a target (oracle phenotype, no structure)
# ---------------------------------------------------------------------------
def genome_literal(target: List[int], A: int, nmax: int = 8) -> Genome:
    g = Genome(macros=[], program=[Instr(OP_LIT, s) for s in target],
               A=A, L=len(target), nmax=nmax)
    validate(g)
    return g


# ---------------------------------------------------------------------------
# evolve to a champion phenotype (normal exp597 GA on the clean env)
# ---------------------------------------------------------------------------
def evolve_champion(env: HEnv, cfg: HConfig, gens: int,
                    rng: np.random.Generator) -> Individual:
    gacfg = _gacfg(cfg)
    target = env.target()
    pop = [Individual(g=random_structured_program(rng, gacfg))
           for _ in range(cfg.pop)]
    _reeval_pop(pop, target, cfg)
    for _ in range(gens):
        pop = _one_generation(pop, target, cfg, gacfg, rng)
    return max(pop, key=lambda i: i.J)


# ---------------------------------------------------------------------------
# propose / survive / fix decomposition of recombination during evolution
# ---------------------------------------------------------------------------
def recomb_event_decomposition(env: HEnv, cfg: HConfig, gens: int,
                               rng: np.random.Generator) -> dict:
    """Instrument factorisation: count recombination macros PROPOSED (a child
    whose factorisation introduced a macro referencing >=2 distinct macros),
    those that SURVIVE (the child is not MDL-worse than its parent), and whether
    any FIXES (present in the final champion)."""
    gacfg = _gacfg(cfg)
    target = env.target()
    pop = [Individual(g=random_structured_program(rng, gacfg))
           for _ in range(cfg.pop)]
    _reeval_pop(pop, target, cfg)
    proposed = survived = 0
    for _ in range(gens):
        # one manual generation so we can watch each factorisation
        pop.sort(key=lambda i: i.J, reverse=True)
        newpop = [pop[i] for i in range(cfg.elitism)]
        while len(newpop) < cfg.pop:
            idx = rng.integers(0, len(pop), size=cfg.tournament)
            parent = max((pop[int(j)] for j in idx), key=lambda i: i.J)
            childg = parent.g.copy()
            if rng.random() < cfg.p_mut:
                before_r = n_recomb(childg)
                childg2 = _mutate(rng, gacfg, childg, cfg)
                after_r = n_recomb(childg2)
                if after_r > before_r:              # a recombination was proposed
                    proposed += 1
                    if description_length(childg2) <= description_length(parent.g):
                        survived += 1
                childg = childg2
            child = Individual(g=childg)
            _eval(child, target, cfg)
            newpop.append(child)
        pop = newpop
    champ = max(pop, key=lambda i: i.J)
    fixed = n_recomb(champ.g)
    return {"proposed": proposed, "survived": survived, "fixed": fixed,
            "champ_F": champ.F}
