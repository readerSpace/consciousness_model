"""exp599a -- Codec Evolution / Representation-Language Selection (engine).

exp598 proved recombination is not selected because the FIXED genetic codec gives
it no MDL benefit (dL_compose < 0 at realistic reuse; even F=1 + an optimal
factoriser never recombine; REP subsumes repetition).  exp599 stops fixing the
codec: the genetic LANGUAGE D itself becomes a selected trait.

    (G, D) -> x ,     J = F - lambda * [ L(G | D) + L(D) ]

Two candidate languages:

    D0 = {LIT, CALL, REP}
    D1 = D0 + {COMPOSE}      COMPOSE a b  ==  expand(macro a) ++ expand(macro b)

A COMPOSE instruction expresses a two-module composition in ONE opcode instead of
[CALL a, CALL b] (two opcodes), so it saves exactly OPCODE_BITS per composition.
Owning the extra instruction is NOT free: D1 pays a fixed language cost L(D1)=
C_opcode.  Hence D1 wins only when a world contains enough reusable composition
that the saved opcodes outweigh C_opcode.

Three worlds make the prediction sharp:

    flat / REP        ABABAB...        -> D0 (REP already suffices)
    compositional     reused pairs      -> D1 (COMPOSE pays)
    random            no reuse          -> D0 (COMPOSE cost wasted)

The break-even is causal: sweeping composition reuse k, the exact MDL crossover
k*_MDL is where L(D1)+L(G|D1) < L(D0)+L(G|D0); evolution's adoption point k_evo
is compared to it.

Self-contained codec (does NOT touch genome.py, so other experiments are intact).
All randomness flows through explicit numpy Generators.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from genome import gamma_bits                       # reuse the exact Elias-gamma cost

OP_LIT, OP_CALL, OP_REP, OP_COMPOSE = 0, 1, 2, 3
OPCODE_BITS = 2                                      # <=4 opcodes
LANG_D0 = "D0"
LANG_D1 = "D1"


@dataclass(frozen=True)
class Ins:
    op: int
    a: int = 0
    b: int = 0


@dataclass
class Prog:
    macros: List[List[Ins]]
    program: List[Ins]
    A: int
    L: int

    def copy(self) -> "Prog":
        return Prog([list(m) for m in self.macros], list(self.program),
                    self.A, self.L)


# ---------------------------------------------------------------------------
# bit costs
# ---------------------------------------------------------------------------
def _sym_bits(A: int) -> int:
    return max(1, (A - 1).bit_length())


def ins_bits(ins: Ins, A: int) -> int:
    if ins.op == OP_LIT:
        return OPCODE_BITS + _sym_bits(A)
    if ins.op == OP_CALL:
        return OPCODE_BITS + gamma_bits(ins.a + 1)
    if ins.op == OP_REP:
        return OPCODE_BITS + gamma_bits(ins.a) + gamma_bits(ins.b + 1)
    if ins.op == OP_COMPOSE:
        return OPCODE_BITS + gamma_bits(ins.a + 1) + gamma_bits(ins.b + 1)
    raise ValueError(ins.op)


def _list_bits(instrs: List[Ins], A: int) -> int:
    return gamma_bits(len(instrs) + 1) + sum(ins_bits(i, A) for i in instrs)


def uses_compose(g: Prog) -> bool:
    if any(i.op == OP_COMPOSE for i in g.program):
        return True
    return any(i.op == OP_COMPOSE for m in g.macros for i in m)


def description_length(g: Prog, language: str, c_opcode: int) -> int:
    """L(D) + L(G|D).  COMPOSE is only legal under D1; a D0 program that uses it
    is infinite (disallowed)."""
    if language == LANG_D0 and uses_compose(g):
        return 10 ** 9
    b = gamma_bits(len(g.macros) + 1)
    for m in g.macros:
        b += _list_bits(m, g.A)
    b += _list_bits(g.program, g.A)
    LD = c_opcode if language == LANG_D1 else 0
    return b + LD


# ---------------------------------------------------------------------------
# expansion (genotype -> phenotype), capped at L
# ---------------------------------------------------------------------------
def _expand_instrs(instrs: List[Ins], expanded: List[List[int]], cap: int) -> List[int]:
    out: List[int] = []
    for ins in instrs:
        if ins.op == OP_LIT:
            out.append(ins.a)
        elif ins.op == OP_CALL:
            out.extend(expanded[ins.a])
        elif ins.op == OP_REP:
            body = expanded[ins.b]
            for _ in range(ins.a):
                out.extend(body)
                if len(out) >= cap:
                    break
        elif ins.op == OP_COMPOSE:
            out.extend(expanded[ins.a])
            out.extend(expanded[ins.b])
        else:
            raise ValueError(ins.op)
        if len(out) >= cap:
            break
    return out[:cap]


def expand(g: Prog) -> List[int]:
    expanded: List[List[int]] = []
    for i, m in enumerate(g.macros):
        expanded.append(_expand_instrs(m, expanded, g.L))   # macro i refs < i
    seq = _expand_instrs(g.program, expanded, g.L)
    if len(seq) < g.L:
        seq = seq + [0] * (g.L - len(seq))
    return seq[: g.L]


def fitness(g: Prog, target: List[int]) -> float:
    phen = expand(g)
    L = len(target)
    return sum(1 for i in range(L) if phen[i] == target[i]) / L


# ---------------------------------------------------------------------------
# worlds
# ---------------------------------------------------------------------------
@dataclass
class World:
    kind: str
    target: List[int]
    A: int
    motifs: List[List[int]]
    pairs: List[Tuple[int, int]]
    seq: List[int]                                   # pair indices (compositional)


def flat_world(A: int, L: int, period: int, rng: np.random.Generator) -> World:
    motif = rng.integers(0, A, size=period).tolist()
    target = [motif[i % period] for i in range(L)]
    return World("flat", target, A, [motif], [], [])


def compositional_world(A: int, n_pairs: int, k: int, motif_len: int,
                        rng: np.random.Generator, n_motifs: int = 3) -> World:
    """n_pairs DISTINCT motif-pairs, each reused k times, in APERIODIC order so
    REP cannot capture it; COMPOSE can express every pair in one opcode."""
    motifs = [rng.integers(0, A, size=motif_len).tolist() for _ in range(n_motifs)]
    allpairs = [(i, j) for i in range(n_motifs) for j in range(n_motifs) if i != j]
    pairs = allpairs[:n_pairs]
    seq: List[int] = []
    for p in range(len(pairs)):
        seq += [p] * k
    rng.shuffle(seq)                                 # aperiodic
    target: List[int] = []
    for p in seq:
        a, b = pairs[p]
        target += motifs[a] + motifs[b]
    return World("compositional", target, A, motifs, pairs, seq)


def random_world(A: int, L: int, rng: np.random.Generator) -> World:
    return World("random", rng.integers(0, A, size=L).tolist(), A, [], [], [])


# ---------------------------------------------------------------------------
# hand-built representations (exact, F=1) for the compositional world
# ---------------------------------------------------------------------------
def _motif_macros(world: World) -> List[List[Ins]]:
    return [[Ins(OP_LIT, s) for s in m] for m in world.motifs]


def g_D0_inline(world: World) -> Prog:
    program: List[Ins] = []
    for p in world.seq:
        a, b = world.pairs[p]
        program += [Ins(OP_CALL, a), Ins(OP_CALL, b)]
    return Prog(_motif_macros(world), program, world.A, len(world.target))


def g_D0_pairmacros(world: World) -> Prog:
    """D0's compositional option: a macro [CALL a, CALL b] per distinct pair,
    then CALL it (the exp598 recombination path)."""
    M = len(world.motifs)
    macros = _motif_macros(world)
    pair_macro_idx = {}
    for pi, (a, b) in enumerate(world.pairs):
        pair_macro_idx[pi] = M + pi
        macros.append([Ins(OP_CALL, a), Ins(OP_CALL, b)])
    program = [Ins(OP_CALL, pair_macro_idx[p]) for p in world.seq]
    return Prog(macros, program, world.A, len(world.target))


def g_flat_pairmacros(world: World) -> Prog:
    """A flat macro of the pair's LITERALS per distinct pair (no motif sharing),
    then CALL it.  Cheap for SHORT motifs; wasteful for long/shared motifs."""
    macros: List[List[Ins]] = []
    pair_macro_idx = {}
    for pi, (a, b) in enumerate(world.pairs):
        pair_macro_idx[pi] = pi
        body = [Ins(OP_LIT, s) for s in world.motifs[a] + world.motifs[b]]
        macros.append(body)
    program = [Ins(OP_CALL, pair_macro_idx[p]) for p in world.seq]
    return Prog(macros, program, world.A, len(world.target))


def g_D1_compose_inline(world: World) -> Prog:
    """D1: one COMPOSE per pair occurrence (one opcode instead of two CALLs)."""
    program = [Ins(OP_COMPOSE, *world.pairs[p]) for p in world.seq]
    return Prog(_motif_macros(world), program, world.A, len(world.target))


def g_D1_compose_macros(world: World) -> Prog:
    """D1's compositional option: a macro [COMPOSE a b] per distinct pair (one
    opcode body, references shared motif macros), then CALL it.  Cheaper than
    D0's [CALL a, CALL b] pair-macro body by one opcode each."""
    M = len(world.motifs)
    macros = _motif_macros(world)
    pair_macro_idx = {}
    for pi, (a, b) in enumerate(world.pairs):
        pair_macro_idx[pi] = M + pi
        macros.append([Ins(OP_COMPOSE, a, b)])
    program = [Ins(OP_CALL, pair_macro_idx[p]) for p in world.seq]
    return Prog(macros, program, world.A, len(world.target))


# ---------------------------------------------------------------------------
# per-language optimal cost on a world (min over that language's representations)
# ---------------------------------------------------------------------------
def best_cost(world: World, language: str, c_opcode: int) -> Tuple[int, str]:
    tgt = world.target
    cands: Dict[str, Prog] = {
        "literal": Prog([], [Ins(OP_LIT, s) for s in tgt], world.A, len(tgt)),
    }
    if world.kind == "flat":
        period = len(world.motifs[0])
        reps = len(tgt) // period + 1
        cands["rep"] = Prog([[Ins(OP_LIT, s) for s in world.motifs[0]]],
                            [Ins(OP_REP, reps, 0)], world.A, len(tgt))
    if world.kind == "compositional":
        cands["D0_inline"] = g_D0_inline(world)
        cands["D0_pairmacros"] = g_D0_pairmacros(world)
        cands["flat_pairmacros"] = g_flat_pairmacros(world)
        if language == LANG_D1:
            cands["D1_compose_inline"] = g_D1_compose_inline(world)
            cands["D1_compose_macros"] = g_D1_compose_macros(world)
    best_L, best_name = 10 ** 9, "literal"
    for name, g in cands.items():
        if language == LANG_D0 and uses_compose(g):
            continue
        if expand(g) != tgt:
            continue
        L = description_length(g, language, c_opcode)
        if L < best_L:
            best_L, best_name = L, name
    return best_L, best_name


def selected_language(world: World, c_opcode: int) -> dict:
    c0, n0 = best_cost(world, LANG_D0, c_opcode)
    c1, n1 = best_cost(world, LANG_D1, c_opcode)
    return {"cost_D0": c0, "cost_D1": c1, "best_D0": n0, "best_D1": n1,
            "selected": LANG_D1 if c1 < c0 else LANG_D0,
            "margin": c0 - c1}


# ---------------------------------------------------------------------------
# exact MDL crossover in the NUMBER OF DISTINCT COMPOSITIONS (the "many
# combinations" axis) -- D1 is adopted once the saved opcodes beat C_opcode
# ---------------------------------------------------------------------------
def _world_for_ncomp(A: int, n_comp: int, k: int, motif_len: int,
                     rng: np.random.Generator) -> World:
    n_motifs = 2
    while n_motifs * (n_motifs - 1) < n_comp:
        n_motifs += 1
    return compositional_world(A, n_comp, k, motif_len, rng, n_motifs=n_motifs)


def mdl_crossover(A: int, n_comps: List[int], k: int, motif_len: int,
                  c_opcode: int, seed: int = 0) -> dict:
    rows = []
    nstar = None
    for nc in n_comps:
        rng = np.random.default_rng(seed)
        world = _world_for_ncomp(A, nc, k, motif_len, rng)
        sel = selected_language(world, c_opcode)
        rows.append({"n_comp": nc, **sel})
        if nstar is None and sel["selected"] == LANG_D1:
            nstar = nc
    return {"rows": rows, "n_comp_star_MDL": nstar}


# ---------------------------------------------------------------------------
# evolution with the LANGUAGE as a selected trait (phenotype-preserving ops, so
# F stays 1 when seeded from the literal; selection then minimises L(D)+L(G|D))
# ---------------------------------------------------------------------------
def _literal(target: List[int], A: int) -> Prog:
    return Prog([], [Ins(OP_LIT, s) for s in target], A, len(target))


def _compose_to_calls(g: Prog) -> Prog:
    """Rewrite every COMPOSE a b into [CALL a, CALL b] (to make a genome legal
    under D0 without changing its phenotype)."""
    def rw(instrs):
        out = []
        for i in instrs:
            if i.op == OP_COMPOSE:
                out += [Ins(OP_CALL, i.a), Ins(OP_CALL, i.b)]
            else:
                out.append(i)
        return out
    return Prog([rw(m) for m in g.macros], rw(g.program), g.A, g.L)


@dataclass
class LangInd:
    g: Prog
    lang: str
    cost: int = 0


def _factor_candidates(g: Prog) -> List[Prog]:
    """All single repeated-substring extractions of the program."""
    prog = g.program
    n = len(prog)
    cands = []
    if n < 4 or len(g.macros) >= 24:
        return cands
    seen_keys = set()
    for length in range(2, min(12, n // 2) + 1):
        occ: Dict[tuple, List[int]] = {}
        for s in range(0, n - length + 1):
            occ.setdefault(tuple(prog[s:s + length]), []).append(s)
        for key, pos in occ.items():
            if key in seen_keys or any(ins.op == OP_COMPOSE for ins in key):
                continue
            cnt = 0
            last = -1
            for p in pos:
                if p >= last:
                    cnt += 1
                    last = p + length
            if cnt < 2:
                continue
            seen_keys.add(key)
            h = g.copy()
            idx = len(h.macros)
            h.macros.append(list(key))
            out = []
            i = 0
            klen = len(key)
            while i < len(prog):
                if tuple(prog[i:i + klen]) == key:
                    out.append(Ins(OP_CALL, idx))
                    i += klen
                else:
                    out.append(prog[i])
                    i += 1
            h.program = out
            cands.append(h)
    return cands


def _compose_candidates(g: Prog, language: str) -> List[Prog]:
    """Abstract each repeated distinct macro-call pair.  Under D1 generate BOTH
    the inline COMPOSE replacement AND a [COMPOSE a b] pair-macro; under D0
    generate a [CALL a, CALL b] pair-macro (inline is already the current state)."""
    prog = g.program
    cnt: Dict[tuple, int] = {}
    for i in range(len(prog) - 1):
        a, b = prog[i], prog[i + 1]
        if a.op == OP_CALL and b.op == OP_CALL and a.a != b.a:
            cnt[(a.a, b.a)] = cnt.get((a.a, b.a), 0) + 1
    cands: List[Prog] = []
    for (a, b), c in cnt.items():
        if c < 2:
            continue
        if language == LANG_D1:
            # (i) inline COMPOSE replacement
            h = g.copy()
            out, i = [], 0
            while i < len(prog):
                if (i < len(prog) - 1 and prog[i].op == OP_CALL and prog[i].a == a
                        and prog[i + 1].op == OP_CALL and prog[i + 1].a == b):
                    out.append(Ins(OP_COMPOSE, a, b)); i += 2
                else:
                    out.append(prog[i]); i += 1
            h.program = out
            cands.append(h)
            # (ii) [COMPOSE a b] pair-macro
            if len(g.macros) < 24:
                h2 = g.copy()
                idx = len(h2.macros)
                h2.macros.append([Ins(OP_COMPOSE, a, b)])
                out, i = [], 0
                while i < len(prog):
                    if (i < len(prog) - 1 and prog[i].op == OP_CALL and prog[i].a == a
                            and prog[i + 1].op == OP_CALL and prog[i + 1].a == b):
                        out.append(Ins(OP_CALL, idx)); i += 2
                    else:
                        out.append(prog[i]); i += 1
                h2.program = out
                cands.append(h2)
        else:
            if len(g.macros) >= 24:
                continue
            h = g.copy()
            idx = len(h.macros)
            h.macros.append([Ins(OP_CALL, a), Ins(OP_CALL, b)])
            out, i = [], 0
            while i < len(prog):
                if (i < len(prog) - 1 and prog[i].op == OP_CALL and prog[i].a == a
                        and prog[i + 1].op == OP_CALL and prog[i + 1].a == b):
                    out.append(Ins(OP_CALL, idx)); i += 2
                else:
                    out.append(prog[i]); i += 1
        cands.append(h)
    return cands


def _best_reduction(g: Prog, language: str, c_opcode: int) -> Optional[Prog]:
    base = description_length(g, language, c_opcode)
    best, best_L = None, base
    for cand in _factor_candidates(g) + _compose_candidates(g, language):
        L = description_length(cand, language, c_opcode)
        if L < best_L:
            best_L, best = L, cand
    return best


def _factor_lit_motifs(g: Prog) -> Prog:
    """Language-independent motif extraction: repeatedly factor the most frequent
    repeated LIT-only substring into a macro (stops at the motif level, leaving a
    program of motif-calls from which each language optimises COMPOSITION)."""
    h = g.copy()
    while len(h.macros) < 24:
        prog = h.program
        n = len(prog)
        best_key, best_rank = None, (1, 0)
        for length in range(2, min(12, n // 2) + 1):
            occ: Dict[tuple, List[int]] = {}
            for s in range(0, n - length + 1):
                sub = prog[s:s + length]
                if any(ins.op != OP_LIT for ins in sub):
                    continue
                occ.setdefault(tuple(sub), []).append(s)
            for key, pos in occ.items():
                cnt, last = 0, -1
                for p in pos:
                    if p >= last:
                        cnt += 1
                        last = p + length
                if cnt >= 2 and (cnt, length) > best_rank:
                    best_rank, best_key = (cnt, length), key
        if best_key is None:
            break
        idx = len(h.macros)
        h.macros.append(list(best_key))
        klen = len(best_key)
        out, i = [], 0
        while i < len(prog):
            if tuple(prog[i:i + klen]) == best_key:
                out.append(Ins(OP_CALL, idx)); i += klen
            else:
                out.append(prog[i]); i += 1
        h.program = out
    return h


def evolve_language(world: World, c_opcode: int, gens: int = 60, pop: int = 24,
                    seed: int = 0) -> dict:
    """Factor motifs first (shared, language-independent); then evolve COMPOSITION
    handling with the language as a selected trait.  Mutation applies the single
    best MDL-reducing composition step under the individual's language, or flips
    language.  Phenotype preserved (F=1); selection minimises L(D)+L(G|D)."""
    rng = np.random.default_rng(seed)
    tgt = world.target
    base = _factor_lit_motifs(_literal(tgt, world.A))
    inds = [LangInd(base.copy(), LANG_D0 if i % 2 == 0 else LANG_D1)
            for i in range(pop)]
    for ind in inds:
        ind.cost = description_length(ind.g, ind.lang, c_opcode)

    def _compose_all(g: Prog, lang: str) -> Prog:
        h = g
        while True:
            best, best_L = None, description_length(h, lang, c_opcode)
            for cand in _compose_candidates(h, lang):
                L = description_length(cand, lang, c_opcode)
                if L < best_L:
                    best_L, best = L, cand
            if best is None:
                return h
            h = best

    def mutate(ind: LangInd) -> LangInd:
        g, lang = ind.g.copy(), ind.lang
        if rng.random() < 0.3:                      # language flip, then re-optimise
            if lang == LANG_D1:
                g, lang = _compose_to_calls(g), LANG_D0
            else:
                lang = LANG_D1
        g = _compose_all(g, lang)                    # reach this language's optimum
        return LangInd(g, lang, description_length(g, lang, c_opcode))

    for _ in range(gens):
        inds.sort(key=lambda i: i.cost)
        newpop = [inds[0], inds[1]]
        while len(newpop) < pop:
            idx = rng.integers(0, len(inds), size=3)
            parent = min((inds[int(j)] for j in idx), key=lambda i: i.cost)
            newpop.append(mutate(parent))
        inds = newpop
    inds.sort(key=lambda i: i.cost)
    champ = inds[0]
    frac_d1 = float(np.mean([1.0 if i.lang == LANG_D1 else 0.0 for i in inds]))
    assert expand(champ.g) == tgt
    return {"champion_lang": champ.lang, "champion_cost": champ.cost,
            "frac_D1": frac_d1, "uses_compose": uses_compose(champ.g)}


def k_evo_sweep(A: int, n_comps: List[int], k: int, motif_len: int,
                c_opcode: int, seeds: List[int]) -> dict:
    rows = []
    n_evo = None
    for nc in n_comps:
        langs = []
        for s in seeds:
            rng = np.random.default_rng(1000 + s)
            world = _world_for_ncomp(A, nc, k, motif_len, rng)
            res = evolve_language(world, c_opcode, seed=2000 + s)
            langs.append(res["champion_lang"])
        frac = float(np.mean([1.0 if l == LANG_D1 else 0.0 for l in langs]))
        rows.append({"n_comp": nc, "frac_champ_D1": frac})
        if n_evo is None and frac >= 0.5:
            n_evo = nc
    return {"rows": rows, "n_comp_evo": n_evo}

