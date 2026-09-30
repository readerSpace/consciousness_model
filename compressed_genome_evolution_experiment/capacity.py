"""exp593 -- Adaptive Genome Capacity (genome birth / death).

exp588-592 evolved *what* is represented at (mostly) fixed capacity.  exp593
evolves *how much representational capacity the genome has*.  The genome is
variable-length (macros + program, from genome.py); mutation may change its
STRUCTURE, not just its values:

    ADD          add a fresh random macro
    DUP          duplicate an existing macro  M -> (M, M')  and repoint one use
    SPLIT        split a macro into two
    MERGE        merge two macros into one
    DELETE       remove an unused macro / a program instruction
    REUSE        insert a CALL/REP of an existing macro

Crucially there is NO "grow" or "shrink" rule.  Only these edits plus MDL
selection  J(G) = F(G) - lambda * L(G).  The question is whether

    "grow when under-capacity, shrink when redundant"

emerges from selection alone.

Environment is staged in complexity (number of independent tiling rules):
    E0..E4  =  1 -> 2 -> 3 -> 2 -> 1   rules.
Each rule is a distinct motif REUSED (tiled >=2x) in its region, so a macro pays
its way; the MDL-optimal macro count tracks the rule count.

Small enough (A=2, L=12, motif_len=2) that F*(K)=max_{L(G)<=K} F(G) is computed
EXACTLY (via optimal.kx over all phenotypes), so "representational deficit" is a
measured fact, not a guess.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np

import genome as G
from genome import Genome, Instr, OP_LIT, OP_CALL, OP_REP, OP_MIR
import optimal as OPT

# dynamics system (macros clearly pay for themselves here); F*(K) uses a
# separate SMALL system where exact enumeration is feasible (see fstar_curve).
DEFAULT_CFG = dict(
    A=4, L=32, nmax=8, max_macros=8, max_macro_len=6, max_program=48,
    motifs=[np.array([0, 1]), np.array([2, 3]), np.array([1, 2]), np.array([3, 0])],
)

# small system for the exact representational-deficit analysis
SMALL = dict(A=2, L=12,
             motifs=[np.array([0, 1]), np.array([1, 0]), np.array([1, 1])])


# ---------------------------------------------------------------------------
# staged environment
# ---------------------------------------------------------------------------
def make_target(rule_count: int, cfg=DEFAULT_CFG) -> np.ndarray:
    """Tile `rule_count` distinct motifs across L, each reused within its region."""
    Lc = cfg["L"]
    assert Lc % rule_count == 0
    region = Lc // rule_count
    x = np.zeros(Lc, dtype=int)
    for r in range(rule_count):
        m = cfg["motifs"][r % len(cfg["motifs"])]
        seg = np.tile(m, region // len(m) + 1)[:region]
        x[r * region:(r + 1) * region] = seg
    return x


def fitness(pheno, target) -> float:
    return float(np.mean(np.asarray(pheno) == np.asarray(target)))


# ---------------------------------------------------------------------------
# capacity metrics
# ---------------------------------------------------------------------------
def capacity_metrics(g: Genome) -> dict:
    counts = G.macro_call_counts(g)
    n_used = sum(1 for c in counts if c > 0)
    n_params = (sum(1 for ins in g.program if ins.op == OP_LIT)
                + sum(1 for m in g.macros for ins in m if ins.op == OP_LIT))
    return {
        "n_macros": len(g.macros),
        "n_used_macros": n_used,
        "n_program": len(g.program),
        "n_params": n_params,
        "LG": G.program_bits(g),
        "LD": G.macro_bits(g),
        "L_total": G.description_length(g),
    }


# ---------------------------------------------------------------------------
# structural mutation operators (each returns a NEW genome or the input on no-op)
# ---------------------------------------------------------------------------
def _rand_lit(rng, cfg):
    return Instr(OP_LIT, int(rng.integers(0, cfg["A"])))


def op_add_macro(rng, g, cfg):
    if len(g.macros) >= cfg["max_macros"]:
        return g
    h = g.copy()
    ln = int(rng.integers(1, cfg["max_macro_len"] + 1))
    h.macros.append([_rand_lit(rng, cfg) for _ in range(ln)])
    return h


def op_dup_macro(rng, g, cfg):
    """M -> (M, M'); repoint ONE program reference of M to the copy M'."""
    if not g.macros or len(g.macros) >= cfg["max_macros"]:
        return g
    h = g.copy()
    k = int(rng.integers(0, len(h.macros)))
    h.macros.append([Instr(i.op, i.a, i.b) for i in h.macros[k]])  # identical copy
    kp = len(h.macros) - 1
    refs = [idx for idx, ins in enumerate(h.program)
            if ins.op in (OP_CALL, OP_MIR) and ins.a == k
            or (ins.op == OP_REP and ins.b == k)]
    if refs:
        idx = refs[int(rng.integers(0, len(refs)))]
        ins = h.program[idx]
        if ins.op == OP_REP:
            h.program[idx] = Instr(OP_REP, ins.a, kp)
        else:
            h.program[idx] = Instr(ins.op, kp)
    return h


def op_delete_macro(rng, g, cfg):
    """Remove an UNUSED macro (birth/death: prune redundant capacity)."""
    counts = G.macro_call_counts(g)
    unused = [i for i, c in enumerate(counts) if c == 0]
    if not unused:
        return g
    k = unused[int(rng.integers(0, len(unused)))]
    return _remove_macro(g, k)


def op_merge_macro(rng, g, cfg):
    """Merge two macros a<b into one (concatenate bodies); repoint refs of b to a
    with a REP/CALL as appropriate.  Conservative: only when b's body has no refs."""
    if len(g.macros) < 2:
        return g
    h = g.copy()
    a, b = sorted(rng.choice(len(h.macros), size=2, replace=False).tolist())
    # only merge if b's body is pure LIT (safe to append after a)
    if any(ins.op != OP_LIT for ins in h.macros[b]):
        return g
    merged = list(h.macros[a]) + list(h.macros[b])
    if len(merged) > cfg["max_macro_len"] * 2:
        return g
    h.macros[a] = merged
    # repoint program refs of b -> a (best-effort: as CALL a)
    for idx, ins in enumerate(h.program):
        if (ins.op in (OP_CALL, OP_MIR) and ins.a == b):
            h.program[idx] = Instr(OP_CALL, a)
        elif ins.op == OP_REP and ins.b == b:
            h.program[idx] = Instr(OP_CALL, a)
    return _remove_macro(h, b)


def op_split_macro(rng, g, cfg):
    """Split a macro (len>=2) into two; repoint its CALLs into CALL,CALL."""
    cands = [i for i, m in enumerate(g.macros)
             if len(m) >= 2 and all(ins.op == OP_LIT for ins in m)]
    if not cands or len(g.macros) >= cfg["max_macros"]:
        return g
    h = g.copy()
    k = cands[int(rng.integers(0, len(cands)))]
    body = h.macros[k]
    cut = int(rng.integers(1, len(body)))
    left, right = body[:cut], body[cut:]
    h.macros[k] = left
    h.macros.append(right)
    kr = len(h.macros) - 1
    # replace CALL k in program with CALL k, CALL kr (only simple CALLs)
    newprog = []
    for ins in h.program:
        if ins.op == OP_CALL and ins.a == k:
            newprog.append(Instr(OP_CALL, k))
            newprog.append(Instr(OP_CALL, kr))
        else:
            newprog.append(ins)
    h.program = newprog
    return h


def op_reuse(rng, g, cfg):
    """Insert a CALL/REP of an existing macro into the program."""
    if not g.macros or len(g.program) >= cfg["max_program"]:
        return g
    h = g.copy()
    k = int(rng.integers(0, len(h.macros)))
    pos = int(rng.integers(0, len(h.program) + 1))
    if rng.random() < 0.5:
        h.program.insert(pos, Instr(OP_REP, int(rng.integers(2, cfg["nmax"] + 1)), k))
    else:
        h.program.insert(pos, Instr(OP_CALL, k))
    return h


def op_del_prog(rng, g, cfg):
    if len(g.program) <= 1:
        return g
    h = g.copy()
    del h.program[int(rng.integers(0, len(h.program)))]
    return h


def op_ins_prog(rng, g, cfg):
    if len(g.program) >= cfg["max_program"]:
        return g
    h = g.copy()
    pos = int(rng.integers(0, len(h.program) + 1))
    M = len(h.macros)
    if M == 0 or rng.random() < 0.5:
        h.program.insert(pos, _rand_lit(rng, cfg))
    else:
        h.program.insert(pos, Instr(OP_CALL, int(rng.integers(0, M))))
    return h


def op_value(rng, g, cfg):
    """Change one LIT symbol somewhere (macro or program)."""
    h = g.copy()
    pool = [("p", i) for i, ins in enumerate(h.program) if ins.op == OP_LIT]
    for mi, m in enumerate(h.macros):
        pool += [("m", mi, i) for i, ins in enumerate(m) if ins.op == OP_LIT]
    if not pool:
        return g
    c = pool[int(rng.integers(0, len(pool)))]
    if c[0] == "p":
        h.program[c[1]] = _rand_lit(rng, cfg)
    else:
        h.macros[c[1]][c[2]] = _rand_lit(rng, cfg)
    return h


def op_operand(rng, g, cfg):
    """Retarget a REP count or a macro-index operand."""
    idxs = [i for i, ins in enumerate(g.program)
            if ins.op in (OP_REP, OP_CALL, OP_MIR)]
    if not idxs or not g.macros:
        return g
    h = g.copy()
    i = idxs[int(rng.integers(0, len(idxs)))]
    ins = h.program[i]
    M = len(h.macros)
    if ins.op == OP_REP:
        if rng.random() < 0.5:
            n = min(cfg["nmax"], max(1, ins.a + int(rng.integers(-2, 3))))
            h.program[i] = Instr(OP_REP, n, ins.b)
        else:
            h.program[i] = Instr(OP_REP, ins.a, int(rng.integers(0, M)))
    else:
        h.program[i] = Instr(ins.op, int(rng.integers(0, M)))
    return h


def _remove_macro(g, k):
    """Delete macro k and reindex all references (macros>k and program)."""
    h = g.copy()
    del h.macros[k]

    def fix(ins):
        if ins.op in (OP_CALL, OP_MIR):
            a = ins.a
            if a == k:
                return None
            return Instr(ins.op, a - 1 if a > k else a)
        if ins.op == OP_REP:
            b = ins.b
            if b == k:
                return None
            return Instr(OP_REP, ins.a, b - 1 if b > k else b)
        return ins

    new_macros = []
    for mi, m in enumerate(h.macros):
        nb = []
        for ins in m:
            fi = fix(ins)
            if fi is not None:
                nb.append(fi)
        new_macros.append(nb if nb else [_rand_lit_static()])
    h.macros = new_macros
    newprog = []
    for ins in h.program:
        fi = fix(ins)
        if fi is not None:
            newprog.append(fi)
    h.program = newprog if newprog else [_rand_lit_static()]
    return h


def _rand_lit_static():
    return Instr(OP_LIT, 0)


# operator categories for the ADD/DELETE controls
GROWTH_OPS = [op_add_macro, op_dup_macro, op_split_macro, op_reuse, op_ins_prog]
SHRINK_OPS = [op_delete_macro, op_merge_macro, op_del_prog]
NEUTRAL_OPS = [op_value, op_operand]


def mutate_capacity(rng, g, cfg, allow_add=True, allow_delete=True):
    ops = list(NEUTRAL_OPS)
    if allow_add:
        ops += GROWTH_OPS
    if allow_delete:
        ops += SHRINK_OPS
    op = ops[int(rng.integers(0, len(ops)))]
    try:
        h = op(rng, g, cfg)
        if len(h.macros) > cfg["max_macros"] or len(h.program) > cfg["max_program"]:
            return g.copy()
        G.validate(h)
        return h
    except Exception:
        return g.copy()


# ---------------------------------------------------------------------------
# staged GA (warm-started across the complexity schedule; MDL selection)
# ---------------------------------------------------------------------------
@dataclass
class StageRecord:
    stage: int
    rule_count: int
    best_F: float
    metrics: dict


def _random_init(rng, cfg):
    prog = [_rand_lit(rng, cfg) for _ in range(int(rng.integers(2, cfg["L"])))]
    return Genome(macros=[], program=prog, A=cfg["A"], L=cfg["L"], nmax=cfg["nmax"])


def _evaluate(g, target, lam):
    F = fitness(G.expand(g), target)
    Lt = G.description_length(g)
    return F, Lt, F - lam * Lt


def run_staged(schedule: List[int], lam: float, seed: int,
               gens_per_stage: int = 200, pop: int = 120,
               allow_add=True, allow_delete=True,
               cfg=None) -> List[StageRecord]:
    cfg = cfg or DEFAULT_CFG
    rng = np.random.default_rng(seed)
    pop_g = [_random_init(rng, cfg) for _ in range(pop)]
    targets = [make_target(r, cfg) for r in schedule]
    records = []

    def objective(g, target):
        return _evaluate(g, target, lam)

    # initial eval on stage 0
    scored = [(g,) + objective(g, targets[0]) for g in pop_g]  # (g,F,Lt,J)

    for s, rule_count in enumerate(schedule):
        target = targets[s]
        # re-evaluate carried population on the new target
        scored = [(g, *objective(g, target)) for (g, *_ ) in scored]
        for _gen in range(gens_per_stage):
            scored.sort(key=lambda t: t[3], reverse=True)
            elite = [scored[0], scored[1]]
            newpop = list(elite)
            while len(newpop) < pop:
                # tournament on J
                idx = rng.integers(0, len(scored), size=4)
                par = max((scored[int(i)] for i in idx), key=lambda t: t[3])
                child = mutate_capacity(rng, par[0], cfg, allow_add, allow_delete)
                if rng.random() < 0.3:
                    child = mutate_capacity(rng, child, cfg, allow_add, allow_delete)
                newpop.append((child, *objective(child, target)))
            scored = newpop
        best = max(scored, key=lambda t: t[1])          # best by F
        # among genomes at the best F, report the MDL-smallest (effective genome)
        bestF = best[1]
        cohort = [t for t in scored if t[1] >= bestF - 1e-9]
        champ = min(cohort, key=lambda t: t[2])          # smallest L_total at best F
        records.append(StageRecord(
            stage=s, rule_count=rule_count, best_F=champ[1],
            metrics=capacity_metrics(champ[0])))
    return records


# ---------------------------------------------------------------------------
# exact representational-capacity curve  F*(K) = max_{L(G)<=K} F(G)
# ---------------------------------------------------------------------------
def fstar_curve(target: np.ndarray, A_s: int, L_s: int, mlen: int = 6) -> dict:
    """Exact F*(K) over ALL phenotypes (small A,L): for each phenotype x,
    K(x)=min description length, F=match(x,target).  F*(K)=max F with K(x)<=K."""
    import itertools
    tgt = tuple(int(v) for v in target)
    best_at_K = {}
    for xt in itertools.product(range(A_s), repeat=L_s):
        k = OPT.kx(xt, A_s, L_s, mlen=mlen, Mmax=2, nmax=8)
        f = sum(1 for a, b in zip(xt, tgt) if a == b) / L_s
        if k not in best_at_K or f > best_at_K[k]:
            best_at_K[k] = f
    # cumulative: F*(K) = max over k'<=K
    Ks = sorted(best_at_K)
    curve = {}
    run = 0.0
    for k in Ks:
        run = max(run, best_at_K[k])
        curve[k] = run
    return curve


def k_min_for_fitness(curve: dict, f_target: float) -> Optional[int]:
    for k in sorted(curve):
        if curve[k] >= f_target - 1e-9:
            return k
    return None
