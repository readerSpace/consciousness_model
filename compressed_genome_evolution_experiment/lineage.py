"""exp594 -- Duplication / Divergence / Specialization.

exp593 showed macro COUNT tracks demand.  exp594 asks the harder question: when a
new module appears, did it arise by *duplicating an existing useful module and
diverging* (the classic gene-duplication route), or by an unrelated de-novo ADD?
And once two modules coexist, are they *specialized* (each carries a different
function)?

To tell these apart we give every macro an immutable id and a parent id, and
carry that lineage through mutation:

    M_a  --DUP-->  (M_a, M_b)      parent(M_b) = id(M_a)
    ADD  ------->  M_c             parent(M_c) = None   (de novo)

Two functions live in disjoint phenotype regions:
    A occupies R_A, B occupies R_B.
    E0 scores R_A only;  E1 scores R_A and R_B.
B comes in two flavours:
    B_related    motif differs from A's by 1 symbol  (a duplicate of M_A is 1
                 mutation away -> duplication should pay)
    B_unrelated  motif differs from A's in every symbol (a duplicate gives no
                 head start)

Selection is still only  J = F - lambda * L(G).  No rule says "duplicate".

This module provides: the world, a lineage-tracked variable-length genome, the
duplication-aware mutation operators, knockout-based specialization S(M_i, fn),
and a staged GA that returns lineage + adaptation times.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

import genome as G
from genome import Genome, Instr, OP_LIT, OP_CALL, OP_REP, OP_MIR


# ---------------------------------------------------------------------------
# world: two functions in disjoint regions
# ---------------------------------------------------------------------------
@dataclass
class World:
    A: int = 4
    motif_len: int = 3
    reps: int = 4                 # tiles per region
    motif_A: np.ndarray = field(default_factory=lambda: np.array([0, 1, 2]))
    motif_B: np.ndarray = field(default_factory=lambda: np.array([0, 1, 3]))

    @property
    def region_len(self):
        return self.motif_len * self.reps

    @property
    def L(self):
        return 2 * self.region_len

    @property
    def R_A(self):
        return np.arange(0, self.region_len)

    @property
    def R_B(self):
        return np.arange(self.region_len, 2 * self.region_len)

    def target(self) -> np.ndarray:
        t = np.zeros(self.L, dtype=int)
        t[self.R_A] = np.tile(self.motif_A, self.reps)[:self.region_len]
        t[self.R_B] = np.tile(self.motif_B, self.reps)[:self.region_len]
        return t


def world_related(A=4, motif_len=3, reps=5) -> World:
    mA = np.arange(motif_len) % A
    mB = mA.copy(); mB[-1] = (mB[-1] + 1) % A          # 1-symbol change
    return World(A, motif_len, reps, mA, mB)


def world_unrelated(A=4, motif_len=3, reps=5) -> World:
    mA = np.arange(motif_len) % A
    mB = np.array([(v + A // 2 + 1) % A for v in mA])  # every symbol changed
    while np.any(mB == mA):
        mB = (mB + 1) % A
    return World(A, motif_len, reps, mA, mB)


def region_fitness(pheno, target: np.ndarray,
                   positions: np.ndarray) -> float:
    pheno = np.asarray(pheno)
    return float(np.mean(pheno[positions] == target[positions]))


def fitness_active(pheno, target, world, functions: str) -> float:
    pos = world.R_A.copy()
    if "B" in functions:
        pos = np.concatenate([world.R_A, world.R_B])
    return region_fitness(pheno, target, pos)


# ---------------------------------------------------------------------------
# lineage-tracked variable-length genome
# ---------------------------------------------------------------------------
@dataclass
class Module:
    id: int
    parent: Optional[int]


class IdAlloc:
    def __init__(self, start=0):
        self.n = start

    def next(self) -> int:
        self.n += 1
        return self.n


@dataclass
class LinGenome:
    g: Genome
    meta: List[Module]            # parallel to g.macros

    def copy(self) -> "LinGenome":
        return LinGenome(self.g.copy(),
                         [Module(m.id, m.parent) for m in self.meta])

    def macro_id(self, idx: int) -> int:
        return self.meta[idx].id


def new_lingenome(rng, world, nmax=8) -> LinGenome:
    prog = [Instr(OP_LIT, int(rng.integers(0, world.A)))
            for _ in range(int(rng.integers(2, world.L)))]
    return LinGenome(Genome(macros=[], program=prog, A=world.A, L=world.L,
                            nmax=nmax), [])


# ---------------------------------------------------------------------------
# lineage-aware structural mutation operators
# ---------------------------------------------------------------------------
def _lit(rng, world):
    return Instr(OP_LIT, int(rng.integers(0, world.A)))


def op_add(rng, lin, world, cfg, alloc):
    if len(lin.g.macros) >= cfg["max_macros"]:
        return lin
    h = lin.copy()
    ln = int(rng.integers(1, cfg["max_macro_len"] + 1))
    h.g.macros.append([_lit(rng, world) for _ in range(ln)])
    h.meta.append(Module(alloc.next(), None))          # de novo: parent None
    return h


def op_dup(rng, lin, world, cfg, alloc):
    if not lin.g.macros or len(lin.g.macros) >= cfg["max_macros"]:
        return lin
    h = lin.copy()
    k = int(rng.integers(0, len(h.g.macros)))
    h.g.macros.append([Instr(i.op, i.a, i.b) for i in h.g.macros[k]])
    h.meta.append(Module(alloc.next(), h.meta[k].id))  # parent = source id
    kp = len(h.g.macros) - 1
    refs = [idx for idx, ins in enumerate(h.g.program)
            if (ins.op in (OP_CALL, OP_MIR) and ins.a == k)
            or (ins.op == OP_REP and ins.b == k)]
    if refs:
        idx = refs[int(rng.integers(0, len(refs)))]
        ins = h.g.program[idx]
        h.g.program[idx] = (Instr(OP_REP, ins.a, kp) if ins.op == OP_REP
                            else Instr(ins.op, kp))
    return h


def op_delete_macro(rng, lin, world, cfg, alloc):
    counts = G.macro_call_counts(lin.g)
    unused = [i for i, c in enumerate(counts) if c == 0]
    if not unused:
        return lin
    k = unused[int(rng.integers(0, len(unused)))]
    return _remove_macro(lin, k)


def op_reuse(rng, lin, world, cfg, alloc):
    if not lin.g.macros or len(lin.g.program) >= cfg["max_program"]:
        return lin
    h = lin.copy()
    k = int(rng.integers(0, len(h.g.macros)))
    pos = int(rng.integers(0, len(h.g.program) + 1))
    if rng.random() < 0.5:
        h.g.program.insert(pos, Instr(OP_REP, int(rng.integers(2, cfg["nmax"] + 1)), k))
    else:
        h.g.program.insert(pos, Instr(OP_CALL, k))
    return h


def op_ins_prog(rng, lin, world, cfg, alloc):
    if len(lin.g.program) >= cfg["max_program"]:
        return lin
    h = lin.copy()
    pos = int(rng.integers(0, len(h.g.program) + 1))
    M = len(h.g.macros)
    if M == 0 or rng.random() < 0.5:
        h.g.program.insert(pos, _lit(rng, world))
    else:
        h.g.program.insert(pos, Instr(OP_CALL, int(rng.integers(0, M))))
    return h


def op_del_prog(rng, lin, world, cfg, alloc):
    if len(lin.g.program) <= 1:
        return lin
    h = lin.copy()
    del h.g.program[int(rng.integers(0, len(h.g.program)))]
    return h


def op_value(rng, lin, world, cfg, alloc):
    h = lin.copy()
    pool = [("p", i) for i, ins in enumerate(h.g.program) if ins.op == OP_LIT]
    for mi, m in enumerate(h.g.macros):
        pool += [("m", mi, i) for i, ins in enumerate(m) if ins.op == OP_LIT]
    if not pool:
        return lin
    c = pool[int(rng.integers(0, len(pool)))]
    if c[0] == "p":
        h.g.program[c[1]] = _lit(rng, world)
    else:
        h.g.macros[c[1]][c[2]] = _lit(rng, world)
    return h


def op_operand(rng, lin, world, cfg, alloc):
    idxs = [i for i, ins in enumerate(lin.g.program)
            if ins.op in (OP_REP, OP_CALL, OP_MIR)]
    if not idxs or not lin.g.macros:
        return lin
    h = lin.copy()
    i = idxs[int(rng.integers(0, len(idxs)))]
    ins = h.g.program[i]
    M = len(h.g.macros)
    if ins.op == OP_REP:
        if rng.random() < 0.5:
            n = min(cfg["nmax"], max(1, ins.a + int(rng.integers(-2, 3))))
            h.g.program[i] = Instr(OP_REP, n, ins.b)
        else:
            h.g.program[i] = Instr(OP_REP, ins.a, int(rng.integers(0, M)))
    else:
        h.g.program[i] = Instr(ins.op, int(rng.integers(0, M)))
    return h


def _remove_macro(lin: LinGenome, k: int) -> LinGenome:
    h = lin.copy()
    del h.g.macros[k]
    del h.meta[k]

    def fix(ins):
        if ins.op in (OP_CALL, OP_MIR):
            if ins.a == k:
                return None
            return Instr(ins.op, ins.a - 1 if ins.a > k else ins.a)
        if ins.op == OP_REP:
            if ins.b == k:
                return None
            return Instr(OP_REP, ins.a, ins.b - 1 if ins.b > k else ins.b)
        return ins

    new_macros = []
    for m in h.g.macros:
        nb = [fi for ins in m if (fi := fix(ins)) is not None]
        new_macros.append(nb if nb else [Instr(OP_LIT, 0)])
    h.g.macros = new_macros
    np_ = [fi for ins in h.g.program if (fi := fix(ins)) is not None]
    h.g.program = np_ if np_ else [Instr(OP_LIT, 0)]
    return h


GROWTH = [op_add, op_dup, op_reuse, op_ins_prog]
SHRINK = [op_delete_macro, op_del_prog]
NEUTRAL = [op_value, op_operand]


def mutate(rng, lin, world, cfg, alloc, allow_dup=True, allow_add=True):
    ops = list(NEUTRAL) + list(SHRINK)
    for o in GROWTH:
        if o is op_dup and not allow_dup:
            continue
        if o is op_add and not allow_add:
            continue
        ops.append(o)
    o = ops[int(rng.integers(0, len(ops)))]
    try:
        h = o(rng, lin, world, cfg, alloc)
        if (len(h.g.macros) > cfg["max_macros"]
                or len(h.g.program) > cfg["max_program"]):
            return lin.copy()
        G.validate(h.g)
        assert len(h.meta) == len(h.g.macros)
        return h
    except Exception:
        return lin.copy()


# ---------------------------------------------------------------------------
# knockout-based specialization
# ---------------------------------------------------------------------------
def knockout(lin: LinGenome, idx: int) -> Genome:
    """Neutralise macro idx (body -> zeros, same length) preserving alignment."""
    h = lin.g.copy()
    h.macros[idx] = [Instr(OP_LIT, 0) for _ in h.macros[idx]]
    return h


def specialization(lin: LinGenome, world: World) -> List[dict]:
    """For each macro: S(M_i, A) and S(M_i, B) = fitness drop on that region when
    the macro is knocked out."""
    base = G.expand(lin.g)
    tgt = world.target()
    fA = region_fitness(base, tgt, world.R_A)
    fB = region_fitness(base, tgt, world.R_B)
    out = []
    for i in range(len(lin.g.macros)):
        ko = G.expand(knockout(lin, i))
        sA = fA - region_fitness(ko, tgt, world.R_A)
        sB = fB - region_fitness(ko, tgt, world.R_B)
        out.append({"idx": i, "id": lin.meta[i].id, "parent": lin.meta[i].parent,
                    "S_A": sA, "S_B": sB})
    return out


# ---------------------------------------------------------------------------
# staged GA with lineage; returns population, per-stage records, adapt times
# ---------------------------------------------------------------------------
DEFAULT_CFG = dict(nmax=8, max_macros=8, max_macro_len=6, max_program=48)


def _obj(lin, target, world, functions, lam):
    F = fitness_active(G.expand(lin.g), target, world, functions)
    Lt = G.description_length(lin.g)
    return F, Lt, F - lam * Lt


def evolve_stage(pop, world, functions, lam, gens, pop_size, rng, cfg, alloc,
                 allow_dup=True, allow_add=True, threshold=None):
    """Evolve one stage; returns (pop, best_lin, T_reach) where T_reach is the
    first generation best-F >= threshold (None if never)."""
    target = world.target()
    scored = [(lin, *_obj(lin, target, world, functions, lam)) for lin in pop]
    T_reach = None
    for gen in range(gens):
        scored.sort(key=lambda t: t[3], reverse=True)
        bestF = max(t[1] for t in scored)
        if threshold is not None and T_reach is None and bestF >= threshold:
            T_reach = gen
        new = scored[:2]
        while len(new) < pop_size:
            idx = rng.integers(0, len(scored), size=4)
            par = max((scored[int(i)] for i in idx), key=lambda t: t[3])
            child = mutate(rng, par[0], world, cfg, alloc, allow_dup, allow_add)
            if rng.random() < 0.3:
                child = mutate(rng, child, world, cfg, alloc, allow_dup, allow_add)
            new.append((child, *_obj(child, target, world, functions, lam)))
        scored = new
    scored.sort(key=lambda t: (t[1], -t[2]), reverse=True)   # best F, then small L
    best = scored[0][0]
    return [t[0] for t in scored], best, T_reach
