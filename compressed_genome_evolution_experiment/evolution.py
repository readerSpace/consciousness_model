"""Genetic algorithm with four pluggable selection objectives (exp588).

Groups (mapping to the attachment's A/B/C/D table):

    A  fixed-length genome      J = F
    B  program genome           J = F - lambda * L(G)
    C  program genome (MDL)     J = F - lambda * (L(G) + L(D))
    D  program genome (MDL)     J = mean_e [ F_e - lambda*(L(G)+L(D)) ]
                                    over a rotating family of related targets
                                    ("future adaptability")

A is represented in the very same codec as an all-LIT program (pure
memorisation), so every group is measured on one ruler.  The only difference is
which mutation operators are allowed (A cannot grow structure) and the objective.

The key asymmetry the design turns on:

  * B penalises only the *program* section L(G).  A genome can therefore push all
    of its content into unpenalised macros (program = one CALL) -> tiny L(G),
    huge L(D), a macro invoked once.  "Trivially shorter genome."
  * C penalises the *total* description L(G)+L(D).  The only way to shrink that on
    a structured target is to find a small macro and call it many times.
    "Reusable rule."

Randomness flows through an explicit numpy Generator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np

import genome as G
from genome import Genome, Instr, OP_LIT, OP_CALL, OP_REP, OP_MIR
from environments import Environment, fitness


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------
@dataclass
class GAConfig:
    A: int = 4
    L: int = 32
    nmax: int = 8
    pop: int = 160
    generations: int = 250
    tournament: int = 4
    elitism: int = 2
    p_mut: float = 0.9            # prob of applying >=1 mutation to a child
    max_program: int = 40
    max_macros: int = 6
    max_macro_len: int = 8
    lam: float = 0.01             # lambda (compression pressure), bits scaled below
    # bit costs are ~ tens; F in [0,1]; scale lambda by 1/ bit_scale so lam~O(F)
    bit_scale: float = 1.0
    seed: int = 0


# ---------------------------------------------------------------------------
# individual bookkeeping
# ---------------------------------------------------------------------------
@dataclass
class Individual:
    g: Genome
    F: float = 0.0
    LG: int = 0
    LD: int = 0
    J: float = 0.0
    phen: Optional[List[int]] = None


# ---------------------------------------------------------------------------
# random genome construction
# ---------------------------------------------------------------------------
def random_literal_program(rng: np.random.Generator, cfg: GAConfig) -> Genome:
    """All-LIT program of length L (the group-A memorisation genome)."""
    prog = [Instr(OP_LIT, int(rng.integers(0, cfg.A))) for _ in range(cfg.L)]
    return Genome(macros=[], program=prog, A=cfg.A, L=cfg.L, nmax=cfg.nmax)


def random_structured_program(rng: np.random.Generator, cfg: GAConfig) -> Genome:
    """Small random program possibly with a couple of small macros (B/C/D init)."""
    n_macros = int(rng.integers(0, 3))
    macros: List[List[Instr]] = []
    for i in range(n_macros):
        macros.append(_random_macro_body(rng, cfg, max_macro=i))
    prog_len = int(rng.integers(1, cfg.L // 2 + 2))
    prog = [_random_instr(rng, cfg, max_macro=len(macros)) for _ in range(prog_len)]
    g = Genome(macros=macros, program=prog, A=cfg.A, L=cfg.L, nmax=cfg.nmax)
    return g


def _random_macro_body(rng, cfg, max_macro) -> List[Instr]:
    ln = int(rng.integers(1, cfg.max_macro_len + 1))
    return [_random_instr(rng, cfg, max_macro, allow_lit_only=(max_macro == 0))
            for _ in range(ln)]


def _random_instr(rng, cfg, max_macro, allow_lit_only=False) -> Instr:
    if allow_lit_only or max_macro == 0:
        return Instr(OP_LIT, int(rng.integers(0, cfg.A)))
    r = rng.random()
    if r < 0.6:
        return Instr(OP_LIT, int(rng.integers(0, cfg.A)))
    elif r < 0.8:
        return Instr(OP_CALL, int(rng.integers(0, max_macro)))
    elif r < 0.95:
        return Instr(OP_REP, int(rng.integers(1, cfg.nmax + 1)),
                     int(rng.integers(0, max_macro)))
    else:
        return Instr(OP_MIR, int(rng.integers(0, max_macro)))


# ---------------------------------------------------------------------------
# mutation operators
# ---------------------------------------------------------------------------
def mutate_literal(rng, cfg, g: Genome) -> Genome:
    """Group A: flip one LIT symbol; structure never changes."""
    h = g.copy()
    if not h.program:
        return h
    idx = int(rng.integers(0, len(h.program)))
    ins = h.program[idx]
    if ins.op == OP_LIT:
        h.program[idx] = Instr(OP_LIT, int(rng.integers(0, cfg.A)))
    return h


def mutate_structured(rng, cfg, g: Genome) -> Genome:
    """Group B/C/D: one of several structural edits."""
    h = g.copy()
    ops = ["sym", "operand", "ins_prog", "del_prog", "add_macro",
           "edit_macro", "promote", "demote"]
    op = ops[int(rng.integers(0, len(ops)))]
    try:
        if op == "sym":
            _mut_sym(rng, cfg, h)
        elif op == "operand":
            _mut_operand(rng, cfg, h)
        elif op == "ins_prog":
            _mut_insert_prog(rng, cfg, h)
        elif op == "del_prog":
            _mut_delete_prog(rng, cfg, h)
        elif op == "add_macro":
            _mut_add_macro(rng, cfg, h)
        elif op == "edit_macro":
            _mut_edit_macro(rng, cfg, h)
        elif op == "promote":
            _mut_promote(rng, cfg, h)
        elif op == "demote":
            _mut_demote(rng, cfg, h)
    except Exception:
        return g.copy()
    # guard sizes / validity
    if (len(h.program) > cfg.max_program or len(h.macros) > cfg.max_macros):
        return g.copy()
    try:
        G.validate(h)
    except Exception:
        return g.copy()
    return h


def _rand_prog_idx(rng, h):
    return int(rng.integers(0, len(h.program))) if h.program else None


def _mut_sym(rng, cfg, h):
    # mutate a LIT anywhere (program or macro)
    pools = [("p", i) for i, ins in enumerate(h.program) if ins.op == OP_LIT]
    for mi, m in enumerate(h.macros):
        pools += [("m", mi, i) for i, ins in enumerate(m) if ins.op == OP_LIT]
    if not pools:
        return
    choice = pools[int(rng.integers(0, len(pools)))]
    new = Instr(OP_LIT, int(rng.integers(0, cfg.A)))
    if choice[0] == "p":
        h.program[choice[1]] = new
    else:
        h.macros[choice[1]][choice[2]] = new


def _mut_operand(rng, cfg, h):
    # tweak a REP count or a macro-index operand somewhere in the program
    idxs = [i for i, ins in enumerate(h.program) if ins.op in (OP_REP, OP_CALL, OP_MIR)]
    if not idxs:
        return
    i = idxs[int(rng.integers(0, len(idxs)))]
    ins = h.program[i]
    M = len(h.macros)
    if M == 0:
        return
    if ins.op == OP_REP:
        if rng.random() < 0.5:
            n = min(cfg.nmax, max(1, ins.a + int(rng.integers(-2, 3))))
            h.program[i] = Instr(OP_REP, n, ins.b)
        else:
            h.program[i] = Instr(OP_REP, ins.a, int(rng.integers(0, M)))
    else:
        h.program[i] = Instr(ins.op, int(rng.integers(0, M)))


def _mut_insert_prog(rng, cfg, h):
    if len(h.program) >= cfg.max_program:
        return
    pos = int(rng.integers(0, len(h.program) + 1))
    h.program.insert(pos, _random_instr(rng, cfg, max_macro=len(h.macros)))


def _mut_delete_prog(rng, cfg, h):
    if len(h.program) <= 1:
        return
    pos = int(rng.integers(0, len(h.program)))
    del h.program[pos]


def _mut_add_macro(rng, cfg, h):
    if len(h.macros) >= cfg.max_macros:
        return
    body = _random_macro_body(rng, cfg, max_macro=len(h.macros))
    h.macros.append(body)


def _mut_edit_macro(rng, cfg, h):
    if not h.macros:
        return
    mi = int(rng.integers(0, len(h.macros)))
    m = h.macros[mi]
    action = int(rng.integers(0, 3))
    if action == 0 and len(m) < cfg.max_macro_len:      # insert
        pos = int(rng.integers(0, len(m) + 1))
        m.insert(pos, _random_instr(rng, cfg, max_macro=mi))
    elif action == 1 and len(m) > 1:                    # delete
        del m[int(rng.integers(0, len(m)))]
    else:                                               # replace
        if m:
            pos = int(rng.integers(0, len(m)))
            m[pos] = _random_instr(rng, cfg, max_macro=mi)


def _mut_promote(rng, cfg, h):
    """Insert a reference (CALL/REP) to an existing macro into the program.

    This is the operator that *lets* reuse appear; whether it survives is up to
    selection.  It does not copy any current best substring -- it only wires up a
    macro that already exists."""
    if not h.macros or len(h.program) >= cfg.max_program:
        return
    M = len(h.macros)
    k = int(rng.integers(0, M))
    pos = int(rng.integers(0, len(h.program) + 1))
    if rng.random() < 0.5:
        h.program.insert(pos, Instr(OP_REP, int(rng.integers(2, cfg.nmax + 1)), k))
    else:
        h.program.insert(pos, Instr(OP_CALL, k))


def _mut_demote(rng, cfg, h):
    """Replace a program reference with inlined LITs (opposite of promote)."""
    idxs = [i for i, ins in enumerate(h.program) if ins.op in (OP_CALL, OP_REP, OP_MIR)]
    if not idxs or len(h.program) >= cfg.max_program:
        return
    i = idxs[int(rng.integers(0, len(idxs)))]
    h.program[i] = Instr(OP_LIT, int(rng.integers(0, cfg.A)))


# ---------------------------------------------------------------------------
# objectives
# ---------------------------------------------------------------------------
def evaluate(ind: Individual, env: Environment, cfg: GAConfig, group: str) -> None:
    g = ind.g
    ind.phen = G.expand(g)
    ind.F = fitness(ind.phen, env)
    ind.LG = G.program_bits(g)
    ind.LD = G.macro_bits(g)
    lam = cfg.lam / cfg.bit_scale
    if group == "A":
        ind.J = ind.F
    elif group == "B":
        ind.J = ind.F - lam * ind.LG
    elif group in ("C", "D"):
        ind.J = ind.F - lam * (ind.LG + ind.LD)
    else:
        raise ValueError(group)


def evaluate_multi(ind: Individual, envs: List[Environment], cfg: GAConfig) -> None:
    """Group D: average objective over a family of related targets."""
    g = ind.g
    ind.phen = G.expand(g)
    ind.LG = G.program_bits(g)
    ind.LD = G.macro_bits(g)
    lam = cfg.lam / cfg.bit_scale
    Fs = [fitness(ind.phen, e) for e in envs]
    ind.F = float(np.mean(Fs))
    ind.J = ind.F - lam * (ind.LG + ind.LD)


# ---------------------------------------------------------------------------
# the GA
# ---------------------------------------------------------------------------
@dataclass
class RunResult:
    group: str
    best_by_F: Individual
    best_by_J: Individual
    history: dict            # per-generation traces
    evals: int
    final_pop: Optional[List[Individual]] = None


def _init_population(rng, cfg, group) -> List[Individual]:
    pop = []
    for _ in range(cfg.pop):
        if group == "A":
            g = random_literal_program(rng, cfg)
        else:
            g = random_structured_program(rng, cfg)
        pop.append(Individual(g=g))
    return pop


def _mutate(rng, cfg, g, group) -> Genome:
    if group == "A":
        return mutate_literal(rng, cfg, g)
    return mutate_structured(rng, cfg, g)


def _tournament_select(rng, pop, k) -> Individual:
    idx = rng.integers(0, len(pop), size=k)
    best = pop[int(idx[0])]
    for j in idx[1:]:
        if pop[int(j)].J > best.J:
            best = pop[int(j)]
    return best


def run_ga(cfg: GAConfig, env, group: str,
           envs_family: Optional[List[Environment]] = None,
           init_pop: Optional[List[Individual]] = None,
           rng: Optional[np.random.Generator] = None) -> RunResult:
    """Evolve one group against `env` (or a rotating `envs_family` for D).

    If `init_pop` is given the population is warm-started from it (re-evaluated
    against the new env) -- this is what the re-adaptation test uses."""
    if rng is None:
        rng = np.random.default_rng(cfg.seed)
    evals = 0

    def ev(ind):
        nonlocal evals
        if group == "D":
            evaluate_multi(ind, envs_family, cfg)
        else:
            evaluate(ind, env, cfg, group)
        evals += 1

    if init_pop is not None:
        pop = [Individual(g=i.g.copy()) for i in init_pop]
    else:
        pop = _init_population(rng, cfg, group)
    for ind in pop:
        ev(ind)

    hist = {"gen": [], "bestF": [], "meanF": [], "bestJ": [],
            "LG_atbestF": [], "LD_atbestF": [], "reuse_atbestF": [],
            "LG_atbestJ": [], "LD_atbestJ": [], "reuse_atbestJ": []}

    best_by_F = max(pop, key=lambda i: i.F)
    best_by_J = max(pop, key=lambda i: i.J)

    for gen in range(cfg.generations):
        pop.sort(key=lambda i: i.J, reverse=True)
        newpop = [pop[i] for i in range(cfg.elitism)]  # elitism carries objects
        while len(newpop) < cfg.pop:
            parent = _tournament_select(rng, pop, cfg.tournament)
            childg = parent.g.copy()
            if rng.random() < cfg.p_mut:
                childg = _mutate(rng, cfg, childg, group)
                # occasionally apply a second mutation for B/C/D
                if group != "A" and rng.random() < 0.3:
                    childg = _mutate(rng, cfg, childg, group)
            child = Individual(g=childg)
            ev(child)
            newpop.append(child)
        pop = newpop

        cur_bF = max(pop, key=lambda i: i.F)
        cur_bJ = max(pop, key=lambda i: i.J)
        if cur_bF.F > best_by_F.F:
            best_by_F = cur_bF
        if cur_bJ.J > best_by_J.J:
            best_by_J = cur_bJ

        hist["gen"].append(gen)
        hist["bestF"].append(cur_bF.F)
        hist["meanF"].append(float(np.mean([i.F for i in pop])))
        hist["bestJ"].append(cur_bJ.J)
        hist["LG_atbestF"].append(cur_bF.LG)
        hist["LD_atbestF"].append(cur_bF.LD)
        hist["reuse_atbestF"].append(G.reuse_rate(cur_bF.g))
        hist["LG_atbestJ"].append(cur_bJ.LG)
        hist["LD_atbestJ"].append(cur_bJ.LD)
        hist["reuse_atbestJ"].append(G.reuse_rate(cur_bJ.g))

    return RunResult(group=group, best_by_F=best_by_F, best_by_J=best_by_J,
                     history=hist, evals=evals, final_pop=pop)
