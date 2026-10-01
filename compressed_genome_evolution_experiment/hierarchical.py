"""exp597 -- Hierarchical Compositional Escalation (engine).

exp596 showed the WEAK arrow: external curriculum -> NOT NEEDED.  The required
minimal description C_t rises endogenously (coevolution) and plateaus when the
environment is frozen.  But the STRONG arrow failed at L<=16:

    endogenous complexity escalation  =/=>  compositional evolution
    (functional novelty / reuse / recombination / hierarchy did NOT appear)

exp596 also localised WHY: reusable macros need many generations on a (near-)
stable target (exp588 used ~220), but exp596's target moved every macro-step, so
no module ever formed.  exp597 turns that diagnosis into the central variable.

What changes vs exp596:
  * drop the exact-L* (L<=16) constraint from the MAIN run; use L = 32/64/128
    with a BOUNDED-MDL complexity estimate (upper bound U_t = shortest decodable
    genome found that reaches F>=f).  Exact L* is kept only for a small-L audit
    that checks U_t tracks L*.
  * the environment is explicitly HIERARCHICAL: atoms -> motifs -> modules
    (compositions of 2 motifs) -> target (a composition of modules).  It changes
    only every tau_E generations, and changes by small, reachable edits that
    REUSE persistent motifs and RECOMBINE them into new modules.
  * individuals may build macros that reference macros (hierarchy), and get a
    Re-Pair-style `factorize` refactoring operator so module formation is
    actually feasible (it only ever compresses the individual's OWN genome;
    it never reads the target).

The central question exp597 asks:

    does open-ended COMPOSITIONAL evolution require the environment-change
    timescale to match the representation-learning timescale, tau_E ~ tau_module?

Predicted phase transition:
    tau_E << tau_module : structure cannot form before the env changes
    tau_E ~  tau_module : maximal reuse + recombination + continuing novelty
    tau_E >> tau_module : over-specialisation (novelty stops once fitted)

Measured (not n_modules alone):
    D_hierarchy       macro-DAG depth (depth>=2 = macro-of-macro)
    R_reuse           reused module calls / all module calls
    N_recombination   macros composing >=2 distinct earlier macros
    N_functional_novelty(t)  functional-distance > eps AND knockout-confirmed,
                             split into COMPOSITION-derived vs DE-NOVO by lineage
    and crucially whether novelty CONTINUES in the late phase.

All randomness flows through explicit numpy Generators.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from statistics import mean, median
from typing import Dict, List, Optional, Tuple

import numpy as np

import genome as G
from genome import (Genome, Instr, OP_LIT, OP_CALL, OP_REP, OP_MIR,
                    expand, validate, description_length, macro_call_counts)
from evolution import GAConfig, Individual, mutate_structured, random_structured_program
from escalation import (_match, _expanded_macros, _dfunc, _knockout_fitness,
                        _is_recombined, _referenced_macros)


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------
@dataclass
class HConfig:
    A: int = 2
    L: int = 32
    nmax: int = 8

    pop: int = 60
    tournament: int = 4
    elitism: int = 2
    p_mut: float = 0.9
    p_fact: float = 0.5          # fraction of mutations that are factorisations
    lam: float = 0.001           # MDL pressure per bit (small: compress, don't crush F)
    bit_scale: float = 1.0
    K_max: int = 160             # generous: literals FIT, MDL still rewards reuse
    max_program: int = 90
    max_macros: int = 24
    max_macro_len: int = 14

    # environment hierarchy + timescale
    tau_E: int = 100             # generations between environment changes
    env_changes: int = 8         # number of env changes (total_gens = tau_E*changes)
    n_motifs0: int = 3
    motif_len: int = 3
    n_modules0: int = 3

    f_solve: float = 0.9
    novelty_eps: float = 0.25
    knockout_drop: float = 0.02
    record_every: int = 10
    seed: int = 0

    # controls
    use_mdl: bool = True
    allow_factorize: bool = True
    freeze_env: bool = False


def _gacfg(cfg: HConfig) -> GAConfig:
    return GAConfig(A=cfg.A, L=cfg.L, nmax=cfg.nmax, pop=cfg.pop,
                    generations=1, tournament=cfg.tournament,
                    elitism=cfg.elitism, p_mut=cfg.p_mut,
                    max_program=cfg.max_program, max_macros=cfg.max_macros,
                    max_macro_len=cfg.max_macro_len, lam=cfg.lam,
                    bit_scale=cfg.bit_scale, seed=cfg.seed)


# ---------------------------------------------------------------------------
# hierarchical environment: atoms -> motifs -> modules -> target
# ---------------------------------------------------------------------------
@dataclass
class HEnv:
    motifs: List[List[int]]              # pool of motifs (atom strings)
    modules: List[Tuple[int, int]]       # each module = an ordered pair of motifs
    order: List[int]                     # sequence of module indices tiled to L
    A: int
    L: int

    def target(self) -> List[int]:
        seq: List[int] = []
        for mi in self.order:
            a, b = self.modules[mi]
            seq += self.motifs[a] + self.motifs[b]
        if not seq:
            seq = [0]
        reps = (self.L // len(seq)) + 1
        return (seq * reps)[: self.L]

    def copy(self) -> "HEnv":
        return HEnv([list(m) for m in self.motifs],
                    [tuple(md) for md in self.modules],
                    list(self.order), self.A, self.L)


def seed_env(cfg: HConfig, rng: np.random.Generator) -> HEnv:
    # SMALL motif pool reused across DISTINCT 2-motif modules, each module
    # REPEATED in the composition.  This gives genuine multi-level reuse (motifs
    # recur across modules; modules recur in the composition) and makes a module
    # = composition of two distinct motifs the dominant compressible unit, so
    # recombination (a macro referencing two different motif-macros) can pay off.
    motifs = [rng.integers(0, cfg.A, size=cfg.motif_len).tolist()
              for _ in range(cfg.n_motifs0)]
    modules = [(i % cfg.n_motifs0, (i + 2) % cfg.n_motifs0)
               for i in range(cfg.n_modules0)]
    order: List[int] = []
    for mi in range(cfg.n_modules0):
        order += [mi, mi]                           # each module repeated
    return HEnv(motifs, modules, order, cfg.A, cfg.L)


def env_step(env: HEnv, cfg: HConfig, rng: np.random.Generator) -> HEnv:
    """One small, reachable edit that keeps the hierarchy but is novel.

    The edits deliberately REUSE persistent motifs and RECOMBINE them into new
    modules -- the structure an individual that kept a module library can exploit,
    and the reason recombination (not just de-novo) is possible."""
    if cfg.freeze_env:
        return env
    h = env.copy()
    kind = rng.integers(0, 4)
    if kind == 0:                                   # motif point mutation (atoms)
        mi = int(rng.integers(0, len(h.motifs)))
        pos = int(rng.integers(0, len(h.motifs[mi])))
        h.motifs[mi][pos] = int(rng.integers(0, cfg.A))
    elif kind == 1:                                 # RECOMBINE: new module from
        mi = int(rng.integers(0, len(h.modules)))   # existing motifs
        a = int(rng.integers(0, len(h.motifs)))
        b = int(rng.integers(0, len(h.motifs)))
        h.modules[mi] = (a, b)
    elif kind == 2 and len(h.motifs) < 6:           # DE-NOVO: add motif + module
        h.motifs.append(rng.integers(0, cfg.A, size=cfg.motif_len).tolist())
        a = len(h.motifs) - 1
        b = int(rng.integers(0, len(h.motifs)))
        h.modules.append((a, b))
        h.order.append(len(h.modules) - 1)
    else:                                           # reorder composition
        if len(h.order) >= 2:
            i = int(rng.integers(0, len(h.order)))
            j = int(rng.integers(0, len(h.order)))
            h.order[i], h.order[j] = h.order[j], h.order[i]
    return h


# ---------------------------------------------------------------------------
# Re-Pair-style factorisation: compress the individual's OWN genome
# ---------------------------------------------------------------------------
def factorize(g: Genome, cfg: HConfig) -> Genome:
    """Replace the most compressive repeated substring of the program by a new
    macro and CALLs to it.  Phenotype-preserving; only ever shortens the genome.
    Extracting WHOLE repeated units (e.g. a motif, or a module = two distinct
    motif-calls) makes the macro body reference >=2 earlier macros -> hierarchy
    depth and recombination arise endogenously from compression.

    Savings = occ*(len-1) favours a whole repeated unit over its fragments, so
    motifs factor as clean macros first and modules compose them afterwards."""
    if len(g.macros) >= cfg.max_macros or len(g.program) < 4:
        return g
    prog = g.program
    n = len(prog)
    maxlen = min(cfg.max_macro_len, n // 2)
    best_key = None
    best_rank = (1, 0)                               # (count, length), both maximised
    for length in range(2, maxlen + 1):
        occ: Dict[tuple, List[int]] = {}
        for s in range(0, n - length + 1):
            occ.setdefault(tuple(prog[s:s + length]), []).append(s)
        for key, positions in occ.items():
            cnt = 0
            last_end = -1
            for p in positions:                     # greedy non-overlapping count
                if p >= last_end:
                    cnt += 1
                    last_end = p + length
            if cnt >= 2 and (cnt, length) > best_rank:
                best_rank = (cnt, length)           # most FREQUENT unit first
                best_key = key                       # (motifs before modules) ->
    if best_key is None:                             # clean bottom-up hierarchy
        return g
    h = g.copy()
    new_idx = len(h.macros)
    h.macros.append(list(best_key))                 # references only earlier macros
    klen = len(best_key)
    newprog: List[Instr] = []
    i = 0
    while i < len(prog):
        if tuple(prog[i:i + klen]) == best_key:
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


def factorize_full(g: Genome, cfg: HConfig, passes: int = 6) -> Genome:
    """Bottom-up Re-Pair: repeatedly factorise the most frequent pair, building a
    hierarchy of macros-of-macros.  Kept only insofar as each pass validates."""
    h = g
    for _ in range(passes):
        nh = factorize(h, cfg)
        if nh is h or (len(nh.program) == len(h.program)
                       and len(nh.macros) == len(h.macros)):
            break
        h = nh
    return h


# ---------------------------------------------------------------------------
# structural metrics: hierarchy depth, reuse rate, recombination
# ---------------------------------------------------------------------------
def _macro_depths(g: Genome) -> List[int]:
    depths = [1] * len(g.macros)
    for i in range(len(g.macros)):                  # macros reference only j<i
        refs = _referenced_macros(g.macros[i])
        if refs:
            depths[i] = 1 + max(depths[j] for j in refs)
    return depths


def hierarchy_depth(g: Genome, used: List[int]) -> int:
    if not used:
        return 0
    depths = _macro_depths(g)
    return max(depths[i] for i in used)


def reuse_rate_g(g: Genome) -> float:
    counts = macro_call_counts(g)
    total = sum(counts)
    if total == 0:
        return 0.0
    reused = sum(c for c in counts if c >= 2)
    return reused / total


# ---------------------------------------------------------------------------
# champion metrics (structure + functional novelty, with composition lineage)
# ---------------------------------------------------------------------------
def champion_metrics(g: Genome, target: List[int], history: List[tuple],
                     cfg: HConfig) -> dict:
    counts = macro_call_counts(g)
    exps = _expanded_macros(g)
    used = [i for i in range(len(g.macros)) if counts[i] >= 1 and len(exps[i]) >= 2]
    baseF = _match(expand(g), target)

    n_novel = n_comp = n_denovo = 0
    novel_sigs: List[tuple] = []
    for i in used:
        sig = tuple(exps[i])
        dmin = min((_dfunc(sig, h) for h in history), default=1.0)
        if dmin > cfg.novelty_eps:
            ko = _knockout_fitness(g, i, target)
            if baseF - ko >= cfg.knockout_drop:
                n_novel += 1
                novel_sigs.append(sig)
                if _is_recombined(g, i):            # built from >=2 earlier modules
                    n_comp += 1
                else:
                    n_denovo += 1
    return {
        "F": baseF,
        "K": description_length(g),
        "n_modules": len(used),
        "hierarchy_depth": hierarchy_depth(g, used),
        "reuse_rate": reuse_rate_g(g),
        "n_recomb": sum(1 for i in used if _is_recombined(g, i)),
        "n_novel": n_novel,
        "n_comp_novel": n_comp,
        "n_denovo_novel": n_denovo,
        "used_sigs": [tuple(exps[i]) for i in used],
        "novel_sigs": novel_sigs,
    }


# ---------------------------------------------------------------------------
# inner GA (one generation) with factorisation
# ---------------------------------------------------------------------------
def _eval(ind: Individual, target: List[int], cfg: HConfig) -> None:
    ind.phen = expand(ind.g)
    ind.F = _match(ind.phen, target)
    K = description_length(ind.g)
    ind.LG = K
    lam = (cfg.lam / cfg.bit_scale) if cfg.use_mdl else 0.0
    J = ind.F - lam * K
    if K > cfg.K_max:
        J -= 1.0 * (K - cfg.K_max)
    ind.J = J


def _mutate(rng, gacfg, g: Genome, cfg: HConfig) -> Genome:
    if cfg.allow_factorize and rng.random() < cfg.p_fact:
        return factorize_full(g, cfg)
    return mutate_structured(rng, gacfg, g)


def _tournament(rng, pop, k) -> Individual:
    idx = rng.integers(0, len(pop), size=k)
    best = pop[int(idx[0])]
    for j in idx[1:]:
        if pop[int(j)].J > best.J:
            best = pop[int(j)]
    return best


def _one_generation(pop, target, cfg, gacfg, rng):
    for ind in pop:
        if ind.phen is None:
            _eval(ind, target, cfg)
    pop.sort(key=lambda i: i.J, reverse=True)
    newpop = [pop[i] for i in range(cfg.elitism)]
    while len(newpop) < cfg.pop:
        parent = _tournament(rng, pop, cfg.tournament)
        childg = parent.g.copy()
        if rng.random() < cfg.p_mut:
            childg = _mutate(rng, gacfg, childg, cfg)
            if rng.random() < 0.3:
                childg = _mutate(rng, gacfg, childg, cfg)
        child = Individual(g=childg)
        _eval(child, target, cfg)
        newpop.append(child)
    return newpop


def _reeval_pop(pop, target, cfg):
    for ind in pop:
        _eval(ind, target, cfg)


def _upper_bound_complexity(pop, cfg) -> Tuple[int, float]:
    """U_t = shortest decodable genome in the pop that reaches F>=f_solve (a
    rigorous UPPER bound on L*(F>=f); exact L* is validated separately at L=16)."""
    solved = [ind for ind in pop if ind.F >= cfg.f_solve]
    if solved:
        u = min(description_length(ind.g) for ind in solved)
        return u, 1.0
    return min(description_length(ind.g) for ind in pop), 0.0


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------
def run_hierarchical(cfg: HConfig) -> dict:
    rng = np.random.default_rng(cfg.seed)
    gacfg = _gacfg(cfg)
    env = seed_env(cfg, rng)
    pop = [Individual(g=random_structured_program(rng, gacfg))
           for _ in range(cfg.pop)]
    target = env.target()
    _reeval_pop(pop, target, cfg)

    history: List[tuple] = []
    seen: set = set()
    total_gens = cfg.tau_E * cfg.env_changes
    series: Dict[str, list] = {k: [] for k in (
        "gen", "F", "U_t", "solved_frac", "n_modules", "hierarchy_depth",
        "reuse_rate", "n_recomb", "n_novel", "cum_novel",
        "n_comp_novel", "n_denovo_novel")}
    cum_novel = 0
    first_module_gen: Optional[int] = None

    for gen in range(total_gens):
        pop = _one_generation(pop, target, cfg, gacfg, rng)

        if gen % cfg.record_every == 0 or gen == total_gens - 1:
            champ = max(pop, key=lambda i: i.J)
            mm = champion_metrics(champ.g, target, history, cfg)
            cum_novel += mm["n_novel"]
            u, sf = _upper_bound_complexity(pop, cfg)
            if first_module_gen is None and mm["n_modules"] > 0 and mm["reuse_rate"] > 0:
                first_module_gen = gen
            series["gen"].append(gen)
            series["F"].append(mm["F"])
            series["U_t"].append(u)
            series["solved_frac"].append(sf)
            series["n_modules"].append(mm["n_modules"])
            series["hierarchy_depth"].append(mm["hierarchy_depth"])
            series["reuse_rate"].append(mm["reuse_rate"])
            series["n_recomb"].append(mm["n_recomb"])
            series["n_novel"].append(mm["n_novel"])
            series["cum_novel"].append(cum_novel)
            series["n_comp_novel"].append(mm["n_comp_novel"])
            series["n_denovo_novel"].append(mm["n_denovo_novel"])
            for sig in mm["used_sigs"]:
                if sig not in seen:
                    seen.add(sig)
                    history.append(sig)

        if (gen + 1) % cfg.tau_E == 0:              # environment changes
            env = env_step(env, cfg, rng)
            target = env.target()
            _reeval_pop(pop, target, cfg)

    return {
        "series": series,
        "summary": _summarise(series, first_module_gen, cfg),
        "config": {k: getattr(cfg, k) for k in (
            "A", "L", "pop", "tau_E", "env_changes", "lam", "K_max",
            "use_mdl", "allow_factorize", "freeze_env", "seed")},
    }


def _summarise(series: Dict[str, list], first_module_gen, cfg: HConfig) -> dict:
    n = len(series["gen"])
    half = max(1, n // 2)
    nov = series["n_novel"]
    return {
        "max_hierarchy_depth": max(series["hierarchy_depth"]) if n else 0,
        "reuse_rate_max": max(series["reuse_rate"]) if n else 0.0,
        "reuse_rate_late": mean(series["reuse_rate"][half:]) if n else 0.0,
        "reuse_any": any(r > 0 for r in series["reuse_rate"]),
        "recomb_any": any(x > 0 for x in series["n_recomb"]),
        "n_recomb_total": sum(series["n_recomb"]),
        "cum_novel": series["cum_novel"][-1] if n else 0,
        "novel_first_half": sum(nov[:half]),
        "novel_second_half": sum(nov[half:]),
        "comp_novel_total": sum(series["n_comp_novel"]),
        "denovo_novel_total": sum(series["n_denovo_novel"]),
        "U_early": mean(series["U_t"][:half]) if n else 0.0,
        "U_late": mean(series["U_t"][half:]) if n else 0.0,
        "F_mean": mean(series["F"]) if n else 0.0,
        "first_module_gen": first_module_gen,
    }


# ---------------------------------------------------------------------------
# tau_module calibration: generations to first reused module on a FIXED target
# ---------------------------------------------------------------------------
def tau_module(cfg: HConfig, seeds: List[int], max_gens: int = 400) -> Optional[float]:
    vals = []
    for s in seeds:
        rng = np.random.default_rng(1000 + s)
        gacfg = _gacfg(cfg)
        env = seed_env(cfg, rng)
        target = env.target()
        pop = [Individual(g=random_structured_program(rng, gacfg))
               for _ in range(cfg.pop)]
        _reeval_pop(pop, target, cfg)
        found = None
        for gen in range(max_gens):
            pop = _one_generation(pop, target, cfg, gacfg, rng)
            champ = max(pop, key=lambda i: i.J)
            # a MEANINGFUL reused module: the champion both reuses a macro AND
            # actually fits the target (excludes trivial reuse in random genomes)
            if reuse_rate_g(champ.g) > 0 and champ.F >= 0.6:
                found = gen
                break
        if found is not None:
            vals.append(found)
    return float(median(vals)) if vals else None
