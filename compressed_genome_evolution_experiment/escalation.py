"""exp596 -- Endogenous Complexity Escalation (co-evolution engine).

exp588-595 all handed the learner a structure we designed (a target, an
interaction partition, a sequence of partitions) and asked whether it could be
*discovered* / *reused*.  exp596 removes the designer from the loop: the
individual genome ``G`` AND the environment / task ``E`` both evolve,

    (G_t, E_t) -> (G_{t+1}, E_{t+1}),

and we ask whether the *reason to need new structure* is generated from inside
the system -- i.e. whether the minimal description length actually required to
solve the environment keeps rising on its own.

Design (straight from the note):

  individual objective   J_G = F(G,E) - lam * L(G)                 (MDL pressure)
  environment objective  J_E = D(E,G) - alpha * L(E) - beta * Impossible(E)

  D(E,G): high for tasks that are HARD for the current individual but REACHABLE
          from nearby mutations -- a zone-of-proximal-development term.  We make
          it explicit and measurable:
              F0    = best fitness of the current population on E
              reach = extra fitness a few mutations of the champion can reach
              D     = (1 - F0) * reach            # 0 if solved, 0 if stuck
  L(E):   the description length of the environment (E is itself a genome in the
          exp588 codec, so L(E) is a real, decodable bit length).
  Impossible(E) = 1 if L*(F>=f | E) exceeds the individual capacity K_max
          (the task needs more bits than an individual can hold).

Both G and E are Genomes in the exp588 codec, so every "length" here is a
genuine decodable bit length and every "environment mutation" is a step in the
same neighbourhood the individuals live in.

We do NOT declare "open-ended evolution".  We measure the weaker, auditable
claim of *endogenous complexity escalation* -- the quantity that must rise first:

    C_t = L*(F >= f | E_t)        (exp589's EXACT minimal description, A=2 world)

"the genome got longer" (L(G)) is deliberately NOT the headline; the headline is
"the minimal description the ENVIRONMENT requires went up", tracked alongside a
basket of structural measures:

    L(G), K(G)=L(G)+L(D), N_modules, N_interactions,
    N_novel_modules, N_reused, N_recombined.

Controls (built as flags on CoevoConfig):
    coevolution   : the full system
    frozen        : E never changes           -> C_t should plateau
    no-MDL        : lam = 0 (no length pressure) -> redundant bloat (K >> C_t)
    no-recomb     : forbid modules built from >=2 earlier modules -> novelty stalls
    random-env    : E_{t+1} drawn at random (J_E ignored) -> adaptation failure

All randomness flows through explicit numpy Generators (never Python hash()).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean
from typing import Dict, List, Optional, Tuple

import numpy as np

import genome as G
from genome import (Genome, Instr, OP_LIT, OP_CALL, OP_REP, OP_MIR,
                    expand, validate, description_length, program_bits,
                    macro_call_counts, _expand_instrs)
from evolution import (GAConfig, Individual, random_structured_program,
                       mutate_structured)
import optimal as O


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------
@dataclass
class CoevoConfig:
    A: int = 2
    L: int = 16
    nmax: int = 8

    # inner genome GA (per macro-step, population is CARRIED across steps)
    pop: int = 40
    inner_gens: int = 8          # enough to DISCOVER reusable macros each step
    tournament: int = 4
    elitism: int = 2
    p_mut: float = 0.9
    lam: float = 0.006           # genome MDL pressure per bit (scaled by bit_scale)
    bit_scale: float = 1.0
    K_max: int = 24              # individual capacity: max description length (bits)
    max_program: int = 40
    max_macros: int = 6
    max_macro_len: int = 8

    # environment evolution
    macro_steps: int = 40
    n_env_cands: int = 6
    env_mut_steps: int = 2
    env_K_max: int = 48          # keep targets inside the small-world C* regime
    alpha: float = 0.001         # L(E) penalty per bit (kept below the D scale)
    beta: float = 0.6            # Impossible(E) penalty
    probe: int = 12              # reachability probe count
    probe_steps: int = 2         # mutations per probe
    mc_reach: float = 0.5        # minimal criterion: champion must reach >= this
    reach_bonus: float = 1.0     # weight on compositional reachability in J_E
    archive_cap: int = 60        # max distinct past targets kept for novelty
    f_solve: float = 0.9         # threshold defining "solved" (for C_t and Impossible)

    # functional novelty
    novelty_eps: float = 0.15    # normalised functional distance to count as novel
    knockout_drop: float = 0.005  # min fitness drop on knockout to be load-bearing

    seed: int = 0

    # control flags
    freeze_env: bool = False
    use_mdl: bool = True
    allow_recomb: bool = True
    random_env: bool = False


def _gacfg(cfg: CoevoConfig) -> GAConfig:
    return GAConfig(A=cfg.A, L=cfg.L, nmax=cfg.nmax, pop=cfg.pop,
                    generations=cfg.inner_gens, tournament=cfg.tournament,
                    elitism=cfg.elitism, p_mut=cfg.p_mut,
                    max_program=cfg.max_program, max_macros=cfg.max_macros,
                    max_macro_len=cfg.max_macro_len, lam=cfg.lam,
                    bit_scale=cfg.bit_scale, seed=cfg.seed)


# ---------------------------------------------------------------------------
# fitness between two phenotypes (fraction of matching positions)
# ---------------------------------------------------------------------------
def _match(phen: List[int], target: List[int]) -> float:
    n = len(target)
    m = 0
    for i in range(n):
        if phen[i] == target[i]:
            m += 1
    return m / n


# ---------------------------------------------------------------------------
# module (macro) structure helpers
# ---------------------------------------------------------------------------
def _expanded_macros(g: Genome) -> List[List[int]]:
    exp: List[List[int]] = []
    for m in g.macros:
        exp.append(_expand_instrs(m, exp, g.L))
    return exp


def _referenced_macros(instrs: List[Instr]) -> set:
    s = set()
    for ins in instrs:
        if ins.op in (OP_CALL, OP_MIR):
            s.add(ins.a)
        elif ins.op == OP_REP:
            s.add(ins.b)
    return s


def _is_recombined(g: Genome, i: int) -> bool:
    """A macro is 'recombined' if its body composes >=2 distinct earlier modules."""
    return len(_referenced_macros(g.macros[i])) >= 2


def _has_recomb(g: Genome) -> bool:
    return any(_is_recombined(g, i) for i in range(len(g.macros)))


def _count_refs(g: Genome) -> int:
    """N_interactions: reference instructions (CALL/REP/MIR) in program + macros."""
    n = 0
    for ins in g.program:
        if ins.op in (OP_CALL, OP_REP, OP_MIR):
            n += 1
    for m in g.macros:
        for ins in m:
            if ins.op in (OP_CALL, OP_REP, OP_MIR):
                n += 1
    return n


def _dfunc(a: Tuple[int, ...], b: Tuple[int, ...]) -> float:
    """Normalised functional distance between two module expansion strings."""
    la, lb = len(a), len(b)
    m = max(la, lb)
    if m == 0:
        return 0.0
    n = min(la, lb)
    ham = sum(1 for i in range(n) if a[i] != b[i]) + (m - n)
    return ham / m


def _knockout_fitness(g: Genome, i: int, target: List[int]) -> float:
    """Fitness after disconnecting macro i (references -> neutral LIT 0)."""
    h = g.copy()

    def strip(instrs):
        out = []
        for ins in instrs:
            if (ins.op in (OP_CALL, OP_MIR) and ins.a == i) or \
               (ins.op == OP_REP and ins.b == i):
                out.append(Instr(OP_LIT, 0))
            else:
                out.append(ins)
        return out

    h.program = strip(h.program)
    for j in range(len(h.macros)):
        h.macros[j] = strip(h.macros[j])
    try:
        phen = expand(h)
    except Exception:
        return 0.0
    return _match(phen, target)


def module_metrics(g: Genome, target: List[int],
                   history: List[Tuple[int, ...]], cfg: CoevoConfig) -> dict:
    """Structural + functional-novelty accounting for one champion genome.

    history is the library of module expansion strings seen in EARLIER steps;
    a module counts as 'functional novelty' only if it is both functionally far
    from everything in history AND load-bearing (knockout lowers fitness)."""
    counts = macro_call_counts(g)
    exps = _expanded_macros(g)
    used = [i for i in range(len(g.macros)) if counts[i] >= 1 and len(exps[i]) >= 2]

    baseF = _match(expand(g), target)
    n_novel = 0
    novel_sigs: List[Tuple[int, ...]] = []
    for i in used:
        sig = tuple(exps[i])
        dmin = min((_dfunc(sig, h) for h in history), default=1.0)
        if dmin > cfg.novelty_eps:
            ko = _knockout_fitness(g, i, target)
            if baseF - ko >= cfg.knockout_drop:
                n_novel += 1
                novel_sigs.append(sig)

    return {
        "N_modules": len(used),
        "N_interactions": _count_refs(g),
        "N_reused": sum(1 for i in used if counts[i] >= 2),
        "N_recombined": sum(1 for i in used if _is_recombined(g, i)),
        "N_novel": n_novel,
        "novel_sigs": novel_sigs,
        "used_sigs": [tuple(exps[i]) for i in used],
    }


# ---------------------------------------------------------------------------
# C_t = L*(F >= f | E)   (exact, from exp589's optimal.py), memoised by target
# ---------------------------------------------------------------------------
_CSTAR_CACHE: Dict[Tuple, int] = {}


def cstar(target: List[int], cfg: CoevoConfig,
          mlen: int = 6, Mmax: int = 2) -> int:
    """Exact minimal description length achieving fitness >= f_solve on `target`.

    Only phenotypes within Hamming radius floor((1-f)*L) of the target can reach
    F >= f, so we audit just that ball (cheap) and take L*(F>=f)."""
    key = (tuple(target), cfg.A, cfg.L, cfg.f_solve, cfg.nmax, mlen, Mmax)
    cached = _CSTAR_CACHE.get(key)
    if cached is not None:
        return cached
    r = int((1.0 - cfg.f_solve) * cfg.L)
    audit = O.audit_target(target, cfg.A, mlen=mlen, Mmax=Mmax, nmax=cfg.nmax,
                           f_min=cfg.f_solve, enumerate_full=False,
                           hamming_radius=r)
    v = O.lstar_at_least(audit, cfg.f_solve)
    if v is None:
        v = O.kx(tuple(target), cfg.A, cfg.L, mlen=mlen, Mmax=Mmax, nmax=cfg.nmax)
    v = int(v)
    _CSTAR_CACHE[key] = v
    return v


# ---------------------------------------------------------------------------
# genome evaluation under capacity + optional MDL pressure
# ---------------------------------------------------------------------------
def _eval_ind(ind: Individual, target: List[int], cfg: CoevoConfig) -> None:
    g = ind.g
    ind.phen = expand(g)
    ind.F = _match(ind.phen, target)
    ind.LG = program_bits(g)
    ind.LD = G.macro_bits(g)
    K = ind.LG + ind.LD
    lam = (cfg.lam / cfg.bit_scale) if cfg.use_mdl else 0.0
    J = ind.F - lam * K
    if K > cfg.K_max:                       # hard capacity: over-budget is lethal
        J -= 1.0 * (K - cfg.K_max)
    ind.J = J


# ---------------------------------------------------------------------------
# mutation respecting the no-recombination control
# ---------------------------------------------------------------------------
def _mutate_ind(rng, gacfg, g: Genome, cfg: CoevoConfig) -> Genome:
    h = mutate_structured(rng, gacfg, g)
    if not cfg.allow_recomb and _has_recomb(h):
        return g.copy()
    return h


def _fresh_individual(rng, gacfg, cfg: CoevoConfig) -> Individual:
    for _ in range(6):
        g = random_structured_program(rng, gacfg)
        if cfg.allow_recomb or not _has_recomb(g):
            return Individual(g=g)
    # fallback: a literal program never recombines
    prog = [Instr(OP_LIT, int(rng.integers(0, cfg.A))) for _ in range(cfg.L // 2)]
    return Individual(g=Genome(macros=[], program=prog, A=cfg.A, L=cfg.L,
                               nmax=cfg.nmax))


def _tournament(rng, pop, k) -> Individual:
    idx = rng.integers(0, len(pop), size=k)
    best = pop[int(idx[0])]
    for j in idx[1:]:
        if pop[int(j)].J > best.J:
            best = pop[int(j)]
    return best


# ---------------------------------------------------------------------------
# inner genome GA against a fixed environment target (warm-started population)
# ---------------------------------------------------------------------------
def _evolve_genomes(pop: List[Individual], target: List[int],
                    cfg: CoevoConfig, gacfg: GAConfig, rng) -> List[Individual]:
    for ind in pop:
        _eval_ind(ind, target, cfg)
    for _ in range(cfg.inner_gens):
        pop.sort(key=lambda i: i.J, reverse=True)
        newpop = [pop[i] for i in range(cfg.elitism)]
        while len(newpop) < cfg.pop:
            parent = _tournament(rng, pop, cfg.tournament)
            childg = parent.g.copy()
            if rng.random() < cfg.p_mut:
                childg = _mutate_ind(rng, gacfg, childg, cfg)
                if rng.random() < 0.3:
                    childg = _mutate_ind(rng, gacfg, childg, cfg)
            child = Individual(g=childg)
            _eval_ind(child, target, cfg)
            newpop.append(child)
        pop = newpop
    return pop


# ---------------------------------------------------------------------------
# environment step: pick E_{t+1} maximising J_E (or freeze / randomise)
# ---------------------------------------------------------------------------
def _seed_env(cfg: CoevoConfig) -> Genome:
    """Start at the SIMPLEST end (all-zeros, C ~ 2 bits) so escalation is a RISE.

    An empty-effect program padded to length L yields all zeros, the cheapest
    possible target; every subsequent bit of required complexity is endogenous."""
    g = Genome(macros=[], program=[Instr(OP_LIT, 0)], A=cfg.A, L=cfg.L,
               nmax=cfg.nmax)
    validate(g)
    return g


def _random_env(rng, gacfg, cfg: CoevoConfig) -> Genome:
    for _ in range(10):
        g = random_structured_program(rng, gacfg)
        if description_length(g) <= cfg.env_K_max:
            return g
    return _seed_env(cfg)


def _hamming_norm(a: List[int], b: List[int]) -> float:
    n = len(a)
    return sum(1 for i in range(n) if a[i] != b[i]) / n if n else 0.0


def _env_objective(env_g: Genome, champ: Individual, archive: List[tuple],
                   cfg: CoevoConfig, gacfg: GAConfig, rng) -> float:
    """Minimal-Criterion Coevolution objective.

    D(E,G) rewards a task that is REACHABLE (the champion, lightly mutated, can
    still do decently -- excludes pure noise and keeps the challenge STRUCTURED)
    and NOVEL relative to the archive of already-mastered targets.  Pure hardness
    would let the environment cycle between all-0 and all-1 (both C=2) forever;
    the archive forces it, once simple structured tasks are exhausted, to reach
    for MORE COMPLEX structure -- the endogenous complexity ratchet."""
    target = expand(env_g)
    F0 = _match(expand(champ.g), target)
    best_probe = F0
    for _ in range(cfg.probe):
        gm = champ.g.copy()
        for _ in range(cfg.probe_steps):
            gm = _mutate_ind(rng, gacfg, gm, cfg)
        best_probe = max(best_probe, _match(expand(gm), target))
    reachable_F = max(F0, best_probe)
    if reachable_F < cfg.mc_reach:                 # fails the minimal criterion
        return -1e9
    # compositional reachability: a single macro-wiring mutation that fixes many
    # positions signals REPEATED/MOTIF structure (the kind that rewards reusable
    # modules), as opposed to flat literal/run content.
    reach_frac = (best_probe - F0) / (1.0 - F0 + 1e-9)
    reach_frac = min(1.0, max(0.0, reach_frac))
    novelty = min((_hamming_norm(target, a) for a in archive), default=1.0)
    LE = description_length(env_g)
    impossible = 1.0 if cstar(target, cfg) > cfg.K_max else 0.0
    return (novelty + cfg.reach_bonus * reach_frac
            - cfg.alpha * LE - cfg.beta * impossible)


def _env_step(env_g: Genome, champ: Individual, archive: List[tuple],
              cfg: CoevoConfig, gacfg: GAConfig, rng) -> Genome:
    if cfg.freeze_env:
        return env_g
    if cfg.random_env:
        return _random_env(rng, gacfg, cfg)
    best, bestJ = env_g, _env_objective(env_g, champ, archive, cfg, gacfg, rng)
    for _ in range(cfg.n_env_cands):
        cand = env_g.copy()
        for _ in range(cfg.env_mut_steps):
            cand = mutate_structured(rng, gacfg, cand)
        if description_length(cand) > cfg.env_K_max:
            continue
        JE = _env_objective(cand, champ, archive, cfg, gacfg, rng)
        if JE > bestJ:
            best, bestJ = cand, JE
    return best


# ---------------------------------------------------------------------------
# the co-evolution run
# ---------------------------------------------------------------------------
def _slope(xs: List[float], ys: List[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mx, my = mean(xs), mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    return num / den if den else 0.0


def _summarise(series: Dict[str, list], cfg: CoevoConfig) -> dict:
    t = series["t"]
    C = series["C_t"]
    K = series["K"]
    F = series["F"]
    n = len(t)
    half = max(1, n // 2)
    C_early = mean(C[:half])
    C_late = mean(C[half:])
    gap_late = mean([k - c for k, c in zip(K[half:], C[half:])])
    # robust escalation stats (C_t oscillates as the env keeps moving, so the
    # frontier the system REACHES is better captured by mean / high percentile
    # than by a noisy slope)
    Csorted = sorted(C)
    p75 = Csorted[int(0.75 * (n - 1))]
    early_w = mean(C[:max(1, n // 5)])
    late_w = mean(C[-max(1, n // 5):])
    return {
        "C_initial": C[0],
        "C_final": C[-1],
        "C_early_mean": C_early,
        "C_late_mean": C_late,
        "C_mean": mean(C),
        "C_peak": max(C),
        "C_p75": p75,
        "C_early_window": early_w,
        "C_late_window": late_w,
        "C_slope": _slope([float(x) for x in t], [float(c) for c in C]),
        "C_slope_late": _slope([float(x) for x in t[half:]],
                               [float(c) for c in C[half:]]),
        "C_grew": bool(C_late > C_early + 1e-9),
        "F_final": F[-1],
        "F_mean": mean(F),
        "K_final": K[-1],
        "gap_late": gap_late,
        "cum_novel": series["cum_novel"][-1] if series["cum_novel"] else 0,
        "novel_late": sum(series["N_novel"][half:]),
        "reuse_any": any(x > 0 for x in series["N_reused"]),
        "recomb_any": any(x > 0 for x in series["N_recombined"]),
        "N_modules_final": series["N_modules"][-1],
        "N_interactions_final": series["N_interactions"][-1],
    }


def run_coevolution(cfg: CoevoConfig) -> dict:
    rng = np.random.default_rng(cfg.seed)
    gacfg = _gacfg(cfg)

    env_g = _seed_env(cfg)
    pop = [_fresh_individual(rng, gacfg, cfg) for _ in range(cfg.pop)]

    history: List[Tuple[int, ...]] = []       # library of module expansion strings
    seen: set = set()
    archive: List[tuple] = []                 # mastered targets (for env novelty)
    series: Dict[str, list] = {k: [] for k in (
        "t", "F", "LG", "K", "C_t", "LE",
        "N_modules", "N_interactions", "N_reused", "N_recombined",
        "N_novel", "cum_novel")}
    cum_novel = 0

    for t in range(cfg.macro_steps):
        target = expand(env_g)
        pop = _evolve_genomes(pop, target, cfg, gacfg, rng)
        # the MDL objective picks the representation that develops REUSABLE rules
        # (exp588); raw-F champions merely memorise, so module accounting uses J.
        champ = max(pop, key=lambda i: i.J)

        mm = module_metrics(champ.g, target, history, cfg)
        ct = cstar(target, cfg)
        cum_novel += mm["N_novel"]

        series["t"].append(t)
        series["F"].append(champ.F)
        series["LG"].append(program_bits(champ.g))
        series["K"].append(description_length(champ.g))
        series["C_t"].append(ct)
        series["LE"].append(description_length(env_g))
        series["N_modules"].append(mm["N_modules"])
        series["N_interactions"].append(mm["N_interactions"])
        series["N_reused"].append(mm["N_reused"])
        series["N_recombined"].append(mm["N_recombined"])
        series["N_novel"].append(mm["N_novel"])
        series["cum_novel"].append(cum_novel)

        # grow the module library with this champion's used modules
        for sig in mm["used_sigs"]:
            if sig not in seen:
                seen.add(sig)
                history.append(sig)

        # archive the mastered target so the environment must find novel structure
        tt = tuple(target)
        if tt not in archive:
            archive.append(tt)
            if len(archive) > cfg.archive_cap:
                archive.pop(0)

        env_g = _env_step(env_g, champ, archive, cfg, gacfg, rng)

    return {
        "series": series,
        "summary": _summarise(series, cfg),
        "config": {k: getattr(cfg, k) for k in (
            "A", "L", "pop", "inner_gens", "macro_steps", "lam", "K_max",
            "alpha", "beta", "f_solve", "freeze_env", "use_mdl",
            "allow_recomb", "random_env", "seed")},
    }
