"""Derived metrics for Compressed Genome Evolution (exp588).

These turn a finished run into the quantities the attachment asked for:

    F, L(G), L(D), F / (L(G)+L(D))
    Adaptation efficiency = dF / inherited bits
    mutation robustness
    re-adaptation generations after an environment change
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

import genome as G
from genome import Genome
from environments import Environment, fitness
from evolution import (GAConfig, Individual, RunResult, run_ga,
                       _mutate, evaluate)


def summarise_best(ind: Individual) -> dict:
    g = ind.g
    lg = G.program_bits(g)
    ld = G.macro_bits(g)
    tot = lg + ld
    return {
        "F": ind.F,
        "LG": lg,
        "LD": ld,
        "L_total": tot,
        "F_per_bit": (ind.F / tot) if tot > 0 else 0.0,
        "reuse_rate": G.reuse_rate(g),
        "mean_macro_span": G.mean_used_macro_span(g),
        "n_macros": len(g.macros),
        "program_len": len(g.program),
        "call_counts": G.macro_call_counts(g),
    }


def adaptation_efficiency(ind: Individual, A: int) -> float:
    """dF per inherited bit.

    dF is fitness above chance (1/A); inherited bits is the genome's full
    description length (what is copied to offspring)."""
    tot = G.description_length(ind.g)
    dF = ind.F - (1.0 / A)
    return dF / tot if tot > 0 else 0.0


def mutation_robustness(ind: Individual, env: Environment, cfg: GAConfig,
                        group: str, n: int = 200, seed: int = 12345) -> dict:
    """Sample single mutations of the best genome; report the fitness response."""
    rng = np.random.default_rng(seed)
    base = ind.F
    deltas = []
    for _ in range(n):
        child_g = _mutate(rng, cfg, ind.g.copy(), group)
        phen = G.expand(child_g)
        f = fitness(phen, env)
        deltas.append(f - base)
    deltas = np.array(deltas)
    return {
        "mean_delta": float(deltas.mean()),
        "mean_abs_delta": float(np.abs(deltas).mean()),
        "frac_neutral": float(np.mean(np.abs(deltas) < 1e-9)),
        "frac_beneficial": float(np.mean(deltas > 1e-9)),
        "frac_deleterious": float(np.mean(deltas < -1e-9)),
        "worst": float(deltas.min()),
    }


def re_adaptation(cfg: GAConfig, group: str,
                  env1: Environment, env2: Environment,
                  gens_phase1: int, gens_phase2: int,
                  threshold: float) -> dict:
    """Evolve on env1, then switch to a related env2 and measure recovery.

    Returns the phase-2 fitness trajectory and the number of generations to
    first reach `threshold` on env2 (None if never)."""
    c1 = _clone_cfg(cfg, generations=gens_phase1)
    rng = np.random.default_rng(cfg.seed)
    r1 = run_ga(c1, env1, group, rng=rng)

    c2 = _clone_cfg(cfg, generations=gens_phase2)
    r2 = run_ga(c2, env2, group, init_pop=r1.final_pop, rng=rng)

    traj = r2.history["bestF"]
    gens_to_thresh = None
    for i, f in enumerate(traj):
        if f >= threshold:
            gens_to_thresh = i
            break
    return {
        "phase1_bestF": r1.best_by_F.F,
        "phase2_startF": traj[0] if traj else None,
        "phase2_final_bestF": r2.best_by_F.F,
        "gens_to_threshold": gens_to_thresh,
        "phase2_trajectory": traj,
        "phase1_best_summary": summarise_best(r1.best_by_F),
    }


def _clone_cfg(cfg: GAConfig, **overrides) -> GAConfig:
    d = cfg.__dict__.copy()
    d.update(overrides)
    return GAConfig(**d)
