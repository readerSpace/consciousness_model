"""exp588 -- Compressed Genome Evolution: full experiment.

Research question (from the design note):

    Does compression pressure merely make genomes SHORT, or does it evolve
    REUSABLE genetic rules?

We separate the two with four groups on one MDL ruler:

    A  fixed-length genome          J = F                      (memorisation baseline)
    B  program genome               J = F - lam*L(G)           (genome-length penalty)
    C  program genome (full MDL)    J = F - lam*(L(G)+L(D))    (MDL-GA)
    D  MDL over related targets     J = mean_e[...]            ("future adaptability")

and across structured vs unstructured (null) environments.

Blocks:
  1. main comparison   A/B/C/(D) x {periodic, structured, unstructured} x seeds
  2. lambda sweep      C on periodic/structured -> representation phase transition
  3. re-adaptation     A vs C: switch to a related target, count recovery gens
  4. mutation robustness of each group's champion
  5. controls          shuffled-structure null

Writes results.json and prints a run log.
"""

from __future__ import annotations

import json
import time
from statistics import mean, pstdev

import numpy as np

import genome as G
from environments import (make_periodic, make_structured, make_unstructured,
                          make_shuffled, related_periodic_family)
from evolution import GAConfig, run_ga
from metrics import (summarise_best, adaptation_efficiency, mutation_robustness,
                     re_adaptation)

# ---------------------------------------------------------------------------
A = 4
L = 32
SEEDS = [1, 2, 3, 4, 5]
LAM = 0.002
POP = 160
GENS = 220


def _agg(dicts, key):
    vals = [d[key] for d in dicts]
    return {"mean": float(mean(vals)),
            "std": float(pstdev(vals)) if len(vals) > 1 else 0.0,
            "vals": [float(v) for v in vals]}


def _fresh_env(kind, seed):
    rng = np.random.default_rng(1000 + seed)
    if kind == "periodic":
        return make_periodic(rng, A, L, period=4)
    if kind == "structured":
        return make_structured(rng, A, L, motif_len=3, n_nuisance=3)
    if kind == "unstructured":
        return make_unstructured(rng, A, L)
    raise ValueError(kind)


# ---------------------------------------------------------------------------
# block 1: main comparison
# ---------------------------------------------------------------------------
def block_main(log):
    log("=" * 72)
    log("BLOCK 1  main comparison  A/B/C  x  {periodic, structured, unstructured}")
    log("=" * 72)
    out = {}
    for kind in ["periodic", "structured", "unstructured"]:
        out[kind] = {}
        for group in ["A", "B", "C"]:
            per_seed = []
            for seed in SEEDS:
                env = _fresh_env(kind, seed)
                cfg = GAConfig(A=A, L=L, pop=POP, generations=GENS,
                               lam=LAM, seed=seed)
                r = run_ga(cfg, env, group)
                champ = r.best_by_F if group == "A" else r.best_by_J
                s = summarise_best(champ)
                s["adapt_eff"] = adaptation_efficiency(r.best_by_F, A)
                per_seed.append(s)
            out[kind][group] = {
                "F": _agg(per_seed, "F"),
                "L_total": _agg(per_seed, "L_total"),
                "LG": _agg(per_seed, "LG"),
                "LD": _agg(per_seed, "LD"),
                "reuse_rate": _agg(per_seed, "reuse_rate"),
                "mean_macro_span": _agg(per_seed, "mean_macro_span"),
                "F_per_bit": _agg(per_seed, "F_per_bit"),
                "adapt_eff": _agg(per_seed, "adapt_eff"),
            }
            m = out[kind][group]
            log(f"  {kind:>12} {group}: "
                f"F={m['F']['mean']:.3f}+-{m['F']['std']:.3f} "
                f"Ltot={m['L_total']['mean']:6.1f} "
                f"LG={m['LG']['mean']:5.1f} LD={m['LD']['mean']:6.1f} "
                f"reuse={m['reuse_rate']['mean']:4.1f} "
                f"span={m['mean_macro_span']['mean']:4.1f} "
                f"AdEff(1e3)={1000*m['adapt_eff']['mean']:.2f}")
    return out


# ---------------------------------------------------------------------------
# block 2: lambda sweep -> representation phase transition
# ---------------------------------------------------------------------------
def block_lambda_sweep(log):
    log("=" * 72)
    log("BLOCK 2  lambda sweep (group C)  ->  memorisation -> rule transition")
    log("=" * 72)
    lambdas = [0.0, 1e-4, 3e-4, 1e-3, 2e-3, 4e-3, 8e-3, 1.6e-2, 3.2e-2]
    out = {}
    for kind in ["periodic", "structured"]:
        out[kind] = []
        for lam in lambdas:
            per_seed = []
            for seed in SEEDS:
                env = _fresh_env(kind, seed)
                cfg = GAConfig(A=A, L=L, pop=POP, generations=GENS,
                               lam=lam, seed=seed)
                r = run_ga(cfg, env, "C")
                per_seed.append(summarise_best(r.best_by_J))
            row = {
                "lam": lam,
                "F": _agg(per_seed, "F"),
                "L_total": _agg(per_seed, "L_total"),
                "reuse_rate": _agg(per_seed, "reuse_rate"),
                "mean_macro_span": _agg(per_seed, "mean_macro_span"),
            }
            out[kind].append(row)
            log(f"  {kind:>12} lam={lam:<8}: "
                f"F={row['F']['mean']:.3f} "
                f"Ltot={row['L_total']['mean']:6.1f} "
                f"reuse={row['reuse_rate']['mean']:4.1f} "
                f"span={row['mean_macro_span']['mean']:4.1f}")
    return out


# ---------------------------------------------------------------------------
# block 3: re-adaptation after an environment change (A vs C)
# ---------------------------------------------------------------------------
def block_readaptation(log):
    log("=" * 72)
    log("BLOCK 3  re-adaptation: evolve on env1, switch to related env2")
    log("=" * 72)
    out = {}
    threshold = 0.95
    for group in ["A", "C"]:
        per_seed = []
        for seed in SEEDS:
            rng = np.random.default_rng(2000 + seed)
            fam = related_periodic_family(rng, A, L, period=4, k=2)
            env1, env2 = fam[0], fam[1]
            cfg = GAConfig(A=A, L=L, pop=POP, generations=GENS,
                           lam=LAM, seed=seed)
            res = re_adaptation(cfg, group, env1, env2,
                                gens_phase1=GENS, gens_phase2=120,
                                threshold=threshold)
            per_seed.append(res)
        gens = [r["gens_to_threshold"] for r in per_seed
                if r["gens_to_threshold"] is not None]
        startF = [r["phase2_startF"] for r in per_seed]
        out[group] = {
            "gens_to_threshold_mean": float(mean(gens)) if gens else None,
            "gens_to_threshold_vals": [r["gens_to_threshold"] for r in per_seed],
            "phase2_startF_mean": float(mean(startF)),
            "n_recovered": len(gens),
        }
        log(f"  group {group}: recovered {out[group]['n_recovered']}/{len(SEEDS)}, "
            f"gens_to_{threshold}="
            f"{out[group]['gens_to_threshold_mean']}, "
            f"startF_after_switch={out[group]['phase2_startF_mean']:.3f} "
            f"(vals {out[group]['gens_to_threshold_vals']})")
    return out


# ---------------------------------------------------------------------------
# block 4: mutation robustness of each group's champion (structured env)
# ---------------------------------------------------------------------------
def block_robustness(log):
    log("=" * 72)
    log("BLOCK 4  mutation robustness of champions (structured env, seed 1)")
    log("=" * 72)
    out = {}
    env = _fresh_env("structured", 1)
    for group in ["A", "B", "C"]:
        cfg = GAConfig(A=A, L=L, pop=POP, generations=GENS, lam=LAM, seed=1)
        r = run_ga(cfg, env, group)
        champ = r.best_by_F if group == "A" else r.best_by_J
        rob = mutation_robustness(champ, env, cfg, group, n=300)
        out[group] = rob
        log(f"  group {group}: mean_abs_dF={rob['mean_abs_delta']:.4f} "
            f"neutral={rob['frac_neutral']:.2f} "
            f"benef={rob['frac_beneficial']:.2f} "
            f"delet={rob['frac_deleterious']:.2f} worst={rob['worst']:.3f}")
    return out


# ---------------------------------------------------------------------------
# block 5: controls (shuffled-structure null)
# ---------------------------------------------------------------------------
def block_controls(log):
    log("=" * 72)
    log("BLOCK 5  control: shuffled structure behaves like noise")
    log("=" * 72)
    out = {}
    for group in ["A", "B", "C"]:
        struct, shuf = [], []
        for seed in SEEDS:
            rng = np.random.default_rng(3000 + seed)
            base = make_periodic(rng, A, L, period=4)
            sh = make_shuffled(rng, base)
            cfg = GAConfig(A=A, L=L, pop=POP, generations=GENS, lam=LAM, seed=seed)
            struct.append(summarise_best(
                (run_ga(cfg, base, group).best_by_F if group == "A"
                 else run_ga(cfg, base, group).best_by_J)))
            shuf.append(summarise_best(
                (run_ga(cfg, sh, group).best_by_F if group == "A"
                 else run_ga(cfg, sh, group).best_by_J)))
        out[group] = {
            "structured_L_total": _agg(struct, "L_total"),
            "shuffled_L_total": _agg(shuf, "L_total"),
            "structured_F": _agg(struct, "F"),
            "shuffled_F": _agg(shuf, "F"),
        }
        o = out[group]
        log(f"  group {group}: "
            f"structured Ltot={o['structured_L_total']['mean']:6.1f} "
            f"(F={o['structured_F']['mean']:.2f}) | "
            f"shuffled Ltot={o['shuffled_L_total']['mean']:6.1f} "
            f"(F={o['shuffled_F']['mean']:.2f})")
    return out


# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    lines = []

    def log(msg):
        print(msg, flush=True)
        lines.append(msg)

    log(f"exp588 Compressed Genome Evolution  A={A} L={L} pop={POP} gens={GENS} "
        f"lam={LAM} seeds={SEEDS}")
    results = {
        "config": {"A": A, "L": L, "pop": POP, "generations": GENS,
                   "lam": LAM, "seeds": SEEDS},
        "block1_main": block_main(log),
        "block2_lambda_sweep": block_lambda_sweep(log),
        "block3_readaptation": block_readaptation(log),
        "block4_robustness": block_robustness(log),
        "block5_controls": block_controls(log),
    }
    dt = time.time() - t0
    log("=" * 72)
    log(f"done in {dt:.1f}s")
    results["runtime_seconds"] = dt

    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    with open("run.log", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote results.json and run.log")


if __name__ == "__main__":
    main()
