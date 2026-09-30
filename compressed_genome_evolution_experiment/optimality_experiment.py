"""exp589 -- optimality / Pareto audit of Compressed Genome Evolution.

For small problems (alphabet A=2, L in {8,12,16}) we compute the EXACT minimum
description length L*(F>=f) and the exact Pareto frontier over (L, F), in the
same bit units as genome.py, then ask how close the GA's solutions get:

    Delta L = L_GA - L*(F>=F_GA)          (bits above optimal at matched fitness)
    R_L     = L_GA / L*(F>=F_GA)

and where A (memorisation) and a lambda-sweep of C (MDL) sit relative to the
exact frontier -- removing the exp588 confound that A had F=1.00 while C had 0.96.

Targets (A=2):
    periodic      clean rule (period 4)                -> sharp frontier knee
    structured    tile + mirror + nuisance             -> residual gap at F=1
    unstructured  i.i.d. noise (null)                   -> frontier ~ flat at literal

L=8,12 use full A^L enumeration (exact global frontier); L=16 uses the high-F
Hamming ball (exact for F >= f_min), which is all that the placement needs.
"""

from __future__ import annotations

import json
import time
from statistics import mean

import numpy as np

import genome as G
from environments import make_periodic, make_structured, make_unstructured
from evolution import GAConfig, run_ga
from metrics import summarise_best
import optimal as O

A = 2
SIZES = [8, 12, 16]
GA_SEEDS = [1, 2, 3]
LAM_SWEEP = [0.0, 0.003, 0.006, 0.01, 0.015, 0.022]
BALL_RADIUS = {16: 4}          # F >= 0.75 for L=16
F_MIN = {16: 0.75}


def build_target(ttype, L, seed):
    rng = np.random.default_rng(500 + seed + hash_free(ttype))
    if ttype == "periodic":
        return make_periodic(rng, A, L, period=4)
    if ttype == "structured":
        return make_structured(rng, A, L, motif_len=4,
                               n_nuisance=max(1, L // 8))
    if ttype == "unstructured":
        return make_unstructured(rng, A, L)
    raise ValueError(ttype)


def hash_free(s):                      # deterministic small offset, not python hash()
    return {"periodic": 0, "structured": 7, "unstructured": 13}[s]


def ga_champion(target, group, L, lam):
    """Best-by-J (best-by-F for A) champion over the GA seeds."""
    best = None
    for seed in GA_SEEDS:
        cfg = GAConfig(A=A, L=L, pop=140, generations=180, lam=lam,
                       seed=seed, max_macros=4, nmax=8)
        r = run_ga(cfg, target, group)
        ch = r.best_by_F if group == "A" else r.best_by_J
        s = summarise_best(ch)
        key = (s["F"], -s["L_total"])          # max F, then min bits
        if best is None or key > best[0]:
            best = (key, s)
    return best[1]


def run_cell(ttype, L, log):
    seed = 1
    target = build_target(ttype, L, seed)
    tgt = target.target
    full = L <= 12
    if full:
        aud = O.audit_target(tgt, A=A, enumerate_full=True)
        fmin = 0.0
    else:
        r = BALL_RADIUS[L]
        fmin = F_MIN[L]
        aud = O.audit_target(tgt, A=A, enumerate_full=False,
                             hamming_radius=r, f_min=fmin)

    def lstar(f):
        return O.lstar_at_least(aud, f)

    # A: memorisation
    a = ga_champion(target, "A", L, lam=0.0)
    a_ls = lstar(a["F"])
    a["Lstar"] = a_ls
    a["deltaL"] = (a["L_total"] - a_ls) if a_ls else None
    a["R_L"] = (a["L_total"] / a_ls) if a_ls else None

    # C: lambda sweep
    c_points = []
    for lam in LAM_SWEEP:
        s = ga_champion(target, "C", L, lam=lam)
        ls = lstar(s["F"]) if s["F"] >= fmin - 1e-9 else None
        s["lam"] = lam
        s["Lstar"] = ls
        s["deltaL"] = (s["L_total"] - ls) if ls else None
        s["R_L"] = (s["L_total"] / ls) if ls else None
        c_points.append(s)

    # representative C: smallest genome among sweep with F >= 0.95 * F_A
    thresh = 0.95 * a["F"]
    elig = [c for c in c_points if c["F"] >= thresh and c["Lstar"] is not None]
    c_rep = min(elig, key=lambda c: c["L_total"]) if elig else \
        min([c for c in c_points if c["Lstar"] is not None],
            key=lambda c: c["R_L"] if c["R_L"] else 1e9, default=None)

    log(f"  [{ttype} L={L}]  L*(F=1.0)={lstar(1.0)}  "
        f"L*(F>=0.9)={lstar(0.9)}  frontier={_fmt_front(aud['frontier'])}")
    log(f"      A : F={a['F']:.3f} L_GA={a['L_total']:>3} "
        f"L*={a['Lstar']} dL={a['deltaL']} R_L="
        f"{_fmt(a['R_L'])}")
    for c in c_points:
        log(f"      C lam={c['lam']:<6} F={c['F']:.3f} L_GA={c['L_total']:>3} "
            f"L*={c['Lstar']} dL={c['deltaL']} R_L={_fmt(c['R_L'])} "
            f"reuse={c['reuse_rate']:.1f} span={c['mean_macro_span']:.1f}")
    if c_rep:
        log(f"      C*: F={c_rep['F']:.3f} L_GA={c_rep['L_total']} "
            f"L*={c_rep['Lstar']} dL={c_rep['deltaL']} R_L={_fmt(c_rep['R_L'])}")

    return {
        "ttype": ttype, "L": L, "target": list(tgt),
        "rule": target.rule,
        "full_enumeration": full, "f_min": fmin,
        "frontier": aud["frontier"],
        "Lstar_F1": lstar(1.0), "Lstar_F09": lstar(0.9),
        "A": _slim(a), "C_sweep": [_slim(c) for c in c_points],
        "C_representative": _slim(c_rep) if c_rep else None,
    }


def _slim(s):
    if s is None:
        return None
    return {k: s.get(k) for k in ("lam", "F", "L_total", "LG", "LD",
                                  "reuse_rate", "mean_macro_span",
                                  "Lstar", "deltaL", "R_L")}


def _fmt(v):
    return "None" if v is None else f"{v:.3f}"


def _fmt_front(front):
    return "[" + ", ".join(f"({l},{f:.2f})" for l, f in front[:6]) + \
           ("..." if len(front) > 6 else "") + "]"


def make_figure(cells, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    types = ["periodic", "structured", "unstructured"]
    fig, axes = plt.subplots(len(SIZES), len(types),
                             figsize=(12, 10), squeeze=False)
    for i, L in enumerate(SIZES):
        for j, tt in enumerate(types):
            ax = axes[i][j]
            cell = next(c for c in cells if c["L"] == L and c["ttype"] == tt)
            front = cell["frontier"]
            if front:
                xs = [p[0] for p in front]
                ys = [p[1] for p in front]
                ax.step(xs, ys, where="post", color="0.4",
                        label="exact frontier", zorder=1)
                ax.scatter(xs, ys, s=14, color="0.4", zorder=2)
            a = cell["A"]
            ax.scatter([a["L_total"]], [a["F"]], marker="s", s=70,
                       color="crimson", label="A (memorise)", zorder=4)
            cs = [c for c in cell["C_sweep"]]
            ax.scatter([c["L_total"] for c in cs], [c["F"] for c in cs],
                       marker="o", s=36, color="steelblue",
                       label="C (MDL, lambda sweep)", zorder=3)
            ax.set_title(f"{tt}  L={L}", fontsize=9)
            ax.set_xlabel("description length (bits)", fontsize=8)
            ax.set_ylabel("fitness F", fontsize=8)
            ax.tick_params(labelsize=7)
            if i == 0 and j == 0:
                ax.legend(fontsize=7, loc="lower right")
    fig.suptitle("exp589  GA solutions vs exact Pareto frontier (A=2)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return True


def main():
    t0 = time.time()
    lines = []

    def log(msg):
        print(msg, flush=True)
        lines.append(msg)

    log(f"exp589 optimality / Pareto audit  A={A} sizes={SIZES} "
        f"lam_sweep={LAM_SWEEP} ga_seeds={GA_SEEDS}")
    cells = []
    for L in SIZES:
        log("=" * 72)
        log(f"SIZE L={L}")
        log("=" * 72)
        for ttype in ["periodic", "structured", "unstructured"]:
            cells.append(run_cell(ttype, L, log))

    # headline summary: R_L of A vs representative C, averaged over structured/periodic
    log("=" * 72)
    log("HEADLINE  (how close to the minimum description, at matched fitness)")
    log("=" * 72)
    for tt in ["periodic", "structured", "unstructured"]:
        aR = [c["A"]["R_L"] for c in cells if c["ttype"] == tt and c["A"]["R_L"]]
        cR = [c["C_representative"]["R_L"] for c in cells
              if c["ttype"] == tt and c["C_representative"]
              and c["C_representative"]["R_L"]]
        log(f"  {tt:>12}: mean R_L  A={mean(aR):.3f}  C*={mean(cR):.3f}"
            if aR and cR else f"  {tt:>12}: (insufficient data)")

    fig_ok = make_figure(cells, "pareto_frontier.png")
    log(f"figure: {'pareto_frontier.png' if fig_ok else '(matplotlib unavailable)'}")

    dt = time.time() - t0
    log("=" * 72)
    log(f"done in {dt:.1f}s")

    results = {
        "config": {"A": A, "sizes": SIZES, "lam_sweep": LAM_SWEEP,
                   "ga_seeds": GA_SEEDS, "ball_radius": BALL_RADIUS},
        "cells": cells,
        "runtime_seconds": dt,
    }
    with open("optimality_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    with open("optimality_run.log", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote optimality_results.json and optimality_run.log")


if __name__ == "__main__":
    main()
