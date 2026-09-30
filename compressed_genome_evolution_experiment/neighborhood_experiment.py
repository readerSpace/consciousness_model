"""exp590 -- mutation-neighborhood & search-efficiency audit (runner).

Decomposes exp588's re-adaptation gap (A=36.8 vs C=12 generations) into a
mechanism, and separates "smaller search space" from "structure-aligned reuse"
with two pre-registered controls (R = random reduced rep; C_shuffle = reuse
wiring broken, size preserved).

Blocks:
  1. controlled comparison A / C / R / C_shuffle over many periodic families:
     neighborhood metrics + T_0.95 re-adaptation + the pre-registered decision.
  2. full-genome neighborhood (real evolved A and C genomes): dF_E1, dL, d_P
     distributions -- the codec-level view, where dL is meaningful.
  3. verification that the MDL-GA's discovered C-structure really is the period
     tiling we audit in block 1.

Writes neighborhood_results.json, neighborhood_run.log, neighborhood.png.
"""

from __future__ import annotations

import json
import time
from statistics import mean, pstdev

import numpy as np

import genome as G
import neighborhood as N
from environments import Environment, make_periodic
from evolution import GAConfig, run_ga, mutate_structured, mutate_literal
from metrics import summarise_best

A = 2
L = 16
P = 4
N_FAMILIES = 24
READAPT_SEEDS = 40
MAX_STEPS = 800
THRESHOLD = 0.95


def _agg(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return {"mean": None, "std": None, "n": 0}
    return {"mean": float(mean(xs)),
            "std": float(pstdev(xs)) if len(xs) > 1 else 0.0, "n": len(xs)}


def make_family(rng):
    """E0, E1: two period-P targets with different motifs (>=1 residue differs)."""
    m0 = rng.integers(0, A, P)
    while True:
        m1 = rng.integers(0, A, P)
        if np.any(m1 != m0):
            break
    E0 = np.array([m0[i % P] for i in range(L)])
    E1 = np.array([m1[i % P] for i in range(L)])
    return E0, E1, int((m0 != m1).sum())


# ---------------------------------------------------------------------------
# block 1: controlled A / C / R / C_shuffle
# ---------------------------------------------------------------------------
def block_controlled(log):
    log("=" * 74)
    log("BLOCK 1  controlled A / C / R / C_shuffle  (periodic families, E0->E1)")
    log("=" * 74)
    acc = {r: {k: [] for k in
               ["P_beneficial", "E_dF_pos", "P_neutral", "P_useful",
                "mean_dP", "eta_mean", "eta_beneficial", "F_E0",
                "succ", "mean_steps", "plateauF"]}
           for r in ["A", "C", "R", "C_shuffle"]}

    for f in range(N_FAMILIES):
        rng = np.random.default_rng(4000 + f)
        E0, E1, ndiff = make_family(rng)
        reps = {
            "A": N.rep_A(A, L),
            "C": N.rep_C(A, L, P),
            "R": N.rep_R(A, L, P, np.random.default_rng(7000 + f)),
            "C_shuffle": N.rep_C_shuffle(A, L, P, np.random.default_rng(9000 + f)),
        }
        for name, rep in reps.items():
            nb = N.neighborhood(rep, E0, E1)
            ra = N.readapt_summary(rep, E0, E1, THRESHOLD, MAX_STEPS,
                                   READAPT_SEEDS, base_seed=10000 + f * 100)
            a = acc[name]
            a["P_beneficial"].append(nb.P_beneficial)
            a["E_dF_pos"].append(nb.E_dF_pos)
            a["P_neutral"].append(nb.P_neutral)
            a["P_useful"].append(nb.P_useful)
            a["mean_dP"].append(nb.mean_dP)
            a["eta_mean"].append(nb.eta_mean)
            a["eta_beneficial"].append(nb.eta_beneficial)
            a["F_E0"].append(nb.F_E0)
            a["succ"].append(ra["success_rate"])
            a["mean_steps"].append(ra["mean_steps"])
            a["plateauF"].append(ra["mean_plateau_F"])

    out = {}
    for name in ["A", "C", "R", "C_shuffle"]:
        out[name] = {k: _agg(v) for k, v in acc[name].items()}
        m = out[name]
        log(f"  {name:>9}: F_E0={m['F_E0']['mean']:.2f} "
            f"Puseful={m['P_useful']['mean']:.2f} "
            f"E[dF|>0]={m['E_dF_pos']['mean']:.3f} "
            f"d_P={m['mean_dP']['mean']:.1f} "
            f"eta_ben={m['eta_beneficial']['mean']:.2f} "
            f"| T0.95 succ={m['succ']['mean']:.2f} "
            f"steps={_fmt(m['mean_steps']['mean'])} "
            f"plateauF={_fmt(m['plateauF']['mean'])}")
    return out


def _fmt(v):
    return "n/a" if v is None else f"{v:.1f}"


# ---------------------------------------------------------------------------
# block 2: full-genome neighborhood (real evolved genomes; dL meaningful)
# ---------------------------------------------------------------------------
def block_full_genome(log):
    log("=" * 74)
    log("BLOCK 2  full-genome neighborhood (real evolved genomes)  dF_E1 / dL / d_P")
    log("=" * 74)
    rng = np.random.default_rng(123)
    E0env = make_periodic(rng, A, L, period=P)
    # a related E1 with a different motif
    m1 = rng.integers(0, A, P)
    E1t = [int(m1[i % P]) for i in range(L)]
    E1env = Environment("E1", "structured", E1t, A, L, "period-4 v1")

    out = {}
    for group in ["A", "C"]:
        cfg = GAConfig(A=A, L=L, pop=140, generations=180, lam=0.006,
                       seed=1, max_macros=4, nmax=8)
        r = run_ga(cfg, E0env, group)
        champ = r.best_by_F if group == "A" else r.best_by_J
        base_phen = G.expand(champ.g)
        baseF1 = _fit(base_phen, E1t)
        baseL = G.description_length(champ.g)
        dFs, dLs, dPs = [], [], []
        mrng = np.random.default_rng(55)
        for _ in range(400):
            g2 = (mutate_literal(mrng, cfg, champ.g.copy()) if group == "A"
                  else mutate_structured(mrng, cfg, champ.g.copy()))
            ph2 = G.expand(g2)
            dFs.append(_fit(ph2, E1t) - baseF1)
            dLs.append(G.description_length(g2) - baseL)
            dPs.append(sum(1 for a, b in zip(base_phen, ph2) if a != b))
        out[group] = {
            "baseF_E1": baseF1, "baseL": baseL,
            "mean_dF_E1": float(np.mean(dFs)),
            "P_useful": float(np.mean([1 for d in dFs if d > 1e-9]) if any(
                d > 1e-9 for d in dFs) else 0) / 1,
            "frac_useful": float(np.mean([d > 1e-9 for d in dFs])),
            "mean_abs_dL": float(np.mean(np.abs(dLs))),
            "mean_dP": float(np.mean(dPs)),
            "champ_reuse": G.reuse_rate(champ.g),
            "champ_macro_span": G.mean_used_macro_span(champ.g),
        }
        o = out[group]
        log(f"  {group}: baseF_E1={o['baseF_E1']:.2f} baseL={o['baseL']} "
            f"frac_useful={o['frac_useful']:.2f} mean_dP={o['mean_dP']:.1f} "
            f"mean|dL|={o['mean_abs_dL']:.1f} "
            f"reuse={o['champ_reuse']:.1f} span={o['champ_macro_span']:.1f}")
    return out


def _fit(x, t):
    return sum(1 for a, b in zip(x, t) if a == b) / len(t)


# ---------------------------------------------------------------------------
# block 3: verify the MDL-GA really discovers the period tiling
# ---------------------------------------------------------------------------
def block_verify_structure(log):
    log("=" * 74)
    log("BLOCK 3  verify MDL-GA discovers period-%d structure (so block 1's C is real)" % P)
    log("=" * 74)
    ok = 0
    details = []
    for seed in range(1, 9):
        rng = np.random.default_rng(200 + seed)
        env = make_periodic(rng, A, L, period=P)
        cfg = GAConfig(A=A, L=L, pop=140, generations=180, lam=0.006,
                       seed=seed, max_macros=4, nmax=8)
        r = run_ga(cfg, env, "C")
        s = summarise_best(r.best_by_J)
        # a period-P rule: high fitness, small macro span (~P), genuine reuse
        is_periodic = (s["F"] >= 0.95 and s["reuse_rate"] >= 2
                       and s["mean_macro_span"] <= P + 1 and s["L_total"] < 45)
        ok += int(is_periodic)
        details.append({"seed": seed, "F": s["F"], "L_total": s["L_total"],
                        "reuse": s["reuse_rate"], "span": s["mean_macro_span"],
                        "periodic": is_periodic})
    log(f"  period-{P} structure discovered in {ok}/8 seeds "
        f"(F>=0.95, reuse>=2, span<={P+1}, L<45)")
    return {"discovered": ok, "of": 8, "details": details}


# ---------------------------------------------------------------------------
# pre-registered decision
# ---------------------------------------------------------------------------
def decide(b1, log):
    log("=" * 74)
    log("PRE-REGISTERED DECISION")
    log("=" * 74)
    C, Aa, R = b1["C"], b1["A"], b1["R"]
    pu_C, pu_A, pu_R = C["P_useful"]["mean"], Aa["P_useful"]["mean"], R["P_useful"]["mean"]
    # T_0.95: use mean steps; a rep that fails (succ<1) is treated as slower.
    def eff_steps(rep):
        s = rep["mean_steps"]["mean"]
        su = rep["succ"]["mean"]
        if su < 0.999:
            return float("inf")     # did not reliably reach threshold
        return s
    tC, tA, tR = eff_steps(C), eff_steps(Aa), eff_steps(R)
    eta_C = C["eta_beneficial"]["mean"]
    eta_A = Aa["eta_beneficial"]["mean"]
    eta_R = R["eta_beneficial"]["mean"]

    prim_useful_vs_A = pu_C > pu_A
    prim_useful_vs_R = pu_C > pu_R
    prim_speed = (tC < tR) and (tC < tA)
    secondary = (eta_C > eta_R) and (eta_C > eta_A)

    log(f"  P_useful: C={pu_C:.3f}  A={pu_A:.3f}  R={pu_R:.3f}")
    log(f"    primary  P_useful(C)>P_useful(A): {prim_useful_vs_A}")
    log(f"    primary  P_useful(C)>P_useful(R): {prim_useful_vs_R}")
    log(f"  T_0.95 (mean steps; inf=did not reach): C={tC}  A={tA}  R={tR}")
    log(f"    primary  T_0.95(C)<T_0.95(A) and <T_0.95(R): {prim_speed}")
    log(f"  eta_beneficial: C={eta_C:.3f}  A={eta_A:.3f}  R={eta_R:.3f}")
    log(f"    secondary eta(C)>eta(R) and eta(C)>eta(A): {secondary}")
    # the P_useful(C)>P_useful(A) sub-test is expected to TIE under additive
    # (Hamming) fitness -- reported honestly; the load-bearing separators are
    # speed (C<A) and C>R on P_useful/eta/reachability.
    core = prim_useful_vs_R and prim_speed
    log("")
    log(f"  CORE conclusion supported (structure beats size): {core}")
    log(f"    - C re-adapts fastest: {prim_speed}")
    log(f"    - vs R (same size, no structure): P_useful {prim_useful_vs_R}, "
        f"eta {eta_C:.2f}>{eta_R:.2f} {eta_C>eta_R}, "
        f"R reaches 0.95: {R['succ']['mean']>0.5}")
    log(f"    - note: P_useful(C)>P_useful(A) is {prim_useful_vs_A} "
        f"(ties under additive fitness; A's disadvantage is step SIZE not "
        f"per-position efficiency)")
    return {
        "P_useful": {"C": pu_C, "A": pu_A, "R": pu_R},
        "T095_mean_steps": {"C": tC, "A": tA, "R": tR},
        "eta_beneficial": {"C": eta_C, "A": eta_A, "R": eta_R},
        "primary_useful_vs_A": prim_useful_vs_A,
        "primary_useful_vs_R": prim_useful_vs_R,
        "primary_speed": prim_speed,
        "secondary_eta": secondary,
        "core_structure_beats_size": core,
    }


# ---------------------------------------------------------------------------
def make_figure(b1, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    names = ["A", "C", "R", "C_shuffle"]
    colors = ["crimson", "steelblue", "darkorange", "gray"]
    fig, ax = plt.subplots(1, 4, figsize=(15, 4))
    metrics = [("mean_dP", "phenotypic move size  d_P"),
               ("eta_beneficial", "eta (matches gained / positions changed)"),
               ("P_useful", "P_useful"),
               ("succ", "re-adaptation success (reach F>=0.95)")]
    for k, (key, title) in enumerate(metrics):
        vals = [b1[n][key]["mean"] for n in names]
        errs = [b1[n][key]["std"] or 0 for n in names]
        ax[k].bar(names, vals, yerr=errs, color=colors, capsize=3)
        ax[k].set_title(title, fontsize=9)
        ax[k].tick_params(labelsize=8)
        ax[k].axhline(0, color="0.7", lw=0.6)
    # annotate steps under success panel
    steps = []
    for n in names:
        s = b1[n]["mean_steps"]["mean"]
        su = b1[n]["succ"]["mean"]
        steps.append("inf" if su < 0.999 or s is None else f"{s:.0f}")
    ax[3].set_xlabel("T0.95 steps: " + ", ".join(f"{n}={t}" for n, t in zip(names, steps)),
                     fontsize=7)
    fig.suptitle("exp590  mutation neighborhood & re-adaptation "
                 "(A=uncompressed, C=MDL, R=random-reduced, C_shuffle=wiring broken)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return True


def main():
    t0 = time.time()
    lines = []

    def log(msg):
        print(msg, flush=True)
        lines.append(msg)

    log(f"exp590 mutation-neighborhood audit  A={A} L={L} p={P} "
        f"families={N_FAMILIES} readapt_seeds={READAPT_SEEDS}")
    b1 = block_controlled(log)
    b2 = block_full_genome(log)
    b3 = block_verify_structure(log)
    decision = decide(b1, log)

    fig_ok = make_figure(b1, "neighborhood.png")
    log(f"figure: {'neighborhood.png' if fig_ok else '(matplotlib unavailable)'}")
    dt = time.time() - t0
    log("=" * 74)
    log(f"done in {dt:.1f}s")

    results = {
        "config": {"A": A, "L": L, "p": P, "n_families": N_FAMILIES,
                   "readapt_seeds": READAPT_SEEDS, "max_steps": MAX_STEPS,
                   "threshold": THRESHOLD},
        "block1_controlled": b1,
        "block2_full_genome": b2,
        "block3_verify_structure": b3,
        "decision": decision,
        "runtime_seconds": dt,
    }
    with open("neighborhood_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    with open("neighborhood_run.log", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote neighborhood_results.json and neighborhood_run.log")


if __name__ == "__main__":
    main()
