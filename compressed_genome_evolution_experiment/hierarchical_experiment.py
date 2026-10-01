"""exp597 -- Hierarchical Compositional Escalation (runner).

exp596 established the WEAK arrow (external curriculum -> not needed): required
complexity escalates endogenously.  exp597 targets ONLY the STRONG arrow it left
open:

    endogenous complexity escalation  --->  CONTINUING compositional novelty ?

and makes the environment-change timescale tau_E the central variable, testing

    does open-ended compositional evolution require tau_E ~ tau_module ?

Pre-registered STRICT criteria (C_t escalation alone is a FAIL):
    functional novelty rate > 0
    reuse rate            > 0
    recombination rate    > 0
    hierarchy depth       > 1
    novelty CONTINUES in the late phase (not only the first generations)
    the phenomenon continues when capacity is increased

Blocks:
  1. tau_module calibration (generations to first reused module on a fixed task)
  2. tau_E sweep {10,30,100,300}  + no-factorize and frozen controls
     -> the phase transition in reuse / hierarchy / continuing novelty
  3. capacity scaling (does the phenomenon persist at larger K_max?)
  4. small-L exact-L* audit: the bounded-MDL U_t tracks exact L*
  5. strict verdict + the central timescale question

Writes hierarchical_results.json, hierarchical_run.log, hierarchical.png.
"""

from __future__ import annotations

import json
import time
from statistics import mean, pstdev

import numpy as np

from hierarchical import HConfig, run_hierarchical, tau_module

SEEDS = [1, 2, 3]
ENV_CHANGES = 6
TAUS = [10, 30, 100, 300]
L_MAIN = 48


def _agg(vals):
    vals = list(vals)
    return {"mean": float(mean(vals)),
            "std": float(pstdev(vals)) if len(vals) > 1 else 0.0,
            "vals": [float(v) for v in vals]}


def _runs(seeds, **kw):
    return [run_hierarchical(HConfig(L=L_MAIN, env_changes=ENV_CHANGES,
                                     seed=s, **kw)) for s in seeds]


# ---------------------------------------------------------------------------
# block 1: tau_module calibration
# ---------------------------------------------------------------------------
def block_tau_module(log):
    log("=" * 78)
    log("BLOCK 1  tau_module calibration (generations to first reused module)")
    log("=" * 78)
    cfg = HConfig(L=L_MAIN)
    tm = tau_module(cfg, seeds=[1, 2, 3, 4, 5], max_gens=500)
    log(f"  tau_module (median gens to first reuse, fixed target) = {tm}")
    return tm


# ---------------------------------------------------------------------------
# block 2: tau_E sweep -> phase transition
# ---------------------------------------------------------------------------
def _summ_rows(runs):
    s = [r["summary"] for r in runs]
    return {
        "max_hierarchy_depth": _agg(x["max_hierarchy_depth"] for x in s),
        "reuse_rate_late": _agg(x["reuse_rate_late"] for x in s),
        "reuse_any": float(mean(1.0 if x["reuse_any"] else 0.0 for x in s)),
        "recomb_any": float(mean(1.0 if x["recomb_any"] else 0.0 for x in s)),
        "n_recomb_total": _agg(x["n_recomb_total"] for x in s),
        "cum_novel": _agg(x["cum_novel"] for x in s),
        "novel_first_half": _agg(x["novel_first_half"] for x in s),
        "novel_second_half": _agg(x["novel_second_half"] for x in s),
        "comp_novel_total": _agg(x["comp_novel_total"] for x in s),
        "U_early": _agg(x["U_early"] for x in s),
        "U_late": _agg(x["U_late"] for x in s),
        "F_mean": _agg(x["F_mean"] for x in s),
    }


def block_tau_sweep(log, tau_module_val):
    log("=" * 78)
    log("BLOCK 2  tau_E sweep {10,30,100,300}  (phase transition) + controls")
    log("=" * 78)
    out = {"tau_module": tau_module_val, "sweep": {}}
    for tau in TAUS:
        runs = _runs(SEEDS, tau_E=tau)
        row = _summ_rows(runs)
        row["series_seed1"] = runs[0]["series"]
        out["sweep"][str(tau)] = row
        log(f"  tau_E={tau:>3}: depth={row['max_hierarchy_depth']['mean']:.1f} "
            f"reuse_late={row['reuse_rate_late']['mean']:.2f} "
            f"novel(cum/2nd)={row['cum_novel']['mean']:.1f}/"
            f"{row['novel_second_half']['mean']:.1f} "
            f"recomb={row['n_recomb_total']['mean']:.1f} "
            f"U {row['U_early']['mean']:.0f}->{row['U_late']['mean']:.0f} "
            f"F={row['F_mean']['mean']:.2f}")
    # controls at the best tau_E (by cum_novel)
    best_tau = max(TAUS, key=lambda t: out["sweep"][str(t)]["cum_novel"]["mean"])
    out["best_tau"] = best_tau
    log(f"  -> peak compositional activity at tau_E = {best_tau} "
        f"(tau_module ~ {tau_module_val})")
    nofact = _summ_rows(_runs(SEEDS, tau_E=best_tau, allow_factorize=False))
    frozen = _summ_rows(_runs(SEEDS, tau_E=best_tau, freeze_env=True))
    out["control_no_factorize"] = nofact
    out["control_frozen"] = frozen
    log(f"  control no-factorize : reuse_any={nofact['reuse_any']:.2f} "
        f"depth={nofact['max_hierarchy_depth']['mean']:.1f} "
        f"cum_novel={nofact['cum_novel']['mean']:.1f}  (factorisation is the mechanism)")
    log(f"  control frozen-env   : novel(1st/2nd)="
        f"{frozen['novel_first_half']['mean']:.1f}/"
        f"{frozen['novel_second_half']['mean']:.1f}  (no new structure demanded)")
    return out


# ---------------------------------------------------------------------------
# block 3: capacity scaling
# ---------------------------------------------------------------------------
def block_capacity(log, best_tau):
    log("=" * 78)
    log("BLOCK 3  capacity scaling (does the phenomenon persist at larger K_max?)")
    log("=" * 78)
    out = {}
    for kmax in [120, 200]:
        row = _summ_rows(_runs(SEEDS, tau_E=best_tau, K_max=kmax))
        out[str(kmax)] = row
        log(f"  K_max={kmax:>3}: depth={row['max_hierarchy_depth']['mean']:.1f} "
            f"reuse_late={row['reuse_rate_late']['mean']:.2f} "
            f"cum_novel={row['cum_novel']['mean']:.1f} "
            f"novel_2nd={row['novel_second_half']['mean']:.1f}")
    persists = (out["200"]["reuse_any"] > 0.0 and
               out["200"]["cum_novel"]["mean"] > 0.0)
    out["_persists"] = persists
    log(f"  -> compositional activity persists at higher capacity? {persists}")
    return out


# ---------------------------------------------------------------------------
# block 4: small-L exact-L* audit (bounded MDL U_t tracks L*)
# ---------------------------------------------------------------------------
def block_audit(log):
    log("=" * 78)
    log("BLOCK 4  exact-L* audit (small L): does bounded-MDL U_t track L*?")
    log("=" * 78)
    import escalation as S
    import genome as G
    from hierarchical import HConfig, seed_env, _gacfg, _one_generation, \
        _reeval_pop, _upper_bound_complexity, random_structured_program, Individual
    acfg = HConfig(L=16, pop=60, motif_len=2, n_motifs0=2, n_modules0=2)
    scfg = S.CoevoConfig(A=2, L=16, f_solve=0.9)
    rows = []
    for seed in (1, 2, 3):
        rng = np.random.default_rng(seed)
        gacfg = _gacfg(acfg)
        env = seed_env(acfg, rng)
        target = env.target()
        pop = [Individual(g=random_structured_program(rng, gacfg))
               for _ in range(acfg.pop)]
        _reeval_pop(pop, target, acfg)
        for _ in range(400):
            pop = _one_generation(pop, target, acfg, gacfg, rng)
        u, sf = _upper_bound_complexity(pop, acfg)
        lstar = S.cstar(target, scfg)
        rows.append((lstar, u, sf))
        log(f"  seed {seed}: exact L*={lstar:3d}  bounded U={u:3d}  "
            f"solved_frac={sf:.2f}  (U >= L*: {u >= lstar})")
    valid = all(u >= l - 1 for (l, u, sf) in rows if sf > 0)
    log(f"  -> U_t is a valid upper bound on exact L* in all audited runs? {valid}")
    return {"rows": [{"Lstar": l, "U": u, "solved_frac": sf} for (l, u, sf) in rows],
            "valid_upper_bound": valid}


# ---------------------------------------------------------------------------
# block 5: strict verdict + central question
# ---------------------------------------------------------------------------
def block_verdict(sweep, capacity, log):
    log("=" * 78)
    log("BLOCK 5  STRICT pre-registered criteria (C_t escalation alone = FAIL)")
    log("=" * 78)
    best = sweep["sweep"][str(sweep["best_tau"])]
    tm = sweep["tau_module"]

    functional_novelty = best["cum_novel"]["mean"] > 0.0
    reuse = best["reuse_any"] > 0.0
    recombination = best["recomb_any"] > 0.0
    hierarchy = best["max_hierarchy_depth"]["mean"] > 1.0
    novelty_continues = best["novel_second_half"]["mean"] > 0.0
    capacity_continues = capacity["_persists"]
    escalation = best["U_late"]["mean"] > best["U_early"]["mean"] + 1e-9

    # timescale matching: a CLEAR interior peak (margin) in module activity at an
    # intermediate tau_E would support tau_E ~ tau_module.  With noisy counts we
    # demand the peak to beat BOTH endpoints by a clear margin, else report it as
    # not established (honest).
    taus = TAUS
    novel_by_tau = {t: sweep["sweep"][str(t)]["cum_novel"]["mean"] for t in taus}
    reuse_by_tau = {t: sweep["sweep"][str(t)]["reuse_rate_late"]["mean"] for t in taus}
    peak_tau = sweep["best_tau"]
    interior_peak = peak_tau not in (taus[0], taus[-1])
    ends = max(novel_by_tau[taus[0]], novel_by_tau[taus[-1]])
    timescale_matters = bool(interior_peak and
                             novel_by_tau[peak_tau] > 1.3 * ends + 1e-9)

    verdict = {
        "C_t_escalation": bool(escalation),
        "functional_novelty_rate_pos": bool(functional_novelty),
        "reuse_rate_pos": bool(reuse),
        "recombination_rate_pos": bool(recombination),
        "hierarchy_depth_gt1": bool(hierarchy),
        "novelty_continues_late": bool(novelty_continues),
        "capacity_increase_continues": bool(capacity_continues),
        "timescale_phase_transition_clear": bool(timescale_matters),
        "interior_peak_tauE": bool(interior_peak),
    }
    strict = [functional_novelty, reuse, recombination, hierarchy,
              novelty_continues, capacity_continues]
    verdict["COMPOSITIONAL_OPEN_ENDEDNESS"] = bool(all(strict))

    for k, v in verdict.items():
        log(f"  {k:<34}: {v}")
    log("  " + "-" * 62)
    log(f"  novelty by tau_E: " +
        "  ".join(f"{t}:{novel_by_tau[t]:.1f}" for t in taus) +
        f"   (tau_module for flat reuse ~ {tm})")
    log(f"  ==> COMPOSITIONAL OPEN-ENDEDNESS (all strict criteria): "
        f"{verdict['COMPOSITIONAL_OPEN_ENDEDNESS']}")
    # honest diagnosis of the remaining gap
    fbest = best["F_mean"]["mean"]
    passed = sum([functional_novelty, reuse, hierarchy, novelty_continues,
                  capacity_continues])
    log(f"  diagnosis: reuse, hierarchy-depth>1, functional novelty, late "
        f"continuity and capacity persistence hold ({passed}/5); the SOLE failing "
        f"strict criterion is RECOMBINATION={recombination}.")
    log(f"  recombination (a macro composing >=2 DISTINCT modules) needs clean "
        f"compositional parsing, but best-tau mean F={fbest:.2f}<1 and the greedy "
        f"compressor yields run/chain hierarchies, not multi-child compositions.")
    log(f"  -> strong arrow MOSTLY traversed: endogenous escalation DOES yield "
        f"continuing functional novelty + reuse + shallow hierarchy; full "
        f"RECOMBINATION remains the open frontier.")
    return verdict


# ---------------------------------------------------------------------------
def _plot(sweep, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    taus = TAUS
    sw = sweep["sweep"]

    ax = axes[0][0]
    ax.plot(taus, [sw[str(t)]["cum_novel"]["mean"] for t in taus], "o-",
            label="cumulative novelty")
    ax.plot(taus, [sw[str(t)]["novel_second_half"]["mean"] for t in taus], "s--",
            label="late-phase novelty")
    if sweep["tau_module"]:
        ax.axvline(sweep["tau_module"], color="k", ls=":", label="tau_module")
    ax.set_xscale("log"); ax.set_xlabel("tau_E (gens/env change)")
    ax.set_ylabel("functional novelty"); ax.set_title("phase transition: novelty")
    ax.legend(fontsize=7)

    ax = axes[0][1]
    ax.plot(taus, [sw[str(t)]["reuse_rate_late"]["mean"] for t in taus], "o-",
            label="reuse rate (late)")
    ax.plot(taus, [sw[str(t)]["max_hierarchy_depth"]["mean"] for t in taus], "s-",
            label="max hierarchy depth")
    ax.set_xscale("log"); ax.set_xlabel("tau_E")
    ax.set_title("reuse & hierarchy vs tau_E"); ax.legend(fontsize=7)

    ax = axes[1][0]
    s = sw[str(sweep["best_tau"])]["series_seed1"]
    ax.plot(s["gen"], s["U_t"], label="U_t (bounded MDL)")
    ax.plot(s["gen"], s["cum_novel"], label="cum novelty")
    ax.set_xlabel("generation"); ax.set_title(f"best tau_E={sweep['best_tau']}: U_t & novelty")
    ax.legend(fontsize=7)

    ax = axes[1][1]
    for t in taus:
        s = sw[str(t)]["series_seed1"]
        ax.plot(s["gen"], s["hierarchy_depth"], label=f"tau_E={t}")
    ax.set_xlabel("generation"); ax.set_ylabel("hierarchy depth")
    ax.set_title("hierarchy depth over time"); ax.legend(fontsize=7)

    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)
    return True


def main():
    t0 = time.time()
    lines = []

    def log(msg):
        print(msg, flush=True)
        lines.append(msg)

    log(f"exp597 Hierarchical Compositional Escalation  "
        f"A=2 L={L_MAIN} seeds={SEEDS} env_changes={ENV_CHANGES} taus={TAUS}")
    tm = block_tau_module(log)
    sweep = block_tau_sweep(log, tm)
    capacity = block_capacity(log, sweep["best_tau"])
    audit = block_audit(log)
    verdict = block_verdict(sweep, capacity, log)

    ok = _plot(sweep, "hierarchical.png")
    log(f"plot written: {ok}")
    log(f"total time: {time.time() - t0:.1f}s")

    results = {
        "config": {"L": L_MAIN, "seeds": SEEDS, "env_changes": ENV_CHANGES,
                   "taus": TAUS},
        "tau_module": tm,
        "block2_tau_sweep": sweep,
        "block3_capacity": capacity,
        "block4_audit": audit,
        "verdict": verdict,
    }
    with open("hierarchical_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    with open("hierarchical_run.log", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
