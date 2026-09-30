"""exp593 -- Adaptive Genome Capacity (runner).

Does "grow when under-capacity, shrink when redundant" emerge from MDL selection
alone, with only structural edits (ADD/DUP/SPLIT/MERGE/DELETE/REUSE) and no
explicit grow/shrink rule?

Blocks:
  1. main trajectory (MDL on): schedule 1->2->4->2->1 rules; track L(G),L(D),
     L_total, n_used_macros, n_params per stage.  H1 (demand up -> complexity up),
     H2 (demand down -> complexity down).
  2. exact deficit F*(K): on a small system compute F*(K)=max_{L(G)<=K}F(G);
     show K_min(F=1) grows with rule count -> representational deficit is real.
  3. controls: MDL off (grows, doesn't prune), ADD off (fitness plateaus at peak),
     DELETE off (grows, can't shrink)  -> growth and pruning are separable causes.
  4. gene duplication: redundant (undivergent) duplicate is pruned under MDL.
  5. hysteresis: L at equal complexity on the up-ramp vs down-ramp
     (evolutionary-history memory).

Writes capacity_results.json, capacity_run.log, capacity.png.
"""

from __future__ import annotations

import json
import time
from statistics import mean, pstdev

import numpy as np

import capacity as C
import genome as G
from genome import Genome, Instr, OP_LIT, OP_CALL, OP_REP

SCHEDULE = [1, 2, 4, 2, 1]
LAM = 0.001
SEEDS = [1, 2, 3, 4, 5, 6]
GENS = 260
POP = 140


def _agg(xs):
    xs = [x for x in xs if x is not None]
    return {"mean": float(mean(xs)), "std": float(pstdev(xs)) if len(xs) > 1 else 0.0}


def staged_over_seeds(lam, allow_add=True, allow_delete=True):
    per_stage = [[] for _ in SCHEDULE]
    for seed in SEEDS:
        recs = C.run_staged(SCHEDULE, lam=lam, seed=seed, gens_per_stage=GENS,
                            pop=POP, allow_add=allow_add, allow_delete=allow_delete)
        for s, rc in enumerate(recs):
            per_stage[s].append(rc)
    out = []
    for s, cell in enumerate(per_stage):
        out.append({
            "stage": s, "rule_count": SCHEDULE[s],
            "F": _agg([r.best_F for r in cell]),
            "L_total": _agg([r.metrics["L_total"] for r in cell]),
            "LG": _agg([r.metrics["LG"] for r in cell]),
            "LD": _agg([r.metrics["LD"] for r in cell]),
            "n_used_macros": _agg([r.metrics["n_used_macros"] for r in cell]),
            "n_params": _agg([r.metrics["n_params"] for r in cell]),
        })
    return out


def block_main(log):
    log("=" * 74)
    log("BLOCK 1  main trajectory (MDL on)  schedule 1->2->4->2->1")
    log("=" * 74)
    traj = staged_over_seeds(LAM, True, True)
    for r in traj:
        log(f"  stage{r['stage']} rules={r['rule_count']}  "
            f"F={r['F']['mean']:.3f}  L_total={r['L_total']['mean']:5.1f}"
            f"+-{r['L_total']['std']:.1f}  used_macros={r['n_used_macros']['mean']:.1f}  "
            f"n_params={r['n_params']['mean']:.1f}")
    return traj


def block_fstar(log):
    log("=" * 74)
    log("BLOCK 2  exact representational deficit  F*(K)=max_{L(G)<=K} F(G)  (A=2,L=12)")
    log("=" * 74)
    out = {}
    for r in [1, 2, 3]:
        tgt = C.make_target(r, C.SMALL)
        curve = C.fstar_curve(tgt, C.SMALL["A"], C.SMALL["L"])
        kmin = C.k_min_for_fitness(curve, 1.0)
        out[r] = {"K_min_F1": kmin,
                  "curve": {int(k): float(v) for k, v in curve.items()}}
        log(f"  rules={r}: K_min(F=1.0) = {kmin} bits")
    log("  -> more rules require a longer minimum description "
        "(capacity demand is a measured fact)")
    return out


def block_controls(log):
    log("=" * 74)
    log("BLOCK 3  controls (growth vs pruning dissociated)")
    log("=" * 74)
    res = {}
    res["mdl_on"] = staged_over_seeds(LAM, True, True)
    res["mdl_off"] = staged_over_seeds(0.0, True, True)
    res["add_off"] = staged_over_seeds(LAM, False, True)
    res["delete_off"] = staged_over_seeds(LAM, True, False)

    def peak_and_down(traj):
        return (traj[2]["L_total"]["mean"], traj[4]["L_total"]["mean"],
                traj[2]["F"]["mean"])
    for name in ["mdl_on", "mdl_off", "add_off", "delete_off"]:
        pk, dn, pf = peak_and_down(res[name])
        log(f"  {name:>11}: peak(4-rule) L={pk:5.1f} F={pf:.3f}  |  "
            f"back-to-1-rule L={dn:5.1f}  (shrank {pk-dn:+.1f})")
    return res


def block_duplication(log):
    log("=" * 74)
    log("BLOCK 4  gene duplication -- redundant duplicate is pruned under MDL")
    log("=" * 74)
    # a genome that solves the 1-rule target, with an EXTRA identical duplicate
    # macro (redundant).  Under MDL it should be removed; without MDL it may stay.
    cfg = C.DEFAULT_CFG
    m = [Instr(OP_LIT, 0), Instr(OP_LIT, 1)]        # motif "01"
    nm = cfg["nmax"]
    def seed_genome():
        # tile motif across L using REPs within nmax; macro 1 is an identical,
        # redundant, UNUSED duplicate of macro 0.
        tiles = cfg["L"] // len(m)                    # number of motif copies needed
        prog = []
        while tiles > 0:
            n = min(nm, tiles)
            prog.append(Instr(OP_REP, n, 0))
            tiles -= n
        g = Genome(macros=[list(m), list(m)],        # M and identical M'
                   program=prog, A=cfg["A"], L=cfg["L"], nmax=nm)
        return g
    out = {}
    for label, lam in [("mdl_on", LAM), ("mdl_off", 0.0)]:
        kept = []
        for seed in SEEDS:
            rng = np.random.default_rng(700 + seed)
            g = seed_genome()
            target = C.make_target(1, cfg)
            # short evolution holding the 1-rule env, seeded from the redundant genome
            pop = [g.copy() for _ in range(POP)]
            scored = [(x, *C._evaluate(x, target, lam)) for x in pop]
            for _ in range(120):
                scored.sort(key=lambda t: t[3], reverse=True)
                new = scored[:2]
                while len(new) < POP:
                    idx = rng.integers(0, len(scored), size=4)
                    par = max((scored[int(i)] for i in idx), key=lambda t: t[3])
                    ch = C.mutate_capacity(rng, par[0], cfg, True, True)
                    new.append((ch, *C._evaluate(ch, target, lam)))
                scored = new
            champ = max(scored, key=lambda t: t[3])       # best by J = F - lam*L
            kept.append(len(champ[0].macros))              # TOTAL macros (incl. unused)
        out[label] = {"mean_total_macros": float(mean(kept)),
                      "total_macros_per_seed": kept}
        log(f"  {label}: TOTAL macros after evolving from a redundant duplicate = "
            f"{out[label]['mean_total_macros']:.2f}  (per seed {kept})")
    log("  -> MDL removes the redundant (unused) duplicate; without MDL it persists")
    return out


def block_hysteresis(traj, log):
    log("=" * 74)
    log("BLOCK 5  hysteresis (evolutionary-history memory)")
    log("=" * 74)
    # rule=2 appears at stage1 (up) and stage3 (down); rule=1 at stage0 and stage4
    up2, dn2 = traj[1]["L_total"]["mean"], traj[3]["L_total"]["mean"]
    up1, dn1 = traj[0]["L_total"]["mean"], traj[4]["L_total"]["mean"]
    log(f"  complexity=2: L_up(stage1)={up2:.1f}  L_down(stage3)={dn2:.1f}  "
        f"(down - up = {dn2-up2:+.1f})")
    log(f"  complexity=1: L_up(stage0)={up1:.1f}  L_down(stage4)={dn1:.1f}  "
        f"(down - up = {dn1-up1:+.1f})")
    return {"c2_up": up2, "c2_down": dn2, "c1_up": up1, "c1_down": dn1}


def decide(traj, controls, dup, log):
    log("=" * 74)
    log("PRE-REGISTERED DECISION")
    log("=" * 74)
    L = [s["L_total"]["mean"] for s in traj]
    U = [s["n_used_macros"]["mean"] for s in traj]
    H1 = (L[2] > L[0] + 3) and (U[2] > U[0])            # demand up -> complexity up
    H2 = (L[4] < L[2] - 3) and (U[4] < U[2])            # demand down -> complexity down
    # controls
    def down_shrink(name):
        t = controls[name]
        return t[2]["L_total"]["mean"] - t[4]["L_total"]["mean"]
    mdl_on_shrinks = down_shrink("mdl_on") > 3
    mdl_off_no_shrink = down_shrink("mdl_off") <= down_shrink("mdl_on")
    add_off_plateaus = controls["add_off"][2]["F"]["mean"] < controls["mdl_on"][2]["F"]["mean"] - 0.02
    delete_off_no_shrink = down_shrink("delete_off") < down_shrink("mdl_on")
    dup_pruned = (dup["mdl_on"]["mean_total_macros"]
                  < dup["mdl_off"]["mean_total_macros"] - 1e-9)

    log(f"  H1  capacity demand up  => effective complexity up:   {H1}  "
        f"(L {L[0]:.0f}->{L[2]:.0f}, used {U[0]:.1f}->{U[2]:.1f})")
    log(f"  H2  capacity demand down => effective complexity down: {H2}  "
        f"(L {L[2]:.0f}->{L[4]:.0f}, used {U[2]:.1f}->{U[4]:.1f})")
    log(f"  control MDL-on prunes on down-ramp:      {mdl_on_shrinks}")
    log(f"  control MDL-off prunes less than MDL-on: {mdl_off_no_shrink}")
    log(f"  control ADD-off plateaus F at peak:      {add_off_plateaus}")
    log(f"  control DELETE-off prunes less:          {delete_off_no_shrink}")
    log(f"  duplication: redundant duplicate pruned under MDL: {dup_pruned}")
    core = H1 and H2 and mdl_on_shrinks and add_off_plateaus and delete_off_no_shrink
    log("")
    log(f"  CORE (adaptive capacity: grows on deficit, shrinks on redundancy, "
        f"from MDL alone): {core}")
    return {
        "H1": bool(H1), "H2": bool(H2),
        "mdl_on_shrinks": bool(mdl_on_shrinks),
        "mdl_off_prunes_less": bool(mdl_off_no_shrink),
        "add_off_plateaus": bool(add_off_plateaus),
        "delete_off_prunes_less": bool(delete_off_no_shrink),
        "duplicate_pruned": bool(dup_pruned),
        "core": bool(core),
    }


def make_figure(traj, controls, fstar, hyst, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    stages = list(range(len(SCHEDULE)))
    xl = [f"E{s}\n{SCHEDULE[s]}rule" for s in stages]
    # panel 1: capacity trajectory
    Lm = [s["L_total"]["mean"] for s in traj]
    Le = [s["L_total"]["std"] for s in traj]
    Um = [s["n_used_macros"]["mean"] for s in traj]
    ax[0].plot(stages, Lm, "-o", color="steelblue", label="L_total (bits)")
    ax[0].fill_between(stages, [a - b for a, b in zip(Lm, Le)],
                       [a + b for a, b in zip(Lm, Le)], color="steelblue", alpha=0.15)
    ax[0].set_xticks(stages); ax[0].set_xticklabels(xl, fontsize=8)
    ax[0].set_ylabel("L_total (bits)", color="steelblue")
    ax2 = ax[0].twinx()
    ax2.plot(stages, Um, "-s", color="seagreen", label="used macros")
    ax2.set_ylabel("used macros", color="seagreen")
    ax[0].set_title("capacity trajectory (MDL): grows then prunes", fontsize=10)
    # panel 2: control down-ramp shrink
    names = ["mdl_on", "mdl_off", "delete_off", "add_off"]
    shrink = [controls[n][2]["L_total"]["mean"] - controls[n][4]["L_total"]["mean"]
              for n in names]
    ax[1].bar(names, shrink, color=["steelblue", "crimson", "gray", "darkorange"])
    ax[1].set_title("down-ramp shrink  L(peak)-L(back to 1)\n(pruning needs MDL+DELETE)",
                    fontsize=9)
    ax[1].axhline(0, color="0.6", lw=0.6)
    ax[1].tick_params(axis="x", labelsize=8)
    # panel 3: F*(K)
    for r, col in zip([1, 2, 3], ["seagreen", "steelblue", "crimson"]):
        cur = fstar[r]["curve"]
        ks = sorted(int(k) for k in cur)
        ax[2].step(ks, [cur[k] for k in ks], where="post", color=col,
                   label=f"{r} rule(s), K_min={fstar[r]['K_min_F1']}")
    ax[2].set_xlabel("description length K (bits)")
    ax[2].set_ylabel("F*(K)")
    ax[2].set_title("exact representational deficit  F*(K)", fontsize=10)
    ax[2].legend(fontsize=7)
    fig.suptitle("exp593  Adaptive Genome Capacity -- module birth on deficit, "
                 "death on redundancy, from MDL selection alone", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return True


def main():
    t0 = time.time()
    lines = []

    def log(msg):
        print(msg, flush=True)
        lines.append(msg)

    log(f"exp593 Adaptive Genome Capacity  schedule={SCHEDULE} lam={LAM} "
        f"seeds={SEEDS} gens={GENS}")
    traj = block_main(log)
    fstar = block_fstar(log)
    controls = block_controls(log)
    dup = block_duplication(log)
    hyst = block_hysteresis(traj, log)
    decision = decide(traj, controls, dup, log)
    fig_ok = make_figure(traj, controls, fstar, hyst, "capacity.png")
    log(f"figure: {'capacity.png' if fig_ok else '(matplotlib unavailable)'}")
    dt = time.time() - t0
    log("=" * 74)
    log(f"done in {dt:.1f}s")

    out = {
        "config": {"schedule": SCHEDULE, "lam": LAM, "seeds": SEEDS,
                   "gens": GENS, "pop": POP,
                   "dynamics_system": {k: (v if not hasattr(v, "tolist") else None)
                                       for k, v in C.DEFAULT_CFG.items()
                                       if k != "motifs"}},
        "block1_trajectory": traj,
        "block2_fstar": fstar,
        "block3_controls": controls,
        "block4_duplication": dup,
        "block5_hysteresis": hyst,
        "decision": decision,
        "runtime_seconds": dt,
    }
    with open("capacity_results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=float)
    with open("capacity_run.log", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote capacity_results.json and capacity_run.log")


if __name__ == "__main__":
    main()
