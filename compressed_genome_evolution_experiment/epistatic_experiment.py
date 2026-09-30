"""exp591 -- Epistatic Module Audit (runner).

Tests the strongest form of the thesis: it is not compression per se, but the
*alignment* of the compressed representation's module structure with the
environment's epistatic interaction structure, that turns the representation into
an advantageous mutation operator (one that can cross fitness valleys).

Design (see epistatic.py):
    fitness  = block-epistatic (a block scores only when ALL its bits match)
    reps     = A_bit, A_block(oracle), C(contiguous blocks), R, C_shuffle
    conditions = E_aligned / E_shifted / E_random  (does Pi match C's modules?)

Pre-registered decisive pattern:
    E_aligned:  P_useful(C) > P_useful(A_bit),  eta(C) > eta(A_bit),
                and  C ~= A_block  <<  A_bit, R, C_shuffle   (re-adaptation steps)
    E_shifted, E_random:  that advantage VANISHES (C ~= R, and C != A_block).

Writes epistatic_results.json, epistatic_run.log, epistatic.png.
"""

from __future__ import annotations

import json
import time
from statistics import mean, pstdev

import numpy as np

import epistatic as E

L = 16
P = 4
B = L // P
N_FAMILIES = 12
READAPT_SEEDS = 30
MAX_STEPS = 6000
THRESHOLD = 0.999          # essentially "all blocks correct"
N_NEI = 4000
REP_NAMES = ["A_bit", "A_block", "C", "R", "C_shuffle"]


def _agg(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return {"mean": None, "std": None, "n": 0}
    return {"mean": float(mean(xs)),
            "std": float(pstdev(xs)) if len(xs) > 1 else 0.0, "n": len(xs)}


def make_Pi(cond, fam, rng):
    if cond == "aligned":
        return E.contiguous_partition(L, P, 0)
    if cond == "shifted":
        return E.contiguous_partition(L, P, P // 2)
    if cond == "random":
        return E.random_partition(L, P, np.random.default_rng(50000 + fam))
    raise ValueError(cond)


def build_reps(Pi, fam):
    return {
        "A_bit": E.rep_A_bit(L),
        "A_block": E.rep_A_block(L, Pi),
        "C": E.rep_C(L, P),
        "R": E.rep_R(L, P, np.random.default_rng(60000 + fam)),
        "C_shuffle": E.rep_C_shuffle(L, P, np.random.default_rng(70000 + fam)),
    }


def run_condition(cond, log):
    acc = {r: {k: [] for k in
               ["P_useful", "E_dF_pos", "eta_aggregate", "mean_dP",
                "succ", "mean_steps", "plateauF"]} for r in REP_NAMES}
    for fam in range(N_FAMILIES):
        rng = np.random.default_rng(80000 + fam)
        E0 = rng.integers(0, 2, L)
        E1 = 1 - E0                                  # maximally-wrong valley
        Pi = make_Pi(cond, fam, rng)
        reps = build_reps(Pi, fam)
        for name, rep in reps.items():
            nb = E.epi_neighborhood(rep, E0, E1, Pi,
                                    np.random.default_rng(90000 + fam), N_NEI)
            ra = E.epi_readapt_summary(rep, E0, E1, Pi, THRESHOLD, MAX_STEPS,
                                       READAPT_SEEDS, base_seed=100000 + fam * 100)
            a = acc[name]
            a["P_useful"].append(nb.P_useful)
            a["E_dF_pos"].append(nb.E_dF_pos)
            a["eta_aggregate"].append(nb.eta_aggregate)
            a["mean_dP"].append(nb.mean_dP)
            a["succ"].append(ra["success_rate"])
            a["mean_steps"].append(ra["mean_steps"])
            a["plateauF"].append(ra["mean_plateau_F"])
    out = {}
    for name in REP_NAMES:
        out[name] = {k: _agg(v) for k, v in acc[name].items()}
        m = out[name]
        log(f"  {name:>10}: Puseful={m['P_useful']['mean']:.3f} "
            f"E[dF|>0]={m['E_dF_pos']['mean']:.3f} "
            f"eta={m['eta_aggregate']['mean']:.3f} d_P={m['mean_dP']['mean']:.1f} "
            f"| succ={m['succ']['mean']:.2f} "
            f"steps={_fmt(m['mean_steps']['mean'])}")
    return out


def _fmt(v):
    return "n/a" if v is None else f"{v:.0f}"


def decide(results, log):
    log("=" * 74)
    log("PRE-REGISTERED DECISION")
    log("=" * 74)
    verdicts = {}
    for cond in ["aligned", "shifted", "random"]:
        r = results[cond]
        pC = r["C"]["P_useful"]["mean"]
        pA = r["A_bit"]["P_useful"]["mean"]
        eC = r["C"]["eta_aggregate"]["mean"]
        eA = r["A_bit"]["eta_aggregate"]["mean"]
        tC = r["C"]["mean_steps"]["mean"]
        tAbit = r["A_bit"]["mean_steps"]["mean"]
        tAblk = r["A_block"]["mean_steps"]["mean"]
        tR = r["R"]["mean_steps"]["mean"]
        tCs = r["C_shuffle"]["mean_steps"]["mean"]

        useful_gt = pC > pA + 1e-9
        eta_gt = eC > eA + 1e-9
        # C ~= A_block (within 35%) and C << the ungr­adient reps (A_bit,R,C_shuffle)
        def _lt(a, b):
            return (a is not None and b is not None and a < b)
        near_block = (tC is not None and tAblk is not None and
                      abs(tC - tAblk) <= 0.35 * max(tC, tAblk))
        faster_than_rest = _lt(tC, tAbit) and _lt(tC, tR) and _lt(tC, tCs)
        advantage = useful_gt and eta_gt and near_block and faster_than_rest
        verdicts[cond] = {
            "P_useful_C": pC, "P_useful_Abit": pA,
            "eta_C": eC, "eta_Abit": eA,
            "steps": {"C": tC, "A_block": tAblk, "A_bit": tAbit,
                      "R": tR, "C_shuffle": tCs},
            "C_advantage_present": advantage,
            "useful_gt": useful_gt, "eta_gt": eta_gt,
            "C_near_A_block": near_block, "C_faster_than_rest": faster_than_rest,
        }
        log(f"  [{cond}] C_advantage={advantage}  "
            f"(P_useful C={pC:.3f} vs A_bit={pA:.3f}; eta C={eC:.3f} vs {eA:.3f}; "
            f"steps C={_fmt(tC)} A_block={_fmt(tAblk)} A_bit={_fmt(tAbit)} "
            f"R={_fmt(tR)} C_shuf={_fmt(tCs)})")

    core = (verdicts["aligned"]["C_advantage_present"]
            and not verdicts["shifted"]["C_advantage_present"]
            and not verdicts["random"]["C_advantage_present"])
    log("")
    log(f"  CORE (alignment, not compression per se, is causal): {core}")
    log("    - aligned: C gains a fitness gradient A_bit lacks and matches the "
        "hand-designed block operator A_block")
    log("    - shifted & random: the SAME compression C loses the advantage when "
        "its modules no longer match the epistatic blocks")
    verdicts["core_alignment_is_causal"] = core
    return verdicts


def make_figure(results, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    conds = ["aligned", "shifted", "random"]
    colors = {"A_bit": "crimson", "A_block": "seagreen", "C": "steelblue",
              "R": "darkorange", "C_shuffle": "gray"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    # panel 1: P_useful ; panel 2: eta ; panel 3: re-adaptation steps
    def bars(ax, key, title, cap=None):
        for i, cond in enumerate(conds):
            for j, name in enumerate(REP_NAMES):
                v = results[cond][name][key]["mean"]
                if v is None:
                    v = cap or 0
                ax.bar(i * (len(REP_NAMES) + 1) + j, v, color=colors[name],
                       label=name if i == 0 else None)
        ax.set_xticks([i * (len(REP_NAMES) + 1) + (len(REP_NAMES) - 1) / 2
                       for i in range(len(conds))])
        ax.set_xticklabels(conds, fontsize=9)
        ax.set_title(title, fontsize=10)
    bars(axes[0], "P_useful", "P_useful  (fraction of moves with dF>0)")
    axes[0].legend(fontsize=7, ncol=2)
    bars(axes[1], "eta_aggregate", "eta  (fitness gained / positions changed)")
    # steps: cap failures
    for i, cond in enumerate(conds):
        for j, name in enumerate(REP_NAMES):
            v = results[cond][name]["mean_steps"]["mean"]
            v = MAX_STEPS if v is None else v
            axes[2].bar(i * (len(REP_NAMES) + 1) + j, v, color=colors[name])
    axes[2].set_xticks([i * (len(REP_NAMES) + 1) + (len(REP_NAMES) - 1) / 2
                        for i in range(len(conds))])
    axes[2].set_xticklabels(conds, fontsize=9)
    axes[2].set_title("re-adaptation steps (valley crossing; lower=better)",
                      fontsize=10)
    fig.suptitle("exp591  Epistatic Module Audit -- C gains A_block's power ONLY "
                 "when its modules align with the epistatic blocks", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=120)
    import matplotlib.pyplot as _plt
    _plt.close(fig)
    return True


def main():
    t0 = time.time()
    lines = []

    def log(msg):
        print(msg, flush=True)
        lines.append(msg)

    log(f"exp591 Epistatic Module Audit  L={L} p={P} B={B} "
        f"families={N_FAMILIES} readapt_seeds={READAPT_SEEDS} max_steps={MAX_STEPS}")
    results = {}
    for cond in ["aligned", "shifted", "random"]:
        log("=" * 74)
        log(f"CONDITION  E_{cond}")
        log("=" * 74)
        results[cond] = run_condition(cond, log)

    decision = decide(results, log)
    fig_ok = make_figure(results, "epistatic.png")
    log(f"figure: {'epistatic.png' if fig_ok else '(matplotlib unavailable)'}")
    dt = time.time() - t0
    log("=" * 74)
    log(f"done in {dt:.1f}s")

    out = {
        "config": {"L": L, "p": P, "B": B, "n_families": N_FAMILIES,
                   "readapt_seeds": READAPT_SEEDS, "max_steps": MAX_STEPS,
                   "threshold": THRESHOLD},
        "conditions": results,
        "decision": decision,
        "runtime_seconds": dt,
    }
    with open("epistatic_results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    with open("epistatic_run.log", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote epistatic_results.json and epistatic_run.log")


if __name__ == "__main__":
    main()
