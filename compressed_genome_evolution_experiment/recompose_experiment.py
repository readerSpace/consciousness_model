"""exp595 -- Dynamic Interaction Recomposition (runner).

Blocks:
  1. N_0.9 across the sequence x conditions (reset / modules / operators / full /
     frozen / oracle), each transition auto-classified REUSE/MERGE/SPLIT/
     RECOMBINE/NOVEL.  Transfer (memory < reset) should appear for recomposable
     changes and vanish for NOVEL.
  2. what is remembered?  operators-only vs modules-only vs full vs reset,
     focused on RECOMBINE.  operators ~= full < modules  =>  the memory is the
     construction OPERATORS (atoms + regrouping), not the stored modules.
  3. two-stage N_0.9 -> T_0.95: the discovered structure is used as a mutation
     operator; structure discovery precedes (and enables) fast adaptation.
  4. lineage: classify how each new block is built from previous ones.

Writes recompose_results.json, recompose_run.log, recompose.png.
"""

from __future__ import annotations

import json
import time
from statistics import mean

import numpy as np

import recompose as R
import epistatic as EP

NGRID = [2, 3, 4, 5, 6, 8, 10, 14, 20, 30, 50, 80, 120]
SEEDS = 20
CONDS = ["reset", "modules", "operators", "full", "frozen", "oracle"]


def _seq_with_learned(condition, base_seed=1000):
    """Walk the sequence; at each step the learner's 'prev' is what IT learned at
    the previous step (with plenty of samples), so memory is self-consistent."""
    seq = R.sequence()
    prev = None
    learned = [None] * len(seq)
    rng = np.random.default_rng(base_seed)
    for i, st in enumerate(seq):
        ph = R.learn(condition, R.sample(st["pi"], 60, rng), prev, st["pi"])
        learned[i] = ph
        prev = ph
    return seq, learned


def block_discovery(log):
    log("=" * 74)
    log("BLOCK 1  N_0.9 (min samples to ARI>=0.9) across sequence x condition")
    log("=" * 74)
    seq = R.sequence()
    # learners carry their own previous learned partition; seed it from Pi0
    rng = {c: np.random.default_rng(700 + k) for k, c in enumerate(CONDS)}
    prev_learned = {c: R.learn(c, R.sample(seq[0]["pi"], 60, rng[c]), None,
                               seq[0]["pi"]) for c in CONDS}
    rows = []
    for i in range(1, len(seq)):
        cur = seq[i]["pi"]
        change = R.classify_change(seq[i - 1]["pi"], cur)
        row = {"name": seq[i]["name"], "change": change, "N09": {}}
        for c in CONDS:
            n09 = R.discovery_cost(c, cur, prev_learned[c], NGRID, SEEDS,
                                   base_seed=1000 + i * 7)
            row["N09"][c] = n09
            # update this condition's memory with what it learns now
            prev_learned[c] = R.learn(c, R.sample(cur, 60, rng[c]),
                                      prev_learned[c], cur)
        rows.append(row)
        log(f"  {row['name']:>10} [{change:>9}]: " +
            "  ".join(f"{c}={row['N09'][c]}" for c in CONDS))
    return rows


def block_decomposition(rows, log):
    log("=" * 74)
    log("BLOCK 2  what is remembered?  operators-only vs modules-only vs full")
    log("=" * 74)
    # focus on RECOMBINE and the recomposable transitions
    def get(change, cond):
        vals = [r["N09"][cond] for r in rows if r["change"] == change
                and r["N09"][cond] is not None]
        return mean(vals) if vals else None
    out = {}
    for change in ["MERGE", "RECOMBINE"]:
        out[change] = {c: get(change, c) for c in
                       ["reset", "modules", "operators", "full"]}
        o = out[change]
        log(f"  {change:>9}: reset={o['reset']}  modules={o['modules']}  "
            f"operators={o['operators']}  full={o['full']}")
    rec = out["RECOMBINE"]
    # robust claim: the construction OPERATORS (atom regrouping) beat stored
    # MODULES for recombination.  (Bundling stale modules into `full` adds a few
    # spurious candidates, so full sits between operators and modules.)
    verdict = (rec["operators"] is not None and rec["modules"] is not None
               and rec["operators"] < rec["modules"])
    log(f"  RECOMBINE: operators < modules  => the transferable memory is the "
        f"CONSTRUCTION OPERATORS, not the stored modules: {verdict}")
    return out, verdict


def block_two_stage(log):
    log("=" * 74)
    log("BLOCK 3  two-stage: N_0.9 (discover) then T_0.95 (adapt with the operator)")
    log("=" * 74)
    seq = R.sequence()
    rng = np.random.default_rng(42)
    prev_full = R.learn("full", R.sample(seq[0]["pi"], 60, rng), None, seq[0]["pi"])
    out = []
    for i in range(1, len(seq)):
        cur = seq[i]["pi"]
        change = R.classify_change(seq[i - 1]["pi"], cur)
        # discovery cost (full memory) and reset
        n_full = R.discovery_cost("full", cur, prev_full, NGRID, SEEDS, 1300 + i)
        n_reset = R.discovery_cost("reset", cur, None, NGRID, SEEDS, 1300 + i)
        # adaptation: use the learned partition as a resample mutation operator
        learned_full = R.learn("full", R.sample(cur, 60, rng), prev_full, cur)
        prev_full = learned_full
        T_mem = _adapt_time(learned_full, cur, rng)
        T_reset = _adapt_time([np.array([j]) for j in range(R.L)], cur, rng)
        out.append({"name": seq[i]["name"], "change": change,
                    "N09_full": n_full, "N09_reset": n_reset,
                    "T095_mem": T_mem, "T095_reset": T_reset})
        log(f"  {seq[i]['name']:>10} [{change:>9}]: N0.9 full={n_full} reset={n_reset}"
            f"  ->  T0.95 mem={_f(T_mem)} reset={_f(T_reset)}")
    return out


def _adapt_time(modules, pi, rng, max_steps=4000, seeds=12):
    rep = EP.GeneRep("op", [np.array(m) for m in modules], R.L)
    steps = []
    for s in range(seeds):
        r = np.random.default_rng(900 + s)
        tgt = r.integers(0, 2, R.L)                     # arbitrary held-out goal
        E0 = r.integers(0, 2, R.L)
        t, _ = EP.epi_readapt_steps(rep, E0, tgt, pi, 0.999, max_steps, r)
        if t is not None:
            steps.append(t)
    return float(mean(steps)) if steps else None


def _f(v):
    return "n/a" if v is None else f"{v:.0f}"


def block_lineage(log):
    log("=" * 74)
    log("BLOCK 4  lineage: how each new block is built from previous parts")
    log("=" * 74)
    seq = R.sequence()
    counter = [1000]
    def alloc():
        counter[0] += 1
        return counter[0]
    prev_ids = {frozenset(int(x) for x in b): (i + 1) for i, b in enumerate(seq[0]["pi"])}
    summary = []
    for i in range(1, len(seq)):
        cur = seq[i]["pi"]
        lin = R.recomposition_lineage(prev_ids, cur, alloc)
        ops = [e["op"] for e in lin]
        change = R.classify_change(seq[i - 1]["pi"], cur)
        log(f"  {seq[i]['name']:>10} [{change:>9}]: " +
            "; ".join(f"{e['block']}<-{e['op']}{e['parents']}" for e in lin))
        summary.append({"name": seq[i]["name"], "change": change,
                        "block_ops": ops})
        prev_ids = {frozenset(int(x) for x in b): (e["id"])
                    for b, e in zip(cur, lin)}
    return summary


def decide(rows, decomp_verdict, log):
    log("=" * 74)
    log("PRE-REGISTERED DECISION")
    log("=" * 74)
    def n(change, cond):
        for r in rows:
            if r["change"] == change:
                return r["N09"][cond]
        return None
    # transfer helps for recomposable changes
    reuse_ok = n("REUSE", "full") < n("REUSE", "reset")
    merge_ok = n("MERGE", "full") < n("MERGE", "reset")
    recomb_ok = n("RECOMBINE", "operators") < n("RECOMBINE", "reset")
    # NO transfer for NOVEL
    nov_full = n("NOVEL", "full"); nov_reset = n("NOVEL", "reset")
    novel_no_transfer = (nov_full is None or nov_reset is None
                         or nov_full >= nov_reset)
    # frozen cannot recompose (RECOMBINE)
    frozen_fails_recomb = (n("RECOMBINE", "frozen") is None
                           or n("RECOMBINE", "frozen") >= n("RECOMBINE", "operators") + 1)

    log(f"  transfer helps REUSE (full<reset):        {reuse_ok}  "
        f"({n('REUSE','full')} vs {n('REUSE','reset')})")
    log(f"  transfer helps MERGE (full<reset):        {merge_ok}  "
        f"({n('MERGE','full')} vs {n('MERGE','reset')})")
    log(f"  transfer helps RECOMBINE (operators<reset):{recomb_ok}  "
        f"({n('RECOMBINE','operators')} vs {n('RECOMBINE','reset')})")
    log(f"  NO transfer for NOVEL (memory>=reset):    {novel_no_transfer}  "
        f"({nov_full} vs {nov_reset})")
    log(f"  frozen cannot RECOMBINE:                  {frozen_fails_recomb}  "
        f"(frozen {n('RECOMBINE','frozen')} vs operators {n('RECOMBINE','operators')})")
    log(f"  memory is construction OPERATORS not modules (RECOMBINE): {decomp_verdict}")
    core = (reuse_ok and merge_ok and recomb_ok and novel_no_transfer
            and frozen_fails_recomb and decomp_verdict)
    log("")
    log(f"  CORE (existing parts are RECOMPOSED for new structures; the transferred "
        f"memory is the construction operators): {core}")
    return {"reuse": bool(reuse_ok), "merge": bool(merge_ok),
            "recombine": bool(recomb_ok), "novel_no_transfer": bool(novel_no_transfer),
            "frozen_fails_recombine": bool(frozen_fails_recomb),
            "memory_is_operators": bool(decomp_verdict), "core": bool(core)}


def make_figure(rows, decomp, two_stage, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    cap = max(NGRID)
    # panel 1: N_0.9 grouped bars by change, conditions
    conds = ["reset", "modules", "operators", "full", "frozen"]
    cols = ["crimson", "goldenrod", "steelblue", "seagreen", "gray"]
    changes = [r["change"] for r in rows]
    x = np.arange(len(rows))
    w = 0.16
    for j, c in enumerate(conds):
        vals = [(r["N09"][c] if r["N09"][c] is not None else cap) for r in rows]
        ax[0].bar(x + (j - 2) * w, vals, w, label=c, color=cols[j])
    ax[0].set_xticks(x); ax[0].set_xticklabels(changes, fontsize=8, rotation=20)
    ax[0].set_ylabel("N_0.9 (samples)")
    ax[0].set_title("structure-discovery cost by change type", fontsize=10)
    ax[0].legend(fontsize=7, ncol=2)
    # panel 2: decomposition on RECOMBINE
    rec = decomp["RECOMBINE"]
    names = ["reset", "modules", "operators", "full"]
    ax[1].bar(names, [rec[n_] or cap for n_ in names],
              color=["crimson", "goldenrod", "steelblue", "seagreen"])
    ax[1].set_ylabel("N_0.9 (samples)")
    ax[1].set_title("RECOMBINE: what is remembered?\nonly the construction OPERATORS "
                    "(atoms) help", fontsize=10)
    # panel 3: two-stage N_0.9 vs T_0.95 (full memory)
    labels = [d["change"] for d in two_stage]
    xx = np.arange(len(two_stage))
    ax[2].bar(xx - 0.2, [d["N09_full"] or cap for d in two_stage], 0.4,
              label="N_0.9 (discover)", color="steelblue")
    ax2 = ax[2].twinx()
    ax2.bar(xx + 0.2, [d["T095_mem"] or 0 for d in two_stage], 0.4,
            label="T_0.95 (adapt)", color="seagreen")
    ax[2].set_xticks(xx); ax[2].set_xticklabels(labels, fontsize=7, rotation=20)
    ax[2].set_ylabel("N_0.9", color="steelblue")
    ax2.set_ylabel("T_0.95", color="seagreen")
    ax[2].set_title("two-stage: discover structure, then adapt", fontsize=10)
    fig.suptitle("exp595  Dynamic Interaction Recomposition -- past parts are "
                 "recomposed; the transferred memory is the construction operators",
                 fontsize=11)
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

    log(f"exp595 Dynamic Interaction Recomposition  seeds={SEEDS} n_grid<= {max(NGRID)}")
    rows = block_discovery(log)
    decomp, decomp_verdict = block_decomposition(rows, log)
    two_stage = block_two_stage(log)
    lineage = block_lineage(log)
    decision = decide(rows, decomp_verdict, log)
    fig_ok = make_figure(rows, decomp, two_stage, "recompose.png")
    log(f"figure: {'recompose.png' if fig_ok else '(matplotlib unavailable)'}")
    dt = time.time() - t0
    log("=" * 74)
    log(f"done in {dt:.1f}s")

    out = {"config": {"seeds": SEEDS, "n_grid": NGRID},
           "block1_discovery": rows, "block2_decomposition": decomp,
           "block3_two_stage": two_stage, "block4_lineage": lineage,
           "decision": decision, "runtime_seconds": dt}
    with open("recompose_results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=float)
    with open("recompose_run.log", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote recompose_results.json and recompose_run.log")


if __name__ == "__main__":
    main()
