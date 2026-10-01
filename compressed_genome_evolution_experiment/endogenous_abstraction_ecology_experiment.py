"""Runner for exp604 -- Endogenous Abstraction Ecology."""

from __future__ import annotations

import json
import time
from statistics import mean, pstdev

import endogenous_abstraction_ecology as E


SEEDS = [1, 2, 3]
STEPS = 30
CONDITIONS = ("adaptive_language", "fixed_language", "no_retention", "keep_all")
METRICS = ("late_births", "late_reuse", "n_birth", "n_pruned", "n_retained_final",
           "mean_A_late", "mean_description", "max_D_language", "final_D_language", "max_arity")


def _aggregate(runs):
    return {metric: {"mean": float(mean(run["summary"][metric] for run in runs)),
                     "std": float(pstdev(run["summary"][metric] for run in runs)),
                     "values": [run["summary"][metric] for run in runs]}
            for metric in METRICS}


def block_conditions(log):
    log("=" * 78)
    log("BLOCK 1  endogenous task ecology across adaptive and control languages")
    log("=" * 78)
    raw, out = {}, {}
    for condition in CONDITIONS:
        raw[condition] = [E.run_ecology(condition, STEPS, seed) for seed in SEEDS]
        out[condition] = _aggregate(raw[condition])
        row = out[condition]
        log(f"  {condition:>18}: late_birth={row['late_births']['mean']:.1f} "
            f"late_reuse={row['late_reuse']['mean']:.1f} prune={row['n_pruned']['mean']:.1f} "
            f"A_late={row['mean_A_late']['mean']:.1f} Dmax={row['max_D_language']['mean']:.1f} "
            f"mean_desc={row['mean_description']['mean']:.1f}")
    return raw, out


def block_life_history(raw, log):
    log("=" * 78)
    log("BLOCK 2  adaptive instruction life histories (seed 1)")
    log("=" * 78)
    life = raw["adaptive_language"][0]["life_history"]
    for name, history in life.items():
        log(f"  {name}: birth={history['birth']} use={history['use']} reuse={history['reuse']} "
            f"retain_steps={history['retained_steps']} prune={history['prune']}")
    return life


def block_verdict(aggregate, log):
    adaptive = aggregate["adaptive_language"]
    fixed = aggregate["fixed_language"]
    no_retention = aggregate["no_retention"]
    keep_all = aggregate["keep_all"]
    verdict = {
        "late_instruction_birth": adaptive["late_births"]["mean"] > 0,
        "late_reuse_of_old_instructions": adaptive["late_reuse"]["mean"] > 0,
        "retention_and_pruning_both_occur": (adaptive["n_pruned"]["mean"] > 0
                                               and adaptive["n_retained_final"]["mean"] > 0),
        "accumulated_advantage_persists_late": adaptive["mean_A_late"]["mean"] > 0,
        "adaptive_beats_fixed_and_no_retention": (adaptive["mean_A_late"]["mean"]
                                                    > no_retention["mean_A_late"]["mean"]
                                                    > fixed["mean_A_late"]["mean"]),
        "adaptive_forgetting_beats_keep_all_bloat": (adaptive["mean_description"]["mean"]
                                                       < keep_all["mean_description"]["mean"]),
        "language_not_saturated_at_capacity": adaptive["max_D_language"]["mean"] < E.CAPACITY,
    }
    verdict["SELF_SUSTAINING_CUMULATIVE_ABSTRACTION"] = all(verdict.values())
    log("=" * 78)
    log("BLOCK 3  pre-registered verdict")
    log("=" * 78)
    for key, value in verdict.items():
        log(f"  {key:<64}: {value}")
    return verdict


def _plot(raw, aggregate):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    adaptive = raw["adaptive_language"][0]["records"]
    axes[0].plot([row["t"] for row in adaptive], [row["A_advantage"] for row in adaptive], label="A_t")
    axes[0].step([row["t"] for row in adaptive], [row["D_language"] for row in adaptive], where="mid", label="L(D)")
    axes[0].set(xlabel="ecology step", title="Adaptive language: advantage and retained cost")
    axes[0].legend(fontsize=8)
    names = list(CONDITIONS)
    axes[1].bar(names, [aggregate[name]["mean_description"]["mean"] for name in names])
    axes[1].set(ylabel="mean L(D)+L(G|D)", title="Forgetting avoids keep-all bloat")
    axes[1].tick_params(axis="x", labelrotation=20, labelsize=8)
    fig.tight_layout(); fig.savefig("endogenous_abstraction_ecology.png", dpi=110); plt.close(fig)
    return True


def main():
    started, lines = time.time(), []

    def log(message):
        print(message, flush=True)
        lines.append(message)

    log(f"exp604 Endogenous Abstraction Ecology  steps={STEPS} seeds={SEEDS} capacity={E.CAPACITY}")
    raw, aggregate = block_conditions(log)
    life = block_life_history(raw, log)
    verdict = block_verdict(aggregate, log)
    plotted = _plot(raw, aggregate)
    log(f"plot written: {plotted}")
    log(f"total time: {time.time() - started:.1f}s")
    results = {"config": {"steps": STEPS, "seeds": SEEDS, "capacity": E.CAPACITY},
               "conditions": aggregate, "adaptive_life_history_seed1": life,
               "verdict": verdict}
    with open("endogenous_abstraction_ecology_results.json", "w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)
    with open("endogenous_abstraction_ecology_run.log", "w", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()