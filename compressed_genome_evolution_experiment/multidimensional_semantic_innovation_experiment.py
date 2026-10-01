"""Runner for exp607 -- Multidimensional Semantic Innovation."""

from __future__ import annotations

from dataclasses import asdict
import json
import time
from statistics import mean, pstdev

import multidimensional_semantic_innovation as M


SEEDS = (1, 2, 3)
TIME_STEPS = (100, 300, 1000, 3000)
CONDITIONS = M.CONDITIONS
METRICS = ("n_parametric_novelty", "n_structural_novelty", "late_parametric_novelty",
           "late_structural_novelty", "structural_novelty_rate_late", "operator_classes",
           "n_birth", "n_retained_final", "total_reuses", "frontier_depth",
           "max_D_language", "capacity_fraction")


def _aggregate(runs):
    return {metric: {"mean": float(mean(run["summary"][metric] for run in runs)),
                     "std": float(pstdev(run["summary"][metric] for run in runs)),
                     "values": [run["summary"][metric] for run in runs]}
            for metric in METRICS}


def _run_grid(mode, configs, log):
    output = {}
    for condition in CONDITIONS:
        output[condition] = {}
        for label, config in configs:
            runs = [M.run_innovation(mode, condition, config, seed) for seed in SEEDS]
            metrics = _aggregate(runs)
            output[condition][label] = {"config": asdict(config), "metrics": metrics,
                                        "series_seed1": runs[0]["records"]}
            log(f"  {mode:>24} {condition:>17} T={label:>4}: "
                f"late(struct/param)={metrics['late_structural_novelty']['mean']:.1f}/"
                f"{metrics['late_parametric_novelty']['mean']:.1f} "
                f"classes={metrics['operator_classes']['mean']:.0f} "
                f"reuse={metrics['total_reuses']['mean']:.0f} frontier={metrics['frontier_depth']['mean']:.0f}")
    return output


def block_time(log):
    log("=" * 78)
    log("BLOCK 1  parametric-only versus compositional-generative time sweep")
    log("=" * 78)
    configs = [(str(steps), M.Config(steps=steps)) for steps in TIME_STEPS]
    return {mode: _run_grid(mode, configs, log) for mode in (M.PARAMETRIC, M.COMPOSITIONAL)}


def block_verdict(sweep, log):
    parametric = sweep[M.PARAMETRIC]["adaptive_language"]["3000"]["metrics"]
    adaptive = sweep[M.COMPOSITIONAL]["adaptive_language"]["3000"]["metrics"]
    fixed = sweep[M.COMPOSITIONAL]["fixed_language"]["3000"]["metrics"]
    no_retention = sweep[M.COMPOSITIONAL]["no_retention"]["3000"]["metrics"]
    rates = [sweep[M.COMPOSITIONAL]["adaptive_language"][str(steps)]["metrics"]
             ["structural_novelty_rate_late"]["mean"] for steps in TIME_STEPS]
    verdict = {
        "parametric_only_has_no_structural_novelty": parametric["late_structural_novelty"]["mean"] == 0,
        "compositional_late_structural_novelty_positive": adaptive["late_structural_novelty"]["mean"] > 0,
        "operator_classes_increase_beyond_one": adaptive["operator_classes"]["mean"] > 1,
        "new_operator_classes_retained_and_reused": (adaptive["n_retained_final"]["mean"] > 1
                                                        and adaptive["total_reuses"]["mean"] > 0),
        "fixed_and_no_retention_plateau": (fixed["late_structural_novelty"]["mean"] == 0
                                             and no_retention["late_structural_novelty"]["mean"] == 0),
        "adaptive_expands_structural_frontier": adaptive["frontier_depth"]["mean"] > fixed["frontier_depth"]["mean"],
        "structural_rate_stays_positive": min(rates) > 0.25,
        "language_not_capacity_saturated": adaptive["capacity_fraction"]["mean"] < 0.95,
    }
    verdict["OPEN_ENDED_SEMANTIC_INNOVATION_CANDIDATE"] = all(verdict.values())
    log("=" * 78)
    log("BLOCK 2  pre-registered verdict")
    log("=" * 78)
    for name, value in verdict.items():
        log(f"  {name:<64}: {value}")
    return verdict


def _plot(sweep):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for mode in (M.PARAMETRIC, M.COMPOSITIONAL):
        for condition in ("adaptive_language", "fixed_language"):
            metrics = sweep[mode][condition]
            axes[0].plot(TIME_STEPS, [metrics[str(steps)]["metrics"]["structural_novelty_rate_late"]["mean"]
                                      for steps in TIME_STEPS], marker="o", label=f"{mode}/{condition}")
    adaptive = sweep[M.COMPOSITIONAL]["adaptive_language"]
    axes[1].plot(TIME_STEPS, [adaptive[str(steps)]["metrics"]["operator_classes"]["mean"]
                              for steps in TIME_STEPS], marker="o", label="operator classes")
    axes[1].plot(TIME_STEPS, [adaptive[str(steps)]["metrics"]["frontier_depth"]["mean"]
                              for steps in TIME_STEPS], marker="o", label="structural frontier")
    axes[0].set(xlabel="steps", ylabel="late structural novelty rate", title="Structural novelty")
    axes[1].set(xlabel="steps", title="Adaptive compositional growth")
    for axis in axes:
        axis.legend(fontsize=7)
    fig.tight_layout(); fig.savefig("multidimensional_semantic_innovation.png", dpi=110); plt.close(fig)
    return True


def main():
    started, lines = time.time(), []

    def log(message):
        print(message, flush=True)
        lines.append(message)

    log(f"exp607 Multidimensional Semantic Innovation  seeds={list(SEEDS)}")
    sweep = block_time(log)
    verdict = block_verdict(sweep, log)
    plotted = _plot(sweep)
    log(f"plot written: {plotted}")
    log(f"total time: {time.time() - started:.1f}s")
    results = {"config": {"seeds": list(SEEDS), "time_steps": list(TIME_STEPS)},
               "sweep": sweep, "verdict": verdict}
    with open("multidimensional_semantic_innovation_results.json", "w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)
    with open("multidimensional_semantic_innovation_run.log", "w", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()