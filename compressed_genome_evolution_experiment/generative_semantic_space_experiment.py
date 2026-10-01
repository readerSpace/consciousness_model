"""Runner for exp606 -- Generative Semantic Space."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from statistics import mean, pstdev

import generative_semantic_space as G


SEEDS = (1, 2, 3)
TIME_STEPS = (100, 300, 1000, 3000)
CAPACITIES = (16, 32, 64, 128)
METRICS = ("n_semantic_novelty", "late_semantic_novelty", "novelty_rate_late",
           "n_birth", "n_retained", "uses", "reuses", "frontier_depth",
           "generated_frontier_depth", "mean_A_late", "max_D_language", "capacity_fraction")


def _aggregate(runs):
    return {metric: {"mean": float(mean(run["summary"][metric] for run in runs)),
                     "std": float(pstdev(run["summary"][metric] for run in runs)),
                     "values": [run["summary"][metric] for run in runs]}
            for metric in METRICS}


def _run(condition, configs, log):
    output = {}
    for label, config in configs:
        runs = [G.run_space(condition, config, seed) for seed in SEEDS]
        metrics = _aggregate(runs)
        output[label] = {"config": asdict(config), "metrics": metrics,
                         "series_seed1": runs[0]["records"]}
        log(f"  {condition:>17} {label:>5}: late_novel={metrics['late_semantic_novelty']['mean']:.1f} "
            f"rate={metrics['novelty_rate_late']['mean']:.3f} "
            f"birth/retain/reuse={metrics['n_birth']['mean']:.0f}/"
            f"{metrics['n_retained']['mean']:.0f}/{metrics['reuses']['mean']:.0f} "
            f"frontier={metrics['frontier_depth']['mean']:.0f} "
            f"D/K={metrics['capacity_fraction']['mean']:.2f}")
    return output


def block_time(log):
    log("=" * 78)
    log("BLOCK 1  time sweep: T={100,300,1000,3000}, K=64")
    log("=" * 78)
    configs = [(str(steps), G.Config(steps=steps, capacity=64)) for steps in TIME_STEPS]
    return {condition: _run(condition, configs, log) for condition in
            ("adaptive_language", "fixed_language")}


def block_capacity(log):
    log("=" * 78)
    log("BLOCK 2  capacity sweep: K={16,32,64,128}, T=1000")
    log("=" * 78)
    configs = [(str(capacity), G.Config(steps=1000, capacity=capacity)) for capacity in CAPACITIES]
    return {condition: _run(condition, configs, log) for condition in
            ("adaptive_language", "fixed_language")}


def block_verdict(time_sweep, capacity_sweep, log):
    adaptive_long = time_sweep["adaptive_language"]["3000"]["metrics"]
    fixed_long = time_sweep["fixed_language"]["3000"]["metrics"]
    adaptive_rates = [time_sweep["adaptive_language"][str(steps)]["metrics"]["novelty_rate_late"]["mean"]
                      for steps in TIME_STEPS]
    capacity_frontiers = [capacity_sweep["adaptive_language"][str(capacity)]["metrics"]["frontier_depth"]["mean"]
                          for capacity in CAPACITIES]
    verdict = {
        "late_semantic_novelty_positive": adaptive_long["late_semantic_novelty"]["mean"] > 0,
        "new_abstraction_retained_and_reused": (adaptive_long["n_retained"]["mean"] > 0
                                                  and adaptive_long["reuses"]["mean"] > 0),
        "adaptive_expands_reachable_semantic_frontier": (adaptive_long["frontier_depth"]["mean"]
                                                           > fixed_long["frontier_depth"]["mean"]),
        "fixed_language_plateaus_substantially_earlier": (fixed_long["late_semantic_novelty"]["mean"] == 0
                                                             and fixed_long["frontier_depth"]["mean"]
                                                             < adaptive_long["frontier_depth"]["mean"] / 10),
        "late_novelty_rate_stays_above_c": min(adaptive_rates) > 0.25,
        "capacity_above_definition_cost_preserves_mechanism": min(capacity_frontiers[1:]) > capacity_frontiers[0],
        "language_not_capacity_saturated": max(
            capacity_sweep["adaptive_language"][str(capacity)]["metrics"]["capacity_fraction"]["mean"]
            for capacity in CAPACITIES[1:]) < 0.95,
    }
    verdict["GENERATIVE_SEMANTIC_SPACE_CANDIDATE"] = all(verdict.values())
    log("=" * 78)
    log("BLOCK 3  pre-registered verdict")
    log("=" * 78)
    for name, value in verdict.items():
        log(f"  {name:<64}: {value}")
    return verdict


def _plot(time_sweep, capacity_sweep):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for condition in ("adaptive_language", "fixed_language"):
        axes[0].plot(TIME_STEPS, [time_sweep[condition][str(steps)]["metrics"]["novelty_rate_late"]["mean"]
                                  for steps in TIME_STEPS], marker="o", label=condition)
        axes[1].plot(CAPACITIES, [capacity_sweep[condition][str(capacity)]["metrics"]["frontier_depth"]["mean"]
                                  for capacity in CAPACITIES], marker="o", label=condition)
    axes[0].set(xlabel="steps", ylabel="late novelty rate", title="Continuing semantic novelty")
    axes[1].set(xlabel="capacity", ylabel="reachable semantic depth", title="Language expands frontier")
    for axis in axes:
        axis.legend(fontsize=8)
    fig.tight_layout(); fig.savefig("generative_semantic_space.png", dpi=110); plt.close(fig)
    return True


def main():
    started, lines = time.time(), []

    def log(message):
        print(message, flush=True)
        lines.append(message)

    log(f"exp606 Generative Semantic Space  seeds={list(SEEDS)}")
    time_sweep = block_time(log)
    capacity_sweep = block_capacity(log)
    verdict = block_verdict(time_sweep, capacity_sweep, log)
    plotted = _plot(time_sweep, capacity_sweep)
    log(f"plot written: {plotted}")
    log(f"total time: {time.time() - started:.1f}s")
    results = {"config": {"seeds": list(SEEDS), "time_steps": list(TIME_STEPS),
                            "capacities": list(CAPACITIES)}, "time_sweep": time_sweep,
               "capacity_sweep": capacity_sweep, "verdict": verdict}
    with open("generative_semantic_space_results.json", "w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)
    with open("generative_semantic_space_run.log", "w", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()