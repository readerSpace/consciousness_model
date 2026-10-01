"""Runner for exp609 -- Meta-Operator Evolution."""

from __future__ import annotations

from dataclasses import asdict
import json
import time
from statistics import mean, pstdev

import meta_operator_evolution as M


SEEDS = (1, 2, 3)
TIME_STEPS = (100, 300, 1000)
METRICS = ("n_operator_classes", "late_operator_classes", "n_meta_rule_classes",
           "n_functional_meta_rules", "late_functional_meta_rules", "meta_rule_reuse",
           "frontier_operator", "max_D_meta", "capacity_fraction")


def _aggregate(runs):
    return {metric: {"mean": float(mean(run["summary"][metric] for run in runs)),
                     "std": float(pstdev(run["summary"][metric] for run in runs))}
            for metric in METRICS}


def run_sweep(log):
    sweep = {}
    for mode in (M.FIXED_META, M.ADAPTIVE_META):
        sweep[mode] = {}
        for steps in TIME_STEPS:
            config = M.Config(steps=steps, capacity=300_000)
            runs = [M.run_meta_evolution(mode, config, seed) for seed in SEEDS]
            metrics = _aggregate(runs)
            sweep[mode][str(steps)] = {"config": asdict(config), "metrics": metrics,
                                        "series_seed1": runs[0]["records"],
                                        "meta_lineage_seed1": runs[0]["meta_lineage"]}
            log(f"  {mode:>22} T={steps:>4}: op={metrics['n_operator_classes']['mean']:.0f} "
                f"late-op={metrics['late_operator_classes']['mean']:.0f} "
                f"late-meta={metrics['late_functional_meta_rules']['mean']:.0f} "
                f"frontier={metrics['frontier_operator']['mean']:.0f}")
    return sweep


def verdict(sweep, log):
    fixed = sweep[M.FIXED_META]["1000"]["metrics"]
    adaptive = sweep[M.ADAPTIVE_META]["1000"]["metrics"]
    checks = {
        "adaptive_has_late_functional_meta_rule": adaptive["late_functional_meta_rules"]["mean"] > 0,
        "adaptive_has_more_operator_classes": adaptive["n_operator_classes"]["mean"] > fixed["n_operator_classes"]["mean"],
        "adaptive_expands_operator_frontier": adaptive["frontier_operator"]["mean"] > fixed["frontier_operator"]["mean"],
        "fixed_meta_has_no_functional_birth": fixed["n_functional_meta_rules"]["mean"] == 0,
        "meta_language_not_capacity_saturated": adaptive["capacity_fraction"]["mean"] < 0.95,
    }
    checks["META_OPERATOR_EVOLUTION_CANDIDATE"] = all(checks.values())
    for name, value in checks.items():
        log(f"  {name:<62}: {value}")
    return checks


def main():
    started, lines = time.time(), []

    def log(message):
        print(message, flush=True)
        lines.append(message)

    log(f"exp609 Meta-Operator Evolution  seeds={list(SEEDS)}")
    sweep = run_sweep(log)
    checks = verdict(sweep, log)
    log(f"total time: {time.time() - started:.1f}s")
    with open("meta_operator_evolution_results.json", "w", encoding="utf-8") as file:
        json.dump({"config": {"seeds": list(SEEDS), "time_steps": list(TIME_STEPS)},
                   "sweep": sweep, "verdict": checks}, file, ensure_ascii=False, indent=2)
    with open("meta_operator_evolution_run.log", "w", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()