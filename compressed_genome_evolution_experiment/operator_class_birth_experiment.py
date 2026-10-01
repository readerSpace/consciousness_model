"""Runner for exp608 -- Operator-Class Birth."""

from __future__ import annotations

from dataclasses import asdict
import json
import time
from statistics import mean, pstdev

import operator_class_birth as O


SEEDS = (1, 2, 3)
TIME_STEPS = (100, 300, 1000)
CONDITIONS = (
    ("fixed_environment_operators", O.FIXED_ENV, O.ADAPTIVE_LANGUAGE),
    ("generative_environment_operators", O.GENERATIVE_ENV, O.ADAPTIVE_LANGUAGE),
    ("generative_fixed_individual_language", O.GENERATIVE_ENV, O.FIXED_LANGUAGE),
    ("generative_adaptive_individual_language", O.GENERATIVE_ENV, O.ADAPTIVE_LANGUAGE),
)
METRICS = ("n_operator_classes", "n_functional_classes", "late_functional_classes",
           "functional_class_rate_late", "total_parent_reuse", "frontier_operator",
           "max_D_language", "capacity_fraction")


def _aggregate(runs):
    return {metric: {"mean": float(mean(run["summary"][metric] for run in runs)),
                     "std": float(pstdev(run["summary"][metric] for run in runs))}
            for metric in METRICS}


def run_sweep(log):
    sweep = {}
    for label, environment, language in CONDITIONS:
        sweep[label] = {}
        for steps in TIME_STEPS:
            config = O.Config(steps=steps, capacity=300_000)
            runs = [O.run_birth(environment, language, config, seed) for seed in SEEDS]
            metrics = _aggregate(runs)
            sweep[label][str(steps)] = {"config": asdict(config), "metrics": metrics,
                                        "series_seed1": runs[0]["records"],
                                        "lineage_seed1": runs[0]["lineage"]}
            log(f"  {label:>38} T={steps:>4}: late={metrics['late_functional_classes']['mean']:.0f} "
                f"classes={metrics['n_functional_classes']['mean']:.0f} "
                f"reuse={metrics['total_parent_reuse']['mean']:.0f} "
                f"frontier={metrics['frontier_operator']['mean']:.0f}")
    return sweep


def verdict(sweep, log):
    adaptive = sweep["generative_adaptive_individual_language"]["1000"]["metrics"]
    fixed_environment = sweep["fixed_environment_operators"]["1000"]["metrics"]
    fixed_language = sweep["generative_fixed_individual_language"]["1000"]["metrics"]
    rates = [sweep["generative_adaptive_individual_language"][str(steps)]["metrics"]
             ["functional_class_rate_late"]["mean"] for steps in TIME_STEPS]
    checks = {
        "generative_adaptive_has_late_functional_birth": adaptive["late_functional_classes"]["mean"] > 0,
        "functional_class_rate_stays_positive": min(rates) > 0,
        "fixed_environment_plateaus": fixed_environment["late_functional_classes"]["mean"] == 0,
        "fixed_language_plateaus": fixed_language["n_functional_classes"]["mean"] == 0,
        "adaptive_expands_frontier": adaptive["frontier_operator"]["mean"] > fixed_language["frontier_operator"]["mean"],
        "language_not_capacity_saturated": adaptive["capacity_fraction"]["mean"] < 0.95,
    }
    checks["OPERATOR_CLASS_BIRTH_CANDIDATE"] = all(checks.values())
    for name, value in checks.items():
        log(f"  {name:<62}: {value}")
    return checks


def main():
    started, lines = time.time(), []

    def log(message):
        print(message, flush=True)
        lines.append(message)

    log(f"exp608 Operator-Class Birth  seeds={list(SEEDS)}")
    sweep = run_sweep(log)
    checks = verdict(sweep, log)
    log(f"total time: {time.time() - started:.1f}s")
    with open("operator_class_birth_results.json", "w", encoding="utf-8") as file:
        json.dump({"config": {"seeds": list(SEEDS), "time_steps": list(TIME_STEPS)},
                   "sweep": sweep, "verdict": checks}, file, ensure_ascii=False, indent=2)
    with open("operator_class_birth_run.log", "w", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()