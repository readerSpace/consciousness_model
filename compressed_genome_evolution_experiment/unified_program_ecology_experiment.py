"""Runner for exp610 -- Unified Program Ecology."""

from __future__ import annotations

from dataclasses import asdict
import json
import time
from statistics import mean, pstdev

import unified_program_ecology as U


SEEDS = (1, 2, 3)
TIME_STEPS = (100, 300, 1000)
METRICS = ("n_functional_semantic_classes", "late_functional_semantic_classes",
           "n_functional_transformer_classes", "late_functional_transformer_classes",
           "max_ancestry_depth", "transformer_reuse", "frontier_program", "capacity_fraction")


def _aggregate(runs):
    return {metric: {"mean": float(mean(run["summary"][metric] for run in runs)),
                     "std": float(pstdev(run["summary"][metric] for run in runs))}
            for metric in METRICS}


def run_sweep(log):
    sweep = {}
    for mode in (U.SPECIALIZED, U.UNIFIED):
        sweep[mode] = {}
        for steps in TIME_STEPS:
            config = U.Config(steps=steps, capacity=300_000)
            runs = [U.run_ecology(mode, config, seed) for seed in SEEDS]
            metrics = _aggregate(runs)
            sweep[mode][str(steps)] = {"config": asdict(config), "metrics": metrics,
                                        "series_seed1": runs[0]["records"],
                                        "transformers_seed1": runs[0]["transformers"]}
            log(f"  {mode:>25} T={steps:>4}: semantic={metrics['n_functional_semantic_classes']['mean']:.0f} "
                f"late-sem={metrics['late_functional_semantic_classes']['mean']:.0f} "
                f"late-transform={metrics['late_functional_transformer_classes']['mean']:.0f} "
                f"depth={metrics['max_ancestry_depth']['mean']:.0f}")
    return sweep


def verdict(sweep, log):
    specialized = sweep[U.SPECIALIZED]["1000"]["metrics"]
    unified = sweep[U.UNIFIED]["1000"]["metrics"]
    checks = {
        "unified_reproduces_specialized_semantic_classes": unified["n_functional_semantic_classes"]["mean"] == specialized["n_functional_semantic_classes"]["mean"],
        "unified_has_late_functional_semantic_classes": unified["late_functional_semantic_classes"]["mean"] > 0,
        "unified_has_late_functional_transformer": unified["late_functional_transformer_classes"]["mean"] > 0,
        "ancestry_depth_grows": unified["max_ancestry_depth"]["mean"] > 100,
        "specialized_has_no_first_class_transformer_birth": specialized["n_functional_transformer_classes"]["mean"] == 0,
        "unified_language_not_capacity_saturated": unified["capacity_fraction"]["mean"] < 0.95,
    }
    checks["UNIFIED_PROGRAM_ECOLOGY_CANDIDATE"] = all(checks.values())
    for name, value in checks.items():
        log(f"  {name:<68}: {value}")
    return checks


def main():
    started, lines = time.time(), []

    def log(message):
        print(message, flush=True)
        lines.append(message)

    log(f"exp610 Unified Program Ecology  seeds={list(SEEDS)}")
    sweep = run_sweep(log)
    checks = verdict(sweep, log)
    log(f"total time: {time.time() - started:.1f}s")
    with open("unified_program_ecology_results.json", "w", encoding="utf-8") as file:
        json.dump({"config": {"seeds": list(SEEDS), "time_steps": list(TIME_STEPS)},
                   "sweep": sweep, "verdict": checks}, file, ensure_ascii=False, indent=2)
    with open("unified_program_ecology_run.log", "w", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()