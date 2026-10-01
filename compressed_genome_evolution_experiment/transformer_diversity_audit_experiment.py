"""Runner for exp611 -- Transformer Diversity / Self-Modification Audit."""

from __future__ import annotations

from dataclasses import asdict
import json
import time
from statistics import mean, pstdev

import transformer_diversity_audit as E


SEEDS = (1, 2, 3)
TIME_STEPS = (100, 300, 1000)
METRICS = ("n_raw_programs", "n_functional_semantic_classes", "n_functional_transformer_classes",
           "late_functional_transformer_classes", "transformer_reuse", "max_ancestry_depth",
           "capacity_fraction")


def _aggregate(runs):
    return {metric: {"mean": float(mean(run["summary"][metric] for run in runs)),
                     "std": float(pstdev(run["summary"][metric] for run in runs))}
            for metric in METRICS}


def run_sweep(log):
    sweep = {}
    for condition in E.CONDITIONS:
        sweep[condition] = {}
        for steps in TIME_STEPS:
            config = E.Config(steps=steps)
            runs = [E.run_audit(condition, config, seed) for seed in SEEDS]
            metrics = _aggregate(runs)
            sweep[condition][str(steps)] = {"config": asdict(config), "metrics": metrics,
                                             "series_seed1": runs[0]["records"],
                                             "transformers_seed1": runs[0]["transformers"]}
            log(f"  {condition:>27} T={steps:>4}: transformers="
                f"{metrics['n_functional_transformer_classes']['mean']:.0f} "
                f"late={metrics['late_functional_transformer_classes']['mean']:.0f} "
                f"raw={metrics['n_raw_programs']['mean']:.0f} depth={metrics['max_ancestry_depth']['mean']:.0f}")
    return sweep


def verdict(sweep, log):
    adaptive = sweep[E.UNIFIED_ADAPTIVE]["1000"]["metrics"]
    controls = [sweep[condition]["1000"]["metrics"] for condition in E.CONDITIONS[1:]]
    counts = [sweep[E.UNIFIED_ADAPTIVE][str(steps)]["metrics"]
              ["n_functional_transformer_classes"]["mean"] for steps in TIME_STEPS]
    checks = {
        "functional_transformer_classes_increase": counts[0] < counts[1] < counts[2],
        "adaptive_has_late_functional_transformers": adaptive["late_functional_transformer_classes"]["mean"] > 0,
        "all_mechanism_controls_have_zero_functional_transformers": all(
            metrics["n_functional_transformer_classes"]["mean"] == 0 for metrics in controls),
        "random_rewrite_has_raw_programs_but_no_functional_transformer": (
            sweep[E.RANDOM_REWRITE]["1000"]["metrics"]["n_raw_programs"]["mean"] > 0
            and sweep[E.RANDOM_REWRITE]["1000"]["metrics"]["n_functional_transformer_classes"]["mean"] == 0),
        "adaptive_language_not_capacity_saturated": adaptive["capacity_fraction"]["mean"] < 0.95,
    }
    checks["TRANSFORMER_DIVERSITY_CANDIDATE"] = all(checks.values())
    for name, value in checks.items():
        log(f"  {name:<72}: {value}")
    return checks


def main():
    started, lines = time.time(), []

    def log(message):
        print(message, flush=True)
        lines.append(message)

    log(f"exp611 Transformer Diversity Audit  seeds={list(SEEDS)}")
    sweep = run_sweep(log)
    checks = verdict(sweep, log)
    log(f"total time: {time.time() - started:.1f}s")
    with open("transformer_diversity_audit_results.json", "w", encoding="utf-8") as file:
        json.dump({"config": {"seeds": list(SEEDS), "time_steps": list(TIME_STEPS)},
                   "sweep": sweep, "verdict": checks}, file, ensure_ascii=False, indent=2)
    with open("transformer_diversity_audit_run.log", "w", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()