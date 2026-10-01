"""Runner for exp605 -- Scaling / Plateau Audit."""

from __future__ import annotations

from dataclasses import asdict
import json
import time
from statistics import mean, pstdev

import scaling_plateau_audit as S


SEEDS = (1, 2, 3)
TIME_STEPS = (30, 100, 300, 1000)
CAPACITIES = (64, 128, 256, 512)
MAX_ARITIES = (8, 16, 32, 64)
METRICS = ("n_birth", "n_reinvent", "n_semantic_novelty", "late_births",
           "late_semantic_novelty", "remaining_catalog", "mean_A_late",
           "late_A_slope", "max_D_language", "capacity_fraction", "max_arity")


def _aggregate(runs):
    return {metric: {"mean": float(mean(run["summary"][metric] for run in runs)),
                     "std": float(pstdev(run["summary"][metric] for run in runs)),
                     "values": [run["summary"][metric] for run in runs]}
            for metric in METRICS}


def _run_grid(configs, log):
    output = {}
    for label, config in configs:
        runs = [S.run_audit("adaptive_language", config, seed) for seed in SEEDS]
        aggregate = _aggregate(runs)
        output[label] = {"config": asdict(config), "metrics": aggregate,
                         "series_seed1": runs[0]["records"]}
        log(f"  {label:>8}: novelty={aggregate['n_semantic_novelty']['mean']:.1f} "
            f"late_novel={aggregate['late_semantic_novelty']['mean']:.1f} "
            f"late_birth={aggregate['late_births']['mean']:.1f} "
            f"remaining={aggregate['remaining_catalog']['mean']:.1f} "
            f"A_late={aggregate['mean_A_late']['mean']:.1f} "
            f"D/K={aggregate['capacity_fraction']['mean']:.2f} "
            f"max_arity={aggregate['max_arity']['mean']:.0f}")
    return output


def block_time(log):
    log("=" * 78)
    log("BLOCK 1  time sweep: T={30,100,300,1000}, K=512, Amax=64")
    log("=" * 78)
    return _run_grid([(str(steps), S.AuditConfig(steps, 512, 64)) for steps in TIME_STEPS], log)


def block_capacity(log):
    log("=" * 78)
    log("BLOCK 2  capacity sweep: K={64,128,256,512}, T=300, Amax=64")
    log("=" * 78)
    return _run_grid([(str(capacity), S.AuditConfig(300, capacity, 64)) for capacity in CAPACITIES], log)


def block_catalog(log):
    log("=" * 78)
    log("BLOCK 3  catalog sweep: Amax={8,16,32,64}, T=300, K=512")
    log("=" * 78)
    return _run_grid([(str(max_arity), S.AuditConfig(300, 512, max_arity))
                      for max_arity in MAX_ARITIES], log)


def block_verdict(time_sweep, capacity_sweep, catalog_sweep, log):
    long = time_sweep["1000"]["metrics"]
    wide = catalog_sweep["64"]["metrics"]
    capacity_frontier = [capacity_sweep[str(capacity)]["metrics"]["max_arity"]["mean"]
                         for capacity in CAPACITIES]
    verdict = {
        "capacity_boundary_bloat_L_to_K": any(
            capacity_sweep[str(capacity)]["metrics"]["capacity_fraction"]["mean"] >= 0.95
            for capacity in CAPACITIES),
        "capacity_expansion_unlocks_reachability": capacity_frontier == sorted(capacity_frontier)
                                                     and capacity_frontier[-1] > capacity_frontier[0],
        "catalog_consumed_at_widest_setting": wide["remaining_catalog"]["mean"] == 0,
        "late_semantic_novelty_continues": long["late_semantic_novelty"]["mean"] > 0,
        "reinvention_cycle_without_late_novelty": (long["late_births"]["mean"] > 0
                                                     and long["n_reinvent"]["mean"] > 0
                                                     and long["late_semantic_novelty"]["mean"] == 0),
        "late_advantage_plateaus": abs(long["late_A_slope"]["mean"]) < 1.0,
    }
    verdict["BOUNDED_CATALOG_PLATEAU"] = (verdict["catalog_consumed_at_widest_setting"]
                                            and not verdict["late_semantic_novelty_continues"]
                                            and verdict["reinvention_cycle_without_late_novelty"])
    verdict["OPEN_ENDEDNESS_CANDIDATE"] = (verdict["late_semantic_novelty_continues"]
                                             and not verdict["capacity_boundary_bloat_L_to_K"]
                                             and not verdict["catalog_consumed_at_widest_setting"])
    log("=" * 78)
    log("BLOCK 4  plateau / false-open-endedness verdict")
    log("=" * 78)
    for name, value in verdict.items():
        log(f"  {name:<64}: {value}")
    return verdict


def _plot(time_sweep, capacity_sweep, catalog_sweep):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(TIME_STEPS, [time_sweep[str(step)]["metrics"]["n_semantic_novelty"]["mean"]
                              for step in TIME_STEPS], marker="o", label="total novelty")
    axes[0].plot(TIME_STEPS, [time_sweep[str(step)]["metrics"]["late_semantic_novelty"]["mean"]
                              for step in TIME_STEPS], marker="o", label="late novelty")
    axes[0].set(xlabel="steps", title="Time sweep")
    axes[0].legend(fontsize=8)
    axes[1].plot(CAPACITIES, [capacity_sweep[str(capacity)]["metrics"]["max_arity"]["mean"]
                              for capacity in CAPACITIES], marker="o", label="reachable arity")
    axes[1].plot(CAPACITIES, [capacity_sweep[str(capacity)]["metrics"]["capacity_fraction"]["mean"]
                              for capacity in CAPACITIES], marker="o", label="max L(D)/K")
    axes[1].set(xlabel="capacity", title="Capacity sweep")
    axes[1].legend(fontsize=8)
    axes[2].bar([str(value) for value in MAX_ARITIES],
                [catalog_sweep[str(value)]["metrics"]["remaining_catalog"]["mean"]
                 for value in MAX_ARITIES])
    axes[2].set(xlabel="catalog Amax", ylabel="remaining semantic catalog", title="Catalog sweep")
    fig.tight_layout(); fig.savefig("scaling_plateau_audit.png", dpi=110); plt.close(fig)
    return True


def main():
    started, lines = time.time(), []

    def log(message):
        print(message, flush=True)
        lines.append(message)

    log(f"exp605 Scaling / Plateau Audit  seeds={list(SEEDS)}")
    time_sweep = block_time(log)
    capacity_sweep = block_capacity(log)
    catalog_sweep = block_catalog(log)
    verdict = block_verdict(time_sweep, capacity_sweep, catalog_sweep, log)
    plotted = _plot(time_sweep, capacity_sweep, catalog_sweep)
    log(f"plot written: {plotted}")
    log(f"total time: {time.time() - started:.1f}s")
    results = {"config": {"seeds": list(SEEDS), "time_steps": list(TIME_STEPS),
                            "capacities": list(CAPACITIES), "max_arities": list(MAX_ARITIES)},
               "time_sweep": time_sweep, "capacity_sweep": capacity_sweep,
               "catalog_sweep": catalog_sweep, "verdict": verdict}
    with open("scaling_plateau_audit_results.json", "w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)
    with open("scaling_plateau_audit_run.log", "w", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()