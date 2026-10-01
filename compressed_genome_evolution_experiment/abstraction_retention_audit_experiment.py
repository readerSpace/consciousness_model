"""Runner for exp603 -- Abstraction Retention Audit."""

from __future__ import annotations

import json
import time

import abstraction_retention_audit as A


def block_accounting(log):
    log("=" * 78)
    log("BLOCK 1  dependency-aware maintenance costs")
    log("=" * 78)
    costs = A.operation_accounting()
    for name, bits in costs.items():
        log(f"  {name:>27}: maintenance bits={bits}")
    return costs


def block_workload(name, audit, log):
    log("=" * 78)
    log(f"BLOCK  {name}  retain/delete controls and MDL free choice")
    log("=" * 78)
    out = {condition: {key: row[key] for key in ("retains", "L_language", "L_tasks_given_D", "L_total")}
           for condition, row in audit["conditions"].items()}
    for condition, row in out.items():
        log(f"  {condition:>27}: keep={row['retains']} L(D)={row['L_language']:2d} "
            f"L(tasks|D)={row['L_tasks_given_D']:3d} total={row['L_total']:3d}")
    log(f"  -> MDL free choice: {audit['free_choice']['state']} "
        f"keep={audit['free_choice']['retains']} total={audit['free_choice']['L_total']}")
    return {"arities": audit["arities"], "conditions": out,
            "free_choice": audit["free_choice"]["state"],
            "free_choice_retains": audit["free_choice"]["retains"],
            "free_choice_total": audit["free_choice"]["L_total"]}


def block_verdict(single, family, log):
    pruning = single["free_choice"] == "delete_O2" and single["free_choice_retains"] == ["O1"]
    retained = family["free_choice"] == "full_history" and family["free_choice_retains"] == ["O1", "O2"]
    no_hidden_dependency = (single["conditions"]["delete_O1_compiled_O2"]["L_language"]
                            > single["conditions"]["full_history"]["L_language"])
    verdict = {
        "MDL_PRUNES_UNAMORTIZED_HIGHER_ABSTRACTION": pruning,
        "MDL_RETAINS_INTERMEDIATE_ABSTRACTIONS_WHEN_FUTURE_REUSE_PAYS": retained,
        "DELETE_O1_PAYS_COMPILED_O2_COST": no_hidden_dependency,
    }
    verdict["ABSTRACTION_RETENTION_IS_SELECTED_BY_MDL"] = all(verdict.values())
    log("=" * 78)
    log("BLOCK 4  conservative verdict")
    log("=" * 78)
    for key, value in verdict.items():
        log(f"  {key:<68}: {value}")
    return verdict


def _plot(single, family):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    names = list(single["conditions"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, dataset, title in ((axes[0], single, "one O3 task: prune O2"),
                                 (axes[1], family, "three O3 tasks: retain O1+O2")):
        axis.bar(names, [dataset["conditions"][name]["L_total"] for name in names])
        axis.set(ylabel="L(D)+L(tasks|D)", title=title)
        axis.tick_params(axis="x", labelrotation=20, labelsize=7)
    fig.tight_layout(); fig.savefig("abstraction_retention_audit.png", dpi=110); plt.close(fig)
    return True


def main():
    started, lines = time.time(), []

    def log(message):
        print(message, flush=True)
        lines.append(message)

    log("exp603 Abstraction Retention Audit")
    accounting = block_accounting(log)
    trace = A.retention_trajectory()
    single = block_workload("2  single O3", trace["single_O3"], log)
    family = block_workload("3  O3 family (three independent future tasks)", trace["O3_family_three"], log)
    verdict = block_verdict(single, family, log)
    plotted = _plot(single, family)
    log(f"plot written: {plotted}")
    log(f"total time: {time.time() - started:.1f}s")
    results = {"accounting": accounting, "single_O3": single, "O3_family_three": family,
               "verdict": verdict}
    with open("abstraction_retention_audit_results.json", "w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)
    with open("abstraction_retention_audit_run.log", "w", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()