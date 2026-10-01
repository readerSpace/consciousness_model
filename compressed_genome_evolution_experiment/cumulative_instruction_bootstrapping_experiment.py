"""Runner for exp602 -- Cumulative Instruction Bootstrapping."""

from __future__ import annotations

import json
import time

import cumulative_instruction_bootstrapping as B


N_TUPLES = [1, 2, 3, 4, 6, 8, 12, 16, 20, 30]


def _expr_shape(expression):
    if expression.op == "REF":
        return f"r{expression.index}"
    return f"{expression.op}({','.join(_expr_shape(child) for child in expression.children)})"


def block_o1(log):
    log("=" * 78)
    log("BLOCK 1  E1 discovers O1 in the exp601 distant primitive basis")
    log("=" * 78)
    cost, tokens = B.discover_o1()
    result = {"K_O1_P0": cost, "definition": list(tokens), "retained_bits": B.USEFUL_O1.retained_bits}
    log(f"  O1 definition={list(tokens)}  K_P0(O1)={cost}  retained decoder bits={B.USEFUL_O1.retained_bits}")
    return result


def block_o2_controls(log):
    log("=" * 78)
    log("BLOCK 2  E2 four-way abstraction: base / useful history / irrelevant history")
    log("=" * 78)
    out = {}
    for history in (B.BASE, B.USEFUL_O1, B.IRRELEVANT):
        cost, expression = B.semantic_complexity(4, history)
        threshold = B.birth_threshold(4, history, N_TUPLES)
        at_threshold = B.audit_birth(4, history, threshold) if threshold is not None else None
        out[history.name] = {
            "K_O2": cost, "expression": _expr_shape(expression), "n_birth_MDL": threshold,
            "history_bits_sunk": history.retained_bits,
            "total_at_birth": at_threshold["total"] if at_threshold else None,
            "fixed_at_birth": at_threshold["fixed"] if at_threshold else None,
        }
        log(f"  {history.name:>27}: K_O2={cost:2d} n*_birth={threshold} "
            f"history_bits(sunk)={history.retained_bits:2d} expr={out[history.name]['expression']}")
    return out


def block_cumulative_trace(log):
    log("=" * 78)
    log("BLOCK 3  cumulative abstraction trace O1 -> O2 -> O3")
    log("=" * 78)
    out = []
    histories = (B.BASE, B.USEFUL_O1, B.USEFUL_O1_O2)
    for row, history in zip(B.cumulative_trace(), histories):
        threshold = B.birth_threshold(row["arity"], history, N_TUPLES)
        base_threshold = B.birth_threshold(row["arity"], B.BASE, N_TUPLES)
        record = {**row, "n_birth_MDL": threshold, "n_birth_P0_MDL": base_threshold}
        out.append(record)
        log(f"  stage={row['stage']} arity={row['arity']}: K_P0={row['K_P0']:2d} "
            f"K_history={row['K_history']:2d} ratio={row['ratio']:.3f} "
            f"n*_P0={base_threshold} n*_history={threshold}")
    return out


def block_verdict(o2, trace, log):
    useful = o2[B.USEFUL_O1.name]
    base = o2[B.BASE.name]
    irrelevant = o2[B.IRRELEVANT.name]
    history_specific = (useful["K_O2"] < base["K_O2"] == irrelevant["K_O2"]
                        and useful["n_birth_MDL"] < base["n_birth_MDL"] == irrelevant["n_birth_MDL"])
    cumulative = all(row["K_history"] < row["K_P0"] for row in trace[1:])
    accelerated = all(row["n_birth_MDL"] < row["n_birth_P0_MDL"] for row in trace[1:])
    verdict = {
        "USEFUL_HISTORY_NOT_LANGUAGE_SIZE_CAUSES_O2_ADVANTAGE": history_specific,
        "CUMULATIVE_HISTORY_LOWERED_LATER_SEMANTIC_COMPLEXITY": cumulative,
        "CUMULATIVE_HISTORY_ACCELERATED_BIRTH_THRESHOLDS": accelerated,
    }
    verdict["CUMULATIVE_REPRESENTATIONAL_EVOLUTION"] = all(verdict.values())
    log("=" * 78)
    log("BLOCK 4  conservative verdict")
    log("=" * 78)
    for key, value in verdict.items():
        log(f"  {key:<64}: {value}")
    return verdict


def _plot(o2, trace):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    names = ["base", "useful", "irrelevant"]
    values = [o2[B.BASE.name]["K_O2"], o2[B.USEFUL_O1.name]["K_O2"],
              o2[B.IRRELEVANT.name]["K_O2"]]
    axes[0].bar(names, values); axes[0].set(ylabel="K(f2) bits", title="Only useful history shortens O2")
    axes[1].plot([row["stage"] for row in trace], [row["K_P0"] for row in trace], "o-", label="original P0")
    axes[1].plot([row["stage"] for row in trace], [row["K_history"] for row in trace], "s-", label="accumulated basis")
    axes[1].set(xlabel="abstraction stage", ylabel="semantic bits", title="Cumulative representation")
    axes[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig("cumulative_instruction_bootstrapping.png", dpi=110); plt.close(fig)
    return True


def main():
    started, lines = time.time(), []

    def log(message):
        print(message, flush=True)
        lines.append(message)

    log("exp602 Cumulative Instruction Bootstrapping")
    o1 = block_o1(log)
    o2 = block_o2_controls(log)
    trace = block_cumulative_trace(log)
    verdict = block_verdict(o2, trace, log)
    plotted = _plot(o2, trace)
    log(f"plot written: {plotted}")
    log(f"total time: {time.time() - started:.1f}s")
    results = {"config": {"n_tuples": N_TUPLES}, "o1": o1, "o2_controls": o2,
               "cumulative_trace": trace, "verdict": verdict}
    with open("cumulative_instruction_bootstrapping_results.json", "w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)
    with open("cumulative_instruction_bootstrapping_run.log", "w", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()