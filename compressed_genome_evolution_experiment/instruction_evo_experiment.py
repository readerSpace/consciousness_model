"""Runner for exp599b -- Emergent Genetic Instruction Set.

Run: python instruction_evo_experiment.py
Writes instruction_evo_results.json, instruction_evo_run.log, and (when
matplotlib is available) instruction_evo.png.
"""

from __future__ import annotations

import json
import time

import codec_evo as C
import instruction_evo as I
import numpy as np


N_COMPS = [1, 2, 4, 6, 8, 12]
SEEDS = [1, 2, 3]


def _summary(costs):
    analysis = costs["analysis"]
    return {
        key: costs[key] for key in ("fixed", "birth_on", "oracle", "born",
                                    "n_candidates", "n_opcodes", "D_language",
                                    "R_opcode_use", "L_D", "L_G_given_D")
    } | {"compose_equivalent": bool(analysis and analysis.compose_equivalent)}


def block_controls(log):
    log("=" * 78)
    log("BLOCK 1  instruction-birth OFF / ON / oracle across three worlds")
    log("=" * 78)
    rng = np.random.default_rng(7)
    worlds = {
        "flat_REP": C.flat_world(2, 96, 2, rng),
        "compositional": C._world_for_ncomp(2, 8, 3, 4, rng),
        "random": C.random_world(2, 96, rng),
    }
    results = {}
    for name, world in worlds.items():
        costs = I.condition_costs(world)
        results[name] = _summary(costs)
        log(f"  {name:>14}: fixed={costs['fixed']:3d} birth_ON={costs['birth_on']:3d} "
            f"oracle={costs['oracle']:3d} born={costs['born']} "
            f"compose_equiv={bool(costs['analysis'] and costs['analysis'].compose_equivalent)}")
    return results


def block_threshold(log):
    log("=" * 78)
    log("BLOCK 2  exact MDL threshold for primitive-defined instruction birth")
    log("=" * 78)
    rows, nstar = [], None
    for n_comp in N_COMPS:
        costs = I.condition_costs(C._world_for_ncomp(2, n_comp, 3, 4,
                                                      np.random.default_rng(1)))
        row = {"n_comp": n_comp, **_summary(costs)}
        rows.append(row)
        if nstar is None and costs["born"]:
            nstar = n_comp
        log(f"  n_comp={n_comp:2d}: fixed={costs['fixed']:3d} ON={costs['birth_on']:3d} "
            f"oracle={costs['oracle']:3d} L(D)={costs['L_D']:2d} "
            f"L(G|D)={costs['L_G_given_D']:3d} born={costs['born']}")
    log(f"  -> n_opcode_birth*_MDL = {nstar}")
    return {"rows": rows, "n_opcode_birth_MDL": nstar}


def block_evolution(log):
    log("=" * 78)
    log("BLOCK 3  grammar-mutation proposal selection -> n_opcode_birth_evo")
    log("=" * 78)
    result = I.birth_evolution_sweep(N_COMPS, SEEDS)
    for row in result["rows"]:
        log(f"  n_comp={row['n_comp']:2d}: fraction with retained born opcode="
            f"{row['frac_born']:.2f}")
    log(f"  -> n_opcode_birth_evo = {result['n_opcode_birth_evo']}")
    return result


def block_verdict(controls, threshold, evolution, log):
    comp = controls["compositional"]
    controls_ok = (not controls["flat_REP"]["born"] and not controls["random"]["born"]
                   and comp["born"] and comp["birth_on"] == comp["oracle"] < comp["fixed"])
    semantic_ok = comp["compose_equivalent"]
    thresholds_match = threshold["n_opcode_birth_MDL"] == evolution["n_opcode_birth_evo"]
    verdict = {
        "controls_match_prediction": controls_ok,
        "invented_opcode_is_compose_equivalent": semantic_ok,
        "n_opcode_birth_MDL": threshold["n_opcode_birth_MDL"],
        "n_opcode_birth_evo": evolution["n_opcode_birth_evo"],
        "evo_tracks_MDL": thresholds_match,
    }
    verdict["PRIMITIVE_GRAMMAR_INVENTS_COMPOSE_EQUIVALENT"] = bool(
        controls_ok and semantic_ok and thresholds_match)
    log("=" * 78)
    log("BLOCK 4  conservative verdict")
    log("=" * 78)
    for key, value in verdict.items():
        log(f"  {key:<45}: {value}")
    return verdict


def _plot(threshold, evolution):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    rows = threshold["rows"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    x = [row["n_comp"] for row in rows]
    axes[0].plot(x, [row["fixed"] for row in rows], "o-", label="birth OFF")
    axes[0].plot(x, [row["birth_on"] for row in rows], "s-", label="birth ON")
    axes[0].plot(x, [row["oracle"] for row in rows], "^-", label="oracle")
    axes[0].set(xlabel="distinct compositions", ylabel="L(D)+L(G|D)",
                title="Primitive opcode birth lowers total description")
    axes[0].legend(fontsize=8)
    axes[1].plot(x, [row["frac_born"] for row in evolution["rows"]], "o-")
    axes[1].set(xlabel="distinct compositions", ylabel="fraction born",
                ylim=(-.05, 1.05), title="Proposal-selection birth threshold")
    fig.tight_layout()
    fig.savefig("instruction_evo.png", dpi=110)
    plt.close(fig)
    return True


def main():
    t0, lines = time.time(), []

    def log(message):
        print(message, flush=True)
        lines.append(message)

    log("exp599b Emergent Genetic Instruction Set  primitive grammar={LITERAL,REF,CONCAT,REPEAT}")
    controls = block_controls(log)
    threshold = block_threshold(log)
    evolution = block_evolution(log)
    verdict = block_verdict(controls, threshold, evolution, log)
    plotted = _plot(threshold, evolution)
    log(f"plot written: {plotted}")
    log(f"total time: {time.time() - t0:.1f}s")
    results = {"config": {"n_comps": N_COMPS, "seeds": SEEDS, "max_tokens": 3},
               "controls": controls, "threshold": threshold,
               "birth_evolution": evolution, "verdict": verdict}
    with open("instruction_evo_results.json", "w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)
    with open("instruction_evo_run.log", "w", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()