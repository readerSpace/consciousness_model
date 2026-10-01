"""Runner for exp601 -- Primitive Basis Ablation / Computational Necessity."""

from __future__ import annotations

import json
import time

import codec_evo as C
import numpy as np
import primitive_basis_ablation as P


N_COMPS = [1, 2, 3, 4, 6, 8, 12]


def _summary(result):
    witness = result["witness"]
    return {
        key: result[key] for key in ("fixed", "total", "born", "expressible", "K_concat",
                                     "reusable", "L_D", "L_G_given_D")
    } | {"witness": list(witness.tokens) if witness else []}


def block_basis_complexity(log):
    log("=" * 78)
    log("BLOCK 1  semantic complexity K_P(a||b) and instruction birth")
    log("=" * 78)
    raw = P.basis_audit(N_COMPS)
    out = {name: _summary(result) | {"n_birth_MDL": result["n_birth_MDL"]}
           for name, result in raw.items()}
    for name, row in out.items():
        log(f"  {name:>27}: expressible={row['expressible']} K={row['K_concat']} "
            f"reusable={row['reusable']} threshold={row['n_birth_MDL']} "
            f"fixed={row['fixed']} total={row['total']} witness={row['witness']}")
    return out


def block_environment_controls(log):
    log("=" * 78)
    log("BLOCK 2  flat/random controls for expressible bases")
    log("=" * 78)
    rng = np.random.default_rng(7)
    out = {}
    for name, world in (("flat_REP", C.flat_world(2, 96, 2, rng)),
                        ("random", C.random_world(2, 96, rng))):
        out[name] = {basis: _summary(P.audit_basis(world, basis))
                     for basis in ("full", "distant_box_pair_flatten")}
        log(f"  {name:>27}: full born={out[name]['full']['born']} "
            f"distant born={out[name]['distant_box_pair_flatten']['born']}")
    return out


def block_verdict(bases, controls, log):
    necessity = all(not bases[name]["expressible"] and not bases[name]["born"]
                    for name in ("minus_ref", "minus_order", "minus_merge"))
    reuse = (bases["minus_reuse"]["expressible"] and not bases["minus_reuse"]["born"])
    distance = (bases["full"]["K_concat"] < bases["distant_box_pair_flatten"]["K_concat"]
                and bases["full"]["n_birth_MDL"] < bases["distant_box_pair_flatten"]["n_birth_MDL"])
    controls_ok = all(not row["born"] for control in controls.values() for row in control.values())
    verdict = {
        "REF_ORDER_MERGE_ARE_COMPUTATIONALLY_NECESSARY": necessity,
        "REUSE_IS_SELECTIVELY_NECESSARY": reuse,
        "K_P_PREDICTS_LATER_BIRTH_FOR_DISTANT_BASIS": distance,
        "NO_BIRTH_IN_NONCOMPOSITIONAL_WORLDS": controls_ok,
    }
    verdict["BASIS_DISTANCE_GOVERNS_INSTRUCTION_BIRTH_WITHIN_AUDITED_BASES"] = all(verdict.values())
    log("=" * 78)
    log("BLOCK 3  conservative verdict")
    log("=" * 78)
    for key, value in verdict.items():
        log(f"  {key:<62}: {value}")
    return verdict


def _plot(bases):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    names = ["full", "distant_box_pair_flatten"]
    k_values = [bases[name]["K_concat"] for name in names]
    thresholds = [bases[name]["n_birth_MDL"] for name in names]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].bar(names, k_values); axes[0].set(ylabel="K_P(a||b) bits", title="semantic basis distance")
    axes[1].bar(names, thresholds); axes[1].set(ylabel="n*_birth,MDL", title="instruction birth threshold")
    for axis in axes:
        axis.tick_params(axis="x", labelrotation=18, labelsize=8)
    fig.tight_layout(); fig.savefig("primitive_basis_ablation.png", dpi=110); plt.close(fig)
    return True


def main():
    started, lines = time.time(), []

    def log(message):
        print(message, flush=True)
        lines.append(message)

    log("exp601 Primitive Basis Ablation / Computational Necessity")
    bases = block_basis_complexity(log)
    controls = block_environment_controls(log)
    audit_verdict = block_verdict(bases, controls, log)
    plotted = _plot(bases)
    log(f"plot written: {plotted}")
    log(f"total time: {time.time() - started:.1f}s")
    results = {"config": {"n_comps": N_COMPS}, "bases": bases,
               "controls": controls, "verdict": audit_verdict}
    with open("primitive_basis_ablation_results.json", "w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)
    with open("primitive_basis_ablation_run.log", "w", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()