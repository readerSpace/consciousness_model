"""Runner for exp600 -- Instruction Necessity / Alternative-Grammar Audit."""

from __future__ import annotations

import json
import time

import alternative_grammar_audit as A
import codec_evo as C
import numpy as np


CAPABLE = ("postfix_ref_concat", "copy_append", "stack_merge")
N_COMPS = [1, 2, 4, 6, 8, 12]


def _summary(result):
    analysis = result["analysis"]
    definition = result["definition"]
    return {
        "fixed": result["fixed"], "total": result["total"], "born": result["born"],
        "n_candidates": result["n_candidates"], "L_D": result["L_D"],
        "L_G_given_D": result["L_G_given_D"],
        "tokens": [token.name for token in definition.tokens] if definition else [],
        "concatenate_equivalent": bool(analysis and analysis.concatenate_equivalent),
    }


def block_convergence(log):
    log("=" * 78)
    log("BLOCK 1  distinct primitive grammars: semantic convergence audit")
    log("=" * 78)
    world = C._world_for_ncomp(2, 8, 3, 4, np.random.default_rng(1))
    raw = A.convergence_audit(world, CAPABLE)
    out = {name: _summary(result) for name, result in raw.items()}
    for name, result in out.items():
        log(f"  {name:>20}: fixed={result['fixed']:3d} born={result['total']:3d} "
            f"L(D)={result['L_D']:2d} tokens={result['tokens']} "
            f"concat_equiv={result['concatenate_equivalent']}")
    return out


def block_controls(log):
    log("=" * 78)
    log("BLOCK 2  environmental and grammar-feasibility controls")
    log("=" * 78)
    rng = np.random.default_rng(7)
    worlds = {"flat_REP": C.flat_world(2, 96, 2, rng),
              "random": C.random_world(2, 96, rng)}
    out = {}
    for world_name, world in worlds.items():
        out[world_name] = {name: _summary(A.audit_grammar(world, name)) for name in CAPABLE}
        assert all(not row["born"] for row in out[world_name].values())
        log(f"  {world_name:>20}: all capable grammars retain no opcode")
    incapable = _summary(A.audit_grammar(
        C._world_for_ncomp(2, 8, 3, 4, np.random.default_rng(1)), "copy_repeat_only"))
    out["copy_repeat_only"] = incapable
    log(f"  {'copy_repeat_only':>20}: fixed={incapable['fixed']:3d} total={incapable['total']:3d} "
        f"born={incapable['born']} (no binary merge primitive)")
    return out


def block_thresholds(log):
    log("=" * 78)
    log("BLOCK 3  instruction-birth thresholds by grammar")
    log("=" * 78)
    out = {name: A.threshold(name, N_COMPS) for name in (*CAPABLE, "copy_repeat_only")}
    for name, value in out.items():
        log(f"  {name:>20}: n_opcode_birth*_MDL={value}")
    return out


def verdict(convergence, controls, thresholds, log):
    semantic_convergence = all(row["born"] and row["concatenate_equivalent"]
                               for row in convergence.values())
    environmental_specific = all(not row["born"] for world in ("flat_REP", "random")
                                 for row in controls[world].values())
    feasibility_control = not controls["copy_repeat_only"]["born"]
    thresholds_match = all(thresholds[name] == 2 for name in CAPABLE)
    result = {
        "semantic_convergence_across_grammars": semantic_convergence,
        "no_birth_in_flat_or_random_controls": environmental_specific,
        "no_birth_without_binary_merge_primitive": feasibility_control,
        "capable_grammar_thresholds_match": thresholds_match,
    }
    result["COMPOSITION_SEMANTICS_CONVERGENT_WITHIN_AUDITED_GRAMMARS"] = all(result.values())
    log("=" * 78)
    log("BLOCK 4  conservative audit verdict")
    log("=" * 78)
    for key, value in result.items():
        log(f"  {key:<55}: {value}")
    return result


def _plot(convergence, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    names = list(convergence)
    fixed = [convergence[name]["fixed"] for name in names]
    born = [convergence[name]["total"] for name in names]
    fig, ax = plt.subplots(figsize=(8, 4))
    index = np.arange(len(names))
    ax.bar(index - .18, fixed, .36, label="fixed language")
    ax.bar(index + .18, born, .36, label="born instruction")
    ax.set(xticks=index, xticklabels=names, ylabel="L(D)+L(G|D)",
           title="Independent grammars converge on a shorter composition operation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return True


def main():
    started, lines = time.time(), []

    def log(message):
        print(message, flush=True)
        lines.append(message)

    log("exp600 Instruction Necessity / Alternative-Grammar Audit")
    convergence = block_convergence(log)
    controls = block_controls(log)
    thresholds = block_thresholds(log)
    audit_verdict = verdict(convergence, controls, thresholds, log)
    plotted = _plot(convergence, "alternative_grammar_audit.png")
    log(f"plot written: {plotted}")
    log(f"total time: {time.time() - started:.1f}s")
    results = {"config": {"capable_grammars": CAPABLE, "n_comps": N_COMPS},
               "convergence": convergence, "controls": controls,
               "thresholds": thresholds, "verdict": audit_verdict}
    with open("alternative_grammar_audit_results.json", "w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)
    with open("alternative_grammar_audit_run.log", "w", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()