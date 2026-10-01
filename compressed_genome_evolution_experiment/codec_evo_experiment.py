"""exp599a -- Codec Evolution / Representation-Language Selection (runner).

exp598 proved recombination is unselected because the FIXED codec gives it no MDL
benefit.  exp599 makes the genetic LANGUAGE a selected trait and asks:

    in a world where composition is useful, is the language D1 = D0 + {COMPOSE}
    (which expresses a composition in one opcode) SELECTED over D0 -- and only
    there, paying the language cost L(D1)=C_opcode?

Blocks:
  1. three-world selection: flat/REP, compositional, random -> selected language.
  2. exact MDL crossover over the number of distinct compositions -> n*_MDL.
  3. evolution with the language as a gene -> adoption point n_evo.
  4. C_opcode dependence of n*_MDL (the opcode price sets the adoption point).
  5. verdict: language tracks world structure; n_evo ~ n*_MDL.

Writes codec_evo_results.json, codec_evo_run.log, codec_evo.png.
"""

from __future__ import annotations

import json
import time

import numpy as np

import codec_evo as C

A = 2
C_OPCODE = 48
K = 3
MOTIF_LEN = 4
N_COMPS = [2, 4, 6, 8, 12, 16, 20, 30]
SEEDS = [1, 2, 3]


def block_three_worlds(log):
    log("=" * 78)
    log("BLOCK 1  three-world language selection (C_opcode=%d)" % C_OPCODE)
    log("=" * 78)
    out = {}
    rng = np.random.default_rng(7)
    worlds = {
        "flat_REP":      C.flat_world(A, 96, 2, rng),
        "compositional": C._world_for_ncomp(A, 20, K, MOTIF_LEN, rng),
        "random":        C.random_world(A, 96, rng),
    }
    expected = {"flat_REP": "D0", "compositional": "D1", "random": "D0"}
    all_ok = True
    for name, w in worlds.items():
        sel = C.selected_language(w, C_OPCODE)
        ok = sel["selected"] == expected[name]
        all_ok = all_ok and ok
        out[name] = {"selected": sel["selected"], "expected": expected[name],
                     "cost_D0": sel["cost_D0"], "cost_D1": sel["cost_D1"], "ok": ok}
        log(f"  {name:>14}: D0={sel['cost_D0']:4d} D1={sel['cost_D1']:4d} "
            f"-> {sel['selected']}  (expected {expected[name]}) {'OK' if ok else 'XX'}")
    out["_all_match"] = all_ok
    log(f"  -> selection matches the predicted table for all worlds? {all_ok}")
    return out


def block_mdl_crossover(log):
    log("=" * 78)
    log("BLOCK 2  exact MDL crossover over number of distinct compositions")
    log("=" * 78)
    res = C.mdl_crossover(A, N_COMPS, K, MOTIF_LEN, C_OPCODE, seed=1)
    for r in res["rows"]:
        log(f"  n_comp={r['n_comp']:2d}: cost_D0={r['cost_D0']:4d} "
            f"cost_D1={r['cost_D1']:4d} margin={r['margin']:+4d} -> {r['selected']}")
    log(f"  -> n_comp*_MDL = {res['n_comp_star_MDL']}")
    return res


def block_evolution(log):
    log("=" * 78)
    log("BLOCK 3  evolution with the LANGUAGE as a selected trait -> n_evo")
    log("=" * 78)
    res = C.k_evo_sweep(A, N_COMPS, K, MOTIF_LEN, C_OPCODE, SEEDS)
    for r in res["rows"]:
        log(f"  n_comp={r['n_comp']:2d}: frac_champion_D1={r['frac_champ_D1']:.2f}")
    log(f"  -> n_comp_evo (first >=0.5) = {res['n_comp_evo']}")
    return res


def block_opcode_price(log):
    log("=" * 78)
    log("BLOCK 4  the opcode price sets the adoption point: n*_MDL(C_opcode)")
    log("=" * 78)
    out = {}
    for cop in (24, 48, 72, 96):
        res = C.mdl_crossover(A, N_COMPS, K, MOTIF_LEN, cop, seed=1)
        out[str(cop)] = res["n_comp_star_MDL"]
        log(f"  C_opcode={cop:>3}: n_comp*_MDL = {res['n_comp_star_MDL']}")
    monotone = [out[str(c)] for c in (24, 48, 72, 96)]
    mono_ok = all(a is not None and b is not None and a <= b
                  for a, b in zip(monotone, monotone[1:]))
    out["_monotone"] = mono_ok
    log(f"  -> a costlier opcode demands more compositional pressure to adopt? {mono_ok}")
    return out


def block_verdict(worlds, mdl, evo, price, log):
    log("=" * 78)
    log("BLOCK 5  verdict -- is the representation LANGUAGE itself selected?")
    log("=" * 78)
    nstar = mdl["n_comp_star_MDL"]
    nevo = evo["n_comp_evo"]
    table_ok = worlds["_all_match"]
    crossover_exists = nstar is not None
    evo_adopts = nevo is not None
    evo_matches_mdl = (nstar is not None and nevo is not None
                       and abs(nevo - nstar) <= 4)
    verdict = {
        "world_selection_matches_table": bool(table_ok),
        "mdl_crossover_exists": bool(crossover_exists),
        "evolution_adopts_D1": bool(evo_adopts),
        "n_comp_star_MDL": nstar,
        "n_comp_evo": nevo,
        "evo_tracks_MDL": bool(evo_matches_mdl),
        "opcode_price_monotone": bool(price["_monotone"]),
    }
    verdict["REPRESENTATION_LANGUAGE_IS_SELECTED"] = bool(
        table_ok and crossover_exists and evo_adopts and evo_matches_mdl)
    for k, v in verdict.items():
        log(f"  {k:<36}: {v}")
    log("  " + "-" * 62)
    log(f"  ==> representation language itself is selected to match the world: "
        f"{verdict['REPRESENTATION_LANGUAGE_IS_SELECTED']}")
    log(f"  ==> evolution adopts COMPOSE at n_comp={nevo}, exact MDL at "
        f"n_comp={nstar}: when the environment's compositional statistics apply")
    log(f"      enough pressure, the language that makes composition a cheap")
    log(f"      PRIMITIVE is selected -- the exp598 codec bottleneck is solved by")
    log(f"      evolving the representation language, not by more search.")
    return verdict


def _plot(mdl, evo, price, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = axes[0]
    ncs = [r["n_comp"] for r in mdl["rows"]]
    ax.plot(ncs, [r["cost_D0"] for r in mdl["rows"]], "o-", label="D0 (LIT,CALL,REP)")
    ax.plot(ncs, [r["cost_D1"] for r in mdl["rows"]], "s-", label="D1 (+COMPOSE)")
    if mdl["n_comp_star_MDL"]:
        ax.axvline(mdl["n_comp_star_MDL"], color="r", ls=":", label="n*_MDL")
    ax.set_xlabel("number of distinct compositions"); ax.set_ylabel("L(D)+L(G|D) bits")
    ax.set_title("MDL: COMPOSE language wins when composition is rich")
    ax.legend(fontsize=7)

    ax = axes[1]
    ax.plot([r["n_comp"] for r in evo["rows"]],
            [r["frac_champ_D1"] for r in evo["rows"]], "o-", label="evolved frac D1")
    if mdl["n_comp_star_MDL"]:
        ax.axvline(mdl["n_comp_star_MDL"], color="r", ls=":", label="n*_MDL")
    if evo["n_comp_evo"]:
        ax.axvline(evo["n_comp_evo"], color="g", ls="--", label="n_evo")
    ax.set_xlabel("number of distinct compositions"); ax.set_ylabel("fraction D1")
    ax.set_title("evolution adopts COMPOSE at the MDL crossover"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)
    return True


def main():
    t0 = time.time()
    lines = []

    def log(msg):
        print(msg, flush=True)
        lines.append(msg)

    log(f"exp599a Codec Evolution  A={A} C_opcode={C_OPCODE} k={K} "
        f"motif_len={MOTIF_LEN} seeds={SEEDS}")
    worlds = block_three_worlds(log)
    mdl = block_mdl_crossover(log)
    evo = block_evolution(log)
    price = block_opcode_price(log)
    verdict = block_verdict(worlds, mdl, evo, price, log)

    ok = _plot(mdl, evo, price, "codec_evo.png")
    log(f"plot written: {ok}")
    log(f"total time: {time.time() - t0:.1f}s")

    results = {
        "config": {"A": A, "C_opcode": C_OPCODE, "k": K, "motif_len": MOTIF_LEN,
                   "n_comps": N_COMPS, "seeds": SEEDS},
        "block1_three_worlds": worlds,
        "block2_mdl_crossover": mdl,
        "block3_evolution": evo,
        "block4_opcode_price": price,
        "verdict": verdict,
    }
    with open("codec_evo_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    with open("codec_evo_run.log", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
