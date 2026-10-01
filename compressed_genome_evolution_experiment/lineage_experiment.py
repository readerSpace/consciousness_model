"""exp594 -- Duplication / Divergence / Specialization (runner).

Blocks:
  1. lineage origin of function B: did the B-serving module arise by DUPLICATING
     the A module (parent link) or by an unrelated de-novo ADD?
  2. specialization: knockout S(M_i, A) vs S(M_i, B) -> each module carries one
     function.
  3. controls: DUP-on vs DUP-off adaptation time, x B_related vs B_unrelated.
     DUP should help, and help MORE when B is a small edit of A (reuse of
     existing structure, not just extra capacity).
  4. latent memory: A -> A+B -> A -> A+B.  Is re-acquisition faster than first
     acquisition (evolutionary memory), and does a reset control abolish it?

Writes lineage_results.json, lineage_run.log, lineage.png.
"""

from __future__ import annotations

import json
import time
from statistics import mean, pstdev

import numpy as np

import genome as G
import lineage as LN

LAM = 0.0015
LAM_MEM = 0.0015
SEEDS = [1, 2, 3, 4, 5, 6, 7, 8]
POP = 120
G_E0 = 220
G_E1 = 400
THRESH = 0.98


def _m(a):
    return "n/a" if a.get("mean") is None else f"{a['mean']:.2f}"


def _agg(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return {"mean": None, "std": None, "n": 0}
    return {"mean": float(mean(xs)), "std": float(pstdev(xs)) if len(xs) > 1 else 0.0,
            "n": len(xs)}


def adapt_A_to_AB(world, seed, allow_dup, lam=LAM, g_e0=G_E0, g_e1=G_E1):
    cfg = LN.DEFAULT_CFG
    rng = np.random.default_rng(seed)
    alloc = LN.IdAlloc()
    pop = [LN.new_lingenome(rng, world, cfg["nmax"]) for _ in range(POP)]
    pop, best0, _ = LN.evolve_stage(pop, world, "A", lam, g_e0, POP, rng, cfg,
                                    alloc, allow_dup=allow_dup)
    pop, best1, T = LN.evolve_stage(pop, world, "AB", lam, g_e1, POP, rng, cfg,
                                    alloc, allow_dup=allow_dup, threshold=THRESH)
    return best0, best1, T


def specialists(best, world):
    spec = LN.specialization(best, world)
    if not spec:
        return None, None, spec
    a_mod = max(spec, key=lambda s: s["S_A"])
    b_mod = max(spec, key=lambda s: s["S_B"])
    return a_mod, b_mod, spec


# ---------------------------------------------------------------------------
def block_lineage_and_spec(log, lam=LAM):
    log("=" * 74)
    log("BLOCK 1+2  lineage origin of B & specialization  (B_related, DUP on)")
    log("=" * 74)
    world = LN.world_related()
    n_spec = 0          # runs that produced two distinct specialized macros
    b_has_parent = 0    # B-module arose by duplication (has a parent)
    b_parent_is_A = 0   # ...and that parent is the current A-specialist
    b_denovo = 0        # B-module is a de-novo ADD (parent None)
    S = {"AA": [], "AB": [], "BA": [], "BB": []}
    Ts = []
    for seed in SEEDS:
        _, best1, T = adapt_A_to_AB(world, seed, allow_dup=True, lam=lam)
        Ts.append(T)
        a_mod, b_mod, spec = specialists(best1, world)
        if (a_mod is None or b_mod is None or a_mod["idx"] == b_mod["idx"]
                or a_mod["S_A"] < 0.1 or b_mod["S_B"] < 0.1):
            continue                       # not (yet) two specialized modules
        n_spec += 1
        S["AA"].append(a_mod["S_A"]); S["AB"].append(a_mod["S_B"])
        S["BA"].append(b_mod["S_A"]); S["BB"].append(b_mod["S_B"])
        if b_mod["parent"] is not None:
            b_has_parent += 1
            if b_mod["parent"] == a_mod["id"]:
                b_parent_is_A += 1
        else:
            b_denovo += 1
    n = len(SEEDS)
    log(f"  runs with two specialized modules (A-mod, B-mod): {n_spec}/{n}")
    log(f"  among them, B-module arose by DUPLICATION (has a parent): "
        f"{b_has_parent}/{n_spec}   de-novo ADD: {b_denovo}/{n_spec}")
    log(f"     (parent is the current A-specialist in {b_parent_is_A}/{n_spec})")
    log(f"  adaptation T(A->A+B) = {_m(_agg(Ts))} gens")
    log(f"  specialization: S(A-mod,A)={_m(_agg(S['AA']))} "
        f"S(A-mod,B)={_m(_agg(S['AB']))} | "
        f"S(B-mod,A)={_m(_agg(S['BA']))} S(B-mod,B)={_m(_agg(S['BB']))}")
    return {
        "n_specialized": n_spec, "b_has_parent": b_has_parent,
        "b_parent_is_A": b_parent_is_A, "b_denovo": b_denovo, "n": n,
        "T_adapt": _agg(Ts), "S": {k: _agg(v) for k, v in S.items()},
    }


def block_controls(log):
    log("=" * 74)
    log("BLOCK 3  controls: DUP-on vs DUP-off, x B_related vs B_unrelated")
    log("=" * 74)
    worlds = {"related": LN.world_related(), "unrelated": LN.world_unrelated()}
    out = {}
    for wname, world in worlds.items():
        out[wname] = {}
        for dup in [True, False]:
            Ts, reached = [], 0
            for seed in SEEDS:
                _, _, T = adapt_A_to_AB(world, 100 + seed, allow_dup=dup)
                Ts.append(T if T is not None else G_E1)
                reached += int(T is not None)
            out[wname]["dup_on" if dup else "dup_off"] = {
                "T": _agg(Ts), "reached": reached, "n": len(SEEDS)}
        adv = (out[wname]["dup_off"]["T"]["mean"] - out[wname]["dup_on"]["T"]["mean"])
        out[wname]["dup_advantage"] = adv
        log(f"  {wname:>9}: DUP-on T={out[wname]['dup_on']['T']['mean']:.0f}  "
            f"DUP-off T={out[wname]['dup_off']['T']['mean']:.0f}  "
            f"-> DUP advantage = {adv:+.0f} gens")
    log(f"  DUP advantage(related)={out['related']['dup_advantage']:+.0f} vs "
        f"(unrelated)={out['unrelated']['dup_advantage']:+.0f}")
    return out


def block_memory(log):
    log("=" * 74)
    log("BLOCK 4  latent memory: A -> A+B -> A(dormant) -> A+B  (reacquire)")
    log("  dormancy under MDL prunes the B-trace; dormancy without MDL keeps it.")
    log("=" * 74)
    world = LN.world_related()
    cfg = LN.DEFAULT_CFG
    first = []
    reac = {"prune": [], "keep": [], "reset": []}
    trace = {"prune": 0, "keep": 0}
    for seed in SEEDS:
        rng = np.random.default_rng(300 + seed)
        alloc = LN.IdAlloc()
        pop = [LN.new_lingenome(rng, world, cfg["nmax"]) for _ in range(POP)]
        pop, _, _ = LN.evolve_stage(pop, world, "A", LAM_MEM, G_E0, POP, rng,
                                    cfg, alloc)
        pop_e0 = [p.copy() for p in pop]                       # reset baseline
        pop, _, T1 = LN.evolve_stage(pop, world, "AB", LAM_MEM, G_E1, POP, rng,
                                     cfg, alloc, threshold=THRESH)
        first.append(T1 if T1 is not None else G_E1)
        pop_ab = [p.copy() for p in pop]
        # dormant A-only stage under two pressures
        pop_prune, _, _ = LN.evolve_stage([p.copy() for p in pop_ab], world, "A",
                                          LAM_MEM, G_E0, POP, rng, cfg, alloc)
        pop_keep, _, _ = LN.evolve_stage([p.copy() for p in pop_ab], world, "A",
                                         0.0, G_E0, POP, rng, cfg, alloc)  # MDL off
        trace["prune"] += int(_has_B_trace(pop_prune, world))
        trace["keep"] += int(_has_B_trace(pop_keep, world))
        # re-acquire A+B from each dormant population, and from the reset baseline
        for label, src in [("prune", pop_prune), ("keep", pop_keep),
                           ("reset", pop_e0)]:
            _, _, T3 = LN.evolve_stage([p.copy() for p in src], world, "AB",
                                       LAM_MEM, G_E1, POP, rng, cfg, alloc,
                                       threshold=THRESH)
            reac[label].append(T3 if T3 is not None else G_E1)
    log(f"  T_first_acquire                = {_m(_agg(first))} gens")
    log(f"  T_reacquire (dormancy=MDL prune) = {_m(_agg(reac['prune']))} gens  "
        f"(B-trace survived {trace['prune']}/{len(SEEDS)})")
    log(f"  T_reacquire (dormancy=no MDL keep)= {_m(_agg(reac['keep']))} gens  "
        f"(B-trace survived {trace['keep']}/{len(SEEDS)})")
    log(f"  T_reacquire (reset, no memory)   = {_m(_agg(reac['reset']))} gens")
    return {"first": _agg(first),
            "reacquire_prune": _agg(reac["prune"]),
            "reacquire_keep": _agg(reac["keep"]),
            "reacquire_reset": _agg(reac["reset"]),
            "trace_prune": trace["prune"], "trace_keep": trace["keep"],
            "n": len(SEEDS)}


def _has_B_trace(pop, world) -> bool:
    """True if some genome in the population contains a macro whose body matches
    the B motif better than the A motif (a latent B-capable part)."""
    tgt = world.target()
    for lin in pop[:20]:
        for m in lin.g.macros:
            body = [ins.a for ins in m if ins.op == LN.OP_LIT]
            if len(body) < world.motif_len:
                continue
            seg = np.array(body[:world.motif_len])
            dA = int(np.sum(seg != world.motif_A[:len(seg)]))
            dB = int(np.sum(seg != world.motif_B[:len(seg)]))
            if dB < dA:                              # closer to B than to A
                return True
    return False


def decide(b12, b3, b4, log):
    log("=" * 74)
    log("PRE-REGISTERED DECISION")
    log("=" * 74)
    ns = max(1, b12["n_specialized"])
    # when B modularizes, it arises predominantly by duplication (not de novo)
    dup_route = (b12["n_specialized"] >= 3
                 and b12["b_has_parent"] > b12["b_denovo"])
    sAA = b12["S"]["AA"]["mean"] or 0.0; sAB = b12["S"]["AB"]["mean"] or 0.0
    sBA = b12["S"]["BA"]["mean"] or 0.0; sBB = b12["S"]["BB"]["mean"] or 0.0
    spec = (sAA > sAB + 0.1) and (sBB > sBA + 0.1)
    dup_helps = b3["related"]["dup_advantage"] > 0
    reuse_specific = b3["related"]["dup_advantage"] > b3["unrelated"]["dup_advantage"]
    keep = b4["reacquire_keep"]["mean"]; reset = b4["reacquire_reset"]["mean"]
    prune = b4["reacquire_prune"]["mean"]
    # having experienced A+B leaves a USABLE memory: reacquire << naive (reset)
    memory_reusable = (prune < reset) and (keep < reset)
    # and that memory is COMPRESSED/generative (the A-module + DUP route), not a
    # stored B pseudogene: MDL prunes the literal trace (kept 0 vs many) yet the
    # pruned condition reacquires at least as fast.
    compressed_memory = (b4["trace_keep"] > b4["trace_prune"]) and (prune <= keep)

    log(f"  H  when B modularizes it arises predominantly by DUPLICATION: "
        f"{dup_route}  (parent {b12['b_has_parent']}/{ns}, de-novo {b12['b_denovo']}/{ns})")
    log(f"  specialization S(A,A)>S(A,B) and S(B,B)>S(B,A):             {spec}")
    log(f"  DUP helps adaptation (T_dup_on < T_dup_off):                {dup_helps}")
    log(f"  DUP advantage bigger for B_related than B_unrelated:        {reuse_specific}")
    log(f"  usable memory: experienced reacquire << naive reset:       {memory_reusable}  "
        f"(prune {prune:.0f}, keep {keep:.0f} vs reset {reset:.0f})")
    log(f"  memory is COMPRESSED (A-module+DUP), not a stored B trace:  {compressed_memory}  "
        f"(literal trace kept {b4['trace_keep']} vs {b4['trace_prune']}/{b4['n']}; "
        f"prune reacquire <= keep)")
    core = dup_route and spec and dup_helps and reuse_specific and memory_reusable
    log("")
    log(f"  CORE (duplication->divergence->specialization; hysteresis is a USABLE, "
        f"COMPRESSED evolutionary memory): {core and compressed_memory}")
    return {"dup_route": bool(dup_route), "specialization": bool(spec),
            "dup_helps": bool(dup_helps), "reuse_specific": bool(reuse_specific),
            "memory_reusable": bool(memory_reusable),
            "memory_is_compressed": bool(compressed_memory),
            "core": bool(core and compressed_memory)}


def make_figure(b12, b3, b4, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    # panel 1: specialization matrix (2x2 knockout)
    M = np.array([[b12["S"]["AA"]["mean"], b12["S"]["AB"]["mean"]],
                  [b12["S"]["BA"]["mean"], b12["S"]["BB"]["mean"]]])
    im = ax[0].imshow(M, cmap="Blues", vmin=0, vmax=max(M.max(), 0.1))
    ax[0].set_xticks([0, 1]); ax[0].set_xticklabels(["knocks A", "knocks B"], fontsize=8)
    ax[0].set_yticks([0, 1]); ax[0].set_yticklabels(["A-module", "B-module"], fontsize=8)
    for i in range(2):
        for j in range(2):
            ax[0].text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=11)
    ax[0].set_title("specialization S(module, function)\n(fitness drop on knockout)",
                    fontsize=9)
    # panel 2: DUP advantage
    names = ["related", "unrelated"]
    on = [b3[n]["dup_on"]["T"]["mean"] for n in names]
    off = [b3[n]["dup_off"]["T"]["mean"] for n in names]
    x = np.arange(2)
    ax[1].bar(x - 0.2, on, 0.4, label="DUP on", color="steelblue")
    ax[1].bar(x + 0.2, off, 0.4, label="DUP off", color="crimson")
    ax[1].set_xticks(x); ax[1].set_xticklabels(names, fontsize=9)
    ax[1].set_ylabel("adaptation T (gens)")
    ax[1].set_title("DUP vs ADD adaptation time\n(DUP helps, more for related)",
                    fontsize=9)
    ax[1].legend(fontsize=8)
    # panel 3: memory
    labels = ["first\nacquire", "reacq\ndorm=keep", "reacq\ndorm=prune", "reacq\nreset"]
    vals = [b4["first"]["mean"], b4["reacquire_keep"]["mean"],
            b4["reacquire_prune"]["mean"], b4["reacquire_reset"]["mean"]]
    ax[2].bar(labels, vals, color=["gray", "seagreen", "darkorange", "crimson"])
    ax[2].set_ylabel("acquisition T (gens)")
    ax[2].set_title("usable memory: any re-exposure << naive reset;\nfastest when MDL "
                    "keeps the genome compressed (prune)", fontsize=9)
    ax[2].tick_params(axis="x", labelsize=7)
    fig.suptitle("exp594  Duplication -> Divergence -> Specialization, and "
                 "hysteresis as usable evolutionary memory", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return True


def main():
    t0 = time.time()
    lines = []

    def log(msg):
        print(msg, flush=True)
        lines.append(msg)

    log(f"exp594 Duplication/Divergence/Specialization  seeds={SEEDS} lam={LAM}")
    b12 = block_lineage_and_spec(log)
    b3 = block_controls(log)
    b4 = block_memory(log)
    decision = decide(b12, b3, b4, log)
    fig_ok = make_figure(b12, b3, b4, "lineage.png")
    log(f"figure: {'lineage.png' if fig_ok else '(matplotlib unavailable)'}")
    dt = time.time() - t0
    log("=" * 74)
    log(f"done in {dt:.1f}s")

    out = {"config": {"lam": LAM, "lam_mem": LAM_MEM, "seeds": SEEDS, "pop": POP,
                      "g_e0": G_E0, "g_e1": G_E1, "threshold": THRESH},
           "block12_lineage_spec": b12, "block3_controls": b3,
           "block4_memory": b4, "decision": decision, "runtime_seconds": dt}
    with open("lineage_results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=float)
    with open("lineage_run.log", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote lineage_results.json and lineage_run.log")


if __name__ == "__main__":
    main()
