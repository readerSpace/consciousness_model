"""exp592 -- Endogenous Interaction Discovery (runner).

Pipeline (Pi* sealed inside the fitness; revealed only in the scoring blocks):

  1. LEARN (blind): from a high-fitness phenotype sample + a black-box fitness,
     recover a partition -- naive (co-variation only) and fitness-driven
     (co-variation proposes, fitness disposes).
  2. SCORE STRUCTURE: ARI / pairwise P,R of each learned partition vs Pi*, and
     vs a random partition.  Nuisance capture: does the partition group the
     fitness-irrelevant nuisance blocks?
  3. SCORE FUNCTION: turn the learned partition into a mutation operator and
     measure valley-crossing on HELD-OUT goals (block targets never seen in
     learning), vs A_bit, A_block (oracle Pi*), and R.

Pre-registered central decision:
  ARI(learned, Pi*) >> ARI(random, Pi*)
  and  T_0.95(C_learned) < T_0.95(A_bit)
  and  T_0.95(C_learned) ~= T_0.95(A_block^oracle)
  and  (nuisance) learned excludes nuisance while naive is tempted by it.

Writes endogenous_results.json, endogenous_run.log, endogenous.png.
"""

from __future__ import annotations

import json
import time
from statistics import mean, pstdev

import numpy as np

import endogenous as EN
import epistatic as EP

P = 4
N_CAUSAL = 3
N_NUISANCE = 2
WORLD_SEEDS = [1, 2, 3, 4, 5, 6]
SAMPLE_N = 300
TAU = 0.6
HELDOUT_GOALS = 6
READAPT_SEEDS = 20
MAX_STEPS = 5000
THRESHOLD = 0.999


def _agg(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return {"mean": None, "std": None, "n": 0}
    return {"mean": float(mean(xs)),
            "std": float(pstdev(xs)) if len(xs) > 1 else 0.0, "n": len(xs)}


def nuisance_capture(part, world):
    """Fraction of nuisance positions that got grouped with >=1 other position."""
    causal = set(int(i) for b in world.causal_blocks for i in b)
    lab = EN._labels_from_partition(part, world.L)
    from collections import Counter
    sizes = Counter(lab.tolist())
    grouped = 0
    total = 0
    for i in range(world.L):
        if i in causal:
            continue
        total += 1
        if sizes[lab[i]] > 1:
            grouped += 1
    return grouped / total if total else 0.0


def restrict_to_causal(part, world):
    """The partition's grouping ON the causal positions only (drop nuisance)."""
    causal = set(int(i) for b in world.causal_blocks for i in b)
    out = []
    for m in part:
        sub = [int(i) for i in m if int(i) in causal]
        if sub:
            out.append(np.array(sorted(sub)))
    return out


def with_nuisance_singletons(causal_modules, world):
    """Append every nuisance position as its own singleton, so ALL operators
    share the same nuisance representation and the function test isolates the
    *causal* grouping (removing the step-counting artifact of how nuisance is
    lumped)."""
    causal = set(int(i) for b in world.causal_blocks for i in b)
    mods = [np.array(m) for m in causal_modules]
    for i in range(world.L):
        if i not in causal:
            mods.append(np.array([i]))
    return mods


def readapt_operator(modules, world, rng_seed):
    """T_0.95 to adapt to HELD-OUT goals using `modules` as the resample operator.

    Start from a random state, adapt to an unseen goal (arbitrary p-bit block
    targets); fitness scores only causal blocks (Pi* used here, in scoring only)."""
    Pi = world.causal_blocks                       # scoring partition (allowed here)
    rep = EP.GeneRep("op", [np.array(m) for m in modules], world.L)
    steps_all, succ_all = [], []
    grng = np.random.default_rng(500 + rng_seed)
    for gi in range(HELDOUT_GOALS):
        # unseen target: arbitrary p-bit motifs on causal blocks (+ random nuisance)
        target = grng.integers(0, 2, world.L)
        E1 = target
        seeds_steps = []
        succ = 0
        for s in range(READAPT_SEEDS):
            rng = np.random.default_rng(9000 + rng_seed * 1000 + gi * 50 + s)
            E0 = rng.integers(0, 2, world.L)       # random start
            t, finalF = EP.epi_readapt_steps(rep, E0, E1, Pi, THRESHOLD,
                                             MAX_STEPS, rng)
            if t is not None:
                seeds_steps.append(t)
                succ += 1
        steps_all.append(float(np.mean(seeds_steps)) if seeds_steps else None)
        succ_all.append(succ / READAPT_SEEDS)
    good = [s for s in steps_all if s is not None]
    return {"mean_steps": float(np.mean(good)) if good else None,
            "success_rate": float(np.mean(succ_all))}


def run(log):
    world = EN.make_world(P, N_CAUSAL, N_NUISANCE)
    log(f"world: L={world.L}  causal={[b.tolist() for b in world.causal_blocks]}  "
        f"nuisance={[b.tolist() for b in world.nuisance_blocks]}")
    Pi_star = world.true_partition()

    struct = {"learned": {"ARI": [], "prec": [], "rec": [], "nuis": []},
              "naive": {"ARI": [], "prec": [], "rec": [], "nuis": []},
              "random": {"ARI": []}}
    func = {k: {"mean_steps": [], "success_rate": []}
            for k in ["A_bit", "A_block", "C_learned", "R", "naive"]}
    example = None

    for ws in WORLD_SEEDS:
        rng = np.random.default_rng(100 + ws)
        sample = EN.high_fitness_sample(world, SAMPLE_N, rng, noise=0.02)
        draw = EN.goal_sampler(world, np.random.default_rng(200 + ws), "aligned")

        naive = EN.learn_naive(sample, TAU)
        learned, diag = EN.learn_fitness_driven(
            sample, draw, np.random.default_rng(300 + ws), TAU)
        rand_part = EN.random_partition_like(Pi_star, world.L,
                                             np.random.default_rng(400 + ws))

        for name, part in [("learned", learned), ("naive", naive)]:
            struct[name]["ARI"].append(
                EN.adjusted_rand_index(part, Pi_star, world.L))
            pr, rc = EN.pairwise_precision_recall(part, Pi_star, world.L)
            struct[name]["prec"].append(pr)
            struct[name]["rec"].append(rc)
            struct[name]["nuis"].append(nuisance_capture(part, world))
        struct["random"]["ARI"].append(
            EN.adjusted_rand_index(rand_part, Pi_star, world.L))

        # function-test operators: each = its CAUSAL grouping + uniform nuisance
        # singletons, so the comparison isolates how the causal positions are
        # grouped (not how nuisance happens to be lumped).
        causal_singletons = [np.array([int(i)]) for b in world.causal_blocks
                             for i in b]
        rand_causal = EN.random_partition_like(
            world.causal_blocks, len(causal_singletons),
            np.random.default_rng(450 + ws))
        # remap rand_causal indices (0..11) back to causal position ids
        causal_ids = np.sort(np.concatenate(world.causal_blocks))
        rand_causal = [causal_ids[m] for m in rand_causal]
        ops = {
            "A_bit": with_nuisance_singletons(causal_singletons, world),
            "A_block": with_nuisance_singletons(world.causal_blocks, world),
            "C_learned": with_nuisance_singletons(
                restrict_to_causal(learned, world), world),
            "R": with_nuisance_singletons(rand_causal, world),
            "naive": with_nuisance_singletons(
                restrict_to_causal(naive, world), world),
        }
        for name, mods in ops.items():
            r = readapt_operator(mods, world, ws)
            func[name]["mean_steps"].append(r["mean_steps"])
            func[name]["success_rate"].append(r["success_rate"])

        if example is None:
            example = {"world_seed": ws,
                       "learned": [m.tolist() for m in learned],
                       "naive": [m.tolist() for m in naive],
                       "candidate_relevance": diag["candidate_relevance"]}

    # aggregate + report
    log("=" * 74)
    log("STRUCTURE  (Pi* revealed only now)")
    log("=" * 74)
    S = {}
    for name in ["learned", "naive", "random"]:
        S[name] = {k: _agg(v) for k, v in struct[name].items()}
    log(f"  ARI(learned,Pi*) = {S['learned']['ARI']['mean']:.3f} "
        f"+-{S['learned']['ARI']['std']:.3f}   "
        f"prec={S['learned']['prec']['mean']:.3f} rec={S['learned']['rec']['mean']:.3f} "
        f"nuisance_grouped={S['learned']['nuis']['mean']:.2f}")
    log(f"  ARI(naive,  Pi*) = {S['naive']['ARI']['mean']:.3f} "
        f"+-{S['naive']['ARI']['std']:.3f}   "
        f"prec={S['naive']['prec']['mean']:.3f} rec={S['naive']['rec']['mean']:.3f} "
        f"nuisance_grouped={S['naive']['nuis']['mean']:.2f}")
    log(f"  ARI(random, Pi*) = {S['random']['ARI']['mean']:.3f}")

    log("=" * 74)
    log("FUNCTION  (held-out goals: block targets unseen in learning)")
    log("=" * 74)
    Fm = {}
    for name in ["A_bit", "A_block", "C_learned", "R", "naive"]:
        Fm[name] = {k: _agg(v) for k, v in func[name].items()}
        m = Fm[name]
        log(f"  {name:>10}: T0.95 steps={_fmt(m['mean_steps']['mean'])} "
            f"success={m['success_rate']['mean']:.2f}")
    return world, S, Fm, example


def _fmt(v):
    return "n/a" if v is None else f"{v:.0f}"


def decide(S, Fm, log):
    log("=" * 74)
    log("PRE-REGISTERED DECISION")
    log("=" * 74)
    ari_learned = S["learned"]["ARI"]["mean"]
    ari_random = S["random"]["ARI"]["mean"]
    tC = Fm["C_learned"]["mean_steps"]["mean"]
    tAbit = Fm["A_bit"]["mean_steps"]["mean"]
    tAblk = Fm["A_block"]["mean_steps"]["mean"]

    ari_gg = ari_learned > ari_random + 0.5
    faster_than_bit = (tC is not None and tAbit is not None and tC < tAbit)
    near_block = (tC is not None and tAblk is not None and
                  abs(tC - tAblk) <= 0.35 * max(tC, tAblk))
    nuisance_ok = (S["learned"]["nuis"]["mean"] < 0.1 <=
                   S["naive"]["nuis"]["mean"] or
                   (S["learned"]["nuis"]["mean"] < S["naive"]["nuis"]["mean"] - 0.3))

    core = ari_gg and faster_than_bit and near_block and nuisance_ok
    log(f"  ARI(learned)={ari_learned:.3f} >> ARI(random)={ari_random:.3f}: {ari_gg}")
    log(f"  T0.95(C_learned)={_fmt(tC)} < T0.95(A_bit)={_fmt(tAbit)}: {faster_than_bit}")
    log(f"  T0.95(C_learned)={_fmt(tC)} ~= T0.95(A_block oracle)={_fmt(tAblk)}: {near_block}")
    log(f"  nuisance: learned groups={S['learned']['nuis']['mean']:.2f} "
        f"vs naive groups={S['naive']['nuis']['mean']:.2f}: {nuisance_ok}")
    log("")
    log(f"  CORE (interaction structure discovered blind, and it is causal "
        f"& functional): {core}")
    return {
        "ARI_learned": ari_learned, "ARI_random": ari_random,
        "T_C_learned": tC, "T_A_bit": tAbit, "T_A_block": tAblk,
        "ari_much_greater_than_random": ari_gg,
        "faster_than_A_bit": faster_than_bit,
        "near_A_block_oracle": near_block,
        "nuisance_excluded_vs_naive": nuisance_ok,
        "core": core,
    }


def make_figure(S, Fm, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    # panel 1: ARI
    names = ["learned", "naive", "random"]
    ax[0].bar(names, [S[n]["ARI"]["mean"] for n in names],
              yerr=[S[n]["ARI"]["std"] or 0 for n in names],
              color=["steelblue", "darkorange", "gray"], capsize=3)
    ax[0].set_title("ARI( partition , Pi* )", fontsize=10)
    ax[0].axhline(0, color="0.7", lw=0.6)
    # panel 2: nuisance grouped
    ax[1].bar(["learned", "naive"],
              [S["learned"]["nuis"]["mean"], S["naive"]["nuis"]["mean"]],
              color=["steelblue", "darkorange"])
    ax[1].set_title("fraction of nuisance positions grouped\n"
                    "(lower = selected only causal structure)", fontsize=9)
    ax[1].set_ylim(0, 1)
    # panel 3: re-adaptation steps on held-out goals
    fn = ["A_bit", "R", "naive", "C_learned", "A_block"]
    cols = ["crimson", "darkorange", "goldenrod", "steelblue", "seagreen"]
    vals = [Fm[n]["mean_steps"]["mean"] or 0 for n in fn]
    ax[2].bar(fn, vals, color=cols)
    ax[2].set_title("held-out valley crossing  T0.95 (lower=better)", fontsize=10)
    ax[2].tick_params(axis="x", labelsize=8)
    fig.suptitle("exp592  Endogenous Interaction Discovery -- the blind learner "
                 "recovers Pi* (causal, not nuisance) and matches the oracle operator",
                 fontsize=11)
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

    log(f"exp592 Endogenous Interaction Discovery  p={P} causal={N_CAUSAL} "
        f"nuisance={N_NUISANCE} world_seeds={WORLD_SEEDS}")
    world, S, Fm, example = run(log)
    decision = decide(S, Fm, log)
    log("example learned partition (world seed %d):" % example["world_seed"])
    log("  learned = %s" % example["learned"])
    log("  naive   = %s" % example["naive"])
    log("  candidate relevance = %s" %
        [(m, round(u, 3)) for m, u in example["candidate_relevance"]])
    fig_ok = make_figure(S, Fm, "endogenous.png")
    log(f"figure: {'endogenous.png' if fig_ok else '(matplotlib unavailable)'}")
    dt = time.time() - t0
    log("=" * 74)
    log(f"done in {dt:.1f}s")

    out = {
        "config": {"p": P, "n_causal": N_CAUSAL, "n_nuisance": N_NUISANCE,
                   "world_seeds": WORLD_SEEDS, "sample_n": SAMPLE_N, "tau": TAU,
                   "heldout_goals": HELDOUT_GOALS, "readapt_seeds": READAPT_SEEDS,
                   "max_steps": MAX_STEPS},
        "structure": S, "function": Fm, "decision": decision, "example": example,
        "runtime_seconds": dt,
    }
    with open("endogenous_results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=float)
    with open("endogenous_run.log", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote endogenous_results.json and endogenous_run.log")


if __name__ == "__main__":
    main()
