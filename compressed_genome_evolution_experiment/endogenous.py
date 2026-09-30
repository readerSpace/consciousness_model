"""exp592 -- Endogenous Interaction Discovery.

exp591 handed the true interaction partition Pi* to the experiment and only
*manipulated* C's alignment with it.  exp592 removes that: the learner sees ONLY
raw phenotypes and a black-box fitness, never Pi*.  We ask whether a compression
/ fitness-driven learner *discovers* the interaction structure by itself, and
whether the discovered partition then works as an advantageous mutation operator
on unseen goals.

Two learners, both blind to Pi*:

  naive_compressor(sample)
      clusters positions by co-variation in a set of high-fitness phenotypes.
      It groups ANYTHING that co-varies -- including a fitness-IRRELEVANT
      nuisance block with its own strong structure.  This is the foil.

  fitness_driven(sample, fitness_sampler)
      uses the SAME co-variation to *propose* candidate modules, then keeps a
      candidate only if perturbing it actually moves the (black-box) fitness.
      Compressibility proposes; fitness disposes.  -> selects the causal blocks
      and drops the nuisance = a minimal *sufficient* evolutionary representation.

Leakage control: the learners take only (sample) / (sample, fitness_sampler).
`fitness_sampler` yields (target, fitness_callable) for random goals; the
callable returns a scalar.  Pi* is sealed inside the callable and is revealed to
NOTHING until the post-hoc scoring in the experiment.

Goal distribution (matches the attachment's hold-out example): during learning
each block's target is drawn from the two "aligned" motifs {0000, 1111}; the
re-adaptation test then uses UNSEEN block values (arbitrary p-bit patterns), so a
learner that merely memorised {0000,1111} cannot pass -- only one that captured
the block *partition* generalises.

All randomness flows through explicit numpy Generators.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# environment: causal blocks (Pi*) + fitness-irrelevant nuisance blocks
# ---------------------------------------------------------------------------
@dataclass
class World:
    L: int
    causal_blocks: List[np.ndarray]      # Pi* on the causal positions
    nuisance_blocks: List[np.ndarray]    # strongly structured but fitness-irrelevant
    p: int

    def causal_positions(self) -> np.ndarray:
        return np.sort(np.concatenate(self.causal_blocks)) if self.causal_blocks \
            else np.array([], dtype=int)

    def true_partition(self) -> List[np.ndarray]:
        """Pi* on ALL positions: causal blocks are real modules; every nuisance
        position is its own singleton (it interacts with nothing)."""
        parts = [b.copy() for b in self.causal_blocks]
        causal = set(int(i) for b in self.causal_blocks for i in b)
        for i in range(self.L):
            if i not in causal:
                parts.append(np.array([i]))
        return parts


def make_world(p=4, n_causal=3, n_nuisance=2) -> World:
    L = (n_causal + n_nuisance) * p
    causal = [np.arange(b * p, (b + 1) * p) for b in range(n_causal)]
    nuisance = [np.arange((n_causal + b) * p, (n_causal + b + 1) * p)
                for b in range(n_nuisance)]
    return World(L=L, causal_blocks=causal, nuisance_blocks=nuisance, p=p)


# aligned motifs used during LEARNING (the hold-out example's {0000, 1111})
def _aligned_motifs(p: int) -> List[np.ndarray]:
    return [np.zeros(p, dtype=int), np.ones(p, dtype=int)]


def fitness_callable(world: World, causal_targets: List[np.ndarray]) -> Callable:
    """Return a *black-box* fitness for one goal: fraction of causal blocks that
    match exactly.  Nuisance positions are ignored.  Pi* lives only inside here."""
    blocks = world.causal_blocks
    def F(x: np.ndarray) -> float:
        hits = 0
        for blk, tgt in zip(blocks, causal_targets):
            if np.array_equal(x[blk], tgt):
                hits += 1
        return hits / len(blocks)
    return F


def goal_sampler(world: World, rng: np.random.Generator,
                 motif_set: str = "aligned") -> Callable:
    """Returns a function drawing a random goal -> (causal_targets, F).

    motif_set='aligned' draws each causal block target from {0000,1111}
    (learning); 'arbitrary' draws any p-bit pattern (held-out test)."""
    motifs = _aligned_motifs(world.p)
    def draw():
        if motif_set == "aligned":
            ct = [motifs[int(rng.integers(0, len(motifs)))].copy()
                  for _ in world.causal_blocks]
        else:
            ct = [rng.integers(0, 2, world.p) for _ in world.causal_blocks]
        return ct, fitness_callable(world, ct)
    return draw


# ---------------------------------------------------------------------------
# a high-fitness phenotype sample (the observational data the learner sees)
# ---------------------------------------------------------------------------
def high_fitness_sample(world: World, n: int, rng: np.random.Generator,
                        noise: float = 0.0) -> np.ndarray:
    """n phenotypes that score well across random aligned goals.  Causal blocks
    carry that goal's target (from {0000,1111}); nuisance blocks carry their OWN
    strongly-structured value (also {0000,1111} per nuisance block) -- structured
    and co-varying, but fitness-irrelevant (the temptation for a naive compressor)."""
    motifs = _aligned_motifs(world.p)
    X = np.zeros((n, world.L), dtype=int)
    for k in range(n):
        for blk in world.causal_blocks:
            X[k, blk] = motifs[int(rng.integers(0, len(motifs)))]
        for blk in world.nuisance_blocks:
            X[k, blk] = motifs[int(rng.integers(0, len(motifs)))]
        if noise > 0:
            flip = rng.random(world.L) < noise
            X[k, flip] ^= 1
    return X


# ---------------------------------------------------------------------------
# co-variation clustering (proposes candidate modules) -- blind
# ---------------------------------------------------------------------------
def covariation_partition(sample: np.ndarray, tau: float = 0.6) -> List[np.ndarray]:
    """Group positions into connected components of |correlation| > tau across
    the sample.  Uses only the phenotype sample (no fitness, no Pi*)."""
    L = sample.shape[1]
    with np.errstate(invalid="ignore", divide="ignore"):
        C = np.corrcoef(sample.T)
    C = np.nan_to_num(C, nan=0.0)         # constant columns -> 0
    adj = np.abs(C) > tau
    np.fill_diagonal(adj, False)
    # connected components
    seen = [False] * L
    comps = []
    for i in range(L):
        if seen[i]:
            continue
        stack = [i]
        comp = []
        seen[i] = True
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in range(L):
                if adj[u, v] and not seen[v]:
                    seen[v] = True
                    stack.append(v)
        comps.append(np.array(sorted(comp)))
    return comps


# ---------------------------------------------------------------------------
# fitness-relevance of a candidate module (blind: only the black-box fitness)
# ---------------------------------------------------------------------------
def module_fitness_relevance(module: np.ndarray, L: int, draw_goal: Callable,
                             rng: np.random.Generator,
                             n_goals: int = 40, n_ref: int = 6,
                             max_enum: int = 8) -> float:
    """Expected fitness gain achievable by optimally setting ONLY this module,
    averaged over random goals and reference states.  Causal blocks -> >0
    (can complete their block); nuisance / partial modules -> ~0.

    `L` is the phenotype length (known from the sample, not Pi*).  Never sees
    Pi*; only calls the black-box F drawn from `draw_goal`."""
    s = len(module)
    if s <= max_enum:
        values = [np.array([(v >> b) & 1 for b in range(s)]) for v in range(2 ** s)]
    else:
        values = [rng.integers(0, 2, s) for _ in range(2 ** max_enum)]
    gains = []
    for _ in range(n_goals):
        _, F = draw_goal()
        for _ in range(n_ref):
            ref = rng.integers(0, 2, L)
            base = F(ref)
            best = base
            for v in values:
                x = ref.copy()
                x[module] = v
                f = F(x)
                if f > best:
                    best = f
            gains.append(best - base)
    return float(np.mean(gains)) if gains else 0.0


# ---------------------------------------------------------------------------
# the two learners
# ---------------------------------------------------------------------------
def learn_naive(sample: np.ndarray, tau: float = 0.6) -> List[np.ndarray]:
    """Cluster by co-variation and KEEP every group (tempted by nuisance)."""
    return covariation_partition(sample, tau)


def learn_fitness_driven(sample: np.ndarray, draw_goal: Callable,
                         rng: np.random.Generator, tau: float = 0.6,
                         relevance_eps: float = 1e-6) -> Tuple[List[np.ndarray], dict]:
    """Propose candidate modules by co-variation, keep only fitness-relevant ones;
    dissolve the rest to singletons.  Blind to Pi*."""
    L = sample.shape[1]
    cand = covariation_partition(sample, tau)
    modules = []
    diagnostics = {"candidate_relevance": []}
    for m in cand:
        if len(m) == 1:
            modules.append(m)
            continue
        U = module_fitness_relevance(m, L, draw_goal, rng)
        diagnostics["candidate_relevance"].append((m.tolist(), float(U)))
        if U > relevance_eps:
            modules.append(m)                      # fitness-relevant -> keep
        else:
            for i in m:                            # nuisance -> singletons
                modules.append(np.array([int(i)]))
    return modules, diagnostics


# ---------------------------------------------------------------------------
# partition comparison: Adjusted Rand Index + pairwise precision/recall
# ---------------------------------------------------------------------------
def _labels_from_partition(part: List[np.ndarray], L: int) -> np.ndarray:
    lab = -np.ones(L, dtype=int)
    for g, m in enumerate(part):
        for i in m:
            lab[int(i)] = g
    # any unassigned -> own label
    nxt = len(part)
    for i in range(L):
        if lab[i] < 0:
            lab[i] = nxt
            nxt += 1
    return lab


def adjusted_rand_index(partA: List[np.ndarray], partB: List[np.ndarray],
                        L: int) -> float:
    a = _labels_from_partition(partA, L)
    b = _labels_from_partition(partB, L)
    # contingency
    ua = np.unique(a); ub = np.unique(b)
    cont = np.zeros((len(ua), len(ub)), dtype=np.int64)
    ia = {v: k for k, v in enumerate(ua)}
    ib = {v: k for k, v in enumerate(ub)}
    for x, y in zip(a, b):
        cont[ia[x], ib[y]] += 1
    def comb2(n):
        return n * (n - 1) // 2
    sum_comb = sum(comb2(int(v)) for v in cont.flatten())
    sum_a = sum(comb2(int(v)) for v in cont.sum(axis=1))
    sum_b = sum(comb2(int(v)) for v in cont.sum(axis=0))
    n = comb2(L)
    if n == 0:
        return 1.0
    expected = sum_a * sum_b / n
    max_index = 0.5 * (sum_a + sum_b)
    if max_index - expected == 0:
        return 1.0
    return (sum_comb - expected) / (max_index - expected)


def pairwise_precision_recall(pred: List[np.ndarray], true: List[np.ndarray],
                              L: int) -> Tuple[float, float]:
    """Precision/recall over *same-module* pairs (only pairs that are together in
    `true` count for recall; pairs together in `pred` count for precision)."""
    a = _labels_from_partition(pred, L)
    b = _labels_from_partition(true, L)
    tp = fp = fn = 0
    for i in range(L):
        for j in range(i + 1, L):
            same_pred = a[i] == a[j]
            same_true = b[i] == b[j]
            if same_pred and same_true:
                tp += 1
            elif same_pred and not same_true:
                fp += 1
            elif not same_pred and same_true:
                fn += 1
    prec = tp / (tp + fp) if (tp + fp) else 1.0
    rec = tp / (tp + fn) if (tp + fn) else 1.0
    return prec, rec


def random_partition_like(true: List[np.ndarray], L: int,
                          rng: np.random.Generator) -> List[np.ndarray]:
    """A random partition with the same block-size multiset as `true`."""
    sizes = [len(m) for m in true]
    perm = rng.permutation(L)
    out = []
    k = 0
    for s in sizes:
        out.append(np.sort(perm[k:k + s]))
        k += s
    return out
