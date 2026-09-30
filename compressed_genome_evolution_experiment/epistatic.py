"""exp591 -- Epistatic Module Audit.

exp590 showed that under *additive* (Hamming) fitness the compressed
representation C beats A only by making coordinated large moves (equal
per-position efficiency).  exp591 raises the bar with a *sign-epistatic*
(Royal-Road-style) fitness: a block scores only when ALL its bits match, so a
1-bit operator cannot climb -- it must cross a fitness valley.

    F(x) = (1/B) * sum_b  1[ x[block_b] == target[block_b] ]

If the mutation operator's module boundary coincides with the epistatic block
boundary, one module move can jump 0000 -> 1111 in a single mutation and cross
the valley.  Otherwise it cannot.

Representations (each a partition of positions into *genes*; a gene stores the
FULL bit-pattern of its positions, so EVERY representation reproduces E0 exactly
-- the only thing that differs is which positions mutate together):

    A_bit      singletons                      standard 1-bit GA
    A_block    the true epistatic partition    expert-designed block operator (oracle)
    C          contiguous p-blocks             the structure the MDL-GA discovers
    R          random partition (|B| genes)    same #genes, structure-agnostic
    C_shuffle  C's block sizes, membership      reuse wiring broken, sizes preserved
               shuffled

Conditions -- whether the epistatic partition Pi aligns with C's modules:

    E_aligned   Pi = contiguous p-blocks  ( == C )
    E_shifted   Pi = contiguous p-blocks shifted by p/2   ( C straddles two blocks )
    E_random    Pi = a fixed random size-p partition       ( C misaligned )

A_block always uses Pi (the oracle); C always uses contiguous blocks, so C == A_block
only under E_aligned.  This isolates "compression that matches the interaction
structure" from "compression in general".

The decisive, pre-registered pattern: the C advantage appears in E_aligned and
VANISHES under E_shifted / E_random.

All randomness flows through an explicit numpy Generator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# epistatic partition Pi and block fitness
# ---------------------------------------------------------------------------
def contiguous_partition(L: int, p: int, shift: int = 0) -> List[np.ndarray]:
    """Blocks of p consecutive positions, cyclically shifted by `shift`."""
    assert L % p == 0
    groups = [[] for _ in range(L // p)]
    for i in range(L):
        b = ((i - shift) % L) // p
        groups[b].append(i)
    return [np.array(sorted(g)) for g in groups]


def random_partition(L: int, p: int, rng: np.random.Generator) -> List[np.ndarray]:
    assert L % p == 0
    perm = rng.permutation(L)
    return [np.sort(perm[i:i + p]) for i in range(0, L, p)]


def block_fitness(x: np.ndarray, target: np.ndarray,
                  Pi: List[np.ndarray]) -> float:
    """Fraction of epistatic blocks that match the target EXACTLY."""
    hits = 0
    for blk in Pi:
        if np.array_equal(x[blk], target[blk]):
            hits += 1
    return hits / len(Pi)


# ---------------------------------------------------------------------------
# gene-based representation: a partition into modules, each storing full bits
# ---------------------------------------------------------------------------
@dataclass
class GeneRep:
    name: str
    modules: List[np.ndarray]     # partition of range(L); module m -> positions
    L: int
    A: int = 2

    def encode_target(self, target: np.ndarray) -> List[np.ndarray]:
        """theta reproducing `target` exactly (every rep can do this)."""
        return [target[m].copy() for m in self.modules]

    def decode(self, theta: List[np.ndarray]) -> np.ndarray:
        x = np.zeros(self.L, dtype=int)
        for m, g in zip(self.modules, theta):
            x[m] = g
        return x

    def module_sizes(self) -> List[int]:
        return [len(m) for m in self.modules]


def rep_A_bit(L: int) -> GeneRep:
    return GeneRep("A_bit", [np.array([i]) for i in range(L)], L)


def rep_A_block(L: int, Pi: List[np.ndarray]) -> GeneRep:
    return GeneRep("A_block", [blk.copy() for blk in Pi], L)


def rep_C(L: int, p: int) -> GeneRep:
    return GeneRep("C", contiguous_partition(L, p), L)


def rep_R(L: int, p: int, rng: np.random.Generator) -> GeneRep:
    return GeneRep("R", random_partition(L, p, rng), L)


def rep_C_shuffle(L: int, p: int, rng: np.random.Generator) -> GeneRep:
    # same module sizes as C (all p), membership permuted
    return GeneRep("C_shuffle", random_partition(L, p, rng), L)


# ---------------------------------------------------------------------------
# neighborhood: sample single-gene resamples from the E0-optimum, score on E1
# ---------------------------------------------------------------------------
@dataclass
class EpiNeighborhood:
    F_E0: float
    F_E1_start: float
    P_useful: float
    E_dF_pos: float
    P_neutral: float
    mean_dP: float
    eta_aggregate: float      # sum(beneficial dF) / sum(beneficial dP)  [fitness per pos]
    eta_mean: float           # mean over samples of dF/dP


def epi_neighborhood(rep: GeneRep, E0: np.ndarray, E1: np.ndarray,
                     Pi: List[np.ndarray], rng: np.random.Generator,
                     n_samples: int = 3000) -> EpiNeighborhood:
    theta = rep.encode_target(E0)             # F_E0 = 1.0 for every rep
    base = rep.decode(theta)
    F_E0 = block_fitness(base, E0, Pi)
    F_E1_start = block_fitness(base, E1, Pi)
    dFs, dPs = [], []
    ben_dF, ben_dP = 0.0, 0
    n_useful = n_neutral = total = 0
    M = len(rep.modules)
    for _ in range(n_samples):
        m = int(rng.integers(0, M))
        size = len(rep.modules[m])
        g2 = rng.integers(0, rep.A, size=size)
        th2 = [g.copy() for g in theta]
        th2[m] = g2
        ph2 = rep.decode(th2)
        dP = int((ph2 != base).sum())
        if dP == 0:
            continue
        dF = block_fitness(ph2, E1, Pi) - F_E1_start
        dFs.append(dF)
        dPs.append(dP)
        total += 1
        if dF > 1e-12:
            n_useful += 1
            ben_dF += dF
            ben_dP += dP
        elif abs(dF) <= 1e-12:
            n_neutral += 1
    if total == 0:
        return EpiNeighborhood(F_E0, F_E1_start, 0, 0, 1, 0, 0, 0)
    pos = [d for d in dFs if d > 1e-12]
    return EpiNeighborhood(
        F_E0=F_E0, F_E1_start=F_E1_start,
        P_useful=n_useful / total,
        E_dF_pos=float(np.mean(pos)) if pos else 0.0,
        P_neutral=n_neutral / total,
        mean_dP=float(np.mean(dPs)),
        eta_aggregate=(ben_dF / ben_dP) if ben_dP else 0.0,
        eta_mean=float(np.mean([d / p for d, p in zip(dFs, dPs)])),
    )


# ---------------------------------------------------------------------------
# re-adaptation: valley crossing.  1+1 ES resampling one random gene per step.
# ---------------------------------------------------------------------------
def epi_readapt_steps(rep: GeneRep, E0: np.ndarray, E1: np.ndarray,
                      Pi: List[np.ndarray], threshold: float, max_steps: int,
                      rng: np.random.Generator) -> Tuple[Optional[int], float]:
    theta = rep.encode_target(E0)
    bestF = block_fitness(rep.decode(theta), E1, Pi)
    if bestF >= threshold:
        return 0, bestF
    M = len(rep.modules)
    for step in range(1, max_steps + 1):
        m = int(rng.integers(0, M))
        size = len(rep.modules[m])
        g2 = rng.integers(0, rep.A, size=size)
        th2 = [g.copy() for g in theta]
        th2[m] = g2
        f = block_fitness(rep.decode(th2), E1, Pi)
        # accept improvements; allow neutral drift (needed for A_bit to move at all)
        if f > bestF or (abs(f - bestF) < 1e-12 and rng.random() < 0.5):
            theta, bestF = th2, f
        if bestF >= threshold:
            return step, bestF
    return None, bestF


def epi_readapt_summary(rep: GeneRep, E0, E1, Pi, threshold, max_steps,
                        n_seeds, base_seed) -> dict:
    steps, plateau = [], []
    for s in range(n_seeds):
        rng = np.random.default_rng(base_seed + s)
        t, finalF = epi_readapt_steps(rep, E0, E1, Pi, threshold, max_steps, rng)
        if t is not None:
            steps.append(t)
        else:
            plateau.append(finalF)
    return {
        "success_rate": len(steps) / n_seeds,
        "mean_steps": float(np.mean(steps)) if steps else None,
        "median_steps": float(np.median(steps)) if steps else None,
        "mean_plateau_F": float(np.mean(plateau)) if plateau else None,
    }
