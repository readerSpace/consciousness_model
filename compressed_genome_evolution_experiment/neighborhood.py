"""exp590 -- mutation-neighborhood & search-efficiency audit.

Why did exp588 show re-adaptation of C = 12 generations vs A = 36.8?  Two rival
explanations:

    (a) C simply has fewer free parameters (a smaller search space), or
    (b) C's parameters are *aligned with the environment's structure*, so one
        mutation moves a whole co-adapted block of phenotype positions the right
        way at once.

To separate them we model each representation as a fixed decoder over a small
parameter vector theta, and vary only the *grouping* of phenotype positions:

    decode(theta)[i] = theta[ group_of[i] ]        (one shared symbol per group)

    A          group_of = [0,1,...,L-1]        K = L      (no compression)
    C          group_of = [i % p ...]          K = p      (the period the MDL-GA
                                                            actually discovers)
    R          group_of = random labels        K = p      (same size, structure-
                                                            agnostic reduced rep)
    C_shuffle  group_of = random *balanced*     K = p      (same group sizes =
               partition (size L/p each)                    macro length / CALL
                                                            count, membership
                                                            permuted -> reuse
                                                            wiring broken)

A mutation flips one theta bit; for binary alphabet that changes every position
in the group.  We record, w.r.t. the *new* environment E1 (a related member of
the same periodic family):

    dF_E0, dF_E1            fitness change
    d_P                    phenotypic Hamming distance = size of the flipped group
    P_beneficial = P(dF_E1 > 0)
    E[dF_E1 | dF_E1 > 0]
    P_neutral   = P(dF_E1 = 0)
    P_useful    = P(F_E1(g') > F_E1(g))
    eta_mu      = (matches gained on E1) / (positions changed)   -- per-position
                  usefulness of a mutation; separates "big but wasteful moves"
                  (misaligned) from "big and aligned moves".

and the re-adaptation cost T_0.95 = mutations (1+1 ES steps) to reach F_E1 >= 0.95.

Everything is decodable and uses an explicit numpy Generator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# representation = a grouping of the L positions into K shared parameters
# ---------------------------------------------------------------------------
@dataclass
class Representation:
    name: str
    group_of: np.ndarray      # int array length L, values in [0, K)
    K: int
    A: int
    L: int

    def decode(self, theta: np.ndarray) -> np.ndarray:
        return theta[self.group_of]

    def group_sizes(self) -> List[int]:
        return [int((self.group_of == g).sum()) for g in range(self.K)]


def rep_A(A: int, L: int) -> Representation:
    return Representation("A", np.arange(L), L, A, L)


def rep_C(A: int, L: int, p: int) -> Representation:
    return Representation("C", np.arange(L) % p, p, A, L)


def rep_R(A: int, L: int, p: int, rng: np.random.Generator) -> Representation:
    g = rng.integers(0, p, size=L)
    # ensure every group is used (else effective K < p); resample offenders
    for _ in range(20):
        if len(np.unique(g)) == p:
            break
        g = rng.integers(0, p, size=L)
    return Representation("R", g, p, A, L)


def rep_C_shuffle(A: int, L: int, p: int, rng: np.random.Generator) -> Representation:
    # balanced partition with the SAME group sizes as C (each L/p), membership
    # permuted so the periodic alignment is destroyed but macro length / call
    # count are preserved.
    assert L % p == 0, "C_shuffle needs L divisible by p"
    labels = np.repeat(np.arange(p), L // p)
    rng.shuffle(labels)
    return Representation("C_shuffle", labels, p, A, L)


# ---------------------------------------------------------------------------
# best theta for a target under a grouping (per-group majority vote)
# ---------------------------------------------------------------------------
def best_theta(rep: Representation, target: np.ndarray) -> np.ndarray:
    theta = np.zeros(rep.K, dtype=int)
    for g in range(rep.K):
        idx = (rep.group_of == g)
        vals = target[idx]
        if len(vals) == 0:
            theta[g] = 0
            continue
        counts = np.bincount(vals, minlength=rep.A)
        theta[g] = int(np.argmax(counts))
    return theta


def fitness(pheno: np.ndarray, target: np.ndarray) -> float:
    return float((pheno == target).mean())


# ---------------------------------------------------------------------------
# 1-step neighborhood metrics (exhaustive over the K.(A-1) single-symbol edits)
# ---------------------------------------------------------------------------
@dataclass
class Neighborhood:
    P_beneficial: float
    E_dF_pos: float
    P_neutral: float
    P_useful: float
    mean_dP: float
    eta_mean: float           # mean over all edits of (matches gained)/(positions changed)
    eta_beneficial: float     # same, restricted to beneficial edits
    F_E0: float
    F_E1: float


def neighborhood(rep: Representation, E0: np.ndarray, E1: np.ndarray) -> Neighborhood:
    theta = best_theta(rep, E0)
    base = rep.decode(theta)
    F_E0 = fitness(base, E0)
    F_E1 = fitness(base, E1)
    dFs, dPs, etas = [], [], []
    n_benef = 0
    n_neutral = 0
    n_useful = 0
    eta_benef_num = 0
    eta_benef_den = 0
    total = 0
    for g in range(rep.K):
        for s in range(rep.A):
            if s == theta[g]:
                continue
            th2 = theta.copy()
            th2[g] = s
            ph2 = rep.decode(th2)
            dP = int((ph2 != base).sum())
            if dP == 0:
                continue
            matches_gained = int((ph2 == E1).sum() - (base == E1).sum())
            dF = matches_gained / rep.L
            dFs.append(dF)
            dPs.append(dP)
            etas.append(matches_gained / dP)
            total += 1
            if dF > 1e-12:
                n_benef += 1
                n_useful += 1
                eta_benef_num += matches_gained
                eta_benef_den += dP
            elif abs(dF) <= 1e-12:
                n_neutral += 1
    if total == 0:
        return Neighborhood(0, 0, 1, 0, 0, 0, 0, F_E0, F_E1)
    pos = [d for d in dFs if d > 1e-12]
    return Neighborhood(
        P_beneficial=n_benef / total,
        E_dF_pos=float(np.mean(pos)) if pos else 0.0,
        P_neutral=n_neutral / total,
        P_useful=n_useful / total,
        mean_dP=float(np.mean(dPs)),
        eta_mean=float(np.mean(etas)),
        eta_beneficial=(eta_benef_num / eta_benef_den) if eta_benef_den else 0.0,
        F_E0=F_E0, F_E1=F_E1,
    )


# ---------------------------------------------------------------------------
# re-adaptation: 1+1 ES on theta from the E0-optimum, measuring steps to E1
# ---------------------------------------------------------------------------
def readapt_steps(rep: Representation, E0: np.ndarray, E1: np.ndarray,
                  threshold: float, max_steps: int,
                  rng: np.random.Generator) -> Optional[int]:
    theta = best_theta(rep, E0)
    base = rep.decode(theta)
    bestF = fitness(base, E1)
    if bestF >= threshold:
        return 0
    for step in range(1, max_steps + 1):
        g = int(rng.integers(0, rep.K))
        s = int(rng.integers(0, rep.A))
        if s == theta[g]:
            continue
        th2 = theta.copy()
        th2[g] = s
        f = fitness(rep.decode(th2), E1)
        if f > bestF or (abs(f - bestF) < 1e-12 and rng.random() < 0.3):
            theta = th2
            bestF = f
        if bestF >= threshold:
            return step
    return None


def readapt_summary(rep: Representation, E0: np.ndarray, E1: np.ndarray,
                    threshold: float, max_steps: int, n_seeds: int,
                    base_seed: int) -> dict:
    steps = []
    plateau = []
    for s in range(n_seeds):
        rng = np.random.default_rng(base_seed + s)
        t = readapt_steps(rep, E0, E1, threshold, max_steps, rng)
        if t is not None:
            steps.append(t)
        else:
            # record plateau fitness reached
            rng2 = np.random.default_rng(base_seed + s)
            plateau.append(_plateau_fitness(rep, E0, E1, max_steps, rng2))
    return {
        "success_rate": len(steps) / n_seeds,
        "mean_steps": float(np.mean(steps)) if steps else None,
        "median_steps": float(np.median(steps)) if steps else None,
        "mean_plateau_F": float(np.mean(plateau)) if plateau else None,
    }


def _plateau_fitness(rep, E0, E1, max_steps, rng):
    theta = best_theta(rep, E0)
    bestF = fitness(rep.decode(theta), E1)
    for _ in range(max_steps):
        g = int(rng.integers(0, rep.K))
        s = int(rng.integers(0, rep.A))
        th2 = theta.copy(); th2[g] = s
        f = fitness(rep.decode(th2), E1)
        if f > bestF or (abs(f - bestF) < 1e-12 and rng.random() < 0.3):
            theta, bestF = th2, f
    return bestF
