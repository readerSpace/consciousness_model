"""Environments (targets) for Compressed Genome Evolution (exp588).

An environment is a target symbol sequence of length L over an alphabet of size
A.  Fitness of a phenotype is the fraction of positions that match the target.

We deliberately build *structured* targets out of exactly the regularities the
genome DSL can express (periodic repetition, mirror symmetry, nested repeats,
concatenation of a small set of blocks), then corrupt a few positions with
fixed "nuisance" symbols the optimum must pay literal budget for.  Against these,
compression can in principle find a short rule.  The *unstructured* target is
uniform noise: there is no rule to find, so compression must buy nothing -- the
null control (cf. the phylogenetic-compression project's "structureless null
does not shrink by even one bit").

The `shuffled` control permutes a structured target's positions, destroying the
structure while preserving the symbol histogram; it should behave like noise.

All randomness flows through an explicit numpy Generator (never Python's salted
hash()).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class Environment:
    name: str
    kind: str            # 'structured' | 'unstructured' | 'shuffled'
    target: List[int]
    A: int
    L: int
    # a compact human description of the generating rule (documentation only):
    rule: str = ""
    # optimal reachable description length is unknown in general; left None.


def fitness(phenotype: List[int], env: Environment) -> float:
    t = env.target
    L = env.L
    m = 0
    for i in range(L):
        if phenotype[i] == t[i]:
            m += 1
    return m / L


# ---------------------------------------------------------------------------
# structured target: nested repeats + mirror + a few nuisance positions
# ---------------------------------------------------------------------------
def make_structured(rng: np.random.Generator, A: int, L: int,
                    motif_len: int = 3, n_nuisance: int = 3,
                    name: str = "structured") -> Environment:
    """Target = tile(motif) over the first 2/3, mirror-block over the rest,
    then overwrite `n_nuisance` random positions with fixed noise symbols.

    This mixes periodic repetition (tile), symmetry (mirror), and irreducible
    literal content (nuisance) in one target."""
    motif = rng.integers(0, A, size=motif_len).tolist()
    head_len = (2 * L) // 3
    # periodic head
    head = [motif[i % motif_len] for i in range(head_len)]
    # mirrored tail built from a second small block
    block = rng.integers(0, A, size=max(2, motif_len)).tolist()
    tail = []
    while len(tail) < (L - head_len):
        tail.extend(block + block[::-1])
    tail = tail[: L - head_len]
    seq = head + tail
    # nuisance corruption
    pos = rng.choice(L, size=min(n_nuisance, L), replace=False)
    for p in pos:
        seq[int(p)] = int(rng.integers(0, A))
    return Environment(
        name=name, kind="structured", target=seq, A=A, L=L,
        rule=f"tile(motif{motif}) ++ mirror(block{block}) with {n_nuisance} nuisance",
    )


def make_periodic(rng: np.random.Generator, A: int, L: int,
                  period: int = 4, name: str = "periodic") -> Environment:
    motif = rng.integers(0, A, size=period).tolist()
    seq = [motif[i % period] for i in range(L)]
    return Environment(name=name, kind="structured", target=seq, A=A, L=L,
                       rule=f"period-{period} tile {motif}")


# ---------------------------------------------------------------------------
# unstructured target: uniform noise (the null)
# ---------------------------------------------------------------------------
def make_unstructured(rng: np.random.Generator, A: int, L: int,
                      name: str = "unstructured") -> Environment:
    seq = rng.integers(0, A, size=L).tolist()
    return Environment(name=name, kind="unstructured", target=seq, A=A, L=L,
                       rule="uniform i.i.d. noise (no rule)")


# ---------------------------------------------------------------------------
# shuffled control: structured histogram, destroyed structure
# ---------------------------------------------------------------------------
def make_shuffled(rng: np.random.Generator, base: Environment,
                  name: str = "shuffled") -> Environment:
    seq = list(base.target)
    perm = rng.permutation(base.L)
    shuffled = [seq[i] for i in perm]
    return Environment(name=name, kind="shuffled", target=shuffled,
                       A=base.A, L=base.L,
                       rule=f"positions of {base.name} permuted (histogram kept)")


# ---------------------------------------------------------------------------
# a related family for re-adaptation / non-stationary (group D) experiments
# ---------------------------------------------------------------------------
def related_periodic_family(rng: np.random.Generator, A: int, L: int,
                            period: int, k: int) -> List[Environment]:
    """k environments sharing the *rule* (same period) but different motifs.

    A genome that captured "period-p tiling" only needs to relearn p symbols to
    jump between family members; a genome that memorised L symbols must relearn
    everything.  This is the concrete substrate for the re-adaptation test."""
    fam = []
    for j in range(k):
        motif = rng.integers(0, A, size=period).tolist()
        seq = [motif[i % period] for i in range(L)]
        fam.append(Environment(name=f"periodic{period}_v{j}", kind="structured",
                               target=seq, A=A, L=L,
                               rule=f"period-{period} tile {motif}"))
    return fam
