"""exp595 -- Dynamic Interaction Recomposition.

exp592 discovered a FIXED interaction structure blind.  exp594 showed genetic
parts can be duplicated and specialised.  exp595 makes the true interaction
structure CHANGE over time -- but by RECOMPOSING existing parts, not by inventing
everything anew -- and asks whether a learner that keeps a module library +
lineage discovers each new structure *faster* than one starting from scratch,
and whether that advantage is specifically about REUSE / SPLIT / MERGE /
RECOMBINE (vs a NOVEL structure that shares nothing).

Everything is over ATOMS -- fixed pairs of positions that always co-vary
(A=[0,1], B=[2,3], ... H=[14,15]).  A true structure Pi*_t groups atoms into
blocks (a block scores, epistatically, only when all its positions match).  The
learner never sees Pi*_t; it only observes high-fitness phenotypes and clusters
positions by co-variation (as in exp592).

The primary metric is NOT fitness but the *cost of discovering the new
structure*:

    N_0.9(t) = min { n samples : ARI(Pi_hat_t, Pi*_t) >= 0.9 }

Memory changes the clustering UNIT and the candidate library:
    reset          units = positions,  no library
    operators-only units = ATOMS (the construction primitives)
    modules-only   units = positions,  library = previous blocks (candidates)
    full           units = atoms,      library = previous blocks
    frozen         units = previous blocks, fixed (may merge, may NOT split)
    oracle         knows Pi*_t

Prediction: memory beats reset for REUSE/MERGE/SPLIT/RECOMBINE and NOT for NOVEL
(negative transfer / structural inertia).  And decomposing memory,
operators-only ~= full < modules-only for RECOMBINE -> what is remembered is the
way to BUILD modules (the atoms/operators), not the old modules themselves.

All randomness flows through explicit numpy Generators.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from endogenous import adjusted_rand_index, _labels_from_partition

L = 16
ATOMS = [np.arange(2 * i, 2 * i + 2) for i in range(8)]   # A..H, 2 positions each


# ---------------------------------------------------------------------------
# the time-varying structure sequence (atom groupings + one NOVEL that cuts atoms)
# ---------------------------------------------------------------------------
def _blocks(*groups) -> List[np.ndarray]:
    """Each group is a list of atom indices; return position-set blocks."""
    return [np.sort(np.concatenate([ATOMS[a] for a in g])) for g in groups]


def sequence() -> List[dict]:
    # atom indices: A0 B1 C2 D3 E4 F5 G6 H7
    Pi0 = _blocks([0, 1], [2, 3], [4, 5], [6, 7])          # AB CD EF GH
    Pi_reuse = _blocks([0, 1], [2, 3], [4, 5], [6, 7])     # same partition (new goals)
    Pi_merge = _blocks([0, 1, 2, 3], [4, 5], [6, 7])       # ABCD  EF GH  (merge AB,CD)
    Pi_split = _blocks([0, 1], [2, 3], [4, 5], [6, 7])     # AB CD EF GH  (split ABCD)
    Pi_recomb = _blocks([0, 2], [1, 3], [4, 5], [6, 7])    # AC BD EF GH  (recombine)
    # NOVEL: blocks that CUT atoms (pair position 0 with 2, etc.)
    novel = [np.array([0, 4, 8, 12]), np.array([1, 5, 9, 13]),
             np.array([2, 6, 10, 14]), np.array([3, 7, 11, 15])]
    return [
        {"name": "Pi0", "change": "INIT", "pi": Pi0},
        {"name": "Pi_reuse", "change": "REUSE", "pi": Pi_reuse},
        {"name": "Pi_merge", "change": "MERGE", "pi": Pi_merge},
        {"name": "Pi_split", "change": "SPLIT", "pi": Pi_split},
        {"name": "Pi_recomb", "change": "RECOMBINE", "pi": Pi_recomb},
        {"name": "Pi_novel", "change": "NOVEL", "pi": novel},
    ]


# ---------------------------------------------------------------------------
# high-fitness phenotype samples: each block's value ~ {all-0, all-1} per goal
# ---------------------------------------------------------------------------
def sample(pi: List[np.ndarray], n: int, rng: np.random.Generator) -> np.ndarray:
    X = np.zeros((n, L), dtype=int)
    for k in range(n):
        for blk in pi:
            X[k, blk] = int(rng.integers(0, 2))            # all-0 or all-1 block
    return X


# ---------------------------------------------------------------------------
# co-variation clustering over arbitrary UNITS (each unit = a set of positions)
# ---------------------------------------------------------------------------
def cluster_units(samples: np.ndarray, units: List[np.ndarray],
                  tau: float = 0.6) -> List[np.ndarray]:
    U = len(units)
    if samples.shape[0] < 2:
        return [np.array(sorted(int(i) for u in units for i in u))]  # can't tell -> all one
    vals = np.stack([samples[:, u[0]] for u in units])     # [U, n], unit rep = first pos
    with np.errstate(invalid="ignore", divide="ignore"):
        C = np.corrcoef(vals)
    C = np.nan_to_num(C, nan=0.0)
    adj = np.abs(C) > tau
    np.fill_diagonal(adj, False)
    seen = [False] * U
    comps = []
    for i in range(U):
        if seen[i]:
            continue
        stack = [i]; seen[i] = True; comp = []
        while stack:
            u = stack.pop(); comp.append(u)
            for v in range(U):
                if adj[u, v] and not seen[v]:
                    seen[v] = True; stack.append(v)
        comps.append(comp)
    # expand unit groups to position blocks
    return [np.sort(np.concatenate([units[u] for u in comp])) for comp in comps]


def _score(partition: List[np.ndarray], samples: np.ndarray) -> float:
    """High when within-block positions are identical and cross-block are not."""
    lab = _labels_from_partition(partition, L)
    within = wc = cross = cc = 0.0
    for i in range(L):
        for j in range(i + 1, L):
            agree = float(np.mean(samples[:, i] == samples[:, j]))
            if lab[i] == lab[j]:
                within += agree; wc += 1
            else:
                cross += agree; cc += 1
    return (within / wc if wc else 1.0) - (cross / cc if cc else 0.0)


# ---------------------------------------------------------------------------
# candidate libraries from a previous learned partition
# ---------------------------------------------------------------------------
def _atoms_of(blk) -> List[int]:
    s = set(int(x) for x in blk)
    return [i for i in range(len(ATOMS)) if set(ATOMS[i].tolist()) <= s]


def _from_atom_groups(groups: List[List[int]]) -> List[np.ndarray]:
    return [np.sort(np.concatenate([ATOMS[a] for a in g])) for g in groups if g]


def _prev_atom_groups(prev: List[np.ndarray]) -> List[List[int]]:
    return [_atoms_of(b) for b in prev]


def block_edit_candidates(prev) -> List[List[np.ndarray]]:
    """MODULE-level edits: keep blocks whole; merge pairs; split a block to atoms.
    Cannot move an atom from one block to another (no RECOMBINE)."""
    groups = _prev_atom_groups(prev)
    cands = [_from_atom_groups(groups)]
    B = len(groups)
    for a in range(B):
        for b in range(a + 1, B):                          # MERGE a,b
            g = [groups[k] for k in range(B) if k not in (a, b)] + [groups[a] + groups[b]]
            cands.append(_from_atom_groups(g))
    for a in range(B):                                     # SPLIT block a to atoms
        if len(groups[a]) >= 2:
            g = [groups[k] for k in range(B) if k != a] + [[x] for x in groups[a]]
            cands.append(_from_atom_groups(g))
    return cands


def atom_move_candidates(prev) -> List[List[np.ndarray]]:
    """OPERATOR-level edits: move/swap atoms between blocks (enables RECOMBINE)."""
    groups = [list(g) for g in _prev_atom_groups(prev)]
    B = len(groups)
    cands = [_from_atom_groups(groups)]
    # single-atom move a -> block b
    for a in range(B):
        for atom in list(groups[a]):
            for b in range(B):
                if b == a:
                    continue
                g = [list(x) for x in groups]
                g[a].remove(atom); g[b].append(atom)
                cands.append(_from_atom_groups(g))
    # pairwise atom swap between two blocks (this is a RECOMBINE)
    for a in range(B):
        for b in range(a + 1, B):
            for xa in groups[a]:
                for xb in groups[b]:
                    g = [list(x) for x in groups]
                    g[a].remove(xa); g[a].append(xb)
                    g[b].remove(xb); g[b].append(xa)
                    cands.append(_from_atom_groups(g))
    return cands


def _best_candidate(cands, samples, floor=None):
    best, best_s = None, -1e9
    for c in cands:
        s = _score(c, samples)
        if s > best_s:
            best, best_s = c, s
    return best, best_s


# ---------------------------------------------------------------------------
# the six learners: given samples + previous learned partition -> Pi_hat
# ---------------------------------------------------------------------------
def learn(condition: str, samples: np.ndarray, prev: Optional[List[np.ndarray]],
          pi_true: Optional[List[np.ndarray]] = None, tau: float = 0.6):
    positions = [np.array([i]) for i in range(L)]
    if condition == "oracle":
        return [b.copy() for b in pi_true]
    if condition == "reset" or not prev:
        return cluster_units(samples, positions, tau)
    fresh = cluster_units(samples, positions, tau)
    fresh_s = _score(fresh, samples)

    if condition == "modules":                             # block-level edits only
        best, bs = _best_candidate(block_edit_candidates(prev), samples)
    elif condition == "operators":                         # atom-level edits only
        best, bs = _best_candidate(atom_move_candidates(prev), samples)
    elif condition == "full":                              # both
        best, bs = _best_candidate(
            block_edit_candidates(prev) + atom_move_candidates(prev), samples)
    elif condition == "frozen":                            # keep blocks; may only MERGE
        cands = [_from_atom_groups(_prev_atom_groups(prev))]
        g = _prev_atom_groups(prev); B = len(g)
        for a in range(B):
            for b in range(a + 1, B):
                cands.append(_from_atom_groups(
                    [g[k] for k in range(B) if k not in (a, b)] + [g[a] + g[b]]))
        best, bs = _best_candidate(cands, samples)
    else:
        raise ValueError(condition)
    # a memory candidate is used only if it explains the data as well as fresh
    return best if bs >= fresh_s - 1e-9 else fresh


# ---------------------------------------------------------------------------
# discovery cost N_0.9 and adaptation time T_0.95
# ---------------------------------------------------------------------------
def discovery_cost(condition: str, pi_true: List[np.ndarray],
                   prev: Optional[List[np.ndarray]], n_grid: List[int],
                   seeds: int, base_seed: int, tau: float = 0.6) -> Optional[int]:
    """min n with mean ARI(Pi_hat, Pi*) >= 0.9."""
    for n in n_grid:
        aris = []
        for s in range(seeds):
            rng = np.random.default_rng(base_seed + s * 101 + n)
            X = sample(pi_true, n, rng)
            ph = learn(condition, X, prev, pi_true, tau)
            aris.append(adjusted_rand_index(ph, pi_true, L))
        if float(np.mean(aris)) >= 0.9:
            return n
    return None


def learned_partition(condition: str, pi_true: List[np.ndarray],
                      prev: Optional[List[np.ndarray]], n: int,
                      rng: np.random.Generator, tau: float = 0.6):
    X = sample(pi_true, n, rng)
    return learn(condition, X, prev, pi_true, tau)


# ---------------------------------------------------------------------------
# change classification & lineage recomposition (from block sets)
# ---------------------------------------------------------------------------
def _as_sets(pi):
    return [frozenset(int(x) for x in b) for b in pi]


def classify_change(prev: List[np.ndarray], cur: List[np.ndarray]) -> str:
    P, Cur = set(_as_sets(prev)), set(_as_sets(cur))
    if P == Cur:
        return "REUSE"
    # atom-respecting?
    atom_sets = [frozenset(a.tolist()) for a in ATOMS]
    def atom_respecting(pi):
        return all(any(aset <= b for b in _as_sets(pi)) for aset in atom_sets)
    if not atom_respecting(cur):
        return "NOVEL"
    # MERGE: some cur block = union of >=2 prev blocks
    if any(all((pb <= cb) for pb in P if pb & cb) and
           sum(1 for pb in P if pb <= cb) >= 2 for cb in Cur):
        return "MERGE"
    # SPLIT: some prev block = union of >=2 cur blocks
    if any(sum(1 for cb in Cur if cb <= pb) >= 2 for pb in P):
        return "SPLIT"
    return "RECOMBINE"


def recomposition_lineage(prev_blocks_ids: Dict[frozenset, int],
                          cur: List[np.ndarray], alloc) -> List[dict]:
    """Assign each current block an id + provenance from previous block ids."""
    out = []
    for b in cur:
        s = frozenset(int(x) for x in b)
        if s in prev_blocks_ids:
            out.append({"block": sorted(s), "id": prev_blocks_ids[s],
                        "op": "REUSE", "parents": [prev_blocks_ids[s]]})
        else:
            parents = [pid for ps, pid in prev_blocks_ids.items()
                       if ps & s]                            # overlapping ancestors
            op = ("MERGE" if len(parents) >= 2 and
                  all(ps <= s for ps in prev_blocks_ids if prev_blocks_ids[ps] in parents)
                  else "RECOMBINE/SPLIT" if parents else "NOVEL")
            out.append({"block": sorted(s), "id": alloc(), "op": op,
                        "parents": parents})
    return out
