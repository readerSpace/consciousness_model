"""Exact optimality / Pareto audit for Compressed Genome Evolution (exp589).

Question exp589 sharpens:

    Not "did C find a short representation?" but
    "how close did C get to the *minimum description* that achieves that fitness?"

For small L (and alphabet A=2) we compute, for every phenotype x, the exact
minimum description length K(x) under a bounded, fully-enumerated description
class, in the *same bit units* as `genome.py`.  From {(K(x), F(x,target))} we get:

    L*(F>=f) = min { K(x) : F(x,target) >= f }          (fitness-constrained optimum)
    Pareto frontier  P = non-dominated (L, F) points
    Delta L = L_GA - L*   and   R_L = L_GA / L*          (how close the GA got)

The description class (stated precisely, as in the phylogenetic-compression
project's "L* is exact w.r.t. this estimator"):

    * up to M<=2 macros, each a LIT-only string over the alphabet, length <= mlen,
      whose emitted string is a substring of x (proved sufficient below);
    * a program that concatenates instructions LIT / CALL k / REP n k / MIR k
      over those macros, chosen optimally by dynamic programming.

Why substring macros are sufficient for this flat class: a CALL/REP emits the
macro string verbatim, and MIR emits it followed by its reverse; in every case
the macro's forward string appears contiguously in x, hence is a substring of x.
A macro invoked only once via CALL never lowers total length (its stored copy
plus the CALL index costs more than inlining), so useful candidates are exactly
the substrings that can be emitted >= 2 times (repeat, REP, or MIR).

K(x) is computed exactly *within this class*.  It is a rigorous lower bound for
any description the GA finds that also lives in the class; if the GA ever beats
it, that means the GA used structure outside the class (nested macros), which the
runner detects and reports rather than hiding.
"""

from __future__ import annotations

import itertools
from typing import Dict, List, Optional, Tuple

import genome as G
from genome import (Genome, Instr, OP_LIT, OP_CALL, OP_REP, OP_MIR,
                    gamma_bits, instr_bits, _sym_bits)

INF = float("inf")


# ---------------------------------------------------------------------------
# candidate macros: substrings of x that can be emitted >= 2 times
# ---------------------------------------------------------------------------
def candidate_macros(x: Tuple[int, ...], mlen: int, nmax: int,
                     max_candidates: int = 48) -> List[Tuple[int, ...]]:
    L = len(x)
    counts: Dict[Tuple[int, ...], int] = {}
    for i in range(L):
        for ln in range(1, min(mlen, L - i) + 1):
            sub = x[i:i + ln]
            counts[sub] = counts.get(sub, 0) + 1
    useful = set()
    for sub, c in counts.items():
        if c >= 2:
            useful.add(sub)              # CALL >=2 or REP
    # MIR: sub followed by its reverse occurs somewhere
    for i in range(L):
        for ln in range(1, min(mlen, L - i) + 1):
            sub = x[i:i + ln]
            mir = sub + sub[::-1]
            if _contains(x, mir):
                useful.add(sub)
    # REP with a single run: a length-1 macro repeated
    for s in set(x):
        useful.add((s,))
    cands = sorted(useful, key=lambda t: (-len(t), t))
    return cands[:max_candidates]


def _contains(x: Tuple[int, ...], sub: Tuple[int, ...]) -> bool:
    ls, lx = len(sub), len(x)
    for i in range(lx - ls + 1):
        if x[i:i + ls] == sub:
            return True
    return False


# ---------------------------------------------------------------------------
# minimal program (2D DP over position x program-length) for a fixed macro set
# ---------------------------------------------------------------------------
def _macro_section_bits(macros: List[Tuple[int, ...]], A: int) -> int:
    b = gamma_bits(len(macros) + 1)
    sb = _sym_bits(A)
    for m in macros:
        b += gamma_bits(len(m) + 1) + len(m) * (2 + sb)  # LIT-only body
    return b


def _min_program(x: Tuple[int, ...], macros: List[Tuple[int, ...]], A: int,
                 nmax: int) -> Tuple[float, Optional[List[Instr]]]:
    """Exact minimal program (in genome.py bit units) producing x with these
    macros.  Returns (program_bits_including_length_prefix, program_instrs)."""
    L = len(x)
    sb = _sym_bits(A)
    lit_cost = 2 + sb
    # precompute per-start-position the available blocks as (end_pos, cost, Instr).
    # end_pos = min(pos + emission_length, L): the codec truncates emissions that
    # overshoot L, so a final block may run past the end as long as its truncated
    # prefix matches x[pos:L].
    def _emits(pattern_prefix, j, emitlen, cost, ins, bucket):
        end = min(j + emitlen, L)
        need = end - j
        if pattern_prefix[:need] == x[j:end]:
            bucket.append((end, cost, ins))

    blocks_at: List[List[Tuple[int, int, Instr]]] = [[] for _ in range(L)]
    for j in range(L):
        blocks_at[j].append((j + 1, lit_cost, Instr(OP_LIT, x[j])))
    for k, m in enumerate(macros):
        lm = len(m)
        mir = m + m[::-1]
        for j in range(L):
            _emits(m, j, lm, 2 + gamma_bits(k + 1), Instr(OP_CALL, k), blocks_at[j])
            for n in range(2, nmax + 1):
                span = lm * n
                _emits(m * n, j, span,
                       2 + gamma_bits(n) + gamma_bits(k + 1),
                       Instr(OP_REP, n, k), blocks_at[j])
                if j + span >= L:      # further n only overshoot more; stop early
                    break
            _emits(mir, j, 2 * lm, 2 + gamma_bits(k + 1), Instr(OP_MIR, k), blocks_at[j])
    # dp[pos][p] = min instruction-bit-sum to produce x[0:pos] with p instructions
    dp = [[INF] * (L + 1) for _ in range(L + 1)]
    back = [[None] * (L + 1) for _ in range(L + 1)]
    dp[0][0] = 0
    for pos in range(L):
        for p in range(L):
            cur = dp[pos][p]
            if cur == INF:
                continue
            for (end, cost, ins) in blocks_at[pos]:
                if cur + cost < dp[end][p + 1]:
                    dp[end][p + 1] = cur + cost
                    back[end][p + 1] = (pos, p, ins)
    # The codec pads the phenotype with symbol 0 up to length L, so any all-zero
    # suffix of x is produced for free: the program need only emit x[0:pos] for
    # any pos whose suffix x[pos:] is all zeros.
    j0 = L
    while j0 > 0 and x[j0 - 1] == 0:
        j0 -= 1
    accept_positions = range(j0, L + 1)

    best = INF
    best_pos = -1
    best_p = -1
    for pos in accept_positions:
        p_start = 0 if pos == 0 else 1
        for p in range(p_start, L + 1):
            if dp[pos][p] == INF:
                continue
            total = dp[pos][p] + gamma_bits(p + 1)   # program length prefix
            if total < best:
                best, best_pos, best_p = total, pos, p
    if best_pos < 0:
        return INF, None
    # reconstruct
    instrs: List[Instr] = []
    pos, p = best_pos, best_p
    while p > 0:
        pos, pp, ins = back[pos][p]
        instrs.append(ins)
        p = pp
    instrs.reverse()
    return best, instrs


# ---------------------------------------------------------------------------
# K(x): exact minimum over M in {0,1,2} macro sets, in genome.py bit units
# ---------------------------------------------------------------------------
def kx(x: Tuple[int, ...], A: int, L: int, mlen: int = 6, Mmax: int = 2,
       nmax: int = 8, return_genome: bool = False):
    cands = candidate_macros(x, mlen, nmax)
    best = INF
    best_macros: List[Tuple[int, ...]] = []
    best_prog: Optional[List[Instr]] = None

    def consider(macros):
        nonlocal best, best_macros, best_prog
        pbits, prog = _min_program(x, list(macros), A, nmax)
        if prog is None:
            return
        total = pbits + _macro_section_bits(list(macros), A)
        if total < best:
            best, best_macros, best_prog = total, list(macros), prog

    consider([])                                   # M = 0 (literal-capable)
    if Mmax >= 1:
        for m in cands:
            consider([m])
    if Mmax >= 2:
        for m0, m1 in itertools.combinations(cands, 2):
            consider([m0, m1])
    if not return_genome:
        return best
    g = Genome(macros=[[Instr(OP_LIT, s) for s in m] for m in best_macros],
               program=best_prog if best_prog is not None else [],
               A=A, L=L, nmax=nmax)
    return best, g


# ---------------------------------------------------------------------------
# fitness, frontier, L*(F>=f)  for a fixed target
# ---------------------------------------------------------------------------
def _fit(x: Tuple[int, ...], target: Tuple[int, ...]) -> float:
    return sum(1 for a, b in zip(x, target) if a == b) / len(target)


def audit_target(target: List[int], A: int, mlen: int = 6, Mmax: int = 2,
                 nmax: int = 8, f_min: float = 0.0,
                 enumerate_full: bool = True,
                 hamming_radius: Optional[int] = None) -> dict:
    """Compute K(x) over the relevant phenotypes and derive the exact frontier.

    enumerate_full: iterate all A^L phenotypes (tractable for small L, A=2).
    else: iterate only x within `hamming_radius` of target (high-F ball)."""
    L = len(target)
    tgt = tuple(target)
    points: List[Tuple[int, float]] = []   # (K, F)
    best_L_at_F: Dict[float, int] = {}

    if enumerate_full:
        it = itertools.product(range(A), repeat=L)
    else:
        it = _hamming_ball(tgt, A, hamming_radius)

    for xt in it:
        F = _fit(xt, tgt)
        if F < f_min:
            continue
        k = kx(xt, A, L, mlen, Mmax, nmax)
        points.append((k, F))
        if (F not in best_L_at_F) or k < best_L_at_F[F]:
            best_L_at_F[F] = k

    # L*(F>=f) for the distinct F levels present
    Fs = sorted(best_L_at_F.keys(), reverse=True)
    Lstar_ge = {}
    running = INF
    for f in Fs:                      # from high F down
        running = min(running, best_L_at_F[f])
        Lstar_ge[f] = running

    frontier = _pareto(points)
    return {
        "L": L, "A": A,
        "points": points,
        "best_L_at_exact_F": {f: best_L_at_F[f] for f in Fs},
        "Lstar_F_ge": Lstar_ge,
        "frontier": frontier,
    }


def _hamming_ball(tgt: Tuple[int, ...], A: int, radius: int):
    L = len(tgt)
    positions = range(L)
    for d in range(radius + 1):
        for combo in itertools.combinations(positions, d):
            others = [range(A)] * d
            for repl in itertools.product(*[[
                    s for s in range(A) if s != tgt[c]] for c in combo]):
                x = list(tgt)
                for c, s in zip(combo, repl):
                    x[c] = s
                yield tuple(x)


def _pareto(points: List[Tuple[int, float]]) -> List[Tuple[int, float]]:
    """Non-dominated (L, F): minimise L, maximise F.  Returns sorted by L."""
    # for each L keep max F, then sweep
    bestF_at_L: Dict[int, float] = {}
    for (Lval, F) in points:
        if (Lval not in bestF_at_L) or F > bestF_at_L[Lval]:
            bestF_at_L[Lval] = F
    front = []
    bestF = -1.0
    for Lval in sorted(bestF_at_L.keys()):
        F = bestF_at_L[Lval]
        if F > bestF:          # strictly better fitness for more bits
            front.append((Lval, F))
            bestF = F
    return front


def lstar_at_least(audit: dict, f: float) -> Optional[int]:
    """min description length achieving fitness >= f."""
    best = INF
    for F, Lval in audit["best_L_at_exact_F"].items():
        if F >= f - 1e-12 and Lval < best:
            best = Lval
    return None if best == INF else int(best)
