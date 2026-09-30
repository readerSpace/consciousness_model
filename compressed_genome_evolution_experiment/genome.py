"""Decodable genome codec for Compressed Genome Evolution (exp588).

The core here is deliberately application-neutral: it knows only about
`symbol`, `instruction`, `program`, `macro`, `expand`, and `bits`.  It does not
mention fitness, selection, environment, or evolution -- those live in
`environments.py` / `evolution.py`.  `test_compressed_genome.py` checks this
mechanically (mirrors the phylogenetic-compression project's
`test_the_core_is_domain_neutral`).

A genome is a tiny program in a flat bytecode plus a table of reusable macros:

    genome = (macros, program)
    macros  = [macro_0, macro_1, ...]     # macro_i may only CALL/REP/MIR macro_j, j<i (a DAG)
    program = [instr, instr, ...]          # the "call program"; may reference any macro

Instructions (atomic, flat -- no inline nested blocks):

    LIT  s      emit symbol s               (s in 0..A-1)
    CALL k      expand macro k
    REP  n k    expand macro k, n times
    MIR  k      expand macro k, then its reverse

`expand` concatenates the program's emissions and pads/truncates to exactly L.

Serialization is a genuine prefix code, so the bit lengths are the lengths of a
string that `decode` reads back exactly:

    number of macros M            Elias gamma(M+1)
    each macro: len Lm            Elias gamma(Lm+1) then Lm instructions
    program:    len Lp            Elias gamma(Lp+1) then Lp instructions
    instruction                   2-bit opcode + operands

    operand symbol s              ceil(log2(A)) bits
    operand macro index k         Elias gamma(k+1)
    operand count  n              Elias gamma(n)

L(G) = bits of the *program* section (what B and C both penalize).
L(D) = bits of the *macro* section  (what only C penalizes).
The interpreter itself is fixed and shared across the population, so it is a
constant and is not counted -- which is exactly why hiding the whole answer in
one giant macro costs L(D) but not L(G).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

# ---------------------------------------------------------------------------
# opcodes
# ---------------------------------------------------------------------------
OP_LIT = 0
OP_CALL = 1
OP_REP = 2
OP_MIR = 3
OPCODE_BITS = 2  # four opcodes


@dataclass(frozen=True)
class Instr:
    op: int
    a: int = 0  # LIT: symbol ; CALL/MIR: macro index ; REP: count
    b: int = 0  # REP: macro index ; unused otherwise

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return {
            OP_LIT: f"LIT {self.a}",
            OP_CALL: f"CALL {self.a}",
            OP_REP: f"REP {self.a} m{self.b}",
            OP_MIR: f"MIR {self.a}",
        }[self.op]


@dataclass
class Genome:
    macros: List[List[Instr]]
    program: List[Instr]
    A: int          # alphabet size
    L: int          # phenotype length
    nmax: int = 8   # max repeat count

    # -- structural helpers --------------------------------------------------
    def copy(self) -> "Genome":
        return Genome(
            macros=[list(m) for m in self.macros],
            program=list(self.program),
            A=self.A,
            L=self.L,
            nmax=self.nmax,
        )


# ---------------------------------------------------------------------------
# Elias gamma (for positive integers >= 1)
# ---------------------------------------------------------------------------
def gamma_bits(x: int) -> int:
    """Bit length of Elias-gamma code of x (x >= 1)."""
    if x < 1:
        raise ValueError(f"gamma needs x>=1, got {x}")
    return 2 * x.bit_length() - 1


def _gamma_encode(x: int, out: List[int]) -> None:
    if x < 1:
        raise ValueError(f"gamma needs x>=1, got {x}")
    n = x.bit_length()
    out.extend([0] * (n - 1))          # n-1 leading zeros
    for i in range(n - 1, -1, -1):     # n bits of x, MSB first
        out.append((x >> i) & 1)


def _gamma_decode(bits: List[int], pos: int) -> Tuple[int, int]:
    zeros = 0
    while bits[pos] == 0:
        zeros += 1
        pos += 1
    # now bits[pos] is the leading 1 of an (zeros+1)-bit number
    val = 0
    for _ in range(zeros + 1):
        val = (val << 1) | bits[pos]
        pos += 1
    return val, pos


# ---------------------------------------------------------------------------
# symbol operand width
# ---------------------------------------------------------------------------
def _sym_bits(A: int) -> int:
    return max(1, (A - 1).bit_length())


# ---------------------------------------------------------------------------
# per-instruction bit cost
# ---------------------------------------------------------------------------
def instr_bits(ins: Instr, A: int) -> int:
    sb = _sym_bits(A)
    if ins.op == OP_LIT:
        return OPCODE_BITS + sb
    if ins.op == OP_CALL:
        return OPCODE_BITS + gamma_bits(ins.a + 1)
    if ins.op == OP_MIR:
        return OPCODE_BITS + gamma_bits(ins.a + 1)
    if ins.op == OP_REP:
        return OPCODE_BITS + gamma_bits(ins.a) + gamma_bits(ins.b + 1)
    raise ValueError(ins.op)


def _instr_list_bits(instrs: List[Instr], A: int) -> int:
    return gamma_bits(len(instrs) + 1) + sum(instr_bits(i, A) for i in instrs)


def program_bits(g: Genome) -> int:
    """L(G): serialized bits of the main call program (penalized by B and C)."""
    return _instr_list_bits(g.program, g.A)


def macro_bits(g: Genome) -> int:
    """L(D): serialized bits of the macro table (penalized only by C)."""
    b = gamma_bits(len(g.macros) + 1)
    for m in g.macros:
        b += _instr_list_bits(m, g.A)
    return b


def description_length(g: Genome) -> int:
    """Total serialized length L(G) + L(D), computed independently of encode()."""
    return program_bits(g) + macro_bits(g)


# ---------------------------------------------------------------------------
# expansion (genotype -> phenotype)
# ---------------------------------------------------------------------------
def _expand_instrs(instrs: List[Instr], expanded_macros: List[List[int]],
                   cap: int) -> List[int]:
    """Expand a flat instruction list, stopping once `cap` symbols are produced
    (bounds cost of large REP counts; the phenotype only ever needs L symbols)."""
    out: List[int] = []
    for ins in instrs:
        if ins.op == OP_LIT:
            out.append(ins.a)
        elif ins.op == OP_CALL:
            out.extend(expanded_macros[ins.a])
        elif ins.op == OP_REP:
            body = expanded_macros[ins.b]
            for _ in range(ins.a):
                out.extend(body)
                if len(out) >= cap:
                    break
        elif ins.op == OP_MIR:
            body = expanded_macros[ins.a]
            out.extend(body)
            out.extend(body[::-1])
        else:
            raise ValueError(ins.op)
        if len(out) >= cap:
            break
    return out[:cap]


def expand(g: Genome) -> List[int]:
    """Deterministically expand to exactly L symbols (pad with 0 / truncate).

    Every intermediate expansion is capped at L symbols, so REP counts can never
    blow up cost and a macro is never materialised beyond the phenotype length."""
    validate(g)
    expanded_macros: List[List[int]] = []
    for i, m in enumerate(g.macros):
        # macro i may only reference macros < i -> DAG, no recursion
        expanded_macros.append(_expand_instrs(m, expanded_macros, g.L))
    seq = _expand_instrs(g.program, expanded_macros, g.L)
    if len(seq) < g.L:
        seq = seq + [0] * (g.L - len(seq))
    return seq[: g.L]


# ---------------------------------------------------------------------------
# validation (enforces the DAG constraint and symbol ranges)
# ---------------------------------------------------------------------------
def validate(g: Genome) -> None:
    M = len(g.macros)
    for i, m in enumerate(g.macros):
        for ins in m:
            _validate_instr(ins, g.A, g.nmax, max_macro=i)  # only < i
    for ins in g.program:
        _validate_instr(ins, g.A, g.nmax, max_macro=M)      # any existing macro


def _validate_instr(ins: Instr, A: int, nmax: int, max_macro: int) -> None:
    if ins.op == OP_LIT:
        if not (0 <= ins.a < A):
            raise ValueError(f"LIT symbol {ins.a} out of range [0,{A})")
    elif ins.op in (OP_CALL, OP_MIR):
        if not (0 <= ins.a < max_macro):
            raise ValueError(f"macro index {ins.a} not in [0,{max_macro})")
    elif ins.op == OP_REP:
        if not (1 <= ins.a <= nmax):
            raise ValueError(f"REP count {ins.a} not in [1,{nmax}]")
        if not (0 <= ins.b < max_macro):
            raise ValueError(f"REP macro index {ins.b} not in [0,{max_macro})")
    else:
        raise ValueError(f"bad opcode {ins.op}")


# ---------------------------------------------------------------------------
# encode / decode  (bit-exact round trip)
# ---------------------------------------------------------------------------
def _encode_instr(ins: Instr, A: int, out: List[int]) -> None:
    sb = _sym_bits(A)
    for i in range(OPCODE_BITS - 1, -1, -1):
        out.append((ins.op >> i) & 1)
    if ins.op == OP_LIT:
        for i in range(sb - 1, -1, -1):
            out.append((ins.a >> i) & 1)
    elif ins.op in (OP_CALL, OP_MIR):
        _gamma_encode(ins.a + 1, out)
    elif ins.op == OP_REP:
        _gamma_encode(ins.a, out)
        _gamma_encode(ins.b + 1, out)


def _encode_instr_list(instrs: List[Instr], A: int, out: List[int]) -> None:
    _gamma_encode(len(instrs) + 1, out)
    for ins in instrs:
        _encode_instr(ins, A, out)


def encode(g: Genome) -> List[int]:
    """Serialize to a bit list; ``len(encode(g)) == description_length(g)``."""
    validate(g)
    out: List[int] = []
    # macro section first (L(D)), then program section (L(G))
    _gamma_encode(len(g.macros) + 1, out)
    for m in g.macros:
        _encode_instr_list(m, g.A, out)
    _encode_instr_list(g.program, g.A, out)
    return out


def _decode_instr(bits: List[int], pos: int, A: int) -> Tuple[Instr, int]:
    sb = _sym_bits(A)
    op = 0
    for _ in range(OPCODE_BITS):
        op = (op << 1) | bits[pos]
        pos += 1
    if op == OP_LIT:
        s = 0
        for _ in range(sb):
            s = (s << 1) | bits[pos]
            pos += 1
        return Instr(OP_LIT, s), pos
    if op in (OP_CALL, OP_MIR):
        k1, pos = _gamma_decode(bits, pos)
        return Instr(op, k1 - 1), pos
    if op == OP_REP:
        n, pos = _gamma_decode(bits, pos)
        k1, pos = _gamma_decode(bits, pos)
        return Instr(OP_REP, n, k1 - 1), pos
    raise ValueError(f"bad opcode {op}")


def _decode_instr_list(bits: List[int], pos: int, A: int) -> Tuple[List[Instr], int]:
    n1, pos = _gamma_decode(bits, pos)
    n = n1 - 1
    instrs = []
    for _ in range(n):
        ins, pos = _decode_instr(bits, pos, A)
        instrs.append(ins)
    return instrs, pos


def decode(bits: List[int], A: int, L: int, nmax: int = 8) -> Genome:
    pos = 0
    m1, pos = _gamma_decode(bits, pos)
    M = m1 - 1
    macros = []
    for _ in range(M):
        m, pos = _decode_instr_list(bits, pos, A)
        macros.append(m)
    program, pos = _decode_instr_list(bits, pos, A)
    g = Genome(macros=macros, program=program, A=A, L=L, nmax=nmax)
    validate(g)
    return g


# ---------------------------------------------------------------------------
# reuse accounting (how many times each macro is actually invoked)
# ---------------------------------------------------------------------------
def macro_call_counts(g: Genome) -> List[int]:
    """Total invocations of each macro across program + other macros.

    A REP n k counts as n invocations of k; MIR counts as 1 (structural reuse).
    Counts propagate through the DAG (a macro invoked inside another macro is
    counted with the outer macro's multiplicity)."""
    M = len(g.macros)
    direct = [0] * M  # weighted direct references from the program

    def add(instrs, weight, target):
        for ins in instrs:
            if ins.op == OP_CALL or ins.op == OP_MIR:
                target[ins.a] += weight
            elif ins.op == OP_REP:
                target[ins.b] += weight * ins.a

    add(g.program, 1, direct)
    # propagate through macros in reverse topological order (high index first)
    total = list(direct)
    for i in range(M - 1, -1, -1):
        w = total[i]
        if w:
            add(g.macros[i], w, total)
    return total


def reuse_rate(g: Genome) -> float:
    """Mean invocations per *used* macro whose body carries >= 2 literals.

    A macro that expands to a single symbol, or that is used only once, is not
    genuine reuse; this metric isolates "a nontrivial block called many times",
    which is what distinguishes rule discovery from literal memorisation."""
    counts = macro_call_counts(g)
    reuse = []
    for i, m in enumerate(g.macros):
        n_lit = _macro_literal_span(g, i)
        if n_lit >= 2 and counts[i] >= 1:
            reuse.append(counts[i])
    if not reuse:
        return 0.0
    return float(sum(reuse)) / len(reuse)


def mean_used_macro_span(g: Genome) -> float:
    """Average expanded size of macros that are actually invoked.

    Distinguishes "a small motif reused many times" (small span, high reuse ->
    a genuine rule) from "a big chunk of the answer stored once" (large span ->
    hiding, the B-group shortcut)."""
    counts = macro_call_counts(g)
    spans = [ _macro_literal_span(g, i) for i in range(len(g.macros)) if counts[i] >= 1 ]
    if not spans:
        return 0.0
    return float(sum(spans)) / len(spans)


def _macro_literal_span(g: Genome, idx: int) -> int:
    """Number of symbols macro idx expands to (its 'size')."""
    expanded: List[List[int]] = []
    for i in range(idx + 1):
        expanded.append(_expand_instrs(g.macros[i], expanded, g.L))
    return len(expanded[idx])
