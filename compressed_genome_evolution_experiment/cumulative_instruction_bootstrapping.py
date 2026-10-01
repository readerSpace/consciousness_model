"""exp602 -- Cumulative Instruction Bootstrapping.

O1 is first discovered in the typed distant basis of exp601.  It is then a real
callable operation in a later definition grammar.  Exact dynamic enumeration of
typed normal forms measures whether retained useful history lowers the semantic
description cost and MDL birth threshold of O2 (four-way concatenation), unlike
an equally expensive but irrelevant retained instruction.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Dict, Iterable, List, Sequence, Tuple

import codec_evo as C
import numpy as np
import primitive_basis_ablation as P
from genome import gamma_bits


REF_BITS = 3
RAW_JOIN_BITS = 8  # BOX + PAIR + FLATTEN in the exp601 typed distant basis
DERIVED_CALL_BITS = 2


@dataclass(frozen=True)
class Expr:
    op: str
    children: Tuple["Expr", ...] = ()
    index: int = -1

    def bits(self) -> int:
        if self.op == "REF":
            return REF_BITS
        return self.op_bits() + sum(child.bits() for child in self.children)

    def op_bits(self) -> int:
        return RAW_JOIN_BITS if self.op == "RAW_JOIN" else DERIVED_CALL_BITS


@dataclass(frozen=True)
class Operation:
    name: str
    arity: int
    bits: int = DERIVED_CALL_BITS
    useful: bool = True


O1 = Operation("O1", 2)
O2 = Operation("O2", 4)
O_IRRELEVANT = Operation("O_IRRELEVANT_DUP", 1, useful=False)


@dataclass(frozen=True)
class History:
    name: str
    operations: Tuple[Operation, ...]
    retained_bits: int


BASE = History("base", (), 0)
USEFUL_O1 = History("history_useful_O1", (O1,), 14)
IRRELEVANT = History("history_irrelevant", (O_IRRELEVANT,), 14)
USEFUL_O1_O2 = History("history_useful_O1_O2", (O1, O2), 14 + 18)


def discover_o1() -> Tuple[int, Tuple[str, ...]]:
    """Recover O1 from exp601 rather than supplying a named primitive."""
    cost, witness = P.semantic_complexity("distant_box_pair_flatten")
    assert cost == 14 and witness is not None
    return cost, witness.tokens


def evaluate(expr: Expr, args: Sequence[Sequence[int]]) -> List[int]:
    if expr.op == "REF":
        return list(args[expr.index])
    values = [evaluate(child, args) for child in expr.children]
    if expr.op in ("RAW_JOIN", "O1", "O2"):
        out: List[int] = []
        for value in values:
            out.extend(value)
        return out
    if expr.op == "O_IRRELEVANT_DUP":
        return values[0] + values[0]
    raise ValueError(expr.op)


def _candidate_ops(history: History) -> Tuple[Operation, ...]:
    return (Operation("RAW_JOIN", 2, RAW_JOIN_BITS),) + tuple(
        operation for operation in history.operations if operation.useful)


def semantic_complexity(arity: int, history: History) -> Tuple[int, Expr]:
    """Exact dynamic enumeration of typed ordered concatenation normal forms.

    A state is a contiguous argument interval.  Every legal operation partitions
    it into its arity-many nonempty contiguous child intervals, so the dynamic
    program covers every well-typed expression that preserves argument order.
    """
    cache: Dict[Tuple[int, int], Tuple[int, Expr]] = {}

    def solve(start: int, end: int) -> Tuple[int, Expr]:
        key = (start, end)
        if key in cache:
            return cache[key]
        if end - start == 1:
            result = (REF_BITS, Expr("REF", index=start))
            cache[key] = result
            return result
        best_cost, best_expr = 10 ** 9, None
        for operation in _candidate_ops(history):
            if end - start < operation.arity:
                continue
            def partitions(remaining: int, parts: int, prefix: Tuple[int, ...]):
                if parts == 1:
                    yield prefix + (remaining,)
                else:
                    for size in range(1, remaining - parts + 2):
                        yield from partitions(remaining - size, parts - 1, prefix + (size,))
            for sizes in partitions(end - start, operation.arity, ()):
                cursor, children, cost = start, [], operation.bits
                for size in sizes:
                    child_cost, child = solve(cursor, cursor + size)
                    cursor += size
                    children.append(child)
                    cost += child_cost
                if cost < best_cost:
                    best_cost, best_expr = cost, Expr(operation.name, tuple(children))
        assert best_expr is not None
        cache[key] = (best_cost, best_expr)
        return cache[key]

    return solve(0, arity)


@dataclass(frozen=True)
class TupleWorld:
    motifs: Tuple[Tuple[int, ...], ...]
    tuples: Tuple[Tuple[int, ...], ...]
    sequence: Tuple[int, ...]
    A: int
    arity: int
    target: Tuple[int, ...]


def tuple_world(arity: int, n_tuples: int, reuse: int, seed: int = 1) -> TupleWorld:
    n_motifs = 2
    while n_motifs ** arity < n_tuples:
        n_motifs += 1
    rng = np.random.default_rng(seed)
    motifs = tuple(tuple(rng.integers(0, 2, size=4).tolist()) for _ in range(n_motifs))
    tuples = tuple(tuple(item) for item in list(product(range(n_motifs), repeat=arity))[:n_tuples])
    sequence = [index for index in range(n_tuples) for _ in range(reuse)]
    rng.shuffle(sequence)
    target: List[int] = []
    for tuple_index in sequence:
        for motif_index in tuples[tuple_index]:
            target.extend(motifs[motif_index])
    return TupleWorld(motifs, tuples, tuple(sequence), 2, arity, tuple(target))


def _motif_bits(world: TupleWorld) -> int:
    result = gamma_bits(len(world.motifs) + 1)
    for motif in world.motifs:
        result += C._list_bits([C.Ins(C.OP_LIT, symbol) for symbol in motif], world.A)
    return result


def _fixed_cost(world: TupleWorld) -> int:
    motif_macros = [[C.Ins(C.OP_LIT, symbol) for symbol in motif] for motif in world.motifs]
    direct = [C.Ins(C.OP_CALL, motif_index) for tuple_index in world.sequence
              for motif_index in world.tuples[tuple_index]]
    candidates = [C.Prog(motif_macros, direct, world.A, len(world.target))]
    tuple_macros = [list(macro) for macro in motif_macros]
    offset = len(tuple_macros)
    for item in world.tuples:
        tuple_macros.append([C.Ins(C.OP_CALL, motif_index) for motif_index in item])
    candidates.append(C.Prog(tuple_macros, [C.Ins(C.OP_CALL, offset + index)
                                             for index in world.sequence], world.A, len(world.target)))
    flat_macros = [[C.Ins(C.OP_LIT, symbol) for motif_index in item for symbol in world.motifs[motif_index]]
                   for item in world.tuples]
    candidates.append(C.Prog(flat_macros, [C.Ins(C.OP_CALL, index) for index in world.sequence],
                             world.A, len(world.target)))
    literal = C.Prog([], [C.Ins(C.OP_LIT, symbol) for symbol in world.target], world.A, len(world.target))
    candidates.append(literal)
    return min(C.description_length(candidate, C.LANG_D0, 0) for candidate in candidates)


def _derived_genome_cost(world: TupleWorld) -> int:
    calls = gamma_bits(len(world.sequence) + 1)
    calls += sum(C.OPCODE_BITS + sum(gamma_bits(motif_index + 1)
                                     for motif_index in world.tuples[tuple_index])
                 for tuple_index in world.sequence)
    return _motif_bits(world) + calls


def audit_birth(arity: int, history: History, n_tuples: int, reuse: int = 3) -> Dict[str, object]:
    world = tuple_world(arity, n_tuples, reuse)
    fixed = _fixed_cost(world)
    complexity, expression = semantic_complexity(arity, history)
    derived = _derived_genome_cost(world)
    total = complexity + derived
    born = total < fixed
    return {
        "arity": arity, "history": history.name, "fixed": fixed,
        "total": total if born else fixed, "born": born,
        "K": complexity, "expression": expression,
        "L_D": complexity if born else 0,
        "L_G_given_D": derived if born else fixed,
        "history_bits_sunk": history.retained_bits,
    }


def birth_threshold(arity: int, history: History, n_tuples: Sequence[int]) -> int | None:
    for n_tuples_value in n_tuples:
        if audit_birth(arity, history, n_tuples_value)["born"]:
            return n_tuples_value
    return None


def cumulative_trace() -> List[Dict[str, object]]:
    """Stage-specific history versus the original basis for O1, O2, and O3."""
    rows = []
    for stage, arity, history in ((1, 2, BASE), (2, 4, USEFUL_O1), (3, 8, USEFUL_O1_O2)):
        base_k, _ = semantic_complexity(arity, BASE)
        prior_k, _ = semantic_complexity(arity, history)
        rows.append({"stage": stage, "arity": arity, "K_P0": base_k,
                     "K_history": prior_k, "ratio": prior_k / base_k,
                     "history": history.name})
    return rows