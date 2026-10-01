"""exp601 -- Primitive Basis Ablation / Computational Necessity.

For each primitive basis P this module computes K_P(a||b), the shortest
definition with concatenation semantics, and tests whether that semantic cost
predicts the MDL threshold at which an instruction is retained.  Ablations
separate function expressibility (REF/order/merge) from call reuse.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Dict, List, Sequence, Tuple

import codec_evo as C
import numpy as np
from genome import gamma_bits


@dataclass(frozen=True)
class Basis:
    name: str
    tokens: Tuple[str, ...]
    bits: Dict[str, int]
    reusable: bool = True
    max_tokens: int = 3


@dataclass(frozen=True)
class Definition:
    basis: str
    tokens: Tuple[str, ...]

    def bits(self) -> int:
        return sum(BASES[self.basis].bits[token] for token in self.tokens)


BASES = {
    "full": Basis("full", ("REF0", "REF1", "LIT0", "LIT1", "MERGE", "REPEAT2"),
                  {"REF0": 3, "REF1": 3, "LIT0": 3, "LIT1": 3, "MERGE": 2, "REPEAT2": 3}),
    # No references: constants cannot represent arbitrary input sequences.
    "minus_ref": Basis("minus_ref", ("LIT0", "LIT1", "MERGE", "REPEAT2"),
                       {"LIT0": 3, "LIT1": 3, "MERGE": 2, "REPEAT2": 3}),
    # Inputs can be introduced only as a fixed (a,b) stack pair; the only merge
    # consumes them as b||a, so the program cannot choose their evaluation order.
    "minus_order": Basis("minus_order", ("ARG0_THEN_ARG1", "LIT0", "LIT1", "REVERSE_MERGE"),
                         {"ARG0_THEN_ARG1": 4, "LIT0": 3, "LIT1": 3, "REVERSE_MERGE": 2}),
    "minus_merge": Basis("minus_merge", ("REF0", "REF1", "LIT0", "LIT1", "REPEAT2"),
                         {"REF0": 3, "REF1": 3, "LIT0": 3, "LIT1": 3, "REPEAT2": 3}),
    # Same semantic basis as full, but an invented instruction cannot be called
    # at each recurring pair, so it cannot amortize its definition length.
    "minus_reuse": Basis("minus_reuse", ("REF0", "REF1", "LIT0", "LIT1", "MERGE", "REPEAT2"),
                         {"REF0": 3, "REF1": 3, "LIT0": 3, "LIT1": 3, "MERGE": 2, "REPEAT2": 3},
                         reusable=False),
    # A typed, lower-level route: input a must be boxed before PAIR accepts it,
    # then FLATTEN converts the pair value back to a sequence.  It expresses the
    # same function but has a genuinely longer minimal definition.
    "distant_box_pair_flatten": Basis(
        "distant_box_pair_flatten", ("REF0", "REF1", "BOX", "PAIR", "FLATTEN", "LIT0"),
        {"REF0": 3, "REF1": 3, "BOX": 3, "PAIR": 2, "FLATTEN": 3, "LIT0": 3}, max_tokens=5),
}


def evaluate(definition: Definition, args: Sequence[Sequence[int]]) -> List[int]:
    """Execute one basis; type errors make a candidate invalid."""
    stack: List[object] = []
    for token in definition.tokens:
        if token == "REF0":
            stack.append(list(args[0]))
        elif token == "REF1":
            stack.append(list(args[1]))
        elif token == "ARG0_THEN_ARG1":
            stack.extend((list(args[0]), list(args[1])))
        elif token == "LIT0":
            stack.append([0])
        elif token == "LIT1":
            stack.append([1])
        elif token == "REPEAT2":
            if not stack or not isinstance(stack[-1], list):
                raise ValueError("REPEAT2 needs a sequence")
            stack.append(stack.pop() * 2)
        elif token in ("MERGE", "REVERSE_MERGE"):
            if len(stack) < 2 or not isinstance(stack[-1], list) or not isinstance(stack[-2], list):
                raise ValueError("merge needs two sequences")
            right, left = stack.pop(), stack.pop()
            stack.append(left + right if token == "MERGE" else right + left)
        elif token == "BOX":
            if not stack or not isinstance(stack[-1], list):
                raise ValueError("BOX needs a sequence")
            stack.append(("box", stack.pop()))
        elif token == "PAIR":
            if len(stack) < 2 or not isinstance(stack[-2], tuple) or not isinstance(stack[-1], list):
                raise ValueError("PAIR needs boxed-left and sequence-right")
            right = stack.pop()
            _, left = stack.pop()
            stack.append(("pair", left, right))
        elif token == "FLATTEN":
            if not stack or not isinstance(stack[-1], tuple) or stack[-1][0] != "pair":
                raise ValueError("FLATTEN needs a pair")
            _, left, right = stack.pop()
            stack.append(left + right)
        else:
            raise ValueError(token)
    if len(stack) != 1 or not isinstance(stack[0], list):
        raise ValueError("definition must leave one sequence")
    return stack[0]


def definitions(basis_name: str) -> List[Definition]:
    basis = BASES[basis_name]
    out = []
    for length in range(1, basis.max_tokens + 1):
        for tokens in product(basis.tokens, repeat=length):
            definition = Definition(basis_name, tokens)
            try:
                evaluate(definition, ([0], [1]))
            except ValueError:
                continue
            out.append(definition)
    return out


_CASES = (([], []), ([0], [1]), ([1, 0], [0]), ([1], [0, 1]), ([0, 1], [1, 1]))


def is_concatenation(definition: Definition) -> bool:
    try:
        return all(evaluate(definition, case) == list(case[0]) + list(case[1]) for case in _CASES)
    except ValueError:
        return False


def semantic_complexity(basis_name: str) -> Tuple[int | None, Definition | None]:
    matches = [definition for definition in definitions(basis_name) if is_concatenation(definition)]
    if not matches:
        return None, None
    witness = min(matches, key=lambda definition: (definition.bits(), len(definition.tokens)))
    return witness.bits(), witness


def _derived_genome_bits(world: C.World) -> int:
    macro_bits = gamma_bits(len(world.motifs) + 1)
    for motif in world.motifs:
        macro_bits += C._list_bits([C.Ins(C.OP_LIT, symbol) for symbol in motif], world.A)
    call_bits = gamma_bits(len(world.seq) + 1)
    call_bits += sum(2 + gamma_bits(world.pairs[index][0] + 1) + gamma_bits(world.pairs[index][1] + 1)
                     for index in world.seq)
    return macro_bits + call_bits


def audit_basis(world: C.World, basis_name: str) -> Dict[str, object]:
    basis = BASES[basis_name]
    fixed = C.best_cost(world, C.LANG_D0, 0)[0]
    complexity, witness = semantic_complexity(basis_name)
    reusable = basis.reusable and world.kind == "compositional"
    total = fixed
    born = False
    if complexity is not None and reusable:
        candidate = complexity + _derived_genome_bits(world)
        if candidate < fixed:
            total, born = candidate, True
    return {
        "basis": basis_name, "fixed": fixed, "total": total, "born": born,
        "expressible": complexity is not None, "K_concat": complexity,
        "witness": witness, "reusable": basis.reusable,
        "L_D": complexity if born else 0,
        "L_G_given_D": total - (complexity if born else 0),
    }


def birth_threshold(basis_name: str, n_comps: Sequence[int]) -> int | None:
    for n_comp in n_comps:
        world = C._world_for_ncomp(2, n_comp, 3, 4, np.random.default_rng(1))
        if audit_basis(world, basis_name)["born"]:
            return n_comp
    return None


def basis_audit(n_comps: Sequence[int], basis_names: Sequence[str] = tuple(BASES)) -> Dict[str, Dict[str, object]]:
    world = C._world_for_ncomp(2, 8, 3, 4, np.random.default_rng(1))
    out = {}
    for basis_name in basis_names:
        result = audit_basis(world, basis_name)
        result["n_birth_MDL"] = birth_threshold(basis_name, n_comps)
        out[basis_name] = result
    return out