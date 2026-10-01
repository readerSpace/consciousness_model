"""exp600 -- Instruction Necessity / Alternative-Grammar Audit.

The audit asks whether composition is an artefact of exp599b's postfix syntax.
It enumerates short decoder programs in independently named primitive grammars,
then compares their input-output semantics rather than their syntax.  A fourth,
composition-incapable grammar is a feasibility control.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Dict, Iterable, List, Sequence, Tuple

import codec_evo as C
import numpy as np
from genome import gamma_bits


@dataclass(frozen=True)
class Token:
    name: str


@dataclass(frozen=True)
class Grammar:
    name: str
    alphabet: Tuple[Token, ...]
    token_bits: Dict[str, int]


@dataclass(frozen=True)
class Definition:
    grammar: str
    tokens: Tuple[Token, ...]

    def bits(self) -> int:
        return sum(GRAMMARS[self.grammar].token_bits[token.name] for token in self.tokens)


@dataclass(frozen=True)
class SemanticAnalysis:
    valid: bool
    concatenate_equivalent: bool
    definition_bits: int
    depth: int


@dataclass(frozen=True)
class DerivedProgram:
    motifs: Tuple[Tuple[int, ...], ...]
    calls: Tuple[Tuple[int, int], ...]
    definition: Definition
    A: int
    L: int


def _grammar(name: str, names: Sequence[str], bits: Dict[str, int]) -> Grammar:
    return Grammar(name, tuple(Token(token) for token in names), bits)


GRAMMARS = {
    "postfix_ref_concat": _grammar(
        "postfix_ref_concat",
        ("REF0", "REF1", "LITERAL0", "LITERAL1", "CONCAT", "REPEAT2"),
        {"REF0": 3, "REF1": 3, "LITERAL0": 3, "LITERAL1": 3, "CONCAT": 2, "REPEAT2": 3},
    ),
    "copy_append": _grammar(
        "copy_append",
        ("COPY0", "COPY1", "ZERO", "ONE", "SLICE1", "APPEND", "DUPLICATE"),
        {"COPY0": 3, "COPY1": 3, "ZERO": 3, "ONE": 3, "SLICE1": 3, "APPEND": 2,
         "DUPLICATE": 3},
    ),
    "stack_merge": _grammar(
        "stack_merge",
        ("PUSH0", "PUSH1", "PUSH_ZERO", "PUSH_ONE", "MERGE", "DUP_TOP"),
        {"PUSH0": 3, "PUSH1": 3, "PUSH_ZERO": 3, "PUSH_ONE": 3, "MERGE": 2, "DUP_TOP": 3},
    ),
    "copy_repeat_only": _grammar(
        "copy_repeat_only",
        ("COPY0", "COPY1", "ZERO", "ONE", "DUPLICATE"),
        {"COPY0": 3, "COPY1": 3, "ZERO": 3, "ONE": 3, "DUPLICATE": 3},
    ),
}


def _kind(token: Token) -> str:
    return token.name


def evaluate(definition: Definition, args: Sequence[Sequence[int]]) -> List[int]:
    """Run a grammar-specific decoder without a named compose operation."""
    stack: List[List[int]] = []
    for token in definition.tokens:
        name = _kind(token)
        if name in ("REF0", "COPY0", "PUSH0"):
            stack.append(list(args[0]))
        elif name in ("REF1", "COPY1", "PUSH1"):
            stack.append(list(args[1]))
        elif name in ("LITERAL0", "ZERO", "PUSH_ZERO"):
            stack.append([0])
        elif name in ("LITERAL1", "ONE", "PUSH_ONE"):
            stack.append([1])
        elif name in ("CONCAT", "APPEND", "MERGE"):
            if len(stack) < 2:
                raise ValueError("binary merge needs two stack values")
            right, left = stack.pop(), stack.pop()
            stack.append(left + right)
        elif name in ("REPEAT2", "DUPLICATE"):
            if not stack:
                raise ValueError("duplication needs a stack value")
            stack.append(stack.pop() * 2)
        elif name == "SLICE1":
            if not stack:
                raise ValueError("SLICE needs a stack value")
            stack.append(stack.pop()[:1])
        elif name == "DUP_TOP":
            if not stack:
                raise ValueError("DUP_TOP needs a stack value")
            stack.append(list(stack[-1]))
        else:
            raise ValueError(name)
    if len(stack) != 1:
        raise ValueError("definition must leave exactly one value")
    return stack[0]


def definitions(grammar_name: str, max_tokens: int = 3) -> List[Definition]:
    grammar = GRAMMARS[grammar_name]
    out = []
    for length in range(1, max_tokens + 1):
        for tokens in product(grammar.alphabet, repeat=length):
            definition = Definition(grammar_name, tokens)
            try:
                evaluate(definition, ([0], [1]))
            except ValueError:
                continue
            out.append(definition)
    return out


def analyse(definition: Definition) -> SemanticAnalysis:
    cases = (([], []), ([0], [1]), ([1, 0], [0]), ([1], [0, 1]), ([0, 1], [1, 1]))
    try:
        converges = all(evaluate(definition, case) == list(case[0]) + list(case[1])
                        for case in cases)
    except ValueError:
        return SemanticAnalysis(False, False, definition.bits(), 0)
    return SemanticAnalysis(True, converges, definition.bits(), len(definition.tokens))


def _derived(world: C.World, definition: Definition) -> DerivedProgram:
    calls = []
    for pair_index in world.seq:
        left, right = world.pairs[pair_index]
        if evaluate(definition, (world.motifs[left], world.motifs[right])) != (
                world.motifs[left] + world.motifs[right]):
            raise ValueError("definition cannot encode this world")
        calls.append((left, right))
    return DerivedProgram(tuple(tuple(motif) for motif in world.motifs), tuple(calls),
                          definition, world.A, len(world.target))


def _expand(program: DerivedProgram) -> List[int]:
    out: List[int] = []
    for left, right in program.calls:
        out.extend(evaluate(program.definition, (program.motifs[left], program.motifs[right])))
    return out[:program.L]


def _genome_bits(program: DerivedProgram) -> int:
    macro_bits = gamma_bits(len(program.motifs) + 1)
    for motif in program.motifs:
        macro_bits += C._list_bits([C.Ins(C.OP_LIT, symbol) for symbol in motif], program.A)
    calls = gamma_bits(len(program.calls) + 1)
    calls += sum(2 + gamma_bits(left + 1) + gamma_bits(right + 1)
                 for left, right in program.calls)
    return macro_bits + calls


def audit_grammar(world: C.World, grammar_name: str, max_tokens: int = 3) -> Dict[str, object]:
    """Find the MDL-best born opcode in one grammar and classify it post-hoc."""
    fixed = C.best_cost(world, C.LANG_D0, 0)[0]
    best, winner = fixed, None
    candidates = definitions(grammar_name, max_tokens)
    if world.kind == "compositional":
        for definition in candidates:
            try:
                program = _derived(world, definition)
            except ValueError:
                continue
            if _expand(program) != world.target:
                continue
            total = definition.bits() + _genome_bits(program)
            if total < best:
                best, winner = total, definition
    analysis = analyse(winner) if winner is not None else None
    return {
        "grammar": grammar_name, "fixed": fixed, "total": best,
        "born": winner is not None, "definition": winner,
        "analysis": analysis, "n_candidates": len(candidates),
        "L_D": winner.bits() if winner is not None else 0,
        "L_G_given_D": best - (winner.bits() if winner is not None else 0),
    }


def convergence_audit(world: C.World, grammar_names: Iterable[str] = GRAMMARS,
                      max_tokens: int = 3) -> Dict[str, Dict[str, object]]:
    return {name: audit_grammar(world, name, max_tokens) for name in grammar_names}


def threshold(grammar_name: str, n_comps: Sequence[int]) -> int | None:
    for n_comp in n_comps:
        world = C._world_for_ncomp(2, n_comp, 3, 4, np.random.default_rng(1))
        if audit_grammar(world, grammar_name)["born"]:
            return n_comp
    return None