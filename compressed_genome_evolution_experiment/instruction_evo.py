"""exp599b -- Emergent Genetic Instruction Set.

Unlike exp599a, this module never exposes a COMPOSE opcode.  A derived opcode
is born as a short stack program over the fixed primitives LITERAL, REF,
CONCAT, and REPEAT.  Its behaviour is measured after birth on reference-input
cases; a name is never used to decide whether it is compose-equivalent.

The experiment compares a fixed language, instruction birth, and an oracle
definition.  All three pay exact program bits, and a born definition also pays
its serialized decoder length L(D).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Dict, Iterable, List, Sequence, Tuple

import codec_evo as C
import numpy as np
from genome import gamma_bits


P_LITERAL, P_REF, P_CONCAT, P_REPEAT = range(4)
PRIMITIVE_BITS = 2
USE_OPCODE_BITS = 2


@dataclass(frozen=True)
class Primitive:
    """One token in a postfix decoder program.

    REF has argument 0 or 1, LITERAL has a binary symbol, and REPEAT has a
    positive repeat count.  CONCAT consumes the two top stack values.
    """

    op: int
    arg: int = 0


@dataclass(frozen=True)
class OpcodeDef:
    tokens: Tuple[Primitive, ...]

    def bits(self) -> int:
        return sum(primitive_bits(token) for token in self.tokens)

    def depth(self) -> int:
        depths: List[int] = []
        for token in self.tokens:
            if token.op in (P_LITERAL, P_REF):
                depths.append(1)
            elif token.op == P_REPEAT:
                if not depths:
                    return 0
                depths[-1] += 1
            elif token.op == P_CONCAT:
                if len(depths) < 2:
                    return 0
                right, left = depths.pop(), depths.pop()
                depths.append(max(left, right) + 1)
        return depths[0] if len(depths) == 1 else 0


@dataclass(frozen=True)
class Analysis:
    valid: bool
    compose_equivalent: bool
    definition_bits: int
    language_depth: int


@dataclass(frozen=True)
class DerivedProg:
    """A genome whose derived calls are interpreted only by an OpcodeDef."""

    motifs: Tuple[Tuple[int, ...], ...]
    calls: Tuple[Tuple[int, int], ...]
    definition: OpcodeDef
    A: int
    L: int


def primitive_bits(token: Primitive) -> int:
    if token.op == P_LITERAL:
        return PRIMITIVE_BITS + 1
    if token.op == P_REF:
        return PRIMITIVE_BITS + 1
    if token.op == P_CONCAT:
        return PRIMITIVE_BITS
    if token.op == P_REPEAT:
        return PRIMITIVE_BITS + gamma_bits(token.arg)
    raise ValueError(token.op)


def evaluate(definition: OpcodeDef, args: Sequence[Sequence[int]]) -> List[int]:
    """Execute a primitive definition; malformed programs raise ValueError."""
    stack: List[List[int]] = []
    for token in definition.tokens:
        if token.op == P_LITERAL:
            stack.append([token.arg])
        elif token.op == P_REF:
            if token.arg >= len(args):
                raise ValueError("reference outside opcode arity")
            stack.append(list(args[token.arg]))
        elif token.op == P_CONCAT:
            if len(stack) < 2:
                raise ValueError("CONCAT needs two values")
            right, left = stack.pop(), stack.pop()
            stack.append(left + right)
        elif token.op == P_REPEAT:
            if not stack or token.arg < 1:
                raise ValueError("REPEAT needs a value and positive count")
            stack.append(stack.pop() * token.arg)
        else:
            raise ValueError(token.op)
    if len(stack) != 1:
        raise ValueError("definition must leave exactly one value")
    return stack[0]


def primitive_definitions(max_tokens: int = 3) -> List[OpcodeDef]:
    """Mechanically enumerate the finite primitive grammar, without semantic labels.

    The token alphabet includes all required primitives.  Definitions are
    deduplicated structurally and rejected only by the generic stack evaluator.
    """
    alphabet = (
        Primitive(P_LITERAL, 0), Primitive(P_LITERAL, 1),
        Primitive(P_REF, 0), Primitive(P_REF, 1),
        Primitive(P_CONCAT), Primitive(P_REPEAT, 2),
    )
    out: List[OpcodeDef] = []
    for length in range(1, max_tokens + 1):
        for tokens in product(alphabet, repeat=length):
            definition = OpcodeDef(tokens)
            try:
                evaluate(definition, ([0], [1]))
            except ValueError:
                continue
            out.append(definition)
    return out


def analyse(definition: OpcodeDef) -> Analysis:
    """Post-hoc functional test; no opcode name participates in this decision."""
    cases = (
        ([], []), ([0], [1]), ([1, 0], [0]), ([1], [0, 1]),
        ([0, 1], [1, 1]),
    )
    try:
        equivalent = all(evaluate(definition, case) == list(case[0]) + list(case[1])
                         for case in cases)
    except ValueError:
        return Analysis(False, False, definition.bits(), definition.depth())
    return Analysis(True, equivalent, definition.bits(), definition.depth())


def compose_equivalents(max_tokens: int = 3) -> List[OpcodeDef]:
    return [definition for definition in primitive_definitions(max_tokens)
            if analyse(definition).compose_equivalent]


def _direct_genome(world: C.World) -> C.Prog:
    return C.g_D0_inline(world)


def _derived_genome(world: C.World, definition: OpcodeDef) -> DerivedProg:
    """Encode every pair with the born binary opcode.

    The encoder does not know a special operation: it validates the candidate
    definition's output for each pair before replacing two CALL instructions.
    """
    calls: List[Tuple[int, int]] = []
    for pair_index in world.seq:
        left, right = world.pairs[pair_index]
        if evaluate(definition, (world.motifs[left], world.motifs[right])) != (
                world.motifs[left] + world.motifs[right]):
            raise ValueError("definition cannot encode this world")
        calls.append((left, right))
    return DerivedProg(tuple(tuple(motif) for motif in world.motifs), tuple(calls),
                       definition, world.A, len(world.target))


def expand_derived(genome: DerivedProg) -> List[int]:
    out: List[int] = []
    for left, right in genome.calls:
        out.extend(evaluate(genome.definition, (genome.motifs[left], genome.motifs[right])))
    return out[:genome.L]


def derived_genome_bits(genome: DerivedProg) -> int:
    """L(G|D) for generic two-argument derived calls, with no special semantics."""
    macro_bits = gamma_bits(len(genome.motifs) + 1)
    for motif in genome.motifs:
        macro_bits += C._list_bits([C.Ins(C.OP_LIT, symbol) for symbol in motif], genome.A)
    call_bits = gamma_bits(len(genome.calls) + 1)
    call_bits += sum(USE_OPCODE_BITS + gamma_bits(left + 1) + gamma_bits(right + 1)
                     for left, right in genome.calls)
    return macro_bits + call_bits


def genome_bits(g: C.Prog) -> int:
    return C.description_length(g, C.LANG_D0, 0)


def _best_born(world: C.World, definitions: Iterable[OpcodeDef]) -> Tuple[int, OpcodeDef | None]:
    best_total, best_definition = C.best_cost(world, C.LANG_D0, 0)[0], None
    for definition in definitions:
        try:
            genome = _derived_genome(world, definition)
        except ValueError:
            continue
        if expand_derived(genome) != world.target:
            continue
        total = definition.bits() + derived_genome_bits(genome)
        if total < best_total:
            best_total, best_definition = total, definition
    return best_total, best_definition


def condition_costs(world: C.World, max_tokens: int = 3) -> Dict[str, object]:
    """Exact L(D)+L(G|D) in OFF, ON, and oracle conditions."""
    fixed = C.best_cost(world, C.LANG_D0, 0)[0]
    if world.kind != "compositional":
        return {
            "fixed": fixed, "birth_on": fixed, "oracle": fixed,
            "born": False, "definition": None, "analysis": None,
            "n_candidates": len(primitive_definitions(max_tokens)),
            "n_opcodes": 0, "D_language": 0, "R_opcode_use": 0,
            "L_D": 0, "L_G_given_D": fixed,
        }
    candidates = primitive_definitions(max_tokens)
    emergent, born = _best_born(world, candidates)
    oracle = OpcodeDef((Primitive(P_REF, 0), Primitive(P_REF, 1), Primitive(P_CONCAT)))
    oracle_total, _ = _best_born(world, [oracle])
    selected = born is not None
    return {
        "fixed": fixed,
        "birth_on": emergent,
        "oracle": oracle_total,
        "born": selected,
        "definition": born,
        "analysis": analyse(born) if born is not None else None,
        "n_candidates": len(candidates),
        "n_opcodes": 1 if selected else 0,
        "D_language": born.depth() if born is not None else 0,
        "R_opcode_use": len(world.seq) if selected else 0,
        "L_D": born.bits() if born is not None else 0,
        "L_G_given_D": emergent - (born.bits() if born is not None else 0),
    }


def _world_for_ncomp(n_comp: int, seed: int = 1) -> C.World:
    return C._world_for_ncomp(2, n_comp, 3, 4, np_rng(seed))


def np_rng(seed: int):
    return np.random.default_rng(seed)


def evolve_instruction_birth(world: C.World, seed: int = 0,
                             max_tokens: int = 3) -> Dict[str, object]:
    """A reproducible birth-and-selection process over grammar mutations.

    Each mutation proposes one previously unavailable primitive program in a
    random order.  Only a proposal that lowers the *full* description is
    retained.  This is deliberately a small language-evolution process rather
    than an oracle label: proposals are evaluated by decoder execution and MDL.
    """
    fixed = condition_costs(world, max_tokens)["fixed"]
    candidates = primitive_definitions(max_tokens)
    order = np.random.default_rng(seed).permutation(len(candidates))
    best, winner, proposed = int(fixed), None, 0
    if world.kind == "compositional":
        for candidate_index in order:
            proposed += 1
            candidate = candidates[int(candidate_index)]
            try:
                genome = _derived_genome(world, candidate)
            except ValueError:
                continue
            if expand_derived(genome) != world.target:
                continue
            total = candidate.bits() + derived_genome_bits(genome)
            if total < best:
                best, winner = total, candidate
    analysis = analyse(winner) if winner is not None else None
    return {
        "fixed": fixed, "total": best, "born": winner is not None,
        "definition": winner, "analysis": analysis, "proposals": proposed,
        "n_opcodes": int(winner is not None),
        "D_language": winner.depth() if winner is not None else 0,
        "R_opcode_use": len(world.seq) if winner is not None else 0,
        "L_D": winner.bits() if winner is not None else 0,
        "L_G_given_D": best - (winner.bits() if winner is not None else 0),
    }


def threshold_sweep(n_comps: Sequence[int], max_tokens: int = 3) -> Dict[str, object]:
    rows = []
    nstar = None
    for n_comp in n_comps:
        costs = condition_costs(_world_for_ncomp(n_comp), max_tokens)
        row = {"n_comp": n_comp, **costs}
        row["emergent_matches_oracle"] = costs["birth_on"] == costs["oracle"]
        rows.append(row)
        if nstar is None and costs["born"]:
            nstar = n_comp
    return {"rows": rows, "n_opcode_birth_MDL": nstar}


def birth_evolution_sweep(n_comps: Sequence[int], seeds: Sequence[int],
                          max_tokens: int = 3) -> Dict[str, object]:
    rows = []
    n_evo = None
    for n_comp in n_comps:
        world = _world_for_ncomp(n_comp)
        trials = [evolve_instruction_birth(world, seed, max_tokens) for seed in seeds]
        fraction = sum(trial["born"] for trial in trials) / len(trials)
        rows.append({"n_comp": n_comp, "frac_born": fraction})
        if n_evo is None and fraction >= 0.5:
            n_evo = n_comp
    return {"rows": rows, "n_opcode_birth_evo": n_evo}