"""Tests for exp599b -- Emergent Genetic Instruction Set.

Run: python -m pytest test_instruction_evo.py -q
"""

from __future__ import annotations

import numpy as np

import codec_evo as C
import instruction_evo as I


def test_primitive_decoder_executes_compose_without_named_opcode():
    definition = I.OpcodeDef((I.Primitive(I.P_REF, 0),
                              I.Primitive(I.P_REF, 1),
                              I.Primitive(I.P_CONCAT)))
    assert I.evaluate(definition, ([1, 0], [0, 1])) == [1, 0, 0, 1]
    assert I.analyse(definition).compose_equivalent


def test_enumeration_contains_all_primitives_and_discovers_compose_by_behaviour():
    candidates = I.primitive_definitions()
    ops = {token.op for definition in candidates for token in definition.tokens}
    assert {I.P_LITERAL, I.P_REF, I.P_CONCAT, I.P_REPEAT} <= ops
    equivalents = I.compose_equivalents()
    assert len(equivalents) >= 1
    assert all(I.analyse(definition).compose_equivalent for definition in equivalents)


def test_instruction_birth_is_off_in_flat_and_random_controls():
    rng = np.random.default_rng(3)
    for world in (C.flat_world(2, 96, 2, rng), C.random_world(2, 96, rng)):
        costs = I.condition_costs(world)
        assert not costs["born"]
        assert costs["birth_on"] == costs["fixed"] == costs["oracle"]


def test_emergent_definition_matches_oracle_and_beats_fixed_composition():
    world = C._world_for_ncomp(2, 2, 3, 4, np.random.default_rng(1))
    costs = I.condition_costs(world)
    assert costs["born"]
    assert costs["birth_on"] == costs["oracle"] < costs["fixed"]
    assert costs["analysis"].compose_equivalent
    assert costs["L_D"] > 0
    assert costs["L_G_given_D"] < costs["fixed"]


def test_derived_genome_expands_via_primitive_definition():
    world = C._world_for_ncomp(2, 2, 3, 4, np.random.default_rng(1))
    definition = I.compose_equivalents()[0]
    genome = I._derived_genome(world, definition)
    assert I.expand_derived(genome) == world.target
    assert I.derived_genome_bits(genome) + definition.bits() < C.best_cost(
        world, C.LANG_D0, 0)[0]


def test_birth_threshold_matches_mdl_and_is_seed_deterministic():
    n_comps = [1, 2, 4, 6]
    exact = I.threshold_sweep(n_comps)
    evolved = I.birth_evolution_sweep(n_comps, [1, 2, 3])
    assert exact["n_opcode_birth_MDL"] == 2
    assert evolved["n_opcode_birth_evo"] == exact["n_opcode_birth_MDL"]


def test_malformed_primitive_program_is_rejected():
    malformed = I.OpcodeDef((I.Primitive(I.P_CONCAT),))
    assert not I.analyse(malformed).valid
