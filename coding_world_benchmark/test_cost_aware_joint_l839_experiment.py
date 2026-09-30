"""Tests for L8.39: three question kinds over a context-conditioned joint."""
from __future__ import annotations

import pytest

from coding_world_benchmark.contextual_joint_l838_experiment import ContextualJoint
from coding_world_benchmark.cost_aware_dialogue_l836_experiment import (
    IMPOSSIBLE, MULTIPLE_OPTIMAL, RESOLVED,
)
from coding_world_benchmark.cost_aware_joint_l839_experiment import (
    ALL_KINDS, CONTEXT, RELATION, UNARY, choose_joint, context_questions,
    informative_by_kind, joint_questions, relation_questions,
    run_joint_dialogue, search_unary_blind, unary_questions, worlds,
)
from coding_world_benchmark.holdout_joint_dialogue_l8391_experiment import (
    ARMS, RATE, RUNTIME, SURFACES, build_families, build_universes,
    run_experiment, run_family,
)
from coding_world_benchmark.nway_semantics_l837_experiment import CONTRADICTION


@pytest.fixture(scope="module")
def universes():
    return build_universes()


@pytest.fixture(scope="module")
def report():
    return run_experiment()


def _space(family):
    return worlds(ContextualJoint(SURFACES, family.sets))


def _family(name):
    return next(f for f in build_families() if f.name == name)


def _universe(family, universes):
    nameable, coined = universes
    return coined if family.coined else nameable


# -- the world set ---------------------------------------------------------

def test_worlds_is_the_product_over_contexts():
    joint = ContextualJoint(SURFACES, (
        ("sim", frozenset({(RUNTIME, RATE), (RATE, RUNTIME)})),
        ("lab", frozenset({(RUNTIME, RATE)}))))
    assert len(worlds(joint)) == 2


def test_an_empty_context_leaves_no_world():
    joint = ContextualJoint(SURFACES, (
        ("sim", frozenset({(RUNTIME, RATE)})), ("lab", frozenset())))
    assert worlds(joint) == frozenset()


def test_an_empty_world_set_is_a_contradiction_not_a_resolution(universes):
    family = _family("CONTRADICTION")
    choice = choose_joint(_space(family), family.contexts,
                          _universe(family, universes), SURFACES)
    assert choice.state == CONTRADICTION


# -- what each kind can name ----------------------------------------------

def test_unary_questions_need_a_nameable_meaning(universes):
    nameable, coined = universes
    family = _family("COUPLING_ONLY")
    space, contexts = _space(family), family.contexts
    assert unary_questions(space, contexts, coined, SURFACES) == []
    assert unary_questions(space, contexts, nameable, SURFACES) != []


def test_relation_and_context_questions_name_no_meaning(universes):
    nameable, coined = universes
    family = _family("COUPLING_ONLY")
    space, contexts = _space(family), family.contexts
    asked = (relation_questions(space, contexts, coined, SURFACES)
             + context_questions(space, contexts, coined, SURFACES))
    assert asked
    for question in asked:
        for facts in nameable:
            assert facts.name not in question.text


def test_only_context_questions_are_answerable_without_recall(universes):
    family = _family("COUPLING_ONLY")
    space, contexts = _space(family), family.contexts
    universe = _universe(family, universes)
    for kind in ALL_KINDS:
        for question in joint_questions(space, contexts, universe, SURFACES, (kind,)):
            assert question.cost.needs_context is (kind != CONTEXT)


# -- safety ---------------------------------------------------------------

def test_every_question_partitions_the_worlds(universes):
    for family in build_families():
        space = _space(family)
        universe = _universe(family, universes)
        for question in joint_questions(space, family.contexts, universe, SURFACES):
            blocks = [block for _, block in question.outcomes]
            assert sum(len(block) for block in blocks) == len(space)
            assert frozenset().union(*blocks) == space


def test_no_session_ever_resolves_to_the_wrong_world(report):
    for run in report["runs"]:
        for arm in run["arms"].values():
            assert not arm["wrong"]
    assert report["metrics"]["false_joint_resolution_rate"] == 0.0


def test_nothing_askable_abstains_rather_than_guessing(report):
    run = next(r for r in report["runs"] if r["name"] == "NOTHING_ASKABLE")
    assert run["worlds"] == 2
    assert all(count == 0 for count in run["informative"].values())
    for arm in run["arms"].values():
        assert arm["state"] == IMPOSSIBLE and not arm["resolved"] and arm["turns"] == 0


def test_only_informative_questions_are_ever_asked(report):
    assert report["metrics"]["unnecessary_question_rate"] == 0.0


# -- the decisive family ---------------------------------------------------

def test_the_decisive_world_offers_no_unary_question_at_all(report):
    run = next(r for r in report["runs"] if r["name"] == "COUPLING_ONLY")
    assert run["informative"][UNARY] == 0
    assert run["informative"][RELATION] > 0
    assert run["unary_blind"]


def test_unary_only_cannot_finish_it_and_the_other_kinds_can(report):
    run = next(r for r in report["runs"] if r["name"] == "COUPLING_ONLY")
    assert run["arms"]["unary_only"]["state"] == IMPOSSIBLE
    assert run["arms"]["unary_relation"]["resolved"]
    assert run["arms"]["full"]["resolved"]


def test_the_zero_comes_from_nameability_not_from_the_pool(report):
    """The same world with names restored is solvable by unary questions alone."""
    control = next(r for r in report["runs"] if r["name"] == "NAMEABLE_CONTROL")
    assert control["informative"][UNARY] > 0
    assert control["arms"]["unary_only"]["resolved"]


def test_the_context_question_saves_an_episode_recall(report):
    run = next(r for r in report["runs"] if r["name"] == "COUPLING_ONLY")
    assert (run["arms"]["full"]["context_recalls"]
            < run["arms"]["unary_relation"]["context_recalls"])
    assert run["arms"]["full"]["turns"] == run["arms"]["unary_relation"]["turns"]


# -- no priority by kind ---------------------------------------------------

def test_each_kind_wins_somewhere(report):
    assert report["metrics"]["distinct_winning_kinds"] == len(ALL_KINDS)
    assert report["metrics"]["no_kind_priority"]


def test_a_marginal_uncertainty_is_answered_by_a_unary_question(report):
    run = next(r for r in report["runs"] if r["name"] == "MARGINAL_WINS")
    assert run["opening_kinds"] == (UNARY,)
    assert run["informative"][RELATION] == 0 and run["informative"][CONTEXT] == 0


def test_one_episode_leaves_the_relation_as_the_only_route(report):
    run = next(r for r in report["runs"] if r["name"] == "RELATION_WINS")
    assert run["opening_kinds"] == (RELATION,)
    assert run["arms"]["unary_only"]["state"] == IMPOSSIBLE


def test_tied_structural_questions_are_reported_rather_than_picked(universes):
    family = _family("COUPLING_ONLY")
    choice = choose_joint(_space(family), family.contexts,
                          _universe(family, universes), SURFACES)
    assert choice.state == MULTIPLE_OPTIMAL
    assert {question.kind for question in choice.questions} == {CONTEXT}


# -- the negative half -----------------------------------------------------

def test_no_unary_blind_world_exists_when_every_meaning_is_nameable(universes):
    nameable = universes[0][:3]
    assert search_unary_blind(nameable, contexts=("c1",)) == []
    assert search_unary_blind(nameable, contexts=("c1", "c2")) == []


def test_the_search_refuses_an_unnameable_universe(universes):
    with pytest.raises(AssertionError):
        search_unary_blind(universes[1][:3])


def test_singleton_marginals_force_a_singleton_world(universes):
    """The proposition the search is checking, stated directly."""
    nameable = universes[0]
    for family in build_families():
        space = _space(family)
        if not space:
            continue
        contexts = family.contexts
        blind = not joint_questions(space, contexts, nameable, SURFACES, (UNARY,))
        if blind:
            assert len(space) == 1


# -- the rule is L8.36's ---------------------------------------------------

def test_the_chooser_is_l836s_with_a_different_pool(universes):
    family = _family("NAMEABLE_CONTROL")
    space, contexts = _space(family), family.contexts
    universe = _universe(family, universes)
    full = choose_joint(space, contexts, universe, SURFACES)
    unary = choose_joint(space, contexts, universe, SURFACES, (UNARY,))
    # Same state machine, different question space -- and the cheaper structural
    # question wins only because it is available, not because of its kind.
    assert full.state == MULTIPLE_OPTIMAL and unary.state in (MULTIPLE_OPTIMAL,)
    assert {q.kind for q in full.questions} == {CONTEXT}
    assert {q.kind for q in unary.questions} == {UNARY}


def test_a_session_terminates_and_reports_its_state(universes):
    family = _family("COUPLING_ONLY")
    session = run_joint_dialogue(_space(family), family.contexts,
                                 _universe(family, universes), SURFACES,
                                 family.truth, ALL_KINDS)
    assert session.state == RESOLVED
    assert session.resolved and not session.wrong
    assert session.kinds == (CONTEXT, RELATION)


def test_every_arm_is_reported_for_every_family(report):
    for run in report["runs"]:
        assert set(run["arms"]) == {label for label, _ in ARMS}
    assert report["metrics"]["joint_dialogue_accuracy"] == 1.0
