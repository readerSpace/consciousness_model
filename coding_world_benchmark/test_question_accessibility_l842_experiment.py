"""Tests for L8.42: what can be asked, kept out of what it would cost."""
from __future__ import annotations

import dataclasses

import pytest

from coding_world_benchmark.cost_aware_dialogue_l836_experiment import (
    IMPOSSIBLE, RESOLVED, Cost,
)
from coding_world_benchmark.cost_aware_joint_l839_experiment import (
    CONTEXT, UNARY, choose_joint, joint_questions, worlds,
)
from coding_world_benchmark.holdout_accessibility_l8421_experiment import (
    LAB, PILOT, SIM, SURFACES, THREE_TRUTH, base_universe, build_families,
    run_experiment, three_world, two_world,
)
from coding_world_benchmark.holdout_joint_dialogue_l8391_experiment import (
    build_universes,
)
from coding_world_benchmark.question_accessibility_l842_experiment import (
    BLOCKED, Access, accessible_pool, askable, cheapest_unavailable, diagnose,
    named_contexts, run_accessible,
)


@pytest.fixture(scope="module")
def report():
    return run_experiment()


def _setup(world=three_world):
    joint = world()
    return worlds(joint), joint.contexts, base_universe()


def _run(report, name):
    return next(r for r in report["runs"] if r["name"] == name)


# -- accessibility is not a cost -------------------------------------------

def test_the_cost_vector_gained_no_accessibility_field():
    assert [f.name for f in dataclasses.fields(Cost)] == [
        "turns", "answer_space", "needs_context"]


def test_a_question_costs_the_same_whatever_the_access():
    space, contexts, universe = _setup()
    full = {q.text: q.cost for q in accessible_pool(
        SURFACES, Access.full(contexts))(space, contexts, universe)}
    limited = accessible_pool(SURFACES, Access.of(known=contexts,
                                                  detail=(SIM, PILOT)))(
        space, contexts, universe)
    assert limited
    for question in limited:
        assert full[question.text] == question.cost


def test_the_reachable_pool_is_a_subset_of_the_whole_pool():
    space, contexts, universe = _setup()
    everything = {q.text for q in joint_questions(space, contexts, universe,
                                                  SURFACES)}
    for access in (Access.of(known=contexts, detail=(SIM,)),
                   Access.of(known=(LAB,)),
                   Access.full(contexts)):
        reachable = {q.text for q in accessible_pool(SURFACES, access)(
            space, contexts, universe)}
        assert reachable <= everything


def test_an_unreachable_question_can_be_strictly_cheaper(report):
    run = _run(report, "HIGH_COST_BUT_ASKABLE")
    assert run["dominated_cost_accepted"]
    assert run["asked_costs"] == [(1, 2, 1)]
    assert run["unavailable_costs"] == [(1, 2, 0)]


def test_the_cheaper_one_is_preferred_as_soon_as_it_is_reachable(report):
    run = _run(report, "HIGH_COST_BUT_ASKABLE")
    assert run["kinds"] == [UNARY]
    assert run["full_access_kinds"] == [CONTEXT]
    assert report["metrics"]["cheaper_preferred_when_reachable"]


# -- safety ----------------------------------------------------------------

def test_no_unaskable_question_is_ever_asked(report):
    assert report["metrics"]["unaskable_question_rate"] == 0.0
    for run in report["runs"]:
        assert run["unaskable_asked"] == 0


def test_nothing_resolves_to_the_wrong_world(report):
    assert report["metrics"]["false_joint_resolution_rate"] == 0.0
    for run in report["runs"]:
        assert not run["wrong"]


def test_a_blocked_state_asks_nothing(report):
    run = _run(report, "INSUFFICIENT_ACCESS")
    assert run["state"] == BLOCKED
    assert run["turns"] == 0 and run["available"] == 0
    assert run["informative"] > 0


# -- the three access conditions -------------------------------------------

def test_full_access_finishes_in_one_question(report):
    run = _run(report, "ORACLE_ACCESS")
    assert run["state"] == RESOLVED and run["turns"] == 1


def test_a_detour_finishes_it_when_the_best_question_is_out_of_reach(report):
    run = _run(report, "INDIRECT_ACCESS")
    assert run["state"] == RESOLVED and run["turns"] == 2
    assert set(run["kinds"]) == {CONTEXT}
    assert report["metrics"]["access_cost_of_detour"] == 1


def test_the_detour_uses_only_the_episodes_it_may_refer_to():
    space, contexts, universe = _setup()
    access = Access.of(known=contexts, detail=(SIM, PILOT))
    session = run_accessible(space, contexts, universe, SURFACES, THREE_TRUTH,
                             access)
    assert session.resolved and not session.wrong
    for question in session.asked:
        assert askable(question, contexts, access)


# -- BLOCKED is not IMPOSSIBLE ---------------------------------------------

def test_blocked_is_a_claim_that_access_would_finish_it(report):
    assert report["metrics"]["blocked_prediction_accuracy"] == 1.0
    for run in report["runs"]:
        if run["state"] == BLOCKED:
            assert run["unlockable"] and run["full_access_resolves"]


def test_impossible_is_a_claim_that_access_would_not(report):
    assert report["metrics"]["impossible_prediction_accuracy"] == 1.0
    run = _run(report, "NOTHING_TO_ASK")
    assert run["state"] == IMPOSSIBLE
    assert run["informative"] == 0 and not run["full_access_resolves"]


def test_opening_an_episode_turns_blocked_into_resolved(report):
    unlock = report["unlock"]
    assert unlock["before"] == BLOCKED
    assert unlock["after_resolved"] and not unlock["after_wrong"]
    assert unlock["after_turns"] == 1
    assert report["metrics"]["blocked_then_resolved"]


def test_an_impossible_world_is_not_unlocked_by_opening_everything(report):
    stays = report["impossible_stays"]
    assert stays["state"] == IMPOSSIBLE and not stays["resolved_with_full_access"]
    assert report["metrics"]["impossible_stays_impossible"]


# -- the predicate ---------------------------------------------------------

def test_a_question_naming_every_episode_needs_every_episode():
    space, contexts, universe = _setup()
    everything = joint_questions(space, contexts, universe, SURFACES)
    unnamed = [q for q in everything
               if q.kind == CONTEXT and not named_contexts(q, contexts)]
    for question in unnamed:
        assert not askable(question, contexts, Access.of(known=(SIM,)))
        assert askable(question, contexts, Access.full(contexts))


def test_opening_grants_both_levels():
    access = Access.of(known=(LAB,))
    assert LAB in access.known and LAB not in access.detail
    opened = access.opening(LAB)
    assert LAB in opened.known and LAB in opened.detail


def test_detail_access_implies_referability():
    access = Access.of(known=(), detail=(LAB,))
    assert access.known == frozenset({LAB})


# -- the planner is unchanged ---------------------------------------------

def test_full_access_reproduces_l839s_own_choice():
    space, contexts, universe = _setup()
    theirs = choose_joint(space, contexts, universe, SURFACES)
    session = run_accessible(space, contexts, universe, SURFACES, THREE_TRUTH,
                             Access.full(contexts))
    assert session.asked[0].text == theirs.questions[0].text


def test_every_family_lands_where_it_claims(report):
    expected = {f.name: (f.expect_state, f.expect_turns) for f in build_families()}
    for run in report["runs"]:
        assert (run["state"], run["turns"]) == expected[run["name"]]
    assert report["metrics"]["diagnosis_accuracy"] == 1.0


def test_unavailable_questions_are_kept_for_the_comparison():
    joint = two_world()
    space, contexts = worlds(joint), joint.contexts
    universe = base_universe()
    blocked_out = cheapest_unavailable(space, contexts, universe, SURFACES,
                                       Access.of(known=(LAB,), detail=(LAB,)))
    assert blocked_out
    assert all(q.kind == CONTEXT for q in blocked_out)


def test_a_contradiction_is_still_a_contradiction_not_a_block():
    from coding_world_benchmark.contextual_joint_l838_experiment import (
        ContextualJoint,
    )
    from coding_world_benchmark.nway_semantics_l837_experiment import (
        CONTRADICTION,
    )
    universe = base_universe()
    joint = ContextualJoint(SURFACES, ((SIM, frozenset()),))
    space, contexts = worlds(joint), joint.contexts
    found = diagnose(space, contexts, universe, SURFACES, Access.full(contexts))
    assert found.state == CONTRADICTION
