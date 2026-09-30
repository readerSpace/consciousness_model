import inspect

import pytest

from coding_world_benchmark.contextual_polysemy_l835_experiment import (
    ContextualObservation,
)
from coding_world_benchmark.cost_aware_dialogue_l836_experiment import (
    ASK, IMPOSSIBLE, MEANING, MONOSEMY, MULTIPLE_OPTIMAL, NOISE, POLYSEMY, RESOLVED,
    REVISION, STRUCTURE, Cost, Hypothesis, all_questions, choose, hypotheses,
    preserves_solvability, run_dialogue, total_cost,
)
from coding_world_benchmark.cross_context_identification_l829_experiment import (
    Observation, universe_from,
)
from coding_world_benchmark.holdout_binding_l8211_experiment import build_holdout
from coding_world_benchmark.holdout_cost_dialogue_l8361_experiment import (
    CONTEXTS, MEANINGS, build_families, run_experiment,
)
from coding_world_benchmark.holdout_lexicon_l8281_experiment import widen_store
from coding_world_benchmark.typed_operation_l819_experiment import ValueType


@pytest.fixture(scope="module")
def universe():
    return universe_from(widen_store(build_holdout(24)[0]))


@pytest.fixture(scope="module")
def report():
    return run_experiment()


def equals(name):
    return Observation("EQUALS", name)


def evidence(*items):
    return tuple(ContextualObservation(o, c, t) for o, c, t in items)


AMBIGUOUS_EVIDENCE = (
    (equals("success_rate"), "sim", 0), (equals("success_rate"), "sim", 1),
    (equals("energy_error"), "lab", 2), (equals("energy_error"), "lab", 3))


# --------------------------------------------------------------------------
# no lambda, and cost is not a number
# --------------------------------------------------------------------------

def test_there_is_no_weight_anywhere():
    """``gain - lambda * cost`` would smuggle in a constant nobody can justify."""
    import coding_world_benchmark.cost_aware_dialogue_l836_experiment as module

    # Code only: the docstrings say "no weights anywhere", which would trip a
    # naive substring check on the very claim it is making.
    def code(obj):
        lines = inspect.getsource(obj).splitlines()
        kept, inside = [], False
        for line in lines:
            ticks = line.count('"""')
            if ticks:
                inside = not inside if ticks == 1 else inside
                continue
            if not inside:
                kept.append(line.split("#")[0])
        return "\n".join(kept)

    source = code(module.choose) + code(module.Cost)
    for smell in ("lambda_", "weight", "0.5 *", "alpha", "* cost", "**"):
        assert smell not in source, smell


def test_cost_is_compared_by_pareto_dominance():
    cheap = Cost(1, 2, False)
    context_needing = Cost(1, 2, True)
    wide = Cost(1, 3, False)
    assert cheap.dominates(context_needing)
    assert cheap.dominates(wide)
    # A three-way answered in one turn and a yes/no are genuinely incomparable
    # against two turns; neither dominates.
    assert not wide.dominates(Cost(2, 2, False))
    assert not Cost(2, 2, False).dominates(wide)
    assert not cheap.dominates(cheap)


# --------------------------------------------------------------------------
# the lexicographic order
# --------------------------------------------------------------------------

def test_capability_is_a_veto_not_an_optimisation(universe):
    """A question that lands somewhere permanently stuck is inadmissible.

    No amount of cheapness redeems spending a turn to arrive nowhere, so this is
    checked before ambiguity and before cost.
    """
    space = hypotheses(evidence(*AMBIGUOUS_EVIDENCE), universe)
    for question in all_questions(space, CONTEXTS, MEANINGS):
        for _, block in question.outcomes:
            if len(block) > 1:
                assert all_questions(block, CONTEXTS, MEANINGS) or \
                       not preserves_solvability(question, CONTEXTS, MEANINGS)


def test_ambiguity_is_consulted_before_cost(report):
    """The three-way choice has the bigger answer space and still wins.

    It finishes in one turn where a yes/no cannot, so it never reaches the cost
    comparison at all.
    """
    runs = {run["name"]: run for run in report["runs"]}
    expensive = runs["EXPENSIVE_BUT_NECESSARY"]
    assert expensive["state"] == ASK
    assert expensive["chosen_cost"][1] == 3  # answer space of three
    assert expensive["cheaper_available"] == 0


def test_an_equally_informative_question_that_costs_more_is_dropped(report):
    """Same information, plus a context recall: Pareto-dominated."""
    assert report["metrics"]["dominated_questions_dropped"] >= 1
    runs = {run["name"]: run for run in report["runs"]}
    assert runs["STRUCTURE_ONLY"]["chosen_cost"] == (1, 2, 0)


# --------------------------------------------------------------------------
# the question about the model
# --------------------------------------------------------------------------

def test_no_meaning_question_is_even_informative_there(universe):
    """The layer's reason to exist, and it is a capability claim, not a price one.

    POLYSEMY and REVISION predict the same meaning for every observed context, so
    asking what the word means cannot separate them at all.
    """
    space = hypotheses(evidence(*AMBIGUOUS_EVIDENCE), universe)
    assert {h.model for h in space} == {POLYSEMY, REVISION}
    predictions = {h.predictions for h in space}
    assert len(predictions) == 1  # identical predictions everywhere
    meaning_questions = [q for q in all_questions(space, CONTEXTS, MEANINGS)
                         if q.kind == MEANING]
    assert meaning_questions == []


def test_the_structural_question_resolves_it(universe):
    space = hypotheses(evidence(*AMBIGUOUS_EVIDENCE), universe)
    truth = next(h for h in space if h.model == POLYSEMY)
    settled, asked = run_dialogue(space, CONTEXTS, MEANINGS, truth)
    assert len(settled) == 1 and next(iter(settled)) == truth
    assert len(asked) == 1 and asked[0].kind == STRUCTURE


def test_a_tie_is_reported_not_broken(report):
    runs = {run["name"]: run for run in report["runs"]}
    structure = runs["STRUCTURE_ONLY"]
    assert structure["state"] == MULTIPLE_OPTIMAL
    assert structure["options"] == 2


# --------------------------------------------------------------------------
# the hypothesis space
# --------------------------------------------------------------------------

def test_a_discarding_account_loses_to_one_that_explains_everything(universe):
    """L8.30's rule, applied to the space rather than to one belief.

    Keeping a NOISE reading alongside a conditioning that fits would let a worse
    account dilute the value of every question.
    """
    space = hypotheses(evidence(*AMBIGUOUS_EVIDENCE), universe)
    assert not any(h.model == NOISE for h in space)
    noisy = hypotheses(evidence(
        (equals("success_rate"), "sim", 0), (equals("success_rate"), "lab", 1),
        (equals("energy_error"), "sim", 2), (equals("success_rate"), "lab", 3)), universe)
    assert {h.model for h in noisy} == {NOISE}


def test_a_revision_predicts_by_when_not_by_which_context(universe):
    """Why the predictions are computed from the evidence rather than the labels."""
    space = hypotheses(evidence(*AMBIGUOUS_EVIDENCE), universe)
    revision = next(h for h in space if h.model == REVISION)
    assert dict(revision.predictions) == {"sim": "success_rate", "lab": "energy_error"}


# --------------------------------------------------------------------------
# the hold-out
# --------------------------------------------------------------------------

def test_the_invariants(report):
    found = report["metrics"]
    assert found["false_resolution_rate"] == 0.0
    assert found["avoidable_interaction_rate"] == 0.0
    assert found["excess_cost_given_equal_resolution"] == 0.0


def test_nothing_is_asked_when_the_evidence_already_decided(report):
    runs = {run["name"]: run for run in report["runs"]}
    for name in ("RESOLVED", "POLYSEMY_SETTLED"):
        assert runs[name]["state"] == RESOLVED
        assert runs[name]["chosen"] is None
        assert runs[name]["asked"] == 0


def test_every_family_behaves_as_specified(report):
    assert report["metrics"]["state_accuracy"] == 1.0
    assert report["metrics"]["question_kind_accuracy"] == 1.0
    assert len(build_families()) == 4


def test_an_unseparable_space_is_reported_not_spent_on(report):
    found = report["impossible"]
    assert found["state"] == IMPOSSIBLE
    assert found["questions"] == 0
    assert found["reported_rather_than_spent"]


def test_total_cost_keeps_the_vector(universe):
    space = hypotheses(evidence(*AMBIGUOUS_EVIDENCE), universe)
    truth = next(h for h in space if h.model == REVISION)
    _, asked = run_dialogue(space, CONTEXTS, MEANINGS, truth)
    assert total_cost(asked).vector == (1, 2, 0)
    assert total_cost(()).vector == (0, 0, 0)


def test_the_report_formats(report):
    from coding_world_benchmark.holdout_cost_dialogue_l8361_experiment import (
        format_experiment,
    )

    text = format_experiment(report)
    assert "capability claim rather than a price one" in text
    assert "Pareto-dominated" in text
