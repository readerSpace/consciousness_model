import pytest

from coding_world_benchmark.contextual_polysemy_l835_experiment import (
    AMBIGUOUS, DISPUTED, MIN_BLOCK, MONOSEMY, NOISE, POLYSEMOUS, POLYSEMY, REVISION,
    SINGLE, UNEXPLAINED, UNIDENTIFIABLE, ContextualLexicon, ContextualObservation,
    explain,
)
from coding_world_benchmark.cross_context_identification_l829_experiment import (
    Observation, available_probes, universe_from,
)
from coding_world_benchmark.holdout_binding_l8211_experiment import build_holdout
from coding_world_benchmark.holdout_lexicon_l8281_experiment import widen_store
from coding_world_benchmark.holdout_polysemy_l8351_experiment import (
    SURFACE, build_families, run_experiment,
)
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


# --------------------------------------------------------------------------
# a contradiction is not an answer
# --------------------------------------------------------------------------

def test_the_same_contradiction_has_three_readings(universe):
    """Noise, revision and polysemy look identical when the disagreement lands.

    What separates them is which variable makes it go away, and that is read off
    the data rather than chosen.
    """
    rate, energy = equals("success_rate"), equals("energy_error")
    noisy = explain(evidence((rate, "sim", 0), (rate, "lab", 1),
                             (energy, "sim", 2), (rate, "lab", 3)), universe)
    revised = explain(evidence((rate, "sim", 0), (rate, "lab", 1),
                               (energy, "sim", 2), (energy, "lab", 3)), universe)
    polysemous = explain(evidence((rate, "sim", 0), (energy, "lab", 1),
                                  (rate, "sim", 2), (energy, "lab", 3)), universe)
    assert (noisy.kind, revised.kind, polysemous.kind) == (NOISE, REVISION, POLYSEMY)


def test_nothing_is_split_when_nothing_disagrees(universe):
    found = explain(evidence((equals("success_rate"), "sim", 0),
                             (equals("success_rate"), "lab", 1)), universe)
    assert found.kind == MONOSEMY and found.state == SINGLE
    assert not found.conditioned


# --------------------------------------------------------------------------
# the two rules that stop it being a sense-splitting machine
# --------------------------------------------------------------------------

def test_a_block_of_one_is_the_noise_hypothesis_wearing_a_hat(universe):
    """One is not a pattern, and this needs no threshold to say so.

    A partition whose block holds a single observation explains exactly what
    "that observation was noise" explains, with one more parameter.
    """
    assert MIN_BLOCK == 2
    rate, energy = equals("success_rate"), equals("energy_error")
    found = explain(evidence((rate, "sim", 0), (rate, "sim", 1),
                             (rate, "sim", 2), (energy, "odd", 3)), universe)
    assert found.kind == NOISE
    assert found.senses == {"*": "success_rate"}


def test_an_explanation_both_variables_fit_is_no_explanation(universe):
    """The contexts happen to be time-ordered, so the data cannot choose."""
    rate, energy = equals("success_rate"), equals("energy_error")
    found = explain(evidence((rate, "sim", 0), (rate, "sim", 1),
                             (energy, "lab", 2), (energy, "lab", 3)), universe)
    assert found.kind == AMBIGUOUS
    assert found.state == DISPUTED
    assert found.senses == {}


def test_a_split_is_refused_when_the_senses_are_not_separable(universe):
    """L8.31 again: splitting an unidentifiable pair invents a distinction."""
    rate, energy = equals("success_rate"), equals("energy_error")
    items = evidence((rate, "sim", 0), (energy, "lab", 1),
                     (rate, "sim", 2), (energy, "lab", 3))
    narrow = explain(items, universe, (Observation("TYPE", ValueType.NUMBER),))
    assert narrow.kind == UNIDENTIFIABLE
    full = explain(items, universe, available_probes(universe))
    assert full.kind == POLYSEMY


def test_a_conditioning_whose_blocks_agree_explains_nothing(universe):
    """Two contexts, same meaning: the variable is not doing any work."""
    rate = equals("success_rate")
    found = explain(evidence((rate, "sim", 0), (rate, "sim", 1),
                             (rate, "lab", 2), (rate, "lab", 3)), universe)
    assert found.kind == MONOSEMY


# --------------------------------------------------------------------------
# the lexicon
# --------------------------------------------------------------------------

def test_a_conditioned_entry_answers_per_context(universe):
    lexicon = ContextualLexicon(universe)
    for observation, context, time in (
            (equals("success_rate"), "sim", 0), (equals("energy_error"), "lab", 1),
            (equals("success_rate"), "sim", 2), (equals("energy_error"), "lab", 3)):
        lexicon.observe(SURFACE, observation, context, time)
    assert lexicon.meaning(SURFACE, "sim") == "success_rate"
    assert lexicon.meaning(SURFACE, "lab") == "energy_error"
    assert lexicon.meaning(SURFACE, None) is None  # a sense needs a context
    assert lexicon.meaning(SURFACE, "unseen") is None


def test_a_revised_entry_answers_with_the_current_meaning(universe):
    lexicon = ContextualLexicon(universe)
    for observation, context, time in (
            (equals("success_rate"), "sim", 0), (equals("success_rate"), "lab", 1),
            (equals("energy_error"), "sim", 2), (equals("energy_error"), "lab", 3)):
        lexicon.observe(SURFACE, observation, context, time)
    assert lexicon.explanations[SURFACE].kind == REVISION
    assert lexicon.meaning(SURFACE, "sim") == "energy_error"


def test_a_disputed_entry_answers_nothing(universe):
    lexicon = ContextualLexicon(universe)
    for observation, context, time in (
            (equals("success_rate"), "sim", 0), (equals("success_rate"), "sim", 1),
            (equals("energy_error"), "lab", 2), (equals("energy_error"), "lab", 3)):
        lexicon.observe(SURFACE, observation, context, time)
    assert lexicon.meaning(SURFACE, "sim") is None
    assert lexicon.lookup(f"…{SURFACE}を…", "sim") is None


# --------------------------------------------------------------------------
# the hold-out
# --------------------------------------------------------------------------

def test_the_invariant(report):
    """Adding a sense whenever evidence disagrees reaches perfect accuracy by
    never being wrong about anything, so this is the number that matters."""
    assert report["metrics"]["false_sense_split_rate"] == 0.0
    assert report["metrics"]["wrong_answer_rate"] == 0.0


def test_genuine_polysemy_is_not_missed_either(report):
    assert report["metrics"]["missed_polysemy_rate"] == 0.0
    assert report["metrics"]["context_conditioned_resolution_rate"] == 1.0


def test_revision_and_polysemy_are_told_apart(report):
    assert report["metrics"]["revision_vs_polysemy_accuracy"] == 1.0
    runs = {run["name"]: run for run in report["runs"]}
    assert runs["REVISION"]["kind"] == REVISION
    assert runs["TRUE_POLYSEMY"]["kind"] == POLYSEMY


def test_erasing_context_breaks_exactly_the_one_family_that_needs_it(report):
    """The strongest claim available here.

    If removing the labels merely made things worse, context would be a
    convenience. It makes TRUE_POLYSEMY inexplicable while leaving every other
    family untouched, so it was the variable the meaning depended on.
    """
    runs = {run["name"]: run for run in report["runs"]}
    assert runs["TRUE_POLYSEMY"]["without_context_kind"] == UNEXPLAINED
    assert runs["TRUE_POLYSEMY"]["without_context_resolves"] is False
    for name in ("NOISY_MONOSEMY", "REVISION", "MONOSEMY"):
        assert runs[name]["without_context_kind"] == runs[name]["kind"], name
        assert runs[name]["without_context_resolves"] is True


def test_the_ambiguous_family_is_ambiguous_only_because_context_exists(report):
    """A detail worth keeping: erasing the labels removes the ambiguity, because
    with nothing to compete against, time explains it on its own."""
    runs = {run["name"]: run for run in report["runs"]}
    assert runs["AMBIGUOUS"]["kind"] == AMBIGUOUS
    assert runs["AMBIGUOUS"]["without_context_kind"] == REVISION


def test_every_family_is_explained_correctly(report):
    assert report["metrics"]["explanation_accuracy"] == 1.0
    assert len(build_families()) == 5


def test_the_unidentifiable_check_cuts_both_ways(report):
    found = report["unidentifiable"]
    assert found["refused_when_inseparable"]
    assert found["allowed_when_separable"]


def test_the_report_formats(report):
    from coding_world_benchmark.holdout_polysemy_l8351_experiment import (
        format_experiment,
    )

    text = format_experiment(report)
    assert "false_sense_split_rate" in text and "context erased" in text
