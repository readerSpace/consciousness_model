import pytest

from coding_world_benchmark.argument_binding_l821_experiment import run_bound_query
from coding_world_benchmark.holdout_binding_l8211_experiment import build_holdout
from coding_world_benchmark.holdout_binding_l8211_experiment import build_probes as binding_probes
from coding_world_benchmark.holdout_language_l8221_experiment import run_holdout
from coding_world_benchmark.holdout_program_l8201_experiment import run_holdout as run_program
from coding_world_benchmark.query_algebra_l820_experiment import canonical_render
from coding_world_benchmark.semantic_proposer_l822_experiment import (
    OPERATOR_FORMS,
    PROPOSAL_MARGIN,
    Proposer,
    assert_holdout_is_unseen,
    run_proposed_query,
)


@pytest.fixture(scope="module")
def store():
    return build_holdout(24)[0]


@pytest.fixture(scope="module")
def proposer():
    return Proposer.trained()


@pytest.fixture(scope="module")
def cold():
    return run_holdout(24, proposed=True)


@pytest.fixture(scope="module")
def cold_base():
    return run_holdout(24, proposed=False)


MEAN_BIG = "MEAN(PROJECT(FILTER(episodes, lattice_size == 64x64), runtime_seconds))"


def test_held_out_forms_contain_no_hand_written_cue():
    # Without this the experiment would be measuring the cue tables: a held-out
    # form carrying a cue never reaches the proposer at all.
    assert_holdout_is_unseen()
    assert all(split["holdout"] for split in OPERATOR_FORMS.values())


def test_an_unseen_wording_becomes_a_correct_execution(store, proposer):
    text = "64x64で走らせたケースについて処理時間をならして"

    assert run_bound_query(store, text).decision == "NOT_AN_OPERATION"

    result = run_proposed_query(store, text, proposer)
    assert result.decision == "ANSWER"
    assert result.source == "proposed"
    assert canonical_render(result.expr) == MEAN_BIG


def test_unseen_wording_inside_an_unseen_shape(store, proposer):
    text = "32x32と比べて64x64の処理時間をならしたら大きい？"

    assert run_bound_query(store, text).decision == "ABSTAIN"
    result = run_proposed_query(store, text, proposer)

    assert result.decision == "ANSWER"
    assert canonical_render(result.expr).startswith("COMPARE(MEAN(")


def test_a_proposal_is_not_authority(store, proposer):
    """Scores widen the candidate set; types and uniqueness still decide.

    「ピークをならして」 carries subword evidence for MAX and for MEAN, both of which
    type-check over a Number field, so two well-typed programs survive and the
    request is refused.  A proposer with authority would run whichever scored
    higher -- here that is a coin flip between max and mean.
    """
    text = "64x64のケースの処理時間をピークをならして"
    scored = dict(proposer.operator_model.scores(text))

    assert {"MAX", "MEAN"} <= set(scored)
    assert abs(scored["MAX"] - scored["MEAN"]) < PROPOSAL_MARGIN

    result = run_proposed_query(store, text, proposer)
    assert result.decision == "ABSTAIN"
    assert result.value is None


def test_a_word_with_no_evidence_buys_no_coverage(store, proposer):
    result = run_proposed_query(store, "64x64のケースの処理時間をブリブリして", proposer)

    assert result.decision == "NOT_AN_OPERATION"
    assert "no evidence" in result.reason


def test_the_proposer_only_sees_what_the_cues_could_not_read(store, proposer):
    # Every probe the symbolic pipeline already reads must come back identical,
    # so any measured gain is new coverage rather than a reshuffle.
    for probe in binding_probes(build_holdout(24)[1]):
        base = run_bound_query(store, probe.text)
        if base.decision == "NOT_AN_OPERATION" and base.reason == "no operation in this request":
            continue
        proposed = run_proposed_query(store, probe.text, proposer)
        assert proposed.decision == base.decision
        assert proposed.value == base.value


def test_controls_survive_the_new_layer(store, proposer):
    assert run_proposed_query(store, "さっきの続きをやって", proposer).decision == "NOT_AN_OPERATION"
    gate = run_proposed_query(store, "実行時間の中央値は？", proposer)
    assert gate.decision == "ABSTAIN" and gate.stage == "gate"


def test_coverage_rises_with_no_wrong_and_no_unsafe_execution(cold, cold_base):
    metrics, before = cold["metrics"], cold_base["metrics"]

    assert metrics["unseen_language_execution_rate"] > before["unseen_language_execution_rate"]
    assert metrics["unseen_composition_accuracy"] > before["unseen_composition_accuracy"]
    assert metrics["wrong_answer_rate"] == 0.0
    assert metrics["unsafe_execution_rate"] == 0.0
    assert metrics["refusal_accuracy"] == 1.0


def test_safe_coverage_gain_is_the_single_number(cold, cold_base):
    # correct minus wrong: "refuse everything" and "answer everything" both lose.
    assert cold["metrics"]["safe_coverage_gain"] > cold_base["metrics"]["safe_coverage_gain"]


def test_every_proposed_execution_came_through_the_symbolic_pipeline(cold):
    executed = [r for r in cold["records"] if r["decision"] == "ANSWER"]

    assert executed
    assert not any(r["unsafe"] for r in executed)


def test_no_regression_on_the_program_holdout():
    metrics = run_program(24, bound=True)["metrics"]

    assert metrics["ast_exact_match"] == 1.0
    assert metrics["held_out_shape_accuracy"] == 1.0
    assert metrics["wrong_answer_rate"] == 0.0


def test_results_are_deterministic():
    assert run_holdout(16)["metrics"] == run_holdout(16)["metrics"]
