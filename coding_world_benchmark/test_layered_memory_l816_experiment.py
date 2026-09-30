import pytest

from coding_world_benchmark.conversation_memory_l73_experiment import _compact_lines
from coding_world_benchmark.layered_memory_l816_experiment import (
    CONFIGURATIONS,
    LayerConfig,
    build_conversation,
    build_probes,
    reference_part,
    retrieve,
    run_experiment,
)


@pytest.fixture(scope="module")
def report():
    return run_experiment()


def _by_name(report, name):
    return next(item for item in report["results"] if item["config"] == name)


def test_compression_really_discards_the_parameters():
    # The premise of the whole experiment, checked against L7.3 itself rather
    # than assumed: only marker-carrying lines survive, so the parameters are
    # not anywhere in the long-term state to be found later.
    memory = build_conversation(4)
    turn = memory.turns[0]

    kept = _compact_lines(turn.response)

    assert turn.conclusion_line in kept
    assert turn.parameter_line not in kept
    assert all(turn.parameter_line not in line for line in kept)


def test_compressed_state_answers_semantics_but_not_wording(report):
    compressed = _by_name(report, "compressed_only")

    assert compressed["per_kind"]["COMPRESSED_FACT"] == "hit"
    assert compressed["per_kind"]["VERBATIM_OLD"] == "misleading"
    assert compressed["per_kind"]["RECENT_VERBATIM"] == "misleading"


def test_a_short_window_loses_everything_older_than_itself(report):
    window = _by_name(report, "short_term_only")

    assert window["per_kind"]["RECENT_VERBATIM"] == "hit"
    assert window["per_kind"]["VERBATIM_OLD"] == "misleading"
    assert window["per_kind"]["COMPRESSED_FACT"] == "misleading"


def test_the_three_layers_together_answer_every_probe(report):
    layered = _by_name(report, "layered")

    assert layered["evidence_recall"] == 1.0
    assert layered["misleading_context_rate"] == 0.0


def test_layering_beats_the_full_transcript_on_both_axes(report):
    layered = _by_name(report, "layered")
    full = _by_name(report, "full_transcript")

    assert layered["evidence_recall"] >= full["evidence_recall"]
    assert layered["mean_context_chars"] < full["mean_context_chars"] / 10


def test_only_the_grounded_configuration_abstains(report):
    abstaining = {
        item["config"] for item in report["results"] if item["abstention_accuracy"] == 1.0
    }

    assert abstaining == {"layered"}


def test_verbatim_layer_abstains_instead_of_paraphrasing():
    memory = build_conversation(8)
    config = LayerConfig("layered", 2, True, True)

    result = retrieve(memory, config, "超伝導ギャップの測定値をそのまま出して")

    assert result.can_answer is False
    assert result.evidence == ""
    assert "no episode could be identified" in result.reason


def test_projection_words_are_split_off_before_resolving():
    # Without the split, パラメータ and 原文 are treated as episode names, cover
    # nothing, and the reference resolves to nothing at all.
    reference = reference_part("さっきの実験のパラメータを原文のまま出して")

    assert "パラメータ" not in reference
    assert "原文" not in reference
    assert "さっき" in reference


def test_stripping_projection_also_fixes_the_sonomama_substring():
    # 「そのまま」 contains 「その」, so L8.13's substring marker test fires on a
    # request that refers to nothing.  Stripping the projection first removes it.
    reference = reference_part("超伝導ギャップの測定値をそのまま出して")

    assert "その" not in reference


def test_aggregate_counts_conjunctively():
    memory = build_conversation(24)
    config = LayerConfig("layered", 2, True, True)

    result = retrieve(memory, config, "イジング模型の実験は何件やった？")

    # 模型 alone appears in all 24 requests; only the conjunction is 5.
    assert "該当 5 件" in result.evidence
    assert "該当 24 件" not in result.evidence


def test_every_configuration_is_measured_on_every_probe(report):
    kinds = {probe.kind for probe in report["probes"]}

    assert len(report["results"]) == len(CONFIGURATIONS)
    for result in report["results"]:
        assert set(result["per_kind"]) == kinds


def test_compression_ratio_is_reported_against_the_real_transcript(report):
    assert 0.0 < report["compression_ratio"] < 1.0
    assert report["compressed_chars"] < report["transcript_chars"]


def test_results_are_deterministic():
    assert _by_name(run_experiment(12), "layered") == _by_name(run_experiment(12), "layered")
