import pytest

from coding_world_benchmark.behavior_monitor_l87_experiment import BehaviorEvent
from coding_world_benchmark.conversation_memory_l73_experiment import ConversationKnowledgeMemory
from coding_world_benchmark.episode_relation_l817_experiment import (
    RelationKind,
    build_relation_graph,
    parse_relation,
    resolve_reference_relational,
)
from coding_world_benchmark.episodic_memory_l813_experiment import EpisodicMemory
from coding_world_benchmark.goal_execution_loop_l812_experiment import run_goal_loop, trace_with
from coding_world_benchmark.holdout_relation_l8171_experiment import run_holdout as run_relation_holdout
from coding_world_benchmark.holdout_resolution_l8151_experiment import run_holdout as run_l8151
from coding_world_benchmark.structural_resolver_l815_experiment import resolve_reference_structural


class _NullPath:
    def is_file(self) -> bool:
        return False


_MATERIALS = ("酸化チタン", "窒化ガリウム", "炭化ケイ素", "硫化モリブデン", "ホウ化ジルコニウム")


def _session(count=5, revise_at=None):
    store = ConversationKnowledgeMemory(_NullPath())
    store.facts = []
    episodic = EpisodicMemory(store)
    for index in range(count):
        if revise_at is not None and index == revise_at:
            request = f"r{index:03d}: m_000.py を修正して"
            response = "m_000.py を修正しました。結論: 傾向は変わらない。"
        else:
            request = f"r{index:03d}: {_MATERIALS[index]}の測定コードを作成して"
            response = f"m_{index:03d}.py を作成しました。結論: 値は {index} だった。"

        def executor(_r, _d, response=response):
            return response, trace_with(BehaviorEvent.FILE_WRITTEN)

        episodic.record(run_goal_loop(request, executor))
    return episodic


@pytest.fixture(scope="module")
def cold():
    return run_relation_holdout(40)


def test_relation_phrases_are_matched_longest_first():
    assert parse_relation("その前の実験を続けて")[0] is RelationKind.PRECEDING
    assert parse_relation("Xの後にやった方を続けて")[0] is RelationKind.FOLLOWING
    assert parse_relation("最初にやったやつを続けて")[0] is RelationKind.FIRST
    assert parse_relation("ただの参照を続けて")[0] is None


def test_the_two_l8151_failures_are_fixed():
    episodic = _session(5)

    previous = resolve_reference_relational(episodic, "その前の実験を続けて")
    following = resolve_reference_relational(episodic, "炭化ケイ素の後にやった方を続けて")

    assert previous.episode_id == "ep-004"
    assert previous.episode_id != episodic.episodes[-1].episode_id
    assert following.episode_id == "ep-004"


def test_an_unparseable_relation_cannot_reach_the_bare_rule():
    # The structural fix, asserted directly rather than via a measured rate:
    # without it 「二つ前の」 leaves no content token, 「前の」 reads as a plain
    # continuation marker, and the latest episode comes back.
    episodic = _session(5)

    result = resolve_reference_relational(episodic, "二つ前の実験を続けて")

    assert result.decision == "CLARIFY"
    assert result.episode_id is None


def test_a_bare_reference_still_means_the_latest_episode():
    episodic = _session(5)

    assert resolve_reference_relational(episodic, "さっきの続きをやって").episode_id == "ep-005"


def test_relation_edges_are_derived_from_the_episodes():
    graph = build_relation_graph(_session(5, revise_at=3))

    assert graph.targets("ep-001", "followed_by") == ("ep-002",)
    assert graph.targets("ep-001", "revises") == ("ep-004",)


def test_revision_is_followed_by_edge_not_by_adjacency():
    episodic = _session(5, revise_at=3)

    result = resolve_reference_relational(episodic, "酸化チタンの修正版を続けて")

    assert result.episode_id == "ep-004"


def test_ambiguous_and_missing_anchors_clarify():
    episodic = _session(5)

    missing = resolve_reference_relational(episodic, "m_999.py の後にやった方を続けて")

    assert missing.decision == "CLARIFY"
    assert missing.episode_id is None


def test_relation_resolver_removes_the_l8151_corruption():
    before = run_l8151(50, resolver=resolve_reference_structural)["metrics"]
    after = run_l8151(50, resolver=resolve_reference_relational)["metrics"]

    assert before["wrong_episode_reuse_rate"] > 0.0
    assert after["wrong_episode_reuse_rate"] == 0.0
    assert after["reference_resolution_accuracy"] > before["reference_resolution_accuracy"]


def test_covered_relations_transfer_to_a_third_vocabulary(cold):
    per_kind = cold["per_kind_accuracy"]

    assert per_kind["COVERED_PRECEDING"] == 1.0
    assert per_kind["COVERED_FIRST"] == 1.0
    assert per_kind["COVERED_FOLLOWING"] == 1.0
    assert per_kind["COVERED_REVISION"] == 1.0
    assert cold["metrics"]["covered_relation_accuracy"] == 1.0


def test_the_phrase_table_turns_out_to_be_load_bearing_for_safety(cold):
    """Residual defect from L8.17.1, asserted so a later fix must acknowledge it.

    The design assumed an unrecognised relation phrase was merely a missed
    capability, because it falls through to L8.15 and L8.15 clarifies rather
    than guesses.  That holds only when the query names no anchor.  With one --
    「Xの直後のを続けて」, where 直後の is absent from the table -- L8.15 grounds X
    and hands back X itself, reproducing the off-by-one this layer exists to
    remove.  So `wrong_relation_reuse_rate` is 0.067 on the cold hold-out, not
    zero, and every wrong answer comes from that single path.

    Not repaired here: the defect was found on a cold hold-out, and fixing it
    against those probes would turn them into development data.  Whoever closes
    it in L8.18 updates this test against a fresh hold-out.
    """
    metrics = cold["metrics"]
    wrong = [record for record in cold["records"] if record["wrong"]]

    assert metrics["wrong_relation_reuse_rate"] > 0.0
    assert {record["kind"] for record in wrong} == {"UNCOVERED_ANCHORED"}
    assert metrics["new_task_contamination_rate"] == 0.0


def test_unanchored_unknown_relations_still_fail_safely(cold):
    unsafe = [
        record
        for record in cold["records"]
        if record["kind"] in {"UNCOVERED_MULTISTEP", "UNCOVERED_SYNONYM"} and record["wrong"]
    ]

    assert unsafe == []


def test_relation_holdout_is_deterministic():
    assert run_relation_holdout(20)["metrics"] == run_relation_holdout(20)["metrics"]
