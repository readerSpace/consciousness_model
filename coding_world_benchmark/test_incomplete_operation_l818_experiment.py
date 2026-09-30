import pytest

from coding_world_benchmark.behavior_monitor_l87_experiment import BehaviorEvent
from coding_world_benchmark.context_isolation_l814_experiment import run_size
from coding_world_benchmark.conversation_memory_l73_experiment import ConversationKnowledgeMemory
from coding_world_benchmark.episode_relation_l817_experiment import resolve_reference_relational
from coding_world_benchmark.episodic_memory_l813_experiment import EpisodicMemory
from coding_world_benchmark.goal_execution_loop_l812_experiment import run_goal_loop, trace_with
from coding_world_benchmark.holdout_incomplete_l8181_experiment import run_holdout
from coding_world_benchmark.holdout_relation_l8171_experiment import run_holdout as run_relation_holdout
from coding_world_benchmark.incomplete_operation_l818_experiment import (
    SlotKind,
    execute,
    has_relation_cue,
    parse_query_ir,
    resolve_reference_gated,
)

_OBJECTS = ("系外惑星", "超新星残骸", "銀河団", "重力レンズ", "パルサー")


class _NullPath:
    def is_file(self) -> bool:
        return False


def _session(count=5):
    store = ConversationKnowledgeMemory(_NullPath())
    store.facts = []
    episodic = EpisodicMemory(store)
    for index in range(count):
        request = f"obs{index:03d}: {_OBJECTS[index]}の前処理コードを作成して"
        response = f"obs_{index:03d}.py を作成しました。結論: 較正誤差に敏感。"

        def executor(_r, _d, response=response):
            return response, trace_with(BehaviorEvent.FILE_WRITTEN)

        episodic.record(run_goal_loop(request, executor))
    return episodic


@pytest.fixture(scope="module")
def cold():
    return run_holdout(30, gated=True)


def test_a_cue_inside_a_compound_is_not_a_relation():
    assert has_relation_cue("その前の実験を続けて")
    assert has_relation_cue("二つ前の実験を続けて")
    assert has_relation_cue("Xの直後のを続けて")

    assert not has_relation_cue("前処理のコードを修正して")
    assert not has_relation_cue("後処理を直して")
    assert not has_relation_cue("次元削減の実装を見せて")
    assert not has_relation_cue("事前チェックを追加して")


def test_an_evoked_but_unresolved_slot_blocks_even_with_a_perfect_anchor():
    """The invariant, and the paradox it removes.

    銀河団 identifies exactly one episode, so the anchor slot is filled with
    high confidence.  Under the old design that made the wrong answer *more*
    likely, because the anchor was returned in place of the relation's result.
    The gate is not conditioned on the anchor for exactly this reason.
    """
    episodic = _session()
    ir = parse_query_ir(episodic, "銀河団の前処理の直後のを続けて")

    assert ir.slot(SlotKind.ANCHOR).resolved is not None
    assert ir.slot(SlotKind.RELATION).evoked
    assert ir.slot(SlotKind.RELATION).resolved is None
    assert SlotKind.RELATION in ir.blocking

    outcome = execute(episodic, "銀河団の前処理の直後のを続けて")
    assert outcome.decision == "ABSTAIN"
    assert outcome.episode_id is None


def test_a_request_that_evokes_no_relation_is_untouched():
    episodic = _session()
    ir = parse_query_ir(episodic, "銀河団の前処理を続けて")

    assert not ir.slot(SlotKind.RELATION).evoked
    assert ir.blocking == ()
    assert execute(episodic, "銀河団の前処理を続けて").episode_id == "ep-003"


def test_bare_reference_still_resolves():
    assert execute(_session(), "さっきの続きをやって").episode_id == "ep-005"


def test_known_relation_still_resolves():
    assert execute(_session(), "その前の実験を続けて").episode_id == "ep-004"


def test_an_unimplemented_operation_abstains_rather_than_looking_up():
    episodic = _session()

    outcome = execute(episodic, "一番多く使った観測手法は？")

    assert outcome.decision == "ABSTAIN"
    assert SlotKind.OPERATION in outcome.blocking


def test_an_implemented_operation_answers():
    outcome = execute(_session(), "系外惑星の実験は何件やった？")

    assert outcome.decision == "ANSWER"
    assert outcome.value == 1


def test_preregistered_predictions_hold_on_the_cold_holdout(cold):
    per_family = cold["per_family_accuracy"]

    assert per_family["KNOWN_RELATION"] == 1.0
    assert per_family["UNKNOWN_RELATION_ANCHORED"] == 1.0
    assert per_family["UNKNOWN_RELATION_BARE"] == 1.0
    assert per_family["BARE_REFERENCE"] == 1.0
    assert per_family["NON_RELATION_LOOKALIKE"] == 1.0
    assert per_family["UNKNOWN_OPERATION"] == 1.0
    assert per_family["NEW_TASK"] == 1.0


def test_both_safety_numbers_are_zero_at_once(cold):
    # Either alone is trivially satisfiable -- refuse everything, or guess
    # everything.  The pair is the claim.
    metrics = cold["metrics"]

    assert metrics["wrong_reuse_rate"] == 0.0
    assert metrics["unnecessary_abstention_rate"] == 0.0
    assert metrics["uncovered_abstention_rate"] == 1.0
    assert metrics["covered_relation_accuracy"] == 1.0


def test_the_guard_costs_no_capability_on_lookalike_text(cold):
    lookalikes = [r for r in cold["records"] if r["family"] == "NON_RELATION_LOOKALIKE"]

    assert lookalikes
    assert not any(record["abstained"] for record in lookalikes)


def test_the_l8171_residual_is_closed_without_losing_coverage():
    before = run_relation_holdout(40, resolver=resolve_reference_relational)["metrics"]
    after = run_relation_holdout(40, resolver=resolve_reference_gated)["metrics"]

    assert before["wrong_relation_reuse_rate"] > 0.0
    assert after["wrong_relation_reuse_rate"] == 0.0
    assert after["covered_relation_accuracy"] == before["covered_relation_accuracy"] == 1.0


def test_no_regression_on_the_earlier_benchmarks():
    metrics = run_size(100, resolver=resolve_reference_gated)["metrics"]

    assert metrics["reference_resolution_accuracy"] == 1.0
    assert metrics["wrong_episode_reuse_rate"] == 0.0
    assert metrics["stale_intent_leak_rate"] == 0.0
    assert metrics["new_task_contamination_rate"] == 0.0


def test_holdout_is_deterministic():
    assert run_holdout(20)["metrics"] == run_holdout(20)["metrics"]
