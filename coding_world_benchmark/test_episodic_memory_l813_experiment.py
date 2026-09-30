import gzip
import json

from coding_world_benchmark.behavior_monitor_l87_experiment import BehaviorEvent
from coding_world_benchmark.conversation_memory_l73_experiment import ConversationKnowledgeMemory
from coding_world_benchmark.episodic_memory_l813_experiment import (
    EpisodicMemory,
    build_session,
    format_episodic_state,
    format_experiment,
    run_experiment,
)
from coding_world_benchmark.goal_execution_loop_l812_experiment import run_goal_loop, trace_with


NEW_WORK = "ナビエストークス方程式のソルバーを作成して"


def _session(tmp_path):
    return build_session(tmp_path)


def _record(episodic, request, response, event=BehaviorEvent.FILE_WRITTEN):
    def executor(_request, _directives):
        return response, trace_with(event)

    return episodic.record(run_goal_loop(request, executor))


def test_new_work_sees_nothing_from_earlier_episodes(tmp_path):
    episodic = _session(tmp_path)

    leaked = episodic.memory.recall(NEW_WORK)
    scoped = episodic.scoped_recall(NEW_WORK)

    # L7.3 on its own matches on an incidental shared word and hands back an
    # unrelated episode's fact.  The episode boundary is what removes it.
    assert leaked, "expected the flat L7.3 pool to leak, otherwise this proves nothing"
    assert scoped == ()
    assert episodic.scoped_context(NEW_WORK) == ""


def test_bare_reference_resolves_to_the_latest_episode(tmp_path):
    episodic = _session(tmp_path)

    resolution = episodic.resolve_reference("それを修正して")

    assert resolution.decision == "CONTINUE_EPISODE"
    assert resolution.episode_id == episodic.episodes[-1].episode_id
    assert episodic.scoped_recall("それを修正して")


def test_named_older_episode_beats_recency(tmp_path):
    episodic = _session(tmp_path)

    resolution = episodic.resolve_reference("さっきのSU2の計算を続けて")

    assert resolution.decision == "CONTINUE_EPISODE"
    assert resolution.episode_id == "ep-002"
    assert resolution.episode_id != episodic.episodes[-1].episode_id
    facts = episodic.scoped_recall("さっきのSU2の計算を続けて")
    assert facts
    assert all(fact.key in set(episodic.episode("ep-002").fact_keys) for fact in facts)


def test_unknown_name_clarifies_instead_of_substituting_the_latest(tmp_path):
    episodic = _session(tmp_path)

    resolution = episodic.resolve_reference("さっきのSU4の計算を続けて")

    assert resolution.decision == "CLARIFY"
    assert resolution.episode_id is None
    assert "no episode matches" in resolution.reason
    assert episodic.scoped_recall("さっきのSU4の計算を続けて") == ()


def test_name_matching_two_episodes_clarifies_with_both_candidates(tmp_path):
    episodic = _session(tmp_path)

    resolution = episodic.resolve_reference("さっきの波動方程式の作業を続けて")

    assert resolution.decision == "CLARIFY"
    assert set(resolution.candidates) == {"ep-001", "ep-004"}


def test_artifact_filename_is_a_reference_without_a_marker(tmp_path):
    episodic = _session(tmp_path)

    resolution = episodic.resolve_reference("lorentz_particle.py を修正して")

    assert resolution.decision == "CONTINUE_EPISODE"
    assert resolution.episode_id == "ep-003"


def test_an_unseen_filename_starts_new_work(tmp_path):
    episodic = _session(tmp_path)

    resolution = episodic.resolve_reference("navier_stokes.py を作成して")

    assert resolution.decision == "NEW_EPISODE"
    assert episodic.scoped_recall("navier_stokes.py を作成して") == ()


def test_generic_noun_alone_never_identifies_an_episode(tmp_path):
    episodic = _session(tmp_path)

    # "計算" appears in ep-002's request but is not a name; on its own it must
    # fall through to the bare-reference rule rather than capture ep-002.
    resolution = episodic.resolve_reference("さっきの計算を続けて")

    assert resolution.episode_id != "ep-002"


def test_episodes_index_facts_rather_than_copying_them(tmp_path):
    episodic = _session(tmp_path)
    stored_keys = {fact.key for fact in episodic.memory.facts}

    for episode in episodic.episodes:
        assert episode.fact_keys
        assert set(episode.fact_keys) <= stored_keys

    values = {fact.value for fact in episodic.memory.facts}
    serialized = json.dumps([episode.fact_keys for episode in episodic.episodes], ensure_ascii=False)
    assert not any(value in serialized for value in values)


def test_recording_does_not_touch_the_l73_file(tmp_path):
    facts_path = tmp_path / "knowledge.json.gz"
    store = ConversationKnowledgeMemory(facts_path)
    store.learn("既存の作業", "結論: 既存の事実です。")
    store.save()
    original = facts_path.read_bytes()

    episodic = EpisodicMemory(store, tmp_path / "episodes.json.gz")
    _record(episodic, "波動方程式を解くコードを作成して", "wave.py を作成しました。結論: CFL 条件を満たす。")
    episodic.save()

    assert facts_path.read_bytes() == original
    with gzip.open(facts_path, "rt", encoding="utf-8") as stream:
        assert set(json.load(stream)) == {"version", "facts"}


def test_episodes_round_trip_through_disk(tmp_path):
    path = tmp_path / "episodes.json.gz"
    episodic = EpisodicMemory(ConversationKnowledgeMemory(tmp_path / "k.gz"), path)
    _record(episodic, "SU2ゲージ理論の計算コードを作成して", "su2_gauge.py を作成しました。結論: Wilson ループ。")
    episodic.save()

    restored = EpisodicMemory(ConversationKnowledgeMemory(tmp_path / "k.gz"), path)

    assert [item.episode_id for item in restored.episodes] == ["ep-001"]
    assert restored.episodes[0].artifacts == ("su2_gauge.py",)
    assert restored.episodes[0].names == episodic.episodes[0].names


def test_reference_before_any_episode_starts_new_work(tmp_path):
    episodic = EpisodicMemory(ConversationKnowledgeMemory(tmp_path / "k.gz"))

    assert episodic.resolve_reference("それを修正して").decision == "NEW_EPISODE"


def test_experiment_probes_all_pass_with_zero_leak(tmp_path):
    report = run_experiment()

    assert all(item["decision_ok"] and item["episode_ok"] for item in report["results"])
    assert report["metrics"]["stale_intent_leak_rate"] == 0.0
    assert report["metrics"]["wrong_episode_reuse_rate"] == 0.0
    assert report["metrics"]["reference_resolution_accuracy"] == 1.0
    assert report["metrics"]["clarification_accuracy"] == 1.0
    assert report["metrics"]["unnecessary_clarification_rate"] == 0.0
    assert "Episodic State Memory Experiment (L8.13)" in format_experiment(report)
    assert "Episodic State (L8.13)" in format_episodic_state(report["episodes"])
