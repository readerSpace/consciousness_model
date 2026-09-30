"""L8.15.1: cold hold-out for the L8.15 resolver.

L8.15's weights and thresholds were chosen against L8.14, which makes L8.14
development data and its 1.00 scores uninformative about generalisation.  This
module is the honest measurement, on the same protocol L8.6.1 used: the
resolver is **frozen**, and everything it is asked about is new.

What changes from L8.14, deliberately in three independent directions:

* **vocabulary** -- signal processing instead of lattice models, so no component
  the weights were chosen against reappears;
* **phrasing** -- four paraphrases that L8.14 never used, including ones that
  may not carry a continuation marker at all;
* **relation** -- references to *when* an episode happened rather than what it
  was called ("最初にやったやつ", "その前の", "X の後にやった方"), which the
  score has no feature for.

The third group is expected to be weak.  It is included rather than omitted
because a hold-out that only asks what the design already covers measures
nothing, and because where it fails is the argument for the episode relation
graph rather than for more weight tuning.  Nothing here is fixed after seeing
the numbers; that would turn this file into development data too.
"""
from __future__ import annotations

from dataclasses import dataclass
import random

from .behavior_monitor_l87_experiment import BehaviorEvent
from .context_isolation_l814_experiment import default_resolver
from .conversation_memory_l73_experiment import ConversationKnowledgeMemory
from .episodic_memory_l813_experiment import EpisodicMemory
from .goal_execution_loop_l812_experiment import run_goal_loop, trace_with
from .structural_resolver_l815_experiment import query_tokens, resolve_reference_structural

_PREFIXES = ("リアルタイム", "マルチチャネル")
_TARGETS = ("心電図", "脳波", "音声信号", "地震波", "株価系列")
_METHODS = ("カルマン", "ウェーブレット", "フーリエ", "ベイズ", "クラスタ")
_OPERATIONS = ("雑音除去", "特徴抽出", "異常検知", "周期推定", "欠損補完")

MAX_EPISODES = 125

HOLDOUT_KINDS: tuple[str, ...] = (
    "PHRASE_NEW_VOCAB",
    "PARAPHRASE",
    "TEMPORAL_FIRST",
    "TEMPORAL_PREVIOUS",
    "RELATION_AFTER",
    "AMBIGUOUS",
    "UNKNOWN_ENTITY",
    "ARTIFACT",
    "NEW_TASK",
)


@dataclass(frozen=True)
class HoldoutEpisode:
    index: int
    run_id: str
    artifact: str
    phrase: str
    request: str
    response: str


@dataclass(frozen=True)
class HoldoutQuery:
    text: str
    kind: str
    expected_decision: str
    expected_episode: str | None


class _NullPath:
    def is_file(self) -> bool:
        return False


def generate_episodes(count: int) -> tuple[HoldoutEpisode, ...]:
    if count > MAX_EPISODES:
        raise ValueError(f"at most {MAX_EPISODES} distinct episodes can be generated")
    episodes = []
    for index in range(count):
        target = _TARGETS[index % 5]
        method = _METHODS[(index // 5) % 5]
        operation = _OPERATIONS[(index // 25) % 5]
        if index % 3 == 2:
            phrase = f"{_PREFIXES[index % 2]}{target}の{method}{operation}"
        else:
            phrase = f"{target}の{method}{operation}"
        run_id = f"run{index:03d}"
        artifact = f"proc_{index:03d}.py"
        episodes.append(
            HoldoutEpisode(
                index=index,
                run_id=run_id,
                artifact=artifact,
                phrase=phrase,
                request=f"{run_id}: {phrase}のコードを作成して",
                response=f"{artifact} を作成しました。結論: {phrase} は前処理の窓長に敏感。",
            )
        )
    return tuple(episodes)


def build_memory(episodes: tuple[HoldoutEpisode, ...]) -> EpisodicMemory:
    store = ConversationKnowledgeMemory(_NullPath(), max_facts=max(len(episodes) * 2, 240))
    store.facts = []
    episodic = EpisodicMemory(store)
    for generated in episodes:
        def executor(_request, _directives, response=generated.response):
            return response, trace_with(BehaviorEvent.FILE_WRITTEN)

        episodic.record(run_goal_loop(generated.request, executor))
    return episodic


def _uniquely_owned(episodic: EpisodicMemory, phrase: str) -> str | None:
    """Ground truth, computed against the corpus rather than assumed."""
    from .structural_resolver_l815_experiment import episode_tokens

    tokens = set(query_tokens(phrase))
    owners = [
        episode.episode_id for episode in episodic.episodes if tokens <= episode_tokens(episode)
    ]
    return owners[0] if len(owners) == 1 else None


_PARAPHRASES = (
    "{phrase}のやつ、もう一度見て",
    "前に作った{phrase}を直して",
    "{phrase}の続きをお願い",
    "その{phrase}について続けたい",
)


def build_queries(
    episodes: tuple[HoldoutEpisode, ...], episodic: EpisodicMemory, seed: int = 81510, per_kind: int = 6
) -> tuple[HoldoutQuery, ...]:
    rng = random.Random(seed)
    indices = list(range(len(episodes)))
    queries: list[HoldoutQuery] = []

    def sample(count: int) -> list[int]:
        return rng.sample(indices, min(count, len(indices)))

    for index in sample(per_kind):
        generated = episodes[index]
        owner = _uniquely_owned(episodic, generated.phrase)
        if owner is None:
            continue
        queries.append(
            HoldoutQuery(f"さっきの{generated.phrase}の件を続けて", "PHRASE_NEW_VOCAB", "CONTINUE_EPISODE", owner)
        )

    for position, index in enumerate(sample(per_kind)):
        generated = episodes[index]
        owner = _uniquely_owned(episodic, generated.phrase)
        if owner is None:
            continue
        template = _PARAPHRASES[position % len(_PARAPHRASES)]
        queries.append(
            HoldoutQuery(template.format(phrase=generated.phrase), "PARAPHRASE", "CONTINUE_EPISODE", owner)
        )

    for _ in range(per_kind):
        queries.append(
            HoldoutQuery("最初にやったやつを続けて", "TEMPORAL_FIRST", "CONTINUE_EPISODE", "ep-001")
        )

    if len(episodes) >= 2:
        previous = f"ep-{len(episodes) - 1:03d}"
        for _ in range(per_kind):
            queries.append(
                HoldoutQuery("その前の実験を続けて", "TEMPORAL_PREVIOUS", "CONTINUE_EPISODE", previous)
            )

    for index in sample(per_kind):
        if index + 1 >= len(episodes):
            continue
        anchor = episodes[index]
        if _uniquely_owned(episodic, anchor.phrase) is None:
            continue
        queries.append(
            HoldoutQuery(
                f"{anchor.phrase}の後にやった方を続けて",
                "RELATION_AFTER",
                "CONTINUE_EPISODE",
                f"ep-{index + 2:03d}",
            )
        )

    for target in _TARGETS[:per_kind]:
        if sum(1 for episode in episodes if target in episode.phrase) > 1:
            queries.append(HoldoutQuery(f"さっきの{target}の件を続けて", "AMBIGUOUS", "CLARIFY", None))

    for offset in range(per_kind):
        queries.append(
            HoldoutQuery(f"さっきの run{700 + offset:03d} を続けて", "UNKNOWN_ENTITY", "CLARIFY", None)
        )

    for index in sample(per_kind):
        generated = episodes[index]
        queries.append(
            HoldoutQuery(f"{generated.artifact} を実行して", "ARTIFACT", "CONTINUE_EPISODE", f"ep-{index + 1:03d}")
        )

    for topic in ("光格子時計", "触媒反応速度", "銀河回転曲線", "生体膜輸送", "結晶成長", "熱電変換")[:per_kind]:
        queries.append(
            HoldoutQuery(f"今のとは別に新しく{topic}のコードを作って", "NEW_TASK", "NEW_EPISODE", None)
        )

    return tuple(queries)


def _ratio(numerator: int, denominator: int) -> float | str:
    if denominator == 0:
        return "n/a"
    return round(numerator / denominator, 3)


def run_holdout(count: int = 50, seed: int = 81510, resolver=None) -> dict[str, object]:
    resolver = resolver or resolve_reference_structural
    episodes = generate_episodes(count)
    episodic = build_memory(episodes)
    queries = build_queries(episodes, episodic, seed=seed)

    records = []
    for query in queries:
        resolution = resolver(episodic, query.text)
        facts = episodic.scoped_recall_for(resolution, query.text)
        records.append(
            {
                "kind": query.kind,
                "text": query.text,
                "decision": resolution.decision,
                "expected_decision": query.expected_decision,
                "episode": resolution.episode_id,
                "expected_episode": query.expected_episode,
                "fact_count": len(facts),
                "correct": resolution.decision == query.expected_decision
                and resolution.episode_id == query.expected_episode,
            }
        )

    continuing = [item for item in records if item["expected_decision"] == "CONTINUE_EPISODE"]
    clarifying = [item for item in records if item["expected_decision"] == "CLARIFY"]
    fresh = [item for item in records if item["expected_decision"] == "NEW_EPISODE"]

    metrics = {
        "reference_resolution_accuracy": _ratio(
            sum(1 for item in continuing if item["correct"]), len(continuing)
        ),
        "wrong_episode_reuse_rate": _ratio(
            sum(
                1
                for item in continuing + clarifying
                if item["episode"] is not None and item["episode"] != item["expected_episode"]
            ),
            len(continuing + clarifying),
        ),
        "stale_intent_leak_rate": _ratio(
            sum(1 for item in fresh if item["fact_count"] > 0), len(fresh)
        ),
        "new_task_contamination_rate": _ratio(
            sum(1 for item in fresh if item["decision"] != "NEW_EPISODE"), len(fresh)
        ),
        "clarification_accuracy": _ratio(
            sum(1 for item in clarifying if item["decision"] == "CLARIFY"), len(clarifying)
        ),
        "unnecessary_clarification_rate": _ratio(
            sum(1 for item in continuing + fresh if item["decision"] == "CLARIFY"),
            len(continuing) + len(fresh),
        ),
    }
    per_kind = {
        kind: _ratio(
            sum(1 for item in records if item["kind"] == kind and item["correct"]),
            sum(1 for item in records if item["kind"] == kind),
        )
        for kind in HOLDOUT_KINDS
    }
    return {"episode_count": count, "records": records, "metrics": metrics, "per_kind_accuracy": per_kind}


def run_experiment(count: int = 50) -> dict[str, object]:
    return {
        "baseline": run_holdout(count, resolver=default_resolver),
        "structural": run_holdout(count, resolver=resolve_reference_structural),
    }


def format_experiment(report: dict[str, object]) -> str:
    lines = [
        "## Cold hold-out for the structural resolver (L8.15.1)",
        "",
        "resolver は凍結。語彙・言い回し・関係表現のすべてが L8.14 と別。",
        "",
        "### Metrics",
        "",
        "| resolver | ref_acc | wrong_reuse | stale_leak | new_task_contam | clarify_acc | unnecessary_clarify |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for label in ("baseline", "structural"):
        metric = report[label]["metrics"]
        lines.append(
            "| {l} | {a} | {w} | {s} | {c} | {cl} | {u} |".format(
                l="L8.13" if label == "baseline" else "L8.15",
                a=metric["reference_resolution_accuracy"],
                w=metric["wrong_episode_reuse_rate"],
                s=metric["stale_intent_leak_rate"],
                c=metric["new_task_contamination_rate"],
                cl=metric["clarification_accuracy"],
                u=metric["unnecessary_clarification_rate"],
            )
        )
    lines += ["", "### Accuracy by query kind", "", "| kind | L8.13 | L8.15 |", "| --- | --- | --- |"]
    for kind in HOLDOUT_KINDS:
        lines.append(
            f"| {kind} | {report['baseline']['per_kind_accuracy'][kind]} | "
            f"{report['structural']['per_kind_accuracy'][kind]} |"
        )
    return "\n".join(lines)


def main() -> None:  # pragma: no cover - manual entry point
    print(format_experiment(run_experiment()))


if __name__ == "__main__":  # pragma: no cover
    main()
