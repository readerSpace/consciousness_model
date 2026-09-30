"""L8.14: measure whether L8.13's episode resolution survives a long session.

L8.13 is frozen for this experiment.  Nothing here edits
``_REFERENCE_STOPWORDS`` or the resolver, because the point is to find out
where the name-matching design stops working rather than to patch it into
passing.  A benchmark that is tuned until it passes measures the tuning.

The protocol follows L8.6.1: the *generator* holds the ground truth and the
resolver never sees it.  Episodes are synthesised first; reference queries are
derived from the generated corpus afterwards, and the resolver is handed the
query **string only**.  Which episode a query should resolve to is computed
from the corpus, not asserted by hand, so a query is only labelled unambiguous
when the generator can show the phrase identifies exactly one episode.

Why a long session degrades at all: a domain phrase is built from components
(``古典一次元`` + ``イジング`` + ``相転移``) and each component is reused by other
episodes as the corpus grows.  The full phrase still names exactly one episode
to a reader, but the resolver matches token-by-token and cannot rank, so once
any component is shared it sees several candidates and asks for
clarification.  That is the specific limit this experiment is built to expose,
and the per-kind breakdown shows which reference types survive it.
"""
from __future__ import annotations

from dataclasses import dataclass
import random

from .behavior_monitor_l87_experiment import BehaviorEvent
from .conversation_memory_l73_experiment import ConversationKnowledgeMemory
from .episodic_memory_l813_experiment import EpisodicMemory, _named_tokens
from .goal_execution_loop_l812_experiment import run_goal_loop, trace_with

_PREFIXES = ("古典", "量子")
_DIMENSIONS = ("一次元", "二次元", "三次元", "四次元")
_MODELS = ("イジング", "ハイゼンベルク", "ポッツ", "シュウィンガー", "スピングラス")
_ASPECTS = ("相転移", "臨界指数", "秩序変数", "相関長", "緩和時間", "比熱")

#: 4 x 5 x 6 = 60 distinct component triples, doubled by the prefix.
MAX_EPISODES = len(_PREFIXES) * 60

SCALING_SIZES: tuple[int, ...] = (2, 5, 10, 20, 50, 100)

QUERY_KINDS: tuple[str, ...] = (
    "BARE_RECENT",
    "ARTIFACT",
    "ENTITY_SPECIFIC",
    "ENTITY_PHRASE",
    "AMBIGUOUS",
    "UNKNOWN_ENTITY",
    "NEW_TASK",
)


@dataclass(frozen=True)
class GeneratedEpisode:
    index: int
    model_id: str
    artifact: str
    phrase: str
    request: str
    response: str


@dataclass(frozen=True)
class ReferenceQuery:
    """A probe.  ``expected_*`` is generator-side truth; only ``text`` is shown."""

    text: str
    kind: str
    expected_decision: str
    expected_episode_id: str | None


def generate_episodes(count: int) -> tuple[GeneratedEpisode, ...]:
    if count > MAX_EPISODES:
        raise ValueError(f"at most {MAX_EPISODES} distinct episodes can be generated")
    episodes = []
    for index in range(count):
        prefix = _PREFIXES[index // 60]
        phrase = (
            f"{prefix}{_DIMENSIONS[index % 4]}{_MODELS[index % 5]}の{_ASPECTS[index % 6]}"
        )
        model_id = f"mdl{index:03d}"
        artifact = f"sim_{index:03d}.py"
        episodes.append(
            GeneratedEpisode(
                index=index,
                model_id=model_id,
                artifact=artifact,
                phrase=phrase,
                request=f"{model_id}: {phrase}を解くコードを作成して",
                response=(
                    f"{artifact} を作成しました。結論: {phrase} の計算は格子サイズに依存する。"
                ),
            )
        )
    return tuple(episodes)


def build_memory(episodes: tuple[GeneratedEpisode, ...]) -> EpisodicMemory:
    store = ConversationKnowledgeMemory(_NULL_PATH, max_facts=max(len(episodes) * 2, 240))
    store.facts = []
    episodic = EpisodicMemory(store)
    for generated in episodes:
        def executor(_request, _directives, response=generated.response):
            return response, trace_with(BehaviorEvent.FILE_WRITTEN)

        episodic.record(run_goal_loop(generated.request, executor))
    return episodic


class _NullPath:
    """A path that never exists and is never written, so no file is produced."""

    def is_file(self) -> bool:
        return False


_NULL_PATH = _NullPath()


def _token_owners(episodic: EpisodicMemory) -> dict[str, list[str]]:
    owners: dict[str, list[str]] = {}
    for episode in episodic.episodes:
        for name in episode.names:
            owners.setdefault(name, []).append(episode.episode_id)
    return owners


def build_queries(
    episodes: tuple[GeneratedEpisode, ...], episodic: EpisodicMemory, seed: int = 8140, per_kind: int = 6
) -> tuple[ReferenceQuery, ...]:
    """Derive probes from the finished corpus.  Truth comes from the corpus."""
    rng = random.Random(seed)
    owners = _token_owners(episodic)
    indices = list(range(len(episodes)))
    queries: list[ReferenceQuery] = []

    def sample(count: int) -> list[int]:
        return rng.sample(indices, min(count, len(indices)))

    latest_id = episodic.episodes[-1].episode_id
    for _ in range(per_kind):
        queries.append(ReferenceQuery("さっきの続きをやって", "BARE_RECENT", "CONTINUE_EPISODE", latest_id))

    for index in sample(per_kind):
        generated = episodes[index]
        queries.append(
            ReferenceQuery(
                f"{generated.artifact} を実行して",
                "ARTIFACT",
                "CONTINUE_EPISODE",
                f"ep-{index + 1:03d}",
            )
        )

    for index in sample(per_kind):
        generated = episodes[index]
        queries.append(
            ReferenceQuery(
                f"さっきの {generated.model_id} を続けて",
                "ENTITY_SPECIFIC",
                "CONTINUE_EPISODE",
                f"ep-{index + 1:03d}",
            )
        )

    # A phrase query is only unambiguous when the full phrase belongs to one
    # episode, which the generator verifies against the corpus rather than
    # assuming.  The resolver still only receives the sentence.
    for index in sample(per_kind):
        generated = episodes[index]
        phrase_tokens = set(_named_tokens(generated.phrase))
        matching = {
            episode.episode_id
            for episode in episodic.episodes
            if phrase_tokens <= set(episode.names)
        }
        if len(matching) != 1:
            continue
        queries.append(
            ReferenceQuery(
                f"さっきの{generated.phrase}の件を続けて",
                "ENTITY_PHRASE",
                "CONTINUE_EPISODE",
                f"ep-{index + 1:03d}",
            )
        )

    shared = sorted(token for token, holders in owners.items() if len(holders) > 1)
    for token in shared[:per_kind]:
        queries.append(
            ReferenceQuery(f"さっきの{token}の件を続けて", "AMBIGUOUS", "CLARIFY", None)
        )

    for offset in range(per_kind):
        queries.append(
            ReferenceQuery(
                f"さっきの mdl{900 + offset:03d} を続けて",
                "UNKNOWN_ENTITY",
                "CLARIFY",
                None,
            )
        )

    for topic in ("光格子時計", "地震波干渉法", "流体潤滑", "銀河回転曲線", "触媒反応速度", "生体膜輸送")[:per_kind]:
        queries.append(
            ReferenceQuery(
                f"今のとは別に新しく{topic}のコードを作って",
                "NEW_TASK",
                "NEW_EPISODE",
                None,
            )
        )

    return tuple(queries)


def _ratio(numerator: int, denominator: int) -> float | str:
    if denominator == 0:
        return "n/a"
    return round(numerator / denominator, 3)


def default_resolver(episodic: EpisodicMemory, text: str):
    """The frozen L8.13 resolver.  Passing anything else names a system under test."""
    return episodic.resolve_reference(text)


def run_size(count: int, seed: int = 8140, resolver=None) -> dict[str, object]:
    resolver = resolver or default_resolver
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
                "expected_episode": query.expected_episode_id,
                "fact_count": len(facts),
                "correct": resolution.decision == query.expected_decision
                and resolution.episode_id == query.expected_episode_id,
            }
        )

    continuing = [item for item in records if item["expected_decision"] == "CONTINUE_EPISODE"]
    clarifying = [item for item in records if item["expected_decision"] == "CLARIFY"]
    fresh = [item for item in records if item["expected_decision"] == "NEW_EPISODE"]
    referencing = continuing + clarifying
    artifacts = [item for item in records if item["kind"] == "ARTIFACT"]

    per_kind = {
        kind: _ratio(
            sum(1 for item in records if item["kind"] == kind and item["correct"]),
            sum(1 for item in records if item["kind"] == kind),
        )
        for kind in QUERY_KINDS
    }

    metrics = {
        "reference_resolution_accuracy": _ratio(
            sum(1 for item in continuing if item["correct"]), len(continuing)
        ),
        "wrong_episode_reuse_rate": _ratio(
            sum(
                1
                for item in referencing
                if item["episode"] is not None and item["episode"] != item["expected_episode"]
            ),
            len(referencing),
        ),
        "stale_intent_leak_rate": _ratio(
            sum(1 for item in fresh if item["fact_count"] > 0), len(fresh)
        ),
        "new_task_contamination_rate": _ratio(
            sum(1 for item in fresh if item["decision"] != "NEW_EPISODE"), len(fresh)
        ),
        "artifact_reference_accuracy": _ratio(
            sum(1 for item in artifacts if item["correct"]), len(artifacts)
        ),
        "clarification_accuracy": _ratio(
            sum(1 for item in clarifying if item["decision"] == "CLARIFY"), len(clarifying)
        ),
        "unnecessary_clarification_rate": _ratio(
            sum(1 for item in continuing + fresh if item["decision"] == "CLARIFY"),
            len(continuing) + len(fresh),
        ),
    }
    return {
        "episode_count": count,
        "query_count": len(records),
        "records": records,
        "metrics": metrics,
        "per_kind_accuracy": per_kind,
    }


def run_experiment(
    sizes: tuple[int, ...] = SCALING_SIZES, seed: int = 8140, resolver=None
) -> dict[str, object]:
    runs = [run_size(size, seed=seed, resolver=resolver) for size in sizes]
    baseline = runs[0]["metrics"]["reference_resolution_accuracy"]
    degradation = {}
    for run in runs:
        accuracy = run["metrics"]["reference_resolution_accuracy"]
        if isinstance(baseline, float) and isinstance(accuracy, float):
            degradation[run["episode_count"]] = round(baseline - accuracy, 3)
        else:
            degradation[run["episode_count"]] = "n/a"
    return {"runs": runs, "episode_scaling_degradation": degradation, "baseline_size": sizes[0]}


def format_experiment(report: dict[str, object]) -> str:
    runs = report["runs"]
    lines = [
        "## Context Isolation / Episodic Generalization (L8.14)",
        "",
        "L8.13 は凍結。`_REFERENCE_STOPWORDS` も resolver も変更していない。",
        "正解は generator のみが持ち、resolver には query 文字列だけを渡している。",
        "",
        "【Metrics by episode count】",
        "",
        "| N | ref_acc | wrong_reuse | stale_leak | new_task_contam | artifact_acc | clarify_acc | unnecessary_clarify |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for run in runs:
        metric = run["metrics"]
        lines.append(
            "| {n} | {a} | {w} | {s} | {c} | {ar} | {cl} | {uc} |".format(
                n=run["episode_count"],
                a=metric["reference_resolution_accuracy"],
                w=metric["wrong_episode_reuse_rate"],
                s=metric["stale_intent_leak_rate"],
                c=metric["new_task_contamination_rate"],
                ar=metric["artifact_reference_accuracy"],
                cl=metric["clarification_accuracy"],
                uc=metric["unnecessary_clarification_rate"],
            )
        )

    lines.extend(["", "【Accuracy by reference kind】", "", "| N | " + " | ".join(QUERY_KINDS) + " |", "| --- |" + " --- |" * len(QUERY_KINDS)])
    for run in runs:
        per_kind = run["per_kind_accuracy"]
        lines.append(
            f"| {run['episode_count']} | " + " | ".join(str(per_kind[kind]) for kind in QUERY_KINDS) + " |"
        )

    lines.extend(["", f"【episode_scaling_degradation  D(N) = A({report['baseline_size']}) - A(N)】", ""])
    for size, value in report["episode_scaling_degradation"].items():
        lines.append(f"- N={size}: `{value}`")
    return "\n".join(lines)


def main() -> None:  # pragma: no cover - manual entry point
    print(format_experiment(run_experiment()))


if __name__ == "__main__":  # pragma: no cover
    main()
