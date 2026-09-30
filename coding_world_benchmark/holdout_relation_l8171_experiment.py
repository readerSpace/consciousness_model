"""L8.17.1: cold hold-out for the relation resolver.

L8.17 was built against the failures L8.15.1 reported, which makes L8.15.1
development data.  This is the honest measurement, with the resolver frozen and
three things new at once: a third domain's vocabulary, relation phrasings that
are deliberately absent from L8.17's phrase table, and multi-step relations the
design does not attempt at all.

Capability and safety are scored separately, because they are different claims:

* ``relation_accuracy`` -- the traversal landed on the episode the generator
  meant.  An unrecognised phrasing is expected to score zero here.
* ``wrong_relation_reuse_rate`` -- the traversal landed on a *different*
  episode.  This is the invariant, and it must be zero **including** for the
  phrasings the design cannot parse, because a relation that is misread is far
  worse than one that is refused.

A clarification counts as neither: it is a miss, not a corruption.  Keeping the
two apart is what stops "refuse everything" from looking like success and
"guess confidently" from looking like capability.
"""
from __future__ import annotations

from dataclasses import dataclass
import random

from .behavior_monitor_l87_experiment import BehaviorEvent
from .conversation_memory_l73_experiment import ConversationKnowledgeMemory
from .episode_relation_l817_experiment import resolve_reference_relational
from .episodic_memory_l813_experiment import EpisodicMemory
from .goal_execution_loop_l812_experiment import run_goal_loop, trace_with
from .structural_resolver_l815_experiment import resolve_reference_structural

_MATERIALS = ("酸化チタン", "窒化ガリウム", "炭化ケイ素", "硫化モリブデン", "ホウ化ジルコニウム")
_METHODS = ("インピーダンス", "ラマン", "光電子", "中性子", "熱重量")
_ASPECTS = ("薄膜成長", "格子欠陥", "電荷輸送", "界面反応", "相安定性")

MAX_EPISODES = 125


@dataclass(frozen=True)
class HoldoutEpisode:
    index: int
    run_id: str
    artifact: str
    phrase: str
    request: str
    response: str
    revises: str | None = None


class _NullPath:
    def is_file(self) -> bool:
        return False


def generate_episodes(count: int = 40) -> tuple[HoldoutEpisode, ...]:
    if count > MAX_EPISODES:
        raise ValueError(f"at most {MAX_EPISODES} distinct episodes can be generated")
    episodes: list[HoldoutEpisode] = []
    for index in range(count):
        material = _MATERIALS[index % 5]
        method = _METHODS[(index // 5) % 5]
        aspect = _ASPECTS[(index // 25) % 5]
        phrase = f"{material}の{method}{aspect}"
        run_id = f"mat{index:03d}"
        artifact = f"mat_{index:03d}.py"
        # Every seventh episode revises its immediate predecessor, so the graph
        # carries revises edges rather than only temporal ones.  It must be the
        # predecessor and not an older episode: revising a revision chains the
        # artifact back to the first episode and makes every edge point there,
        # which silently invalidates the probe's ground truth.
        if index >= 7 and index % 7 == 0:
            source = episodes[index - 1]
            request = f"{run_id}: {source.artifact} を修正して"
            response = f"{source.artifact} を修正しました。結論: 補正後も傾向は変わらない。"
            episodes.append(
                HoldoutEpisode(index, run_id, source.artifact, phrase, request, response, source.artifact)
            )
            continue
        episodes.append(
            HoldoutEpisode(
                index,
                run_id,
                artifact,
                phrase,
                f"{run_id}: {phrase}のコードを作成して",
                f"{artifact} を作成しました。結論: {phrase} は前処理条件に依存する。",
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


@dataclass(frozen=True)
class RelationQuery:
    text: str
    kind: str
    expected_episode: str | None
    covered: bool
    note: str


def _unique_phrase_owner(episodic: EpisodicMemory, phrase: str) -> str | None:
    from .structural_resolver_l815_experiment import episode_tokens, query_tokens

    tokens = set(query_tokens(phrase))
    owners = [e.episode_id for e in episodic.episodes if tokens <= episode_tokens(e)]
    return owners[0] if len(owners) == 1 else None


def build_queries(
    episodes: tuple[HoldoutEpisode, ...], episodic: EpisodicMemory, seed: int = 81710, per_kind: int = 5
) -> tuple[RelationQuery, ...]:
    rng = random.Random(seed)
    total = len(episodes)
    queries: list[RelationQuery] = []

    def anchors(count: int) -> list[int]:
        usable = [i for i in range(total - 1) if _unique_phrase_owner(episodic, episodes[i].phrase)]
        return rng.sample(usable, min(count, len(usable)))

    for _ in range(per_kind):
        queries.append(
            RelationQuery("その前の実験を続けて", "COVERED_PRECEDING", f"ep-{total - 1:03d}", True,
                          "covered phrasing, third vocabulary")
        )
    for _ in range(per_kind):
        queries.append(
            RelationQuery("最初にやったやつを続けて", "COVERED_FIRST", "ep-001", True, "covered phrasing")
        )
    for index in anchors(per_kind):
        queries.append(
            RelationQuery(f"{episodes[index].phrase}の後にやった方を続けて", "COVERED_FOLLOWING",
                          f"ep-{index + 2:03d}", True, "anchored traversal in new vocabulary")
        )
    for index in range(total):
        if episodes[index].revises and len(queries) < 100:
            source_index = index - 1
            queries.append(
                RelationQuery(f"{episodes[source_index].phrase}の修正版を続けて", "COVERED_REVISION",
                              f"ep-{index + 1:03d}", True, "revises edge, not temporal adjacency")
            )

    # Phrasings absent from the table.  Expected to miss; required not to corrupt.
    for _ in range(per_kind):
        queries.append(
            RelationQuery("二つ前の実験を続けて", "UNCOVERED_MULTISTEP", f"ep-{total - 2:03d}", False,
                          "multi-step relation the design does not attempt")
        )
    for _ in range(per_kind):
        queries.append(
            RelationQuery("ひとつ手前の実験を続けて", "UNCOVERED_SYNONYM", f"ep-{total - 1:03d}", False,
                          "synonym of a covered relation, absent from the table")
        )
    for index in anchors(per_kind):
        queries.append(
            RelationQuery(f"{episodes[index].phrase}の直後のを続けて", "UNCOVERED_ANCHORED",
                          f"ep-{index + 2:03d}", False, "anchored, but 直後の is not in the table")
        )

    for _ in range(per_kind):
        queries.append(
            RelationQuery("mat999.py の後にやった方を続けて", "MISSING_ANCHOR", None, True,
                          "control: the anchor does not exist")
        )
    for _ in range(per_kind):
        queries.append(
            RelationQuery("酸化チタンの後にやった方を続けて", "AMBIGUOUS_ANCHOR", None, True,
                          "control: the anchor names several episodes")
        )
    for topic in ("光格子時計", "銀河回転曲線", "生体膜輸送", "乱流遷移", "超音速燃焼")[:per_kind]:
        queries.append(
            RelationQuery(f"今のとは別に新しく{topic}のコードを作って", "NEW_TASK", None, True,
                          "control: relation machinery must not capture new work")
        )
    return tuple(queries)


def _ratio(numerator: int, denominator: int) -> float | str:
    if denominator == 0:
        return "n/a"
    return round(numerator / denominator, 3)


def run_holdout(count: int = 40, seed: int = 81710, resolver=None) -> dict[str, object]:
    resolver = resolver or resolve_reference_relational
    episodes = generate_episodes(count)
    episodic = build_memory(episodes)
    queries = build_queries(episodes, episodic, seed=seed)

    records = []
    for query in queries:
        resolution = resolver(episodic, query.text)
        landed = resolution.episode_id
        expected = query.expected_episode
        if query.kind == "NEW_TASK":
            correct = resolution.decision == "NEW_EPISODE"
            wrong = resolution.decision == "CONTINUE_EPISODE"
        elif expected is None:
            correct = resolution.decision == "CLARIFY"
            wrong = landed is not None
        else:
            correct = landed == expected
            wrong = landed is not None and landed != expected
        records.append(
            {
                "kind": query.kind,
                "covered": query.covered,
                "text": query.text,
                "decision": resolution.decision,
                "episode": landed,
                "expected": expected,
                "correct": correct,
                "wrong": wrong,
                "note": query.note,
            }
        )

    relational = [r for r in records if r["kind"] not in {"NEW_TASK"}]
    covered = [r for r in relational if r["covered"]]
    uncovered = [r for r in relational if not r["covered"]]
    fresh = [r for r in records if r["kind"] == "NEW_TASK"]

    metrics = {
        "relation_accuracy": _ratio(sum(1 for r in relational if r["correct"]), len(relational)),
        "covered_relation_accuracy": _ratio(sum(1 for r in covered if r["correct"]), len(covered)),
        "uncovered_relation_accuracy": _ratio(sum(1 for r in uncovered if r["correct"]), len(uncovered)),
        "wrong_relation_reuse_rate": _ratio(sum(1 for r in relational if r["wrong"]), len(relational)),
        "uncovered_wrong_rate": _ratio(sum(1 for r in uncovered if r["wrong"]), len(uncovered)),
        "new_task_contamination_rate": _ratio(sum(1 for r in fresh if r["wrong"]), len(fresh)),
    }
    per_kind = {
        kind: _ratio(
            sum(1 for r in records if r["kind"] == kind and r["correct"]),
            sum(1 for r in records if r["kind"] == kind),
        )
        for kind in dict.fromkeys(r["kind"] for r in records)
    }
    return {"records": records, "metrics": metrics, "per_kind_accuracy": per_kind}


def run_experiment(count: int = 40) -> dict[str, object]:
    return {
        "structural": run_holdout(count, resolver=resolve_reference_structural),
        "relational": run_holdout(count, resolver=resolve_reference_relational),
    }


def format_experiment(report: dict[str, object]) -> str:
    lines = [
        "## Cold hold-out for the relation resolver (L8.17.1)",
        "",
        "resolver 凍結。第3のドメイン語彙、表に無い関係表現、多段関係を含む。",
        "",
        "| resolver | relation_acc | covered | uncovered | **wrong_relation_reuse** | uncovered_wrong | new_task_contam |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for label, key in (("L8.15", "structural"), ("L8.17", "relational")):
        m = report[key]["metrics"]
        lines.append(
            f"| {label} | {m['relation_accuracy']} | {m['covered_relation_accuracy']} | "
            f"{m['uncovered_relation_accuracy']} | **{m['wrong_relation_reuse_rate']}** | "
            f"{m['uncovered_wrong_rate']} | {m['new_task_contamination_rate']} |"
        )
    kinds = list(report["relational"]["per_kind_accuracy"])
    lines += ["", "| kind | L8.15 | L8.17 |", "| --- | --- | --- |"]
    for kind in kinds:
        lines.append(
            f"| {kind} | {report['structural']['per_kind_accuracy'].get(kind)} | "
            f"{report['relational']['per_kind_accuracy'][kind]} |"
        )
    return "\n".join(lines)


def main() -> None:  # pragma: no cover - manual entry point
    print(format_experiment(run_experiment()))


if __name__ == "__main__":  # pragma: no cover
    main()
