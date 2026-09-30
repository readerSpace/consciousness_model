"""L8.18.1: cold hold-out for the incomplete-operation gate.

Fourth vocabulary, resolver frozen, and the rule under test was written down in
L8.18's docstring before this file existed.  The point of the pre-registration
is that two of the families exist to make the guard *fail* if it is too eager:

* ``NON_RELATION_LOOKALIKE`` -- ordinary requests carrying 前処理 / 後処理 / 次元.
  A guard that keys on the characters rather than on their use abstains here and
  loses capability it never needed to lose.
* ``KNOWN_OPERATION`` / ``BARE_REFERENCE`` / ``NAMED_REFERENCE`` -- unchanged
  behaviour, so "abstain more" cannot be mistaken for "safer".

Safety is therefore reported as a pair.  ``wrong_reuse_rate`` alone is
satisfied by refusing everything; ``unnecessary_abstention_rate`` alone is
satisfied by guessing everything.  Only both at zero means anything.
"""
from __future__ import annotations

from dataclasses import dataclass
import random

from .behavior_monitor_l87_experiment import BehaviorEvent
from .conversation_memory_l73_experiment import ConversationKnowledgeMemory
from .episode_relation_l817_experiment import resolve_reference_relational
from .episodic_memory_l813_experiment import EpisodicMemory
from .goal_execution_loop_l812_experiment import run_goal_loop, trace_with
from .incomplete_operation_l818_experiment import execute, resolve_reference_gated

_OBJECTS = ("系外惑星", "超新星残骸", "銀河団", "重力レンズ", "パルサー")
_INSTRUMENTS = ("電波干渉", "近赤外分光", "X線撮像", "偏光測定", "時系列測光")
_STAGES = ("前処理", "後処理", "次元削減", "較正", "合成")


@dataclass(frozen=True)
class HoldoutEpisode:
    index: int
    artifact: str
    phrase: str
    request: str
    response: str
    stage: str


class _NullPath:
    def is_file(self) -> bool:
        return False


def generate_episodes(count: int = 30) -> tuple[HoldoutEpisode, ...]:
    episodes = []
    for index in range(count):
        obj = _OBJECTS[index % 5]
        instrument = _INSTRUMENTS[(index // 5) % 5]
        stage = _STAGES[index % 5]
        phrase = f"{obj}の{instrument}{stage}"
        artifact = f"obs_{index:03d}.py"
        episodes.append(
            HoldoutEpisode(
                index,
                artifact,
                phrase,
                f"obs{index:03d}: {phrase}のコードを作成して",
                f"{artifact} を作成しました。結論: {phrase} は較正誤差に敏感。",
                stage,
            )
        )
    return tuple(episodes)


def build_memory(episodes):
    store = ConversationKnowledgeMemory(_NullPath(), max_facts=max(len(episodes) * 2, 240))
    store.facts = []
    episodic = EpisodicMemory(store)
    for generated in episodes:
        def executor(_r, _d, response=generated.response):
            return response, trace_with(BehaviorEvent.FILE_WRITTEN)

        episodic.record(run_goal_loop(generated.request, executor))
    return episodic


@dataclass(frozen=True)
class GatedQuery:
    text: str
    family: str
    expected: str  # ANSWER:<ep> | ABSTAIN | NEW_TASK | COUNT:<n>
    note: str


def _unique(episodic, phrase):
    from .structural_resolver_l815_experiment import episode_tokens, query_tokens

    tokens = set(query_tokens(phrase))
    owners = [e.episode_id for e in episodic.episodes if tokens <= episode_tokens(e)]
    return owners[0] if len(owners) == 1 else None


def build_queries(episodes, episodic, seed: int = 81810, per_kind: int = 5):
    rng = random.Random(seed)
    total = len(episodes)
    usable = [i for i in range(total - 1) if _unique(episodic, episodes[i].phrase)]
    queries: list[GatedQuery] = []

    for _ in range(per_kind):
        queries.append(GatedQuery("その前の実験を続けて", "KNOWN_RELATION", f"ANSWER:ep-{total - 1:03d}",
                                  "covered relation, fourth vocabulary"))
    for index in rng.sample(usable, min(per_kind, len(usable))):
        queries.append(GatedQuery(f"{episodes[index].phrase}の後にやった方を続けて", "KNOWN_RELATION",
                                  f"ANSWER:ep-{index + 2:03d}", "covered anchored relation"))

    for index in rng.sample(usable, min(per_kind, len(usable))):
        queries.append(GatedQuery(f"{episodes[index].phrase}の直後のを続けて", "UNKNOWN_RELATION_ANCHORED",
                                  "ABSTAIN", "the L8.17.1 defect: anchor resolvable, relation not"))
    for _ in range(per_kind):
        queries.append(GatedQuery("二つ前の実験を続けて", "UNKNOWN_RELATION_BARE", "ABSTAIN",
                                  "multi-step relation, nothing else to go on"))

    for _ in range(per_kind):
        queries.append(GatedQuery("さっきの続きをやって", "BARE_REFERENCE", f"ANSWER:ep-{total:03d}",
                                  "control: ordinary bare reference must be unaffected"))
    lookalikes = [i for i in usable if episodes[i].stage in {"前処理", "後処理", "次元削減"}]
    for index in lookalikes[:per_kind]:
        queries.append(GatedQuery(f"{episodes[index].phrase}を続けて", "NON_RELATION_LOOKALIKE",
                                  f"ANSWER:ep-{index + 1:03d}",
                                  "control: 前処理/後処理/次元 must not read as a relation"))
    for index in rng.sample(usable, min(per_kind, len(usable))):
        queries.append(GatedQuery(f"{episodes[index].phrase}の続きをお願い", "NAMED_REFERENCE",
                                  f"ANSWER:ep-{index + 1:03d}", "control: named reference unaffected"))

    for _ in range(per_kind):
        queries.append(GatedQuery("系外惑星の実験は何件やった？", "KNOWN_OPERATION", "COUNT:6",
                                  "control: an implemented operation still answers"))
    for _ in range(per_kind):
        queries.append(GatedQuery("一番多く使った観測手法は？", "UNKNOWN_OPERATION", "ABSTAIN",
                                  "operation evoked but unimplemented; must not degrade to a lookup"))
    for topic in ("光格子時計", "触媒反応速度", "乱流遷移", "生体膜輸送", "熱電変換")[:per_kind]:
        queries.append(GatedQuery(f"今のとは別に新しく{topic}のコードを作って", "NEW_TASK", "NEW_TASK",
                                  "control: the gate must not capture new work"))
    return tuple(queries)


def _ratio(numerator, denominator):
    if denominator == 0:
        return "n/a"
    return round(numerator / denominator, 3)


_RELATIONAL = {"KNOWN_RELATION", "UNKNOWN_RELATION_ANCHORED", "UNKNOWN_RELATION_BARE"}
_MUST_ANSWER = {"BARE_REFERENCE", "NON_RELATION_LOOKALIKE", "NAMED_REFERENCE", "KNOWN_OPERATION"}


def run_holdout(count: int = 30, seed: int = 81810, gated: bool = True):
    episodes = generate_episodes(count)
    episodic = build_memory(episodes)
    queries = build_queries(episodes, episodic, seed=seed)

    records = []
    for query in queries:
        if gated:
            outcome = execute(episodic, query.text)
            decision, episode_id, value = outcome.decision, outcome.episode_id, outcome.value
        else:
            resolution = resolve_reference_relational(episodic, query.text)
            decision = {"CONTINUE_EPISODE": "ANSWER", "NEW_EPISODE": "NEW_TASK"}.get(
                resolution.decision, "ABSTAIN"
            )
            episode_id, value = resolution.episode_id, None

        kind, _, argument = query.expected.partition(":")
        if kind == "ANSWER":
            correct = decision == "ANSWER" and episode_id == argument
            wrong = decision == "ANSWER" and episode_id != argument
        elif kind == "COUNT":
            correct = decision == "ANSWER" and str(value) == argument
            wrong = decision == "ANSWER" and str(value) != argument
        elif kind == "ABSTAIN":
            correct = decision == "ABSTAIN"
            wrong = decision == "ANSWER"
        else:
            correct = decision == "NEW_TASK"
            wrong = decision == "ANSWER"
        records.append(
            {
                "family": query.family,
                "text": query.text,
                "decision": decision,
                "episode": episode_id,
                "value": value,
                "expected": query.expected,
                "correct": correct,
                "wrong": wrong,
                "abstained": decision == "ABSTAIN",
            }
        )

    covered = [r for r in records if r["family"] == "KNOWN_RELATION"]
    uncovered = [r for r in records if r["family"].startswith("UNKNOWN_")]
    must_answer = [r for r in records if r["family"] in _MUST_ANSWER]
    fresh = [r for r in records if r["family"] == "NEW_TASK"]

    metrics = {
        "covered_relation_accuracy": _ratio(sum(1 for r in covered if r["correct"]), len(covered)),
        "uncovered_abstention_rate": _ratio(sum(1 for r in uncovered if r["abstained"]), len(uncovered)),
        "wrong_reuse_rate": _ratio(sum(1 for r in records if r["wrong"]), len(records)),
        "unnecessary_abstention_rate": _ratio(
            sum(1 for r in must_answer + fresh if r["abstained"]), len(must_answer) + len(fresh)
        ),
        "new_task_contamination_rate": _ratio(sum(1 for r in fresh if r["wrong"]), len(fresh)),
    }
    per_family = {
        family: _ratio(
            sum(1 for r in records if r["family"] == family and r["correct"]),
            sum(1 for r in records if r["family"] == family),
        )
        for family in dict.fromkeys(r["family"] for r in records)
    }
    return {"records": records, "metrics": metrics, "per_family_accuracy": per_family}


def run_experiment(count: int = 30):
    return {"relational": run_holdout(count, gated=False), "gated": run_holdout(count, gated=True)}


def format_experiment(report):
    lines = [
        "## Cold hold-out for the incomplete-operation gate (L8.18.1)",
        "",
        "第4語彙。規則は L8.18 の docstring に**事前登録済み**。",
        "",
        "| resolver | covered_relation | uncovered_abstention | **wrong_reuse** | **unnecessary_abstention** | new_task_contam |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for label, key in (("L8.17", "relational"), ("L8.18", "gated")):
        m = report[key]["metrics"]
        lines.append(
            f"| {label} | {m['covered_relation_accuracy']} | {m['uncovered_abstention_rate']} | "
            f"**{m['wrong_reuse_rate']}** | **{m['unnecessary_abstention_rate']}** | "
            f"{m['new_task_contamination_rate']} |"
        )
    families = list(report["gated"]["per_family_accuracy"])
    lines += ["", "| family | L8.17 | L8.18 |", "| --- | --- | --- |"]
    for family in families:
        lines.append(
            f"| {family} | {report['relational']['per_family_accuracy'].get(family)} | "
            f"{report['gated']['per_family_accuracy'][family]} |"
        )
    return "\n".join(lines)


def main() -> None:  # pragma: no cover
    print(format_experiment(run_experiment()))


if __name__ == "__main__":  # pragma: no cover
    main()
