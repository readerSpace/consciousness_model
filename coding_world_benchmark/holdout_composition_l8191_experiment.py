"""L8.19.1: cold compositional hold-out for the typed operations.

Unseen vocabulary is not the interesting axis any more.  L8.19's primitives are
COUNT, MEAN, MODE, ARGMAX and COMPARE, and the question is whether the system
learned *primitives* or only the example sentences they came in.  So the probes
are unseen **combinations** of seen operators -- COUNT(FILTER(..)),
MEAN(FILTER(..)), ARGMAX(FILTER(..)), COMPARE(MEAN(A), MEAN(B)) -- over a fifth
vocabulary, with the resolver and the signatures frozen.

Three refusal families keep the score honest, and they refuse for three
*different* reasons, which is the point of separating the checks:

* ``TYPE_ERROR`` -- every slot resolved, and the signature still forbids it
  (MEAN of a name, ARGMAX of a categorical);
* ``PRECONDITION`` -- well-typed, but the data the signature needs was never
  recorded (ARGMAX over a field no episode has);
* ``UNKNOWN_OPERATION`` -- 中央値 is evoked and unimplemented, so L8.18's gate
  holds it.

And two control families that must **not** refuse, because "abstain more" is
not the same as "safer": ordinary references, and valid compositions.
"""
from __future__ import annotations

from dataclasses import dataclass

from .behavior_monitor_l87_experiment import BehaviorEvent
from .conversation_memory_l73_experiment import ConversationKnowledgeMemory
from .episodic_memory_l813_experiment import EpisodicMemory
from .goal_execution_loop_l812_experiment import run_goal_loop, trace_with
from .typed_operation_l819_experiment import TypedStore, build_store, execute_typed

_SUBJECTS = ("翼型", "熱交換器", "燃焼器", "配管網", "軸受")
_SOLVERS = ("有限体積", "境界要素", "格子ボルツマン", "渦法", "陰解法")
_SIZES = ("32x32", "64x64", "128x128")


class _NullPath:
    def is_file(self) -> bool:
        return False


@dataclass(frozen=True)
class HoldoutEpisode:
    index: int
    phrase: str
    request: str
    response: str
    lattice: str
    runtime: float
    rate: float


def generate_episodes(count: int = 24) -> tuple[HoldoutEpisode, ...]:
    episodes = []
    for index in range(count):
        phrase = f"{_SUBJECTS[index % 5]}の{_SOLVERS[(index // 5) % 5]}解析"
        # Deliberately skewed: an even split has no most-common value, and a
        # MODE probe over it asserts an answer that does not exist.
        lattice = ("32x32", "64x64", "64x64", "128x128")[index % 4]
        runtime = round(10.0 + index * 1.5 + (0 if lattice == "32x32" else 20.0), 1)
        rate = round(0.50 + (index % 7) * 0.05, 2)
        response = (
            f"cfd_{index:03d}.py を作成しました。\n"
            f"格子サイズは {lattice}、乱数種は {20260000 + index} を使用しました。\n"
            f"実行時間は {runtime} 秒。成功率は {rate}。\n"
            f"結論: {phrase} は境界条件に敏感。"
        )
        episodes.append(
            HoldoutEpisode(index, phrase, f"cfd{index:03d}: {phrase}のコードを作成して",
                           response, lattice, runtime, rate)
        )
    return tuple(episodes)


def build_holdout_store(count: int = 24) -> tuple[TypedStore, tuple[HoldoutEpisode, ...]]:
    episodes = generate_episodes(count)
    store = ConversationKnowledgeMemory(_NullPath(), max_facts=max(count * 2, 240))
    store.facts = []
    episodic = EpisodicMemory(store)
    responses = {}
    for generated in episodes:
        def executor(_r, _d, response=generated.response):
            return response, trace_with(BehaviorEvent.FILE_WRITTEN)

        episode = episodic.record(run_goal_loop(generated.request, executor))
        responses[episode.episode_id] = generated.response
    return build_store(episodic, responses), episodes


@dataclass(frozen=True)
class CompositionQuery:
    text: str
    family: str
    expected: object  # value, or "ABSTAIN", or "NOT_AN_OPERATION"
    note: str


def _oracle(episodes, lattice=None):
    return [e for e in episodes if lattice is None or e.lattice == lattice]


def build_queries(episodes) -> tuple[CompositionQuery, ...]:
    import statistics

    sixty_four = _oracle(episodes, "64x64")
    thirty_two = _oracle(episodes, "32x32")
    best = max(sixty_four, key=lambda e: e.rate)
    lattice_counts = {size: len(_oracle(episodes, size)) for size in _SIZES}
    top_lattice = max(lattice_counts, key=lambda key: lattice_counts[key])

    queries = [
        CompositionQuery("64x64を使った実験はいくつ？", "COUNT_FILTERED", len(sixty_four),
                         "COUNT(FILTER(lattice))"),
        CompositionQuery("64x64を使った実験の平均実行時間は？", "MEAN_FILTERED",
                         round(statistics.fmean([e.runtime for e in sixty_four]), 4),
                         "MEAN(FILTER(lattice), runtime)"),
        CompositionQuery("64x64を使った実験で成功率が一番高いのは？", "ARGMAX_FILTERED",
                         f"ep-{best.index + 1:03d}", "ARGMAX(FILTER(lattice), success_rate)"),
        CompositionQuery("平均成功率は？", "MEAN_PLAIN",
                         round(statistics.fmean([e.rate for e in episodes]), 4), "MEAN(success_rate)"),
        CompositionQuery("一番多く使った格子サイズは？", "MODE_PLAIN", top_lattice, "MODE(lattice)"),
        CompositionQuery("32x32と比べて64x64の平均実行時間は大きい？", "COMPARE_GROUPS",
                         "greater" if statistics.fmean([e.runtime for e in sixty_four])
                         > statistics.fmean([e.runtime for e in thirty_two]) else "less",
                         "COMPARE(MEAN(A), MEAN(B))"),

        CompositionQuery("実験名の平均は？", "TYPE_ERROR", "ABSTAIN", "MEAN over Text"),
        CompositionQuery("格子サイズが一番大きい実験は？", "TYPE_ERROR", "ABSTAIN",
                         "ARGMAX over Categorical"),
        CompositionQuery("エネルギー誤差が一番大きい実験は？", "PRECONDITION", "ABSTAIN",
                         "well-typed, but no episode records the field"),
        CompositionQuery("64x64と32x32を使った実験はいくつ？", "PRECONDITION", "ABSTAIN",
                         "one field cannot hold two values at once"),
        CompositionQuery("実行時間の中央値は？", "UNKNOWN_OPERATION", "ABSTAIN",
                         "evoked and unimplemented; L8.18's gate holds it"),

        CompositionQuery("さっきの続きをやって", "NOT_AN_OPERATION", "NOT_AN_OPERATION",
                         "control: an ordinary reference is not an operation"),
        CompositionQuery("64x64を使った実験を続けて", "NOT_AN_OPERATION", "NOT_AN_OPERATION",
                         "control: a filter-shaped reference must not become a computation"),
    ]
    return tuple(queries)


def _ratio(numerator, denominator):
    if denominator == 0:
        return "n/a"
    return round(numerator / denominator, 3)


_VALID = {"COUNT_FILTERED", "MEAN_FILTERED", "ARGMAX_FILTERED", "MEAN_PLAIN", "MODE_PLAIN", "COMPARE_GROUPS"}
_COMPOSED = {"COUNT_FILTERED", "MEAN_FILTERED", "ARGMAX_FILTERED", "COMPARE_GROUPS"}
_INVALID = {"TYPE_ERROR", "PRECONDITION", "UNKNOWN_OPERATION"}


def run_holdout(count: int = 24) -> dict[str, object]:
    store, episodes = build_holdout_store(count)
    records = []
    for query in build_queries(episodes):
        outcome = execute_typed(store, query.text)
        if query.expected == "ABSTAIN":
            correct = outcome.decision == "ABSTAIN"
            wrong = outcome.decision == "ANSWER"
        elif query.expected == "NOT_AN_OPERATION":
            correct = outcome.decision == "NOT_AN_OPERATION"
            wrong = outcome.decision == "ANSWER"
        else:
            correct = outcome.decision == "ANSWER" and str(outcome.value) == str(query.expected)
            wrong = outcome.decision == "ANSWER" and str(outcome.value) != str(query.expected)
        records.append(
            {
                "family": query.family,
                "text": query.text,
                "note": query.note,
                "decision": outcome.decision,
                "value": outcome.value,
                "expected": query.expected,
                "stage": outcome.stage,
                "correct": correct,
                "wrong": wrong,
                "abstained": outcome.decision == "ABSTAIN",
            }
        )

    valid = [r for r in records if r["family"] in _VALID]
    composed = [r for r in records if r["family"] in _COMPOSED]
    invalid = [r for r in records if r["family"] in _INVALID]
    type_family = [r for r in records if r["family"] == "TYPE_ERROR"]
    controls = [r for r in records if r["family"] == "NOT_AN_OPERATION"]

    metrics = {
        "valid_execution_rate": _ratio(sum(1 for r in valid if r["correct"]), len(valid)),
        "composition_accuracy": _ratio(sum(1 for r in composed if r["correct"]), len(composed)),
        "invalid_execution_abstention_rate": _ratio(
            sum(1 for r in invalid if r["abstained"]), len(invalid)
        ),
        "wrong_operation_rate": _ratio(sum(1 for r in records if r["wrong"]), len(records)),
        "type_error_escape_rate": _ratio(
            sum(1 for r in type_family if r["decision"] == "ANSWER"), len(type_family)
        ),
        "unnecessary_abstention_rate": _ratio(
            sum(1 for r in valid + controls if r["abstained"]), len(valid) + len(controls)
        ),
    }
    return {"records": records, "metrics": metrics}


def format_experiment(report) -> str:
    lines = ["## Cold compositional hold-out (L8.19.1)", "", "【Metrics】", ""]
    for key, value in report["metrics"].items():
        lines.append(f"- {key}: `{value}`")
    lines += ["", "| family | query | decision | stage | ok |", "| --- | --- | --- | --- | --- |"]
    for record in report["records"]:
        lines.append(
            f"| {record['family']} | {record['text']} | {record['decision']} "
            f"| {record['stage'] or '-'} | {'OK' if record['correct'] else 'NG'} |"
        )
    return "\n".join(lines)


def main() -> None:  # pragma: no cover
    print(format_experiment(run_holdout()))


if __name__ == "__main__":  # pragma: no cover
    main()
