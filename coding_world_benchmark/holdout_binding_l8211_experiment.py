"""L8.21.1: cold hold-out for argument binding, L8.20 kept alongside as the control.

The novelty axis here is phrasing, not vocabulary, and that is the honest choice:
binding is about which mention fills which hole, so what has to be new is the
order and the redundancy of the mentions -- the field named before its value, a
field named that belongs to the filter rather than the projection, two fields
offered for one hole, none offered at all.

L8.20 is run on the same probes, because the interesting number is not that
L8.21 scores higher.  It is that one family separates them in a way accuracy
alone would hide: 「実行時間と成功率の平均は？」 offers two Number fields for one
projection hole.  L8.20 takes the first and returns a number.  Nothing in the
answer says it chose, so it is a wrong answer that reads as a right one, and it
is the first time since L8.18 that a layer has been able to produce one.
"""
from __future__ import annotations

from dataclasses import dataclass

from .argument_binding_l821_experiment import run_bound_query
from .behavior_monitor_l87_experiment import BehaviorEvent
from .conversation_memory_l73_experiment import ConversationKnowledgeMemory
from .episodic_memory_l813_experiment import EpisodicMemory
from .goal_execution_loop_l812_experiment import run_goal_loop, trace_with
from .query_algebra_l820_experiment import (
    Argmax, Compare, Count, Filter, Max, Mean, Mode, Project, Source,
    canonical_render, render, run_query,
)
from .typed_operation_l819_experiment import build_store

_TARGETS = ("化合物", "酵素", "受容体", "膜輸送体", "転写因子")
_ASSAYS = ("結合親和性", "阻害定数", "熱安定性", "選択性", "代謝安定性")
_SIZES = ("32x32", "64x64", "64x64", "128x128")


class _NullPath:
    def is_file(self) -> bool:
        return False


@dataclass(frozen=True)
class Ep:
    index: int
    lattice: str
    seed: str
    runtime: float
    rate: float


def build_holdout(count: int = 24):
    store = ConversationKnowledgeMemory(_NullPath(), max_facts=max(count * 2, 240))
    store.facts = []
    episodic = EpisodicMemory(store)
    responses, rows = {}, []
    for index in range(count):
        phrase = f"{_TARGETS[index % 5]}の{_ASSAYS[(index // 5) % 5]}スクリーニング"
        lattice = _SIZES[index % 4]
        seed = f"{30260000 + index}"
        runtime = round(8.0 + index * 1.25 + (0.0 if lattice == "32x32" else 15.0), 2)
        rate = round(0.40 + (index % 6) * 0.07, 2)
        response = (
            f"scr_{index:03d}.py を作成しました。\n"
            f"格子サイズは {lattice}、乱数種は {seed} を使用しました。\n"
            f"実行時間は {runtime} 秒。成功率は {rate}。\n"
            f"結論: {phrase} は温度条件に敏感。"
        )

        def executor(_r, _d, response=response):
            return response, trace_with(BehaviorEvent.FILE_WRITTEN)

        episode = episodic.record(run_goal_loop(f"scr{index:03d}: {phrase}のコードを作成して", executor))
        responses[episode.episode_id] = response
        rows.append(Ep(index, lattice, seed, runtime, rate))
    return build_store(episodic, responses), tuple(rows)


@dataclass(frozen=True)
class Probe:
    text: str
    family: str
    expected: object  # rendered AST, or "ABSTAIN" / "NOT_AN_OPERATION"
    note: str


def build_probes(rows) -> tuple[Probe, ...]:
    import statistics

    big = [r for r in rows if r.lattice == "64x64"]
    small = [r for r in rows if r.lattice == "32x32"]
    target_seed = big[1].seed
    best = max(big, key=lambda r: r.rate)

    def r(expr):
        return canonical_render(expr)

    return (
        Probe("格子サイズ64x64の実行時間の平均は？", "FIELD_THEN_VALUE",
              r(Mean(Project(Filter(Source(), "lattice_size", "64x64"), "runtime_seconds"))),
              "the shape L8.20 got wrong: filter field named before the projection field"),
        Probe("64x64を使った実験の平均成功率は？", "VALUE_ONLY",
              r(Mean(Project(Filter(Source(), "lattice_size", "64x64"), "success_rate"))),
              "no field names the filter; the value's domain has to supply it"),
        Probe(f"乱数種{target_seed}の実験の平均実行時間は？", "OTHER_FIELD_VALUE",
              r(Mean(Project(Filter(Source(), "seed", target_seed), "runtime_seconds"))),
              "the value belongs to seed, so the filter must bind there, not to lattice"),
        Probe("格子サイズ32x32と比べて格子サイズ64x64の実行時間の平均は大きい？", "NESTED_WITH_BINDING",
              r(Compare(
                  Mean(Project(Filter(Source(), "lattice_size", "32x32"), "runtime_seconds")),
                  Mean(Project(Filter(Source(), "lattice_size", "64x64"), "runtime_seconds")))),
              "composition and binding at once, both operands naming their filter field"),
        Probe("格子サイズ64x64の実験で成功率が一番高いのは？", "ARGMAX_BIND",
              r(Argmax(Filter(Source(), "lattice_size", "64x64"), "success_rate")),
              "ARGMAX orders by one field while another filters"),
        Probe("64x64を使った実験の実行時間の最大は？", "MAX_BIND",
              r(Max(Project(Filter(Source(), "lattice_size", "64x64"), "runtime_seconds"))),
              "MAX with a filter, binding unchanged"),

        Probe("実行時間と成功率の平均は？", "AMBIGUOUS_TWO_FIELDS", "ABSTAIN",
              "two Number fields for one projection hole; choosing silently is a wrong answer"),
        Probe("平均は？", "NO_FIELD", "ABSTAIN", "nothing determines the projection"),
        Probe("実験名の平均は？", "TYPE_ERROR", "ABSTAIN", "the named field cannot fill the hole"),

        Probe("さっきの続きをやって", "CONTROL_REFERENCE", "NOT_AN_OPERATION",
              "control: an ordinary reference must not become a query"),
        Probe("実行時間の中央値は？", "CONTROL_GATE", "ABSTAIN",
              "control: L8.18's gate still holds an unimplemented operation"),
    )


_MUST_ANSWER = {
    "FIELD_THEN_VALUE", "VALUE_ONLY", "OTHER_FIELD_VALUE",
    "NESTED_WITH_BINDING", "ARGMAX_BIND", "MAX_BIND",
}
_MUST_REFUSE = {"AMBIGUOUS_TWO_FIELDS", "NO_FIELD", "TYPE_ERROR"}


def _ratio(numerator, denominator):
    if denominator == 0:
        return "n/a"
    return round(numerator / denominator, 3)


def run_holdout(count: int = 24, bound: bool = True) -> dict[str, object]:
    store, rows = build_holdout(count)
    runner = run_bound_query if bound else run_query
    records = []
    for probe in build_probes(rows):
        result = runner(store, probe.text)
        answered = result.decision == "ANSWER"
        got = canonical_render(result.expr) if result.expr is not None else None
        if probe.expected == "ABSTAIN":
            correct, wrong = result.decision == "ABSTAIN", answered
        elif probe.expected == "NOT_AN_OPERATION":
            correct, wrong = result.decision == "NOT_AN_OPERATION", answered
        else:
            correct = answered and got == probe.expected
            wrong = answered and got != probe.expected
        shape_ok = (
            got is not None
            and isinstance(probe.expected, str)
            and probe.expected.split("(")[0] == got.split("(")[0]
        )
        records.append(
            {
                "family": probe.family,
                "text": probe.text,
                "note": probe.note,
                "decision": result.decision,
                "stage": result.stage,
                "got": got,
                "expected": probe.expected,
                "correct": correct,
                "wrong": wrong,
                "shape_ok": shape_ok,
                "abstained": result.decision != "ANSWER",
            }
        )

    answering = [r for r in records if r["family"] in _MUST_ANSWER]
    refusing = [r for r in records if r["family"] in _MUST_REFUSE]
    controls = [r for r in records if r["family"].startswith("CONTROL")]

    metrics = {
        "binding_accuracy": _ratio(sum(1 for r in answering if r["correct"]), len(answering)),
        "shape_accuracy": _ratio(sum(1 for r in answering if r["shape_ok"]), len(answering)),
        "wrong_answer_rate": _ratio(sum(1 for r in records if r["wrong"]), len(records)),
        "ambiguous_abstention_rate": _ratio(sum(1 for r in refusing if r["correct"]), len(refusing)),
        "unnecessary_abstention_rate": _ratio(
            sum(1 for r in answering if r["abstained"]), len(answering)
        ),
        "control_accuracy": _ratio(sum(1 for r in controls if r["correct"]), len(controls)),
    }
    return {"records": records, "metrics": metrics}


def run_experiment(count: int = 24) -> dict[str, object]:
    return {"l820": run_holdout(count, bound=False), "l821": run_holdout(count, bound=True)}


def format_experiment(report) -> str:
    lines = [
        "## Cold hold-out for argument binding (L8.21.1)", "",
        "| compiler | binding_acc | shape_acc | **wrong_answer** | ambiguous_abstention | unnecessary_abstention | control |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for label, key in (("L8.20", "l820"), ("L8.21", "l821")):
        m = report[key]["metrics"]
        lines.append(
            f"| {label} | {m['binding_accuracy']} | {m['shape_accuracy']} | **{m['wrong_answer_rate']}** "
            f"| {m['ambiguous_abstention_rate']} | {m['unnecessary_abstention_rate']} | {m['control_accuracy']} |"
        )
    lines += ["", "| family | L8.20 | L8.21 |", "| --- | --- | --- |"]
    by_family = {r["family"]: r for r in report["l820"]["records"]}
    for record in report["l821"]["records"]:
        other = by_family[record["family"]]
        lines.append(
            f"| {record['family']} | {other['decision']}{'' if other['correct'] else ' (NG)'} "
            f"| {record['decision']}{'' if record['correct'] else ' (NG)'} |"
        )
    return "\n".join(lines)


def main() -> None:  # pragma: no cover
    print(format_experiment(run_experiment()))


if __name__ == "__main__":  # pragma: no cover
    main()
