"""L8.24.1: the four families that separate arbitration from tidy merging.

L8.23.1 measured the two sensors as a tie with different failures, so the probes
here are built around *which* sensor can see each case:

* ``SPAN_ONLY`` -- two held-out forms in one request; the sentence reading
  contends and only the span is decisive;
* ``WHOLE_ONLY`` -- wording no span covers, where the sentence still leans;
* ``COMPLEMENTARY`` -- operator visible to the span, field only to the sentence,
  so the program has to be assembled from two sources;
* ``AMBIGUOUS`` / ``NO_EVIDENCE`` -- must refuse.

Solving the first three with a slot-wise union would be easy and wrong, which is
why refusal families and the strength requirement are carried alongside.

Beyond accuracy the run reports **capability complementarity**: the sets of
queries each pipeline answers correctly, and their differences.  L8.22 and L8.23
scored the same on fresh probes while disagreeing about which requests they
could handle, and a single accuracy number hides exactly that.
"""
from __future__ import annotations

from dataclasses import dataclass

from .evidence_arbitration_l824_experiment import run_arbitrated_query
from .holdout_binding_l8211_experiment import build_holdout
from .query_algebra_l820_experiment import (
    Count, Filter, Mean, Project, Source, canonical_render, infer,
)
from .semantic_proposer_l822_experiment import Proposer, run_proposed_query
from .span_grounded_proposer_l823_experiment import SpanProposer, run_span_query


@dataclass(frozen=True)
class Probe:
    text: str
    family: str
    expected: str
    note: str


def build_probes() -> tuple[Probe, ...]:
    mean_runtime = canonical_render(
        Mean(Project(Filter(Source(), "lattice_size", "64x64"), "runtime_seconds")))
    mean_rate = canonical_render(
        Mean(Project(Filter(Source(), "lattice_size", "64x64"), "success_rate")))
    count_big = canonical_render(Count(Filter(Source(), "lattice_size", "64x64")))
    return (
        Probe("64x64の分のランタイム量をアベレージ的に", "SPAN_ONLY", mean_runtime,
              "the sentence contends between MAX/MEAN; only the span is decisive"),
        Probe("64x64のケースをまとめて数えあげて", "WHOLE_ONLY", count_big,
              "no span clears coverage; the sentence still leans COUNT"),
        Probe("64x64のケースの成功という割合をならした", "COMPLEMENTARY", mean_rate,
              "operator from the span, field from the sentence"),
        Probe("64x64のケースのランタイム量をならしで", "COMPLEMENTARY", mean_runtime,
              "the other direction: field from the span, operator thin"),

        Probe("64x64のならならアベレのピークの値", "AMBIGUOUS", "ABSTAIN",
              "spans for MEAN and MAX both survive; neither may be chosen"),
        Probe("64x64のケースの処理時間をホニャララして", "NO_EVIDENCE", "NOT_AN_OPERATION",
              "no sensor has any support"),

        Probe("さっきの続きをやって", "CONTROL_REFERENCE", "NOT_AN_OPERATION",
              "control: an ordinary reference must not become a query"),
        Probe("実行時間の中央値は？", "CONTROL_GATE", "ABSTAIN",
              "control: L8.18's gate still holds an unimplemented operation"),
        Probe("64x64を使った実験の平均実行時間は？", "CONTROL_SYMBOLIC", mean_runtime,
              "control: a request the cue tables read never reaches a sensor"),
    )


_MUST_ANSWER = {"SPAN_ONLY", "WHOLE_ONLY", "COMPLEMENTARY", "CONTROL_SYMBOLIC"}
_MUST_REFUSE = {"AMBIGUOUS", "NO_EVIDENCE", "CONTROL_REFERENCE", "CONTROL_GATE"}

PIPELINES = ("l822", "l823", "l824")


def _ratio(numerator, denominator):
    if denominator == 0:
        return "n/a"
    return round(numerator / denominator, 3)


def run_pipeline(name: str, count: int = 24) -> dict[str, object]:
    store, _ = build_holdout(count)
    sentence = Proposer.trained()
    spans = SpanProposer.trained()

    records = []
    for probe in build_probes():
        if name == "l822":
            result = run_proposed_query(store, probe.text, sentence)
        elif name == "l823":
            result = run_span_query(store, probe.text, spans, sentence)
        else:
            result = run_arbitrated_query(store, probe.text, spans, sentence)

        answered = result.decision == "ANSWER"
        got = canonical_render(result.expr) if result.expr is not None else None
        if probe.expected == "ABSTAIN":
            correct, wrong = result.decision == "ABSTAIN", answered
        elif probe.expected == "NOT_AN_OPERATION":
            correct, wrong = result.decision == "NOT_AN_OPERATION", answered
        else:
            correct = answered and got == probe.expected
            wrong = answered and got != probe.expected

        unsafe = False
        if answered and result.expr is not None:
            try:
                infer(result.expr)
            except Exception:  # pragma: no cover
                unsafe = True
        if answered and probe.family == "CONTROL_GATE":
            unsafe = True

        records.append({
            "family": probe.family, "text": probe.text, "note": probe.note,
            "decision": result.decision, "source": result.source, "got": got,
            "correct": correct, "wrong": wrong, "unsafe": unsafe,
        })

    answering = [r for r in records if r["family"] in _MUST_ANSWER]
    refusing = [r for r in records if r["family"] in _MUST_REFUSE]
    correct = sum(1 for r in records if r["correct"])
    wrong = sum(1 for r in records if r["wrong"])

    metrics = {
        "execution_accuracy": _ratio(sum(1 for r in answering if r["correct"]), len(answering)),
        "refusal_accuracy": _ratio(sum(1 for r in refusing if r["correct"]), len(refusing)),
        "wrong_answer_rate": _ratio(wrong, len(records)),
        "unsafe_execution_rate": _ratio(sum(1 for r in records if r["unsafe"]), len(records)),
        "safe_coverage_gain": round((correct - wrong) / len(records), 3),
    }
    return {"records": records, "metrics": metrics,
            "solved": frozenset(r["text"] for r in records if r["correct"])}


def run_experiment(count: int = 24) -> dict[str, object]:
    runs = {name: run_pipeline(name, count) for name in PIPELINES}
    complementarity = {}
    for left in PIPELINES:
        for right in PIPELINES:
            if left >= right:
                continue
            a, b = runs[left]["solved"], runs[right]["solved"]
            complementarity[f"{left}\\{right}"] = len(a - b)
            complementarity[f"{right}\\{left}"] = len(b - a)
            complementarity[f"{left}&{right}"] = len(a & b)
    return {"runs": runs, "complementarity": complementarity}


def format_experiment(report) -> str:
    lines = ["## Slot-wise evidence arbitration (L8.24.1)", "",
             "| pipeline | execution | refusal | **wrong_answer** | **unsafe** | safe_coverage_gain |",
             "| --- | --- | --- | --- | --- | --- |"]
    for name in PIPELINES:
        m = report["runs"][name]["metrics"]
        lines.append(
            f"| {name.upper()} | {m['execution_accuracy']} | {m['refusal_accuracy']} "
            f"| **{m['wrong_answer_rate']}** | **{m['unsafe_execution_rate']}** | {m['safe_coverage_gain']} |")
    lines += ["", "【capability complementarity】", ""]
    for key, value in report["complementarity"].items():
        lines.append(f"- |C({key})| = `{value}`")
    lines += ["", "| family | L8.22 | L8.23 | L8.24 | witness |", "| --- | --- | --- | --- | --- |"]
    for index, probe in enumerate(build_probes()):
        cells = []
        for name in PIPELINES:
            record = report["runs"][name]["records"][index]
            cells.append(f"{record['decision']}{'' if record['correct'] else ' (NG)'}")
        witness = report["runs"]["l824"]["records"][index]["source"]
        lines.append(f"| {probe.family} | " + " | ".join(cells) + f" | {witness} |")
    return "\n".join(lines)


def main() -> None:  # pragma: no cover
    print(format_experiment(run_experiment()))


if __name__ == "__main__":  # pragma: no cover
    main()
