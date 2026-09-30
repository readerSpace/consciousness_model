"""L8.23.1: fresh probes for span grounding, with L8.22 kept as the control.

L8.22.1 became development data the moment its refused probe was used to design
this layer, so the wordings here are new and the two proposers are run on the
same set.  What has to show up is not simply a higher score:

* the dilution case has to execute, and the source has to be the span -- if the
  sentence reading still carried it, grounding did nothing;
* the thin-overlap case has to keep executing from the **sentence**, because
  span evidence is stricter and refusing it would be a regression traded for
  the fix;
* the ambiguous and no-evidence families have to refuse exactly as before, and
  `wrong_answer_rate` has to stay at 0.00.

`span_grounded_rate` reports which witness carried each execution, so "spans
helped" is a measured attribution rather than an inference from the totals.
"""
from __future__ import annotations

from dataclasses import dataclass

from .holdout_binding_l8211_experiment import build_holdout
from .query_algebra_l820_experiment import (
    Compare, Count, Filter, Mean, Project, Source, canonical_render, infer,
)
from .semantic_proposer_l822_experiment import Proposer, run_proposed_query
from .span_grounded_proposer_l823_experiment import SpanProposer, run_span_query


@dataclass(frozen=True)
class Probe:
    text: str
    family: str
    expected: str
    note: str


def _programs():
    mean_runtime = canonical_render(
        Mean(Project(Filter(Source(), "lattice_size", "64x64"), "runtime_seconds")))
    mean_rate = canonical_render(
        Mean(Project(Filter(Source(), "lattice_size", "64x64"), "success_rate")))
    count_big = canonical_render(Count(Filter(Source(), "lattice_size", "64x64")))
    nested = canonical_render(Compare(
        Mean(Project(Filter(Source(), "lattice_size", "32x32"), "runtime_seconds")),
        Mean(Project(Filter(Source(), "lattice_size", "64x64"), "runtime_seconds"))))
    return mean_runtime, mean_rate, count_big, nested


def build_probes() -> tuple[Probe, ...]:
    mean_runtime, mean_rate, count_big, nested = _programs()
    return (
        Probe("64x64の分のランタイム値をアベレージで", "SPAN_FLAT", mean_runtime,
              "two held-out forms in one request: the case whole-sentence scoring diluted"),
        Probe("64x64のケースの成功の比率をならした値は？", "SPAN_FLAT", mean_rate,
              "a different field, same shape"),
        Probe("64x64で走らせた一連のケースをまとめてカウントを出して", "SPAN_FILLER", count_big,
              "long filler around one operator span"),
        Probe("32x32と比べて64x64のランタイム値をならした値は大きい？", "SPAN_NESTED", nested,
              "held-out wording inside the nested shape"),

        Probe("64x64のケースの数えた結果は？", "SENTENCE_FALLBACK", count_big,
              "thin overlap: no span clears coverage, the sentence still leans COUNT"),

        Probe("64x64のケースのランタイム値をピークの値でならした値", "AMBIGUOUS", "ABSTAIN",
              "spans for MAX and MEAN both survive; neither may be chosen"),
        Probe("64x64のケースの処理時間をモニョモニョして", "NO_EVIDENCE", "NOT_AN_OPERATION",
              "no overlap anywhere; no coverage is the honest outcome"),

        Probe("さっきの続きをやって", "CONTROL_REFERENCE", "NOT_AN_OPERATION",
              "control: an ordinary reference must not become a query"),
        Probe("実行時間の中央値は？", "CONTROL_GATE", "ABSTAIN",
              "control: L8.18's gate still holds an unimplemented operation"),
        Probe("64x64を使った実験の平均実行時間は？", "CONTROL_SYMBOLIC", mean_runtime,
              "control: a request the cue tables read must not reach a proposer at all"),
    )


_MUST_ANSWER = {"SPAN_FLAT", "SPAN_FILLER", "SPAN_NESTED", "SENTENCE_FALLBACK", "CONTROL_SYMBOLIC"}
_MUST_REFUSE = {"AMBIGUOUS", "NO_EVIDENCE", "CONTROL_REFERENCE", "CONTROL_GATE"}


def _ratio(numerator, denominator):
    if denominator == 0:
        return "n/a"
    return round(numerator / denominator, 3)


def run_holdout(count: int = 24, grounded: bool = True) -> dict[str, object]:
    store, _ = build_holdout(count)
    sentence = Proposer.trained()
    spans = SpanProposer.trained() if grounded else None

    records = []
    for probe in build_probes():
        result = (
            run_span_query(store, probe.text, spans, sentence)
            if grounded
            else run_proposed_query(store, probe.text, sentence)
        )
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
    executed = [r for r in records if r["decision"] == "ANSWER"]
    correct = sum(1 for r in records if r["correct"])
    wrong = sum(1 for r in records if r["wrong"])

    metrics = {
        "execution_accuracy": _ratio(sum(1 for r in answering if r["correct"]), len(answering)),
        "refusal_accuracy": _ratio(sum(1 for r in refusing if r["correct"]), len(refusing)),
        "wrong_answer_rate": _ratio(wrong, len(records)),
        "unsafe_execution_rate": _ratio(sum(1 for r in records if r["unsafe"]), len(records)),
        "span_grounded_rate": _ratio(sum(1 for r in executed if r["source"] == "span"), len(executed)),
        "safe_coverage_gain": round((correct - wrong) / len(records), 3),
    }
    return {"records": records, "metrics": metrics}


def run_experiment(count: int = 24) -> dict[str, object]:
    return {"l822": run_holdout(count, grounded=False), "l823": run_holdout(count, grounded=True)}


def format_experiment(report) -> str:
    lines = ["## Span-grounded proposal, fresh probes (L8.23.1)", "",
             "| proposer | execution | refusal | **wrong_answer** | **unsafe** | span_grounded | safe_coverage_gain |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for label, key in (("L8.22", "l822"), ("L8.23", "l823")):
        m = report[key]["metrics"]
        lines.append(
            f"| {label} | {m['execution_accuracy']} | {m['refusal_accuracy']} | **{m['wrong_answer_rate']}** "
            f"| **{m['unsafe_execution_rate']}** | {m['span_grounded_rate']} | {m['safe_coverage_gain']} |")
    lines += ["", "| family | L8.22 | L8.23 | witness |", "| --- | --- | --- | --- |"]
    previous = {}
    for record in report["l822"]["records"]:
        previous.setdefault(record["family"], []).append(record)
    seen = {}
    for record in report["l823"]["records"]:
        index = seen.get(record["family"], 0)
        seen[record["family"]] = index + 1
        other = previous[record["family"]][index]
        lines.append(
            f"| {record['family']} | {other['decision']}{'' if other['correct'] else ' (NG)'} "
            f"| {record['decision']}{'' if record['correct'] else ' (NG)'} | {record['source']} |")
    return "\n".join(lines)


def main() -> None:  # pragma: no cover
    print(format_experiment(run_experiment()))


if __name__ == "__main__":  # pragma: no cover
    main()
