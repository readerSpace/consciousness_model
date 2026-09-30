"""L8.22.1: language and composition held out independently.

Two things can be unfamiliar about a request, and lumping them together hides
which one the learner actually helped with.  So the probes form a 2x2 -- the
wording is either in the cue tables or only in the proposer's held-out forms,
and the shape is either a flat aggregate or the nested comparison -- plus the
families that exist to stop the guard being graded generously:

* ``AMBIGUOUS_UNSEEN`` -- 「ピークをならして」 carries subword evidence for MAX and for
  MEAN at once.  Both type-check over a Number field, so two well-typed programs
  survive and the request must be refused.  A proposer with authority would run
  whichever scored higher.
* ``UNKNOWN_WORD`` -- wording with no subword overlap with anything trained.
  The honest outcome is no coverage at all, not a guess.
* the reference and gate controls, unchanged from earlier layers.

L8.21 is run on the same probes.  The claim is not a higher score; it is that
every point gained is a refusal turned into a correct execution, with
``wrong_answer_rate`` still 0.00 -- which is what
``safe_coverage_gain = P(correct) - P(wrong)`` is there to state in one number.
"""
from __future__ import annotations

from dataclasses import dataclass

from .argument_binding_l821_experiment import run_bound_query
from .holdout_binding_l8211_experiment import build_holdout
from .query_algebra_l820_experiment import (
    Compare, Count, Filter, Mean, Project, Source, canonical_render, infer,
)
from .semantic_proposer_l822_experiment import Proposer, run_proposed_query


@dataclass(frozen=True)
class Probe:
    text: str
    family: str
    expected: str  # a canonical program, "ABSTAIN", or "NOT_AN_OPERATION"
    note: str


def build_probes() -> tuple[Probe, ...]:
    mean_runtime_big = canonical_render(
        Mean(Project(Filter(Source(), "lattice_size", "64x64"), "runtime_seconds"))
    )
    nested = canonical_render(
        Compare(
            Mean(Project(Filter(Source(), "lattice_size", "32x32"), "runtime_seconds")),
            Mean(Project(Filter(Source(), "lattice_size", "64x64"), "runtime_seconds")),
        )
    )
    count_big = canonical_render(Count(Filter(Source(), "lattice_size", "64x64")))

    return (
        Probe("64x64を使った実験の平均実行時間は？", "KNOWN_LANG_KNOWN_SHAPE",
              mean_runtime_big, "baseline: both compilers read this"),
        Probe("64x64を使った実験はいくつ？", "KNOWN_LANG_KNOWN_SHAPE",
              count_big, "baseline: a second operator"),

        Probe("64x64で走らせたケースについて処理時間をならして", "UNSEEN_LANG_KNOWN_SHAPE",
              mean_runtime_big, "ならして and 処理時間 are both outside the cue tables"),
        Probe("64x64のケースの数を教えて", "UNSEEN_LANG_KNOWN_SHAPE",
              count_big, "数を教えて generalises from 数えて"),
        Probe("64x64で走らせた分のランタイムの値をアベレージ", "UNSEEN_LANG_KNOWN_SHAPE",
              mean_runtime_big, "two held-out forms at once"),

        Probe("32x32と比べて64x64の平均実行時間は大きい？", "KNOWN_LANG_UNSEEN_SHAPE",
              nested, "composition L8.20 introduced; must not regress"),

        Probe("32x32と比べて64x64の処理時間をならしたら大きい？", "UNSEEN_LANG_UNSEEN_SHAPE",
              nested, "unseen wording inside an unseen shape"),

        Probe("64x64のケースの処理時間をピークをならして", "AMBIGUOUS_UNSEEN", "ABSTAIN",
              "evidence for MAX and MEAN at once; two well-typed programs survive"),
        Probe("64x64のケースの処理時間をブリブリして", "UNKNOWN_WORD", "NOT_AN_OPERATION",
              "no subword overlap with anything trained; no coverage is the honest outcome"),

        Probe("さっきの続きをやって", "CONTROL_REFERENCE", "NOT_AN_OPERATION",
              "control: an ordinary reference must not become a query"),
        Probe("実行時間の中央値は？", "CONTROL_GATE", "ABSTAIN",
              "control: L8.18's gate still holds an unimplemented operation"),
    )


_UNSEEN_LANGUAGE = {"UNSEEN_LANG_KNOWN_SHAPE", "UNSEEN_LANG_UNSEEN_SHAPE"}
_UNSEEN_SHAPE = {"KNOWN_LANG_UNSEEN_SHAPE", "UNSEEN_LANG_UNSEEN_SHAPE"}
_MUST_REFUSE = {"AMBIGUOUS_UNSEEN", "UNKNOWN_WORD", "CONTROL_REFERENCE", "CONTROL_GATE"}


def _ratio(numerator, denominator):
    if denominator == 0:
        return "n/a"
    return round(numerator / denominator, 3)


def run_holdout(count: int = 24, proposed: bool = True) -> dict[str, object]:
    store, _ = build_holdout(count)
    proposer = Proposer.trained() if proposed else None

    records = []
    for probe in build_probes():
        result = (
            run_proposed_query(store, probe.text, proposer)
            if proposed
            else run_bound_query(store, probe.text)
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
            except Exception:  # pragma: no cover - would mean the type pass was bypassed
                unsafe = True
        if answered and probe.family == "CONTROL_GATE":
            unsafe = True

        records.append(
            {
                "family": probe.family,
                "text": probe.text,
                "note": probe.note,
                "decision": result.decision,
                "source": getattr(result, "source", "symbolic"),
                "got": got,
                "correct": correct,
                "wrong": wrong,
                "unsafe": unsafe,
            }
        )

    unseen_language = [r for r in records if r["family"] in _UNSEEN_LANGUAGE]
    unseen_shape = [r for r in records if r["family"] in _UNSEEN_SHAPE]
    refusing = [r for r in records if r["family"] in _MUST_REFUSE]
    correct = sum(1 for r in records if r["correct"])
    wrong = sum(1 for r in records if r["wrong"])

    metrics = {
        "unseen_language_execution_rate": _ratio(
            sum(1 for r in unseen_language if r["correct"]), len(unseen_language)
        ),
        "unseen_composition_accuracy": _ratio(
            sum(1 for r in unseen_shape if r["correct"]), len(unseen_shape)
        ),
        "wrong_answer_rate": _ratio(wrong, len(records)),
        "unsafe_execution_rate": _ratio(sum(1 for r in records if r["unsafe"]), len(records)),
        "refusal_accuracy": _ratio(sum(1 for r in refusing if r["correct"]), len(refusing)),
        "safe_coverage_gain": round((correct - wrong) / len(records), 3),
    }
    return {"records": records, "metrics": metrics}


def run_experiment(count: int = 24) -> dict[str, object]:
    return {"l821": run_holdout(count, proposed=False), "l822": run_holdout(count, proposed=True)}


def format_experiment(report) -> str:
    lines = [
        "## Language and composition held out independently (L8.22.1)", "",
        "| compiler | unseen_language | unseen_composition | **wrong_answer** | **unsafe_execution** | refusal | safe_coverage_gain |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for label, key in (("L8.21", "l821"), ("L8.22", "l822")):
        m = report[key]["metrics"]
        lines.append(
            f"| {label} | {m['unseen_language_execution_rate']} | {m['unseen_composition_accuracy']} "
            f"| **{m['wrong_answer_rate']}** | **{m['unsafe_execution_rate']}** | {m['refusal_accuracy']} "
            f"| {m['safe_coverage_gain']} |"
        )
    lines += ["", "| family | L8.21 | L8.22 | source |", "| --- | --- | --- | --- |"]
    previous = {}
    for record in report["l821"]["records"]:
        previous.setdefault(record["family"], []).append(record)
    seen: dict[str, int] = {}
    for record in report["l822"]["records"]:
        index = seen.get(record["family"], 0)
        seen[record["family"]] = index + 1
        other = previous[record["family"]][index]
        lines.append(
            f"| {record['family']} | {other['decision']}{'' if other['correct'] else ' (NG)'} "
            f"| {record['decision']}{'' if record['correct'] else ' (NG)'} | {record['source']} |"
        )
    return "\n".join(lines)


def main() -> None:  # pragma: no cover
    print(format_experiment(run_experiment()))


if __name__ == "__main__":  # pragma: no cover
    main()
