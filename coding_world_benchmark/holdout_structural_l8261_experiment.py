"""L8.26.1: the four cells, and what each sensor is actually worth.

The families are chosen so that a pipeline cannot pass by being good at one
thing.  ``STRUCTURAL_ONLY`` needs a witness the lexical pair does not have,
``LEXICAL_ONLY`` needs one structure cannot supply, ``CONFLICT`` has to be
refused, and the controls have to behave exactly as they did five layers ago.

``CONFLICT`` is the cell that did not exist before.  L8.24 could only exercise
disagreement on constructed evidence, because two lexical sensors reading the
same string agree or fall silent together.  With a sensor that reads relations
instead of vocabulary, 「成功比率を処理時間のかわりにならした」 produces two field
candidates that are each strongly supported and neither dominated, so the
request is refused on a real sentence rather than in a unit test.

Ablation reports what each sensor contributes: ``ΔC_s = C_all \\ C_-s``.  One
honest caveat about it -- the structural sensor locates candidates through the
span sensor's boundaries, using their positions but never their labels, so
removing spans also removes structure's ability to attach to anything.  ΔC for
the span sensor is inflated by that coupling and is not a clean independent
contribution; ΔC for whole and for structure are.
"""
from __future__ import annotations

from dataclasses import dataclass

from .evidence_arbitration_l824_experiment import run_arbitrated_query
from .holdout_binding_l8211_experiment import build_holdout
from .query_algebra_l820_experiment import (
    Filter, Mean, Project, Source, canonical_render, infer,
)
from .semantic_proposer_l822_experiment import NgramClassifier, Proposer
from .span_grounded_proposer_l823_experiment import SpanProposer
from .structural_sensor_l826_experiment import StructuralSensor, run_structural_query


class _NoSpans(SpanProposer):
    def detect(self, text):  # noqa: D102
        return ()

    def propose(self, text):  # noqa: D102
        return (), ()


class _NoWhole(Proposer):
    def operators(self, text):  # noqa: D102
        return ()

    def fields(self, text):  # noqa: D102
        return ()


class _NoStructure(StructuralSensor):
    def bindings(self, text):  # noqa: D102
        return ()


@dataclass(frozen=True)
class Probe:
    text: str
    family: str
    expected: str
    note: str


def build_probes() -> tuple[Probe, ...]:
    runtime = canonical_render(
        Mean(Project(Filter(Source(), "lattice_size", "64x64"), "runtime_seconds")))
    rate = canonical_render(
        Mean(Project(Filter(Source(), "lattice_size", "64x64"), "success_rate")))
    return (
        Probe("64x64のケースの処理時間をならす形に", "LEXICAL_ONLY", runtime,
              "one field span; structure adds nothing the lexical pair lacked"),
        Probe("64x64のケースをカウントする流れで", "LEXICAL_ONLY",
              canonical_render(__import__(
                  "coding_world_benchmark.query_algebra_l820_experiment",
                  fromlist=["Count"]).Count(Filter(Source(), "lattice_size", "64x64"))),
              "operator only, no argument to attach"),

        Probe("64x64のケースで処理時間ではなく成功比率をならした", "STRUCTURAL_ONLY", rate,
              "two field spans; only attachment says which the head governs"),
        Probe("64x64のケースで成功比率ではなく処理時間をならした", "STRUCTURAL_ONLY", runtime,
              "the same shape with the answer on the other side"),

        Probe("64x64のケースの成功比率を処理時間のかわりにならした", "CONFLICT", "ABSTAIN",
              "two candidates, both strongly attached, neither dominated"),

        Probe("64x64のケースの処理時間をホニャララして", "NO_EVIDENCE", "NOT_AN_OPERATION",
              "control: no sensor has support"),
        Probe("さっきの続きをやって", "CONTROL_REFERENCE", "NOT_AN_OPERATION",
              "control: an ordinary reference must not become a query"),
        Probe("実行時間の中央値は？", "CONTROL_GATE", "ABSTAIN",
              "control: L8.18's gate still holds an unimplemented operation"),
        Probe("64x64を使った実験の平均実行時間は？", "CONTROL_SYMBOLIC", runtime,
              "control: the cue tables read this; no sensor is consulted"),
    )


_MUST_ANSWER = {"LEXICAL_ONLY", "STRUCTURAL_ONLY", "CONTROL_SYMBOLIC"}
_MUST_REFUSE = {"CONFLICT", "NO_EVIDENCE", "CONTROL_REFERENCE", "CONTROL_GATE"}


def _ratio(numerator, denominator):
    if denominator == 0:
        return "n/a"
    return round(numerator / denominator, 3)


def _sensors(ablate: str | None):
    operators, fields = __import__(
        "coding_world_benchmark.semantic_proposer_l822_experiment",
        fromlist=["build_training_examples"]).build_training_examples()
    span_cls = _NoSpans if ablate == "span" else SpanProposer
    whole_cls = _NoWhole if ablate == "whole" else Proposer
    structure_cls = _NoStructure if ablate == "structure" else StructuralSensor
    from .span_grounded_proposer_l823_experiment import build_aligned_training

    span_pairs, field_pairs = build_aligned_training()
    return (
        span_cls(NgramClassifier.train(span_pairs), NgramClassifier.train(field_pairs)),
        whole_cls(NgramClassifier.train(operators), NgramClassifier.train(fields)),
        structure_cls(),
    )


def run_pipeline(name: str, count: int = 24, ablate: str | None = None) -> dict[str, object]:
    store, _ = build_holdout(count)
    spans, sentence, structure = _sensors(ablate)

    records = []
    for probe in build_probes():
        if name == "l824":
            result = run_arbitrated_query(store, probe.text, spans, sentence)
        else:
            result = run_structural_query(store, probe.text, spans, sentence, structure)

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
            "decision": result.decision, "source": result.source,
            "correct": correct, "wrong": wrong, "unsafe": unsafe,
        })

    answering = [r for r in records if r["family"] in _MUST_ANSWER]
    refusing = [r for r in records if r["family"] in _MUST_REFUSE]
    return {
        "records": records,
        "solved": frozenset(r["text"] for r in records if r["correct"]),
        "metrics": {
            "execution_accuracy": _ratio(sum(1 for r in answering if r["correct"]), len(answering)),
            "refusal_accuracy": _ratio(sum(1 for r in refusing if r["correct"]), len(refusing)),
            "wrong_answer_rate": _ratio(sum(1 for r in records if r["wrong"]), len(records)),
            "unsafe_execution_rate": _ratio(sum(1 for r in records if r["unsafe"]), len(records)),
        },
    }


def run_experiment(count: int = 24) -> dict[str, object]:
    runs = {"l824": run_pipeline("l824", count), "l826": run_pipeline("l826", count)}
    ablations = {
        sensor: run_pipeline("l826", count, ablate=sensor)
        for sensor in ("span", "whole", "structure")
    }
    contribution = {
        sensor: sorted(runs["l826"]["solved"] - run["solved"])
        for sensor, run in ablations.items()
    }
    structure_scores = sorted(
        {bind.strength for probe in build_probes() for bind in StructuralSensor().bindings(probe.text)}
    )
    return {
        "runs": runs, "ablations": ablations, "contribution": contribution,
        "structure_score_values": structure_scores,
    }


def format_experiment(report) -> str:
    lines = ["## Structural evidence (L8.26.1)", "",
             "| pipeline | execution | refusal | **wrong_answer** | **unsafe** |",
             "| --- | --- | --- | --- | --- |"]
    for name in ("l824", "l826"):
        m = report["runs"][name]["metrics"]
        lines.append(f"| {name.upper()} | {m['execution_accuracy']} | {m['refusal_accuracy']} "
                     f"| **{m['wrong_answer_rate']}** | **{m['unsafe_execution_rate']}** |")
    lines += ["", "【ablation: ΔC_s = C_all \\ C_-s】", ""]
    for sensor, lost in report["contribution"].items():
        caveat = "  (inflated: structure attaches through span boundaries)" if sensor == "span" else ""
        lines.append(f"- without **{sensor}**: loses `{len(lost)}`{caveat}")
        for text in lost:
            lines.append(f"    - {text}")
    lines += ["", "【structural strengths observed】", "",
              "- " + ", ".join(str(value) for value in report["structure_score_values"])]
    lines += ["", "| family | L8.24 | L8.26 | witness |", "| --- | --- | --- | --- |"]
    for index, probe in enumerate(build_probes()):
        left = report["runs"]["l824"]["records"][index]
        right = report["runs"]["l826"]["records"][index]
        lines.append(
            f"| {probe.family} | {left['decision']}{'' if left['correct'] else ' (NG)'} "
            f"| {right['decision']}{'' if right['correct'] else ' (NG)'} | {right['source']} |")
    return "\n".join(lines)


def main() -> None:  # pragma: no cover
    print(format_experiment(run_experiment()))


if __name__ == "__main__":  # pragma: no cover
    main()
