"""L8.27.1: a cold hold-out that the previous two layers fail on.

The families are the shapes an edge can have and a set cannot.  ``UNNAMED``
governs an argument the lexical sensors cannot read; ``OBLIQUE`` puts the legible
field behind ではなく / のかわりに so proximity points at the wrong one;
``COORDINATION`` gives the head a plural argument the algebra has no node for;
``MODIFIER`` hangs the argument off a light noun so a の-chain has to be closed
before anything attaches.  The controls are the same four L8.26.1 carried.

Two disclosures, because both matter for reading the numbers.

**The hand-written families are development-informed.**  The wrong answers that
motivated this layer were found on these sentences, so their L8.24/L8.26 column
is a diagnosis, not an independent estimate.  The ``generated`` sweep exists for
that reason: it enumerates five shapes over held-out operator and field forms
with no sentence chosen by hand, and the ground truth is the program the
generator rendered.  Read the generated table as the estimate and the family
table as the explanation.

**One L8.26.1 expectation is reclassified.**  That hold-out scores
「64x64のケースの成功比率を処理時間のかわりにならした」 as a required refusal, on the
grounds that two field candidates were each strongly attached and neither
dominated.  With を exposed, they are not symmetric -- 成功比率 is the argument and
処理時間 is what it replaces -- so L8.27 answers it, correctly.  L8.26.1 is left
frozen and untouched; it measures L8.24 against L8.26, and both are unchanged.
What moved is the claim that the sentence is inherently ambiguous, which was a
statement about the sensors of the time and not about the sentence.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product

from .evidence_arbitration_l824_experiment import run_arbitrated_query
from .evidence_graph_l827_experiment import build_graph, run_graph_query
from .holdout_binding_l8211_experiment import build_holdout
from .query_algebra_l820_experiment import (
    Count, Filter, Max, Mean, Project, Source, canonical_render, infer,
)
from .semantic_proposer_l822_experiment import FIELD_FORMS, OPERATOR_FORMS, Proposer
from .span_grounded_proposer_l823_experiment import SpanProposer
from .structural_sensor_l826_experiment import StructuralSensor, run_structural_query

_PIPELINES = ("l824", "l826", "l827")


def _run(name, store, text, spans, sentence, structure):
    if name == "l824":
        return run_arbitrated_query(store, text, spans, sentence)
    if name == "l826":
        return run_structural_query(store, text, spans, sentence, structure)
    return run_graph_query(store, text, spans, sentence, structure)


@dataclass(frozen=True)
class Probe:
    text: str
    family: str
    expected: str  # rendered AST, or "ABSTAIN" / "NOT_AN_OPERATION"
    note: str


def _mean(field: str, value: str = "64x64") -> str:
    return canonical_render(Mean(Project(Filter(Source(), "lattice_size", value), field)))


def _max(field: str, value: str = "64x64") -> str:
    return canonical_render(Max(Project(Filter(Source(), "lattice_size", value), field)))


def build_probes() -> tuple[Probe, ...]:
    runtime, rate = _mean("runtime_seconds"), _mean("success_rate")
    return (
        Probe("64x64のケースでランタイムの値ではなく成功の比率をならして", "UNNAMED", "ABSTAIN",
              "the governed argument is unreadable; the legible field is the excluded one"),
        Probe("処理にかかった時間を見ながら64x64のケースの成功の比率をならして", "UNNAMED", "ABSTAIN",
              "the legible field sits in a subordinate clause the head does not govern"),

        Probe("64x64のケースで処理時間ではなく成功比率をならした", "OBLIQUE", rate,
              "を marks the argument; ではなく marks what it is not"),
        Probe("64x64のケースの成功比率を処理時間のかわりにならした", "OBLIQUE", rate,
              "reclassified from L8.26.1's CONFLICT: のかわりに is oblique, not symmetric"),

        Probe("64x64のケースで処理時間と成功比率をならした", "COORDINATION", "ABSTAIN",
              "one coordinated argument naming two fields; the algebra has no plural node"),

        Probe("64x64のケースのサクセスレートの値をアベレージ", "MODIFIER", rate,
              "の closes into one phrase, so the light noun 値 does not win the attachment"),
        Probe("64x64のケースのランタイム量をピークの値で", "MODIFIER", _max("runtime_seconds"),
              "the same chain with the operator itself behind a modifier"),

        Probe("64x64のケースの処理時間をホニャララして", "NO_EVIDENCE", "NOT_AN_OPERATION",
              "control: no sensor has support"),
        Probe("さっきの続きをやって", "CONTROL_REFERENCE", "NOT_AN_OPERATION",
              "control: an ordinary reference must not become a query"),
        Probe("実行時間の中央値は？", "CONTROL_GATE", "ABSTAIN",
              "control: L8.18's gate still holds an unimplemented operation"),
        Probe("64x64を使った実験の平均実行時間は？", "CONTROL_SYMBOLIC", runtime,
              "control: the cue tables read this; no sensor is consulted"),
    )


_MUST_REFUSE = {"UNNAMED", "COORDINATION", "NO_EVIDENCE", "CONTROL_REFERENCE", "CONTROL_GATE"}


# --------------------------------------------------------------------------
# the generated sweep: no sentence chosen by hand
# --------------------------------------------------------------------------

_SHAPES = ("direct", "oblique", "replace", "modifier", "coordination")


def generate_probes(limit: int = 2) -> tuple[Probe, ...]:
    """Five shapes over held-out forms; ground truth is the rendered program."""
    operators = {
        label: split["holdout"][:limit] for label, split in OPERATOR_FORMS.items()
        if label in ("MEAN", "MAX")
    }
    fields = {
        label: split["holdout"][:limit] for label, split in FIELD_FORMS.items()
        if label in ("runtime_seconds", "success_rate")
    }
    build = {"MEAN": Mean, "MAX": Max}

    probes: list[Probe] = []
    for (op_label, op_forms), (field_label, field_forms) in product(
            operators.items(), fields.items()):
        other = next(name for name in fields if name != field_label)
        distractor = fields[other][0]
        program = canonical_render(build[op_label](
            Project(Filter(Source(), "lattice_size", "64x64"), field_label)))
        for op_form, field_form, shape in product(op_forms, field_forms, _SHAPES):
            if shape == "direct":
                text = f"64x64のケースの{field_form}を{op_form}"
                expected = program
            elif shape == "oblique":
                text = f"64x64のケースで{distractor}ではなく{field_form}を{op_form}"
                expected = program
            elif shape == "replace":
                text = f"64x64のケースの{field_form}を{distractor}のかわりに{op_form}"
                expected = program
            elif shape == "modifier":
                text = f"64x64のケースの{field_form}の値を{op_form}"
                expected = program
            else:
                text = f"64x64のケースで{distractor}と{field_form}を{op_form}"
                expected = "ABSTAIN"
            probes.append(Probe(text, shape, expected, f"{op_label}/{field_label}"))
    return tuple(dict.fromkeys(probes))


def _ratio(numerator, denominator):
    if denominator == 0:
        return "n/a"
    return round(numerator / denominator, 3)


def _score(probe: Probe, result) -> dict[str, object]:
    answered = result.decision == "ANSWER"
    got = canonical_render(result.expr) if result.expr is not None else None
    if probe.expected in ("ABSTAIN", "NOT_AN_OPERATION"):
        correct, wrong = result.decision == probe.expected, answered
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
    return {
        "family": probe.family, "text": probe.text, "note": probe.note,
        "decision": result.decision, "source": result.source, "got": got,
        "answerable": probe.expected not in ("ABSTAIN", "NOT_AN_OPERATION"),
        "correct": correct, "wrong": wrong, "unsafe": unsafe,
        "answered": answered,
    }


def _metrics(records) -> dict[str, object]:
    answerable = [r for r in records if r["answerable"]]
    refusing = [r for r in records if not r["answerable"]]
    answered = [r for r in records if r["answered"]]
    conflict = [r for r in records if r["family"] in ("COORDINATION", "coordination")]
    return {
        "coverage": _ratio(sum(1 for r in answerable if r["correct"]), len(answerable)),
        "wrong_answer_rate": _ratio(sum(1 for r in records if r["wrong"]), len(records)),
        "unnecessary_abstention_rate": _ratio(
            sum(1 for r in answerable if not r["answered"]), len(answerable)),
        "calibration": _ratio(sum(1 for r in answered if r["correct"]), len(answered)),
        "conflict_abstention": _ratio(sum(1 for r in conflict if r["correct"]), len(conflict)),
        "refusal_accuracy": _ratio(sum(1 for r in refusing if r["correct"]), len(refusing)),
        "unsafe_execution_rate": _ratio(sum(1 for r in records if r["unsafe"]), len(records)),
    }


def _sensors():
    return SpanProposer.trained(), Proposer.trained(), StructuralSensor()


def run_suite(probes, count: int = 24) -> dict[str, object]:
    store, _ = build_holdout(count)
    spans, sentence, structure = _sensors()
    runs = {}
    for name in _PIPELINES:
        records = [_score(probe, _run(name, store, probe.text, spans, sentence, structure))
                   for probe in probes]
        runs[name] = {
            "records": records,
            "solved": frozenset(r["text"] for r in records if r["correct"]),
            "metrics": _metrics(records),
        }
    return runs


def run_experiment(count: int = 24) -> dict[str, object]:
    families = run_suite(build_probes(), count)
    generated = run_suite(generate_probes(), count)
    store, _ = build_holdout(count)
    spans, sentence, structure = _sensors()
    graphs = {
        probe.text: build_graph(probe.text, store, spans, sentence, structure).render()
        for probe in build_probes()
    }
    complementarity = {
        f"{a}\\{b}": len(families[a]["solved"] | generated[a]["solved"]
                         - (families[b]["solved"] | generated[b]["solved"]))
        for a, b in (("l827", "l826"), ("l826", "l827"), ("l827", "l824"))
    }
    return {"families": families, "generated": generated, "graphs": graphs,
            "complementarity": complementarity, "probe_count": len(generate_probes())}


_AXES = ("coverage", "wrong_answer_rate", "unnecessary_abstention_rate",
         "calibration", "conflict_abstention")


def format_experiment(report) -> str:
    lines = ["## Evidence graph (L8.27.1)", ""]
    for title, key in (("generated sweep (%d probes, no sentence chosen by hand)"
                        % report["probe_count"], "generated"),
                       ("hand-written families (development-informed; see the docstring)",
                        "families")):
        lines += [f"### {title}", "",
                  "| pipeline | coverage | **wrong_answer** | unnecessary_abstention "
                  "| calibration | conflict_abstention |", "| --- | --- | --- | --- | --- | --- |"]
        for name in _PIPELINES:
            m = report[key][name]["metrics"]
            lines.append(
                f"| {name.upper()} | {m['coverage']} | **{m['wrong_answer_rate']}** "
                f"| {m['unnecessary_abstention_rate']} | {m['calibration']} "
                f"| {m['conflict_abstention']} |")
        lines.append("")

    lines += ["### complementarity |C_a \\ C_b|", ""]
    for pair, size in report["complementarity"].items():
        lines.append(f"- `{pair}` = {size}")

    lines += ["", "### per family", "",
              "| family | L8.24 | L8.26 | L8.27 |", "| --- | --- | --- | --- |"]
    for index, probe in enumerate(build_probes()):
        cells = []
        for name in _PIPELINES:
            record = report["families"][name]["records"][index]
            cells.append(f"{record['decision']}{'' if record['correct'] else ' (NG)'}")
        lines.append(f"| {probe.family} | " + " | ".join(cells) + " |")

    lines += ["", "### graphs", ""]
    for text, drawing in report["graphs"].items():
        lines.append(f"- `{text}`")
        lines.append(f"    - {drawing}")
    return "\n".join(lines)


def main() -> None:  # pragma: no cover
    print(format_experiment(run_experiment()))


if __name__ == "__main__":  # pragma: no cover
    main()
