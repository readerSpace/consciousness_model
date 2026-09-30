"""L8.28.1: how good the hypotheses are, and what promoting them would cost.

The population this layer addresses is the residual L8.27 left behind: requests
whose argument was located and refused because nothing could name it.  So the
probes are built to *produce* that state, along two axes that separate the two
things that could be doing the work.

**Overlap.**  ``OVERLAP_*`` families use real held-out forms that share characters
with the trained ones -- 成功の比率 against 成功比率.  ``ZERO_*`` families use
invented forms with no shared n-gram at all, checked by
``assert_zero_overlap`` rather than asserted in prose.  A hypothesis that
survives in a ZERO family was not read off the spelling.

**Contrast.**  ``*_CONTRAST`` families put the unknown phrase in 「Xではなく Y」 or
「Yを Xのかわりに」, where a labelled sibling excludes one candidate.  ``*_BARE``
families remove that, leaving type and attestation alone.

The cell that matters is ``ZERO_CONTRAST``: no overlap, so spelling says nothing,
and the only remaining evidence is the operator's type hole plus a construction
that says "not the other one".  If a unique correct hypothesis survives there,
structure alone named a word the system has never seen.

Three numbers keep the claim honest.

``false_hypothesis_rate``
    Hypotheses produced for a request that had no unnamed argument.  Must be 0:
    a generator that speculates about phrases the sensors read perfectly well is
    not a generator, it is noise.

``execution_invariance``
    Every decision identical to L8.27's.  This layer generates and promotes
    nothing, so a difference here is a bug, not a result.

``decidable_precision``
    Of the cases where exactly one hypothesis survives, how often is it right.
    This is the **cost a promotion stage would inherit**, measured here without
    paying it -- which is the entire reason generation and promotion were split.

One suspicion about that last number is worth testing rather than caveating.
The schema attests exactly two Number fields, so "exclude one by contrast" and
"name the other" are the same operation, and a decidable hypothesis is correct
for a reason that may not survive a wider schema.  ``widen_store`` attests a
third Number field (``energy_error``, declared since L8.19 and never emitted) and
the whole sweep is re-run against it.  If decidability holds up, the mechanism is
doing the work; if it collapses, the 1.0 above was counting the schema.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product

from .evidence_graph_l827_experiment import run_graph_query
from .holdout_binding_l8211_experiment import build_holdout
from .lexical_hypothesis_l828_experiment import (
    SOURCES, run_hypothesis_query, subword_overlap,
)
from .semantic_proposer_l822_experiment import FIELD_FORMS, OPERATOR_FORMS, Proposer
from .typed_operation_l819_experiment import TypedStore
from .span_grounded_proposer_l823_experiment import SpanProposer
from .structural_sensor_l826_experiment import StructuralSensor

#: Invented surface forms with no character n-gram in common with any trained
#: form.  Checked, not asserted -- see ``assert_zero_overlap``.
INVENTED: dict[str, tuple[str, ...]] = {
    "runtime_seconds": ("ホゲ尺", "ヌルポ幅"),
    "success_rate": ("フガ率", "ピヨピヨ度"),
}

#: Held-out real forms the span sensor cannot read: the inserted の breaks every
#: trained n-gram, so they land in ``UNKNOWN`` while still sharing characters.
UNREADABLE: dict[str, tuple[str, ...]] = {
    "runtime_seconds": ("処理にかかる間",),
    "success_rate": ("成功の比率", "成功の割合"),
}


def assert_zero_overlap() -> None:
    """The invented forms must carry no subword evidence at all.

    Without this the ZERO families would silently become weak OVERLAP families
    and the layer's central claim would be unfalsifiable.
    """
    spans = SpanProposer.trained()
    for label, forms in INVENTED.items():
        for form in forms:
            found = subword_overlap(form, spans)
            if found:
                raise AssertionError(f"{form} ({label}) overlaps {found}")


def widen_store(store: TypedStore) -> TypedStore:
    """Attest a third Number field, so contrast no longer decides by elimination."""
    widened = {
        episode_id: dict(values, energy_error=round(0.001 * (index + 1), 4))
        for index, (episode_id, values) in enumerate(store.fields.items())
    }
    return TypedStore(store.episodic, widened)


@dataclass(frozen=True)
class Probe:
    text: str
    family: str
    truth: str | None  # the field the unknown phrase denotes, or None when named
    note: str


def _sibling(label: str) -> str:
    return "success_rate" if label == "runtime_seconds" else "runtime_seconds"


def build_probes() -> tuple[Probe, ...]:
    """Four cells plus the controls that must generate nothing."""
    readable = {"runtime_seconds": "処理時間", "success_rate": "成功比率"}
    probes: list[Probe] = []
    for bank, overlap in ((UNREADABLE, "OVERLAP"), (INVENTED, "ZERO")):
        for label, forms in bank.items():
            other = readable[_sibling(label)]
            for form in forms:
                probes.append(Probe(
                    f"64x64のケースで{other}ではなく{form}をならして",
                    f"{overlap}_CONTRAST", label,
                    "a labelled sibling excludes one candidate"))
                probes.append(Probe(
                    f"{other}を見ながら64x64のケースの{form}をならして",
                    f"{overlap}_BARE", label,
                    "the sibling is present but in no contrastive relation"))
    probes += [
        Probe("64x64のケースの処理時間をならす形に", "CONTROL_READABLE", None,
              "control: the sensors read this; no hypothesis may be produced"),
        Probe("64x64のケースのサクセスレートの値をアベレージ", "CONTROL_READABLE", None,
              "control: a modifier chain the graph resolves"),
        Probe("さっきの続きをやって", "CONTROL_REFERENCE", None,
              "control: not a query at all"),
        Probe("実行時間の中央値は？", "CONTROL_GATE", None,
              "control: L8.18's gate refuses before any of this runs"),
        Probe("64x64を使った実験の平均実行時間は？", "CONTROL_SYMBOLIC", None,
              "control: the cue tables read this; no sensor is consulted"),
    ]
    return tuple(probes)


def _controls() -> tuple[Probe, ...]:
    """Requests with nothing unnamed. ``false_hypothesis_rate`` is measured on these."""
    return tuple(probe for probe in build_probes() if probe.truth is None)


def generate_probes(limit: int = 2) -> tuple[Probe, ...]:
    """The same four cells enumerated over operators and both fields."""
    operators = tuple(OPERATOR_FORMS["MEAN"]["holdout"][:limit]) + tuple(
        OPERATOR_FORMS["MAX"]["holdout"][:limit])
    readable = {label: FIELD_FORMS[label]["train"][0]
                for label in ("runtime_seconds", "success_rate")}
    probes: list[Probe] = []
    for bank, overlap in ((UNREADABLE, "OVERLAP"), (INVENTED, "ZERO")):
        for (label, forms), operator in product(bank.items(), operators):
            other = readable[_sibling(label)]
            for form in forms:
                probes.append(Probe(
                    f"64x64のケースで{other}ではなく{form}を{operator}",
                    f"{overlap}_CONTRAST", label, f"{operator}"))
                probes.append(Probe(
                    f"64x64のケースの{form}を{other}のかわりに{operator}",
                    f"{overlap}_CONTRAST", label, f"{operator} (swapped terms)"))
                probes.append(Probe(
                    f"{other}を見ながら64x64のケースの{form}を{operator}",
                    f"{overlap}_BARE", label, f"{operator}"))
    # The controls ride along so ``false_hypothesis_rate`` is a measured number
    # on this suite rather than "n/a" for want of anything to divide by.
    probes += list(_controls())
    return tuple(dict.fromkeys(probes))


def _ratio(numerator, denominator):
    if denominator == 0:
        return "n/a"
    return round(numerator / denominator, 3)


def _score(probe: Probe, outcome, baseline) -> dict[str, object]:
    labels = [item.label for item in outcome.hypotheses]
    rank = labels.index(probe.truth) + 1 if probe.truth in labels else 0
    return {
        "family": probe.family, "text": probe.text, "truth": probe.truth,
        "decision": outcome.result.decision,
        "unnamed": bool(outcome.hypotheses),
        "expected_unnamed": probe.truth is not None,
        "recalled": rank > 0,
        "top1": rank == 1,
        "reciprocal": 1.0 / rank if rank else 0.0,
        "size": len(labels),
        "decidable": outcome.decidable,
        "decidable_correct": outcome.decidable and rank == 1,
        "invariant": outcome.result.decision == baseline.decision
                     and outcome.result.reason == baseline.reason,
        "wrong": outcome.result.decision == "ANSWER" and probe.truth is not None,
        "hypotheses": [str(item) for item in outcome.hypotheses],
    }


def _metrics(records) -> dict[str, object]:
    unnamed = [r for r in records if r["expected_unnamed"]]
    named = [r for r in records if not r["expected_unnamed"]]
    decidable = [r for r in unnamed if r["decidable"]]
    return {
        "unnamed_rate": _ratio(sum(1 for r in unnamed if r["unnamed"]), len(unnamed)),
        "hypothesis_recall": _ratio(sum(1 for r in unnamed if r["recalled"]), len(unnamed)),
        "top1_accuracy": _ratio(sum(1 for r in unnamed if r["top1"]), len(unnamed)),
        "mrr": _ratio(sum(r["reciprocal"] for r in unnamed), len(unnamed)),
        "mean_set_size": _ratio(sum(r["size"] for r in unnamed), len(unnamed)),
        "decidable_rate": _ratio(len(decidable), len(unnamed)),
        "decidable_precision": _ratio(
            sum(1 for r in decidable if r["decidable_correct"]), len(decidable)),
        "false_hypothesis_rate": _ratio(sum(1 for r in named if r["unnamed"]), len(named)),
        "execution_invariance": _ratio(sum(1 for r in records if r["invariant"]), len(records)),
        "wrong_answer_rate": _ratio(sum(1 for r in records if r["wrong"]), len(records)),
    }


def run_suite(probes, count: int = 24, disable: tuple[str, ...] = (),
              widen: bool = False) -> dict[str, object]:
    store, _ = build_holdout(count)
    if widen:
        store = widen_store(store)
    spans, sentence, structure = SpanProposer.trained(), Proposer.trained(), StructuralSensor()
    records = []
    for probe in probes:
        baseline = run_graph_query(store, probe.text, spans, sentence, structure)
        outcome = run_hypothesis_query(store, probe.text, spans, sentence, structure, disable)
        records.append(_score(probe, outcome, baseline))
    return {"records": records, "metrics": _metrics(records)}


def by_family(records) -> dict[str, dict[str, object]]:
    families: dict[str, list] = {}
    for record in records:
        families.setdefault(record["family"], []).append(record)
    return {name: _metrics(rows) for name, rows in sorted(families.items())}


def run_experiment(count: int = 24) -> dict[str, object]:
    assert_zero_overlap()
    families = run_suite(build_probes(), count)
    generated = run_suite(generate_probes(), count)
    ablations = {
        source: run_suite(generate_probes(), count, disable=(source,))
        for source in SOURCES
    }
    widened = run_suite(generate_probes(), count, widen=True)
    return {
        "families": families,
        "generated": generated,
        "generated_by_family": by_family(generated["records"]),
        "widened": widened,
        "widened_by_family": by_family(widened["records"]),
        "ablations": ablations,
        "probe_count": len(generate_probes()),
    }


_HEADLINE = ("hypothesis_recall", "top1_accuracy", "mrr", "decidable_rate",
             "decidable_precision", "mean_set_size")


def format_experiment(report) -> str:
    generated = report["generated"]["metrics"]
    lines = [
        "## Lexical hypothesis generation (L8.28.1)", "",
        f"### generated sweep ({report['probe_count']} probes)", "",
        "| metric | value |", "| --- | --- |",
    ]
    for key in _HEADLINE:
        lines.append(f"| {key} | {generated[key]} |")
    lines += [
        f"| **false_hypothesis_rate** | **{generated['false_hypothesis_rate']}** |",
        f"| **execution_invariance** | **{generated['execution_invariance']}** |",
        f"| **wrong_answer_rate** | **{generated['wrong_answer_rate']}** |",
        "", "### by cell", "",
        "| family | recall | top1 | decidable | decidable_precision |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name, metrics in report["generated_by_family"].items():
        lines.append(f"| {name} | {metrics['hypothesis_recall']} | {metrics['top1_accuracy']} "
                     f"| {metrics['decidable_rate']} | {metrics['decidable_precision']} |")

    lines += ["", "### ablation (source removed)", "",
              "| removed | recall | top1 | decidable | decidable_precision |",
              "| --- | --- | --- | --- | --- |",
              f"| _(none)_ | {generated['hypothesis_recall']} | {generated['top1_accuracy']} "
              f"| {generated['decidable_rate']} | {generated['decidable_precision']} |"]
    for source, run in report["ablations"].items():
        metrics = run["metrics"]
        lines.append(f"| {source} | {metrics['hypothesis_recall']} | {metrics['top1_accuracy']} "
                     f"| {metrics['decidable_rate']} | {metrics['decidable_precision']} |")

    lines += ["", "### a third attested Number field (schema widened)", "",
              "| schema | decidable_rate | decidable_precision | top1 | recall |",
              "| --- | --- | --- | --- | --- |"]
    for name, key in (("2 Number fields", "generated"), ("3 Number fields", "widened")):
        metrics = report[key]["metrics"]
        lines.append(f"| {name} | {metrics['decidable_rate']} "
                     f"| {metrics['decidable_precision']} | {metrics['top1_accuracy']} "
                     f"| {metrics['hypothesis_recall']} |")
    lines += ["", "| family | decidable (2) | decidable (3) |", "| --- | --- | --- |"]
    for name in report["generated_by_family"]:
        left = report["generated_by_family"][name]["decidable_rate"]
        right = report["widened_by_family"].get(name, {}).get("decidable_rate", "n/a")
        lines.append(f"| {name} | {left} | {right} |")

    lines += ["", "### hand-written families", "",
              "| family | decision | hypotheses |", "| --- | --- | --- |"]
    for record in report["families"]["records"]:
        drawn = " / ".join(record["hypotheses"]) or "_(none)_"
        lines.append(f"| {record['family']} | {record['decision']} | {drawn} |")
    return "\n".join(lines)


def main() -> None:  # pragma: no cover
    print(format_experiment(run_experiment()))


if __name__ == "__main__":  # pragma: no cover
    main()
