"""L8.25.1: does the estimate agree with the constant, and where does it not?

The question is not "is calibration better".  It is whether the number L8.24
hard-codes is where the data says the boundary is, and whether it survives a
change of vocabulary.  So the run does three things:

1. replays the L8.24 probes with calibrated admission, to check nothing is lost;
2. runs the same shapes in an **entangled** vocabulary, where labels share
   substrings and the score distribution slides;
3. reports the table itself, because its diagnostic content is the result.

The measured outcome is reported as it came out, including a null: on every
probe reachable here the two admission rules agree.  The whole-sentence score
turns out to be close to bimodal -- near 1.0 when one label has evidence, near
0.5 when two contend -- so the band where they would differ is barely populated.
What calibration buys today is therefore not coverage.  It is that the boundary
is derived from outcomes rather than read off a plot, that it is re-derived per
vocabulary instead of inherited, and that it states in numbers two things a
single global threshold cannot: reliability differs by sensor, and it differs by
vocabulary.
"""
from __future__ import annotations

from dataclasses import dataclass

from .evidence_arbitration_l824_experiment import run_arbitrated_query
from .evidence_calibration_l825_experiment import (
    ENTANGLED, MIN_SUPPORT, RELIABILITY_FLOOR, SEPARATED, band, calibrate, run_calibrated_query,
)
from .holdout_binding_l8211_experiment import build_holdout
from .query_algebra_l820_experiment import (
    Count, Filter, Mean, Project, Source, canonical_render, infer,
)

WHOLE_STRONG_L824 = 0.75


@dataclass(frozen=True)
class Probe:
    text: str
    family: str
    expected: str
    note: str


def _targets():
    return (
        canonical_render(Mean(Project(Filter(Source(), "lattice_size", "64x64"), "runtime_seconds"))),
        canonical_render(Mean(Project(Filter(Source(), "lattice_size", "64x64"), "success_rate"))),
        canonical_render(Count(Filter(Source(), "lattice_size", "64x64"))),
    )


def separated_probes() -> tuple[Probe, ...]:
    runtime, rate, count = _targets()
    return (
        Probe("64x64のケースの処理時間をならす形に", "PLAIN", runtime, "both sensors agree"),
        Probe("64x64のケースの成功比率をならした", "PLAIN", rate, "a second field"),
        Probe("64x64のケースをカウントする流れで", "PLAIN", count, "operator only"),
        Probe("64x64で走らせた分のランタイムの値をアベレージ", "DILUTED", runtime,
              "the contended sentence reading must stay out"),
        Probe("64x64のケースの処理時間をピークをならして", "AMBIGUOUS", "ABSTAIN",
              "two operator spans survive"),
        Probe("さっきの続きをやって", "CONTROL_REFERENCE", "NOT_AN_OPERATION", "control"),
        Probe("実行時間の中央値は？", "CONTROL_GATE", "ABSTAIN", "control"),
    )


def entangled_probes() -> tuple[Probe, ...]:
    runtime, rate, count = _targets()
    return (
        Probe("64x64のケースの実行時間を平坦化", "PLAIN", runtime, "shared 実行 across field labels"),
        Probe("64x64のケースの実行成功率を平坦化して集計", "PLAIN", rate, "shared 集計 across operators"),
        Probe("64x64のケースを個数集計", "PLAIN", count, "operator only, entangled vocabulary"),
        Probe("さっきの続きをやって", "CONTROL_REFERENCE", "NOT_AN_OPERATION", "control"),
    )


def _ratio(numerator, denominator):
    if denominator == 0:
        return "n/a"
    return round(numerator / denominator, 3)


def _score(probe: Probe, result) -> tuple[bool, bool, bool]:
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
    return correct, wrong, unsafe


def run_holdout(count: int = 24) -> dict[str, object]:
    store, _ = build_holdout(count)
    tables = {bank.name: calibrate(bank, store) for bank in (SEPARATED, ENTANGLED)}

    runs: dict[str, dict] = {}
    for bank, probes in ((SEPARATED, separated_probes()), (ENTANGLED, entangled_probes())):
        sensors = bank.sensors()
        rows = []
        for probe in probes:
            calibrated = run_calibrated_query(store, probe.text, bank, tables[bank.name], sensors)
            fixed = (
                run_arbitrated_query(store, probe.text, *sensors)
                if bank is SEPARATED
                else None
            )
            correct, wrong, unsafe = _score(probe, calibrated)
            row = {
                "family": probe.family, "text": probe.text, "note": probe.note,
                "decision": calibrated.decision, "source": calibrated.source,
                "correct": correct, "wrong": wrong, "unsafe": unsafe,
                "fixed_decision": fixed.decision if fixed else None,
                "agrees_with_fixed": fixed is None or fixed.decision == calibrated.decision,
            }
            rows.append(row)
        runs[bank.name] = {
            "records": rows,
            "metrics": {
                "accuracy": _ratio(sum(1 for r in rows if r["correct"]), len(rows)),
                "wrong_answer_rate": _ratio(sum(1 for r in rows if r["wrong"]), len(rows)),
                "unsafe_execution_rate": _ratio(sum(1 for r in rows if r["unsafe"]), len(rows)),
                "agreement_with_fixed_threshold": _ratio(
                    sum(1 for r in rows if r["agrees_with_fixed"]), len(rows)
                ),
            },
        }

    diagnostics = {}
    for name, table in tables.items():
        entries = {
            key: round(table.reliability(key), 3)
            for key in table.total
            if table.total[key] >= MIN_SUPPORT and table.reliability(key) is not None
        }
        diagnostics[name] = entries
    return {"runs": runs, "tables": tables, "diagnostics": diagnostics}


def format_experiment(report) -> str:
    lines = ["## Calibrated admission (L8.25.1)", "",
             "| vocabulary | accuracy | **wrong_answer** | **unsafe** | agreement with fixed 0.75 |",
             "| --- | --- | --- | --- | --- |"]
    for name, run in report["runs"].items():
        m = run["metrics"]
        lines.append(
            f"| {name} | {m['accuracy']} | **{m['wrong_answer_rate']}** | "
            f"**{m['unsafe_execution_rate']}** | {m['agreement_with_fixed_threshold']} |")
    lines += ["", "【estimated reliability】", ""]
    for name, entries in report["diagnostics"].items():
        lines.append(f"- **{name}**")
        for key, value in sorted(entries.items()):
            sensor, slot, score_band, coverage_band = key
            admitted = "admit" if value >= RELIABILITY_FLOOR else "reject"
            fixed = ""
            if sensor == "whole":
                fixed = " | fixed-0.75: " + ("admit" if score_band >= WHOLE_STRONG_L824 else "reject")
            lines.append(
                f"    - {sensor}/{slot} score≥{score_band} cov≥{coverage_band}: "
                f"`{value}` -> {admitted}{fixed}")
    return "\n".join(lines)


def main() -> None:  # pragma: no cover
    print(format_experiment(run_holdout()))


if __name__ == "__main__":  # pragma: no cover
    main()
