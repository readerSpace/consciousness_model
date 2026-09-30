"""L8.20.1: programs as ground truth, and composition as the held-out axis.

Natural language is a poor gold standard -- "is this the right meaning" has no
sharp answer.  A program does: generate the program first, render it into
Japanese, compile the Japanese back, and compare the two trees.  Agreement is
exact AST equality, and the value is checked against an oracle computed
directly from the generated episodes rather than by running the same evaluator
twice.

**This is not a learning result, and the distinction matters.** The compiler in
L8.20 is hand-written, so "held out" cannot mean "absent from training data".
It means absent from the cases the compiler was written against.  Four shapes
were in front of me while writing it -- COUNT(FILTER), MEAN(PROJECT(FILTER)),
COMPARE(MEAN, MEAN) and MODE(PROJECT) -- and the rest were never tried:
comparing counts instead of means, MAX where MEAN had been, ARGMAX under a
filter, two filters stacked on different fields.  Every one of them is reachable
from the same grammar, so a recursive compiler should handle them for free.  The
question this file answers is whether it actually does, or whether the nesting
fix was special-cased around the one example that motivated it.  That is a
falsifiable question about the implementation, not a claim about generalisation
in a learned model; L8.21 is where a learned component would make it one.

English renderings are included as a separate family.  The cue tables are
Japanese-only, so the expectation is that they do not compile -- what is
measured is that they fail without computing anything, not that they work.
"""
from __future__ import annotations

from dataclasses import dataclass, field as dc_field
import statistics

from .holdout_composition_l8191_experiment import build_holdout_store
from .query_algebra_l820_experiment import (
    Argmax,
    Compare,
    Count,
    Expr,
    Filter,
    Max,
    Mean,
    Mode,
    Project,
    Source,
    canonical_render,
    render,
    run_query,
)

#: Shapes the compiler was written against.  Everything else is held out.
DEVELOPMENT_SHAPES = frozenset(
    {"COUNT(FILTER)", "MEAN(PROJECT(FILTER))", "COMPARE(MEAN,MEAN)", "MODE(PROJECT)"}
)


@dataclass(frozen=True)
class ProgramProbe:
    shape: str
    program: Expr
    renderings: tuple[str, ...]
    expected_value: object
    language: str = "ja"
    note: str = ""


def _mean(values):
    return round(statistics.fmean(values), 4)


def build_probes(episodes) -> tuple[ProgramProbe, ...]:
    def where(lattice=None):
        return [e for e in episodes if lattice is None or e.lattice == lattice]

    big, small = where("64x64"), where("32x32")
    seed_of_first_big = f"{20260000 + big[0].index}"
    best_big = max(big, key=lambda e: e.rate)
    counts = {size: len(where(size)) for size in {e.lattice for e in episodes}}
    top = max(counts, key=lambda key: counts[key])

    probes = [
        # --- shapes the compiler was written against -------------------------
        ProgramProbe(
            "COUNT(FILTER)",
            Count(Filter(Source(), "lattice_size", "64x64")),
            ("64x64を使った実験はいくつ？", "格子サイズ64x64の実験は何件？"),
            len(big),
        ),
        ProgramProbe(
            "MEAN(PROJECT(FILTER))",
            Mean(Project(Filter(Source(), "lattice_size", "64x64"), "runtime_seconds")),
            ("64x64を使った実験の平均実行時間は？", "格子サイズ64x64の実行時間の平均を教えて"),
            _mean([e.runtime for e in big]),
        ),
        ProgramProbe(
            "COMPARE(MEAN,MEAN)",
            Compare(
                Mean(Project(Filter(Source(), "lattice_size", "32x32"), "runtime_seconds")),
                Mean(Project(Filter(Source(), "lattice_size", "64x64"), "runtime_seconds")),
            ),
            ("32x32と比べて64x64の平均実行時間は大きい？",),
            "greater" if _mean([e.runtime for e in big]) > _mean([e.runtime for e in small]) else "less",
        ),
        ProgramProbe(
            "MODE(PROJECT)",
            Mode(Project(Source(), "lattice_size")),
            ("一番多く使った格子サイズは？",),
            top,
        ),
        # --- held out: same grammar, combinations never written against -------
        ProgramProbe(
            "COMPARE(COUNT,COUNT)",
            Compare(
                Count(Filter(Source(), "lattice_size", "32x32")),
                Count(Filter(Source(), "lattice_size", "64x64")),
            ),
            ("32x32と比べて64x64の実験は何件？",),
            "greater" if len(big) > len(small) else ("less" if len(big) < len(small) else "equal"),
            note="COMPARE over counts rather than means",
        ),
        ProgramProbe(
            "MAX(PROJECT(FILTER))",
            Max(Project(Filter(Source(), "lattice_size", "64x64"), "runtime_seconds")),
            ("64x64を使った実験の実行時間の最大は？",),
            max(e.runtime for e in big),
            note="MAX where only MEAN had been exercised",
        ),
        ProgramProbe(
            "ARGMAX(FILTER)",
            Argmax(Filter(Source(), "lattice_size", "64x64"), "success_rate"),
            ("64x64を使った実験で成功率が一番高いのは？",),
            f"ep-{best_big.index + 1:03d}",
            note="ARGMAX under a filter",
        ),
        ProgramProbe(
            "COUNT(FILTER(FILTER))",
            Count(Filter(Filter(Source(), "lattice_size", "64x64"), "seed", seed_of_first_big)),
            (f"64x64で乱数種{seed_of_first_big}の実験はいくつ？",),
            len([e for e in big if f"{20260000 + e.index}" == seed_of_first_big]),
            note="two filters stacked on different fields",
        ),
        ProgramProbe(
            "MEAN(PROJECT)",
            Mean(Project(Source(), "success_rate")),
            ("平均成功率は？",),
            _mean([e.rate for e in episodes]),
            note="aggregate with no filter at all",
        ),
        # --- language coverage, measured rather than assumed ------------------
        ProgramProbe(
            "COUNT(FILTER)",
            Count(Filter(Source(), "lattice_size", "64x64")),
            ("How many experiments used a 64x64 lattice?",),
            len(big),
            language="en",
            note="cue tables are Japanese-only; must fail without computing",
        ),
    ]
    return tuple(probes)


def _ratio(numerator, denominator):
    if denominator == 0:
        return "n/a"
    return round(numerator / denominator, 3)


def run_holdout(count: int = 24, bound: bool = False) -> dict[str, object]:
    """``bound`` swaps in the L8.21 compiler so the two are measured on one set."""
    from .argument_binding_l821_experiment import run_bound_query

    runner = run_bound_query if bound else run_query
    store, episodes = build_holdout_store(count)
    records = []
    for probe in build_probes(episodes):
        for text in probe.renderings:
            result = runner(store, text)
            ast_match = result.expr is not None and canonical_render(result.expr) == canonical_render(probe.program)
            answered = result.decision == "ANSWER"
            value_match = answered and str(result.value) == str(probe.expected_value)
            records.append(
                {
                    "shape": probe.shape,
                    "held_out": probe.shape not in DEVELOPMENT_SHAPES,
                    "language": probe.language,
                    "text": text,
                    "note": probe.note,
                    "decision": result.decision,
                    "ast_match": ast_match,
                    "value_match": value_match,
                    "wrong": answered and not value_match,
                    "abstained": result.decision != "ANSWER",
                    "got": render(result.expr) if result.expr else result.reason,
                }
            )

    japanese = [r for r in records if r["language"] == "ja"]
    held_out = [r for r in japanese if r["held_out"]]
    development = [r for r in japanese if not r["held_out"]]
    english = [r for r in records if r["language"] == "en"]

    metrics = {
        "ast_exact_match": _ratio(sum(1 for r in japanese if r["ast_match"]), len(japanese)),
        "development_shape_accuracy": _ratio(
            sum(1 for r in development if r["ast_match"]), len(development)
        ),
        "held_out_shape_accuracy": _ratio(sum(1 for r in held_out if r["ast_match"]), len(held_out)),
        "execution_match": _ratio(sum(1 for r in japanese if r["value_match"]), len(japanese)),
        "wrong_answer_rate": _ratio(sum(1 for r in records if r["wrong"]), len(records)),
        "unnecessary_abstention_rate": _ratio(
            sum(1 for r in japanese if r["abstained"]), len(japanese)
        ),
        "unsupported_language_wrong_rate": _ratio(
            sum(1 for r in english if r["wrong"]), len(english)
        ),
    }
    return {"records": records, "metrics": metrics}


def format_experiment(report) -> str:
    lines = ["## Programs as ground truth (L8.20.1)", "", "【Metrics】", ""]
    for key, value in report["metrics"].items():
        lines.append(f"- {key}: `{value}`")
    lines += ["", "| shape | held out | lang | AST | value | decision |", "| --- | --- | --- | --- | --- | --- |"]
    for record in report["records"]:
        lines.append(
            f"| {record['shape']} | {'yes' if record['held_out'] else 'no'} | {record['language']} "
            f"| {'OK' if record['ast_match'] else 'NG'} | {'OK' if record['value_match'] else 'NG'} "
            f"| {record['decision']} |"
        )
    return "\n".join(lines)


def main() -> None:  # pragma: no cover
    print(format_experiment(run_holdout()))


if __name__ == "__main__":  # pragma: no cover
    main()
