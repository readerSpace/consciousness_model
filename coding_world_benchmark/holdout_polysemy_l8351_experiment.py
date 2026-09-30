"""L8.35.1: four ways evidence can disagree, and only one of them is polysemy.

The families exist because ``contradiction => polysemy`` is the implementation
this suite has to be able to fail.

``TRUE_POLYSEMY``    the sense really does depend on the episode
``NOISY_MONOSEMY``   one meaning, one broken observation
``REVISION``         the meaning changed, and *time* explains it, not context
``UNIDENTIFIABLE``   two senses that no available observation separates
``AMBIGUOUS``        the contexts happen to be time-ordered, so neither variable wins
``MONOSEMY``         the control: nothing disagrees, so nothing may be split

``false_sense_split_rate`` is the number that matters. Adding a sense whenever
evidence disagrees reaches perfect accuracy by never being wrong about anything,
so a suite that only contains genuine polysemy measures nothing at all.

The ablation is the strongest claim available here. Running the same evidence
with the context labels erased must **break** -- not merely do worse. If
``TRUE_POLYSEMY`` becomes inexplicable and the request abstains once context is
removed, then context was not a convenience that improved things; it was the
variable the meaning depended on, and the episode boundary L8.13 introduced
twenty-two layers ago is where it comes from.
"""
from __future__ import annotations

from dataclasses import dataclass

from .contextual_polysemy_l835_experiment import (
    AMBIGUOUS, DISPUTED, MONOSEMY, NOISE, POLYSEMOUS, POLYSEMY, REVISION, SINGLE,
    UNEXPLAINED, UNIDENTIFIABLE, ContextualLexicon, ContextualObservation, explain,
)
from .cross_context_identification_l829_experiment import (
    Observation, available_probes, universe_from,
)
from .holdout_binding_l8211_experiment import build_holdout
from .holdout_lexicon_l8281_experiment import widen_store
from .typed_operation_l819_experiment import ValueType

SURFACE = "フガ率"


def _equals(name: str) -> Observation:
    return Observation("EQUALS", name)


@dataclass(frozen=True)
class Family:
    name: str
    evidence: tuple[tuple[Observation, str, int], ...]
    expect_kind: str
    #: context -> meaning that a correct reading should produce
    expect: dict[str | None, str | None]
    truly_polysemous: bool
    note: str


def build_families() -> tuple[Family, ...]:
    rate, energy = _equals("success_rate"), _equals("energy_error")
    return (
        Family("TRUE_POLYSEMY",
               ((rate, "sim", 0), (energy, "lab", 1), (rate, "sim", 2), (energy, "lab", 3)),
               POLYSEMY, {"sim": "success_rate", "lab": "energy_error"}, True,
               "contexts interleave in time, so time cannot explain the split"),
        Family("NOISY_MONOSEMY",
               ((rate, "sim", 0), (rate, "lab", 1), (energy, "sim", 2), (rate, "lab", 3)),
               NOISE, {"sim": "success_rate", "lab": "success_rate"}, False,
               "one odd observation; the odd block would have size one"),
        Family("REVISION",
               ((rate, "sim", 0), (rate, "lab", 1), (energy, "sim", 2), (energy, "lab", 3)),
               REVISION, {"sim": "energy_error", "lab": "energy_error"}, False,
               "both contexts appear on both sides, so context cannot explain it"),
        Family("AMBIGUOUS",
               ((rate, "sim", 0), (rate, "sim", 1), (energy, "lab", 2), (energy, "lab", 3)),
               AMBIGUOUS, {"sim": None, "lab": None}, False,
               "the contexts happen to be time-ordered; neither variable wins"),
        Family("MONOSEMY",
               ((rate, "sim", 0), (rate, "lab", 1), (rate, "sim", 2)),
               MONOSEMY, {"sim": "success_rate", "lab": "success_rate"}, False,
               "control: nothing disagrees, so nothing may be split"),
    )


def _ratio(numerator, denominator):
    if denominator == 0:
        return "n/a"
    return round(numerator / denominator, 3)


def _observations(family: Family) -> tuple[ContextualObservation, ...]:
    return tuple(ContextualObservation(o, c, t) for o, c, t in family.evidence)


def run_family(family: Family, universe) -> dict:
    found = explain(_observations(family), universe)

    lexicon = ContextualLexicon(universe)
    for observation, context, time in family.evidence:
        lexicon.observe(SURFACE, observation, context, time)
    readings = {context: lexicon.meaning(SURFACE, context)
                for context in family.expect}

    # The ablation: the same evidence with the context labels erased.
    erased = tuple(ContextualObservation(item.observation, "-", item.time)
                   for item in _observations(family))
    without = explain(erased, universe)

    wrong = any(
        readings[context] is not None and expected is not None
        and readings[context] != expected
        for context, expected in family.expect.items())
    return {
        "name": family.name, "note": family.note,
        "kind": found.kind, "state": found.state, "senses": found.senses,
        "expected_kind": family.expect_kind,
        "kind_correct": found.kind == family.expect_kind,
        "readings": readings, "expected": family.expect,
        "readings_correct": readings == family.expect,
        "wrong_reading": wrong,
        "truly_polysemous": family.truly_polysemous,
        "split": found.kind == POLYSEMY,
        "false_split": found.kind == POLYSEMY and not family.truly_polysemous,
        "missed_split": family.truly_polysemous and found.kind != POLYSEMY,
        "without_context_kind": without.kind,
        "without_context_resolves": without.kind in (MONOSEMY, NOISE, REVISION),
    }


def run_unidentifiable(universe) -> dict:
    """Two senses L8.31 says nothing here can separate: a split invents a distinction."""
    narrow = (Observation("TYPE", ValueType.NUMBER),)
    evidence = (
        ContextualObservation(_equals("success_rate"), "sim", 0),
        ContextualObservation(_equals("energy_error"), "lab", 1),
        ContextualObservation(_equals("success_rate"), "sim", 2),
        ContextualObservation(_equals("energy_error"), "lab", 3),
    )
    with_probes = explain(evidence, universe, narrow)
    with_full = explain(evidence, universe, available_probes(universe))
    return {
        "narrow_language": with_probes.kind,
        "full_language": with_full.kind,
        "refused_when_inseparable": with_probes.kind == UNIDENTIFIABLE,
        "allowed_when_separable": with_full.kind == POLYSEMY,
    }


def run_experiment(count: int = 24) -> dict:
    universe = universe_from(widen_store(build_holdout(count)[0]))
    runs = [run_family(family, universe) for family in build_families()]
    polysemous = [r for r in runs if r["truly_polysemous"]]
    monosemous = [r for r in runs if not r["truly_polysemous"]]
    timed = [r for r in runs if r["name"] in ("TRUE_POLYSEMY", "REVISION")]
    return {
        "runs": runs,
        "unidentifiable": run_unidentifiable(universe),
        "metrics": {
            "false_sense_split_rate": _ratio(
                sum(1 for r in monosemous if r["false_split"]), len(monosemous)),
            "missed_polysemy_rate": _ratio(
                sum(1 for r in polysemous if r["missed_split"]), len(polysemous)),
            "context_conditioned_resolution_rate": _ratio(
                sum(1 for r in polysemous if r["readings_correct"]), len(polysemous)),
            "revision_vs_polysemy_accuracy": _ratio(
                sum(1 for r in timed if r["kind_correct"]), len(timed)),
            "wrong_answer_rate": _ratio(
                sum(1 for r in runs if r["wrong_reading"]), len(runs)),
            "explanation_accuracy": _ratio(
                sum(1 for r in runs if r["kind_correct"]), len(runs)),
        },
    }


def format_experiment(report) -> str:
    found = report["metrics"]
    lines = ["## Context-conditioned polysemy (L8.35.1)", "",
             "| metric | value |", "| --- | --- |",
             f"| **false_sense_split_rate** | **{found['false_sense_split_rate']}** |",
             f"| missed_polysemy_rate | {found['missed_polysemy_rate']} |",
             f"| context_conditioned_resolution_rate "
             f"| {found['context_conditioned_resolution_rate']} |",
             f"| revision_vs_polysemy_accuracy | {found['revision_vs_polysemy_accuracy']} |",
             f"| **wrong_answer_rate** | **{found['wrong_answer_rate']}** |",
             f"| explanation_accuracy | {found['explanation_accuracy']} |",
             "", "### the families", "",
             "| family | explanation | state | senses |",
             "| --- | --- | --- | --- |"]
    for run in report["runs"]:
        senses = ", ".join(f"{k}={v}" for k, v in run["senses"].items()) or "-"
        mark = "" if run["kind_correct"] else " (NG)"
        lines.append(f"| {run['name']} | {run['kind']}{mark} | {run['state']} | {senses} |")

    lines += ["", "### ablation: the same evidence with context erased", "",
              "| family | with context | context erased | still resolves |",
              "| --- | --- | --- | --- |"]
    for run in report["runs"]:
        lines.append(f"| {run['name']} | {run['kind']} | {run['without_context_kind']} "
                     f"| {run['without_context_resolves']} |")
    polysemous = next(r for r in report["runs"] if r["truly_polysemous"])
    lines += ["",
              f"`TRUE_POLYSEMY` goes `{polysemous['kind']}` -> "
              f"`{polysemous['without_context_kind']}` once the labels are erased, and "
              f"stops resolving. Context was not a convenience -- it was the variable "
              f"the meaning depended on."]

    unidentifiable = report["unidentifiable"]
    lines += ["", "### a split refused because there is nothing to split", "",
              f"- with only a type hole available: `{unidentifiable['narrow_language']}`",
              f"- with the full observation language: `{unidentifiable['full_language']}`"]
    return "\n".join(lines)


def main() -> None:  # pragma: no cover
    print(format_experiment(run_experiment()))


if __name__ == "__main__":  # pragma: no cover
    main()
