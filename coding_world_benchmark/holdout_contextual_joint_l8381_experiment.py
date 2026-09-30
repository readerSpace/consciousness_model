"""L8.38.1: the family where looking at one word at a time can see nothing at all.

``RELATIONAL_POLYSEMY`` is the reason the layer exists and it is built to be
unexplainable by the two layers it sits on. Every single-word candidate set is
*identical* across the contexts, so L8.35 applied per word finds no context
effect to report; and the belief sets differ, so L8.37 applied to the pooled
evidence loses the distinction. The difference lives entirely in the
correspondence.

The demonstration is the part that matters, because "the sets differ" on its own
is bookkeeping. **One further observation -- the same one in both contexts --
lands on different meanings.** Telling the system that the first surface is
runtime_seconds makes the second mean success_rate in one episode and
energy_error in the other, and that implication was invisible to every per-word
view of the same evidence.

``CONTEXT_ERASED`` runs the same evidence with the labels removed. It must break
this family and leave the others alone, which is the L8.35 ablation lifted from
one word to the tuple: context was not a convenience, it was what selected the
admissible combinations.

The controls are the ones that make the headline mean something.
``INDEPENDENT_POLYSEMY`` has marginals that *do* differ, so a per-word account
explains it and the layer must not claim credit. ``CONTEXT_IRRELEVANT`` has
identical sets and must be reported as no effect rather than as a context
dependence. ``SYMMETRIC_WITHIN_CONTEXT`` stays ambiguous inside each context
however much is added, and ``CONTRADICTION`` reports rather than picks.
"""
from __future__ import annotations

from dataclasses import dataclass

from .contextual_joint_l838_experiment import (
    NO_EFFECT, RELATIONAL_EFFECT, UNARY_EFFECT, ContextualJoint, erased_verdict,
    per_word_sees_context, per_word_view,
)
from .cross_context_identification_l829_experiment import Observation, universe_from
from .holdout_binding_l8211_experiment import build_holdout
from .holdout_lexicon_l8281_experiment import widen_store
from .nway_semantics_l837_experiment import AMBIGUOUS, CONTRADICTION, RESOLVED, verdict

SURFACES = ("フガ率", "ホゲ尺")
CONTEXTS = ("sim", "lab")

RUNTIME, RATE, ENERGY = "runtime_seconds", "success_rate", "energy_error"

#: The observation both contexts receive, after which they disagree about what
#: the *other* surface means.
PROBE_INDEX = 0
PROBE = Observation("EQUALS", RUNTIME)


def _equals(name: str) -> Observation:
    return Observation("EQUALS", name)


@dataclass(frozen=True)
class Family:
    name: str
    sets: tuple[tuple[str, frozenset[tuple[str, ...]]], ...]
    expect_effect: str
    #: What the second surface should mean in each context once PROBE arrives.
    expect_after: dict[str, str | None]
    note: str


def build_families() -> tuple[Family, ...]:
    return (
        Family(
            "RELATIONAL_POLYSEMY",
            (("sim", frozenset({(RUNTIME, RATE), (RATE, ENERGY)})),
             ("lab", frozenset({(RUNTIME, ENERGY), (RATE, RATE)}))),
            RELATIONAL_EFFECT, {"sim": RATE, "lab": ENERGY},
            "identical single-word candidate sets; only the correspondence differs"),
        Family(
            "INDEPENDENT_POLYSEMY",
            (("sim", frozenset({(RUNTIME, RATE)})),
             ("lab", frozenset({(ENERGY, ENERGY)}))),
            UNARY_EFFECT, {"sim": RATE, "lab": None},
            "control: the marginals differ, so a per-word account explains it"),
        Family(
            "CONTEXT_IRRELEVANT",
            (("sim", frozenset({(RUNTIME, RATE), (RATE, ENERGY)})),
             ("lab", frozenset({(RUNTIME, RATE), (RATE, ENERGY)}))),
            NO_EFFECT, {"sim": RATE, "lab": RATE},
            "control: identical sets, so the context is doing nothing"),
        Family(
            "SYMMETRIC_WITHIN_CONTEXT",
            (("sim", frozenset({(RUNTIME, RATE), (RUNTIME, ENERGY)})),
             ("lab", frozenset({(RUNTIME, RATE), (RUNTIME, ENERGY)}))),
            NO_EFFECT, {"sim": None, "lab": None},
            "control: the probe does not break a symmetry inside a context"),
        Family(
            "CONTRADICTION",
            (("sim", frozenset({(RUNTIME, RATE)})),
             ("lab", frozenset())),
            UNARY_EFFECT, {"sim": RATE, "lab": None},
            "control: an empty context is reported, not filled in"),
    )


def _ratio(numerator, denominator):
    if denominator == 0:
        return "n/a"
    return round(numerator / denominator, 3)


def run_family(family: Family, universe) -> dict:
    joint = ContextualJoint(SURFACES, family.sets)
    after = joint.observe(PROBE_INDEX, PROBE, universe)

    readings = {}
    for context in CONTEXTS:
        meanings = after.meanings_in(context)
        readings[context] = meanings[SURFACES[1]] if meanings else None

    # The same evidence with the labels removed.
    erased_before = erased_verdict(joint)
    erased_after = verdict(sorted(after.erased()))

    wrong = any(
        readings[context] is not None and expected is not None
        and readings[context] != expected
        for context, expected in family.expect_after.items())
    resolved_with_context = sum(1 for value in readings.values() if value is not None)
    resolved_erased = 1 if erased_after == RESOLVED else 0

    return {
        "name": family.name, "note": family.note,
        "effect": joint.effect, "expected_effect": family.expect_effect,
        "effect_correct": joint.effect == family.expect_effect,
        "marginals_agree": joint.marginals_agree,
        "per_word_sees_context": per_word_sees_context(joint),
        "per_word": {s: {c: sorted(v) for c, v in m.items()}
                     for s, m in per_word_view(joint).items()},
        "verdicts": {c: joint.verdict_in(c) for c in CONTEXTS},
        "after_probe": {c: after.verdict_in(c) for c in CONTEXTS},
        "readings": readings, "expected": family.expect_after,
        "readings_correct": all(
            readings[c] == family.expect_after[c] for c in CONTEXTS),
        "wrong": wrong,
        "erased_before": erased_before, "erased_after": erased_after,
        "erasure_breaks_it": resolved_with_context > 0 and resolved_erased == 0,
        "resolved_with_context": resolved_with_context,
        "different_readings": len({v for v in readings.values() if v}) > 1,
    }


def run_experiment(count: int = 24) -> dict:
    universe = universe_from(widen_store(build_holdout(count)[0]))
    runs = [run_family(family, universe) for family in build_families()]
    relational = [r for r in runs if r["effect"] == RELATIONAL_EFFECT]
    others = [r for r in runs if r["effect"] != RELATIONAL_EFFECT]
    return {
        "runs": runs,
        "metrics": {
            "false_joint_resolution_rate": _ratio(
                sum(1 for r in runs if r["wrong"]), len(runs)),
            "context_conditioned_joint_accuracy": _ratio(
                sum(1 for r in runs if r["readings_correct"]), len(runs)),
            "effect_accuracy": _ratio(
                sum(1 for r in runs if r["effect_correct"]), len(runs)),
            "relational_only_gain": sum(
                1 for r in relational if r["different_readings"]),
            "marginal_identity_verified": all(
                r["marginals_agree"] and not r["per_word_sees_context"]
                for r in relational),
            "erasure_breaks_only_relational": (
                all(r["erasure_breaks_it"] for r in relational)
                and not any(r["erasure_breaks_it"] for r in others)),
        },
    }


def format_experiment(report) -> str:
    found = report["metrics"]
    lines = ["## Context-conditioned joint semantics (L8.38.1)", "",
             "| metric | value |", "| --- | --- |",
             f"| **false_joint_resolution_rate** | **{found['false_joint_resolution_rate']}** |",
             f"| context_conditioned_joint_accuracy "
             f"| {found['context_conditioned_joint_accuracy']} |",
             f"| effect_accuracy | {found['effect_accuracy']} |",
             f"| **relational_only_gain** | **{found['relational_only_gain']}** |",
             f"| marginal_identity_verified | {found['marginal_identity_verified']} |",
             f"| erasure_breaks_only_relational "
             f"| {found['erasure_breaks_only_relational']} |",
             "", "### the families", "",
             "| family | effect | per-word sees context | erasure breaks it |",
             "| --- | --- | --- | --- |"]
    for run in report["runs"]:
        mark = "" if run["effect_correct"] else " (NG)"
        lines.append(f"| {run['name']} | {run['effect']}{mark} "
                     f"| {run['per_word_sees_context']} | {run['erasure_breaks_it']} |")

    relational = next(r for r in report["runs"]
                      if r["name"] == "RELATIONAL_POLYSEMY")
    lines += ["", "### the one observation that lands differently", "",
              "What a per-word view of the same evidence contains:", ""]
    for surface, sets in relational["per_word"].items():
        lines.append(f"- `{surface}`: " + ", ".join(
            f"{context} {values}" for context, values in sets.items()))
    lines += ["",
              f"Identical in every context, so there is no context effect for a "
              f"per-word account to find. Then `{SURFACES[0]} = {RUNTIME}` arrives "
              f"in both:", "",
              f"- sim -> `{SURFACES[1]}` means **{relational['readings']['sim']}**",
              f"- lab -> `{SURFACES[1]}` means **{relational['readings']['lab']}**",
              "",
              f"With the labels erased the same observation leaves "
              f"`{relational['erased_after']}` -- the correspondence was the thing "
              f"the context was carrying."]
    return "\n".join(lines)


def main() -> None:  # pragma: no cover
    print(format_experiment(run_experiment()))


if __name__ == "__main__":  # pragma: no cover
    main()
