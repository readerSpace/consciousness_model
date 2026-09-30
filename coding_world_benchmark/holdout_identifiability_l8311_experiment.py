"""L8.31.1: check the analysis against the three layers it claims to explain.

An identifiability analysis that is only ever compared against itself proves
nothing.  So this suite does not score L8.31 on its own terms: it runs L8.31
**first**, records what it says will happen, then runs L8.29 and L8.30 and
compares.  Three predictions, each falsifiable:

1. The set L8.31 says a truthful stream can never narrow past is exactly the set
   L8.29's exhaustive ``cross`` arm ends with.
2. A word L8.31 calls IDENTIFIABLE is one L8.29 actually decides.  The converse
   error -- calling a word identifiable when it is not -- is
   ``false_identifiable_rate`` and is the invariant here, because L8.29 would
   then acquire on partial evidence and hand L8.30 a case it cannot correct.
3. Whether L8.30 ever sees a contradiction is decided in advance by whether the
   stray observation excludes the truth.  L8.30's ``undetected_noise_rate`` is
   therefore predictable before L8.30 runs.

If those agree, the ``nested`` wall, the twin control and ``undetected_noise``
are not three separate demonstrations -- they are one structure seen three times.

The control that cannot be passed by accumulating data is the twins at
N = 10, 100, 1000, 10000.  Intersection is monotone, so once the surviving set
equals the class it cannot shrink again; the sweep shows that plateau rather than
asserting it, and L8.31 called it before any of the evidence arrived.  That is
what finally separates **not enough data yet** from **the language cannot say
it**.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

from .cross_context_identification_l829_experiment import (
    Observation, available_probes, identify, indistinguishable_universe,
    synthetic_universe, truthful, universe_from,
)
from .holdout_binding_l8211_experiment import build_holdout
from .holdout_lexicon_l8281_experiment import widen_store
from .holdout_revision_l8301_experiment import generate_scenarios, run_scenario
from .semantic_identifiability_l831_experiment import (
    IDENTIFIABLE, PARTIALLY_IDENTIFIABLE, STRUCTURALLY_UNIDENTIFIABLE, analyse,
    detectable_as_noise, equivalence_classes, predict_contradiction, reachable, saturate,
)
from .typed_operation_l819_experiment import ValueType

SIZES = (3, 5, 10)
GEOMETRIES = ("banded", "nested", "disjoint")


def _ratio(numerator, denominator):
    if denominator == 0:
        return "n/a"
    return round(numerator / denominator, 3)


def _universes():
    """Every world the earlier layers were measured in, plus the real one."""
    found = [("real", universe_from(widen_store(build_holdout(24)[0])))]
    for geometry in GEOMETRIES:
        for count in SIZES:
            found.append((f"{geometry}/K{count}",
                          synthetic_universe(count, geometry=geometry)))
    found.append(("twins/K3", indistinguishable_universe(3)))
    return found


# --------------------------------------------------------------------------
# prediction 1 and 2: against L8.29
# --------------------------------------------------------------------------

def check_against_identification() -> dict:
    rows, pairs = [], []
    for label, universe in _universes():
        probes = available_probes(universe)
        for facts in universe:
            predicted = reachable(facts.name, universe, probes)
            analysis = analyse(universe, probes, facts.name, probes)
            observed = identify("w", facts.name, universe, "cross",
                                budget=4 * len(universe) + 8)
            # What an exhaustive truthful run actually left standing.
            actual = (frozenset({observed.meaning}) if observed.meaning
                      else frozenset(
                          name for name in (f.name for f in universe)
                          if all(name in _survives(o, universe)
                                 for o in probes if truthful(o, facts.name, universe))))
            rows.append({
                "universe": label, "truth": facts.name,
                "verdict": analysis.verdict,
                "predicted": sorted(predicted), "actual": sorted(actual),
                "class_matches": predicted == actual,
                "said_identifiable": analysis.verdict == IDENTIFIABLE,
                "was_decided": observed.state == "DECIDABLE",
            })
        for left, right in combinations([f.name for f in universe], 2):
            together_predicted = right in reachable(left, universe, probes)
            observed_left = identify("w", left, universe, "cross",
                                     budget=4 * len(universe) + 8)
            together_actual = observed_left.state != "DECIDABLE"
            pairs.append({
                "universe": label, "pair": (left, right),
                "predicted_together": together_predicted,
                "actual_inseparable": together_actual and together_predicted,
            })
    return {"rows": rows, "pairs": pairs}


def _survives(observation: Observation, universe) -> frozenset[str]:
    from .cross_context_identification_l829_experiment import constrain

    return constrain(observation, universe)


def identification_metrics(report) -> dict:
    rows = report["rows"]
    claimed = [r for r in rows if r["said_identifiable"]]
    inseparable = [r for r in rows if len(r["predicted"]) > 1]
    return {
        # The invariant: never promise learnability the later layers cannot deliver.
        "false_identifiable_rate": _ratio(
            sum(1 for r in claimed if not r["was_decided"]), len(claimed)),
        "equivalence_class_accuracy": _ratio(
            sum(1 for r in rows if r["class_matches"]), len(rows)),
        "indistinguishable_pair_recall": _ratio(
            sum(1 for r in inseparable if not r["was_decided"]), len(inseparable)),
        "identifiable_rate": _ratio(len(claimed), len(rows)),
    }


# --------------------------------------------------------------------------
# prediction 3: against L8.30
# --------------------------------------------------------------------------

def check_against_revision() -> dict:
    rows = []
    for geometry in ("disjoint", "banded"):
        for scenario in generate_scenarios(geometry=geometry):
            count = int(scenario.name[1:])
            universe = synthetic_universe(count, geometry=geometry)
            # Two different predictions, deliberately kept apart.
            # ``detectable`` is pure identifiability: can the planted observation
            # ever conflict with the truth at all?  ``contradiction`` adds the
            # sequencing condition -- the belief must have committed first.
            detectable = scenario.stray is not None and detectable_as_noise(
                scenario.truth, scenario.stray, universe)
            predicted = predict_contradiction(scenario.stream, universe)
            observed = run_scenario(scenario, universe)
            rows.append({
                "geometry": geometry, "universe": scenario.name,
                "truth": scenario.truth,
                "predicted_detectable": detectable,
                "predicted_contradiction": predicted,
                "observed_contradiction": observed["had_contradiction"],
                "observed_ended_wrong": observed["ended_wrong"] or observed["undetected_noise"],
                "agrees": predicted == observed["had_contradiction"],
                "detectability_agrees": detectable or not observed["had_contradiction"],
            })
    return {
        "rows": rows,
        "metrics": {
            "predicted_vs_observed_detectability": _ratio(
                sum(1 for r in rows if r["agrees"]), len(rows)),
            "undetectable_implies_no_contradiction": _ratio(
                sum(1 for r in rows if r["detectability_agrees"]), len(rows)),
            "predicted_undetectable_rate": _ratio(
                sum(1 for r in rows if not r["predicted_detectable"]), len(rows)),
            "observed_undetectable_rate": _ratio(
                sum(1 for r in rows if not r["observed_contradiction"]), len(rows)),
        },
    }


# --------------------------------------------------------------------------
# minimal distinguishing sets, checked by using them
# --------------------------------------------------------------------------

def check_distinguishing_sets() -> dict:
    """Propose an extension, then extend the language and see whether it worked."""
    rows = []
    for label, universe in _universes():
        full = available_probes(universe)
        # Strip the language back to the type hole alone, so the search has real
        # work to do: with CONTRAST still available every field is already
        # separable and there would be nothing to propose.
        restricted = tuple(o for o in full if o.kind == "TYPE")
        pool = tuple(o for o in full if o.kind != "TYPE")
        for facts in universe:
            analysis = analyse(universe, restricted, facts.name, pool)
            if analysis.verdict != PARTIALLY_IDENTIFIABLE:
                continue
            extended = restricted + analysis.distinguishing
            after = reachable(facts.name, universe, extended)
            rows.append({
                "universe": label, "truth": facts.name,
                "before": sorted(analysis.of_interest),
                "proposed": [f"{o.kind}({o.payload})" for o in analysis.distinguishing],
                "claimed_resolves": analysis.resolves,
                "after": sorted(after),
                "worked": (len(after) == 1) == analysis.resolves,
                "resolved": len(after) == 1,
            })
    return {
        "rows": rows,
        "metrics": {
            "minimal_distinguishing_set_accuracy": _ratio(
                sum(1 for r in rows if r["worked"]), len(rows)),
            "resolved_rate": _ratio(sum(1 for r in rows if r["resolved"]), len(rows)),
            "mean_set_size": _ratio(sum(len(r["proposed"]) for r in rows), len(rows)),
        },
    }


# --------------------------------------------------------------------------
# the control no amount of data can pass
# --------------------------------------------------------------------------

COUNTS = (10, 100, 1000, 10000)


def check_saturation() -> dict:
    universe = indistinguishable_universe(3)
    probes = available_probes(universe)
    rows = []
    for name in ("twin_a", "metric_00"):
        analysis = analyse(universe, probes, name, probes)
        rows.append({
            "name": name,
            "predicted_verdict": analysis.verdict,
            "predicted_class": sorted(analysis.of_interest or ()),
            "observed": {count: sorted(saturate(name, universe, count)) for count in COUNTS},
        })
    return {
        "rows": rows,
        "metrics": {
            "plateau_matches_prediction": all(
                all(set(sizes) == set(row["predicted_class"]) for sizes in row["observed"].values())
                for row in rows),
        },
    }


# --------------------------------------------------------------------------
# what polysemy would have to look like
# --------------------------------------------------------------------------

def check_polysemy_precheck() -> dict:
    """A DISPUTED surface is two words only if the two meanings are separable.

    L8.30 left multiplicity open: a surface whose evidence cannot be reconciled
    might carry two senses, or might carry one the language cannot pin down.
    L8.31 decides which question is being asked before anything tries to split a
    word -- STRUCTURALLY_UNIDENTIFIABLE means there is nothing to split.
    """
    real = universe_from(widen_store(build_holdout(24)[0]))
    twins = indistinguishable_universe(3)
    current = (Observation("TYPE", ValueType.NUMBER),
               Observation("CONTRAST", "runtime_seconds"))
    pool = tuple(o for o in available_probes(real) if o.kind == "RANGE")
    genuine = analyse(real, current, "success_rate", pool)
    hopeless = analyse(twins, available_probes(twins), "twin_a", available_probes(twins))
    return {
        "separable_case": {
            "class": sorted(genuine.of_interest or ()),
            "verdict": genuine.verdict,
            "reading": "two senses are distinguishable here, so splitting the word is meaningful",
        },
        "inseparable_case": {
            "class": sorted(hopeless.of_interest or ()),
            "verdict": hopeless.verdict,
            "reading": "nothing separates these, so a split would invent a distinction",
        },
        "precheck_agrees": genuine.verdict != STRUCTURALLY_UNIDENTIFIABLE
                           and hopeless.verdict == STRUCTURALLY_UNIDENTIFIABLE,
    }


def run_experiment() -> dict:
    against_identification = check_against_identification()
    return {
        "identification": against_identification,
        "identification_metrics": identification_metrics(against_identification),
        "revision": check_against_revision(),
        "distinguishing": check_distinguishing_sets(),
        "saturation": check_saturation(),
        "polysemy": check_polysemy_precheck(),
    }


def format_experiment(report) -> str:
    found = report["identification_metrics"]
    revision = report["revision"]["metrics"]
    distinguishing = report["distinguishing"]["metrics"]
    lines = [
        "## Semantic identifiability analysis (L8.31.1)", "",
        "### the five numbers", "",
        "| metric | value |", "| --- | --- |",
        f"| **false_identifiable_rate** | **{found['false_identifiable_rate']}** |",
        f"| indistinguishable_pair_recall | {found['indistinguishable_pair_recall']} |",
        f"| equivalence_class_accuracy | {found['equivalence_class_accuracy']} |",
        f"| minimal_distinguishing_set_accuracy "
        f"| {distinguishing['minimal_distinguishing_set_accuracy']} |",
        f"| predicted_vs_observed_detectability "
        f"| {revision['predicted_vs_observed_detectability']} |",
        "",
        "### predicting L8.30 before it runs", "",
        f"- `undetectable_implies_no_contradiction` = "
        f"{revision['undetectable_implies_no_contradiction']}",
        f"- predicted undetectable (identifiability alone): "
        f"{revision['predicted_undetectable_rate']}",
        f"- observed without contradiction (adds the sequencing condition): "
        f"{revision['observed_undetectable_rate']}",
        "",
        "### minimal distinguishing sets", "",
        f"- mean size {distinguishing['mean_set_size']}, "
        f"resolved to a singleton {distinguishing['resolved_rate']}",
        "",
        "| universe | word | class before | proposed | after |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in report["distinguishing"]["rows"][:8]:
        lines.append(f"| {row['universe']} | {row['truth']} | {row['before']} "
                     f"| {', '.join(row['proposed'])} | {row['after']} |")

    lines += ["", "### the control no amount of data passes", "",
              "| word | L8.31 verdict (before any evidence) | "
              + " | ".join(f"N={count}" for count in COUNTS) + " |",
              "| --- | --- | " + " | ".join("---" for _ in COUNTS) + " |"]
    for row in report["saturation"]["rows"]:
        observed = " | ".join(str(row["observed"][count]) for count in COUNTS)
        lines.append(f"| {row['name']} | {row['predicted_verdict']} | {observed} |")

    polysemy = report["polysemy"]
    lines += ["", "### is it two words, or one the language cannot pin down?", "",
              f"- separable: {polysemy['separable_case']['class']} -> "
              f"{polysemy['separable_case']['verdict']} -- "
              f"{polysemy['separable_case']['reading']}",
              f"- inseparable: {polysemy['inseparable_case']['class']} -> "
              f"{polysemy['inseparable_case']['verdict']} -- "
              f"{polysemy['inseparable_case']['reading']}"]
    return "\n".join(lines)


def main() -> None:  # pragma: no cover
    print(format_experiment(run_experiment()))


if __name__ == "__main__":  # pragma: no cover
    main()
