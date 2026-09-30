"""L8.29.1: does a meaning candidate set actually shrink with experience?

The sweep varies **K, the number of same-typed fields**, not the size of the
vocabulary, because L8.28.1 showed that K is what broke identification: at K=2
elimination and identification coincide, and from K=3 they come apart.  Three
arms answer three different questions.

``single``
    One request, as L8.28 had it.  Reproduces that layer's result exactly --
    decidable at K=2, zero from K=3 -- which is the check that this suite is
    measuring the same thing.

``cross``
    H_{t+1} = H_t ∩ C(e_t) over every context a truthful world offers, in a fixed
    order.  Identifies at every K, and the cost is the number to look at.

``eig``
    The next context chosen by expected information gain.

Two synthetic geometries are run because the scaling law turns out to belong to
the world rather than to the algorithm.  In ``banded`` a literal always localises
to a constant-size block, so EIG's cost stops growing with K.  In ``nested`` the
ranges are contained in one another, and a field whose range sits inside every
other field's range **cannot be separated by any literal at all** -- every
truthful value it could be compared against is also admitted by everything above
it.  Only CONTRAST separates those, one candidate at a time.  The twin control is
the limit of that: two fields sharing a type and a range and nameable by nothing,
where UNRESOLVED forever is the correct answer and any resolution is a guess.

The end-to-end section is what stops this being a simulation.  On the widened
real store, an invented word is narrowed across **two actual requests** parsed by
L8.27's graph and L8.28's sources, acquired only when the set is a singleton, and
then allowed -- for the first time in the project -- to fill an argument that the
graph refused for want of a name.  ``false_acquisition_rate`` and
``wrong_answer_rate`` are measured with that opening in place, not with it shut.
"""
from __future__ import annotations

from dataclasses import dataclass
from statistics import median

from .cross_context_identification_l829_experiment import (
    CONTRADICTION, DECIDABLE, GEOMETRIES, UNRESOLVED, AcquiredLexicon, Belief,
    Observation, common_word, identify, indistinguishable_universe,
    observations_from_request, run_acquired_query, synthetic_universe, universe_from,
)
from .evidence_graph_l827_experiment import run_graph_query
from .holdout_binding_l8211_experiment import build_holdout
from .holdout_lexicon_l8281_experiment import widen_store
from .query_algebra_l820_experiment import canonical_render
from .semantic_proposer_l822_experiment import Proposer
from .span_grounded_proposer_l823_experiment import SpanProposer
from .structural_sensor_l826_experiment import StructuralSensor

ARMS = ("single", "cross", "eig")
SIZES = (2, 3, 5, 10, 20, 50)


def _ratio(numerator, denominator):
    if denominator == 0:
        return "n/a"
    return round(numerator / denominator, 3)


def run_sweep(sizes=SIZES, geometry: str = "banded") -> dict:
    """Every field in the universe takes a turn as the unknown word's meaning."""
    results: dict[int, dict[str, dict]] = {}
    for count in sizes:
        universe = synthetic_universe(count, geometry=geometry)
        budget = 4 * count + 8
        for_arm: dict[str, dict] = {}
        for arm in ARMS:
            runs = [identify("w", facts.name, universe, arm, budget) for facts in universe]
            decided = [run for run in runs if run.acquired]
            kinds: dict[str, int] = {}
            for run in runs:
                for entry in run.history:
                    kinds[entry.split(":")[0]] = kinds.get(entry.split(":")[0], 0) + 1
            for_arm[arm] = {
                "decidable_rate": _ratio(len(decided), len(runs)),
                "false_acquisition_rate": _ratio(sum(1 for r in runs if r.wrong), len(runs)),
                "unresolved_rate": _ratio(
                    sum(1 for r in runs if r.state == UNRESOLVED), len(runs)),
                "mean_observations": _ratio(sum(r.observations for r in decided),
                                            len(decided)),
                "median_observations": median([r.observations for r in decided])
                                       if decided else "n/a",
                "observation_mix": kinds,
            }
        results[count] = for_arm
    return results


def run_contradiction(trials: int = 20) -> dict:
    """Contexts that cannot all be about one field must be detected, not averaged."""
    universe = synthetic_universe(10, geometry="banded")
    names = [facts.name for facts in universe]
    detected = 0
    for index in range(trials):
        left = universe[index % len(universe)]
        right = universe[(index + 5) % len(universe)]
        belief = Belief.prior("w", universe)
        # Two literals no single field admits together.
        for facts in (left, right):
            belief = belief.observe(
                Observation("RANGE", round((facts.low + facts.high) / 2, 6)), universe)
        if belief.state == CONTRADICTION:
            detected += 1
    # A consistent pair must NOT be reported as a contradiction.
    consistent = Belief.prior("w", universe).observe(
        Observation("RANGE", round((universe[0].low + universe[0].high) / 2, 6)), universe)
    return {
        "contradiction_detection_rate": _ratio(detected, trials),
        "false_contradiction": consistent.state == CONTRADICTION,
        "names": names[:3],
    }


def run_control() -> dict:
    """Two fields nothing can separate. UNRESOLVED forever is the right answer."""
    universe = indistinguishable_universe(3)
    rows = {}
    for name in ("twin_a", "twin_b", "metric_00"):
        rows[name] = {
            arm: identify("w", name, universe, arm).state for arm in ARMS
        }
    return rows


# --------------------------------------------------------------------------
# end to end, on real requests
# --------------------------------------------------------------------------

#: Two requests about one invented word. Neither identifies it alone: the first
#: removes runtime_seconds and leaves {success_rate, energy_error}, the second
#: compares the word against a magnitude only one of those admits.
CONTEXTS = (
    "64x64のケースで処理時間ではなくフガ率をならして",
    "64x64のケースの0.5以上のフガ率をならして",
)
#: What the invented word means, known to the probe generator and to nothing else.
TRUTH = "success_rate"
#: A request the graph refuses for want of a name, used to see whether acquisition
#: actually recovers the ability to answer.
TARGET = "64x64のケースで処理時間ではなくフガ率をならして"


def run_end_to_end(count: int = 24) -> dict:
    store = widen_store(build_holdout(count)[0])
    sensors = (SpanProposer.trained(), Proposer.trained(), StructuralSensor())
    universe = universe_from(store)

    spans, gathered = [], []
    for request in CONTEXTS:
        span, observations = observations_from_request(request, store, *sensors)
        if not span:
            continue
        spans.append(span)
        gathered.append(observations)

    word = common_word(tuple(spans))
    steps = []
    belief = Belief.prior(word, universe)
    steps.append({"after": "prior", "candidates": sorted(belief.candidates),
                  "state": belief.state})
    for request, observations in zip(CONTEXTS, gathered):
        for observation in observations:
            belief = belief.observe(observation, universe)
        steps.append({"after": request, "candidates": sorted(belief.candidates),
                      "state": belief.state,
                      "used": [f"{o.kind}={o.payload}" for o in observations]})

    from .cross_context_identification_l829_experiment import Identification

    identification = Identification(word, TRUTH, belief.state, belief.meaning,
                                    sum(len(o) for o in gathered), ())
    lexicon = AcquiredLexicon()
    acquired = lexicon.acquire(identification)

    before = run_graph_query(store, TARGET, *sensors)
    after = run_acquired_query(store, TARGET, lexicon, *sensors)
    expected = canonical_render(
        __import__("coding_world_benchmark.query_algebra_l820_experiment",
                   fromlist=["Mean"]).Mean(
            __import__("coding_world_benchmark.query_algebra_l820_experiment",
                       fromlist=["Project"]).Project(
                __import__("coding_world_benchmark.query_algebra_l820_experiment",
                           fromlist=["Filter"]).Filter(
                    __import__("coding_world_benchmark.query_algebra_l820_experiment",
                               fromlist=["Source"]).Source(),
                    "lattice_size", "64x64"), TRUTH)))
    got = canonical_render(after.expr) if after.expr is not None else None

    # Acquisition must be reversible: retract and the system is exactly L8.27.
    lexicon.retract(word)
    reverted = run_acquired_query(store, TARGET, lexicon, *sensors)

    return {
        "word": word, "spans": spans, "steps": steps,
        "single_context_state": Belief.prior(word, universe).observe(
            gathered[0][0], universe).observe(gathered[0][1], universe).state
            if gathered and len(gathered[0]) >= 2 else UNRESOLVED,
        "acquired": acquired, "meaning": identification.meaning,
        "correct": identification.meaning == TRUTH,
        "false_acquisition": acquired and identification.meaning != TRUTH,
        "before": before.decision, "after": after.decision,
        "wrong_answer": after.decision == "ANSWER" and got != expected,
        "recovered": after.decision == "ANSWER" and got == expected,
        "reverted_to": reverted.decision,
        "reversible": reverted.decision == before.decision,
    }


def run_experiment(count: int = 24) -> dict:
    return {
        "sweeps": {geometry: run_sweep(geometry=geometry) for geometry in GEOMETRIES},
        "contradiction": run_contradiction(),
        "control": run_control(),
        "end_to_end": run_end_to_end(count),
    }


def format_experiment(report) -> str:
    lines = ["## Cross-context semantic identification (L8.29.1)", ""]
    for geometry, sweep in report["sweeps"].items():
        lines += [f"### K-sweep, `{geometry}` geometry", "",
                  "| K | single dec. | cross dec. | eig dec. | cross obs | eig obs "
                  "| **false_acq** |", "| --- | --- | --- | --- | --- | --- | --- |"]
        for count, arms in sweep.items():
            false_acq = max(str(arms[arm]["false_acquisition_rate"]) for arm in ARMS)
            lines.append(
                f"| {count} | {arms['single']['decidable_rate']} "
                f"| {arms['cross']['decidable_rate']} | {arms['eig']['decidable_rate']} "
                f"| {arms['cross']['mean_observations']} | {arms['eig']['mean_observations']} "
                f"| **{false_acq}** |")
        lines.append("")

    lines += ["### what the observations were made of (K=10, eig)", ""]
    for geometry, sweep in report["sweeps"].items():
        lines.append(f"- `{geometry}`: {sweep[10]['eig']['observation_mix']}")

    contradiction = report["contradiction"]
    lines += ["", "### contradiction", "",
              f"- `contradiction_detection_rate` = {contradiction['contradiction_detection_rate']}",
              f"- false contradiction on a consistent pair: {contradiction['false_contradiction']}",
              "", "### indistinguishable pair (must stay UNRESOLVED)", "",
              "| field | single | cross | eig |", "| --- | --- | --- | --- |"]
    for name, states in report["control"].items():
        lines.append(f"| {name} | " + " | ".join(states[arm] for arm in ARMS) + " |")

    end = report["end_to_end"]
    lines += ["", "### end to end, on real requests", "",
              f"- recurring word found across spans {end['spans']}: **「{end['word']}」**", ""]
    for step in end["steps"]:
        used = " + ".join(step.get("used", [])) or "-"
        lines.append(f"    - after `{step['after']}` [{used}] -> "
                     f"{step['candidates']} ({step['state']})")
    lines += ["",
              f"- one context alone: **{end['single_context_state']}**",
              f"- acquired: {end['acquired']} -> `{end['meaning']}` "
              f"(correct: {end['correct']})",
              f"- the refused request: {end['before']} -> **{end['after']}** "
              f"(recovered: {end['recovered']})",
              f"- **false_acquisition**: {end['false_acquisition']} ; "
              f"**wrong_answer**: {end['wrong_answer']}",
              f"- retracting returns it to {end['reverted_to']} "
              f"(reversible: {end['reversible']})"]
    return "\n".join(lines)


def main() -> None:  # pragma: no cover
    print(format_experiment(run_experiment()))


if __name__ == "__main__":  # pragma: no cover
    main()
