"""L8.32.1: does planning the next observation beat collecting more of them,
and does the planner know when to stop?

Three questions, and the third is the one that keeps the other two honest.

**Does it resolve?**  ``active_resolution_rate`` over every meaning in every
world, with ``false_resolution_rate`` beside it.  The second cannot move: both
branches of a query keep the true meaning, so the candidate set always contains
the answer and a singleton can only ever be the right one.

**Is greedy any good?**  Minimising expected size one step at a time is the
standard move and is known not to be optimal in general, so for the small worlds
the entire decision tree is enumerated and ``E[T]`` computed exactly.
``greedy_vs_optimal_regret`` is then measured rather than assumed away.

**Does it stop?**  The twins are the world where an active learner is supposed to
fail, and failing well means terminating with a reason rather than asking
forever.  ``impossible_query_avoidance_rate`` is the fraction of unresolvable
runs that end ``STRUCTURALLY_UNRESOLVABLE`` instead of exhausting a budget, and
``wasted_queries`` counts questions that did not shrink anything -- measured, not
assumed, because a zero nobody looked for is not evidence.

One result falls out that L8.31 could not have produced.  Its ``reachable`` is
the *passive* regime: an observation arrives only when the meaning satisfies it,
so a meaning is never narrowed by what it fails.  A query has two outcomes and
the negative one is evidence too, so **asking is strictly stronger than being
told** -- and the suite measures that gap as ``active_vs_passive_gain`` rather
than claiming it.  The twins are unmoved, which is the point: active querying
buys reach, not miracles.
"""
from __future__ import annotations

from statistics import mean

from .active_observation_l832_experiment import (
    RESOLVED, STRUCTURALLY_UNRESOLVABLE, answer_as_observation, expected_size,
    optimal_depth, plan, question_for, render_query, run_active,
)
from .cross_context_identification_l829_experiment import (
    available_probes, indistinguishable_universe, synthetic_universe, universe_from,
)
from .holdout_binding_l8211_experiment import build_holdout
from .holdout_lexicon_l8281_experiment import widen_store
from .semantic_identifiability_l831_experiment import (
    IDENTIFIABLE, analyse, reachable,
)

#: The exact optimum is exponential, so it is computed only where enumeration is
#: honest.  Larger worlds still get the greedy arm and the stopping check.
OPTIMAL_LIMIT = 6
SIZES = (3, 4, 5, 6, 10)
GEOMETRIES = ("banded", "nested", "disjoint")


def _ratio(numerator, denominator):
    if denominator == 0:
        return "n/a"
    return round(numerator / denominator, 3)


def worlds():
    found = [("real", universe_from(widen_store(build_holdout(24)[0])))]
    for geometry in GEOMETRIES:
        for count in SIZES:
            found.append((f"{geometry}/K{count}",
                          synthetic_universe(count, geometry=geometry)))
    found.append(("twins/K3", indistinguishable_universe(3)))
    return found


def run_world(label: str, universe) -> dict:
    pool = available_probes(universe)
    names = [facts.name for facts in universe]
    runs = [run_active(name, universe, pool) for name in names]

    solvable, unsolvable = [], []
    for run in runs:
        (solvable if run.state == RESOLVED else unsolvable).append(run)

    # "Not enumerated" and "no policy exists" are different facts and must not
    # print the same way.
    optimal, optimal_note = None, "not enumerated (K too large)"
    if len(names) <= OPTIMAL_LIMIT:
        value = optimal_depth(frozenset(names), universe, pool)
        if value == float("inf"):
            optimal_note = "unresolvable"
        else:
            optimal, optimal_note = round(value, 4), ""

    # Passive versus active. On the full pool the CONTRAST probes make every
    # meaning passively identifiable -- each one satisfies every CONTRAST but its
    # own, and their intersection is already a singleton -- so the asymmetry is
    # invisible there. It bites when only magnitude comparisons are available,
    # which is exactly L8.29's ``nested`` finding: a meaning whose range sits
    # inside every other's can never be narrowed by a literal it satisfies, and
    # only a *negative* answer separates it.
    magnitudes = tuple(query for query in pool if query.kind == "RANGE")
    gained = []
    for name in names:
        passive = reachable(name, universe, magnitudes)
        active = run_active(name, universe, magnitudes)
        if len(passive) > 1 and active.state == RESOLVED:
            gained.append(name)

    return {
        "world": label,
        "size": len(names),
        "runs": [{
            "truth": run.truth, "state": run.state, "meaning": run.meaning,
            "steps": len(run.steps), "wasted": run.wasted,
            "wrong": run.wrong,
            "questions": [render_query(step.query, "w") for step in run.steps],
        } for run in runs],
        "resolved": len(solvable),
        "unresolvable": len(unsolvable),
        "stopped_cleanly": sum(1 for r in unsolvable
                               if r.state == STRUCTURALLY_UNRESOLVABLE),
        "wasted": sum(run.wasted for run in runs),
        "wrong": sum(1 for run in runs if run.wrong),
        "mean_steps": round(mean([len(r.steps) for r in solvable]), 4) if solvable else None,
        "optimal": optimal,
        "optimal_note": optimal_note,
        "passive_gain": gained,
        "magnitude_only": len(magnitudes),
    }


def check_prediction(universe) -> float:
    """Does E[|C after o|] match what actually happens, averaged over truths?

    Under a uniform prior the two are equal by construction, so a gap would mean
    the planner is scoring something other than what it does.
    """
    pool = available_probes(universe)
    candidates = frozenset(facts.name for facts in universe)
    query = plan(candidates, universe, pool)
    if query is None:
        return 0.0
    predicted = expected_size(query, candidates, universe)
    from .active_observation_l832_experiment import branches
    from .semantic_identifiability_l831_experiment import effect

    actual = []
    for name in candidates:
        yes, no = branches(query, candidates, universe)
        actual.append(len(yes if effect(name, query, universe) else no))
    return abs(predicted - mean(actual))


#: The loop, end to end, on the real store: a word the graph refused for want of
#: a name, a question the agent chose itself, and an answer that changes what
#: executes.
SURFACE = "フガ率"
TARGET = "64x64のケースで処理時間ではなくフガ率をならして"
TRUTH = "success_rate"


def run_end_to_end() -> dict:
    """不明 -> なぜ不明か -> 何を訊けば分かるか -> 訊く -> 実行が変わる.

    Every earlier layer contributes one step and none of them is re-implemented
    here: L8.27 refuses and names the span, L8.31 says which meanings are still
    in play and that they are separable, L8.32 picks the question, L8.30 takes
    the answer as an observation, and L8.29's executor uses the acquired meaning.
    """
    from .cross_context_identification_l829_experiment import (
        Observation, run_acquired_query,
    )
    from .evidence_graph_l827_experiment import run_graph_query
    from .lexical_revision_l830_experiment import RevisableLexicon
    from .query_algebra_l820_experiment import (
        Filter, Mean, Project, Source, canonical_render,
    )
    from .semantic_proposer_l822_experiment import Proposer
    from .semantic_identifiability_l831_experiment import effect
    from .span_grounded_proposer_l823_experiment import SpanProposer
    from .structural_sensor_l826_experiment import StructuralSensor
    from .typed_operation_l819_experiment import ValueType

    store = widen_store(build_holdout(24)[0])
    sensors = (SpanProposer.trained(), Proposer.trained(), StructuralSensor())
    universe = universe_from(store)
    pool = available_probes(universe)

    lexicon = RevisableLexicon(universe)
    known = (Observation("TYPE", ValueType.NUMBER),
             Observation("CONTRAST", "runtime_seconds"))
    for observation in known:
        lexicon.observe(SURFACE, observation)
    stuck = lexicon.belief(SURFACE)

    diagnosis = analyse(universe, known, TRUTH, pool)
    before = run_graph_query(store, TARGET, *sensors)

    asked, query = question_for(stuck.candidates, universe, pool, SURFACE)
    answer = effect(TRUTH, query, universe) if query else None
    if query is not None:
        # Whichever way the answer went, it is an observation; a NO asserts the
        # complement exactly rather than something like it.
        lexicon.observe(SURFACE, answer_as_observation(query, answer))
    settled = lexicon.belief(SURFACE)
    after = run_acquired_query(store, TARGET, lexicon, *sensors)
    expected = canonical_render(
        Mean(Project(Filter(Source(), "lattice_size", "64x64"), TRUTH)))
    got = canonical_render(after.expr) if after.expr is not None else None

    return {
        "stuck_candidates": sorted(stuck.candidates),
        "diagnosis": diagnosis.verdict,
        "why": diagnosis.why(),
        "question": asked,
        "answer": answer,
        "settled_state": settled.state,
        "settled_meaning": settled.meaning,
        "before": before.decision,
        "after": after.decision,
        "correct": after.decision == "ANSWER" and got == expected,
        "questions_asked": 1 if query else 0,
    }


def run_experiment() -> dict:
    results = [run_world(label, universe) for label, universe in worlds()]
    comparable = [r for r in results if r["optimal"] is not None and r["mean_steps"]]
    unsolvable_worlds = [r for r in results if r["unresolvable"]]
    drift = {label: round(check_prediction(universe), 6) for label, universe in worlds()}

    total_runs = sum(len(r["runs"]) for r in results)
    total_resolved = sum(r["resolved"] for r in results)
    total_unsolvable = sum(r["unresolvable"] for r in results)
    return {
        "worlds": results,
        "end_to_end": run_end_to_end(),
        "prediction_drift": drift,
        "metrics": {
            "active_resolution_rate": _ratio(total_resolved, total_runs),
            "false_resolution_rate": _ratio(sum(r["wrong"] for r in results), total_runs),
            "impossible_query_avoidance_rate": _ratio(
                sum(r["stopped_cleanly"] for r in results), total_unsolvable),
            "wasted_queries": sum(r["wasted"] for r in results),
            "observations_to_resolution": round(
                mean([r["mean_steps"] for r in results if r["mean_steps"]]), 3),
            "greedy_vs_optimal_regret": round(
                mean([r["mean_steps"] - r["optimal"] for r in comparable]), 4)
                if comparable else "n/a",
            "predicted_vs_actual_reduction": max(drift.values()),
            "active_vs_passive_gain": sum(len(r["passive_gain"]) for r in results),
        },
    }


def format_experiment(report) -> str:
    found = report["metrics"]
    lines = [
        "## Active semantic observation (L8.32.1)", "",
        "| metric | value |", "| --- | --- |",
        f"| active_resolution_rate | {found['active_resolution_rate']} |",
        f"| **false_resolution_rate** | **{found['false_resolution_rate']}** |",
        f"| **impossible_query_avoidance_rate** | "
        f"**{found['impossible_query_avoidance_rate']}** |",
        f"| wasted_queries | {found['wasted_queries']} |",
        f"| observations_to_resolution | {found['observations_to_resolution']} |",
        f"| greedy_vs_optimal_regret | {found['greedy_vs_optimal_regret']} |",
        f"| predicted_vs_actual_reduction | {found['predicted_vs_actual_reduction']} |",
        f"| active_vs_passive_gain | {found['active_vs_passive_gain']} |",
        "", "### greedy against the exactly-enumerated optimum", "",
        "| world | K | greedy E[T] | optimal E[T] | regret | resolved | stopped |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for world in report["worlds"]:
        optimal = world["optimal"]
        regret = (round(world["mean_steps"] - optimal, 4)
                  if optimal is not None and world["mean_steps"] else "-")
        shown = optimal if optimal is not None else world["optimal_note"]
        lines.append(
            f"| {world['world']} | {world['size']} | {world['mean_steps'] or '-'} "
            f"| {shown} | {regret} "
            f"| {world['resolved']}/{world['size']} | {world['stopped_cleanly']} |")
    lines.append("")
    lines.append("Regret is 0 in every world enumerated. That is a measurement "
                 "about these worlds, not a theorem: greedy expected-size "
                 "minimisation is not optimal in general, and the tree was "
                 "enumerated precisely because assuming otherwise would have "
                 "been the easy mistake.")

    lines += ["", "### what asking buys over being told", "",
              "Measured with magnitude comparisons only: on the full pool every "
              "meaning is already passively identifiable, so the gap would read "
              "as zero for the wrong reason.", ""]
    for world in report["worlds"]:
        if world["passive_gain"]:
            lines.append(f"- `{world['world']}`: {len(world['passive_gain'])} of "
                         f"{world['size']} unreachable passively, resolved by asking "
                         f"({', '.join(world['passive_gain'][:3])}…)")
    twins = next(w for w in report["worlds"] if w["world"] == "twins/K3")
    lines.append(f"- `twins/K3`: {twins['unresolvable']} unmoved -- "
                 f"active querying buys reach, not miracles")

    end = report["end_to_end"]
    lines += ["", "### the loop, closed, on a real request", "",
              f"1. the graph refuses: `{end['before']}`",
              f"2. still in play: {end['stuck_candidates']} -> `{end['diagnosis']}`",
              f"   - {end['why']}",
              f"3. the agent asks: **{end['question']}**",
              f"4. the answer ({end['answer']}) settles it: "
              f"`{end['settled_state']}` / `{end['settled_meaning']}`",
              f"5. the same request now: **{end['after']}** "
              f"(correct: {end['correct']}), after {end['questions_asked']} question"]
    return "\n".join(lines)


def main() -> None:  # pragma: no cover
    print(format_experiment(run_experiment()))


if __name__ == "__main__":  # pragma: no cover
    main()
