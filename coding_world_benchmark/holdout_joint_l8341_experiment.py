"""L8.34.1: the families a solver that just ran L8.29 twice would score zero on.

``JOINT_ONLY`` is the reason the layer exists.  Both surfaces carry the same
evidence and are each ambiguous between the same two fields, so word-at-a-time
inference finishes with two candidates apiece and nothing decided.  The order
relation between them admits only one of the two pairings -- energy_error can
never exceed success_rate in the recorded data -- so the assignment is unique
while both marginals, computed independently, are not.
``independent_vs_joint_gain`` counts exactly these.

``MARGINAL_TRAP`` is the safety cell, and it is the same rule this project has
restated at every level.  With::

    { (runtime, success), (success, runtime) }

each surface is "narrowed to two" and the correspondence is precisely what is
unknown.  Reading marginals would report progress and run the wrong program.
``false_joint_resolution_rate`` is the number that must stay at zero, and
``MARGINAL_ONLY`` is a state rather than a rounding error so the distinction can
be reported.

``INDEPENDENT_ENOUGH`` is the control in the other direction: where word-at-a-time
already succeeds, the joint solver must agree rather than quietly differ.
``IMPOSSIBLE`` checks that relations which cannot all hold produce a contradiction
instead of an arbitrary pick.

The question-planning halves are measured against each other on the same beliefs.
A relational question -- 「別の指標ですか」 -- is a candidate beside the per-word
ones, and whether it wins is left to the planner rather than assumed.
"""
from __future__ import annotations

from dataclasses import dataclass

from .cross_context_identification_l829_experiment import (
    Observation, available_probes, run_acquired_query, universe_from,
)
from .evidence_graph_l827_experiment import run_graph_query
from .holdout_binding_l8211_experiment import build_holdout
from .holdout_lexicon_l8281_experiment import widen_store
from .joint_inference_l834_experiment import (
    CONTRADICTION, DISTINCT, EXCEEDS, JOINT_RESOLVED, MARGINAL_ONLY, SAME_TYPE,
    JointBelief, Relation, independent, joint_queries, plan_joint,
    relations_from_request, resolve,
)
from .lexical_revision_l830_experiment import RevisableLexicon
from .query_algebra_l820_experiment import (
    Filter, Mean, Project, Source, canonical_render,
)
from .semantic_proposer_l822_experiment import Proposer
from .span_grounded_proposer_l823_experiment import SpanProposer
from .structural_sensor_l826_experiment import StructuralSensor
from .typed_operation_l819_experiment import ValueType

SURFACES = ("フガ率", "ホゲ尺")


@dataclass(frozen=True)
class Scenario:
    name: str
    request: str
    #: Observations already established per surface index, as the earlier layers
    #: would have produced them from the sentence.
    evidence: tuple[tuple[int, Observation], ...]
    expect_state: str
    expect: dict[str, str] | None
    note: str


def _number() -> Observation:
    return Observation("TYPE", ValueType.NUMBER)


def _not(name: str) -> Observation:
    return Observation("CONTRAST", name)


def build_scenarios() -> tuple[Scenario, ...]:
    ordered = "フガ率がホゲ尺より大きいケースの平均を出して"
    listed = "フガ率とホゲ尺をそれぞれならして"
    return (
        Scenario(
            "JOINT_ONLY", ordered,
            ((0, _number()), (0, _not("runtime_seconds")),
             (1, _number()), (1, _not("runtime_seconds"))),
            JOINT_RESOLVED, {"フガ率": "success_rate", "ホゲ尺": "energy_error"},
            "each word ambiguous between the same two; the order admits one pairing"),
        Scenario(
            "JOINT_ONLY_SWAPPED", "ホゲ尺がフガ率より大きいケースの平均を出して",
            ((0, _number()), (0, _not("runtime_seconds")),
             (1, _number()), (1, _not("runtime_seconds"))),
            JOINT_RESOLVED, {"フガ率": "energy_error", "ホゲ尺": "success_rate"},
            "the same shape with the order the other way round"),
        Scenario(
            "MARGINAL_TRAP", listed,
            ((0, _number()), (0, _not("energy_error")),
             (1, _number()), (1, _not("energy_error"))),
            MARGINAL_ONLY, None,
            "both marginals narrowed to two; the correspondence is what is unknown"),
        Scenario(
            "INDEPENDENT_ENOUGH", listed,
            ((0, _number()), (0, _not("runtime_seconds")), (0, _not("energy_error")),
             (1, _number()), (1, _not("success_rate")), (1, _not("energy_error"))),
            JOINT_RESOLVED, {"フガ率": "success_rate", "ホゲ尺": "runtime_seconds"},
            "control: word-at-a-time already succeeds, so the joint must agree"),
        Scenario(
            "IMPOSSIBLE", ordered,
            ((0, Observation("EQUALS", "energy_error")),
             (1, Observation("EQUALS", "runtime_seconds"))),
            CONTRADICTION, None,
            "energy_error cannot exceed runtime_seconds; no assignment survives"),
    )


def _ratio(numerator, denominator):
    if denominator == 0:
        return "n/a"
    return round(numerator / denominator, 3)


def run_scenario(scenario: Scenario, universe) -> dict:
    belief = JointBelief.prior(SURFACES, universe)
    grouped: dict[int, list[Observation]] = {}
    for index, observation in scenario.evidence:
        belief = belief.observe(index, observation)
        grouped.setdefault(index, []).append(observation)

    relations = relations_from_request(scenario.request, SURFACES)
    joint = belief
    for relation in relations:
        joint = joint.relate(relation)

    alone = independent(belief, {k: tuple(v) for k, v in grouped.items()})
    solved_independently = all(len(m) == 1 for m in alone)
    meanings = joint.meanings

    return {
        "name": scenario.name, "note": scenario.note,
        "relations": [f"{r.kind}({r.left},{r.right})" for r in relations],
        "independent": [sorted(m) for m in alone],
        "joint": sorted(joint.assignments),
        "marginals": [sorted(m) for m in joint.marginals],
        "state": joint.state,
        "meanings": meanings,
        "expected_state": scenario.expect_state,
        "expected": scenario.expect,
        "state_correct": joint.state == scenario.expect_state,
        "meanings_correct": meanings == scenario.expect,
        "solved_independently": solved_independently,
        "solved_jointly": joint.state == JOINT_RESOLVED,
        "joint_only": joint.state == JOINT_RESOLVED and not solved_independently,
        "false_joint": (joint.state == JOINT_RESOLVED and scenario.expect is not None
                        and meanings != scenario.expect)
                       or (joint.state == JOINT_RESOLVED and scenario.expect is None),
    }


def compare_planning(universe) -> dict:
    """Joint planning against per-word planning, on the same starting belief."""
    belief = JointBelief.prior(SURFACES, universe)
    for index in (0, 1):
        belief = belief.observe(index, _number())
    for relation in (Relation(SAME_TYPE, 0, 1), Relation(DISTINCT, 0, 1)):
        belief = belief.relate(relation)

    pool = tuple(p for p in available_probes(universe) if p.kind != "TYPE")
    truth = {"フガ率": "success_rate", "ホゲ尺": "energy_error"}

    settled, asked_joint = resolve(belief, truth, pool)
    first = plan_joint(belief, pool)

    # The per-word baseline: only questions about a single surface are allowed.
    surface_only = tuple(q for q in joint_queries(belief, pool) if q.kind == "SURFACE")
    working, asked_alone = belief, 0
    while working.state not in (JOINT_RESOLVED, CONTRADICTION) and asked_alone < 16:
        usable = [q for q in surface_only if all(q.split(working))]
        if not usable:
            break
        query = min(
            ((sum(len(b) ** 2 for b in q.split(working)) / len(working.assignments),
              index, q) for index, q in enumerate(usable)),
            key=lambda item: (item[0], item[1]))[2]
        assignment = tuple(truth[s] for s in working.surfaces)
        from .cross_context_identification_l829_experiment import constrain

        answer = assignment[query.index] in constrain(query.observation, universe)
        from .joint_inference_l834_experiment import apply_answer

        working = apply_answer(working, query, answer)
        asked_alone += 1

    return {
        "first_question": first.text if first else None,
        "first_is_relational": bool(first and first.kind == "RELATION"),
        "questions_joint": asked_joint,
        "questions_surface_only": asked_alone,
        "joint_meanings": settled.meanings,
        "surface_only_meanings": working.meanings,
        "both_correct": settled.meanings == truth and working.meanings == truth,
    }


def run_end_to_end(universe, store) -> dict:
    """Two words acquired at once, then used on an ordinary single-word request."""
    sensors = (SpanProposer.trained(), Proposer.trained(), StructuralSensor())
    scenario = build_scenarios()[0]
    belief = JointBelief.prior(SURFACES, universe)
    for index, observation in scenario.evidence:
        belief = belief.observe(index, observation)
    for relation in relations_from_request(scenario.request, SURFACES):
        belief = belief.relate(relation)
    meanings = belief.meanings

    target = "64x64のケースで処理時間ではなくフガ率をならして"
    before = run_graph_query(store, target, *sensors).decision

    lexicon = RevisableLexicon(universe)
    for surface, meaning in (meanings or {}).items():
        lexicon.observe(surface, Observation("EQUALS", meaning))
    after = run_acquired_query(store, target, lexicon, *sensors)
    expected = canonical_render(
        Mean(Project(Filter(Source(), "lattice_size", "64x64"), "success_rate")))
    got = canonical_render(after.expr) if after.expr is not None else None
    return {
        "meanings": meanings, "before": before, "after": after.decision,
        "correct": after.decision == "ANSWER" and got == expected,
    }


def run_experiment(count: int = 24) -> dict:
    store = widen_store(build_holdout(count)[0])
    universe = universe_from(store)
    runs = [run_scenario(scenario, universe) for scenario in build_scenarios()]
    return {
        "runs": runs,
        "planning": compare_planning(universe),
        "end_to_end": run_end_to_end(universe, store),
        "metrics": {
            "joint_resolution_accuracy": _ratio(
                sum(1 for r in runs if r["state_correct"] and r["meanings_correct"]),
                len(runs)),
            "false_joint_resolution_rate": _ratio(
                sum(1 for r in runs if r["false_joint"]), len(runs)),
            "independent_vs_joint_gain": sum(1 for r in runs if r["joint_only"]),
            "marginal_trap_abstention": _ratio(
                sum(1 for r in runs if r["name"] == "MARGINAL_TRAP"
                    and r["state"] == MARGINAL_ONLY),
                sum(1 for r in runs if r["name"] == "MARGINAL_TRAP")),
            "state_accuracy": _ratio(sum(1 for r in runs if r["state_correct"]), len(runs)),
        },
    }


def format_experiment(report) -> str:
    found = report["metrics"]
    planning = report["planning"]
    end = report["end_to_end"]
    lines = [
        "## Multi-unknown joint inference (L8.34.1)", "",
        "| metric | value |", "| --- | --- |",
        f"| joint_resolution_accuracy | {found['joint_resolution_accuracy']} |",
        f"| **false_joint_resolution_rate** | **{found['false_joint_resolution_rate']}** |",
        f"| **independent_vs_joint_gain** | **{found['independent_vs_joint_gain']}** |",
        f"| marginal_trap_abstention | {found['marginal_trap_abstention']} |",
        f"| state_accuracy | {found['state_accuracy']} |",
        "", "### the families", "",
        "| family | independent | joint | state | meanings |",
        "| --- | --- | --- | --- | --- |",
    ]
    for run in report["runs"]:
        meanings = ", ".join(f"{k}={v}" for k, v in (run["meanings"] or {}).items()) or "-"
        lines.append(f"| {run['name']} | {run['independent']} | {len(run['joint'])} pair(s) "
                     f"| {run['state']} | {meanings} |")

    lines += ["", "### planning over the pair versus over each word", "",
              f"- first question chosen: **{planning['first_question']}** "
              f"(relational: {planning['first_is_relational']})",
              f"- questions, planning jointly: **{planning['questions_joint']}**",
              f"- questions, per-word only: **{planning['questions_surface_only']}**",
              f"- both arrive at the same meanings: {planning['both_correct']}",
              "", "### two words acquired at once, then used", "",
              f"- resolved: {end['meanings']}",
              f"- an ordinary single-word request: {end['before']} -> **{end['after']}** "
              f"(correct: {end['correct']})"]
    return "\n".join(lines)


def main() -> None:  # pragma: no cover
    print(format_experiment(run_experiment()))


if __name__ == "__main__":  # pragma: no cover
    main()
