"""L8.42.1: the best question, the reachable question, and the one that is neither.

Four states of the same world, distinguished only by what the person can refer to
and recall:

``ORACLE_ACCESS``
    everything open.  One question -- 「lab の場面で フガ率 は runtime／success／
    energy のどれですか？」 -- splits three worlds three ways and finishes.

``INDIRECT_ACCESS``
    the ``lab`` episode's details are closed, so that question is not in the
    action space.  The episodes are still referable, so 「この対応は sim と lab で
    同じですか？」 is, and **two** of those finish it.  The planner is not told
    to prefer them; they are what is left.

``INSUFFICIENT_ACCESS``
    only ``lab`` can be referred to at all.  Informative questions still exist --
    the count is reported -- and none is reachable, so nothing is asked.  That is
    ``BLOCKED``, and the claim is checked by running the same planner with full
    access and watching it finish.

``NOTHING_TO_ASK``
    unnameable twins no relation distinguishes, with everything open.  No
    question is informative at any access level, so this is ``IMPOSSIBLE`` and
    the check is that full access still does not finish it.

``HIGH_COST_BUT_ASKABLE`` is the control that keeps the two ideas apart.  The
question that gets asked costs ``(1, 2, needs_context=True)`` and an *unavailable*
one costs ``(1, 2, False)`` -- strictly cheaper, and strictly dominating under
L8.36's Pareto rule.  With full access that cheaper one is chosen; with the
episodes out of reach the dominated one is asked anyway.  If accessibility had
been folded into the cost vector this could not happen: unavailable would just be
another price, and the planner would have had to be talked out of it rather than
never offered it.

``ACCESS_UNLOCKS`` runs the state change: ``BLOCKED``, the person opens the
episode, and the same planner finishes in one question.  That is the difference
``BLOCKED`` exists to carry -- reported as ``IMPOSSIBLE`` it would have
abandoned a task one action away from done.
"""
from __future__ import annotations

from dataclasses import dataclass

from .contextual_joint_l838_experiment import ContextualJoint
from .cost_aware_dialogue_l836_experiment import IMPOSSIBLE, RESOLVED
from .cost_aware_joint_l839_experiment import worlds
from .cross_context_identification_l829_experiment import universe_from
from .holdout_binding_l8211_experiment import build_holdout
from .holdout_joint_dialogue_l8391_experiment import (
    TWIN_A, TWIN_B, build_universes,
)
from .holdout_lexicon_l8281_experiment import widen_store
from .question_accessibility_l842_experiment import (
    BLOCKED, Access, askable, cheapest_unavailable, diagnose, run_accessible,
)

SURFACES = ("フガ率", "ホゲ尺")
SIM, LAB, PILOT = "sim", "lab", "pilot"
RUNTIME, RATE, ENERGY = "runtime_seconds", "success_rate", "energy_error"


def base_universe():
    return universe_from(widen_store(build_holdout(24)[0]))


def three_world():
    """Two fixed episodes and one open episode with three admissible pairings."""
    return ContextualJoint(SURFACES, (
        (SIM, frozenset({(RUNTIME, RATE)})),
        (LAB, frozenset({(RUNTIME, RATE), (RATE, ENERGY), (ENERGY, RUNTIME)})),
        (PILOT, frozenset({(RATE, ENERGY)}))))


def two_world():
    return ContextualJoint(SURFACES, (
        (SIM, frozenset({(RUNTIME, RATE)})),
        (LAB, frozenset({(RUNTIME, RATE), (RATE, ENERGY)}))))


def twins_world():
    return ContextualJoint(SURFACES, (
        (SIM, frozenset({(TWIN_A, TWIN_B), (TWIN_B, TWIN_A)})),))


THREE_TRUTH = ((SIM, (RUNTIME, RATE)), (LAB, (ENERGY, RUNTIME)),
               (PILOT, (RATE, ENERGY)))
TWO_TRUTH = ((SIM, (RUNTIME, RATE)), (LAB, (RATE, ENERGY)))
TWINS_TRUTH = ((SIM, (TWIN_A, TWIN_B)),)


@dataclass(frozen=True)
class Family:
    name: str
    world: object
    truth: object
    access: Access
    coined: bool
    expect_state: str
    expect_turns: int | None
    note: str


def build_families() -> tuple[Family, ...]:
    everything = Access.full((SIM, LAB, PILOT))
    return (
        Family("ORACLE_ACCESS", three_world, THREE_TRUTH, everything, False,
               RESOLVED, 1, "everything open: one three-way question finishes it"),
        Family("INDIRECT_ACCESS", three_world, THREE_TRUTH,
               Access.of(known=(SIM, LAB, PILOT), detail=(SIM, PILOT)), False,
               RESOLVED, 2,
               "the lab details are closed; two reachable questions finish it"),
        Family("INSUFFICIENT_ACCESS", three_world, THREE_TRUTH,
               Access.of(known=(LAB,)), False, BLOCKED, 0,
               "informative questions exist and none is reachable"),
        Family("HIGH_COST_BUT_ASKABLE", two_world, TWO_TRUTH,
               Access.of(known=(LAB,), detail=(LAB,)), False, RESOLVED, 1,
               "control: the reachable question is the dominated one, and is asked"),
        Family("NOTHING_TO_ASK", twins_world, TWINS_TRUTH,
               Access.full((SIM,)), True, IMPOSSIBLE, 0,
               "no question is informative at any access level"),
    )


def _ratio(numerator, denominator):
    if denominator == 0:
        return "n/a"
    return round(numerator / denominator, 3)


def run_family(family: Family, universes) -> dict:
    nameable, coined = universes
    universe = coined if family.coined else nameable
    joint = family.world()
    space = worlds(joint)
    contexts = joint.contexts

    found = diagnose(space, contexts, universe, SURFACES, family.access,
                     family.truth)
    session = run_accessible(space, contexts, universe, SURFACES, family.truth,
                             family.access)
    full = run_accessible(space, contexts, universe, SURFACES, family.truth,
                          Access.full(contexts))

    blocked_out = cheapest_unavailable(space, contexts, universe, SURFACES,
                                       family.access)
    asked_costs = [q.cost for q in session.asked]
    dominated = any(other.cost.dominates(cost)
                    for cost in asked_costs for other in blocked_out)

    state = found.state if not session.asked else session.state
    if found.state in (BLOCKED, IMPOSSIBLE):
        state = found.state

    return {
        "name": family.name, "note": family.note,
        "worlds": len(space), "access_known": sorted(family.access.known),
        "access_detail": sorted(family.access.detail),
        "informative": found.informative, "available": found.available,
        "state": state, "expect_state": family.expect_state,
        "state_correct": state == family.expect_state,
        "turns": session.turns, "expect_turns": family.expect_turns,
        "turns_correct": (family.expect_turns is None
                          or session.turns == family.expect_turns),
        "kinds": [q.kind for q in session.asked],
        "asked_costs": [cost.vector for cost in asked_costs],
        "unavailable_costs": sorted({q.cost.vector for q in blocked_out}),
        "dominated_cost_accepted": dominated,
        "resolved": session.resolved, "wrong": session.wrong,
        "unaskable_asked": session.unaskable_asked,
        "unlockable": found.unlockable,
        "full_access_resolves": full.resolved,
        "full_access_turns": full.turns,
        "full_access_kinds": [q.kind for q in full.asked],
        "full_access_costs": [q.cost.vector for q in full.asked],
    }


def run_unlock(universes) -> dict:
    """The state change: BLOCKED, the person opens the episode, one question."""
    universe = universes[0]
    joint = three_world()
    space, contexts = worlds(joint), joint.contexts
    before_access = Access.of(known=(LAB,))
    before = diagnose(space, contexts, universe, SURFACES, before_access,
                      THREE_TRUTH)
    after_access = before_access.opening(LAB)
    after = run_accessible(space, contexts, universe, SURFACES, THREE_TRUTH,
                           after_access)
    return {
        "before": before.state, "before_available": before.available,
        "after": after.state, "after_turns": after.turns,
        "after_resolved": after.resolved, "after_wrong": after.wrong,
        "correct": (before.state == BLOCKED and after.resolved
                    and not after.wrong),
    }


def run_impossible_stays(universes) -> dict:
    """The control on the other side: IMPOSSIBLE must not be unlocked by access."""
    universe = universes[1]
    joint = twins_world()
    space, contexts = worlds(joint), joint.contexts
    found = diagnose(space, contexts, universe, SURFACES,
                     Access.of(known=(SIM,)), TWINS_TRUTH)
    opened = run_accessible(space, contexts, universe, SURFACES, TWINS_TRUTH,
                            Access.full(contexts))
    return {"state": found.state, "resolved_with_full_access": opened.resolved,
            "correct": found.state == IMPOSSIBLE and not opened.resolved}


def run_experiment() -> dict:
    universes = build_universes()
    runs = [run_family(family, universes) for family in build_families()]
    unlock = run_unlock(universes)
    stays = run_impossible_stays(universes)

    blocked = [r for r in runs if r["state"] == BLOCKED]
    impossible = [r for r in runs if r["state"] == IMPOSSIBLE]
    oracle = next(r for r in runs if r["name"] == "ORACLE_ACCESS")
    indirect = next(r for r in runs if r["name"] == "INDIRECT_ACCESS")
    expensive = next(r for r in runs if r["name"] == "HIGH_COST_BUT_ASKABLE")

    return {
        "runs": runs, "unlock": unlock, "impossible_stays": stays,
        "metrics": {
            "false_joint_resolution_rate": _ratio(
                sum(1 for r in runs if r["wrong"]), len(runs)),
            "unaskable_question_rate": _ratio(
                sum(r["unaskable_asked"] for r in runs),
                sum(r["turns"] for r in runs)),
            "diagnosis_accuracy": _ratio(
                sum(1 for r in runs if r["state_correct"]), len(runs)),
            "turns_accuracy": _ratio(
                sum(1 for r in runs if r["turns_correct"]), len(runs)),
            "blocked_prediction_accuracy": _ratio(
                sum(1 for r in blocked if r["full_access_resolves"]),
                len(blocked)),
            "impossible_prediction_accuracy": _ratio(
                sum(1 for r in impossible if not r["full_access_resolves"]),
                len(impossible)),
            "oracle_turns": oracle["turns"],
            "accessible_turns": indirect["turns"],
            "access_cost_of_detour": indirect["turns"] - oracle["turns"],
            "dominated_cost_accepted": expensive["dominated_cost_accepted"],
            # Measured rather than asserted: with the episodes reachable the
            # cheaper question is the one the same planner picks.
            "cheaper_preferred_when_reachable": (
                expensive["full_access_costs"] != expensive["asked_costs"]
                and all(cost == (1, 2, 0)
                        for cost in expensive["full_access_costs"])),
            "blocked_then_resolved": unlock["correct"],
            "impossible_stays_impossible": stays["correct"],
        },
    }


def format_experiment(report) -> str:
    found = report["metrics"]
    lines = ["## Question accessibility (L8.42.1)", "",
             "| metric | value |", "| --- | --- |",
             f"| **false_joint_resolution_rate** | **{found['false_joint_resolution_rate']}** |",
             f"| **unaskable_question_rate** | **{found['unaskable_question_rate']}** |",
             f"| diagnosis_accuracy | {found['diagnosis_accuracy']} |",
             f"| turns_accuracy | {found['turns_accuracy']} |",
             f"| **blocked_prediction_accuracy** "
             f"| **{found['blocked_prediction_accuracy']}** |",
             f"| **impossible_prediction_accuracy** "
             f"| **{found['impossible_prediction_accuracy']}** |",
             f"| oracle turns / reachable turns "
             f"| {found['oracle_turns']} / {found['accessible_turns']} "
             f"(detour {found['access_cost_of_detour']}) |",
             f"| **dominated cost accepted when it is all there is** "
             f"| **{found['dominated_cost_accepted']}** |",
             f"| blocked -> opened -> resolved | {found['blocked_then_resolved']} |",
             f"| impossible stays impossible | {found['impossible_stays_impossible']} |",
             "", "### the families", "",
             "| family | referable | recallable | informative | reachable "
             "| state | turns |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for run in report["runs"]:
        mark = "" if run["state_correct"] else " (NG)"
        lines.append(
            f"| {run['name']} | {'、'.join(run['access_known']) or '-'} "
            f"| {'、'.join(run['access_detail']) or '-'} "
            f"| {run['informative']} | {run['available']} "
            f"| {run['state']}{mark} | {run['turns']} |")

    expensive = next(r for r in report["runs"]
                     if r["name"] == "HIGH_COST_BUT_ASKABLE")
    lines += ["", "### unavailable is not expensive", "",
              f"With the episodes out of reach the planner asks a question "
              f"costing `{expensive['asked_costs']}`, while questions costing "
              f"`{expensive['unavailable_costs']}` sit unavailable -- strictly "
              f"cheaper under L8.36's Pareto rule, and strictly preferred when "
              f"they are reachable. A cost vector cannot express that, which is "
              f"why `askable` is not in it. With the same world fully reachable "
              f"the same planner picks `{expensive['full_access_kinds']}` at "
              f"`{expensive['full_access_costs']}` instead "
              f"(cheaper_preferred_when_reachable = "
              f"{found['cheaper_preferred_when_reachable']}).",
              "", "### the state change", ""]
    unlock = report["unlock"]
    lines += [f"- before opening the episode: **{unlock['before']}** "
              f"({unlock['before_available']} reachable questions)",
              f"- the person opens it: **{unlock['after']}** in "
              f"{unlock['after_turns']} question",
              f"- and the world that is genuinely out of reach stays that way: "
              f"`{report['impossible_stays']['state']}`, still unresolved with "
              f"every episode open."]
    return "\n".join(lines)


def main() -> None:  # pragma: no cover
    print(format_experiment(run_experiment()))


if __name__ == "__main__":  # pragma: no cover
    main()
