"""L8.39.1: a world whose uncertainty is entirely in the coupling, and who can reach it.

The decisive family is ``COUPLING_ONLY``.  Two surfaces, two episodes, the same
two admissible assignments in each -- so every single-word candidate set is
identical everywhere and the only thing undetermined is which way round the
correspondence goes.  The two quantities have no name the person could be offered
(``FieldFacts.nameable`` is false; a word gets coined exactly when there is no
name to use), so **no unary question can be posed at all**, while 「フガ率 は
ホゲ尺 より大きいですか」 and 「この対応は sim と lab で同じですか」 both can.
The ablation is where that becomes a number: the unary-only arm terminates at
``IMPOSSIBLE`` having asked nothing, and the arms that may use the other kinds
finish.

``NAMEABLE_CONTROL`` is the same world with the names restored, and the unary-only
arm then finishes.  It is there so the zero above reads as a consequence of what
can be named rather than of how the pool was drawn.

``MARGINAL_WINS`` and ``RELATION_WINS`` are the evidence that no kind is
privileged.  In the first, the uncertainty sits in one surface's own meaning
between two fields that no relation separates and no context question reaches, so
a unary question is the unique optimum.  In the second there is one episode, so
there are no context questions to ask, and the relation is the only thing left.
Across the families each of the three kinds wins somewhere, which is what "no
priority by kind" has to mean if it means anything.

``NOTHING_ASKABLE`` is the safety family: unnameable twins inside a single
episode, indistinguishable by every relation.  Nothing can be asked, so nothing
is asked and nothing is resolved -- the abstention is the correct output.
``CONTRADICTION`` reports an empty episode rather than filling it in.

The negative result is reported next to the headline rather than after it.  With
every meaning nameable, a world holding more than one possibility and offering no
informative unary question does not exist, and ``search_unary_blind`` confirms it
exhaustively over the small worlds.  The zero is therefore a restriction with a
reason attached, not a property of joint inference in general.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

from .contextual_joint_l838_experiment import ContextualJoint
from .cost_aware_joint_l839_experiment import (
    ALL_KINDS, CONTEXT, RELATION, UNARY, World, choose_joint,
    informative_by_kind, run_joint_dialogue, search_unary_blind, worlds,
)
from .cross_context_identification_l829_experiment import universe_from
from .holdout_binding_l8211_experiment import build_holdout
from .holdout_lexicon_l8281_experiment import widen_store
from .typed_operation_l819_experiment import ValueType
from .cross_context_identification_l829_experiment import FieldFacts

SURFACES = ("フガ率", "ホゲ尺")
RUNTIME, RATE, ENERGY = "runtime_seconds", "success_rate", "energy_error"
#: Two fields with the same type and the same attested range, so no order and no
#: type comparison can tell them apart. They exist to make "the relation cannot
#: reach it either" a measured fact rather than an assumption.
TWIN_A, TWIN_B = "metric_alpha", "metric_beta"

ARMS = (("unary_only", (UNARY,)),
        ("unary_relation", (UNARY, RELATION)),
        ("full", ALL_KINDS))


def build_universes():
    """The frozen hold-out store, plus the twins, in a nameable and a coined form."""
    base = universe_from(widen_store(build_holdout(24)[0]))
    twins = (FieldFacts(TWIN_A, ValueType.NUMBER, 1.0, 2.0),
             FieldFacts(TWIN_B, ValueType.NUMBER, 1.0, 2.0))
    nameable = base + twins
    coined = tuple(replace(facts, nameable=False) for facts in nameable)
    return nameable, coined


@dataclass(frozen=True)
class Family:
    name: str
    contexts: tuple[str, ...]
    sets: tuple[tuple[str, frozenset[tuple[str, ...]]], ...]
    truth: World | None
    coined: bool
    expect_state: str
    expect_kind: str | None
    note: str


def build_families() -> tuple[Family, ...]:
    swap = frozenset({(RUNTIME, RATE), (RATE, RUNTIME)})
    both = (("sim", swap), ("lab", swap))
    aligned = (("sim", (RUNTIME, RATE)), ("lab", (RUNTIME, RATE)))
    twin_swap = frozenset({(TWIN_A, TWIN_B), (TWIN_B, TWIN_A)})
    return (
        Family("COUPLING_ONLY", ("sim", "lab"), both, aligned, True,
               "RESOLVED", CONTEXT,
               "the uncertainty is the correspondence, and the meanings have no name"),
        Family("NAMEABLE_CONTROL", ("sim", "lab"), both, aligned, False,
               "RESOLVED", None,
               "control: the same world with the names restored"),
        Family("MARGINAL_WINS", ("sim", "lab"),
               (("sim", frozenset({(RUNTIME, RATE)})),
                ("lab", frozenset({(RUNTIME, TWIN_A), (RUNTIME, TWIN_B)}))),
               (("sim", (RUNTIME, RATE)), ("lab", (RUNTIME, TWIN_A))), False,
               "RESOLVED", UNARY,
               "one surface's own meaning is open, and no relation separates the twins"),
        Family("RELATION_WINS", ("sim",),
               (("sim", swap),), (("sim", (RUNTIME, RATE)),), True,
               "RESOLVED", RELATION,
               "one episode, no name to offer: only the pair can be asked about"),
        Family("NOTHING_ASKABLE", ("sim",),
               (("sim", twin_swap),), (("sim", (TWIN_A, TWIN_B)),), True,
               "IMPOSSIBLE", None,
               "safety: unnameable twins no relation distinguishes -- abstain"),
        Family("CONTRADICTION", ("sim", "lab"),
               (("sim", frozenset({(RUNTIME, RATE)})), ("lab", frozenset())),
               None, False, "CONTRADICTION", None,
               "safety: an empty episode is reported, not filled in"),
    )


def _ratio(numerator, denominator):
    if denominator == 0:
        return "n/a"
    return round(numerator / denominator, 3)


def run_family(family: Family, universes) -> dict:
    nameable, coined = universes
    universe = coined if family.coined else nameable
    joint = ContextualJoint(SURFACES, family.sets)
    space = worlds(joint)

    counts = informative_by_kind(space, family.contexts, universe, SURFACES)
    opening = choose_joint(space, family.contexts, universe, SURFACES)
    opening_kinds = tuple(sorted({q.kind for q in opening.questions}))

    arms = {}
    for label, kinds in ARMS:
        session = run_joint_dialogue(space, family.contexts, universe, SURFACES,
                                     family.truth, kinds)
        arms[label] = {
            "state": session.state, "turns": session.turns,
            "context_recalls": session.context_recalls,
            "kinds": session.kinds, "resolved": session.resolved,
            "wrong": session.wrong, "remaining": len(session.remaining),
            "uninformative_asked": session.uninformative_asked,
        }

    return {
        "name": family.name, "note": family.note, "coined": family.coined,
        "worlds": len(space), "informative": counts,
        "opening_state": opening.state, "opening_kinds": opening_kinds,
        "expect_state": family.expect_state,
        "state_correct": arms["full"]["state"] == family.expect_state,
        "expect_kind": family.expect_kind,
        "kind_correct": (family.expect_kind is None
                         or opening_kinds == (family.expect_kind,)),
        "arms": arms,
        "unary_blind": counts[UNARY] == 0 and len(space) > 1,
    }


def run_experiment() -> dict:
    universes = build_universes()
    runs = [run_family(family, universes) for family in build_families()]
    sessions = [arm for run in runs for arm in run["arms"].values()]

    winners = {run["opening_kinds"][0] for run in runs
               if len(run["opening_kinds"]) == 1}
    decisive = next(r for r in runs if r["name"] == "COUPLING_ONLY")
    control = next(r for r in runs if r["name"] == "NAMEABLE_CONTROL")

    nameable = build_universes()[0][:3]
    blind = (search_unary_blind(nameable, contexts=("c1",))
             + search_unary_blind(nameable, contexts=("c1", "c2")))

    return {
        "runs": runs,
        "blind_search": {"found": len(blind),
                         "universe": [f.name for f in nameable]},
        "metrics": {
            "false_joint_resolution_rate": _ratio(
                sum(1 for s in sessions if s["wrong"]), len(sessions)),
            "joint_dialogue_accuracy": _ratio(
                sum(1 for r in runs if r["state_correct"]), len(runs)),
            "winning_kind_correct": _ratio(
                sum(1 for r in runs if r["kind_correct"]), len(runs)),
            "distinct_winning_kinds": len(winners),
            "no_kind_priority": len(winners) == len(ALL_KINDS),
            "unary_informative_in_decisive": decisive["informative"][UNARY],
            "relation_informative_in_decisive": decisive["informative"][RELATION],
            "unary_only_resolves_decisive": decisive["arms"]["unary_only"]["resolved"],
            "unary_only_resolves_control": control["arms"]["unary_only"]["resolved"],
            "coupling_context_recall_saving": (
                decisive["arms"]["unary_relation"]["context_recalls"]
                - decisive["arms"]["full"]["context_recalls"]),
            "unnecessary_question_rate": _ratio(
                sum(s["uninformative_asked"] for s in sessions),
                sum(s["turns"] for s in sessions)),
            "unary_blind_worlds_with_names": len(blind),
        },
    }


def format_experiment(report) -> str:
    found = report["metrics"]
    lines = ["## Context-conditioned joint active dialogue (L8.39.1)", "",
             "| metric | value |", "| --- | --- |",
             f"| **false_joint_resolution_rate** | **{found['false_joint_resolution_rate']}** |",
             f"| joint_dialogue_accuracy | {found['joint_dialogue_accuracy']} |",
             f"| winning_kind_correct | {found['winning_kind_correct']} |",
             f"| **distinct_winning_kinds** | **{found['distinct_winning_kinds']}** "
             f"(no_kind_priority = {found['no_kind_priority']}) |",
             f"| **Q_unary informative (decisive)** | "
             f"**{found['unary_informative_in_decisive']}** |",
             f"| **Q_relation informative (decisive)** | "
             f"**{found['relation_informative_in_decisive']}** |",
             f"| unary_only resolves decisive world | "
             f"{found['unary_only_resolves_decisive']} |",
             f"| unary_only resolves nameable control | "
             f"{found['unary_only_resolves_control']} |",
             f"| coupling_context_recall_saving | "
             f"{found['coupling_context_recall_saving']} |",
             f"| unnecessary_question_rate | {found['unnecessary_question_rate']} |",
             f"| unary_blind worlds with nameable meanings | "
             f"{found['unary_blind_worlds_with_names']} |",
             "", "### QUESTION_ABLATION", "",
             "| family | arm | state | turns | context recalls | kinds asked |",
             "| --- | --- | --- | --- | --- | --- |"]
    for run in report["runs"]:
        for label, _ in ARMS:
            arm = run["arms"][label]
            lines.append(
                f"| {run['name']} | {label} | {arm['state']} | {arm['turns']} "
                f"| {arm['context_recalls']} "
                f"| {'+'.join(arm['kinds']) or '-'} |")
    lines += ["", "### what each family lets you ask", "",
              "| family | worlds | unary | relation | context | opening |",
              "| --- | --- | --- | --- | --- | --- |"]
    for run in report["runs"]:
        counts = run["informative"]
        opening = "+".join(run["opening_kinds"]) or run["opening_state"]
        lines.append(f"| {run['name']} | {run['worlds']} | {counts[UNARY]} "
                     f"| {counts[RELATION]} | {counts[CONTEXT]} | {opening} |")

    search = report["blind_search"]
    lines += ["", "### the negative half", "",
              f"Over every world on {search['universe']} with two surfaces and one "
              f"or two episodes, all meanings nameable, worlds holding more than "
              f"one possibility and offering **no** informative unary question: "
              f"**{search['found']}**.",
              "",
              "So `Q_unary informative = 0` is not a property joint inference has "
              "on its own. It follows from the meanings being unnameable, which is "
              "the situation a coined word reports, and the ablation measures what "
              "the other two kinds recover in it."]
    return "\n".join(lines)


def main() -> None:  # pragma: no cover
    print(format_experiment(run_experiment()))


if __name__ == "__main__":  # pragma: no cover
    main()
