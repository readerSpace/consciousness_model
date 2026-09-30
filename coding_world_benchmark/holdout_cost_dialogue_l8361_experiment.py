"""L8.36.1: cheaper only where cheapness costs nothing, and silence where two tie.

The families separate the three comparisons, so that passing cannot come from
being good at one of them.

``STRUCTURE_ONLY``  L8.35's AMBIGUOUS state. POLYSEMY and REVISION predict the
                    same meaning for every observed context, so **not one
                    question about meaning is even informative** and the target
                    of questioning has to move to the model. This is the layer's
                    reason to exist and it is a capability claim, not a price one.
``CHEAP_EQUIVALENT``  in the same state, the general model question costs a
                    context recall and tells you exactly what the two binary ones
                    tell you, so it is Pareto-dominated and dropped.
``MULTIPLE_OPTIMAL``  the two survivors are genuinely tied. Choosing between them
                    would mean inventing a preference, so both are returned.
``EXPENSIVE_BUT_NECESSARY``  three live meanings: a three-way choice has a bigger
                    answer space than a yes/no and finishes in one turn, so it
                    wins on ambiguity before cost is consulted at all.
``IMPOSSIBLE``      is checked on a **constructed** space rather than on evidence,
                    and the reason is worth stating: in this hypothesis language
                    it appears to be unreachable from any evidence, because the
                    planner can always enumerate the live meanings itself and ask
                    about them. Two accounts that predict the same thing
                    everywhere and share a model are the same object. So the code
                    path is exercised directly, the way L8.24 exercised
                    arbitration on constructed evidence for the same reason -- a
                    rule that is never tested has not been tested.

``avoidable_interaction_rate`` and ``excess_cost_given_equal_resolution`` exist to
fail the implementation that asks about everything: a planner can reach any
resolution rate by spending turns, and those two are where that shows up.
"""
from __future__ import annotations

from dataclasses import dataclass

from .contextual_polysemy_l835_experiment import ContextualObservation
from .cost_aware_dialogue_l836_experiment import (
    ASK, IMPOSSIBLE, MEANING, MULTIPLE_OPTIMAL, RESOLVED, STRUCTURE, Cost,
    Hypothesis, all_questions, choose, hypotheses, preserves_solvability,
    run_dialogue, total_cost,
)
from .cross_context_identification_l829_experiment import Observation, universe_from
from .holdout_binding_l8211_experiment import build_holdout
from .holdout_lexicon_l8281_experiment import widen_store
from .typed_operation_l819_experiment import ValueType

CONTEXTS = ("sim", "lab")
MEANINGS = ("success_rate", "energy_error", "runtime_seconds")


def _equals(name: str) -> Observation:
    return Observation("EQUALS", name)


def _number() -> Observation:
    return Observation("TYPE", ValueType.NUMBER)


@dataclass(frozen=True)
class Family:
    name: str
    evidence: tuple[tuple[Observation, str, int], ...]
    meanings: tuple[str, ...]
    expect_state: str
    expect_kind: str | None
    note: str


def build_families() -> tuple[Family, ...]:
    rate, energy = _equals("success_rate"), _equals("energy_error")
    return (
        Family("STRUCTURE_ONLY",
               ((rate, "sim", 0), (rate, "sim", 1), (energy, "lab", 2), (energy, "lab", 3)),
               MEANINGS, MULTIPLE_OPTIMAL, STRUCTURE,
               "no meaning question is informative; the model is what is unknown"),
        Family("EXPENSIVE_BUT_NECESSARY",
               ((_number(), "sim", 0), (_number(), "sim", 1)),
               MEANINGS, ASK, MEANING,
               "a three-way choice finishes in one turn; yes/no does not"),
        Family("RESOLVED",
               ((rate, "sim", 0), (rate, "lab", 1)),
               MEANINGS, RESOLVED, None,
               "control: already settled, so asking anything is avoidable"),
        Family("POLYSEMY_SETTLED",
               ((rate, "sim", 0), (energy, "lab", 1), (rate, "sim", 2), (energy, "lab", 3)),
               MEANINGS, RESOLVED, None,
               "control: the evidence already picks one account"),
    )


def run_impossible() -> dict:
    """Two accounts nothing in the language distinguishes. Constructed; see above."""
    left = Hypothesis("POLYSEMY", (("a", "x"), ("b", "y")), ())
    right = Hypothesis("POLYSEMY", (("a", "y"), ("b", "x")), ())
    space = frozenset({left, right})
    choice = choose(space, CONTEXTS, MEANINGS)
    return {
        "state": choice.state,
        "questions": len(choice.questions),
        "reported_rather_than_spent": choice.state == IMPOSSIBLE and not choice.questions,
    }


def _ratio(numerator, denominator):
    if denominator == 0:
        return "n/a"
    return round(numerator / denominator, 3)


def run_family(family: Family, universe) -> dict:
    evidence = tuple(ContextualObservation(o, c, t) for o, c, t in family.evidence)
    space = hypotheses(evidence, universe)
    pool = all_questions(space, CONTEXTS, family.meanings)
    choice = choose(space, CONTEXTS, family.meanings)

    meaning_pool = [q for q in pool if q.kind == MEANING]
    admissible = [q for q in pool if preserves_solvability(q, CONTEXTS, family.meanings)]

    chosen = choice.questions[0] if choice.questions else None
    # Was anything equally informative available for less?
    equally = [q for q in admissible
               if chosen and abs(q.expected_size() - chosen.expected_size()) < 1e-9]
    cheaper = [q for q in equally if chosen and q.cost.dominates(chosen.cost)]

    # And the dominated ones that were correctly dropped.
    dropped = [q for q in equally
               if chosen and chosen.cost.dominates(q.cost)]

    truth = min(space, key=str) if space else None
    settled, asked = run_dialogue(space, CONTEXTS, family.meanings, truth) \
        if truth is not None else (space, ())

    return {
        "name": family.name, "note": family.note,
        "space": len(space),
        "models": sorted({h.model for h in space}),
        "state": choice.state,
        "expected_state": family.expect_state,
        "state_correct": choice.state == family.expect_state,
        "chosen_kind": chosen.kind if chosen else None,
        "expected_kind": family.expect_kind,
        "kind_correct": (chosen.kind if chosen else None) == family.expect_kind,
        "chosen": chosen.text if chosen else None,
        "chosen_cost": chosen.cost.vector if chosen else None,
        "options": len(choice.questions),
        "informative_meaning_questions": len(meaning_pool),
        "cheaper_available": len(cheaper),
        "dominated_dropped": len(dropped),
        "asked": len(asked),
        "avoidable": choice.state == RESOLVED and len(asked) > 0,
        "resolved": len(settled) == 1,
        "wrong": len(settled) == 1 and truth is not None and next(iter(settled)) != truth,
        "cost": total_cost(asked).vector if asked else (0, 0, 0),
    }


def run_experiment(count: int = 24) -> dict:
    universe = universe_from(widen_store(build_holdout(count)[0]))
    runs = [run_family(family, universe) for family in build_families()]
    asked = [r for r in runs if r["chosen"]]
    return {
        "runs": runs,
        "impossible": run_impossible(),
        "metrics": {
            "false_resolution_rate": _ratio(sum(1 for r in runs if r["wrong"]), len(runs)),
            "avoidable_interaction_rate": _ratio(
                sum(1 for r in runs if r["avoidable"]), len(runs)),
            "excess_cost_given_equal_resolution": _ratio(
                sum(r["cheaper_available"] for r in asked), len(asked)),
            "dominated_questions_dropped": sum(r["dominated_dropped"] for r in runs),
            "state_accuracy": _ratio(sum(1 for r in runs if r["state_correct"]), len(runs)),
            "question_kind_accuracy": _ratio(
                sum(1 for r in runs if r["kind_correct"]), len(runs)),
        },
    }


def format_experiment(report) -> str:
    found = report["metrics"]
    lines = ["## Cost-aware active dialogue (L8.36.1)", "",
             "| metric | value |", "| --- | --- |",
             f"| **false_resolution_rate** | **{found['false_resolution_rate']}** |",
             f"| **avoidable_interaction_rate** | **{found['avoidable_interaction_rate']}** |",
             f"| **excess_cost_given_equal_resolution** "
             f"| **{found['excess_cost_given_equal_resolution']}** |",
             f"| dominated_questions_dropped | {found['dominated_questions_dropped']} |",
             f"| state_accuracy | {found['state_accuracy']} |",
             f"| question_kind_accuracy | {found['question_kind_accuracy']} |",
             "", "### the families", "",
             "| family | hypotheses | state | chosen | cost | meaning qs |",
             "| --- | --- | --- | --- | --- | --- |"]
    for run in report["runs"]:
        mark = "" if run["state_correct"] else " (NG)"
        lines.append(
            f"| {run['name']} | {run['space']} {run['models']} | {run['state']}{mark} "
            f"| {run['chosen_kind'] or '-'} | {run['chosen_cost'] or '-'} "
            f"| {run['informative_meaning_questions']} |")

    impossible = report["impossible"]
    lines += ["", f"A constructed space nothing can separate reports "
              f"`{impossible['state']}` with {impossible['questions']} questions -- "
              f"it says so rather than spending a turn. It is constructed because "
              f"this hypothesis language appears not to reach that state from "
              f"evidence, which is itself worth recording."]

    structure = next(r for r in report["runs"] if r["name"] == "STRUCTURE_ONLY")
    lines += ["", "### what the structural question is for", "",
              f"In `STRUCTURE_ONLY` the space is {structure['models']} and the number of "
              f"**informative meaning questions is {structure['informative_meaning_questions']}**. "
              f"The two accounts predict the same meaning for every observed context, so "
              f"asking what the word means cannot separate them at all. This is a capability "
              f"claim rather than a price one -- the structural question does not win because "
              f"it is cheaper, it wins because nothing else is even a question.",
              "",
              f"`{structure['options']}` questions survive the comparison and they are "
              f"genuinely tied, so the planner reports `{structure['state']}` instead of "
              f"inventing a preference. A third -- the combined model question -- asks for the "
              f"same information while also requiring a context recall, so it is "
              f"Pareto-dominated and dropped."]
    return "\n".join(lines)


def main() -> None:  # pragma: no cover
    print(format_experiment(run_experiment()))


if __name__ == "__main__":  # pragma: no cover
    main()
