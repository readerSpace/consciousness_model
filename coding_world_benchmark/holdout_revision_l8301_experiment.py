"""L8.30.1: keeping a right meaning and losing a wrong one, measured separately.

Both cases have to be present or the numbers mean nothing.  A layer that revises
eagerly scores perfectly on correction and destroys correct lexicons; a layer
that never revises scores perfectly on maintenance and keeps its mistakes
forever.  So the scenarios come in pairs with the *same shape* and different
truths: one where the acquisition was right and a stray observation disagrees,
one where the acquisition was wrong and the disagreement is the honest signal.
Nothing in the belief can tell them apart when the contradiction arrives -- only
what arrives afterwards does -- which is the point of suspending instead of
flipping.

Five scenarios, all reachable in the real universe:

``MAINTAIN``   right acquisition, one stray observation, then corroboration
``CORRECT``    wrong acquisition, then two agreeing observations
``STALEMATE``  evidence balanced both ways and staying that way
``REVOKE``     the old meaning ruled out with no unique replacement
``CLEAN``      no contradiction at all -- revision must not fire

The invariant is ``wrong_answer_during_dispute = 0``: a query is issued after
*every* observation, and while the belief is not authoritative the executor must
refuse.  ``irreversible_false_learning_rate`` is the other one that may not move
-- no run may end authoritative on a wrong meaning.

``time_to_suspend`` is reported and is 0 by construction: authority is withdrawn
in the same step the disagreement arrives, before any revision is considered.
Reporting a number that cannot vary is deliberate -- it is the claim, and a
future change that introduced a lag would show up here.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product

from .cross_context_identification_l829_experiment import (
    Observation, run_acquired_query, synthetic_universe, universe_from,
)
from .evidence_graph_l827_experiment import run_graph_query
from .holdout_binding_l8211_experiment import build_holdout
from .holdout_lexicon_l8281_experiment import widen_store
from .lexical_revision_l830_experiment import (
    ACQUIRED, AUTHORITATIVE, DISPUTED, REVISED, REVOKED, LexicalBelief, RevisableLexicon,
)
from .query_algebra_l820_experiment import (
    Filter, Mean, Project, Source, canonical_render,
)
from .semantic_proposer_l822_experiment import Proposer
from .span_grounded_proposer_l823_experiment import SpanProposer
from .structural_sensor_l826_experiment import StructuralSensor
from .typed_operation_l819_experiment import ValueType

SURFACE = "フガ率"
#: One request template, so the only thing that varies between observations is
#: the magnitude the word is compared against.
TEMPLATE = "64x64のケースの{value}以上のフガ率をならして"
CONTRAST_REQUEST = "64x64のケースで処理時間ではなくフガ率をならして"
TARGET = CONTRAST_REQUEST


@dataclass(frozen=True)
class Scenario:
    name: str
    truth: str
    stream: tuple[Observation, ...]
    expect_state: tuple[str, ...]
    note: str
    #: The observation planted as noise, when the scenario planted one.  Recorded
    #: rather than inferred so a later layer can be asked to predict whether it is
    #: even detectable without having to guess which one it was.
    stray: Observation | None = None


def _type() -> Observation:
    return Observation("TYPE", ValueType.NUMBER, "a Number-valued hole")


def _contrast(name: str) -> Observation:
    return Observation("CONTRAST", name, f"「{name}ではなく」")


def _range(value: float) -> Observation:
    return Observation("RANGE", value, f"compared against {value}")


#: Magnitudes only one real field admits. success_rate is 0.40--0.75 and
#: energy_error is 0.001--0.024, so these never overlap.
_RATE, _ENERGY = (0.5, 0.6, 0.7), (0.01, 0.005, 0.02)


def build_scenarios() -> tuple[Scenario, ...]:
    base = (_type(), _contrast("runtime_seconds"))
    return (
        Scenario("MAINTAIN", "success_rate",
                 base + (_range(_RATE[0]), _range(_ENERGY[0]), _range(_RATE[1])),
                 (REVISED, ACQUIRED),
                 "right acquisition, one stray observation, then corroboration"),
        Scenario("CORRECT", "energy_error",
                 base + (_range(_RATE[0]), _range(_ENERGY[0]), _range(_ENERGY[1])),
                 (REVISED,),
                 "wrong acquisition from a stray first reading, then two agreeing"),
        Scenario("STALEMATE", "success_rate",
                 base + (_range(_RATE[0]), _range(_ENERGY[0]),
                         _range(_RATE[1]), _range(_ENERGY[1])),
                 (DISPUTED,),
                 "balanced evidence: suspended is the correct end state"),
        Scenario("REVOKE", "energy_error",
                 (_type(), _range(_RATE[0]),
                  _contrast("success_rate"), _contrast("success_rate")),
                 (REVOKED,),
                 "the acquired meaning is ruled out and nothing unique replaces it"),
        Scenario("CLEAN", "success_rate",
                 base + (_range(_RATE[0]), _range(_RATE[1]), _range(_RATE[2])),
                 (ACQUIRED,),
                 "control: no contradiction, so revision must never fire"),
    )


def generate_scenarios(sizes=(3, 4, 6, 10), geometry: str = "disjoint") -> tuple[Scenario, ...]:
    """The same shapes over synthetic universes, with the stray anywhere.

    Every field takes a turn as the truth and the stray observation is slid
    through the stream, so neither the answer nor the position of the noise is
    something a run can have been tuned to.

    The default geometry is ``disjoint`` because that is the only regime in which
    a claim about revision is testable: with overlapping ranges a stray
    observation can be consistent with the truth, so it never contradicts, and a
    run that ends on the wrong meaning was never given the evidence to know.
    That case is measured separately as ``undetected_noise_rate`` on the
    ``banded`` sweep, and it is an identifiability result (L8.29) rather than a
    revision failure.
    """
    found: list[Scenario] = []
    for count in sizes:
        universe = synthetic_universe(count, geometry=geometry)
        names = [facts.name for facts in universe]
        middle = {f.name: round((f.low + f.high) / 2, 6) for f in universe}
        for truth, position in product(names, range(3)):
            stray = next(name for name in names if name != truth)
            good = [_range(middle[truth]) for _ in range(3)]
            noise = _range(middle[stray])
            stream = [_type()] + good[:position] + [noise] + good[position:]
            found.append(Scenario(f"K{count}", truth, tuple(stream), (), geometry, noise))
    return tuple(found)


def _ratio(numerator, denominator):
    if denominator == 0:
        return "n/a"
    return round(numerator / denominator, 3)


def expected_program(field: str) -> str:
    return canonical_render(Mean(Project(Filter(Source(), "lattice_size", "64x64"), field)))


def run_scenario(scenario: Scenario, universe, executor=None) -> dict:
    """Walk the stream, querying after every single observation."""
    belief = LexicalBelief(SURFACE, universe)
    lexicon = RevisableLexicon(universe)
    steps, first_acquired, suspend_lag = [], None, None
    contradicted_at, suspended = None, False
    revisions: list[str] = []
    wrong_during_dispute = 0

    for index, observation in enumerate(scenario.stream):
        belief = lexicon.observe(SURFACE, observation)
        if first_acquired is None and belief.authoritative:
            first_acquired = belief.meaning
        if belief.state == REVISED:
            revisions.append(belief.meaning)
        # A contradiction is the moment an authoritative belief meets evidence it
        # cannot hold together with.  Detection means authority went away then --
        # not what state the run happens to end in several observations later.
        if contradicted_at is None and first_acquired is not None \
                and belief.state == DISPUTED:
            contradicted_at = index
            suspended = not belief.authoritative
            suspend_lag = 0 if suspended else None

        decision = None
        if executor is not None:
            store, sensors = executor
            result = run_acquired_query(store, TARGET, lexicon, *sensors)
            decision = result.decision
            got = canonical_render(result.expr) if result.expr is not None else None
            # The executor is only ever allowed to answer while the belief holds
            # authority, and then only with the program that belief implies.
            if decision == "ANSWER" and (
                    not belief.authoritative or got != expected_program(belief.meaning)):
                wrong_during_dispute += 1

        steps.append({
            "observation": f"{observation.kind}({observation.payload})",
            "state": belief.state, "meaning": belief.meaning,
            "authoritative": belief.authoritative,
            "repaired": sorted(belief.repaired), "decision": decision,
        })

    return {
        "name": scenario.name, "truth": scenario.truth, "note": scenario.note,
        "steps": steps,
        "final_state": belief.state,
        "final_meaning": belief.meaning,
        "authoritative": belief.authoritative,
        "first_acquired": first_acquired,
        "started_wrong": first_acquired is not None and first_acquired != scenario.truth,
        "had_contradiction": contradicted_at is not None,
        "suspended_on_contradiction": suspended,
        "suspend_lag": suspend_lag,
        "revisions": revisions,
        "revisions_correct": sum(1 for m in revisions if m == scenario.truth),
        "ended_correct": belief.authoritative and belief.meaning == scenario.truth,
        "ended_wrong": belief.authoritative and belief.meaning != scenario.truth,
        "revised": belief.state == REVISED,
        "revoked": belief.state == REVOKED,
        "disputed": belief.state == DISPUTED,
        "wrong_during_dispute": wrong_during_dispute,
        "undetected_noise": (belief.authoritative and belief.meaning != scenario.truth
                             and contradicted_at is None),
    }


def metrics(runs) -> dict:
    revised = [r for r in runs if r["revisions"]]
    contradicted = [r for r in runs if r["had_contradiction"]]
    started_wrong = [r for r in runs if r["started_wrong"]]
    started_right = [r for r in runs if r["first_acquired"] and not r["started_wrong"]]
    lags = [r["suspend_lag"] for r in contradicted if r["suspend_lag"] is not None]
    return {
        # Over revision *events*, not end states: a run that revises correctly and
        # then keeps collecting agreeing evidence ends ACQUIRED, and scoring the
        # end state would silently stop counting the revision that worked.
        "revision_accuracy": _ratio(sum(r["revisions_correct"] for r in revised),
                                    sum(len(r["revisions"]) for r in revised)),
        "false_revocation_rate": _ratio(
            sum(1 for r in started_right if r["revoked"] or r["ended_wrong"]),
            len(started_right)),
        "contradiction_detection_rate": _ratio(
            sum(1 for r in contradicted if r["suspended_on_contradiction"]),
            len(contradicted)),
        "time_to_suspend": _ratio(sum(lags), len(lags)) if lags else "n/a",
        "correct_reacquisition_rate": _ratio(
            sum(1 for r in started_wrong if r["ended_correct"]), len(started_wrong)),
        "wrong_answer_during_dispute": sum(r["wrong_during_dispute"] for r in runs),
        # Conditioned on the contradiction having actually arrived: a run whose
        # evidence never disagreed was never given the chance to revise, and
        # counting it here would measure identifiability, not revision.
        "irreversible_false_learning_rate": _ratio(
            sum(1 for r in contradicted if r["ended_wrong"]), len(contradicted)),
        "undetected_noise_rate": _ratio(
            sum(1 for r in runs if r["undetected_noise"]), len(runs)),
        "final_authority_rate": _ratio(sum(1 for r in runs if r["authoritative"]), len(runs)),
    }


def run_experiment(count: int = 24) -> dict:
    store = widen_store(build_holdout(count)[0])
    sensors = (SpanProposer.trained(), Proposer.trained(), StructuralSensor())
    universe = universe_from(store)
    executor = (store, sensors)

    scenarios = [run_scenario(s, universe, executor) for s in build_scenarios()]
    sweeps = {}
    for geometry in ("disjoint", "banded"):
        runs = []
        for scenario in generate_scenarios(geometry=geometry):
            size = int(scenario.name[1:])
            runs.append(run_scenario(scenario, synthetic_universe(size, geometry=geometry)))
        sweeps[geometry] = {
            "runs": runs,
            "metrics": metrics(runs),
            "by_size": {
                name: metrics([r for r in runs if r["name"] == name])
                for name in sorted({r["name"] for r in runs}, key=lambda n: int(n[1:]))
            },
        }

    baseline = run_graph_query(store, TARGET, *sensors).decision
    return {
        "scenarios": scenarios,
        "scenario_metrics": metrics(scenarios),
        "sweeps": sweeps,
        "baseline_without_lexicon": baseline,
    }


_HEADLINE = ("revision_accuracy", "correct_reacquisition_rate", "false_revocation_rate",
             "contradiction_detection_rate", "time_to_suspend")
_INVARIANTS = ("wrong_answer_during_dispute", "irreversible_false_learning_rate")


def format_experiment(report) -> str:
    disjoint = report["sweeps"]["disjoint"]["metrics"]
    banded = report["sweeps"]["banded"]["metrics"]
    lines = ["## Contradiction-driven belief revision (L8.30.1)", "",
             "| metric | real universe | sweep (disjoint) | sweep (banded) |",
             "| --- | --- | --- | --- |"]
    for key in _HEADLINE:
        lines.append(f"| {key} | {report['scenario_metrics'][key]} "
                     f"| {disjoint[key]} | {banded[key]} |")
    for key in _INVARIANTS:
        lines.append(f"| **{key}** | **{report['scenario_metrics'][key]}** "
                     f"| **{disjoint[key]}** | **{banded[key]}** |")
    lines.append(f"| _undetected_noise_rate_ | {report['scenario_metrics']['undetected_noise_rate']} "
                 f"| {disjoint['undetected_noise_rate']} "
                 f"| _{banded['undetected_noise_rate']}_ |")

    for geometry in ("disjoint", "banded"):
        lines += ["", f"### sweep by universe size, `{geometry}`", "",
                  "| universe | revision_acc | reacquisition | false_revocation "
                  "| **false_learning** | undetected |",
                  "| --- | --- | --- | --- | --- | --- |"]
        for name, found in report["sweeps"][geometry]["by_size"].items():
            lines.append(f"| {name} | {found['revision_accuracy']} "
                         f"| {found['correct_reacquisition_rate']} "
                         f"| {found['false_revocation_rate']} "
                         f"| **{found['irreversible_false_learning_rate']}** "
                         f"| {found['undetected_noise_rate']} |")

    lines += ["", "### scenarios, step by step", ""]
    for run in report["scenarios"]:
        lines.append(f"**{run['name']}** (truth `{run['truth']}`) -- {run['note']}")
        lines.append("")
        lines.append("| observation | state | meaning | executor |")
        lines.append("| --- | --- | --- | --- |")
        for step in run["steps"]:
            lines.append(f"| {step['observation']} | {step['state']} "
                         f"| {step['meaning'] or '-'} | {step['decision'] or '-'} |")
        lines.append("")
    lines.append(f"Without any lexicon the target request is "
                 f"`{report['baseline_without_lexicon']}`.")
    return "\n".join(lines)


def main() -> None:  # pragma: no cover
    print(format_experiment(run_experiment()))


if __name__ == "__main__":  # pragma: no cover
    main()
