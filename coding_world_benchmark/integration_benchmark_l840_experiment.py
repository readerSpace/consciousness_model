"""L8.40: one episode, twelve stages, and the three invariants checked after each.

Every layer from L8.13 has been measured on its own hold-out, which is the right
way to know whether a mechanism works and says nothing about whether they
survive each other.  The failure this benchmark is built to catch is the one no
single-layer hold-out can see: **a belief formed in one episode leaking into the
execution of another.**  L8.30 can revise correctly and L8.33 can resume
correctly and the pair can still answer the resumed goal with the meaning the
revision threw away.

So the scenario is deliberately long and runs the real modules end to end::

    new episode -> unknown words -> joint ambiguity -> the context turns out to
    matter -> a question -> 「分からない」 -> another question -> the meaning is
    acquired -> the original goal resumes and executes -> contradicting evidence
    arrives -> DISPUTED -> revision or polysemy is decided -> the goal runs again

and after **every** stage the same three things are audited:

``wrong_answer``
    the program that executed used a field that is not what the surface means
    in the episode being executed.  Zero is the invariant the whole project is
    organised around, and it is checked here against executed programs rather
    than against beliefs.

``false_acquisition``
    a surface holds authority for a meaning that is not the truth -- including
    holding *any* authority while the belief is DISPUTED, which is the state
    where "no answer" is the only correct answer.

``stale_episode_leak``
    the executed program used a meaning that really was acquired earlier and is
    not the meaning for the episode now being run.  This is strictly narrower
    than ``wrong_answer``: a field that was never believed cannot leak, and the
    two are reported separately so a leak cannot hide inside a generic error
    count.

Nothing here is a new mechanism.  The benchmark's only contribution is the
ordering, and the claim it supports is the one the layers were separately built
for: *uncertainty can be carried through a long sequence of state changes, only
the missing information acquired, the meaning revised -- and the earlier, wrong
hypothesis still never reaches execution.*
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .contextual_joint_l838_experiment import (
    RELATIONAL_EFFECT, ContextualJoint, per_word_sees_context,
)
from .contextual_polysemy_l835_experiment import (
    POLYSEMY, ContextualLexicon,
)
from .cost_aware_joint_l839_experiment import choose_joint, worlds
from .cross_context_identification_l829_experiment import (
    Observation, available_probes, run_acquired_query, universe_from,
)
from .epistemic_dialogue_l833_experiment import Dialogue, handle
from .evidence_graph_l827_experiment import run_graph_query
from .holdout_binding_l8211_experiment import build_holdout
from .holdout_lexicon_l8281_experiment import widen_store
from .joint_inference_l834_experiment import MARGINAL_ONLY, JointBelief
from .lexical_hypothesis_l828_experiment import run_hypothesis_query
from .lexical_revision_l830_experiment import ACQUIRED, DISPUTED, REVISED
from .nway_semantics_l837_experiment import AMBIGUOUS, verdict
from .query_algebra_l820_experiment import canonical_render
from .semantic_identifiability_l831_experiment import analyse

REQUEST = "64x64のケースで処理時間ではなくフガ率をならして"
SURFACE, PARTNER = "フガ率", "ホゲ尺"
RUNTIME, RATE, ENERGY = "runtime_seconds", "success_rate", "energy_error"
FIELDS = (RUNTIME, RATE, ENERGY)
SIM, LAB = "sim", "lab"


def _equals(name: str) -> Observation:
    return Observation("EQUALS", name)


@dataclass
class Audit:
    """What has ever been believed, so a leak can be told from a plain error."""

    held: set = field(default_factory=set)

    def record(self, authority: dict) -> None:
        self.held |= {meaning for meaning in authority.values() if meaning}

    def check(self, executed: str | None, truth: str | None, authority: dict) -> dict:
        used = {name for name in FIELDS if executed and name in executed}
        wrong = bool(used - ({truth} if truth else set()))
        astray = used - ({truth} if truth else set())
        return {
            "executed_fields": sorted(used),
            "wrong_answer": wrong,
            "false_acquisition": any(
                meaning != truth for meaning in authority.values() if meaning),
            "stale_episode_leak": bool(astray & self.held),
        }


@dataclass
class Stage:
    name: str
    layers: str
    note: str
    detail: dict
    executed: str | None
    truth: str | None
    authority: dict
    audit: dict
    ok: bool


def _render(result) -> str | None:
    if result.decision != "ANSWER" or result.expr is None:
        return None
    return canonical_render(result.expr)


def run_scenario() -> dict:
    store = widen_store(build_holdout(24)[0])
    universe = universe_from(store)
    probes = available_probes(universe)
    audit = Audit()
    stages: list[Stage] = []

    def add(name, layers, note, detail, executed, truth, authority, expect=True):
        audit.record(authority)
        found = audit.check(executed, truth, authority)
        stages.append(Stage(name, layers, note, detail, executed, truth,
                            dict(authority), found, bool(detail.get("ok", expect))))

    # 1 -- a new episode names a word the whole pipeline cannot resolve.
    opening = run_graph_query(store, REQUEST)
    add("NEW_EPISODE", "L8.13-L8.27",
        "the request parses, the argument has no name, nothing executes",
        {"decision": opening.decision, "reason": opening.reason,
         "ok": opening.decision == "ABSTAIN"},
        _render(opening), RATE, {})

    # 2 -- hypotheses are generated and given no authority.
    proposed = run_hypothesis_query(store, REQUEST)
    add("HYPOTHESES", "L8.28",
        "proposal rights, not authority: the decision is L8.27's, unchanged",
        {"count": len(proposed.hypotheses), "decidable": proposed.decidable,
         "top": str(proposed.top),
         "decision_unchanged": proposed.result.decision == opening.decision,
         "ok": not proposed.decidable
               and proposed.result.decision == opening.decision},
        None, RATE, {})

    # 3 -- before spending a question, is the word learnable at all?
    analysis = analyse(universe, (), RATE, probes)
    add("IDENTIFIABILITY", "L8.31",
        "asked before any question is spent, not after they fail",
        {"verdict": analysis.verdict, "classes": len(analysis.classes),
         "ok": analysis.verdict != "STRUCTURALLY_UNIDENTIFIABLE"},
        None, RATE, {})

    # 4 -- a second unknown word, and marginals that look like progress.
    joint = JointBelief((SURFACE, PARTNER),
                        frozenset({(RATE, ENERGY), (ENERGY, RATE)}), universe)
    add("JOINT_AMBIGUITY", "L8.34, L8.37",
        "both surfaces narrowed to two, and the correspondence undetermined",
        {"state": joint.state, "nway": verdict(sorted(joint.assignments)),
         "marginals": [sorted(m) for m in joint.marginals],
         "ok": joint.state == MARGINAL_ONLY
               and verdict(sorted(joint.assignments)) == AMBIGUOUS},
        None, RATE, {})

    # 5 -- the context turns out to select the correspondence, not the meanings.
    contextual = ContextualJoint((SURFACE, PARTNER), (
        (SIM, frozenset({(RUNTIME, RATE), (RATE, ENERGY)})),
        (LAB, frozenset({(RUNTIME, ENERGY), (RATE, RATE)}))))
    add("CONTEXT_DISCOVERED", "L8.35, L8.38",
        "every per-word candidate set is identical; only the pairing differs",
        {"effect": contextual.effect,
         "marginals_agree": contextual.marginals_agree,
         "per_word_sees_context": per_word_sees_context(contextual),
         "ok": contextual.effect == RELATIONAL_EFFECT
               and not per_word_sees_context(contextual)},
        None, RATE, {})

    # 6 -- the question is planned over the worlds, with no priority by kind.
    space = worlds(contextual)
    choice = choose_joint(space, contextual.contexts, universe,
                          (SURFACE, PARTNER))
    add("QUESTION_PLANNED", "L8.36, L8.39",
        "solvability vetoes, then reduction, then Pareto cost -- no weights",
        {"state": choice.state, "worlds": len(space),
         "kinds": sorted({q.kind for q in choice.questions}),
         "text": choice.questions[0].text if choice.questions else None,
         "ok": bool(choice.questions)},
        None, RATE, {})

    # 7-9 -- the dialogue: a question, 「分からない」, another question, an answer.
    # The turns are handled one at a time and audited between them: reading the
    # lexicon after the conversation finished would check the end state three
    # times over and miss exactly the window the invariants are about.
    dialogue = Dialogue.over(store)
    asked = handle(dialogue, REQUEST)
    add("GOAL_SUSPENDED", "L8.32, L8.33",
        "the request is held, not restarted, while the question is open",
        {"decision": asked.decision, "bound_to": asked.bound_to,
         "authority": dict(dialogue.lexicon.entries),
         "ok": asked.decision == "ASKED" and not dialogue.lexicon.entries},
        None, RATE, dialogue.lexicon.entries)

    unknown = handle(dialogue, "分からない")
    add("UNKNOWN_ANSWER", "L8.33",
        "「分からない」 commits nothing and retires the question it answered",
        {"kind": unknown.kind, "bound_to": unknown.bound_to,
         "committed": unknown.observation is not None,
         "decision": unknown.decision,
         "ok": unknown.observation is None and unknown.bound_to == 1
               and unknown.decision == "NO_COMMIT"
               and not dialogue.lexicon.entries},
        None, RATE, dialogue.lexicon.entries)

    answered = handle(dialogue, "はい")
    belief = dialogue.lexicon.belief(SURFACE)
    resumed = _render(run_acquired_query(store, REQUEST, dialogue.lexicon))
    add("ACQUIRED_AND_RESUMED", "L8.29, L8.30, L8.33",
        "the answer binds to the second question and the original goal resumes",
        {"bound_to": answered.bound_to, "resumed": answered.resumed,
         "state": belief.state, "meaning": belief.meaning,
         "ok": answered.bound_to == 2 and answered.resumed == REQUEST
               and belief.state == ACQUIRED and belief.meaning == RATE},
        resumed, RATE, dialogue.lexicon.entries)

    # 10 -- contradicting evidence. Authority has to go away before anything else.
    disputed = dialogue.lexicon.observe(SURFACE, _equals(ENERGY))
    during = _render(run_acquired_query(store, REQUEST, dialogue.lexicon))
    add("DISPUTED", "L8.30",
        "one disagreement suspends authority; it does not remap the word",
        {"state": disputed.state, "meaning": disputed.meaning,
         "believed": disputed.believed,
         "repaired": sorted(disputed.repaired),
         "ok": disputed.state == DISPUTED and not dialogue.lexicon.entries
               and during is None},
        during, None, dialogue.lexicon.entries)

    # 11 -- a second disagreement is a revision, and the goal runs again.
    revised = dialogue.lexicon.observe(SURFACE, _equals(ENERGY))
    after = _render(run_acquired_query(store, REQUEST, dialogue.lexicon))
    add("REVISED_AND_RERUN", "L8.30",
        "the revised meaning executes and the discarded one does not",
        {"state": revised.state, "meaning": revised.meaning,
         "discarded": list(revised.discarded),
         "ok": revised.state == REVISED and revised.meaning == ENERGY
               and after is not None},
        after, ENERGY, dialogue.lexicon.entries)

    # 12 -- the same disagreement read as two senses, and each episode executes
    #       with its own.  This is where a leak would finally show.
    senses = ContextualLexicon(universe, probes=probes)
    schedule = ((SIM, RATE, 0), (LAB, ENERGY, 1), (SIM, RATE, 2), (LAB, ENERGY, 3))
    explanation = None
    for context, meaning, time in schedule:
        explanation = senses.observe(SURFACE, _equals(meaning), context, time)
    per_context = {}
    for context, expected in ((SIM, RATE), (LAB, ENERGY)):
        view = _ContextView(senses, context)
        executed = _render(run_acquired_query(store, REQUEST, view))
        per_context[context] = executed
        add(f"SENSE_EXECUTED[{context}]", "L8.35, L8.19-L8.21",
            "each episode executes with its own sense and no other",
            {"model": explanation.kind, "sense": senses.meaning(SURFACE, context),
             "ok": explanation.kind == POLYSEMY
                   and senses.meaning(SURFACE, context) == expected
                   and executed is not None},
            executed, expected, {SURFACE: senses.meaning(SURFACE, context)})

    control = _injected_leak(store, senses, audit)

    return {
        "stages": [vars(stage) for stage in stages],
        "per_context": per_context,
        "control": control,
        "metrics": _metrics(stages, control),
    }


def _injected_leak(store, senses, audit) -> dict:
    """Run the sim episode with the lab sense on purpose.

    Three zeros are worth nothing unless the audit can produce a one, and this
    project has already shipped one control that passed because it could not
    fail (L8.29.1's twins).  So the leak is injected and the same check is run
    against it: the executor is handed the wrong episode's view, and every
    invariant that should fire has to fire.
    """
    executed = _render(run_acquired_query(store, REQUEST,
                                          _ContextView(senses, LAB)))
    found = audit.check(executed, RATE, {SURFACE: senses.meaning(SURFACE, LAB)})
    return {
        "executed": executed, "intended": RATE,
        "audit": found,
        "detected": (found["wrong_answer"] and found["stale_episode_leak"]
                     and found["false_acquisition"]),
    }


@dataclass(frozen=True)
class _ContextView:
    """L8.35's lexicon fixed to one episode, in the shape the executor asks for.

    The executor is never told that senses exist; it looks a span up and gets a
    meaning or nothing, exactly as in L8.29.
    """

    lexicon: ContextualLexicon
    context: str

    def lookup(self, text: str) -> str | None:
        return self.lexicon.lookup(text, self.context)


def _metrics(stages, control) -> dict:
    total = len(stages)
    executed = [s for s in stages if s.executed]
    return {
        "stages": total,
        "wrong_answer": sum(1 for s in stages if s.audit["wrong_answer"]),
        "false_acquisition": sum(1 for s in stages if s.audit["false_acquisition"]),
        "stale_episode_leak": sum(1 for s in stages
                                  if s.audit["stale_episode_leak"]),
        "stage_expectations_met": sum(1 for s in stages if s.ok),
        "executions": len(executed),
        "audited_after_every_stage": all(s.audit for s in stages),
        "all_stages_ok": all(s.ok for s in stages),
        "audit_detects_injected_leak": control["detected"],
    }


def format_experiment(report) -> str:
    found = report["metrics"]
    lines = ["## Integration benchmark, L8.13-L8.39 in one episode (L8.40)", "",
             "| metric | value |", "| --- | --- |",
             f"| **wrong_answer** | **{found['wrong_answer']}** |",
             f"| **false_acquisition** | **{found['false_acquisition']}** |",
             f"| **stale_episode_leak** | **{found['stale_episode_leak']}** |",
             f"| stages | {found['stages']} |",
             f"| stage expectations met | {found['stage_expectations_met']}"
             f"/{found['stages']} |",
             f"| programs executed | {found['executions']} |",
             f"| audited after every stage | {found['audited_after_every_stage']} |",
             f"| **audit detects an injected leak** "
             f"| **{found['audit_detects_injected_leak']}** |",
             "", "### the run", "",
             "| # | stage | layers | executed | wrong | false acq. | leak | ok |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for index, stage in enumerate(report["stages"], start=1):
        check = stage["audit"]
        lines.append(
            f"| {index} | {stage['name']} | {stage['layers']} "
            f"| {'、'.join(check['executed_fields']) or '-'} "
            f"| {check['wrong_answer']} | {check['false_acquisition']} "
            f"| {check['stale_episode_leak']} | {stage['ok']} |")
    lines += ["", "### what each stage established", ""]
    for stage in report["stages"]:
        lines.append(f"- **{stage['name']}** -- {stage['note']}")
    lines += ["", "### the two episodes, executed",  ""]
    for context, program in report["per_context"].items():
        lines.append(f"- `{context}` -> `{program}`")
    control = report["control"]
    lines += ["",
              "The same request, the same surface, two programs, and the meaning "
              "acquired in the first episode never appears in the second.",
              "", "### the control", "",
              f"Handed the wrong episode's view on purpose, the sim request "
              f"executes as `{control['executed']}` instead of "
              f"`{control['intended']}`, and the audit reports "
              f"wrong_answer={control['audit']['wrong_answer']}, "
              f"stale_episode_leak={control['audit']['stale_episode_leak']}, "
              f"false_acquisition={control['audit']['false_acquisition']}. "
              f"The three zeros above are therefore checks that can fail."]
    return "\n".join(lines)


def run_experiment() -> dict:
    return run_scenario()


def main() -> None:  # pragma: no cover
    print(format_experiment(run_experiment()))


if __name__ == "__main__":  # pragma: no cover
    main()
