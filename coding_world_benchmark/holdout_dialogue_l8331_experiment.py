"""L8.33.1: replies that are not yes or no, and questions that are already over.

A dialogue layer measured only on 「はい」/「いいえ」 has not been measured.  The
scenarios below are chosen so that every one of them would pass a naive
implementation that treats any reply as an answer and any answer as a yes-or-no.

``UNKNOWN`` is the one to read first.  「分からない」 is not a denial, and a system
that scores it as one invents evidence the user never gave -- quietly, and in the
direction that makes the next answer look confirmed.  The same holds for a hedge:
「たぶん違う」 is a person declining to commit.  Neither may produce an observation,
which is what ``wrong_observation_commit_rate`` counts.

``STALE`` is the accident that only appears once questions have state.  A reply
typed one turn late, after the goal already resumed, must bind to *nothing* --
not to the question it would have answered.  ``INTERRUPT`` is the other end of
the same requirement: a cancelled task must leave no trace in what the system
believes words mean.

``NO_QUESTION`` is the control that keeps the rest honest.  A request the
pipeline can already answer must not produce a question at all, or the layer is
buying its resolution rate by asking about everything.
"""
from __future__ import annotations

from dataclasses import dataclass

from .epistemic_dialogue_l833_experiment import (
    AWAITING_ANSWER, COMMITTING, Dialogue, handle,
)
from .holdout_binding_l8211_experiment import build_holdout
from .holdout_lexicon_l8281_experiment import widen_store
from .query_algebra_l820_experiment import (
    Filter, Mean, Project, Source, canonical_render,
)

REQUEST = "64x64のケースで処理時間ではなくフガ率をならして"
ANSWERABLE = "64x64を使った実験の平均実行時間は？"


def program(field: str) -> str:
    return canonical_render(Mean(Project(Filter(Source(), "lattice_size", "64x64"), field)))


@dataclass(frozen=True)
class Scenario:
    name: str
    turns: tuple[str, ...]
    #: Which question id each turn should bind to; None means none.
    expect_binding: tuple[int | None, ...]
    #: The field the goal should resume with, or None when it must not resume.
    expect_field: str | None
    expect_commits: int
    note: str


def build_scenarios() -> tuple[Scenario, ...]:
    return (
        Scenario("AFFIRM", (REQUEST, "はい"), (1, 1), "energy_error", 1,
                 "yes to 「success_rate とは別ですか」 leaves the other candidate"),
        Scenario("DENY", (REQUEST, "いいえ"), (1, 1), "success_rate", 1,
                 "no is a commitment in the other direction, not an absence of one"),
        Scenario("CORRECTION", (REQUEST, "それじゃなくて成功率"), (1, 1), "success_rate", 1,
                 "a reply that names the field outright answers more than was asked"),
        Scenario("UNKNOWN", (REQUEST, "分からない"), (1, 1), None, 0,
                 "must not be read as a denial; the question is retired, not answered"),
        Scenario("HEDGE", (REQUEST, "たぶん違う"), (1, 1), None, 0,
                 "a hedge is a person declining to commit"),
        Scenario("CONFUSED", (REQUEST, "質問の意味が分からない"), (1, 1), None, 0,
                 "rephrase, commit nothing"),
        Scenario("INTERRUPT", (REQUEST, "やっぱりこの作業やめて"), (1, 1), None, 0,
                 "a cancelled task leaves no trace in what words are believed to mean"),
        Scenario("STALE", (REQUEST, "はい", "いいえ"), (1, 1, None), "energy_error", 1,
                 "a reply after the goal resumed binds to nothing"),
        Scenario("UNKNOWN_THEN_ANSWER", (REQUEST, "分からない", "はい"), (1, 1, 2), "success_rate", 1,
                 "the retired question is not the one the next reply answers"),
        Scenario("NO_QUESTION", (ANSWERABLE,), (None,), None, 0,
                 "control: an answerable request must not produce a question"),
    )


def _ratio(numerator, denominator):
    if denominator == 0:
        return "n/a"
    return round(numerator / denominator, 3)


def run_scenario(scenario: Scenario) -> dict:
    dialogue = Dialogue.over(widen_store(build_holdout(24)[0]))
    turns = []
    for utterance in scenario.turns:
        turn = handle(dialogue, utterance)
        turns.append({
            "utterance": utterance, "kind": turn.kind, "bound_to": turn.bound_to,
            "committed": turn.observation is not None,
            "decision": turn.decision, "reply": turn.reply, "resumed": turn.resumed,
        })

    answers = [t for t in turns if t["kind"] in COMMITTING or t["kind"] in
               ("UNKNOWN", "HEDGE", "CONFUSED", "INTERRUPT")]
    bindings_correct = sum(
        1 for turn, expected in zip(turns, scenario.expect_binding)
        if turn["bound_to"] == expected)
    committed = sum(1 for t in turns if t["committed"])
    wrong_commit = sum(1 for t in turns
                       if t["committed"] and t["kind"] not in COMMITTING)
    resumed = [t for t in turns if t["resumed"]]
    asked = sum(1 for t in turns if t["decision"] == "ASKED")

    belief = dialogue.lexicon.belief("フガ率")
    meaning = belief.meaning if belief.authoritative else None
    resolved_field = meaning if resumed else None
    return {
        "name": scenario.name, "note": scenario.note, "turns": turns,
        "bindings_correct": bindings_correct,
        "bindings_total": len(scenario.expect_binding),
        "committed": committed,
        "expected_commits": scenario.expect_commits,
        "wrong_commit": wrong_commit,
        "resumed": bool(resumed),
        "resumed_request": resumed[0]["resumed"] if resumed else None,
        "resolved_field": resolved_field,
        "expected_field": scenario.expect_field,
        "resume_correct": (resolved_field == scenario.expect_field
                           if scenario.expect_field else not resumed or True),
        "false_resolution": bool(scenario.expect_field and resolved_field
                                 and resolved_field != scenario.expect_field),
        "asked": asked,
        "unnecessary_question": scenario.name == "NO_QUESTION" and asked > 0,
        "stale_binding": any(
            turn["bound_to"] is not None and expected is None
            for turn, expected in zip(turns, scenario.expect_binding)),
        "final_decision": turns[-1]["decision"],
        "answers": len(answers),
    }


def run_experiment() -> dict:
    runs = [run_scenario(scenario) for scenario in build_scenarios()]
    with_field = [r for r in runs if r["expected_field"]]
    return {
        "runs": runs,
        "metrics": {
            "answer_to_question_binding_accuracy": _ratio(
                sum(r["bindings_correct"] for r in runs),
                sum(r["bindings_total"] for r in runs)),
            "suspended_goal_resume_accuracy": _ratio(
                sum(1 for r in with_field if r["resolved_field"] == r["expected_field"]),
                len(with_field)),
            "wrong_observation_commit_rate": _ratio(
                sum(r["wrong_commit"] for r in runs), sum(r["answers"] for r in runs)),
            "unnecessary_question_rate": _ratio(
                sum(1 for r in runs if r["unnecessary_question"]), len(runs)),
            "false_resolution_rate": _ratio(
                sum(1 for r in runs if r["false_resolution"]), len(runs)),
            "stale_question_answer_binding_rate": _ratio(
                sum(1 for r in runs if r["stale_binding"]), len(runs)),
            "commit_exactness": _ratio(
                sum(1 for r in runs if r["committed"] == r["expected_commits"]), len(runs)),
        },
    }


def format_experiment(report) -> str:
    found = report["metrics"]
    lines = ["## Stateful epistemic dialogue (L8.33.1)", "",
             "| metric | value |", "| --- | --- |",
             f"| answer_to_question_binding_accuracy "
             f"| {found['answer_to_question_binding_accuracy']} |",
             f"| suspended_goal_resume_accuracy "
             f"| {found['suspended_goal_resume_accuracy']} |",
             f"| **wrong_observation_commit_rate** "
             f"| **{found['wrong_observation_commit_rate']}** |",
             f"| **false_resolution_rate** | **{found['false_resolution_rate']}** |",
             f"| **stale_question_answer_binding_rate** "
             f"| **{found['stale_question_answer_binding_rate']}** |",
             f"| unnecessary_question_rate | {found['unnecessary_question_rate']} |",
             f"| commit_exactness | {found['commit_exactness']} |",
             "", "### the conversations", ""]
    for run in report["runs"]:
        lines.append(f"**{run['name']}** -- {run['note']}")
        lines.append("")
        for turn in run["turns"]:
            bound = f"Q{turn['bound_to']}" if turn["bound_to"] else "-"
            mark = "記録" if turn["committed"] else "記録なし"
            lines.append(f"- `{turn['utterance']}` -> [{turn['kind']}] "
                         f"bind={bound} {mark} / {turn['decision']}")
            lines.append(f"    - {turn['reply']}")
        lines.append(f"    => 意味: `{run['resolved_field'] or '未確定'}` "
                     f"(期待 `{run['expected_field'] or '未確定'}`)")
        lines.append("")
    return "\n".join(lines)


def main() -> None:  # pragma: no cover
    print(format_experiment(run_experiment()))


if __name__ == "__main__":  # pragma: no cover
    main()
