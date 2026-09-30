"""Curiosity example: "この赤い物体は動かせるか？" (is the red object movable?).

Shows Q_t generation from predictive uncertainty and curiosity-driven action
selection ``a* = argmax_a [ I(Q_t; O_{t+1}|a) - λ·Cost(a) ]``.

The agent has no goal it can act on, so it should choose the *informative*
action (PUSH) that would resolve the open question, over cheap-but-useless ones.

Run:  python -m examples.red_object_curiosity
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from consciousness_io import (
    ActionSchema,
    AgentConfig,
    ConsciousAgent,
    Fact,
    PredictedEffect,
    SensorChannel,
    fact,
)


# --- INPUT: a camera that sees a red object but not its movability ---------
def vision_to_facts(frame: dict) -> list[Fact]:
    facts: list[Fact] = []
    for obj, info in frame.get("objects", {}).items():
        facts.append(fact("color", obj, info["color"]))
        if "movable" in info:  # only known after interaction
            facts.append(fact("state", obj, "movable" if info["movable"] else "fixed"))
    return facts


camera = SensorChannel("camera", encode=vision_to_facts)


# --- OUTPUT: several actions with different information value ---------------
def look_effect(beliefs, binding) -> PredictedEffect:
    # Looking cannot reveal movability; outcome is certain-but-uninformative.
    return PredictedEffect(added=(), confidence=1.0)


def push_effect(beliefs, binding) -> PredictedEffect:
    # Pushing WOULD reveal movability, but the outcome is unknown now -> low conf.
    known = any(f.predicate == "state" and f.args[0] == "red_block" for f in beliefs)
    if known:
        return PredictedEffect(added=(), confidence=1.0)
    return PredictedEffect(added=(fact("state", "red_block", "movable"),), confidence=0.2)


def wait_effect(beliefs, binding) -> PredictedEffect:
    return PredictedEffect(added=(), confidence=1.0)


look = ActionSchema("LOOK", effect=look_effect, cost=0.2)
push = ActionSchema("PUSH", effect=push_effect, cost=1.0)
wait = ActionSchema("WAIT", effect=wait_effect, cost=0.1)


def main() -> None:
    agent = ConsciousAgent(
        sensors=[camera],
        actions=[look, push, wait],
        goals=[],  # no achievable goal -> curiosity should dominate
        config=AgentConfig(policy="auto", question_threshold=0.5, curiosity_weight=0.3),
    )

    frame = {"objects": {"red_block": {"color": "red"}}}  # movability unknown
    state = agent.step({"camera": frame})
    print(state.summary())
    print()
    print("open questions Q_t:", [str(q) for q in state.questions])
    print("chosen action a_t:", state.decision.label, "->", state.decision.rationale)
    assert state.decision.action is not None and state.decision.action.name == "PUSH", \
        "curiosity should choose the informative PUSH"
    print("\nOK: the agent chose the information-maximising action.")


if __name__ == "__main__":
    main()
