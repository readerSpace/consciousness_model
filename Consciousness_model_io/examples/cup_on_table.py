"""Goal-directed example: "机の上のコップを取れ" (grasp the cup on the table).

Shows how to *configure* one camera input channel and two action outputs, then
let the agent pick actions by minimising the goal distance
``a* = argmin_a D(P(a, B_t), G_t)``.

Run:  python -m examples.cup_on_table      (from the package root)
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


# --- configure the INPUT ---------------------------------------------------
def vision_to_facts(frame: dict) -> list[Fact]:
    """Turn a (mock) perception dict into observation facts O_t."""
    facts: list[Fact] = []
    for obj, info in frame.get("objects", {}).items():
        if "on" in info:
            facts.append(fact("ON", obj, info["on"]))
        if "at" in info:
            facts.append(fact("AT", obj, tuple(info["at"])))
        if info.get("movable"):
            facts.append(fact("state", obj, "movable"))
        if info.get("reachable"):
            facts.append(fact("REACHABLE", "robot", obj))
    if frame.get("holding"):
        facts.append(fact("HAVE", "robot", frame["holding"]))
    return facts


camera = SensorChannel("camera", encode=vision_to_facts)


# --- configure the OUTPUT --------------------------------------------------
def reach_effect(beliefs, binding) -> PredictedEffect:
    # Reaching makes the cup reachable if we can see where it is.
    if any(f.predicate == "AT" and f.args[0] == "cup" for f in beliefs):
        return PredictedEffect(added=(fact("REACHABLE", "robot", "cup"),), confidence=0.9)
    return PredictedEffect(added=(), confidence=0.4)


def grasp_effect(beliefs, binding) -> PredictedEffect:
    # Grasping succeeds once the cup is reachable and movable.
    reachable = any(f.predicate == "REACHABLE" and f.args == ("robot", "cup") for f in beliefs)
    movable = any(f.predicate == "state" and f.args == ("cup", "movable") for f in beliefs)
    if reachable and movable:
        return PredictedEffect(added=(fact("HAVE", "robot", "cup"),), confidence=0.95)
    return PredictedEffect(added=(), confidence=0.5)


reach = ActionSchema("REACH", effect=reach_effect, cost=1.0)
grasp = ActionSchema("GRASP", effect=grasp_effect, cost=1.5)


def main() -> None:
    agent = ConsciousAgent(
        sensors=[camera],
        actions=[reach, grasp],
        # Intermediate subgoal REACHABLE makes the two-step plan visible under
        # the spec's greedy one-step selection a* = argmin_a D(P(a,B_t), G_t).
        goals=[fact("REACHABLE", "robot", "cup"), fact("HAVE", "robot", "cup")],
        config=AgentConfig(policy="auto"),
    )

    # A tiny scripted world: perception is stable; the chosen action's predicted
    # effect is folded back as the next observation to advance the world.
    frame = {"objects": {"cup": {"on": "table", "at": [1.2, 0.3, 0.8], "movable": True}}}

    for _ in range(4):
        state = agent.step({"camera": frame})
        print(state.summary())
        print()
        if state.goal_distance == 0.0:
            print(">>> goal reached: HAVE(robot, cup)")
            break
        act = state.decision.action
        if act is None:
            print(">>> no progressing action available")
            break
        # World responds: realise the predicted effect of the chosen action.
        for produced in act.predict(agent.beliefs.as_set()).added:
            if produced.predicate == "REACHABLE":
                frame["objects"]["cup"]["reachable"] = True
            elif produced.predicate == "HAVE":
                frame["holding"] = produced.args[1]


if __name__ == "__main__":
    main()
