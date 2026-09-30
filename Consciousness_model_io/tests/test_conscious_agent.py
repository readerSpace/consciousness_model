"""Tests for the configurable-I/O set-theoretic consciousness agent."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from consciousness_io import (
    ActionSchema,
    AgentConfig,
    ConsciousAgent,
    Fact,
    PredictedEffect,
    SensorChannel,
    fact,
    goal_distance,
)
from consciousness_io.facts import BeliefStore


# --- fixtures --------------------------------------------------------------

def _camera_encode(frame):
    facts = []
    for obj, info in frame.get("objects", {}).items():
        if "at" in info:
            facts.append(fact("AT", obj, tuple(info["at"])))
        if info.get("movable"):
            facts.append(fact("state", obj, "movable"))
    if frame.get("holding"):
        facts.append(fact("HAVE", "robot", frame["holding"]))
    return facts


def _grasp_effect(beliefs, binding):
    seen = any(f.predicate == "AT" and f.args[0] == "cup" for f in beliefs)
    return PredictedEffect(added=(fact("HAVE", "robot", "cup"),), confidence=0.95 if seen else 0.5)


def _make_agent(**cfg):
    camera = SensorChannel("camera", encode=_camera_encode)
    grasp = ActionSchema("GRASP", effect=_grasp_effect)
    return ConsciousAgent(
        sensors=[camera],
        actions=[grasp],
        goals=[fact("HAVE", "robot", "cup")],
        config=AgentConfig(**cfg),
    )


# --- facts & belief store --------------------------------------------------

def test_fact_set_semantics():
    a = fact("ON", "cup", "table")
    b = fact("ON", "cup", "table")
    assert {a} == {b}  # value equality -> real set membership


def test_belief_overwrite_by_key():
    store = BeliefStore()
    store.update([fact("AT", "cup", (0, 0, 0))])
    store.update([fact("AT", "cup", (1, 2, 3))])
    positions = [f for f in store if f.predicate == "AT"]
    assert len(positions) == 1 and positions[0].args[1] == (1, 2, 3)


def test_goal_distance_counts_unmet_goals():
    store = BeliefStore().update([fact("HAVE", "robot", "cup")])
    assert goal_distance(store, [fact("HAVE", "robot", "cup")]) == 0.0
    assert goal_distance(store, [fact("HAVE", "robot", "ball")]) == 1.0


# --- the cycle -------------------------------------------------------------

def test_observation_becomes_belief():
    agent = _make_agent()
    state = agent.step({"camera": {"objects": {"cup": {"at": [1, 0, 0], "movable": True}}}})
    predicates = {f.predicate for f in state.beliefs}
    assert "AT" in predicates and "state" in predicates
    # O_t is recorded separately from B_t
    assert len(state.observations) >= 2


def test_compression_produces_Z():
    agent = _make_agent()
    state = agent.step({"camera": {"objects": {"cup": {"at": [1, 0, 0], "movable": True}}}})
    assert len(state.compressed) >= 1  # Z_t non-empty


def test_goal_directed_selection_picks_progressing_action():
    agent = _make_agent(policy="auto")
    state = agent.step({"camera": {"objects": {"cup": {"at": [1, 0, 0], "movable": True}}}})
    assert state.decision.mode == "goal"
    assert state.decision.action is not None and state.decision.action.name == "GRASP"


def test_goal_satisfied_yields_noop():
    agent = _make_agent(policy="auto")
    state = agent.step({"camera": {"objects": {}, "holding": "cup"}})
    assert state.goal_distance == 0.0
    assert state.decision.mode == "noop"


# --- questions & curiosity -------------------------------------------------

def test_uncertain_action_becomes_question():
    def push_effect(beliefs, binding):
        return PredictedEffect(added=(fact("state", "x", "movable"),), confidence=0.2)

    camera = SensorChannel("camera", encode=lambda raw: [fact("color", "x", "red")])
    push = ActionSchema("PUSH", effect=push_effect)
    agent = ConsciousAgent([camera], [push], goals=[], config=AgentConfig(question_threshold=0.5))
    state = agent.step({"camera": None})
    assert any(q.origin == "unpredictable_action" for q in state.questions)


def test_curiosity_prefers_informative_action():
    def look_effect(beliefs, binding):
        return PredictedEffect(added=(), confidence=1.0)

    def push_effect(beliefs, binding):
        return PredictedEffect(added=(fact("state", "x", "movable"),), confidence=0.2)

    camera = SensorChannel("camera", encode=lambda raw: [fact("color", "x", "red")])
    agent = ConsciousAgent(
        [camera],
        [ActionSchema("LOOK", effect=look_effect, cost=0.2),
         ActionSchema("PUSH", effect=push_effect, cost=1.0)],
        goals=[],
        config=AgentConfig(policy="auto", curiosity_weight=0.3),
    )
    state = agent.step({"camera": None})
    assert state.decision.mode == "curiosity"
    assert state.decision.action.name == "PUSH"


# --- workspace -------------------------------------------------------------

def test_workspace_is_capacity_bounded():
    def many(raw):
        return [fact("prop", f"obj{i}", i) for i in range(20)]

    camera = SensorChannel("camera", encode=many)
    noop = ActionSchema("NOOP", effect=lambda b, a: PredictedEffect((), 1.0))
    agent = ConsciousAgent([camera], [noop], goals=[fact("prop", "obj3", 3)],
                           config=AgentConfig(workspace_capacity=5))
    state = agent.step({"camera": None})
    assert len(state.workspace) <= 5


def test_workspace_focuses_on_goal_terms():
    def many(raw):
        return [fact("prop", f"obj{i}", i) for i in range(20)]

    camera = SensorChannel("camera", encode=many)
    noop = ActionSchema("NOOP", effect=lambda b, a: PredictedEffect((), 1.0))
    agent = ConsciousAgent([camera], [noop], goals=[fact("prop", "obj7", 7)],
                           config=AgentConfig(workspace_capacity=5))
    state = agent.step({"camera": None})
    assert any(f.args[0] == "obj7" for f in state.workspace)


# --- configuration guards --------------------------------------------------

def test_missing_required_channel_raises():
    camera = SensorChannel("camera", encode=lambda raw: [])
    noop = ActionSchema("NOOP", effect=lambda b, a: PredictedEffect((), 1.0))
    agent = ConsciousAgent([camera], [noop])
    with pytest.raises(KeyError):
        agent.step({"lidar": None})


def test_optional_channel_may_be_absent():
    bump = SensorChannel("bump", encode=lambda raw: [fact("BUMP", "front")], optional=True)
    camera = SensorChannel("camera", encode=lambda raw: [fact("color", "x", "red")])
    noop = ActionSchema("NOOP", effect=lambda b, a: PredictedEffect((), 1.0))
    agent = ConsciousAgent([camera, bump], [noop])
    state = agent.step({"camera": None})  # bump absent -> fine
    assert any(f.predicate == "color" for f in state.observations)


def test_executor_is_invoked():
    calls = []
    camera = SensorChannel("camera", encode=lambda raw: [fact("color", "x", "red")])
    act = ActionSchema("MOVE", effect=lambda b, a: PredictedEffect((), 1.0),
                       execute=lambda binding: calls.append(binding.name))
    agent = ConsciousAgent([camera], [act], goals=[], config=AgentConfig(policy="curiosity"))
    # force a decision by making the action uncertain
    act2 = ActionSchema("MOVE", effect=lambda b, a: PredictedEffect((fact("y", "1"),), 0.2),
                        execute=lambda binding: calls.append(binding.name))
    agent2 = ConsciousAgent([camera], [act2], goals=[], config=AgentConfig(policy="curiosity"))
    agent2.step({"camera": None})
    assert calls == ["MOVE"]
