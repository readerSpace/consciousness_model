import pytest

from Consciousness_model import ConsciousnessController


def test_integrated_state_is_shared_across_broadcast_consumers():
    controller = ConsciousnessController(capacity=2)
    controller.perceive((("start", "go", "goal"), ("noise", "loops", "noise")))

    integrated = controller.integrate(("start", "goal"))

    assert integrated.concepts
    assert controller.broadcast("planner") is integrated
    assert controller.broadcast("report") is integrated
    assert controller.select_action().kind == "ACT"


def test_reflection_increases_risk_after_a_confident_failure():
    controller = ConsciousnessController()
    controller.perceive((("start", "go", "goal"),))
    controller.integrate(("start", "goal"))
    action = controller.select_action()

    observation = controller.observe_outcome(action, succeeded=False)

    assert observation.prediction_error == pytest.approx(action.confidence)
    assert controller.reflect() > 0.5


def test_broadcast_requires_an_integrated_state():
    with pytest.raises(RuntimeError, match="integrate before broadcast"):
        ConsciousnessController().broadcast("planner")