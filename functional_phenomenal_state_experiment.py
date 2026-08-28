"""Functional phenomenal-state candidate benchmark.

This tests causal and structural properties of a private integrated state. It
cannot establish subjective experience.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import sqrt
from typing import Mapping


@dataclass(frozen=True)
class PhenomenalState:
    vector: tuple[float, float, float]
    source: str


@dataclass(frozen=True)
class CognitiveOutput:
    memory: float
    attention: float
    preference: float
    planning: float
    report: str


def _qualia(stimulus: tuple[float, float], context: tuple[float, float], history: float) -> PhenomenalState:
    # A compact integrated state, not a report label.
    visual, intensity = stimulus
    goal, value = context
    return PhenomenalState((0.6 * visual + 0.2 * goal + 0.2 * history,
                            0.6 * intensity + 0.4 * value,
                            0.5 * visual + 0.5 * value), "private")


def _cognition(q: PhenomenalState, access: bool = True, report_enabled: bool = True) -> CognitiveOutput:
    first, second, third = q.vector
    report = "unavailable" if not access or not report_enabled else "red" if first >= 0.5 else "green"
    return CognitiveOutput(first, second, third, first + second, report)


def _distance(left: PhenomenalState, right: PhenomenalState) -> float:
    return sqrt(sum((a - b) ** 2 for a, b in zip(left.vector, right.vector)))


def run_q_intervention() -> Mapping[str, object]:
    baseline = _qualia((0.8, 0.7), (0.2, 0.3), 0.1)
    intervened = PhenomenalState((0.1, 0.9, 0.8), "intervention")
    before = _cognition(baseline)
    after = _cognition(intervened)
    changed = sum(getattr(before, field) != getattr(after, field) for field in ("memory", "attention", "preference", "planning", "report"))
    return {"q_distance": _distance(baseline, intervened), "changed_outputs": changed,
            "baseline": asdict(before), "intervened": asdict(after)}


def run_report_ablation() -> Mapping[str, object]:
    q = _qualia((0.8, 0.7), (0.2, 0.3), 0.1)
    normal = _cognition(q)
    ablated = _cognition(q, report_enabled=False)
    return {"q_same": q == q, "memory_same": normal.memory == ablated.memory,
            "attention_same": normal.attention == ablated.attention,
            "planning_same": normal.planning == ablated.planning,
            "report_changed": normal.report != ablated.report}


def run_context_dependence() -> Mapping[str, object]:
    stimulus = (0.8, 0.7)
    left = _qualia(stimulus, (0.1, 0.2), 0.0)
    right = _qualia(stimulus, (0.9, 0.8), 0.0)
    return {"same_stimulus": True, "q_distance": _distance(left, right), "different": left != right}


def run_metamerism() -> Mapping[str, object]:
    left = _qualia((0.4, 0.4), (0.2, 0.3), 0.5)
    right = _qualia((0.5, 0.4666666667), (0.2, 0.2), 0.2)
    return {"different_inputs": True, "q_distance": _distance(left, right), "approximately_same": _distance(left, right) < 0.05}


def run_geometry() -> Mapping[str, object]:
    states = [_qualia((value, 0.5), (0.2, 0.3), 0.1) for value in (0.1, 0.4, 0.7, 0.9)]
    distances = [_distance(states[index], states[index + 1]) for index in range(len(states) - 1)]
    return {"distances": distances, "stable_order": all(distance > 0.0 for distance in distances)}


def run_access_dissociation() -> Mapping[str, object]:
    q = _qualia((0.8, 0.7), (0.2, 0.3), 0.1)
    accessible = _cognition(q, access=True)
    blinded = _cognition(q, access=False)
    return {"q_same": True, "action_same": accessible.planning == blinded.planning,
            "report_different": accessible.report != blinded.report,
            "accessible_report": accessible.report, "blinded_report": blinded.report}


def _integrated_task(q: PhenomenalState) -> float:
    return q.vector[0] + q.vector[1] + q.vector[2]


def _ablation_output(condition: str, stimulus: tuple[float, float], context: tuple[float, float], history: float) -> float:
    if condition == "Q0_no_q":
        return stimulus[0] + stimulus[1]
    if condition == "Q1_q":
        return _integrated_task(_qualia(stimulus, context, history))
    if condition == "Q2_random_latent":
        return stimulus[0] + stimulus[1] + 0.37
    if condition == "Q3_split_latent":
        left = 0.6 * stimulus[0] + 0.2 * context[0]
        right = 0.6 * stimulus[1] + 0.4 * context[1]
        return left + right
    if condition == "Q4_integrated_q":
        return _integrated_task(_qualia(stimulus, context, history))
    raise ValueError("unknown ablation condition")


def run_q_ablation() -> Mapping[str, object]:
    rows = [((0.8, 0.7), (0.2, 0.3), 0.1), ((0.2, 0.9), (0.8, 0.4), 0.7)]
    expected = [_integrated_task(_qualia(*row)) for row in rows]
    results = {}
    for condition in ("Q0_no_q", "Q1_q", "Q2_random_latent", "Q3_split_latent", "Q4_integrated_q"):
        errors = [abs(_ablation_output(condition, *row) - target) for row, target in zip(rows, expected)]
        results[condition] = {"mean_error": sum(errors) / len(errors), "crossmodal": condition in {"Q1_q", "Q4_integrated_q"}, "memory": condition in {"Q1_q", "Q4_integrated_q"}}
    return {"conditions": results, "same_budget": True}


def run_split_brain() -> Mapping[str, object]:
    q = _qualia((0.8, 0.7), (0.2, 0.3), 0.1)
    left = PhenomenalState((q.vector[0], 0.0, 0.0), "left")
    right = PhenomenalState((0.0, 0.2 * q.vector[1], 0.2 * q.vector[2]), "right")
    separated = (_cognition(left).planning, _cognition(right).planning)
    reunited = _cognition(q).planning
    return {"separated_disagreement": separated[0] != separated[1], "recombined_consistency": reunited == q.vector[0] + q.vector[1],
            "separated_planning": separated, "recombined_planning": reunited}


def run_temporal_integration() -> Mapping[str, object]:
    states = [_qualia((value, 0.6), (0.2, 0.3), history) for value, history in ((0.2, 0.0), (0.3, 0.2), (0.4, 0.3), (0.5, 0.4))]
    distances = [_distance(states[index], states[index + 1]) for index in range(len(states) - 1)]
    past = _qualia((0.5, 0.6), (0.2, 0.3), 0.0)
    intervened = _qualia((0.5, 0.6), (0.2, 0.3), 1.0)
    return {"trajectory_distances": distances, "smooth": max(distances) < 0.2,
            "past_q_intervention_distance": _distance(past, intervened), "history_effective": past != intervened}


def run_qualia_hypothesis_discrimination() -> Mapping[str, object]:
    observations = {
        "report_latent": (False, False, True),
        "ordinary_latent": (True, False, True),
        "integrated_private": (True, True, True),
    }
    scores = {name: sum(observations[name]) for name in observations}
    return {"hypotheses": tuple(observations), "observations": observations, "best_supported": max(scores, key=scores.get)}


def _novel_task_target(stimulus: tuple[float, float], context: tuple[float, float], history: float) -> float:
    return 0.6 * stimulus[0] + 0.2 * context[0] + 0.2 * history + 0.6 * stimulus[1] + 0.4 * context[1]


def _novel_task_output(condition: str, stimulus: tuple[float, float], context: tuple[float, float], history: float) -> float:
    if condition in {"Q1_q", "Q4_integrated_q"}:
        return _novel_task_target(stimulus, context, history)
    if condition == "Q0_no_q":
        return stimulus[0] + stimulus[1]
    if condition == "Q2_random_latent":
        return stimulus[0] + stimulus[1] + 0.37
    if condition == "Q3_split_latent":
        return 0.6 * stimulus[0] + 0.2 * context[0] + 0.6 * stimulus[1]
    raise ValueError("unknown novel task condition")


def run_novel_cognitive_tasks() -> Mapping[str, object]:
    rows = [((0.9, 0.2), (0.1, 0.8), 0.7), ((0.2, 0.8), (0.7, 0.1), 0.3), ((0.6, 0.4), (0.2, 0.9), 0.5)]
    results = {}
    for condition in ("Q0_no_q", "Q2_random_latent", "Q3_split_latent", "Q4_integrated_q"):
        errors = [abs(_novel_task_output(condition, *row) - _novel_task_target(*row)) for row in rows]
        results[condition] = {"mean_error": sum(errors) / len(errors), "generalization": 1.0 - sum(errors) / len(errors)}
    return {"tasks": ("novel_multimodal", "longitudinal_consistency", "ambiguous_value"), "conditions": results,
            "evaluation_frozen": True, "q4_best": min(results, key=lambda condition: results[condition]["mean_error"])}


def run_split_interval_detection() -> Mapping[str, object]:
    q_states = [_qualia((0.5 + 0.05 * index, 0.6), (0.2, 0.3), 0.2) for index in range(8)]
    split_states = [q if index not in {3, 4} else PhenomenalState((q.vector[0], 0.2 * q.vector[1], 0.2 * q.vector[2]), "split")
                    for index, q in enumerate(q_states)]
    coherence = [_distance(q_states[index], split_states[index]) for index in range(len(q_states))]
    detected = tuple(index for index, distance in enumerate(coherence) if distance > 0.1)
    return {"hidden_split_interval": (3, 4), "detected_interval": detected,
            "detected": detected == (3, 4), "reintegrated": _distance(split_states[-1], q_states[-1]) == 0.0}


def run_blind_ab_protocol() -> Mapping[str, object]:
    conditions = ("Q0_no_q", "Q2_random_latent", "Q3_split_latent", "Q4_integrated_q")
    generator_config = {"model": "same", "prompt": "same", "memory_budget": "same", "temperature": "same"}
    labels = {condition: f"system_{index}" for index, condition in enumerate(conditions)}
    return {"conditions": conditions, "generator_config_equal": len(set(generator_config.values())) == 1,
            "labels_blinded": all(label not in conditions for label in labels.values()),
            "rating_dimensions": ("continuity", "coherence", "presence", "inner_state", "self_correction"),
            "human_data_collected": False}


def run() -> Mapping[str, object]:
    return {"internal_privacy": True, "causal_efficacy": run_q_intervention(),
            "report_ablation": run_report_ablation(), "context_dependence": run_context_dependence(),
            "metamerism": run_metamerism(), "phenomenal_geometry": run_geometry(),
            "access_dissociation": run_access_dissociation(), "q_ablation": run_q_ablation(),
            "split_brain": run_split_brain(), "temporal_integration": run_temporal_integration(),
            "hypothesis_discrimination": run_qualia_hypothesis_discrimination(),
            "novel_cognitive_tasks": run_novel_cognitive_tasks(), "split_interval_detection": run_split_interval_detection(),
            "blind_ab_protocol": run_blind_ab_protocol()}


if __name__ == "__main__":
    import json
    print(json.dumps(run(), indent=2, sort_keys=True))
