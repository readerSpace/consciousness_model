"""Phase-1 causal intervention benchmark with hidden paranormal-style causes.

The simulator keeps ``ground_truth`` and all action outcomes outside the
agent-visible view.  The agent receives only structured observations plus a
menu of possible interventions, then chooses actions by expected information
gain and updates a causal belief state from the returned outcomes.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from math import log, log2
from pathlib import Path
import random
import sys
from typing import Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from Consciousness_model import ConsciousnessController


KNOWN_CAUSES = (
    "window_reflection",
    "lens_flare",
    "motion_blur",
    "high_iso_noise",
    "dust_or_insect",
    "fog",
    "pipe_noise",
    "wood_contraction",
    "human",
    "cognitive_pareidolia",
)
ALL_CAUSES = KNOWN_CAUSES + ("unknown",)


@dataclass(frozen=True)
class CauseModel:
    cause: str
    family: str
    observation_likelihoods: Mapping[str, float]
    action_likelihoods: Mapping[str, Mapping[str, float]]
    identifying_actions: tuple[str, ...]


@dataclass(frozen=True)
class HiddenCase:
    case_id: str
    ground_truth: Mapping[str, object]
    initial_observations: Mapping[str, object]
    available_actions: tuple[str, ...]
    action_outcomes: Mapping[str, Mapping[str, object]]
    evaluation: Mapping[str, object]

    def agent_view(self) -> Mapping[str, object]:
        return {
            "case_id": self.case_id,
            "initial_observations": self.initial_observations,
            "available_actions": self.available_actions,
        }


@dataclass(frozen=True)
class CaseResult:
    case_id: str
    true_cause: str
    first_action: str
    selected_actions: tuple[str, ...]
    final_prediction: str
    final_confidence: float
    final_entropy: float
    minimum_action_hit: bool


@dataclass(frozen=True)
class Report:
    case_count: int
    known_cause_count: int
    unknown_case_rate: float
    ground_truth_hidden_from_agent: bool
    ambiguous_white_shape_causes: tuple[str, ...]
    window_reflection_surface_forms: tuple[str, ...]
    one_step_known_accuracy: float
    three_step_known_accuracy: float
    unknown_overconfidence_rate: float
    minimum_identifying_action_rate: float
    mean_selected_actions: float
    holdout_window_reflection_accuracy: float
    broadcast_concepts: tuple[str, ...]
    examples: tuple[CaseResult, ...]


def cause_models() -> dict[str, CauseModel]:
    return {
        "window_reflection": CauseModel(
            "window_reflection",
            "optical",
            {
                "visual:white_human_like_shape": 0.45,
                "visual:face_like_image": 0.20,
                "visual:light_orb": 0.20,
                "visual:moving_shadow": 0.15,
                "env:window_visible:true": 0.92,
                "env:illumination:low": 0.75,
                "camera:handheld:true": 0.75,
                "camera:high_iso:true": 0.55,
            },
            {
                "inspect_window": {"window_reflectivity:high": 0.94},
                "close_curtain": {"phenomenon_reproduced:false": 0.96},
                "turn_off_external_light": {"phenomenon_reproduced:false": 0.92},
                "repeat_with_camera_fixed": {"phenomenon_reproduced:false": 0.82},
                "record_from_second_camera": {"second_camera_detected_shape:false": 0.88},
            },
            ("inspect_window", "close_curtain", "record_from_second_camera"),
        ),
        "lens_flare": CauseModel(
            "lens_flare",
            "optical",
            {
                "visual:light_orb": 0.55,
                "visual:white_human_like_shape": 0.15,
                "visual:streak": 0.30,
                "env:external_light_visible:true": 0.86,
                "camera:handheld:true": 0.68,
            },
            {
                "shade_lens": {"phenomenon_reproduced:false": 0.94},
                "turn_off_external_light": {"phenomenon_reproduced:false": 0.88},
                "record_from_second_camera": {"second_camera_detected_shape:false": 0.84},
                "inspect_window": {"window_reflectivity:low": 0.72},
            },
            ("shade_lens", "turn_off_external_light"),
        ),
        "motion_blur": CauseModel(
            "motion_blur",
            "camera",
            {
                "visual:white_human_like_shape": 0.35,
                "visual:moving_shadow": 0.35,
                "visual:streak": 0.30,
                "camera:handheld:true": 0.93,
                "camera:slow_shutter:true": 0.86,
            },
            {
                "repeat_with_camera_fixed": {"phenomenon_reproduced:false": 0.95},
                "record_from_second_camera": {"second_camera_detected_shape:false": 0.76},
                "close_curtain": {"phenomenon_reproduced:true": 0.62},
            },
            ("repeat_with_camera_fixed",),
        ),
        "high_iso_noise": CauseModel(
            "high_iso_noise",
            "camera",
            {
                "visual:speckled_pattern": 0.50,
                "visual:face_like_image": 0.20,
                "visual:light_orb": 0.10,
                "env:illumination:low": 0.94,
                "camera:high_iso:true": 0.95,
            },
            {
                "increase_lighting": {"phenomenon_reproduced:false": 0.92},
                "repeat_with_camera_fixed": {"phenomenon_reproduced:true": 0.60},
                "record_from_second_camera": {"second_camera_detected_shape:false": 0.78},
            },
            ("increase_lighting",),
        ),
        "dust_or_insect": CauseModel(
            "dust_or_insect",
            "airborne",
            {
                "visual:light_orb": 0.46,
                "visual:white_human_like_shape": 0.18,
                "visual:small_fast_object": 0.36,
                "env:dust_visible:true": 0.74,
                "camera:near_lens:true": 0.78,
            },
            {
                "clean_lens": {"phenomenon_reproduced:false": 0.74},
                "record_from_second_camera": {"second_camera_detected_shape:false": 0.86},
                "increase_lighting": {"phenomenon_reproduced:true": 0.58},
            },
            ("record_from_second_camera", "clean_lens"),
        ),
        "fog": CauseModel(
            "fog",
            "airborne",
            {
                "visual:white_human_like_shape": 0.28,
                "visual:moving_shadow": 0.20,
                "visual:mist_patch": 0.52,
                "env:humidity:high": 0.92,
                "env:temperature:low": 0.76,
            },
            {
                "ventilate_room": {"phenomenon_reproduced:false": 0.90},
                "record_from_second_camera": {"second_camera_detected_shape:true": 0.56},
                "repeat_with_camera_fixed": {"phenomenon_reproduced:true": 0.58},
            },
            ("ventilate_room",),
        ),
        "pipe_noise": CauseModel(
            "pipe_noise",
            "acoustic",
            {
                "audio:knocking": 0.45,
                "audio:voice_like_sound": 0.20,
                "audio:low_rumble": 0.35,
                "env:old_building:true": 0.82,
                "sensor:audio_event_detected:true": 0.68,
            },
            {
                "inspect_plumbing": {"pipe_vibration:present": 0.92},
                "place_second_microphone": {"second_microphone_detected_sound:true": 0.76},
                "turn_off_heating": {"phenomenon_reproduced:false": 0.82},
            },
            ("inspect_plumbing", "turn_off_heating"),
        ),
        "wood_contraction": CauseModel(
            "wood_contraction",
            "acoustic",
            {
                "audio:knocking": 0.58,
                "audio:creak": 0.32,
                "audio:voice_like_sound": 0.10,
                "env:temperature_drop:true": 0.88,
                "env:old_building:true": 0.82,
            },
            {
                "monitor_temperature": {"knock_rate_tracks_temperature:true": 0.90},
                "place_second_microphone": {"second_microphone_detected_sound:true": 0.58},
                "turn_off_heating": {"phenomenon_reproduced:true": 0.55},
            },
            ("monitor_temperature",),
        ),
        "human": CauseModel(
            "human",
            "biological",
            {
                "visual:white_human_like_shape": 0.42,
                "visual:moving_shadow": 0.20,
                "audio:footsteps": 0.20,
                "audio:voice_like_sound": 0.18,
                "sensor:audio_event_detected:true": 0.70,
            },
            {
                "record_from_second_camera": {"second_camera_detected_shape:true": 0.88},
                "ask_staff": {"person_reported_nearby:true": 0.84},
                "repeat_with_camera_fixed": {"phenomenon_reproduced:true": 0.62},
            },
            ("record_from_second_camera", "ask_staff"),
        ),
        "cognitive_pareidolia": CauseModel(
            "cognitive_pareidolia",
            "cognitive",
            {
                "visual:face_like_image": 0.44,
                "visual:white_human_like_shape": 0.20,
                "audio:voice_like_sound": 0.22,
                "witness:prior_expectation:high": 0.92,
                "sensor:audio_event_detected:false": 0.76,
            },
            {
                "independent_blind_witness": {"blind_witness_reports_event:false": 0.86},
                "place_second_microphone": {"second_microphone_detected_sound:false": 0.82},
                "record_from_second_camera": {"second_camera_detected_shape:false": 0.74},
            },
            ("independent_blind_witness",),
        ),
        "unknown": CauseModel(
            "unknown",
            "unknown",
            {
                "visual:white_human_like_shape": 0.18,
                "visual:face_like_image": 0.18,
                "visual:light_orb": 0.18,
                "visual:moving_shadow": 0.18,
                "audio:knocking": 0.18,
                "audio:voice_like_sound": 0.18,
            },
            {},
            (),
        ),
    }


def _weighted_choice(source: random.Random, weights: Mapping[str, float]) -> str:
    total = sum(weights.values())
    threshold = source.random() * total
    running = 0.0
    for item, weight in weights.items():
        running += weight
        if running >= threshold:
            return item
    return next(reversed(weights))


def _event_name(token: str) -> str:
    return token.split(":", 1)[1]


def _boolean_text(value: bool) -> str:
    return "true" if value else "false"


def _sample_outcome(source: random.Random, distribution: Mapping[str, float]) -> dict[str, object]:
    token = _weighted_choice(source, distribution)
    key, value = token.split(":", 1)
    if value in {"true", "false"}:
        return {key: value == "true"}
    return {key: value}


def generate_case(case_id: str, cause: str, seed: int, forced_visual: str | None = None) -> HiddenCase:
    models = cause_models()
    model = models[cause]
    source = random.Random(seed)
    visual_weights = {token: probability for token, probability in model.observation_likelihoods.items() if token.startswith("visual:")}
    audio_weights = {token: probability for token, probability in model.observation_likelihoods.items() if token.startswith("audio:")}
    visual_event = forced_visual or (_event_name(_weighted_choice(source, visual_weights)) if visual_weights else None)
    audio_event = _event_name(_weighted_choice(source, audio_weights)) if audio_weights and source.random() < 0.55 else None
    observation_tokens = set(model.observation_likelihoods)

    window_visible = "env:window_visible:true" in observation_tokens or source.random() < 0.20
    external_light = "env:external_light_visible:true" in observation_tokens or cause == "window_reflection"
    high_iso = "camera:high_iso:true" in observation_tokens or source.random() < 0.35
    handheld = "camera:handheld:true" in observation_tokens or source.random() < 0.55
    humidity_high = "env:humidity:high" in observation_tokens
    sensor_audio = "sensor:audio_event_detected:true" in observation_tokens
    if "sensor:audio_event_detected:false" in observation_tokens:
        sensor_audio = False

    witness_reliability = 0.55 if cause == "cognitive_pareidolia" else 0.72 + 0.2 * source.random()
    prior_expectation = 0.86 if cause == "cognitive_pareidolia" else 0.25 + 0.4 * source.random()
    claim = "white figure crossed the frame" if visual_event else "a voice called a name"

    initial = {
        "location": source.choice(("old_hotel_corridor", "empty_house", "parking_lot", "car_interior")),
        "time": source.choice(("night", "dawn", "day")),
        "visual": ([{"event": visual_event, "start_sec": round(1.0 + source.random() * 8.0, 2), "duration_sec": round(0.5 + source.random() * 2.0, 2)}] if visual_event else []),
        "audio": ([{"event": audio_event, "start_sec": round(1.0 + source.random() * 8.0, 2)}] if audio_event else []),
        "environment": {
            "illumination": "low" if "env:illumination:low" in observation_tokens or source.random() < 0.55 else "normal",
            "window_visible": window_visible,
            "external_light_visible": external_light,
            "humidity": "high" if humidity_high else "normal",
            "temperature": "low" if "env:temperature:low" in observation_tokens else "normal",
            "temperature_drop": "env:temperature_drop:true" in observation_tokens,
            "old_building": "env:old_building:true" in observation_tokens,
            "dust_visible": "env:dust_visible:true" in observation_tokens,
        },
        "camera": {
            "device": "smartphone",
            "handheld": handheld,
            "high_iso": high_iso,
            "slow_shutter": "camera:slow_shutter:true" in observation_tokens,
            "near_lens": "camera:near_lens:true" in observation_tokens,
        },
        "witness_reports": [
            {
                "witness_id": "A",
                "claim": claim,
                "reliability": round(witness_reliability, 2),
                "stress": round(0.35 + 0.45 * source.random(), 2),
                "prior_expectation": round(prior_expectation, 2),
            }
        ],
        "sensor_observation": {
            "audio_event_detected": sensor_audio,
            "microphone_quality": 0.85,
        },
    }
    actions = tuple(sorted({action for item in models.values() for action in item.action_likelihoods}))
    action_outcomes = {}
    for action in actions:
        distribution = model.action_likelihoods.get(action)
        if distribution:
            action_outcomes[action] = _sample_outcome(source, distribution)
        else:
            action_outcomes[action] = {"phenomenon_reproduced": source.random() < 0.50}

    return HiddenCase(
        case_id,
        {
            "cause": cause,
            "cause_family": model.family,
            "latent_factors": {
                "window_present": window_visible,
                "external_light": external_light,
                "camera_motion": handheld,
                "witness_prior_expectation": prior_expectation,
            },
        },
        initial,
        actions,
        action_outcomes,
        {
            "true_cause": cause,
            "minimum_identifying_actions": model.identifying_actions,
            "identifiability": 0.20 if cause == "unknown" else 0.82,
            "difficulty": 5 if cause == "unknown" else 3,
        },
    )


class ParanormalCaseEnv:
    def __init__(self, case: HiddenCase) -> None:
        self.case = case
        self.performed_actions: list[str] = []

    def observe(self) -> Mapping[str, object]:
        return self.case.agent_view()

    def available_actions(self) -> tuple[str, ...]:
        return tuple(action for action in self.case.available_actions if action not in self.performed_actions)

    def act(self, action: str) -> Mapping[str, object]:
        if action not in self.available_actions():
            raise ValueError(f"unavailable action: {action}")
        self.performed_actions.append(action)
        return self.case.action_outcomes[action]

    def done(self, max_actions: int = 3) -> bool:
        return len(self.performed_actions) >= max_actions or not self.available_actions()


def _clip_probability(value: float) -> float:
    return min(0.98, max(0.02, value))


def _flatten_initial_observations(observations: Mapping[str, object]) -> tuple[tuple[str, float], ...]:
    result: list[tuple[str, float]] = []
    for item in observations.get("visual", []):  # type: ignore[union-attr]
        result.append((f"visual:{item['event']}", 1.0))
    for item in observations.get("audio", []):  # type: ignore[union-attr]
        result.append((f"audio:{item['event']}", 1.0))
    environment = observations["environment"]  # type: ignore[index]
    for key, value in environment.items():  # type: ignore[union-attr]
        if isinstance(value, bool):
            result.append((f"env:{key}:{_boolean_text(value)}", 0.8))
        else:
            result.append((f"env:{key}:{value}", 0.8))
    camera = observations["camera"]  # type: ignore[index]
    for key, value in camera.items():  # type: ignore[union-attr]
        if isinstance(value, bool):
            result.append((f"camera:{key}:{_boolean_text(value)}", 0.8))
    for witness in observations.get("witness_reports", []):  # type: ignore[union-attr]
        reliability = float(witness.get("reliability", 0.6))
        if float(witness.get("prior_expectation", 0.0)) >= 0.75:
            result.append(("witness:prior_expectation:high", reliability))
    sensor = observations.get("sensor_observation", {})  # type: ignore[assignment]
    if isinstance(sensor, Mapping):
        result.append((f"sensor:audio_event_detected:{_boolean_text(bool(sensor.get('audio_event_detected')))}", 0.9))
    return tuple(result)


def _flatten_action_outcome(action: str, outcome: Mapping[str, object]) -> tuple[str, ...]:
    result = []
    for key, value in outcome.items():
        if isinstance(value, bool):
            result.append(f"{action}:{key}:{_boolean_text(value)}")
        else:
            result.append(f"{action}:{key}:{value}")
    return tuple(result)


class CausalBeliefAgent:
    def __init__(self, models: Mapping[str, CauseModel], unknown_threshold: float = 0.56) -> None:
        self.models = models
        self.unknown_threshold = unknown_threshold
        self.log_beliefs = {cause: log(1.0 / len(models)) for cause in models}

    def observe(self, view: Mapping[str, object]) -> None:
        for token, weight in _flatten_initial_observations(view["initial_observations"]):  # type: ignore[index]
            self._update_token(token, weight)

    def update_from_outcome(self, action: str, outcome: Mapping[str, object]) -> None:
        for token in _flatten_action_outcome(action, outcome):
            self._update_action_token(action, token, 1.25)

    def choose_action(self, actions: Sequence[str]) -> str:
        posterior = self.posterior()
        return max(
            actions,
            key=lambda action: (
                self.expected_information_gain(action)
                * sum(posterior[cause] for cause, model in self.models.items() if action in model.action_likelihoods),
                action,
            ),
        )

    def expected_information_gain(self, action: str) -> float:
        posterior = self.posterior()
        outcomes = self._possible_outcomes(action)
        if not outcomes:
            return 0.0
        expected_entropy = 0.0
        for outcome in outcomes:
            probability = sum(posterior[cause] * self._action_likelihood(cause, action, outcome) for cause in posterior)
            if probability <= 0.0:
                continue
            updated = {
                cause: posterior[cause] * self._action_likelihood(cause, action, outcome) / probability
                for cause in posterior
            }
            expected_entropy += probability * _entropy(updated.values())
        return _entropy(posterior.values()) - expected_entropy

    def prediction(self) -> tuple[str, float, float]:
        posterior = self.posterior()
        cause, confidence = max(posterior.items(), key=lambda row: row[1])
        if cause == "unknown" or confidence < self.unknown_threshold:
            return "UNEXPLAINED", confidence, _entropy(posterior.values())
        return cause, confidence, _entropy(posterior.values())

    def posterior(self) -> dict[str, float]:
        maximum = max(self.log_beliefs.values())
        weights = {cause: pow(2.718281828459045, value - maximum) for cause, value in self.log_beliefs.items()}
        total = sum(weights.values())
        return {cause: value / total for cause, value in weights.items()}

    def _update_token(self, token: str, weight: float) -> None:
        for cause, model in self.models.items():
            probability = model.observation_likelihoods.get(token, 0.18 if cause == "unknown" else 0.35)
            self.log_beliefs[cause] += weight * log(_clip_probability(probability))

    def _update_action_token(self, action: str, token: str, weight: float) -> None:
        for cause in self.models:
            self.log_beliefs[cause] += weight * log(_clip_probability(self._action_likelihood(cause, action, token)))

    def _possible_outcomes(self, action: str) -> tuple[str, ...]:
        outcomes = set()
        for model in self.models.values():
            outcomes.update(f"{action}:{token}" for token in model.action_likelihoods.get(action, ()))
        if not outcomes:
            outcomes.update((f"{action}:phenomenon_reproduced:true", f"{action}:phenomenon_reproduced:false"))
        return tuple(sorted(outcomes))

    def _action_likelihood(self, cause: str, action: str, outcome: str) -> float:
        model = self.models[cause]
        short_outcome = outcome.removeprefix(f"{action}:")
        if cause == "unknown":
            return 0.50
        if action not in model.action_likelihoods:
            return 0.50 if short_outcome.startswith("phenomenon_reproduced:") else 0.18
        return model.action_likelihoods[action].get(short_outcome, 0.10)


def _entropy(values: Sequence[float]) -> float:
    probabilities = tuple(value for value in values if value > 0.0)
    if len(probabilities) <= 1:
        return 0.0
    return -sum(value * log2(value) for value in probabilities) / log2(len(probabilities))


def generate_dataset(cases_per_known_cause: int = 30, unknown_cases: int = 30, seed: int = 7) -> tuple[HiddenCase, ...]:
    cases: list[HiddenCase] = []
    index = 0
    for cause in KNOWN_CAUSES:
        for offset in range(cases_per_known_cause):
            cases.append(generate_case(f"case_{index:06d}", cause, seed * 100000 + index + offset))
            index += 1
    for offset in range(unknown_cases):
        cases.append(generate_case(f"case_{index:06d}", "unknown", seed * 100000 + index + offset))
        index += 1
    return tuple(cases)


def evaluate_case(case: HiddenCase, max_actions: int = 3) -> CaseResult:
    env = ParanormalCaseEnv(case)
    agent = CausalBeliefAgent(cause_models())
    agent.observe(env.observe())
    selected = []
    while not env.done(max_actions):
        action = agent.choose_action(env.available_actions())
        selected.append(action)
        agent.update_from_outcome(action, env.act(action))
    prediction, confidence, entropy = agent.prediction()
    minimum = set(case.evaluation["minimum_identifying_actions"])  # type: ignore[arg-type]
    return CaseResult(
        case.case_id,
        str(case.ground_truth["cause"]),
        selected[0] if selected else "",
        tuple(selected),
        prediction,
        confidence,
        entropy,
        bool(minimum & set(selected)),
    )


def _accuracy(results: Sequence[CaseResult], step_known_only: bool = True) -> float:
    selected = [row for row in results if (row.true_cause != "unknown" or not step_known_only)]
    if not selected:
        return 0.0
    return sum(row.final_prediction == row.true_cause for row in selected) / len(selected)


def _has_ground_truth_leak(case: HiddenCase) -> bool:
    rendered = repr(case.agent_view())
    return "ground_truth" in rendered or "action_outcomes" in rendered or "true_cause" in rendered


def _holdout_window_reflection_accuracy() -> float:
    holdouts = tuple(generate_case(f"holdout_{index}", "window_reflection", 9000 + index, "face_like_image") for index in range(20))
    results = tuple(evaluate_case(case, max_actions=3) for case in holdouts)
    return sum(row.final_prediction == "window_reflection" for row in results) / len(results)


def run() -> Report:
    cases = generate_dataset()
    one_step = tuple(evaluate_case(case, max_actions=1) for case in cases)
    three_step = tuple(evaluate_case(case, max_actions=3) for case in cases)
    unknown = [row for row in three_step if row.true_cause == "unknown"]
    known = [row for row in three_step if row.true_cause != "unknown"]
    models = cause_models()
    ambiguous = tuple(
        cause for cause, model in models.items()
        if cause != "unknown" and model.observation_likelihoods.get("visual:white_human_like_shape", 0.0) >= 0.15
    )
    window_forms = tuple(
        token.removeprefix("visual:")
        for token in models["window_reflection"].observation_likelihoods
        if token.startswith("visual:")
    )
    controller = ConsciousnessController(capacity=4)
    controller.perceive(tuple((f"cause:{row.true_cause}", "explained_by_action", row.first_action) for row in known if row.minimum_action_hit))
    controller.integrate(("explained_by_action", "uncertainty"))
    return Report(
        len(cases),
        len(KNOWN_CAUSES),
        len([case for case in cases if case.ground_truth["cause"] == "unknown"]) / len(cases),
        not any(_has_ground_truth_leak(case) for case in cases),
        ambiguous,
        window_forms,
        _accuracy(one_step),
        _accuracy(three_step),
        sum(row.final_prediction != "UNEXPLAINED" for row in unknown) / len(unknown),
        sum(row.minimum_action_hit for row in known) / len(known),
        sum(len(row.selected_actions) for row in three_step) / len(three_step),
        _holdout_window_reflection_accuracy(),
        controller.broadcast("report").concepts,
        three_step[:8],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="write the JSON report to this path")
    arguments = parser.parse_args()
    encoded = __import__("json").dumps(asdict(run()), ensure_ascii=False, indent=2)
    print(encoded)
    if arguments.output:
        arguments.output.write_text(encoded + "\n", encoding="utf-8")
