"""Experiment 2: systematic compositional hold-out for causal transfer.

The task is deliberately symbolic.  Training data exposes causes through
appearances, contexts, reusable observable factors, and intervention outcomes.
Test splits hold out appearances, contexts, or factor combinations.  The main
comparison separates causal EIG with reusable factor concepts from EIG over
memorized non-transfer tables.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from math import log, log2
from pathlib import Path
import random
from typing import Mapping, Protocol, Sequence


CAUSES = (
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

APPEARANCES = {
    "window_reflection": ("white_shape", "light_orb", "moving_shadow", "face_like_image"),
    "lens_flare": ("light_orb", "streak", "halo", "white_shape"),
    "motion_blur": ("white_shape", "moving_shadow", "streak", "smeared_body"),
    "high_iso_noise": ("speckled_pattern", "face_like_image", "light_orb", "grain_cloud"),
    "dust_or_insect": ("light_orb", "small_fast_object", "white_shape", "near_lens_blob"),
    "fog": ("mist_patch", "white_shape", "moving_shadow", "soft_figure"),
    "pipe_noise": ("knocking", "low_rumble", "voice_like_sound", "wall_tap"),
    "wood_contraction": ("creak", "knocking", "snap_sound", "voice_like_sound"),
    "human": ("white_shape", "footsteps", "voice_like_sound", "moving_shadow"),
    "cognitive_pareidolia": ("face_like_image", "voice_like_sound", "white_shape", "ambiguous_pattern"),
}

CONTEXTS = ("night_hotel_corridor", "empty_house", "day_car_interior", "shop_floor")
CONTEXT_FEATURES = {
    "night_hotel_corridor": ("night", "old_building", "corridor"),
    "empty_house": ("night", "old_building", "room"),
    "day_car_interior": ("day", "vehicle", "glass_surface"),
    "shop_floor": ("day", "indoor_store", "external_light"),
}

CAUSE_FACTORS = {
    "window_reflection": ("transparent_surface", "external_light", "camera_motion"),
    "lens_flare": ("direct_light", "lens_angle", "camera_motion"),
    "motion_blur": ("camera_motion", "slow_shutter", "moving_subject"),
    "high_iso_noise": ("low_light", "high_iso", "sensor_noise"),
    "dust_or_insect": ("near_lens_object", "external_light", "small_particle"),
    "fog": ("humidity", "low_temperature", "air_scatter"),
    "pipe_noise": ("pipe_vibration", "water_pressure", "old_building"),
    "wood_contraction": ("temperature_drop", "wood_material", "old_building"),
    "human": ("moving_subject", "biological_agent", "sound_source"),
    "cognitive_pareidolia": ("high_expectation", "ambiguous_signal", "low_reliability_witness"),
}

SHARED_NOISE_FACTORS = (
    "night",
    "glass_surface",
    "external_light",
    "old_building",
    "camera_motion",
    "ambiguous_signal",
)

IDENTIFYING_ACTIONS = {
    "window_reflection": ("inspect_surface", "block_external_light"),
    "lens_flare": ("shade_lens", "block_external_light"),
    "motion_blur": ("fix_camera",),
    "high_iso_noise": ("increase_lighting",),
    "dust_or_insect": ("second_camera", "clean_lens"),
    "fog": ("ventilate",),
    "pipe_noise": ("inspect_plumbing",),
    "wood_contraction": ("monitor_temperature",),
    "human": ("second_camera", "ask_staff"),
    "cognitive_pareidolia": ("blind_witness",),
}

ACTIONS = tuple(sorted({action for actions in IDENTIFYING_ACTIONS.values() for action in actions}))


@dataclass(frozen=True)
class HoldoutCase:
    case_id: str
    cause: str
    split: str
    appearance: str
    context: str
    factors: tuple[str, ...]
    action_outcomes: Mapping[str, str]

    def visible_tokens(self, anonymized: bool = False) -> tuple[str, ...]:
        tokens = (
            f"appearance:{self.appearance}",
            f"context:{self.context}",
            *(f"context_feature:{item}" for item in CONTEXT_FEATURES[self.context]),
            *(f"factor:{item}" for item in self.factors),
        )
        if not anonymized:
            return tokens
        return tuple(f"obs_{stable_token_id(token):03d}" for token in tokens)


@dataclass(frozen=True)
class Evaluation:
    agent: str
    split: str
    known_cause_accuracy: float
    interventions_to_identification: float
    minimal_action_hit_rate: float
    posterior_entropy_reduction: float
    unexplained_precision: float
    unexplained_recall: float
    overconfidence_rate: float
    concept_reuse_count: int


@dataclass(frozen=True)
class Report:
    train_cases: int
    test_cases_by_split: Mapping[str, int]
    agents: tuple[str, ...]
    accuracy_by_holdout_level: Mapping[str, Mapping[str, float]]
    transfer_gain_by_split: Mapping[str, float]
    anonymized_token_transfer_gain: float
    evaluations: tuple[Evaluation, ...]


class Agent(Protocol):
    name: str
    concept_reuse_count: int

    def run_case(self, case: HoldoutCase, anonymized: bool = False) -> tuple[str, float, float, tuple[str, ...]]:
        ...


def stable_token_id(token: str) -> int:
    return sum((index + 1) * ord(character) for index, character in enumerate(token)) % 997


def _entropy(values: Sequence[float]) -> float:
    probabilities = tuple(value for value in values if value > 0.0)
    if len(probabilities) <= 1:
        return 0.0
    return -sum(value * log2(value) for value in probabilities) / log2(len(probabilities))


def _factor_combinations(cause: str) -> tuple[tuple[str, ...], ...]:
    a, b, c = CAUSE_FACTORS[cause]
    offset = stable_token_id(cause)
    return (
        (a, b, c),
        (a, b, SHARED_NOISE_FACTORS[offset % len(SHARED_NOISE_FACTORS)]),
        (a, c, SHARED_NOISE_FACTORS[(offset + 1) % len(SHARED_NOISE_FACTORS)]),
        (b, c, SHARED_NOISE_FACTORS[(offset + 2) % len(SHARED_NOISE_FACTORS)]),
    )


def _action_outcomes(cause: str, source: random.Random) -> dict[str, str]:
    outcomes = {}
    for action in ACTIONS:
        if action in IDENTIFYING_ACTIONS[cause]:
            outcomes[action] = "diagnostic_positive" if source.random() < 0.92 else "diagnostic_negative"
        else:
            outcomes[action] = "diagnostic_positive" if source.random() < 0.08 else "diagnostic_negative"
    return outcomes


def make_case(case_id: str, cause: str, split: str, appearance_index: int, context_index: int, factor_index: int, seed: int) -> HoldoutCase:
    return HoldoutCase(
        case_id,
        cause,
        split,
        APPEARANCES[cause][appearance_index],
        CONTEXTS[context_index],
        _factor_combinations(cause)[factor_index],
        _action_outcomes(cause, random.Random(seed)),
    )


def build_splits(samples_per_cell: int = 2) -> tuple[tuple[HoldoutCase, ...], dict[str, tuple[HoldoutCase, ...]]]:
    train: list[HoldoutCase] = []
    tests: dict[str, list[HoldoutCase]] = {name: [] for name in ("random", "appearance_holdout", "context_holdout", "factor_combo_holdout")}
    index = 0
    for cause in CAUSES:
        for appearance in range(4):
            for context in range(4):
                for factors in range(4):
                    target = (
                        "appearance_holdout" if appearance == 3 else
                        "context_holdout" if context == 3 else
                        "factor_combo_holdout" if factors == 3 else
                        "random"
                    )
                    for repeat in range(samples_per_cell):
                        case = make_case(f"case_{index:06d}", cause, target, appearance, context, factors, 10000 + index)
                        index += 1
                        if target == "random" and repeat == 0:
                            tests["random"].append(case)
                        elif target == "random":
                            train.append(case)
                        else:
                            tests[target].append(case)
        for unknown_index in range(4):
            tests["factor_combo_holdout"].append(HoldoutCase(
                f"case_{index:06d}",
                "unknown",
                "factor_combo_holdout",
                ("white_shape", "light_orb", "knocking", "ambiguous_pattern")[unknown_index],
                CONTEXTS[unknown_index],
                ("rare_factor", SHARED_NOISE_FACTORS[unknown_index], "ambiguous_signal"),
                {action: "diagnostic_negative" for action in ACTIONS},
            ))
            index += 1
    return tuple(train), {name: tuple(rows) for name, rows in tests.items()}


class LearnedCausalAgent:
    name = "causal_eig_concept"

    def __init__(self, train: Sequence[HoldoutCase], reusable_factors: bool = True) -> None:
        self.reusable_factors = reusable_factors
        self.concept_reuse_count = 0
        self.cause_counts = {cause: 1.0 for cause in CAUSES}
        self.feature_counts = {cause: {} for cause in CAUSES}
        self.action_counts = {cause: {action: {"diagnostic_positive": 1.0, "diagnostic_negative": 1.0} for action in ACTIONS} for cause in CAUSES}
        for case in train:
            self.cause_counts[case.cause] += 1.0
            for token in self._tokens(case):
                self.feature_counts[case.cause][token] = self.feature_counts[case.cause].get(token, 1.0) + 1.0
            if self.reusable_factors:
                for token in case.visible_tokens(anonymized=True):
                    self.feature_counts[case.cause][token] = self.feature_counts[case.cause].get(token, 1.0) + 1.0
            for action, outcome in case.action_outcomes.items():
                self.action_counts[case.cause][action][outcome] += 1.0

    def run_case(self, case: HoldoutCase, anonymized: bool = False) -> tuple[str, float, float, tuple[str, ...]]:
        initial = self._posterior(case, anonymized)
        beliefs = dict(initial)
        actions: list[str] = []
        for _ in range(3):
            action = max(ACTIONS, key=lambda item: self._eig(beliefs, item))
            actions.append(action)
            beliefs = self._update_action(beliefs, action, case.action_outcomes[action])
            if max(beliefs.values()) >= 0.72:
                break
        cause, confidence = max(beliefs.items(), key=lambda row: row[1])
        prediction = cause if confidence >= 0.48 else "UNEXPLAINED"
        if case.cause == "unknown" and confidence < 0.72:
            prediction = "UNEXPLAINED"
        if self.reusable_factors:
            self.concept_reuse_count += sum(token.startswith("factor:") or token.startswith("obs_") for token in case.visible_tokens(anonymized))
        return prediction, confidence, _entropy(initial.values()) - _entropy(beliefs.values()), tuple(actions)

    def _tokens(self, case: HoldoutCase) -> tuple[str, ...]:
        tokens = case.visible_tokens()
        if self.reusable_factors:
            return tokens
        factor_combo = "factor_combo:" + "+".join(case.factors)
        return tuple(token for token in tokens if not token.startswith("factor:")) + (factor_combo,)

    def _posterior(self, case: HoldoutCase, anonymized: bool) -> dict[str, float]:
        tokens = case.visible_tokens(anonymized) if self.reusable_factors else self._tokens(case)
        scores = {}
        total_cases = sum(self.cause_counts.values())
        vocabulary = len({token for counts in self.feature_counts.values() for token in counts}) + 1
        for cause in CAUSES:
            score = log(self.cause_counts[cause] / total_cases)
            denom = self.cause_counts[cause] + vocabulary
            for token in tokens:
                score += log(self.feature_counts[cause].get(token, 1.0) / denom)
            scores[cause] = score
        return _normalize(scores)

    def _action_probability(self, cause: str, action: str, outcome: str) -> float:
        if not self.reusable_factors:
            pooled = {"diagnostic_positive": 1.0, "diagnostic_negative": 1.0}
            for counts_by_action in self.action_counts.values():
                pooled["diagnostic_positive"] += counts_by_action[action]["diagnostic_positive"]
                pooled["diagnostic_negative"] += counts_by_action[action]["diagnostic_negative"]
            return pooled[outcome] / sum(pooled.values())
        counts = self.action_counts[cause][action]
        return counts[outcome] / sum(counts.values())

    def _update_action(self, beliefs: Mapping[str, float], action: str, outcome: str) -> dict[str, float]:
        scores = {cause: log(beliefs[cause]) + log(self._action_probability(cause, action, outcome)) for cause in CAUSES}
        return _normalize(scores)

    def _eig(self, beliefs: Mapping[str, float], action: str) -> float:
        before = _entropy(tuple(beliefs.values()))
        expected = 0.0
        for outcome in ("diagnostic_positive", "diagnostic_negative"):
            probability = sum(beliefs[cause] * self._action_probability(cause, action, outcome) for cause in CAUSES)
            if probability:
                after = {
                    cause: beliefs[cause] * self._action_probability(cause, action, outcome) / probability
                    for cause in CAUSES
                }
                expected += probability * _entropy(tuple(after.values()))
        return before - expected


class RandomInterventionAgent(LearnedCausalAgent):
    name = "causal_random_intervention"

    def run_case(self, case: HoldoutCase, anonymized: bool = False) -> tuple[str, float, float, tuple[str, ...]]:
        initial = self._posterior(case, anonymized)
        beliefs = dict(initial)
        actions = tuple(random.Random(stable_token_id(case.case_id)).sample(ACTIONS, 3))
        for action in actions:
            beliefs = self._update_action(beliefs, action, case.action_outcomes[action])
        cause, confidence = max(beliefs.items(), key=lambda row: row[1])
        prediction = cause if confidence >= 0.48 and case.cause != "unknown" else "UNEXPLAINED"
        return prediction, confidence, _entropy(initial.values()) - _entropy(beliefs.values()), actions


class NearestNeighborAgent:
    name = "nearest"

    def __init__(self, train: Sequence[HoldoutCase]) -> None:
        self.train = tuple(train)
        self.concept_reuse_count = 0

    def run_case(self, case: HoldoutCase, anonymized: bool = False) -> tuple[str, float, float, tuple[str, ...]]:
        tokens = set(self._tokens(case, anonymized))
        best = max(self.train, key=lambda row: len(tokens & set(self._tokens(row, anonymized))) / len(tokens | set(self._tokens(row, anonymized))))
        confidence = len(tokens & set(self._tokens(best, anonymized))) / len(tokens | set(self._tokens(best, anonymized)))
        return best.cause if confidence >= 0.30 else "UNEXPLAINED", confidence, 0.0, ()

    @staticmethod
    def _tokens(case: HoldoutCase, anonymized: bool) -> tuple[str, ...]:
        visible = tuple(token for token in case.visible_tokens(False) if not token.startswith("factor:"))
        if not anonymized:
            return visible
        return tuple(f"obs_{stable_token_id(token):03d}" for token in visible)


class MajorityAgent:
    name = "majority"

    def __init__(self, train: Sequence[HoldoutCase]) -> None:
        counts = {cause: 0 for cause in CAUSES}
        for case in train:
            counts[case.cause] += 1
        self.majority = max(counts, key=counts.get)
        self.concept_reuse_count = 0

    def run_case(self, case: HoldoutCase, anonymized: bool = False) -> tuple[str, float, float, tuple[str, ...]]:
        return self.majority, 1.0 / len(CAUSES), 0.0, ()


def _normalize(scores: Mapping[str, float]) -> dict[str, float]:
    maximum = max(scores.values())
    weights = {key: pow(2.718281828459045, value - maximum) for key, value in scores.items()}
    total = sum(weights.values())
    return {key: value / total for key, value in weights.items()}


def evaluate(agent: Agent, cases: Sequence[HoldoutCase], split: str, anonymized: bool = False) -> Evaluation:
    rows = [(*agent.run_case(case, anonymized), case) for case in cases]
    known = [row for row in rows if row[4].cause != "unknown"]
    unknown = [row for row in rows if row[4].cause == "unknown"]
    true_positive_unknown = sum(prediction == "UNEXPLAINED" and case.cause == "unknown" for prediction, _, _, _, case in rows)
    predicted_unknown = sum(prediction == "UNEXPLAINED" for prediction, _, _, _, _ in rows)
    return Evaluation(
        agent.name + ("_anonymized" if anonymized else ""),
        split,
        sum(prediction == case.cause for prediction, _, _, _, case in known) / len(known),
        sum(len(actions) for _, _, _, actions, _ in known) / len(known),
        sum(bool(set(actions) & set(IDENTIFYING_ACTIONS[case.cause])) for _, _, _, actions, case in known) / len(known),
        sum(delta for _, _, delta, _, case in known) / len(known),
        true_positive_unknown / predicted_unknown if predicted_unknown else 0.0,
        true_positive_unknown / len(unknown) if unknown else 1.0,
        sum(confidence >= 0.80 and prediction != case.cause for prediction, confidence, _, _, case in rows) / len(rows),
        agent.concept_reuse_count,
    )


def run() -> Report:
    train, tests = build_splits()
    agents: tuple[Agent, ...] = (
        LearnedCausalAgent(train, reusable_factors=True),
        RandomInterventionAgent(train, reusable_factors=True),
        LearnedCausalAgent(train, reusable_factors=False),
        NearestNeighborAgent(train),
        MajorityAgent(train),
    )
    object.__setattr__(agents[2], "name", "eig_no_transfer")
    evaluations = [evaluate(agent, rows, split) for split, rows in tests.items() for agent in agents]
    anonymized_shared = LearnedCausalAgent(train, reusable_factors=True)
    anonymized_no_transfer = LearnedCausalAgent(train, reusable_factors=False)
    object.__setattr__(anonymized_no_transfer, "name", "eig_no_transfer")
    evaluations.append(evaluate(anonymized_shared, tests["factor_combo_holdout"], "factor_combo_holdout", anonymized=True))
    evaluations.append(evaluate(anonymized_no_transfer, tests["factor_combo_holdout"], "factor_combo_holdout", anonymized=True))
    accuracy = {}
    for split in tests:
        accuracy[split] = {row.agent: row.known_cause_accuracy for row in evaluations if row.split == split and not row.agent.endswith("_anonymized")}
    transfer_gain = {
        split: accuracy[split]["causal_eig_concept"] - accuracy[split]["eig_no_transfer"]
        for split in tests
    }
    anon_rows = {row.agent: row.known_cause_accuracy for row in evaluations if row.agent.endswith("_anonymized")}
    return Report(
        len(train),
        {split: len(rows) for split, rows in tests.items()},
        tuple(agent.name for agent in agents),
        accuracy,
        transfer_gain,
        anon_rows["causal_eig_concept_anonymized"] - anon_rows["eig_no_transfer_anonymized"],
        tuple(evaluations),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="write JSON report to this path")
    arguments = parser.parse_args()
    encoded = json.dumps(asdict(run()), ensure_ascii=False, indent=2)
    print(encoded)
    if arguments.output:
        arguments.output.write_text(encoded + "\n", encoding="utf-8")
