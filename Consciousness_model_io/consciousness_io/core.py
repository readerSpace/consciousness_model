"""Reusable, dependency-free physical analysis primitives."""

from dataclasses import dataclass, field
import math
import random
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple


Vector = Tuple[float, ...]
Rhs = Callable[[float, Vector], Vector]


@dataclass(frozen=True)
class Observation:
    time: float
    features: Dict[str, float]
    metadata: Dict[str, object] = field(default_factory=dict)
    uncertainty: Dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class Law:
    expression: str
    parameters: Dict[str, float]
    domain: Dict[str, float]
    error: float
    validation_r2: float
    extrapolation_score: float
    mdl_gain: float
    support: int


@dataclass(frozen=True)
class RegimeLaw:
    boundary: float
    left: Law
    right: Law
    mdl_gain: float


@dataclass(frozen=True)
class DiscoveryConfig:
    discovery_fraction: float = 0.6
    validation_fraction: float = 0.2
    min_validation_r2: float = 0.8
    min_mdl_gain: float = 5.0
    min_segment_size: int = 24
    regime_complexity_bits: float = 48.0


@dataclass(frozen=True)
class PhysicalObservation:
    time: float
    state: Vector
    noisy_state: Vector
    uncertainty: Vector = ()


def _r2(observed: Sequence[float], predicted: Sequence[float]) -> float:
    mean = sum(observed) / len(observed)
    total = sum((value - mean) ** 2 for value in observed)
    if total <= 1e-15:
        return 1.0 if all(abs(a - b) <= 1e-12 for a, b in zip(observed, predicted)) else 0.0
    return 1.0 - sum((a - b) ** 2 for a, b in zip(observed, predicted)) / total


def _residual_bits(residuals: Sequence[float], parameter_count: int) -> float:
    if not residuals:
        return float("inf")
    scale = max(max(abs(value) for value in residuals), 1e-12)
    return len(residuals) * math.log2(scale + 1.0) + parameter_count * 8.0


def _fit_power(observations: Sequence[Observation]) -> Tuple[float, float]:
    log_x = [math.log(row.features["x1"]) for row in observations]
    log_y = [math.log(row.features["x2"]) for row in observations]
    x_mean = sum(log_x) / len(log_x)
    y_mean = sum(log_y) / len(log_y)
    denominator = sum((value - x_mean) ** 2 for value in log_x)
    if denominator <= 1e-15:
        raise ValueError("x1 must contain more than one distinct value")
    exponent = sum((x - x_mean) * (y - y_mean) for x, y in zip(log_x, log_y)) / denominator
    return math.exp(y_mean - exponent * x_mean), exponent


def _positive_sorted(observations: Sequence[Observation]) -> List[Observation]:
    valid = [row for row in observations if row.features.get("x1", 0.0) > 0 and row.features.get("x2", 0.0) > 0]
    return sorted(valid, key=lambda row: row.features["x1"])


def _split(observations: Sequence[Observation], config: DiscoveryConfig):
    ordered = _positive_sorted(observations)
    if len(ordered) < 10:
        raise ValueError("At least ten positive observations are required")
    discovery_end = max(2, int(len(ordered) * config.discovery_fraction))
    validation_end = max(discovery_end + 2, int(len(ordered) * (config.discovery_fraction + config.validation_fraction)))
    validation_end = min(validation_end, len(ordered) - 2)
    return ordered[:discovery_end], ordered[discovery_end:validation_end], ordered[validation_end:]


def _make_law(fit_data, validation_data, extrapolation_data, domain) -> Law:
    coefficient, exponent = _fit_power(fit_data)
    predict = lambda rows: [math.log(coefficient) + exponent * math.log(row.features["x1"]) for row in rows]
    observed_fit = [math.log(row.features["x2"]) for row in fit_data]
    residuals = [actual - predicted for actual, predicted in zip(observed_fit, predict(fit_data))]
    gain = len(fit_data) * math.log2(max(max(abs(value) for value in observed_fit), 1.0) + 1.0) - _residual_bits(residuals, 2)
    return Law(
        expression=f'x2 = {coefficient:.12g} * x1^{exponent:.12g}',
        parameters={"coefficient": coefficient, "exponent": exponent},
        domain={"x1_min": min(row.features["x1"] for row in domain), "x1_max": max(row.features["x1"] for row in domain)},
        error=math.sqrt(sum(value * value for value in residuals) / len(residuals)),
        validation_r2=_r2([math.log(row.features["x2"]) for row in validation_data], predict(validation_data)),
        extrapolation_score=_r2([math.log(row.features["x2"]) for row in extrapolation_data], predict(extrapolation_data)),
        mdl_gain=gain,
        support=len(fit_data),
    )


def discover_power_law(observations: Sequence[Observation], config: DiscoveryConfig = DiscoveryConfig()) -> Optional[Law]:
    discovery, validation, extrapolation = _split(observations, config)
    law = _make_law(discovery, validation, extrapolation, observations)
    if law.mdl_gain < config.min_mdl_gain or law.validation_r2 < config.min_validation_r2:
        return None
    return law


def discover_regime(observations: Sequence[Observation], config: DiscoveryConfig = DiscoveryConfig()) -> Optional[RegimeLaw]:
    ordered = _positive_sorted(observations)
    if len(ordered) < config.min_segment_size * 2:
        return None
    whole = _make_law(ordered, ordered, ordered, ordered)
    best = None
    for index in range(config.min_segment_size, len(ordered) - config.min_segment_size + 1, 2):
        left, right = ordered[:index], ordered[index:]
        left_law, right_law = _make_law(left, left, left, left), _make_law(right, right, right, right)
        gain = whole.error - left_law.error - right_law.error - config.regime_complexity_bits
        if best is None or gain > best[0]:
            best = (gain, left, right, left_law, right_law)
    if best is None or best[0] < config.min_mdl_gain:
        return None
    gain, left, right, left_law, right_law = best
    return RegimeLaw(math.sqrt(left[-1].features["x1"] * right[0].features["x1"]), left_law, right_law, gain)


def _add(left: Vector, right: Vector) -> Vector:
    return tuple(a + b for a, b in zip(left, right))


def _scale(values: Vector, factor: float) -> Vector:
    return tuple(value * factor for value in values)


def integrate_rk4(rhs: Rhs, initial_state: Sequence[float], times: Sequence[float]) -> Tuple[Vector, ...]:
    if not times:
        return ()
    if any(later <= earlier for earlier, later in zip(times, times[1:])):
        raise ValueError("times must be strictly increasing")
    state = tuple(float(value) for value in initial_state)
    trajectory = [state]
    for start, end in zip(times, times[1:]):
        step = end - start
        k1 = rhs(start, state)
        k2 = rhs(start + step / 2.0, _add(state, _scale(k1, step / 2.0)))
        k3 = rhs(start + step / 2.0, _add(state, _scale(k2, step / 2.0)))
        k4 = rhs(end, _add(state, _scale(k3, step)))
        state = _add(state, _scale(_add(_add(k1, _scale(k2, 2.0)), _add(_scale(k3, 2.0), k4)), step / 6.0))
        trajectory.append(state)
    return tuple(trajectory)


def generate_noisy_observations(rhs: Rhs, initial_state: Sequence[float], times: Sequence[float], noise_sigma: float = 0.0, seed: int = 0) -> Tuple[PhysicalObservation, ...]:
    if noise_sigma < 0:
        raise ValueError("noise_sigma must be non-negative")
    clean = integrate_rk4(rhs, initial_state, times)
    source = random.Random(seed)
    return tuple(PhysicalObservation(float(time), state, tuple(value + source.gauss(0.0, noise_sigma) for value in state), tuple(noise_sigma for _ in state)) for time, state in zip(times, clean))


@dataclass(frozen=True)
class SurrogatePrediction:
    value: float
    uncertainty: float
    nearest_distance: float
    exact_fallback: bool


class KNNModel:
    def __init__(self, max_distance: float = math.inf, neighbors: int = 3) -> None:
        if max_distance < 0 or neighbors < 1:
            raise ValueError("max_distance must be non-negative and neighbors must be positive")
        self.max_distance, self.neighbors = max_distance, neighbors
        self._samples: Tuple[Tuple[Vector, float], ...] = ()

    def fit(self, features: Iterable[Sequence[float]], values: Iterable[float]) -> "KNNModel":
        samples = tuple((tuple(map(float, feature)), float(value)) for feature, value in zip(features, values))
        if not samples or any(len(feature) != len(samples[0][0]) for feature, _ in samples):
            raise ValueError("features must be non-empty and have consistent dimensions")
        self._samples = samples
        return self

    def predict(self, feature: Sequence[float], exact: Optional[Callable[[], float]] = None) -> SurrogatePrediction:
        if not self._samples:
            raise ValueError("fit must be called before predict")
        point = tuple(map(float, feature))
        if len(point) != len(self._samples[0][0]):
            raise ValueError("feature dimension does not match training data")
        ranked = sorted(
            (
                (math.sqrt(sum((a - b) ** 2 for a, b in zip(point, sample))), value)
                for sample, value in self._samples
            ),
            key=lambda item: item[0],
        )
        selected = ranked[: self.neighbors]
        nearest = ranked[0][0]
        prediction = sum(value for _, value in selected) / len(selected)
        fallback = nearest > self.max_distance
        if fallback and exact is not None:
            prediction = float(exact())
        return SurrogatePrediction(prediction, max(distance for distance, _ in selected), nearest, fallback)


def evaluate_ood(predict: Callable[[Sequence[float]], float], samples: Iterable[Tuple[Sequence[float], float]]) -> float:
    rows = tuple(samples)
    if not rows:
        raise ValueError("samples must be non-empty")
    errors = [float(predict(features)) - float(target) for features, target in rows]
    scale = max(abs(float(target)) for _, target in rows)
    return math.sqrt(sum(error * error for error in errors) / len(errors)) / max(scale, 1e-15)
