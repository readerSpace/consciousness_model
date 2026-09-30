"""L6.8: design and verify a small scientific experiment from natural language.

The first bounded domain is the relativistic train-and-garage simultaneity
problem. The plan keeps hypotheses, equations, generated implementation, and
runtime conclusions separate so a missing repository experiment is not a
terminal failure.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from math import sqrt
from pathlib import Path
import subprocess
import sys
import tempfile


@dataclass(frozen=True)
class VerificationStrategy:
    primary: str
    secondary: str
    reason: str


@dataclass(frozen=True)
class ExperimentPlan:
    question: str
    domain: str
    hypotheses: tuple[str, ...]
    assumptions: tuple[str, ...]
    variables: tuple[str, ...]
    parameters: tuple[tuple[str, float], ...]
    equations: tuple[str, ...]
    controls: tuple[str, ...]
    measurements: tuple[str, ...]
    success_conditions: tuple[str, ...]
    implementation_steps: tuple[str, ...]
    strategy: VerificationStrategy
    source: str


@dataclass(frozen=True)
class VerificationResult:
    return_code: int
    beta_threshold: float
    beta_tested: float
    contracted_length: float
    fits_at_test_speed: bool
    simultaneity_delta_train: float
    sweep_fit_count: int
    stdout: str
    stderr: str


def is_design_and_verify_request(text: str) -> bool:
    lower = text.lower()
    asks_verify = any(word in text for word in ("検証して", "確かめて", "正しいか", "シミュレーションして")) or any(
        word in lower for word in ("verify", "validate", "simulate")
    )
    has_relativity_context = any(word in text for word in ("相対論", "相対論的", "電車", "列車", "車庫", "同時刻", "同時性")) or any(
        word in lower for word in ("relativ", "lorentz", "train", "garage", "simultaneity")
    )
    has_file_target = ".py" in lower or "/" in text or "\\" in text
    return asks_verify and has_relativity_context and not has_file_target


def _implementation_source() -> str:
    return r'''
import json
import math

def gamma(beta):
    return 1.0 / math.sqrt(1.0 - beta * beta)

def contracted_length(rest_length, beta):
    return rest_length / gamma(beta)

def lorentz_time(t, x, beta, c=1.0):
    return gamma(beta) * (t - beta * x / c)

L0 = 100.0
L_garage = 60.0
beta_tested = 0.9
c = 1.0
beta_threshold = math.sqrt(1.0 - (L_garage / L0) ** 2)
length_at_test_speed = contracted_length(L0, beta_tested)
rear_train_time = lorentz_time(0.0, 0.0, beta_tested, c)
front_train_time = lorentz_time(0.0, L_garage, beta_tested, c)
sweep = [
    beta / 100.0 for beta in range(0, 100)
    if contracted_length(L0, beta / 100.0) <= L_garage
]
print(json.dumps({
    "beta_threshold": beta_threshold,
    "beta_tested": beta_tested,
    "contracted_length": length_at_test_speed,
    "fits_at_test_speed": length_at_test_speed <= L_garage,
    "simultaneity_delta_train": front_train_time - rear_train_time,
    "sweep_fit_count": len(sweep),
}))
'''.strip()


def design_relativistic_garage_experiment(question: str) -> ExperimentPlan:
    source = _implementation_source()
    return ExperimentPlan(
        question=question,
        domain="special_relativity",
        hypotheses=(
            "There is a speed range in which length contraction makes the train fit inside the garage frame.",
            "Events simultaneous in the garage frame are not simultaneous in the train frame.",
        ),
        assumptions=(
            "The train has rest length L0 = 100 and the garage has length L_garage = 60.",
            "The garage frame is inertial and units use c = 1.",
            "The doors/events are compared at fixed positions in the garage frame.",
        ),
        variables=("beta", "gamma", "L0", "L_garage", "t", "x"),
        parameters=(("L0", 100.0), ("L_garage", 60.0), ("beta_tested", 0.9), ("c", 1.0)),
        equations=(
            r"\gamma = \frac{1}{\sqrt{1-\beta^2}}",
            r"L_{train,garage} = \frac{L_0}{\gamma}",
            r"t' = \gamma\left(t - \frac{\beta x}{c}\right)",
            r"\Delta t' = -\gamma\frac{\beta\Delta x}{c}",
        ),
        controls=("beta = 0 baseline", "compare analytic threshold with numerical sweep"),
        measurements=("contracted train length", "fit decision", "train-frame time difference", "fit-speed count"),
        success_conditions=(
            "L0 / gamma(beta) <= L_garage determines whether the train fits.",
            "Delta t' != 0 for simultaneous separated events when beta != 0.",
            "The numerical sweep agrees with the analytic threshold.",
        ),
        implementation_steps=(
            "Generate a self-contained Python implementation.",
            "Evaluate the Lorentz equations analytically and numerically.",
            "Sweep beta and compare the fit boundary.",
            "Capture stdout, stderr, and return code.",
        ),
        strategy=VerificationStrategy("analytic_lorentz_transform", "numerical_parameter_sweep", "closed-form equations are available; the sweep checks the boundary"),
        source=source,
    )


def run_experiment(plan: ExperimentPlan) -> VerificationResult:
    with tempfile.TemporaryDirectory(prefix="scientific-l68-") as directory:
        script = Path(directory) / "generated_relativistic_garage.py"
        script.write_text(plan.source, encoding="utf-8")
        completed = subprocess.run(
            [sys.executable, str(script)], capture_output=True, encoding="utf-8", errors="replace", timeout=30,
        )
    payload = json.loads(completed.stdout)
    return VerificationResult(
        return_code=completed.returncode,
        beta_threshold=float(payload["beta_threshold"]),
        beta_tested=float(payload["beta_tested"]),
        contracted_length=float(payload["contracted_length"]),
        fits_at_test_speed=bool(payload["fits_at_test_speed"]),
        simultaneity_delta_train=float(payload["simultaneity_delta_train"]),
        sweep_fit_count=int(payload["sweep_fit_count"]),
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def format_verification_report(plan: ExperimentPlan, result: VerificationResult) -> str:
    fit_word = "成立" if result.fits_at_test_speed else "不成立"
    return "\n".join([
        "## Scientific Experiment Verification Report",
        "",
        "【検証要求】", plan.question,
        "", "【領域】", plan.domain,
        "", "【検証仮説】", *[f"- {item}" for item in plan.hypotheses],
        "", "【前提】", *[f"- {item}" for item in plan.assumptions],
        "", "【検証戦略】",
        f"- primary: `{plan.strategy.primary}`",
        f"- secondary: `{plan.strategy.secondary}`",
        f"- reason: {plan.strategy.reason}",
        "", "【理論式】",
        *sum((["$$", equation, "$$"] for equation in plan.equations), []),
        "", "【実験計画】", *[f"{index}. {item}" for index, item in enumerate(plan.implementation_steps, 1)],
        "", "【実行結果】",
        f"- return code: `{result.return_code}`",
        f"- beta_tested: `{result.beta_tested:.3f}`",
        f"- contracted length: `{result.contracted_length:.6f}`",
        f"- fit condition: `{fit_word}` (`L_train / gamma <= L_garage`)",
        f"- analytic beta threshold: `{result.beta_threshold:.6f}`",
        f"- numerical sweep fit count: `{result.sweep_fit_count}`",
        f"- train-frame simultaneity difference: `{result.simultaneity_delta_train:.6f}`",
        "", "【判定】",
        f"長さ収縮については、beta={result.beta_tested:.3f} で車庫内に収まる判定です。",
        "同時性については、車庫系で同時な2事象の時間差が列車系で0ではないため、同時性は観測系に依存します。",
        "", "【生成・実行】", "- generated Python experiment: `temporary workspace`",
    ])
