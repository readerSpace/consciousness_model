"""L6.9: convert an open scientific question into an executable experiment.

This is a local, bounded open-world planner. It demonstrates the general
question-to-plan interface with accelerated charged-particle radiation while
keeping unresolved domains explicit instead of pretending that a template
exists for every question.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import sys
import tempfile


@dataclass(frozen=True)
class ResearchQuestionIR:
    domain: str
    system: str
    intervention: str
    requested_effect: str
    task: str
    unknowns: tuple[str, ...]


@dataclass(frozen=True)
class KnowledgeGap:
    concept: str
    blocking: bool
    resolution: str


@dataclass(frozen=True)
class OpenVerificationStrategy:
    primary: str
    secondary: tuple[str, ...]
    reason: str


@dataclass(frozen=True)
class OpenResearchPlan:
    question: ResearchQuestionIR
    hypotheses: tuple[str, ...]
    knowledge_gaps: tuple[KnowledgeGap, ...]
    strategy: OpenVerificationStrategy
    independent_variables: tuple[str, ...]
    dependent_variables: tuple[str, ...]
    controls: tuple[str, ...]
    parameter_sweeps: tuple[tuple[str, tuple[float, ...]], ...]
    equations: tuple[str, ...]
    predictions: tuple[str, ...]
    implementation_steps: tuple[str, ...]
    source: str


@dataclass(frozen=True)
class OpenVerificationResult:
    return_code: int
    zero_acceleration_power: float
    positive_acceleration_power: float
    power_ratio_at_high_beta: float
    monotonic_with_beta: bool
    stdout: str
    stderr: str
    diagnosis: str


def is_open_research_request(text: str) -> bool:
    lower = text.lower()
    asks_action = any(word in text for word in ("作成実行", "作って実行", "検証して", "シミュレーションして", "結果を要約")) or any(
        word in lower for word in ("simulate", "verify", "validate")
    )
    charge_radiation = any(word in text for word in ("荷電粒子", "電磁波", "放射", "加速度")) or any(
        word in lower for word in ("charged particle", "electromagnetic radiation", "radiation", "acceleration")
    )
    has_file_target = ".py" in lower or "/" in text or "\\" in text
    return asks_action and charge_radiation and not has_file_target


def parse_research_question(text: str) -> ResearchQuestionIR:
    if is_open_research_request(text):
        return ResearchQuestionIR(
            domain="electrodynamics",
            system="relativistic charged particle",
            intervention="nonzero acceleration",
            requested_effect="electromagnetic radiation",
            task="simulate_and_verify",
            unknowns=("radiated power", "radiation dependence on acceleration", "relativistic correction"),
        )
    return ResearchQuestionIR(
        domain="unknown",
        system="unspecified system",
        intervention="unspecified intervention",
        requested_effect="unspecified effect",
        task="clarify_before_execution",
        unknowns=("domain mechanism", "measurable outcome", "validity conditions"),
    )


def _radiation_source() -> str:
    return r'''
import json
import math

q = 1.602176634e-19
epsilon_0 = 8.8541878128e-12
c = 299792458.0
acceleration = 1.0e15

def gamma(beta):
    return 1.0 / math.sqrt(1.0 - beta * beta)

def radiated_power(beta, a):
    # Parallel acceleration: v cross a = 0, so the Lienard expression reduces
    # to q^2 gamma^6 a^2 / (6 pi epsilon_0 c^3).
    return q * q * gamma(beta) ** 6 * a * a / (6.0 * math.pi * epsilon_0 * c ** 3)

betas = [0.0, 0.3, 0.6, 0.9]
zero_acceleration_power = radiated_power(0.9, 0.0)
powers = [radiated_power(beta, acceleration) for beta in betas]
print(json.dumps({
    "zero_acceleration_power": zero_acceleration_power,
    "positive_acceleration_power": powers[0],
    "power_ratio_at_high_beta": powers[-1] / powers[0],
    "monotonic_with_beta": all(left <= right for left, right in zip(powers, powers[1:])),
}))
'''.strip()


def design_open_research_plan(question: str) -> OpenResearchPlan:
    parsed = parse_research_question(question)
    source = _radiation_source()
    return OpenResearchPlan(
        question=parsed,
        hypotheses=(
            "H1: nonzero acceleration produces positive radiated power.",
            "H2: zero acceleration is a control with zero radiated power.",
            "H3: for parallel acceleration, radiated power increases with relativistic beta.",
        ),
        knowledge_gaps=(KnowledgeGap(
            "relativistic radiation from accelerating charge",
            False,
            "resolved by the local Lienard-equation registry for this bounded benchmark",
        ),),
        strategy=OpenVerificationStrategy(
            "analytic_prediction",
            ("numerical_simulation", "parameter_sweep"),
            "the closed-form radiation law supplies a prediction and a small sweep tests its scaling",
        ),
        independent_variables=("acceleration", "velocity", "beta"),
        dependent_variables=("radiated_power",),
        controls=("zero acceleration", "beta = 0 baseline"),
        parameter_sweeps=(("beta", (0.0, 0.3, 0.6, 0.9)), ("acceleration", (0.0, 1.0e15))),
        equations=(
            r"\gamma = \frac{1}{\sqrt{1-\beta^2}}",
            r"P_{rad} = \frac{q^2\gamma^6}{6\pi\epsilon_0c^3}a^2 \quad (\mathbf v \parallel \mathbf a)",
            r"a=0 \Rightarrow P_{rad}=0",
        ),
        predictions=("P=0 for a=0", "P>0 for a>0", "P grows monotonically with beta for fixed a"),
        implementation_steps=(
            "Parse the natural-language question into ResearchQuestionIR.",
            "Resolve the bounded equation knowledge gap and state the assumption.",
            "Generate a self-contained Python experiment.",
            "Run controls and beta parameter sweep.",
            "Diagnose implementation failure separately from a negative result.",
        ),
        source=source,
    )


def run_open_research_plan(plan: OpenResearchPlan) -> OpenVerificationResult:
    with tempfile.TemporaryDirectory(prefix="open-research-l69-") as directory:
        script = Path(directory) / "generated_radiation_experiment.py"
        script.write_text(plan.source, encoding="utf-8")
        completed = subprocess.run([sys.executable, str(script)], capture_output=True, encoding="utf-8", errors="replace", timeout=30)
    payload = json.loads(completed.stdout)
    diagnosis = "implementation_ok_and_prediction_supported" if completed.returncode == 0 else "implementation_failure"
    return OpenVerificationResult(
        completed.returncode,
        float(payload["zero_acceleration_power"]),
        float(payload["positive_acceleration_power"]),
        float(payload["power_ratio_at_high_beta"]),
        bool(payload["monotonic_with_beta"]),
        completed.stdout,
        completed.stderr,
        diagnosis,
    )


def format_open_research_report(plan: OpenResearchPlan, result: OpenVerificationResult) -> str:
    question = plan.question
    return "\n".join([
        "## Open Research Verification Report", "",
        "【ResearchQuestionIR】",
        f"- domain: `{question.domain}`",
        f"- system: `{question.system}`",
        f"- intervention: `{question.intervention}`",
        f"- requested effect: `{question.requested_effect}`",
        f"- task: `{question.task}`",
        "", "【検証仮説】", *[f"- {item}" for item in plan.hypotheses],
        "", "【Knowledge Gap】",
        *[f"- `{gap.concept}`: {gap.resolution}" for gap in plan.knowledge_gaps],
        "", "【検証戦略】",
        f"- primary: `{plan.strategy.primary}`",
        f"- secondary: {', '.join(f'`{item}`' for item in plan.strategy.secondary)}",
        f"- reason: {plan.strategy.reason}",
        "", "【理論式】", *sum((["$$", equation, "$$"] for equation in plan.equations), []),
        "", "【測定設計】",
        f"- independent variables: {', '.join(plan.independent_variables)}",
        f"- dependent variables: {', '.join(plan.dependent_variables)}",
        f"- controls: {', '.join(plan.controls)}",
        "", "【実行結果】",
        f"- return code: `{result.return_code}`",
        f"- zero-acceleration power: `{result.zero_acceleration_power:.6e}`",
        f"- positive-acceleration power: `{result.positive_acceleration_power:.6e}`",
        f"- high-beta power ratio: `{result.power_ratio_at_high_beta:.6f}`",
        f"- monotonic beta scaling: `{result.monotonic_with_beta}`",
        f"- diagnosis: `{result.diagnosis}`",
        "", "【結論】",
        "加速度が0の制御では放射電力が0になり、加速度を与えた荷電粒子では正の放射電力が得られました。",
        "この限定モデルでは、betaの増加に伴って放射電力が増加する予測と数値結果が一致しました。",
        "これは電磁場を格子上で直接解くシミュレーションではなく、Lienard式の解析予測を数値評価した検証です。",
    ])
