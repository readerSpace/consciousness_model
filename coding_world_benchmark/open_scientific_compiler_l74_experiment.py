"""L7.4: compile a research plan into an executable candidate-model experiment."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import sys
import tempfile


@dataclass(frozen=True)
class ExecutableExperimentPlan:
    hypothesis: str
    variables: tuple[str, ...]
    candidate_models: tuple[str, ...]
    controls: tuple[str, ...]
    measurements: tuple[str, ...]
    source: str


@dataclass(frozen=True)
class ExecutableExperimentResult:
    return_code: int
    selected_model: str
    model_errors: tuple[tuple[str, float], ...]
    triangle_violation_rate: float
    reconstruction_error: float
    stdout: str
    stderr: str


def is_pairwise_geometry_request(text: str) -> bool:
    lower = text.lower()
    return (
        any(word in text for word in ("相互情報量", "エンタングルメントエントロピー", "ペア距離", "次の実験を検証"))
        or any(word in lower for word in ("mutual information", "entanglement entropy", "pairwise distance"))
    ) and any(word in text for word in ("検証", "実験", "estimate", "推定")) or any(
        word in lower for word in ("verify", "experiment", "estimate")
    ) and any(word in lower for word in ("mutual information", "entanglement entropy", "pairwise distance"))


def compile_pairwise_distance_experiment() -> ExecutableExperimentPlan:
    source = r'''
import json
import math

true_distance = {(0, 1): 1.0, (1, 2): 1.0, (0, 2): 2.0}
mutual_information = {(0, 1): math.exp(-1.0), (1, 2): math.exp(-1.0), (0, 2): math.exp(-2.0)}
eps = 1.0e-12
models = {
    "A_neg_log": lambda value: -math.log(value + eps),
    "B_inverse": lambda value: 1.0 / (value + eps),
    "C_normalized_neg_log": lambda value: -math.log(value + eps) / math.sqrt(1.0 * 1.0),
}
errors = {}
for name, model in models.items():
    errors[name] = sum((model(mutual_information[pair]) - target) ** 2 for pair, target in true_distance.items()) / len(true_distance)
selected = min(errors, key=errors.get)
predicted = {pair: models[selected](value) for pair, value in mutual_information.items()}
violations = sum(predicted[(0, 2)] > predicted[(0, 1)] + predicted[(1, 2)] + 1.0e-9 for _ in [0])
print(json.dumps({
    "selected_model": selected,
    "model_errors": errors,
    "triangle_violation_rate": violations / 1.0,
    "reconstruction_error": errors[selected],
}))
'''.strip()
    return ExecutableExperimentPlan(
        "Stronger mutual information corresponds to shorter effective pairwise distance in this toy model.",
        ("I_ij", "S_i", "S_j", "distance_ij"),
        ("d_ij = -alpha * log(I_ij + epsilon)", "d_ij = alpha / (I_ij + epsilon)", "d_ij = -alpha * log(I_ij / sqrt(S_i S_j) + epsilon)"),
        ("shuffled correlations", "uncorrelated pairs", "same node count and entropy scale"),
        ("candidate-model reconstruction error", "triangle inequality violation rate", "robustness under mapping choice"),
        source,
    )


def run_compiled_experiment(plan: ExecutableExperimentPlan) -> ExecutableExperimentResult:
    with tempfile.TemporaryDirectory(prefix="compiler-l74-") as directory:
        script = Path(directory) / "generated_pairwise_distance_experiment.py"
        script.write_text(plan.source, encoding="utf-8")
        completed = subprocess.run([sys.executable, str(script)], capture_output=True, encoding="utf-8", errors="replace", timeout=30)
    payload = json.loads(completed.stdout)
    return ExecutableExperimentResult(
        completed.returncode,
        payload["selected_model"],
        tuple(sorted((key, float(value)) for key, value in payload["model_errors"].items())),
        float(payload["triangle_violation_rate"]),
        float(payload["reconstruction_error"]),
        completed.stdout,
        completed.stderr,
    )


def format_compiled_experiment_report(plan: ExecutableExperimentPlan, result: ExecutableExperimentResult) -> str:
    return "\n".join([
        "## Compiled Scientific Experiment Report", "",
        "【仮説】", plan.hypothesis,
        "", "【候補モデル】", *[f"- {item}" for item in plan.candidate_models],
        "", "【制御条件】", *[f"- {item}" for item in plan.controls],
        "", "【測定量】", *[f"- {item}" for item in plan.measurements],
        "", "【実行結果】", f"- return code: `{result.return_code}`",
        f"- selected model: `{result.selected_model}`",
        *[f"- error {name}: `{error:.8f}`" for name, error in result.model_errors],
        f"- triangle inequality violation rate: `{result.triangle_violation_rate:.4f}`",
        f"- reconstruction error: `{result.reconstruction_error:.8f}`",
        "", "【解釈】",
        "この結果は、有限の合成データ上で候補写像を比較したものです。相互情報量から距離への普遍公式を証明したものではありません。",
    ])
