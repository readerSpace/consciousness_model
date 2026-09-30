"""L7.0: route scientific knowledge retrieval and mathematical derivation.

The retrieval corpus is an offline, inspectable seed registry. A live web
adapter can replace it later without changing the evidence and research-plan
interfaces. This keeps the local coding system deterministic and API-free.
"""
from __future__ import annotations

from dataclasses import dataclass


class ScientificIntent:
    REPO_TASK = "repo_task"
    SCIENTIFIC_RESEARCH = "scientific_research"
    MATH_DERIVATION = "math_derivation"
    OPEN_EXPERIMENT = "open_experiment"
    EXECUTE_CODE = "execute_code"


@dataclass(frozen=True)
class ScientificEvidence:
    claim: str
    source_type: str
    source: str
    equations: tuple[str, ...]
    assumptions: tuple[str, ...]
    confidence: float


@dataclass(frozen=True)
class ResearchPlan:
    question: str
    topic: str
    known_results: tuple[str, ...]
    unresolved_points: tuple[str, ...]
    candidate_equations: tuple[str, ...]
    experimentally_testable_claims: tuple[str, ...]
    proposed_experiments: tuple[str, ...]
    evidence: tuple[ScientificEvidence, ...]


@dataclass(frozen=True)
class DerivationResult:
    equation: str
    assumptions: tuple[str, ...]
    steps: tuple[str, ...]
    latex: tuple[str, ...]
    boundary_conditions: tuple[str, ...]
    caveat: str


def classify_scientific_intent(text: str) -> str:
    lower = text.lower()
    if any(word in text for word in ("導出", "解を求め", "波動方程式", "偏微分方程式")) or any(
        word in lower for word in ("derive", "wave equation", "pde")
    ):
        return ScientificIntent.MATH_DERIVATION
    if any(word in text for word in ("論文", "研究", "文献", "量子情報", "時空が創発", "検索して", "相互情報量", "エンタングルメントエントロピー")) or any(
        word in lower for word in ("paper", "papers", "literature", "emergent spacetime", "quantum information", "mutual information", "entanglement entropy", "pairwise distance")
    ):
        return ScientificIntent.SCIENTIFIC_RESEARCH
    if any(word in text for word in ("検証", "実験", "シミュレーション")) or any(
        word in lower for word in ("verify", "experiment", "simulate")
    ):
        return ScientificIntent.OPEN_EXPERIMENT
    if any(word in text for word in ("実行", "動かして")) or any(word in lower for word in ("run", "execute")):
        return ScientificIntent.EXECUTE_CODE
    return ScientificIntent.REPO_TASK


def is_scientific_research_request(text: str) -> bool:
    return classify_scientific_intent(text) == ScientificIntent.SCIENTIFIC_RESEARCH


def is_math_derivation_request(text: str) -> bool:
    return classify_scientific_intent(text) == ScientificIntent.MATH_DERIVATION


_EMERGENT_SPACETIME_EVIDENCE = (
    ScientificEvidence(
        "Holographic entanglement entropy relates boundary-region entropy to the area of a minimal bulk surface.",
        "paper", "https://arxiv.org/abs/hep-th/0603001",
        (r"S_A = \frac{\mathrm{Area}(\gamma_A)}{4G_N}",),
        ("AdS/CFT correspondence", "semiclassical bulk geometry", "appropriate extremal surface exists"), 0.96,
    ),
    ScientificEvidence(
        "Changing entanglement between regions is proposed to change whether a connected spacetime geometry persists.",
        "paper", "https://arxiv.org/abs/1005.3035",
        (r"\text{connected geometry} \leftrightarrow \text{entanglement structure}",),
        ("non-perturbative quantum-gravity description", "entanglement is the relevant coupling structure"), 0.88,
    ),
    ScientificEvidence(
        "Tensor-network quantum-error-correcting codes provide toy models where bulk logical information is encoded on a boundary.",
        "paper", "https://arxiv.org/abs/1503.06237",
        (r"\mathcal{H}_{bulk} \hookrightarrow \mathcal{H}_{boundary}",),
        ("toy tensor-network code", "bulk and boundary degrees of freedom are identified as logical and physical systems"), 0.94,
    ),
)


def retrieve_scientific_evidence(topic: str) -> tuple[ScientificEvidence, ...]:
    normalized = topic.lower()
    if any(word in normalized for word in ("quantum", "entanglement", "spacetime", "時空", "量子情報")):
        return _EMERGENT_SPACETIME_EVIDENCE
    return ()


def build_research_plan(question: str) -> ResearchPlan:
    evidence = retrieve_scientific_evidence(question)
    return ResearchPlan(
        question=question,
        topic="emergent spacetime from quantum information",
        known_results=tuple(item.claim for item in evidence),
        unresolved_points=(
            "Whether a finite quantum-information model reproduces a geometry beyond a qualitative analogy.",
            "Which entanglement observable should be treated as a geometric distance or area proxy.",
            "How robust the inferred geometry is under noise and finite-size effects.",
        ),
        candidate_equations=tuple(equation for item in evidence for equation in item.equations),
        experimentally_testable_claims=(
            "Increasing entanglement connectivity should reduce an inferred graph distance between subsystems.",
            "Removing entanglement links should split or lengthen the inferred geometry.",
            "A tensor-network toy model should preserve logical information while changing boundary reconstruction paths.",
        ),
        proposed_experiments=(
            "Construct a small tensor-network or graph-state family with tunable bond entanglement.",
            "Estimate pairwise distance from mutual information or entanglement entropy.",
            "Sweep bond dimension and noise, then compare connectivity and reconstruction fidelity.",
            "Use a disentangled control with the same number of nodes and local dimensions.",
        ),
        evidence=evidence,
    )


def derive_1d_wave_equation(question: str) -> DerivationResult:
    return DerivationResult(
        equation=r"\frac{\partial^2 u}{\partial t^2} = c^2 \frac{\partial^2 u}{\partial x^2}",
        assumptions=("one spatial dimension", "constant wave speed c", "fixed endpoints at x=0 and x=L"),
        steps=(
            "Assume u(x,t)=X(x)T(t).",
            r"Substitution gives \frac{T''}{c^2T}=\frac{X''}{X}=-k^2.",
            r"The boundary conditions X(0)=X(L)=0 select k_n=\frac{n\pi}{L}.",
            r"Therefore T_n(t) oscillates with \omega_n=ck_n.",
        ),
        latex=(
            r"X_n(x)=\sin\left(\frac{n\pi x}{L}\right)",
            r"\omega_n=\frac{n\pi c}{L}",
            r"u(x,t)=\sum_{n=1}^{\infty}\left[A_n\cos(\omega_nt)+B_n\sin(\omega_nt)\right]\sin\left(\frac{n\pi x}{L}\right)",
        ),
        boundary_conditions=(r"u(0,t)=0", r"u(L,t)=0"),
        caveat="係数 A_n と B_n には初期変位と初期速度が必要です。境界条件が異なる場合は固有関数も変わります。",
    )


def format_research_report(plan: ResearchPlan) -> str:
    lines = ["## Scientific Research Report", "", "【研究質問】", plan.question, "", "【トピック】", plan.topic, "", "【既知結果】"]
    for evidence in plan.evidence:
        lines.extend([f"- {evidence.claim}", f"  - source: [{evidence.source_type}]({evidence.source})", f"  - confidence: `{evidence.confidence:.2f}`"])
        for equation in evidence.equations:
            lines.extend(["  - equation:", "    $$", f"    {equation}", "    $$"])
    lines.extend(["", "【未解決点】", *[f"- {item}" for item in plan.unresolved_points], "", "【検証可能な主張】", *[f"- {item}" for item in plan.experimentally_testable_claims], "", "【提案実験】", *[f"{index}. {item}" for index, item in enumerate(plan.proposed_experiments, 1)], "", "【注意】", "文献の主張と、この計画で新たに検証する予測を分離しています。"])
    return "\n".join(lines)


def format_derivation_report(result: DerivationResult) -> str:
    lines = ["## Mathematical Derivation Report", "", "【方程式】", "$$", result.equation, "$$", "", "【前提】", *[f"- {item}" for item in result.assumptions], "", "【導出手順】", *[f"{index}. {item}" for index, item in enumerate(result.steps, 1)], "", "【境界条件】"]
    for condition in result.boundary_conditions:
        lines.extend(["$$", condition, "$$"])
    lines.extend(["", "【一般解】"])
    for equation in result.latex:
        lines.extend(["$$", equation, "$$"])
    lines.extend(["", "【適用範囲】", result.caveat])
    return "\n".join(lines)
