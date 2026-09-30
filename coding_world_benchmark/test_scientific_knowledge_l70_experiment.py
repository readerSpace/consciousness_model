from coding_world_benchmark.canonical_ir_l63_experiment import canonicalize_request
from coding_world_benchmark.coding_agent_app import LocalWorkspaceAgent
from coding_world_benchmark.scientific_knowledge_l70_experiment import (
    ScientificIntent,
    build_research_plan,
    classify_scientific_intent,
    derive_1d_wave_equation,
    format_derivation_report,
    format_research_report,
    is_math_derivation_request,
    is_scientific_research_request,
)
from coding_world_benchmark.structured_repository_report_l62_experiment import RequestType, classify_request


RESEARCH_REQUEST = "量子情報から時空が創発するか検証するために論文を検索してそれを参考に検証して"
DERIVATION_REQUEST = "波動方程式の解を導出して"


def test_scientific_and_math_requests_have_separate_intents():
    assert classify_scientific_intent(RESEARCH_REQUEST) == ScientificIntent.SCIENTIFIC_RESEARCH
    assert classify_scientific_intent(DERIVATION_REQUEST) == ScientificIntent.MATH_DERIVATION
    assert is_scientific_research_request(RESEARCH_REQUEST)
    assert is_math_derivation_request(DERIVATION_REQUEST)
    assert classify_request(RESEARCH_REQUEST).request_type is RequestType.SCIENTIFIC_RESEARCH
    assert classify_request(DERIVATION_REQUEST).request_type is RequestType.MATH_DERIVATION
    assert canonicalize_request(RESEARCH_REQUEST).action == "scientific_research"
    assert canonicalize_request(DERIVATION_REQUEST).action == "math_derivation"


def test_research_plan_extracts_evidence_claims_equations_and_experiments():
    plan = build_research_plan(RESEARCH_REQUEST)

    assert plan.topic == "emergent spacetime from quantum information"
    assert len(plan.evidence) == 3
    assert all(item.source_type == "paper" for item in plan.evidence)
    assert any("Area" in equation for equation in plan.candidate_equations)
    assert plan.experimentally_testable_claims
    assert plan.proposed_experiments
    assert "arxiv.org" in format_research_report(plan)


def test_wave_equation_derivation_is_assumption_bounded_and_latex_ready():
    result = derive_1d_wave_equation(DERIVATION_REQUEST)
    report = format_derivation_report(result)

    assert result.boundary_conditions == (r"u(0,t)=0", r"u(L,t)=0")
    assert any(r"\omega_n" in equation for equation in result.latex)
    assert "【導出手順】" in report
    assert "【一般解】" in report
    assert "$$" in report
    assert "初期変位" in result.caveat


def test_app_routes_science_and_derivation_before_repository_search(tmp_path):
    agent = LocalWorkspaceAgent(tmp_path)
    research = agent.handle(RESEARCH_REQUEST)
    derivation = agent.handle(DERIVATION_REQUEST)

    assert "Scientific Research Report" in research.text
    assert "【提案実験】" in research.text
    assert "候補ファイル" not in research.text
    assert "Mathematical Derivation Report" in derivation.text
    assert "【一般解】" in derivation.text
    assert "候補ファイル" not in derivation.text
