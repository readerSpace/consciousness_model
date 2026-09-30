from pathlib import Path

from coding_world_benchmark.coding_agent_app import LocalWorkspaceAgent
from coding_world_benchmark.grounded_repository_explanation_l59_experiment import (
    InferenceType,
    explain,
    format_explanation,
)


def _repo(tmp_path: Path) -> Path:
    (tmp_path / "model.py").write_text(
        "def make_initial_state():\n"
        "    rho = 0.45\n"
        "    information = rho\n"
        "    return information\n\n"
        "def expA1_initial_state():\n"
        "    state = make_initial_state()\n"
        "    return evolve(state)\n\n"
        "def exp445_initial_state():\n"
        "    return make_initial_state()\n\n"
        "def evolve(state):\n"
        "    return state\n",
        encoding="utf-8",
    )
    return tmp_path


def test_explanation_claims_are_grounded_and_typed(tmp_path: Path):
    result = explain(_repo(tmp_path), "initial state")
    assert result.ambiguous
    assert result.claims
    assert all(claim.evidence_ids for claim in result.claims)
    kinds = {claim.inference_type for claim in result.claims}
    assert InferenceType.AMBIGUOUS in kinds
    assert InferenceType.CALL_GRAPH_INFERENCE in kinds
    assert InferenceType.DATAFLOW_INFERENCE in kinds
    rendered = format_explanation(result)
    assert "根拠付きRepository説明" in rendered
    assert "model.py:" in rendered
    assert "runtime" in rendered


def test_local_app_uses_grounded_explanation_for_natural_language_request(tmp_path: Path):
    _repo(tmp_path)
    agent = LocalWorkspaceAgent(tmp_path)
    result = agent.handle("初期状態はどのように作られるか解説して")
    assert result.provider == "local"
    assert "根拠付きRepository説明" in result.text
    assert "make_initial_state" in result.text
    assert "根拠:" in result.text


def test_non_explanation_requests_use_local_summary_route(tmp_path: Path):
    result = LocalWorkspaceAgent(_repo(tmp_path)).handle("このコードを確認して")
    assert result.provider == "local"
    assert "ローカル調査結果" in result.text
    assert "model.py" in result.text
