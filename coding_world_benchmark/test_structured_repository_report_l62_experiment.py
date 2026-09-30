from pathlib import Path

from coding_world_benchmark.structured_repository_report_l62_experiment import (
    RequestType,
    classify_request,
    structured_repository_report,
)


def test_classifies_mixed_language_request_and_resolves_file():
    intent = classify_request("qg_dynamic_geometry.pyを要約して explain the main flow")
    assert intent.request_type is RequestType.EXPLAIN
    assert intent.target == "qg_dynamic_geometry.py"


def test_classifies_execute_and_test_as_action_requests():
    execute = classify_request("シミュレーションのpythonファイルを実行して")
    test = classify_request("テストして")

    assert execute.request_type is RequestType.EXECUTE
    assert test.request_type is RequestType.TEST


def test_structured_report_contains_graph_sections_and_evidence(tmp_path: Path):
    (tmp_path / "model.py").write_text(
        "def make_initial_state():\n    rho = 0.45\n    return rho\n\n"
        "def expA1_initial_state():\n    state = make_initial_state()\n    return evolve(state)\n\n"
        "def evolve(state):\n    return state\n", encoding="utf-8"
    )
    report = structured_repository_report(tmp_path, "model.pyを要約して")
    assert "【対象】model.py" in report
    assert "【主要な実験】" in report
    assert "【呼び出し関係】" in report
    assert "【根拠】" in report
    assert "model.py:" in report
