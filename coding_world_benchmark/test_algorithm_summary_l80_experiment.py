from coding_world_benchmark.algorithm_summary_l80_experiment import (
    format_algorithm_summary,
    is_algorithm_summary_request,
    summarize_algorithm,
)


def test_algorithm_summary_extracts_stages_and_evidence(tmp_path):
    module = tmp_path / "Consciousness_model"
    module.mkdir()
    (module / "core.py").write_text(
        "def run():\n"
        "    state = initialize_workspace()\n"
        "    candidates = generate_candidates(state)\n"
        "    choice = select_candidate(candidates)\n"
        "    result = execute_candidate(choice)\n"
        "    return update_memory(result)\n\n"
        "def initialize_workspace():\n    return {}\n\n"
        "def generate_candidates(state):\n    return [state]\n\n"
        "def select_candidate(candidates):\n    return candidates[0]\n\n"
        "def execute_candidate(choice):\n    return choice\n\n"
        "def update_memory(result):\n    memory = result\n    return memory\n",
        encoding="utf-8",
    )
    request = "consciousness_modelのアルゴリズムを要約して"

    assert is_algorithm_summary_request(request)
    summary = summarize_algorithm(tmp_path, request)
    report = format_algorithm_summary(summary)

    assert len(summary.stages) >= 3
    assert len(summary.symbols) >= 2
    assert summary.dataflow
    assert "【コード根拠】" in report
    assert "core.py:" in report
    assert "ファイル一覧" not in report


def test_non_algorithm_summary_is_not_routed_to_algorithm_summary():
    assert not is_algorithm_summary_request("consciousness_modelのREADMEを要約して")
