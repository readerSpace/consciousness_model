from pathlib import Path

from coding_world_benchmark.coding_agent_app import (
    LocalWorkspaceAgent,
    compile_markdown,
    load_session,
    memory_policy_for_message,
    save_session,
    save_work_log,
)
from coding_world_benchmark.local_paper_l77_experiment import PDFMode, PaperKnowledge


def test_workspace_selection_and_safe_local_commands(tmp_path: Path):
    (tmp_path / "main.py").write_text("VALUE = 7\n", encoding="utf-8")
    agent = LocalWorkspaceAgent()
    assert "作業フォルダ" in agent.set_workspace(tmp_path)
    assert "main.py" in agent.handle("/inspect").text
    assert "VALUE = 7" in agent.handle("/read main.py").text
    assert "main.py:1" in agent.handle("/find value").text
    assert "/tests" in agent.handle("/help").text
    assert "/self-benchmark" in agent.handle("/help").text
    assert "/behavior-audit" in agent.handle("/help").text
    assert "/goal <request>" in agent.handle("/help").text


def test_verification_feature_commands_route_to_coding_system(monkeypatch):
    agent = LocalWorkspaceAgent()
    monkeypatch.setattr(agent, "self_benchmark", lambda: "SELF BENCHMARK REPORT")
    monkeypatch.setattr(agent, "holdout_benchmark", lambda: "HOLDOUT REPORT")
    monkeypatch.setattr(agent, "behavior_audit", lambda: "BEHAVIOR AUDIT REPORT")

    assert agent.handle("/self-benchmark").text == "SELF BENCHMARK REPORT"
    assert agent.handle("/holdout-benchmark").text == "HOLDOUT REPORT"
    assert agent.handle("/behavior-audit").text == "BEHAVIOR AUDIT REPORT"


def test_workspace_path_escape_is_rejected(tmp_path: Path):
    agent = LocalWorkspaceAgent(tmp_path)
    result = agent.handle("/read ../outside.txt")
    assert "workspace外" in result.text or "見つかりません" in result.text


def test_natural_language_is_resolved_by_local_investigation(tmp_path: Path):
    (tmp_path / "measure.py").write_text("# 測定処理\n\ndef measure_signal():\n    return 42\n", encoding="utf-8")
    agent = LocalWorkspaceAgent(tmp_path)
    result = agent.handle("このコードの測定処理を確認して")
    assert result.provider == "local"
    assert "ローカル調査結果" in result.text
    assert "measure.py" in result.text


def test_natural_language_without_workspace_answers_without_local_investigation():
    result = LocalWorkspaceAgent().handle("一様磁場中で荷電粒子が運動するシミュレーションを作成して")

    assert result.provider == "no-workspace"
    assert "ローカルフォルダ未指定" in result.text
    assert "ファイル一覧・検索・コード読取は行っていません" in result.text
    assert "GoalIR" in result.text
    assert "TaskGraph" in result.text
    assert "先に" not in result.text


def test_summary_request_uses_local_investigation(tmp_path: Path):
    (tmp_path / "research.md").write_text("# Big Bang以前\npre big ban validation result\n", encoding="utf-8")
    agent = LocalWorkspaceAgent(tmp_path)

    result = agent.handle("pre big banの検証の内容をまとめて")

    assert result.provider == "local"
    assert "ローカル要約" in result.text
    assert "research.md" in result.text
    assert "Big Bang" in result.text


def test_algorithm_summary_returns_process_stages_not_search_listing(tmp_path: Path):
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
    result = LocalWorkspaceAgent(tmp_path).handle("consciousness_modelのアルゴリズムを要約して")

    assert result.provider == "local"
    assert "Algorithm Summary" in result.text
    assert "【アルゴリズム】" in result.text
    assert "初期化" in result.text
    assert "候補生成" in result.text
    assert "【コード根拠】" in result.text
    assert "ローカル要約" not in result.text


def test_external_repair_proposal_precedes_experiment_semantic_route(tmp_path: Path):
    (tmp_path / "closed_loop_abstraction_experiment.py").write_text("def run():\n    return 1\n", encoding="utf-8")
    proposal = """L8.3: Semantic Repair Decomposition
Problem: the repair proposal must reach the patch pipeline
Suggested changes:
- change routing condition
- add helper
Regression tests:
- test_router.py
"""

    result = LocalWorkspaceAgent(tmp_path).handle(proposal)

    assert result.provider == "local"
    assert "Semantic Repair Decomposition" in result.text
    assert "Experiment Semantic Report" not in result.text


def test_external_repair_wrapper_lists_items_instead_of_search_fallback(tmp_path: Path):
    (tmp_path / "closed_loop_abstraction_experiment.py").write_text("def run():\n    return 1\n", encoding="utf-8")
    proposal = """以下の提案から実装項目をリストアップし実装して
L8.6: Self-Modification Benchmark
1. 既知欠陥fixtureを追加
2. failure detectorを追加
3. L8.1〜L8.4 repair pipelineへ接続
4. regression testを自動実行
"""

    result = LocalWorkspaceAgent(tmp_path).handle(proposal)

    assert "Implementation Items" in result.text
    assert "fixture" in result.text
    assert "Autonomous Repair Loop" in result.text
    assert "一致ファイル" not in result.text
    assert "Experiment Semantic Report" not in result.text


def test_task_contract_proposal_outputs_recognized_taskspec_without_validation_memory(tmp_path: Path):
    (tmp_path / "behavior_monitor_l87_experiment.py").write_text("class ExpectedBehavior:\n    pass\n", encoding="utf-8")
    proposal = """以下の提案から実装項目をリストアップし実装して
TaskSpecを画面に表示する
Expected: FILE_MODIFICATION >= 1
Observed: FILE_MODIFICATION = 0
Behavior Monitorで不一致なら異常として扱う
"""

    result = LocalWorkspaceAgent(tmp_path).handle(proposal)

    assert "Recognized TaskSpec" in result.text
    assert "IMPLEMENT_PROPOSAL" in result.text
    assert "task contract visualization" in result.text
    assert "expected behavior generation" in result.text
    assert "gauge_theory_validation" not in result.text
    assert "SU2" not in result.text


def test_task_contract_request_uses_task_contract_memory_policy():
    assert memory_policy_for_message("TaskSpec ExpectedBehavior ObservedBehavior Behavior Monitorを実装して") == "task_contract_only"


def test_scientific_simulation_creation_writes_files_instead_of_search_summary(tmp_path: Path):
    result = LocalWorkspaceAgent(tmp_path).handle("一様磁場中で荷電粒子が運動するシミュレーションを作成して")

    assert "Scientific Simulation Implementation" in result.text
    assert "IMPLEMENT_SCIENTIFIC_SIMULATION" in result.text
    assert (tmp_path / "charged_particle_uniform_B.py").is_file()
    assert (tmp_path / "test_charged_particle_uniform_B.py").is_file()
    assert "ローカル要約" not in result.text
    assert "一致ファイル" not in result.text
    assert "SU2" not in result.text


def test_scientific_simulation_creation_and_execution_runs_generated_tests(tmp_path: Path):
    result = LocalWorkspaceAgent(tmp_path).handle("一様磁場中で荷電粒子が運動するシミュレーションを作成して実行して")

    assert "return code: `0`" in result.text
    assert "test_charged_particle_uniform_B.py" in result.text


def test_scientific_simulation_request_uses_scientific_modeling_memory_policy():
    assert memory_policy_for_message("一様磁場中で荷電粒子が運動するシミュレーションを作成して") == "scientific_modeling_only"


def test_goal_command_renders_goalir_and_taskgraph():
    result = LocalWorkspaceAgent().handle("/goal 一様磁場中で荷電粒子が運動するシミュレーションを作成して")

    assert "Goal Semantics" in result.text
    assert "Requirement Graph" in result.text
    assert "GoalIR" in result.text
    assert "TaskGraph" in result.text
    assert "formulate_physics_model" in result.text
    assert "working artifact exists" in result.text
    assert "search_results_only" in result.text


def test_compact_local_summary_is_short_and_api_free(tmp_path: Path):
    (tmp_path / "research.md").write_text("# Big Bang以前\npre big ban validation result\nextra detail\n", encoding="utf-8")
    agent = LocalWorkspaceAgent(tmp_path)
    summary = agent.summarize_compact("pre big banの検証内容をまとめて", max_chars=1000)
    assert "API不要" in summary
    assert "research.md" in summary
    assert len(summary) <= 1000
    assert "/read <path>" in summary


def test_summary_normalizes_the_typo_and_avoids_single_word_big_matches(tmp_path: Path):
    (tmp_path / "relevant.md").write_text("# Big Bang以前\nBig Bang singularity audit\n", encoding="utf-8")
    (tmp_path / "unrelated.md").write_text("A big collection of unrelated notes.\n", encoding="utf-8")
    agent = LocalWorkspaceAgent(tmp_path)

    result = agent.handle("pre big banの検証内容をまとめて")

    assert "relevant.md" in result.text
    assert "unrelated.md" not in result.text


def test_file_summary_uses_structured_repository_report_offline(tmp_path: Path):
    (tmp_path / "model.py").write_text("def initial_state():\n    return 1\n", encoding="utf-8")
    agent = LocalWorkspaceAgent(tmp_path)
    result = agent.handle("model.pyを要約して")
    assert "Structured Repository Report" in result.text
    assert "【対象】model.py" in result.text
    assert "【根拠】" in result.text


def test_execute_request_runs_resolved_python_script_instead_of_showing_search_results(tmp_path: Path):
    (tmp_path / "README.md").write_text(
        "標準実行:\npython .\\relativistic_tube_simulation.py --beta 0.9\n",
        encoding="utf-8",
    )
    (tmp_path / "relativistic_tube_simulation.py").write_text(
        "import argparse\n"
        "parser = argparse.ArgumentParser()\n"
        "parser.add_argument('--beta')\n"
        "args = parser.parse_args()\n"
        "print('simulation beta=' + args.beta)\n",
        encoding="utf-8",
    )
    result = LocalWorkspaceAgent(tmp_path).handle("シミュレーションのpythonファイルを実行して")

    assert result.provider == "local"
    assert "【実行】" in result.text
    assert "relativistic_tube_simulation.py" in result.text
    assert "simulation beta=0.9" in result.text
    assert "ローカル要約" not in result.text


def test_experiment_configuration_request_returns_semantic_report_not_file_list(tmp_path: Path):
    (tmp_path / "README.md").write_text(
        "Quick start:\npython .\\relativistic_tube_simulation.py --beta 0.9\n",
        encoding="utf-8",
    )
    (tmp_path / "relativistic_tube_simulation.py").write_text(
        "import argparse\n"
        "dt = 0.01\n"
        "steps = 10000\n"
        "initial_position = 0.0\n"
        "initial_velocity = 0.0\n"
        "def initialize_state(beta):\n"
        "    return {'x': initial_position, 'v': initial_velocity, 'beta': beta}\n"
        "def simulate(state, dt, steps):\n"
        "    return [state['x'] + state['beta'] * dt for _ in range(steps)]\n"
        "def measure_trajectory(trajectory):\n"
        "    return max(trajectory)\n"
        "def save_results(value):\n"
        "    print(value)\n"
        "def main():\n"
        "    parser = argparse.ArgumentParser()\n"
        "    parser.add_argument('--beta', type=float, default=0.9)\n"
        "    args = parser.parse_args()\n"
        "    state = initialize_state(args.beta)\n"
        "    trajectory = simulate(state, dt, steps)\n"
        "    save_results(measure_trajectory(trajectory))\n",
        encoding="utf-8",
    )

    result = LocalWorkspaceAgent(tmp_path).handle("pythonでのシミュレーションがどのような設定で行われているか調査して")

    assert "Experiment Semantic Report" in result.text
    assert "【対象実験】" in result.text
    assert "relativistic_tube_simulation.py" in result.text
    assert "`beta`" in result.text
    assert "`dt`" in result.text
    assert "【処理の流れ】" in result.text
    assert "候補ファイル" not in result.text


def test_math_algorithm_request_returns_latex_report_not_search_results(tmp_path: Path):
    (tmp_path / "README.md").write_text(
        "Quick start:\npython .\\relativistic_tube_simulation.py --beta 0.9\n",
        encoding="utf-8",
    )
    (tmp_path / "relativistic_tube_simulation.py").write_text(
        "import math\n"
        "dt = 0.01\n"
        "initial_x = 0.0\n"
        "def simulate(beta, steps):\n"
        "    gamma = 1.0 / math.sqrt(1.0 - beta**2)\n"
        "    x = initial_x\n"
        "    for _ in range(steps):\n"
        "        x = x + beta * dt\n"
        "    return x, gamma\n",
        encoding="utf-8",
    )

    result = LocalWorkspaceAgent(tmp_path).handle("pythonシミュレーションのアルゴリズムを数式で表して")

    assert result.provider == "local"
    assert "Mathematical Algorithm Report" in result.text
    assert "relativistic_tube_simulation.py" in result.text
    assert "【支配方程式】" in result.text
    assert r"\frac{d x}{dt}" in result.text
    assert "【更新則】" in result.text
    assert "$$" in result.text
    assert "\\sqrt" in result.text
    assert "候補ファイル" not in result.text
    assert "ローカル要約" not in result.text


def test_local_paper_request_routes_pdf_to_paper_knowledge(tmp_path: Path, monkeypatch):
    pdf = tmp_path / "paper.pdf"
    pdf.write_bytes(b"placeholder")
    knowledge = PaperKnowledge(str(pdf), PDFMode.TEXT_PDF, 1, (), (), (), (), ())
    monkeypatch.setattr("coding_world_benchmark.coding_agent_app.extract_local_paper", lambda path: knowledge)

    result = LocalWorkspaceAgent(tmp_path).handle("このフォルダのPDF論文を読んで知識化して")

    assert result.provider == "local"
    assert "Local Paper Knowledge" in result.text
    assert "text_pdf" in result.text


def test_natural_language_explanation_auto_selects_embedded_workspace_locally(tmp_path: Path):
    research = tmp_path / "定理発見システム"
    research.mkdir()
    (research / "quantum_state.py").write_text(
        "# pre big ban validation\n"
        "def initial_quantum_state():\n"
        "    return vacuum_state(seed=0)\n",
        encoding="utf-8",
    )
    agent = LocalWorkspaceAgent()

    result = agent.handle(f'"{research}"でpre big ban検証で初期量子状態をどのように生成しているか解説して')

    assert agent.workspace == research.resolve()
    assert result.provider == "local"
    assert "quantum_state.py" in result.text
    assert "initial_quantum_state" in result.text
    assert "/read <path>" in result.text
    assert "ローカル調査結果" in result.text


def test_local_agent_without_workspace_does_not_ask_for_api_credentials(tmp_path: Path):
    result = LocalWorkspaceAgent().handle("hello")
    assert result.provider == "no-workspace"
    assert "ファイル一覧・検索・コード読取は行っていません" in result.text
    assert "API" not in result.text
    assert "sk-" not in result.text


def test_markdown_compiler_creates_readable_structured_tokens():
    tokens = compile_markdown(
        "# Result\n\n**Summary** with `run.py`\n- first item\n> a note\n\n```python\nprint('ok')\n```"
    )
    values = [value for value, _tag in tokens]
    tags = [tag for _value, tag in tokens]
    rendered = "".join(values)
    assert "#" not in rendered
    assert "**" not in rendered
    assert "`" not in rendered
    assert "Result" in rendered
    assert "print('ok')" in rendered
    assert "heading1" in tags
    assert "strong" in tags
    assert "inline_code" in tags
    assert "bullet" in tags
    assert "quote_marker" in tags
    assert "code" in tags


def test_mixed_japanese_english_request_keeps_semantic_keywords():
    assert LocalWorkspaceAgent._semantic_question("初期 quantum state の生成方法を explain") == "initial state"
    assert LocalWorkspaceAgent._is_explanation_request("where does rho data flow?")


def test_session_is_compressed_and_restored_without_secrets(tmp_path: Path):
    path = tmp_path / "conversation.json.gz"
    messages = [{"speaker": "You", "text": "日本語で run tests"},
                {"speaker": "Agent", "text": "**Passed**\n\n```text\n2 tests\n```"}]
    save_session(path, messages, tmp_path)
    workspace, restored = load_session(path)
    assert workspace == tmp_path
    assert restored == messages
    assert b"sk-" not in path.read_bytes()


def test_work_log_is_markdown_and_keeps_mixed_language_content(tmp_path: Path):
    path = save_work_log(tmp_path, [{"speaker": "You", "text": "日本語で run tests"}])
    content = path.read_text(encoding="utf-8")
    assert path.name == "work_log.md"
    assert "# Conscious Coding Agent" in content
    assert "日本語で run tests" in content
