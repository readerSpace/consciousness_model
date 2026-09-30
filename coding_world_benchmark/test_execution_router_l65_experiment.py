from pathlib import Path
import subprocess
import sys

from coding_world_benchmark.execution_router_l65_experiment import (
    build_execution_plan,
    discover_execution_candidates,
    execute_plan,
    format_execution_result,
)


def _simulation_repo(tmp_path: Path) -> Path:
    (tmp_path / "README.md").write_text(
        "# Simulation\n\n"
        "Quick start:\n"
        "```powershell\n"
        "python .\\relativistic_tube_simulation.py --beta 0.9\n"
        "python .\\relativistic_tube_simulation.py --sweep\n"
        "```\n",
        encoding="utf-8",
    )
    (tmp_path / "relativistic_tube_simulation.py").write_text(
        "import argparse\n"
        "parser = argparse.ArgumentParser()\n"
        "parser.add_argument('--beta')\n"
        "parser.add_argument('--sweep', action='store_true')\n"
        "args = parser.parse_args()\n"
        "print('beta=' + str(args.beta))\n",
        encoding="utf-8",
    )
    return tmp_path


def test_readme_command_becomes_execution_candidate_not_summary(tmp_path: Path):
    root = _simulation_repo(tmp_path)
    candidates = discover_execution_candidates(root, "シミュレーションのpythonファイルを実行して")

    assert candidates
    assert candidates[0].script == "relativistic_tube_simulation.py"
    assert candidates[0].args == ("--beta", "0.9")
    assert candidates[0].evidence.startswith("README.md:")


def test_execution_plan_validates_command_inside_workspace(tmp_path: Path):
    root = _simulation_repo(tmp_path)
    plan = build_execution_plan(root, "シミュレーションのpythonファイルを実行して")

    assert plan.command == (sys.executable, "relativistic_tube_simulation.py", "--beta", "0.9")
    assert plan.cwd == str(root.resolve())


def test_execute_plan_captures_return_code_stdout_and_evidence(tmp_path: Path):
    root = _simulation_repo(tmp_path)
    plan = build_execution_plan(root, "シミュレーションのpythonファイルを実行して")
    result = execute_plan(plan)
    rendered = format_execution_result(result)

    assert result.return_code == 0
    assert "beta=0.9" in result.stdout
    assert result.evidence.startswith("README.md:")
    assert "【実行】" in rendered
    assert "【結果】" in rendered


def test_execute_plan_uses_injected_runner_for_attempt_rate(tmp_path: Path):
    root = _simulation_repo(tmp_path)
    plan = build_execution_plan(root, "READMEに書いてあるシミュレーションを実行して")
    calls = []

    def runner(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, "ok\n", "")

    result = execute_plan(plan, runner=runner)

    assert calls
    assert calls[0][0][1:] == ["relativistic_tube_simulation.py", "--beta", "0.9"]
    assert calls[0][1]["cwd"] == root
    assert result.return_code == 0
