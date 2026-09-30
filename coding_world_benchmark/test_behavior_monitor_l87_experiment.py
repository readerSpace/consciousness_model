from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from coding_world_benchmark.behavior_monitor_l87_experiment import (
    POSTCONDITIONS,
    BehaviorEvent,
    ExpectedBehavior,
    MetricBound,
    audit_request,
    compare_behavior,
    expected_behavior_for,
    format_behavior_audit,
    observe,
    record_execution,
)
from coding_world_benchmark.coding_agent_app import LocalWorkspaceAgent
from coding_world_benchmark.structured_repository_report_l62_experiment import RequestType
from coding_world_benchmark.workspace_operations_l75_experiment import is_move_directories_request


def _workspace(root: Path) -> Path:
    (root / "SU2_validation").mkdir()
    (root / "SU2_validation" / "run.py").write_text("print('su2 ok')\n", encoding="utf-8")
    (root / "SU3_validation").mkdir()
    (root / "SU3_validation" / "run.py").write_text("print('su3 ok')\n", encoding="utf-8")
    (root / "simulation.py").write_text(
        "def load_config():\n    return {'steps': 3}\n\n\n"
        "def step(state):\n    return state + 1\n\n\n"
        "def run_simulation():\n    config = load_config()\n    state = 0\n"
        "    for _ in range(config['steps']):\n        state = step(state)\n    return state\n\n\n"
        "if __name__ == '__main__':\n    print(run_simulation())\n",
        encoding="utf-8",
    )
    (root / "README.md").write_text("# demo\n\n```powershell\npython simulation.py\n```\n", encoding="utf-8")
    return root


# --------------------------------------------------------------------------
# the contract is data, and the trace observes real side effects
# --------------------------------------------------------------------------

def test_expected_behavior_is_declarative_and_resolvable():
    expected = expected_behavior_for("SU2とSU3の検証フォルダをまとめて移動して")

    assert expected.intent is RequestType.MOVE_DIRECTORIES
    assert BehaviorEvent.PATH_MOVED in expected.required_events
    assert set(expected.postconditions) <= set(POSTCONDITIONS)
    assert expected.rationale


def test_algorithm_summary_contract_demands_stages_not_search_hits():
    expected = expected_behavior_for("simulation.py のアルゴリズムを要約して")

    bounds = {str(item) for item in expected.metric_bounds}
    assert "algorithm_stage_count >= 2.0" in bounds
    assert "raw_match_ratio < 0.5" in bounds


def test_trace_records_side_effects_and_restores_every_patch(tmp_path: Path):
    real_run, real_move = subprocess.run, shutil.move

    with record_execution(tmp_path) as trace:
        (tmp_path / "made").mkdir()
        (tmp_path / "made" / "x.txt").write_text("x", encoding="utf-8")
        shutil.move(str(tmp_path / "made"), str(tmp_path / "moved"))
        # sys.executable, not "python": a bare python need not exist on Windows.
        subprocess.run([sys.executable, "-c", "print(1)"], capture_output=True,
                       encoding="utf-8", errors="replace", check=False)

    assert trace.count(BehaviorEvent.DIRECTORY_CREATED) == 1
    assert trace.count(BehaviorEvent.PATH_MOVED) == 1
    assert trace.count(BehaviorEvent.SUBPROCESS_STARTED) == 1
    assert "moved/" in trace.created
    assert subprocess.run is real_run and shutil.move is real_move


def test_trace_restores_patches_even_when_the_run_raises(tmp_path: Path):
    real_run = subprocess.run

    with pytest.raises(RuntimeError):
        with record_execution(tmp_path):
            (tmp_path / "partial").mkdir()
            raise RuntimeError("boom")

    assert subprocess.run is real_run


# --------------------------------------------------------------------------
# implementation-independent detection: the handler is a black box
# --------------------------------------------------------------------------

def test_execute_that_only_searches_is_reported_as_behavioural_failure(tmp_path: Path):
    def search_only(_message: str) -> str:
        return "ローカル要約\n- simulation.py (12行): 3: def run_simulation():"

    audit = audit_request(tmp_path, "シミュレーションを実行して", search_only)

    assert audit.intent is RequestType.EXECUTE
    assert not audit.consistent
    assert audit.mismatch.missing_events == ("SUBPROCESS_STARTED",)
    assert "return_code_recorded" in audit.mismatch.failed_postconditions
    assert audit.goal_satisfaction is not None
    assert not audit.goal_satisfaction.achieved


def test_move_that_only_claims_success_is_reported(tmp_path: Path):
    _workspace(tmp_path)

    def claims_without_doing(_message: str) -> str:
        return "## Workspace Operation Report\n\n【検証】\n- 成功: `2/2`\n- 移動しました。"

    audit = audit_request(tmp_path, "SU2とSU3の検証フォルダをまとめて移動して", claims_without_doing)

    assert not audit.consistent
    assert audit.mismatch.missing_events == ("PATH_MOVED",)
    assert set(audit.mismatch.failed_postconditions) == {"destination_present", "source_absent"}


def test_summary_made_of_raw_search_hits_violates_the_metric_bound(tmp_path: Path):
    def raw_hits(_message: str) -> str:
        return "\n".join(f"simulation.py:{line}: def step(state):" for line in range(1, 9))

    audit = audit_request(tmp_path, "このリポジトリを要約して", raw_hits)

    assert not audit.consistent
    assert any("raw_match_ratio" in item for item in audit.mismatch.metric_violations)


def test_a_handler_that_does_the_work_passes(tmp_path: Path):
    _workspace(tmp_path)

    def really_moves(_message: str) -> str:
        destination = tmp_path / "su2_su3"
        destination.mkdir()
        for name in ("SU2_validation", "SU3_validation"):
            shutil.move(str(tmp_path / name), str(destination / name))
        return "移動しました: `su2_su3`"

    audit = audit_request(tmp_path, "SU2とSU3の検証フォルダをまとめて移動して", really_moves)

    assert audit.consistent
    assert audit.observed.count(BehaviorEvent.PATH_MOVED) == 2
    assert audit.goal_satisfaction.achieved


def test_create_goal_that_only_searches_is_goal_failure(tmp_path: Path):
    def search_only(_message: str) -> str:
        return "ローカル要約（検索語一致なし・ファイル概観）\n候補ファイル: 0件"

    audit = audit_request(tmp_path, "一様磁場中で荷電粒子が運動するシミュレーションを作成して", search_only)

    assert audit.goal_satisfaction is not None
    assert not audit.goal_satisfaction.achieved
    assert "FILE_WRITTEN >= 1" in audit.goal_satisfaction.missing_observations


def test_create_goal_that_writes_file_is_goal_success(tmp_path: Path):
    def writes_file(_message: str) -> str:
        (tmp_path / "charged_particle_uniform_B.py").write_text("print('ok')\n", encoding="utf-8")
        return "生成ファイル: `charged_particle_uniform_B.py`"

    audit = audit_request(tmp_path, "一様磁場中で荷電粒子が運動するシミュレーションを作成して", writes_file)

    assert audit.goal_satisfaction.achieved


def test_comparator_reports_every_failing_clause_at_once():
    expected = ExpectedBehavior(
        RequestType.EXECUTE,
        required_events=(BehaviorEvent.SUBPROCESS_STARTED,),
        prohibited_events=(BehaviorEvent.PATH_MOVED,),
        postconditions=("return_code_recorded",),
        metric_bounds=(MetricBound("raw_match_ratio", "<", 0.5),),
    )

    class _Trace:
        events = [BehaviorEvent.PATH_MOVED]

    observed = observe(Path("."), "実行して", "a.py:1: hit", _Trace(), RequestType.EXECUTE)
    mismatch = compare_behavior(expected, observed)

    assert mismatch.missing_events == ("SUBPROCESS_STARTED",)
    assert mismatch.prohibited_events == ("PATH_MOVED",)
    assert mismatch.failed_postconditions == ("return_code_recorded",)
    assert mismatch.metric_violations


# --------------------------------------------------------------------------
# auditing the shipped agent, and the bug that audit found
# --------------------------------------------------------------------------

def test_japanese_particles_no_longer_hide_the_move_target():
    # Found by the L8.7 audit: `\bSU2\b` never matched in "SU2とSU3の…" because a
    # Japanese particle is a word character, so move requests silently fell
    # through to the validation-memory route without moving anything.
    assert is_move_directories_request("SU2とSU3の検証フォルダをまとめて移動して")
    assert is_move_directories_request("SU2 and SU3 validation folders, move them")
    assert not is_move_directories_request("SU2の検証結果を要約して")


@pytest.mark.parametrize("request_text", [
    "シミュレーションを実行して",
    "simulation.py のアルゴリズムを要約して",
    "このリポジトリを要約して",
    "SU2とSU3の検証フォルダをまとめて移動して",
])
def test_shipped_agent_satisfies_its_behavioural_contract(request_text: str, tmp_path: Path):
    root = _workspace(tmp_path)
    agent = LocalWorkspaceAgent(root)

    audit = audit_request(root, request_text, lambda message: agent.handle(message).text)

    assert audit.contracted, "この intent には契約が定義されていない"
    assert audit.consistent, audit.mismatch.statement


def test_audit_report_names_the_contract_rationale(tmp_path: Path):
    audit = audit_request(tmp_path, "テストを実行して", lambda _message: "検索しました")

    report = format_behavior_audit((audit,))

    assert "不一致" in report
    assert "Goal未達" in report
    assert audit.expected.rationale in report
