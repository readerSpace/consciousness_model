from pathlib import Path

from coding_world_benchmark.coding_dialogue import CodingDialogueController
from coding_world_benchmark.coding_world_benchmark import generate_tasks


def test_instruction_is_translated_into_a_coding_goal():
    controller = CodingDialogueController()

    turn = controller.respond("ログイン失敗時に理由を表示するようにして。既存テストは通して")

    assert turn.goal.intent == "MODIFY"
    assert turn.goal.goal == "login_error_message"
    assert turn.goal.target == "authentication"
    assert "existing_tests_must_pass" in turn.goal.constraints


def test_ambiguous_deletion_request_asks_for_a_specification():
    controller = CodingDialogueController()

    turn = controller.respond("ユーザー削除時の履歴をどうするか対応して")

    assert turn.goal.intent == "ASK_USER"
    assert "履歴も消しますか" in turn.text


def test_execution_report_uses_the_actual_trial_result(tmp_path: Path):
    controller = CodingDialogueController()

    report = controller.execute(generate_tasks(1)[0], "C_full", tmp_path)
    explanation = controller.respond("なぜその方法にした？")

    assert "app.py" in report.text
    assert "すべて通過" in report.text
    assert "セッションが残る実装" in explanation.text