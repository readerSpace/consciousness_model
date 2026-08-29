"""Japanese dialogue adapter for Coding World instructions and reports."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .coding_world_benchmark import CodingTask, TrialResult, run_trial
from .portable_japanese_dialogue import JapaneseDialogueEngine, SemanticFrame


Intent = Literal["MODIFY", "ROLLBACK", "STATUS", "EXPLAIN", "ASK_USER"]


@dataclass(frozen=True)
class CodingGoal:
    intent: Intent
    goal: str
    target: str
    constraints: tuple[str, ...]
    scope: tuple[str, ...]
    source_frame: SemanticFrame


@dataclass(frozen=True)
class ChangeRecord:
    identifier: str
    files_changed: tuple[str, ...]
    result: TrialResult


@dataclass(frozen=True)
class DialogueTurn:
    goal: CodingGoal
    text: str


class CodingDialogueController:
    """Keep dialogue as an interface while execution remains trace-driven."""

    def __init__(self, workspace_capacity: int = 8) -> None:
        self.language = JapaneseDialogueEngine(workspace_capacity=workspace_capacity)
        self.history: list[SemanticFrame] = []
        self.changes: list[ChangeRecord] = []

    def interpret(self, text: str) -> CodingGoal:
        frame = self.language.analyze(text)
        normalized = frame.text
        intent = self._intent(normalized)
        goal, target = self._goal_and_target(normalized)
        constraints = tuple(item for marker, item in (
            ("テスト", "existing_tests_must_pass"),
            ("互換", "preserve_api_compatibility"),
            ("APIだけ", "limit_to_api"),
        ) if marker in normalized)
        scope = ("api",) if "APIだけ" in normalized else ()
        if intent == "MODIFY" and "履歴" in normalized and "削除" in normalized and "消す" not in normalized:
            intent = "ASK_USER"
            goal = "clarify_deleted_user_history"
        self.history.append(frame)
        return CodingGoal(intent, goal, target, constraints, scope, frame)

    def respond(self, text: str) -> DialogueTurn:
        goal = self.interpret(text)
        if goal.intent == "ASK_USER":
            return DialogueTurn(goal, "仕様が2通り解釈できます。削除済みユーザーの履歴も消しますか？")
        if goal.intent == "ROLLBACK":
            return DialogueTurn(goal, self._rollback_message(goal))
        if goal.intent in {"STATUS", "EXPLAIN"}:
            return DialogueTurn(goal, self._report_message(goal.intent))
        return DialogueTurn(goal, self._instruction_message(goal))

    def execute(self, task: CodingTask, condition: str, work_root: Path) -> DialogueTurn:
        """Run the coding loop and retain a reportable, factual execution trace."""
        result = run_trial(task, condition, work_root)
        changed = ("app.py",) if result.files_modified else ()
        self.changes.append(ChangeRecord(f"change_{len(self.changes) + 1:03d}", changed, result))
        goal = CodingGoal(
            "STATUS", "report_execution", "repository", (), (),
            self.language.analyze("実行結果を報告して", "system"),
        )
        return DialogueTurn(goal, self._report_message("STATUS"))

    @staticmethod
    def _intent(text: str) -> Intent:
        if any(marker in text for marker in ("元に戻", "rollback", "ロールバック")):
            return "ROLLBACK"
        if any(marker in text for marker in ("して", "ください", "追加", "修正", "表示する")):
            return "MODIFY"
        if any(marker in text for marker in ("何を変更", "結果", "進捗", "状態")):
            return "STATUS"
        if any(marker in text for marker in ("なぜ", "理由", "問題はある")):
            return "EXPLAIN"
        return "MODIFY"

    @staticmethod
    def _goal_and_target(text: str) -> tuple[str, str]:
        if "ログイン" in text or "認証" in text:
            return "login_error_message", "authentication"
        if "ユーザー" in text and "削除" in text:
            return "session_cleanup", "user_deletion"
        if "pagination" in text or "ページ" in text:
            return "add_pagination", "rest_api"
        return "modify_repository", "repository"

    @staticmethod
    def _instruction_message(goal: CodingGoal) -> str:
        constraints = "、".join(goal.constraints) if goal.constraints else "制約なし"
        return f"目標を登録しました: {goal.goal}。対象は {goal.target}、制約は {constraints} です。"

    def _rollback_message(self, goal: CodingGoal) -> str:
        if not self.changes:
            return "元に戻す変更履歴がありません。"
        latest = self.changes[-1]
        scope = "、".join(goal.scope) if goal.scope else "全変更"
        return f"対象は {latest.identifier} です。ロールバック範囲は {scope} と解釈しました。"

    def _report_message(self, intent: Intent) -> str:
        if not self.changes:
            return "まだ実行結果はありません。"
        change = self.changes[-1]
        result = change.result
        files = "、".join(change.files_changed) or "なし"
        tests = "すべて通過" if result.task_success else "失敗あり"
        if intent == "EXPLAIN":
            cause = "セッションが残る実装" if result.task_success else "未解決の失敗"
            return f"直近の原因判断は {cause} です。失敗仮説は {result.failed_hypotheses} 件でした。"
        return (
            f"{change.identifier}: {files} を変更しました。可視・隠しテストは {tests} "
            f"(実行 {result.test_runs} 回、反復 {result.iterations} 回) です。"
        )