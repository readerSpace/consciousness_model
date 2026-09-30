"""Per-environment test verification: evidence in, matrix row out.

`TEST_MATRIX.md` exists so an environment-dependent failure is never counted as
a pass.  This module removes the remaining hand-written step: a row is marked
verified only from a result file that an actual run produced, and it refuses
for a stated reason otherwise.

The producer for Windows is ``verify_local_windows.ps1``, which checks the
execution path in layers (shell -> workspace -> python -> pytest -> suite) so a
break says which layer failed.  Any other runner may write the same shape.

Note on execution paths: the desktop device shell runs a *Linux VM* with the
project folder mounted, so a suite run there is not a Windows run.  Verifying
`local_windows` requires a Windows-native interpreter -- PowerShell on the
desktop -- which is what the script above drives.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import json
from pathlib import Path

VERIFICATION_DIRECTORY = "verification"
REQUIRED_LAYERS = ("shell", "workspace", "python", "pytest", "suite")


@dataclass(frozen=True)
class EnvironmentResult:
    environment: str
    verified: bool
    passed: int
    failed: int
    errors: int
    date: str
    device: str
    python_version: str
    layers: dict[str, bool]
    layer_details: dict[str, str]
    test_file_count: int
    exit_code: int | None
    reason: str

    @property
    def total(self) -> int:
        return self.passed + self.failed + self.errors

    @property
    def failing_layer(self) -> str:
        return next((name for name in REQUIRED_LAYERS if not self.layers.get(name, False)), "")


UNVERIFIED = EnvironmentResult("local_windows", False, 0, 0, 0, "—", "", "", {}, {}, 0, None,
                               "結果ファイルがありません。実機で verify_local_windows.ps1 を実行してください。")


def result_path(package_root: Path, environment: str = "local_windows") -> Path:
    return Path(package_root) / VERIFICATION_DIRECTORY / f"{environment}_result.json"


def count_test_files(package_root: Path) -> int:
    return len([path for path in Path(package_root).glob("test_*.py") if path.is_file()])


def load_verification(package_root: Path, environment: str = "local_windows",
                      today: date | None = None) -> EnvironmentResult:
    """Read a result file and decide whether it may be trusted.

    Trust requires: the file parses, every layer passed, the run actually
    counted tests, and the suite it ran matches the suite on disk now.
    """
    path = result_path(package_root, environment)
    if not path.is_file():
        return UNVERIFIED
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return EnvironmentResult(environment, False, 0, 0, 0, "—", "", "", {}, {}, 0, None,
                                 f"結果ファイルを読めません: {error}")

    raw_layers = payload.get("layers") or {}
    layers = {name: bool(item.get("ok")) for name, item in raw_layers.items() if isinstance(item, dict)}
    details = {name: str(item.get("detail", "")) for name, item in raw_layers.items() if isinstance(item, dict)}
    passed = int(payload.get("passed") or 0)
    failed = int(payload.get("failed") or 0)
    errors = int(payload.get("errors") or 0)
    recorded_files = int(payload.get("test_file_count") or 0)
    current_files = count_test_files(package_root)

    reasons: list[str] = []
    missing = [name for name in REQUIRED_LAYERS if name not in layers]
    if missing:
        reasons.append("記録されていない層: " + ", ".join(missing))
    broken = [name for name in REQUIRED_LAYERS if name in layers and not layers[name]]
    if broken:
        reasons.append("失敗した層: " + ", ".join(broken))
    if passed + failed + errors == 0:
        reasons.append("テストが 1 件も数えられていません")
    if not payload.get("verified"):
        reasons.append("実行側が verified=false を記録しています")
    if recorded_files and current_files and recorded_files != current_files:
        reasons.append(f"記録時のテストファイル数 {recorded_files} が現在の {current_files} と異なります（再実行が必要）")

    verified = not reasons
    return EnvironmentResult(
        payload.get("environment", environment), verified, passed, failed, errors,
        str(payload.get("date", "—")), str(payload.get("device", "")),
        str(payload.get("python_version", "")), layers, details, recorded_files,
        payload.get("exit_code"),
        "すべての層が通過し、テストが実行されました。" if verified else " / ".join(reasons),
    )


def format_matrix_row(result: EnvironmentResult) -> str:
    if not result.verified:
        note = result.reason
        if result.failing_layer:
            note = f"層 `{result.failing_layer}` で停止 — {note}"
        return f"| {result.environment} | {result.date} | **UNVERIFIED** | {note} |"
    outcome = f"{result.passed} passed"
    if result.failed:
        outcome += f" / {result.failed} failed"
    if result.errors:
        outcome += f" / {result.errors} errors"
    device = f"{result.device} " if result.device else ""
    return f"| {result.environment} | {result.date} | {outcome} | {device}{result.python_version} |"


def format_verification_report(result: EnvironmentResult, package_root: Path | None = None) -> str:
    lines = [f"## Environment Verification — {result.environment}", "",
             f"【判定】`{'verified' if result.verified else 'UNVERIFIED'}`",
             f"【理由】{result.reason}", "", "【層】"]
    for name in REQUIRED_LAYERS:
        if name not in result.layers:
            lines.append(f"- `{name}`: 記録なし")
            continue
        mark = "OK" if result.layers[name] else "NG"
        lines.append(f"- `{name}`: {mark} — {result.layer_details.get(name, '')}")
    lines.extend(["", "【結果】", f"- passed: `{result.passed}` / failed: `{result.failed}` / errors: `{result.errors}`",
                  f"- python: `{result.python_version or '不明'}`",
                  f"- exit code: `{result.exit_code}`"])
    if not result.verified and package_root is not None:
        lines.extend(["", "【実行方法】", "```powershell",
                      f"powershell -ExecutionPolicy Bypass -File .\\{Path(package_root).name}\\verify_local_windows.ps1",
                      "```"])
    return "\n".join(lines)


def main() -> None:  # pragma: no cover - manual entry point
    package_root = Path(__file__).resolve().parent
    result = load_verification(package_root)
    print(format_verification_report(result, package_root))
    print()
    print(format_matrix_row(result))


if __name__ == "__main__":  # pragma: no cover
    main()
