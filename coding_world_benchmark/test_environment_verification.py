import json
from pathlib import Path

import pytest

from coding_world_benchmark.environment_verification import (
    REQUIRED_LAYERS,
    count_test_files,
    format_matrix_row,
    format_verification_report,
    load_verification,
    result_path,
)


def _payload(**overrides):
    payload = {
        "environment": "local_windows",
        "device": "satoshi3",
        "date": "2026-09-14",
        "repository": r"C:\\Users\\neko5\\Documents\\projects\\意識モデル",
        "test_file_count": 1,
        "python_version": "Python 3.12.1",
        "pytest_version": "pytest 8.0.0",
        "layers": {name: {"ok": True, "detail": name} for name in REQUIRED_LAYERS},
        "passed": 286,
        "failed": 0,
        "errors": 0,
        "exit_code": 0,
        "verified": True,
    }
    payload.update(overrides)
    return payload


def _write(root: Path, payload) -> Path:
    (root / "test_placeholder.py").write_text("def test_placeholder():\n    assert True\n", encoding="utf-8")
    path = result_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


# --------------------------------------------------------------------------
# absence of evidence is never verification
# --------------------------------------------------------------------------

def test_missing_result_file_stays_unverified(tmp_path: Path):
    result = load_verification(tmp_path)

    assert not result.verified
    assert "実機で" in result.reason
    assert "UNVERIFIED" in format_matrix_row(result)


def test_unreadable_result_file_stays_unverified(tmp_path: Path):
    path = result_path(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text("{ not json", encoding="utf-8")

    result = load_verification(tmp_path)

    assert not result.verified
    assert "読めません" in result.reason


# --------------------------------------------------------------------------
# a layered failure says which layer, and never counts as a pass
# --------------------------------------------------------------------------

@pytest.mark.parametrize("layer", REQUIRED_LAYERS)
def test_a_failed_layer_blocks_verification_and_is_named(layer: str, tmp_path: Path):
    layers = {name: {"ok": name != layer, "detail": name} for name in REQUIRED_LAYERS}
    _write(tmp_path, _payload(layers=layers, verified=False))

    result = load_verification(tmp_path)

    assert not result.verified
    assert result.failing_layer == layer
    assert layer in format_matrix_row(result)


def test_a_run_that_counted_no_tests_is_not_verified(tmp_path: Path):
    _write(tmp_path, _payload(passed=0, failed=0, errors=0))

    result = load_verification(tmp_path)

    assert not result.verified
    assert "1 件も数えられていません" in result.reason


def test_runner_saying_unverified_is_respected(tmp_path: Path):
    _write(tmp_path, _payload(verified=False))

    assert not load_verification(tmp_path).verified


# --------------------------------------------------------------------------
# a stale result is not evidence about the current suite
# --------------------------------------------------------------------------

def test_result_from_a_different_suite_size_is_rejected(tmp_path: Path):
    _write(tmp_path, _payload(test_file_count=999))

    result = load_verification(tmp_path)

    assert not result.verified
    assert "再実行が必要" in result.reason


def test_test_file_count_reflects_the_package(tmp_path: Path):
    (tmp_path / "test_a.py").write_text("", encoding="utf-8")
    (tmp_path / "test_b.py").write_text("", encoding="utf-8")
    (tmp_path / "module.py").write_text("", encoding="utf-8")

    assert count_test_files(tmp_path) == 2


# --------------------------------------------------------------------------
# a genuine run produces a matrix row
# --------------------------------------------------------------------------

def test_complete_run_is_verified_and_rendered(tmp_path: Path):
    _write(tmp_path, _payload())

    result = load_verification(tmp_path)
    row = format_matrix_row(result)

    assert result.verified
    assert result.total == 286
    assert "286 passed" in row
    assert "UNVERIFIED" not in row
    assert "satoshi3" in row


def test_failures_are_carried_into_the_row(tmp_path: Path):
    _write(tmp_path, _payload(passed=285, failed=1, exit_code=1))

    row = format_matrix_row(load_verification(tmp_path))

    assert "285 passed / 1 failed" in row


def test_report_tells_you_how_to_produce_the_evidence(tmp_path: Path):
    report = format_verification_report(load_verification(tmp_path), tmp_path)

    assert "UNVERIFIED" in report
    assert "verify_local_windows.ps1" in report


def test_the_shipped_script_exists_next_to_this_module():
    # The matrix promises a reproducible command; the file backing it must ship.
    assert (Path(__file__).resolve().parent / "verify_local_windows.ps1").is_file()


# --------------------------------------------------------------------------
# shrinking the Windows-only surface: the parsing half is platform neutral
# --------------------------------------------------------------------------

def test_windows_path_candidates_parse_without_windows():
    from coding_world_benchmark.coding_agent_app import windows_path_candidates

    message = '"C:\\Users\\neko5\\Documents\\projects\\意識モデル\\定理発見システム"でpre big ban検証を解説して'
    candidates = windows_path_candidates(message)

    # Longest first, then every shorter prefix, so the lookup can fall back.
    assert candidates[0] == "C:\\Users\\neko5\\Documents\\projects\\意識モデル\\定理発見システム"
    assert "C:\\Users\\neko5\\Documents\\projects\\意識モデル" in candidates
    assert "C:\\Users" in candidates
    assert len(candidates) == len(set(candidates))


def test_windows_path_candidates_ignore_prose_without_a_drive_letter():
    from coding_world_benchmark.coding_agent_app import windows_path_candidates

    assert windows_path_candidates("このリポジトリを要約して") == ()
    # A POSIX path is not a Windows workspace reference.
    assert windows_path_candidates("/tmp/pytest-of-root/x を見て") == ()
