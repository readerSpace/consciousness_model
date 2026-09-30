from pathlib import Path
import subprocess
import sys

import pytest

from coding_world_benchmark.process_execution import (
    child_environment,
    combined_output,
    compatible_runner,
    resolve_command,
    run_text,
)


# --------------------------------------------------------------------------
# a bare "python" is not guaranteed to exist on Windows
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["python", "python3", "py", "PYTHON.EXE"])
def test_bare_interpreter_names_resolve_to_the_running_interpreter(name: str):
    assert resolve_command([name, "-c", "pass"]) == [sys.executable, "-c", "pass"]


@pytest.mark.parametrize("name", ["C:\\bin\\python.exe", "/usr/bin/python3", "./python"])
def test_an_explicit_interpreter_path_is_respected(name: str):
    # Same answer on every platform: only a bare name is ambiguous.
    assert resolve_command([name, "-c", "pass"])[0] == name


def test_other_commands_are_left_alone():
    assert resolve_command(["pytest", "-q"]) == ["pytest", "-q"]
    assert resolve_command([]) == []


def test_child_is_told_to_speak_utf8():
    environment = child_environment()

    assert environment["PYTHONIOENCODING"] == "utf-8"
    assert environment["PYTHONUTF8"] == "1"
    assert child_environment({"EXTRA": "1"})["EXTRA"] == "1"


# --------------------------------------------------------------------------
# the cp932 regression: decoding must never raise, output must never be None
# --------------------------------------------------------------------------

def test_undecodable_bytes_do_not_crash_the_reader(tmp_path: Path):
    # Reproduces the real Windows failure shape without needing a cp932 locale:
    # the child emits bytes the codec cannot map, which used to kill the reader
    # thread and leave stdout=None, so the caller died on `None + str`.
    script = tmp_path / "noisy.py"
    script.write_text(
        "import sys\n"
        "sys.stdout.buffer.write(b'before \\x86\\xff\\xfe after')\n"
        "sys.stdout.buffer.flush()\n",
        encoding="utf-8",
    )

    completed = run_text([sys.executable, str(script)])

    assert isinstance(completed.stdout, str)
    assert isinstance(completed.stderr, str)
    assert "before" in completed.stdout and "after" in completed.stdout
    assert completed.returncode == 0


def test_japanese_output_survives_the_round_trip(tmp_path: Path):
    script = tmp_path / "japanese.py"
    script.write_text("print('【再利用された知識】 検証 SU2')\n", encoding="utf-8")

    completed = run_text([sys.executable, str(script)])

    assert "【再利用された知識】" in completed.stdout
    assert "SU2" in completed.stdout


def test_failing_child_still_yields_text(tmp_path: Path):
    script = tmp_path / "boom.py"
    script.write_text("raise SystemExit('失敗しました')\n", encoding="utf-8")

    completed = run_text([sys.executable, str(script)])

    assert completed.returncode != 0
    assert "失敗しました" in combined_output(completed)


def test_combined_output_tolerates_a_runner_that_returned_none():
    completed = subprocess.CompletedProcess(["x"], 1, None, None)

    assert combined_output(completed) == ""
    assert combined_output(subprocess.CompletedProcess(["x"], 0, "abcdef", ""), 3) == "def"


def test_compatible_runner_accepts_the_legacy_keyword_arguments(tmp_path: Path):
    script = tmp_path / "ok.py"
    script.write_text("print('ok')\n", encoding="utf-8")

    completed = compatible_runner([sys.executable, str(script)], cwd=tmp_path,
                                  capture_output=True, text=True, check=False, timeout=30)

    assert completed.stdout.strip() == "ok"


def test_cwd_is_honoured(tmp_path: Path):
    (tmp_path / "marker.txt").write_text("x", encoding="utf-8")
    script = tmp_path / "where.py"
    script.write_text("from pathlib import Path\nprint(Path('marker.txt').is_file())\n", encoding="utf-8")

    assert run_text([sys.executable, str(script)], cwd=tmp_path).stdout.strip() == "True"
