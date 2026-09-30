"""Child-process capture that behaves the same on every platform.

``subprocess.run(..., text=True)`` decodes the child's pipes with the *locale*
codec.  On a Japanese Windows install that codec is cp932 while the child emits
UTF-8, so one non-cp932 byte raises ``UnicodeDecodeError`` inside the reader
thread; ``communicate`` then returns ``stdout=None`` and the caller dies with
``unsupported operand type(s) for +: 'NoneType' and 'str'``.  A UTF-8 container
never reproduces it -- this cost 31 failures on the first real Windows run.

Every child process in this repository therefore goes through ``run_text``:

* UTF-8 in both directions (the child is told via ``PYTHONIOENCODING``/``PYTHONUTF8``),
* ``errors="replace"``, so decoding can never raise no matter what bytes arrive,
* ``stdout``/``stderr`` guaranteed to be ``str``, never ``None``,
* ``sys.executable`` substituted for a bare ``python``, which is not guaranteed
  to exist on Windows (the Store alias stub exits non-zero).
"""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from typing import Iterable, Mapping, Sequence

PYTHON_NAMES = ("python", "python3", "py")


def resolve_command(command: Sequence[str]) -> list[str]:
    """Replace a bare interpreter name with the interpreter that is running.

    An explicit path is respected as given: only a bare name is ambiguous, and
    only a bare name can fail to exist (on Windows, ``python`` is often the
    Store alias stub).  Deciding on separators rather than on ``Path().name``
    keeps the answer identical on every platform.
    """
    parts = [str(item) for item in command]
    if not parts:
        return parts
    first = parts[0]
    if "/" in first or "\\" in first:
        return parts
    if first.lower().removesuffix(".exe") in PYTHON_NAMES:
        parts[0] = sys.executable
    return parts


def child_environment(extra: Mapping[str, str] | None = None) -> dict[str, str]:
    """Environment that makes the child speak UTF-8 regardless of system locale."""
    environment = dict(os.environ)
    environment["PYTHONIOENCODING"] = "utf-8"
    environment["PYTHONUTF8"] = "1"
    if extra:
        environment.update(extra)
    return environment


def run_text(
    command: Sequence[str],
    *,
    cwd: Path | str | None = None,
    timeout: float | None = None,
    check: bool = False,
    env: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a command and capture its output as text that is always a ``str``."""
    completed = subprocess.run(
        resolve_command(command),
        cwd=str(cwd) if cwd is not None else None,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=check,
        env=child_environment(env),
    )
    return subprocess.CompletedProcess(
        completed.args,
        completed.returncode,
        completed.stdout or "",
        completed.stderr or "",
    )


def combined_output(completed: subprocess.CompletedProcess, limit: int | None = None) -> str:
    """stdout + stderr, tolerating a runner that handed back ``None``."""
    text = (completed.stdout or "") + (completed.stderr or "")
    return text[-limit:] if limit else text


def compatible_runner(command: Iterable[str], cwd: Path | str | None = None, **kwargs):
    """Drop-in for injected ``runner=subprocess.run`` defaults.

    Accepts and ignores the keyword arguments those call sites pass
    (``capture_output``/``text``/``check``) so existing fakes keep the same
    signature, while the real run is always UTF-8 safe.
    """
    return run_text(list(command), cwd=cwd, timeout=kwargs.get("timeout"))
