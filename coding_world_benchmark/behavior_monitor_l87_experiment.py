"""L8.7: audit what the agent actually did against what the intent required.

L8.6.1 measured that behavioural detection is the one capability that transfers
to unseen faults, because it runs a probe instead of matching a code pattern.
This module generalizes that: instead of one probe per known fault, the intent
itself states what must happen, and the run is observed and compared.

    request -> RequestType (existing L6.2 intent) -> ExpectedBehavior
            -> execution trace -> ObservedBehavior -> comparator -> mismatch

The comparison never reads the agent's source.  Events come from the standard
library calls any implementation must make to do the work (subprocess, moves,
directory creation) and from a before/after snapshot of the workspace, so an
agent that answers "I moved it" without moving anything fails the audit even
when no exception was raised.

ExpectedBehavior is data, not code: postconditions and metric bounds are names
resolved through registries, so an expectation can be written down, reviewed and
stored rather than hidden inside an assertion.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import pathlib
import re
import shutil
import subprocess
import sys
from typing import Callable, Iterator

from .structured_repository_report_l62_experiment import RequestType, classify_request
from .goal_semantics_l89_experiment import (
    GoalSatisfaction,
    contract_from_goal,
    evaluate_goal_contract,
    extract_goal,
    observe_goal_behavior,
)


class BehaviorEvent(Enum):
    SUBPROCESS_STARTED = "SUBPROCESS_STARTED"
    DIRECTORY_CREATED = "DIRECTORY_CREATED"
    PATH_MOVED = "PATH_MOVED"
    FILE_WRITTEN = "FILE_WRITTEN"
    REPOSITORY_INDEXED = "REPOSITORY_INDEXED"


# --------------------------------------------------------------------------
# execution trace
# --------------------------------------------------------------------------

@dataclass
class ExecutionTrace:
    events: list[BehaviorEvent] = field(default_factory=list)
    details: list[str] = field(default_factory=list)

    def record(self, event: BehaviorEvent, detail: str = "") -> None:
        self.events.append(event)
        self.details.append(f"{event.value}: {detail}" if detail else event.value)

    def count(self, event: BehaviorEvent) -> int:
        return self.events.count(event)


def _patch_everywhere(module_prefix: str, attribute: str, original, replacement) -> list[tuple[object, str]]:
    """Rebind an attribute in its own module and in every module that imported it."""
    patched: list[tuple[object, str]] = []
    for module in list(sys.modules.values()):
        name = getattr(module, "__name__", "")
        if not name.startswith(module_prefix):
            continue
        if getattr(module, attribute, None) is original:
            setattr(module, attribute, replacement)
            patched.append((module, attribute))
    return patched


def _patch_defaults(module_prefix: str, original, replacement) -> list[tuple[object, str, object]]:
    """Rebind captured defaults such as ``runner=subprocess.run``.

    A default argument binds at definition time, so patching the module
    attribute alone would miss it and the monitor would report a subprocess that
    did happen as missing.
    """
    restore: list[tuple[object, str, object]] = []
    for module in list(sys.modules.values()):
        if not getattr(module, "__name__", "").startswith(module_prefix):
            continue
        for value in list(vars(module).values()):
            if not callable(value) or not hasattr(value, "__kwdefaults__"):
                continue
            for slot, defaults in (("__kwdefaults__", value.__kwdefaults__), ("__defaults__", value.__defaults__)):
                if not defaults:
                    continue
                if slot == "__kwdefaults__":
                    for key, item in defaults.items():
                        if item is original:
                            restore.append((value, f"kw:{key}", original))
                            defaults[key] = replacement
                elif original in defaults:
                    restore.append((value, "pos", defaults))
                    value.__defaults__ = tuple(replacement if item is original else item for item in defaults)
    return restore


def _restore_defaults(restore: list[tuple[object, str, object]]) -> None:
    for target, slot, previous in restore:
        if slot.startswith("kw:"):
            target.__kwdefaults__[slot[3:]] = previous
        else:
            target.__defaults__ = previous


def _snapshot(root: Path) -> set[str]:
    excluded = {".git", "__pycache__", ".pytest_cache", ".venv"}
    return {
        str(path.relative_to(root)).replace("\\", "/") + ("/" if path.is_dir() else "")
        for path in root.rglob("*")
        if not excluded.intersection(path.parts)
    }


@contextmanager
def record_execution(root: Path, package: str = "coding_world_benchmark") -> Iterator[ExecutionTrace]:
    """Observe a run through the stdlib calls any implementation has to make."""
    trace = ExecutionTrace()
    root = Path(root)
    before = _snapshot(root) if root.is_dir() else set()

    real_run = subprocess.run
    real_move = shutil.move
    real_mkdir = pathlib.Path.mkdir
    real_write = pathlib.Path.write_text

    def run(*args, **kwargs):
        trace.record(BehaviorEvent.SUBPROCESS_STARTED, " ".join(map(str, args[0])) if args and args[0] else "")
        return real_run(*args, **kwargs)

    def move(source, destination, *args, **kwargs):
        trace.record(BehaviorEvent.PATH_MOVED, f"{source} -> {destination}")
        return real_move(source, destination, *args, **kwargs)

    def mkdir(self, *args, **kwargs):
        trace.record(BehaviorEvent.DIRECTORY_CREATED, str(self))
        return real_mkdir(self, *args, **kwargs)

    def write_text(self, *args, **kwargs):
        trace.record(BehaviorEvent.FILE_WRITTEN, str(self))
        return real_write(self, *args, **kwargs)

    from . import repository_semantic_l58_experiment as semantic

    real_index = semantic.index_repository

    def index_repository(*args, **kwargs):
        trace.record(BehaviorEvent.REPOSITORY_INDEXED)
        return real_index(*args, **kwargs)

    subprocess.run = run
    shutil.move = move
    pathlib.Path.mkdir = mkdir
    pathlib.Path.write_text = write_text
    patched = _patch_everywhere(package, "index_repository", real_index, index_repository)
    patched += _patch_everywhere(package, "run", real_run, run)
    defaults = _patch_defaults(package, real_run, run)
    semantic.index_repository = index_repository
    try:
        yield trace
    finally:
        subprocess.run = real_run
        shutil.move = real_move
        pathlib.Path.mkdir = real_mkdir
        pathlib.Path.write_text = real_write
        semantic.index_repository = real_index
        _restore_defaults(defaults)
        for module, attribute in patched:
            setattr(module, attribute, real_index if attribute == "index_repository" else real_run)
        # Snapshot inside the finally so a failing run is still audited on its
        # side effects rather than losing them.
        after = _snapshot(root) if root.is_dir() else set()
        trace.created = tuple(sorted(after - before))
        trace.removed = tuple(sorted(before - after))


# --------------------------------------------------------------------------
# observed behavior
# --------------------------------------------------------------------------

_RAW_MATCH = re.compile(r":\s*\d+\s*:")
_STAGE_LINE = re.compile(r"^\s*\d+\.\s+\S+.*:", re.M)


@dataclass(frozen=True)
class ObservedBehavior:
    intent: RequestType
    request: str
    response: str
    events: tuple[BehaviorEvent, ...]
    created_paths: tuple[str, ...]
    removed_paths: tuple[str, ...]
    metrics: dict[str, float]

    def count(self, event: BehaviorEvent) -> int:
        return self.events.count(event)


def _metrics(response: str) -> dict[str, float]:
    lines = [line for line in response.splitlines() if line.strip()]
    raw = sum(1 for line in lines if _RAW_MATCH.search(line))
    return {
        "response_lines": float(len(lines)),
        "raw_match_ratio": round(raw / len(lines), 3) if lines else 0.0,
        "algorithm_stage_count": float(len(_STAGE_LINE.findall(response))),
        "structured_section_count": float(len(re.findall(r"【[^】]+】", response))),
    }


def observe(root: Path, request: str, response: str, trace: ExecutionTrace,
            intent: RequestType | None = None) -> ObservedBehavior:
    return ObservedBehavior(
        intent or classify_request(request).request_type,
        request,
        response,
        tuple(trace.events),
        tuple(getattr(trace, "created", ())),
        tuple(getattr(trace, "removed", ())),
        _metrics(response),
    )


# --------------------------------------------------------------------------
# expected behavior (data, resolved through registries)
# --------------------------------------------------------------------------

Postcondition = Callable[[ObservedBehavior], bool]


def _return_code_recorded(observed: ObservedBehavior) -> bool:
    return bool(re.search(r"return code\s*[`:]?\s*-?\d+", observed.response, re.I))


def _destination_present(observed: ObservedBehavior) -> bool:
    return any(item.endswith("/") for item in observed.created_paths)


def _source_absent(observed: ObservedBehavior) -> bool:
    return bool(observed.removed_paths)


def _names_a_concrete_target(observed: ObservedBehavior) -> bool:
    """A filename or a quoted symbol.  Markdown formatting must not decide this."""
    return bool(re.search(r"[\w./\\-]+\.[A-Za-z]{1,5}\b", observed.response)
                or re.search(r"`[^`]+`", observed.response))

POSTCONDITIONS: dict[str, Postcondition] = {
    "return_code_recorded": _return_code_recorded,
    "destination_present": _destination_present,
    "source_absent": _source_absent,
    "names_a_concrete_target": _names_a_concrete_target,
}


@dataclass(frozen=True)
class MetricBound:
    metric: str
    operator: str
    value: float

    def satisfied_by(self, observed: ObservedBehavior) -> bool:
        actual = observed.metrics.get(self.metric)
        if actual is None:
            return False
        return {
            "<": actual < self.value,
            "<=": actual <= self.value,
            ">": actual > self.value,
            ">=": actual >= self.value,
        }[self.operator]

    def __str__(self) -> str:
        return f"{self.metric} {self.operator} {self.value}"


@dataclass(frozen=True)
class ExpectedBehavior:
    intent: RequestType
    required_events: tuple[BehaviorEvent, ...] = ()
    prohibited_events: tuple[BehaviorEvent, ...] = ()
    postconditions: tuple[str, ...] = ()
    metric_bounds: tuple[MetricBound, ...] = ()
    rationale: str = ""


# Written from what each intent means for the person asking, not from what the
# current implementation happens to do.
EXPECTED_BEHAVIOR: dict[RequestType, ExpectedBehavior] = {
    RequestType.EXECUTE: ExpectedBehavior(
        RequestType.EXECUTE,
        required_events=(BehaviorEvent.SUBPROCESS_STARTED,),
        postconditions=("return_code_recorded",),
        rationale="実行を求められた以上、プロセスが起動し結果が報告されなければならない。",
    ),
    RequestType.TEST: ExpectedBehavior(
        RequestType.TEST,
        required_events=(BehaviorEvent.SUBPROCESS_STARTED,),
        postconditions=("return_code_recorded",),
        rationale="テスト実行要求は検索では満たせない。",
    ),
    RequestType.MOVE_DIRECTORIES: ExpectedBehavior(
        RequestType.MOVE_DIRECTORIES,
        required_events=(BehaviorEvent.PATH_MOVED,),
        postconditions=("destination_present", "source_absent"),
        rationale="移動要求は、移動元が消え移動先が存在して初めて満たされる。",
    ),
    RequestType.DELETE_DIRECTORY: ExpectedBehavior(
        RequestType.DELETE_DIRECTORY,
        required_events=(BehaviorEvent.PATH_MOVED,),
        postconditions=("source_absent",),
        rationale="削除要求は、対象が元の場所から消えて初めて満たされる（実体は trash へ移動）。",
    ),
    RequestType.SUMMARIZE: ExpectedBehavior(
        RequestType.SUMMARIZE,
        postconditions=("names_a_concrete_target",),
        metric_bounds=(MetricBound("raw_match_ratio", "<", 0.5),),
        rationale="要約要求に対して検索ヒットの羅列を返すのは回答になっていない。",
    ),
}

ALGORITHM_SUMMARY = ExpectedBehavior(
    RequestType.SUMMARIZE,
    required_events=(BehaviorEvent.REPOSITORY_INDEXED,),
    postconditions=("names_a_concrete_target",),
    metric_bounds=(MetricBound("algorithm_stage_count", ">=", 2.0),
                   MetricBound("raw_match_ratio", "<", 0.5)),
    rationale="アルゴリズム要約は Context Graph を読み、段階を 2 つ以上示す必要がある。",
)


def expected_behavior_for(request: str, intent: RequestType | None = None) -> ExpectedBehavior | None:
    """The contract the request implies.  None means no contract is defined yet."""
    from .algorithm_summary_l80_experiment import is_algorithm_summary_request

    intent = intent or classify_request(request).request_type
    if is_algorithm_summary_request(request):
        return ALGORITHM_SUMMARY
    return EXPECTED_BEHAVIOR.get(intent)


# --------------------------------------------------------------------------
# comparator
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class BehaviorMismatch:
    intent: RequestType
    missing_events: tuple[str, ...]
    prohibited_events: tuple[str, ...]
    failed_postconditions: tuple[str, ...]
    metric_violations: tuple[str, ...]
    statement: str


@dataclass(frozen=True)
class BehaviorAudit:
    request: str
    intent: RequestType
    expected: ExpectedBehavior | None
    observed: ObservedBehavior
    mismatch: BehaviorMismatch | None
    goal_satisfaction: GoalSatisfaction | None = None

    @property
    def consistent(self) -> bool:
        return self.mismatch is None

    @property
    def contracted(self) -> bool:
        return self.expected is not None


def compare_behavior(expected: ExpectedBehavior, observed: ObservedBehavior) -> BehaviorMismatch | None:
    missing = tuple(event.value for event in expected.required_events if observed.count(event) == 0)
    prohibited = tuple(event.value for event in expected.prohibited_events if observed.count(event) > 0)
    failed = tuple(name for name in expected.postconditions
                   if not POSTCONDITIONS[name](observed))
    violations = tuple(f"{bound} (actual {observed.metrics.get(bound.metric)})"
                       for bound in expected.metric_bounds if not bound.satisfied_by(observed))
    if not (missing or prohibited or failed or violations):
        return None
    parts = []
    if missing:
        parts.append("発生しなかった必須イベント: " + ", ".join(missing))
    if prohibited:
        parts.append("禁止イベントが発生: " + ", ".join(prohibited))
    if failed:
        parts.append("未達の事後条件: " + ", ".join(failed))
    if violations:
        parts.append("指標違反: " + ", ".join(violations))
    statement = f"{expected.intent.value} が要求されたが、" + " / ".join(parts)
    return BehaviorMismatch(expected.intent, missing, prohibited, failed, violations, statement)


def audit_request(root: Path, request: str, handler: Callable[[str], str]) -> BehaviorAudit:
    """Run one request through ``handler`` and audit the behaviour it produced."""
    intent = classify_request(request).request_type
    expected = expected_behavior_for(request, intent)
    with record_execution(root) as trace:
        response = handler(request)
    observed = observe(root, request, response, trace, intent)
    mismatch = compare_behavior(expected, observed) if expected else None
    goal = extract_goal(request)
    goal_satisfaction = evaluate_goal_contract(contract_from_goal(goal), observe_goal_behavior(trace, response))
    return BehaviorAudit(request, intent, expected, observed, mismatch, goal_satisfaction)


def format_behavior_audit(audits: tuple[BehaviorAudit, ...], title: str = "Behavior Monitor (L8.7)") -> str:
    lines = [f"## {title}", "", "| request | intent | contract | events | verdict | goal |", "| --- | --- | --- | --- | --- | --- |"]
    for audit in audits:
        events = ", ".join(dict.fromkeys(event.value for event in audit.observed.events)) or "なし"
        goal_verdict = "達成" if audit.goal_satisfaction and audit.goal_satisfaction.achieved else "未達"
        lines.append(
            f"| {audit.request[:28]} | `{audit.intent.value}` | "
            f"{'あり' if audit.contracted else '未定義'} | {events} | "
            f"{'一致' if audit.consistent and audit.contracted else ('契約なし' if not audit.contracted else '**不一致**')} | {goal_verdict} |"
        )
    mismatches = [audit for audit in audits if audit.mismatch]
    lines.extend(["", "【不一致】"])
    for audit in mismatches:
        lines.append(f"- `{audit.intent.value}`: {audit.mismatch.statement}")
        lines.append(f"    - 契約の根拠: {audit.expected.rationale}")
    if not mismatches:
        lines.append("- なし")
    goal_failures = [audit for audit in audits if audit.goal_satisfaction and not audit.goal_satisfaction.achieved]
    lines.extend(["", "【Goal未達】"])
    lines.extend([f"- {audit.goal_satisfaction.statement}" for audit in goal_failures] or ["- なし"])
    uncontracted = [audit for audit in audits if not audit.contracted]
    lines.extend(["", "【契約未定義の intent】"])
    lines.extend([f"- `{audit.intent.value}`" for audit in dict.fromkeys(uncontracted)] or ["- なし"])
    return "\n".join(lines)


def audit_agent(root: Path, requests: tuple[str, ...]) -> tuple[BehaviorAudit, ...]:
    """Audit the shipped agent over a list of requests."""
    from .coding_agent_app import LocalWorkspaceAgent

    agent = LocalWorkspaceAgent(Path(root))
    return tuple(audit_request(Path(root), request, lambda message: agent.handle(message).text)
                 for request in requests)


DEFAULT_AUDIT_REQUESTS = (
    "シミュレーションを実行して",
    "simulation.py のアルゴリズムを要約して",
    "このリポジトリを要約して",
    "SU2とSU3の検証フォルダをまとめて移動して",
)


def main() -> None:  # pragma: no cover - manual entry point
    import sys as _sys
    import tempfile

    if len(_sys.argv) > 1:
        print(format_behavior_audit(audit_agent(Path(_sys.argv[1]), DEFAULT_AUDIT_REQUESTS)))
        return
    with tempfile.TemporaryDirectory(prefix="l87-audit-") as directory:
        root = Path(directory)
        (root / "SU2_validation").mkdir()
        (root / "SU2_validation" / "run.py").write_text("print('su2 ok')\n", encoding="utf-8")
        (root / "SU3_validation").mkdir()
        (root / "SU3_validation" / "run.py").write_text("print('su3 ok')\n", encoding="utf-8")
        (root / "simulation.py").write_text(
            "def load_config():\n    return {'steps': 3}\n\n\n"
            "def step(state):\n    return state + 1\n\n\n"
            "def run_simulation():\n    config = load_config()\n    state = 0\n"
            "    for _ in range(config['steps']):\n        state = step(state)\n    return state\n\n\n"
            "if __name__ == '__main__':\n    print(run_simulation())\n", encoding="utf-8")
        (root / "README.md").write_text("# demo\n\n```powershell\npython simulation.py\n```\n", encoding="utf-8")
        print(format_behavior_audit(audit_agent(root, DEFAULT_AUDIT_REQUESTS)))


if __name__ == "__main__":  # pragma: no cover
    main()
