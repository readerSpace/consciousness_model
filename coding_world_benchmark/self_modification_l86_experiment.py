"""L8.6: self-modification benchmark over reproduced historical faults.

Each fixture rebuilds, in a miniature workspace, the *shape* of a fault this
agent actually shipped (raw search instead of a summary, a capability that is
implemented but never routed, a semantic change that stopped at
``synthesis_unavailable``, recalled memory appended regardless of policy).
The harness then runs the full loop with no human in it:

    detect -> diagnose -> RepairInstruction -> L8.3 decomposition
           -> L8.5 CodeChangeIR -> L8.2 patch -> L8.4 retry -> regression

Two properties keep the score meaningful rather than self-congratulatory:

* ``diagnose_workspace`` never sees the fixture identity or its expected fault.
  It gets a directory and the probe output, and must infer the entry point, the
  faulty file and the change to request from repository evidence alone.
* One fixture is healthy.  A run that "repairs" it scores a false repair.

The honest caveat is that fixtures and diagnosis heuristics were written
together, so a high score is evidence that the pipeline closes on known fault
shapes -- not that it generalizes to unseen ones.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import ast
import re
import subprocess

from .autonomous_repair_l84_experiment import AutonomousRepairResult, run_autonomous_repair_loop
from .process_execution import combined_output, run_text
from .external_repair_l81_experiment import (
    ProposalStatus,
    RegressionVerifier,
    RepairInstruction,
    parse_repair_proposal,
    plan_repair,
)
from .semantic_change_compiler_l85_experiment import ChangeCompilation, compile_semantic_changes
from .semantic_repair_l83_experiment import RepairDecomposition, decompose_repair


PROBE = "test_behavior.py"
BASELINE = "test_baseline.py"
MODULE = "agent_module.py"


class ExpectedAction(Enum):
    """What the *correct* behaviour is for a fixture, independent of fault type."""

    REPAIR = "REPAIR"      # a correct code fix exists and should be applied automatically
    ABSTAIN = "ABSTAIN"    # out of the formalized vocabulary: escalate, change nothing
    IGNORE = "IGNORE"      # healthy: do not even detect a fault


class FaultClass(Enum):
    UNWIRED_CAPABILITY = "UNWIRED_CAPABILITY"
    MISSING_TYPE = "MISSING_TYPE"
    UNAPPLIED_POLICY_GUARD = "UNAPPLIED_POLICY_GUARD"
    HEALTHY = "HEALTHY"
    UNKNOWN = "UNKNOWN"


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class FaultFixture:
    id: str
    intent: str
    symptom: str
    expected_fault: FaultClass
    files: dict[str, str]
    expected_action: ExpectedAction = ExpectedAction.REPAIR


def _routing_module(route: str, keyword: str, label: str, competing: str = "") -> str:
    """A capability that is implemented but never reached from the entry point."""
    competing_definition = (
        f'\n\ndef {competing}(request):\n    return "{competing.split("_")[0]}-report: " + request\n'
        if competing else ""
    )
    competing_dispatch = (
        f"    _routed = {competing}(request)\n    if _routed is not None:\n        return _routed\n"
        if competing else ""
    )
    return (
        '"""Miniature routing agent used by an L8.6 fault fixture."""\n\n\n'
        'def keyword_search_fallback(request):\n'
        '    return "search-hit: " + request\n\n\n'
        f'def {route}(request):\n'
        f'    if "{keyword}" in request:\n'
        f'        return "{label}: " + request\n'
        '    return None\n'
        f'{competing_definition}\n\n'
        'def handle_request(request):\n'
        f'{competing_dispatch}'
        '    return keyword_search_fallback(request)\n'
    )


def _routing_probe(request: str, label: str) -> str:
    return (
        "from agent_module import handle_request\n\n\n"
        "def test_capability_is_actually_routed():\n"
        f'    assert handle_request("{request}").startswith("{label}:")\n'
    )


def _routing_baseline(label: str) -> str:
    return (
        "from agent_module import handle_request\n\n\n"
        "def test_unrelated_request_keeps_previous_behaviour():\n"
        f'    assert handle_request("unrelated request").startswith("{label}:")\n'
    )


def _routing_fixture(identifier: str, intent: str, symptom: str, route: str, keyword: str,
                     label: str, request: str, competing: str = "") -> FaultFixture:
    return FaultFixture(
        identifier, intent, symptom, FaultClass.UNWIRED_CAPABILITY,
        {
            MODULE: _routing_module(route, keyword, label, competing),
            PROBE: _routing_probe(request, label),
            BASELINE: _routing_baseline(competing.split("_")[0] + "-report" if competing else "search-hit"),
        },
    )


_SEMANTIC_MODULE = (
    '"""Miniature agent that has no TaskIR type yet."""\n\n\n'
    "def handle_request(request):\n"
    '    return "handled: " + request\n'
)

_MEMORY_MODULE = (
    '"""Miniature agent whose memory policy is computed but never applied."""\n\n\n'
    "def core_answer(request):\n"
    '    return "answer: " + request\n\n\n'
    "def recall_knowledge(request):\n"
    '    return "SU2/SU3 の検証知識"\n\n\n'
    "def memory_policy_allows(policy):\n"
    '    return policy == "all"\n\n\n'
    "def respond(request, policy):\n"
    "    answer = core_answer(request)\n"
    "    recalled = recall_knowledge(request)\n"
    "    if recalled:\n"
    '        answer = answer + "\\n【再利用された知識】\\n" + recalled\n'
    "    return answer\n"
)

# The capability is wired correctly; the fault is *inside* the function body, so
# no formalized operation can express the fix.  This fixture exists to measure
# refusal: the pipeline must diagnose UNKNOWN and edit nothing.
_OUT_OF_SCOPE_MODULE = (
    '"""Miniature agent whose routed capability is implemented wrongly."""\n\n\n'
    "def keyword_search_fallback(request):\n"
    '    return "search-hit: " + request\n\n\n'
    "def summarize_route(request):\n"
    '    if "要約" in request:\n'
    '        return "search-hit: " + request\n'
    "    return None\n\n\n"
    "def handle_request(request):\n"
    "    _routed = summarize_route(request)\n"
    "    if _routed is not None:\n"
    "        return _routed\n"
    "    return keyword_search_fallback(request)\n"
)

_HEALTHY_MODULE = (
    '"""Miniature agent with no injected fault."""\n\n\n'
    "def keyword_search_fallback(request):\n"
    '    return "search-hit: " + request\n\n\n'
    "def summarize_route(request):\n"
    '    if "要約" in request:\n'
    '        return "summary: " + request\n'
    "    return None\n\n\n"
    "def handle_request(request):\n"
    "    _routed = summarize_route(request)\n"
    "    if _routed is not None:\n"
    "        return _routed\n"
    "    return keyword_search_fallback(request)\n"
)


FAULT_FIXTURES: tuple[FaultFixture, ...] = (
    _routing_fixture("F1", "SUMMARIZE", "keyword search の結果だけを返す",
                     "summarize_route", "要約", "summary", "このリポジトリを要約して"),
    _routing_fixture("F2", "EXECUTE", "Python を探すだけで実行しない",
                     "execute_route", "実行", "executed", "テストを実行して"),
    _routing_fixture("F3", "MOVE_DIRECTORIES", "target ambiguity で何も移動しない",
                     "move_directories_route", "移動", "moved", "SU2のフォルダを移動して"),
    _routing_fixture("F4", "SCIENTIFIC_RESEARCH", "repo search fallback へ落ちる",
                     "scientific_research_route", "調査", "research", "先行研究を調査して"),
    _routing_fixture("F5", "MATH_DERIVATION", "ファイル検索へ落ちる",
                     "math_derivation_route", "導出", "derivation", "波動方程式を導出して"),
    _routing_fixture("F6", "EXTERNAL_REPAIR_PROPOSAL", "Experiment Semantic Report へ誤routing",
                     "external_repair_route", "提案", "repair", "以下の提案を実装して",
                     competing="experiment_semantic_route"),
    FaultFixture(
        "F7", "SEMANTIC_CHANGE", "意味変更が synthesis_unavailable で停止する", FaultClass.MISSING_TYPE,
        {
            MODULE: _SEMANTIC_MODULE,
            PROBE: (
                "from agent_module import TaskIR\n\n\n"
                "def test_task_ir_carries_action_and_target():\n"
                '    task = TaskIR(action="MOVE", target="SU2")\n'
                '    assert (task.action, task.target) == ("MOVE", "SU2")\n'
            ),
            BASELINE: (
                "from agent_module import handle_request\n\n\n"
                "def test_existing_entry_point_is_untouched():\n"
                '    assert handle_request("x") == "handled: x"\n'
            ),
        },
    ),
    FaultFixture(
        "F8", "MEMORY_POLICY", "SU2/SU3 知識を無関係な回答へ大量表示する", FaultClass.UNAPPLIED_POLICY_GUARD,
        {
            MODULE: _MEMORY_MODULE,
            PROBE: (
                "from agent_module import respond\n\n\n"
                "def test_memory_is_withheld_under_restricted_policy():\n"
                '    assert "再利用された知識" not in respond("SU2の検証", "repair_related_only")\n'
            ),
            BASELINE: (
                "from agent_module import respond\n\n\n"
                "def test_memory_is_still_available_under_all_policy():\n"
                '    assert "再利用された知識" in respond("SU2の検証", "all")\n'
            ),
        },
    ),
    FaultFixture(
        "F9", "SUMMARIZE_BODY", "routing は正しいが実装本文が検索結果を返す（語彙外）", FaultClass.UNKNOWN,
        {
            MODULE: _OUT_OF_SCOPE_MODULE,
            PROBE: _routing_probe("これを要約して", "summary"),
            BASELINE: _routing_baseline("search-hit"),
        },
        ExpectedAction.ABSTAIN,
    ),
    FaultFixture(
        "F0", "CONTROL", "故障を注入していない健全な workspace", FaultClass.HEALTHY,
        {
            MODULE: _HEALTHY_MODULE,
            PROBE: (
                "from agent_module import handle_request\n\n\n"
                "def test_capability_is_actually_routed():\n"
                '    assert handle_request("これを要約して").startswith("summary:")\n'
            ),
            BASELINE: (
                "from agent_module import handle_request\n\n\n"
                "def test_unrelated_request_keeps_previous_behaviour():\n"
                '    assert handle_request("unrelated request").startswith("search-hit:")\n'
            ),
        },
        ExpectedAction.IGNORE,
    ),
)


def build_fixture(root: Path, fixture: FaultFixture) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for name, content in fixture.files.items():
        (root / name).write_text(content, encoding="utf-8")
    return root


# --------------------------------------------------------------------------
# detect
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class FaultObservation:
    detected: bool
    command: tuple[str, ...]
    output: str
    return_code: int


def observe_fault(root: Path, tests: tuple[str, ...] = (PROBE,)) -> FaultObservation:
    """A fault is *detected* when the behavioural probe fails on the real workspace."""
    command = ["python", "-m", "pytest", "-q", *tests]
    completed = run_text(command, cwd=root, timeout=300)
    return FaultObservation(completed.returncode != 0, tuple(command), combined_output(completed, 4000), completed.returncode)


# --------------------------------------------------------------------------
# diagnose (repository evidence only -- no fixture identity)
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class DiagnosisCandidate:
    """One detector's claim.  ``specificity`` ranks how narrow its evidence is.

    Specific evidence beats generic evidence: "this entry argument exists but is
    never read" (2) points at one statement, while "some function is not wired
    anywhere" (1) is true of any dead helper.  Ranking by specificity first and
    confidence second is the generalization of the F8 conflict, where the unwired
    rule would otherwise have claimed the guard helper's own evidence.
    """

    fault_class: FaultClass
    specificity: int
    confidence: float
    target_file: str
    entry: str
    artifact: str
    anchor: str
    members: tuple[str, ...]
    statement: str
    evidence: tuple[str, ...]


@dataclass(frozen=True)
class FaultDiagnosis:
    fault_class: FaultClass
    target_file: str
    entry: str
    artifact: str
    anchor: str
    members: tuple[str, ...]
    statement: str
    evidence: tuple[str, ...]
    confidence: float
    specificity: int = 0
    rejected: tuple[str, ...] = ()


def _module_of(root: Path, tests: tuple[str, ...]) -> str:
    """The module under test, taken from what the probe imports."""
    for test in tests:
        path = root / test
        if not path.is_file():
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and (root / f"{node.module}.py").is_file():
                return f"{node.module}.py"
    return MODULE if (root / MODULE).is_file() else ""


def _entry_of(root: Path, tests: tuple[str, ...], module: str) -> str:
    """The entry point the probe drives, i.e. an imported name it also calls."""
    source = (root / module).read_text(encoding="utf-8", errors="replace") if module else ""
    defined = {node.name for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)} if source else set()
    for test in tests:
        path = root / test
        if not path.is_file():
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        imported = {alias.name for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) for alias in node.names}
        for call in ast.walk(tree):
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id in imported & defined:
                return call.func.id
    return ""


def _function(source: str, name: str) -> ast.FunctionDef | None:
    tree = ast.parse(source)
    return next((node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name), None)


def _called_names(node: ast.AST) -> set[str]:
    return {item.func.id for item in ast.walk(node) if isinstance(item, ast.Call) and isinstance(item.func, ast.Name)}


def _unwired_capability(source: str, entry: str) -> tuple[str, str]:
    """A function defined in the module that nothing ever calls, plus the first route the entry does call."""
    tree = ast.parse(source)
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    referenced: set[str] = set()
    for node in functions:
        referenced |= {item.id for item in ast.walk(node) if isinstance(item, ast.Name)}
    unwired = [node.name for node in functions if node.name != entry and node.name not in referenced]
    entry_node = _function(source, entry)
    if not unwired or entry_node is None:
        return "", ""
    called = [item for item in ast.walk(entry_node) if isinstance(item, ast.Call) and isinstance(item.func, ast.Name)]
    anchor = min(called, key=lambda item: item.lineno).func.id if called else ""
    return unwired[0], anchor


def _missing_name(output: str) -> str:
    patterns = (
        r"cannot import name '([A-Za-z_]\w*)'",
        r"NameError: name '([A-Za-z_]\w*)' is not defined",
        r"has no attribute '([A-Za-z_]\w*)'",
    )
    for pattern in patterns:
        match = re.search(pattern, output)
        if match:
            return match.group(1)
    return ""


def _constructor_members(root: Path, tests: tuple[str, ...], name: str) -> tuple[str, ...]:
    """Fields the probe expects, read from how it constructs and reads the symbol."""
    members: list[str] = []
    for test in tests:
        path = root / test
        if not path.is_file():
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == name:
                members.extend(keyword.arg for keyword in node.keywords if keyword.arg)
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                members.append(node.attr)
    return tuple(dict.fromkeys(item for item in members if item))


def _unapplied_guard(source: str, entry: str) -> tuple[str, str, str]:
    """An entry parameter that the body ignores, the helper that consumes it, and the unguarded branch."""
    entry_node = _function(source, entry)
    if entry_node is None:
        return "", "", ""
    used = {item.id for item in ast.walk(entry_node) if isinstance(item, ast.Name)}
    ignored = next((arg.arg for arg in entry_node.args.args if arg.arg not in used and arg.arg not in {"self", "cls"}), "")
    if not ignored:
        return "", "", ""
    tree = ast.parse(source)
    called_anywhere: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            called_anywhere |= _called_names(node)
    helper = next(
        (node.name for node in tree.body
         if isinstance(node, ast.FunctionDef) and node.name not in called_anywhere and node.name != entry
         and any(arg.arg == ignored for arg in node.args.args)),
        "",
    )
    branch = next((node for node in ast.walk(entry_node)
                   if isinstance(node, ast.If) and ignored not in {item.id for item in ast.walk(node.test) if isinstance(item, ast.Name)}), None)
    anchor = next((item.id for item in ast.walk(branch.test) if isinstance(item, ast.Name)), "") if branch else ""
    return ignored, helper, anchor


def _missing_type_candidate(root: Path, source: str, module: str, observation: FaultObservation,
                           tests: tuple[str, ...], entry: str) -> DiagnosisCandidate | None:
    missing = _missing_name(observation.output)
    if not missing or re.search(rf"^\s*(?:class|def)\s+{re.escape(missing)}\b", source, re.M):
        return None
    return DiagnosisCandidate(
        FaultClass.MISSING_TYPE, 3, 0.92, module, entry, missing, "",
        _constructor_members(root, tests, missing),
        f"probe が要求する `{missing}` が {module} に存在しません。",
        (f"probe: {missing} を import できない", f"{module}: 定義なし"),
    )


def _policy_guard_candidate(root: Path, source: str, module: str, observation: FaultObservation,
                            tests: tuple[str, ...], entry: str) -> DiagnosisCandidate | None:
    if not entry:
        return None
    ignored, helper, anchor = _unapplied_guard(source, entry)
    if not (ignored and helper and anchor):
        return None
    return DiagnosisCandidate(
        FaultClass.UNAPPLIED_POLICY_GUARD, 2, 0.86, module, entry, helper, anchor, (ignored,),
        f"`{entry}` の引数 `{ignored}` が本体で使われず、`{anchor}` の分岐が無条件に出力へ追加されています。",
        (f"{entry}: 引数 {ignored} 未使用", f"{module}: {helper} は定義のみで参照なし"),
    )


def _unwired_candidate(root: Path, source: str, module: str, observation: FaultObservation,
                       tests: tuple[str, ...], entry: str) -> DiagnosisCandidate | None:
    if not entry:
        return None
    artifact, anchor = _unwired_capability(source, entry)
    if not (artifact and anchor):
        return None
    return DiagnosisCandidate(
        FaultClass.UNWIRED_CAPABILITY, 1, 0.88, module, entry, artifact, anchor, (),
        f"`{artifact}` は実装済みですが `{entry}` から一度も呼ばれず、`{anchor}` が先に応答しています。",
        (f"{module}: {artifact} は定義のみで参照なし", f"{entry}: 最初の dispatch は {anchor}"),
    )


DETECTORS = (_missing_type_candidate, _policy_guard_candidate, _unwired_candidate)


def diagnose_candidates(root: Path, observation: FaultObservation,
                        tests: tuple[str, ...] = (PROBE, BASELINE)) -> tuple[DiagnosisCandidate, ...]:
    """Every detector that fires, most specific first.  No fixture identity is used."""
    module = _module_of(root, tests)
    if not module:
        return ()
    source = (root / module).read_text(encoding="utf-8", errors="replace")
    entry = _entry_of(root, tests, module)
    candidates = [detector(root, source, module, observation, tests, entry) for detector in DETECTORS]
    return tuple(sorted((item for item in candidates if item is not None),
                        key=lambda item: (item.specificity, item.confidence), reverse=True))


def diagnose_workspace(root: Path, observation: FaultObservation,
                       tests: tuple[str, ...] = (PROBE, BASELINE)) -> FaultDiagnosis:
    """Classify the fault from repository evidence and probe output alone."""
    if not observation.detected:
        return FaultDiagnosis(FaultClass.HEALTHY, "", "", "", "", (), "probe が通過したため修正対象はありません。", (), 1.0, 0, ())
    candidates = diagnose_candidates(root, observation, tests)
    if not candidates:
        module = _module_of(root, tests)
        return FaultDiagnosis(FaultClass.UNKNOWN, module, _entry_of(root, tests, module) if module else "", "", "", (),
                              "既知の検出器がどれも発火せず、故障の型を repo 証拠から決定できませんでした。", (), 0.0, 0, ())
    best, *rest = candidates
    rejected = tuple(f"{item.fault_class.value}(specificity={item.specificity})" for item in rest)
    return FaultDiagnosis(best.fault_class, best.target_file, best.entry, best.artifact, best.anchor,
                          best.members, best.statement, best.evidence, best.confidence, best.specificity, rejected)


# --------------------------------------------------------------------------
# repair instruction (natural language, so the whole L8.1 path stays exercised)
# --------------------------------------------------------------------------

def compose_repair_proposal(diagnosis: FaultDiagnosis, tests: tuple[str, ...]) -> str:
    if diagnosis.fault_class is FaultClass.UNWIRED_CAPABILITY:
        change = f"既存{diagnosis.anchor}の前に{diagnosis.artifact}を配置"
    elif diagnosis.fault_class is FaultClass.MISSING_TYPE:
        members = ", ".join(diagnosis.members)
        change = f"{diagnosis.artifact}型を追加 ({members})" if members else f"{diagnosis.artifact}型を追加"
    elif diagnosis.fault_class is FaultClass.UNAPPLIED_POLICY_GUARD:
        guard = f"{diagnosis.artifact}({diagnosis.members[0]})"
        change = f"replace branch condition {diagnosis.anchor} -> {guard} and {diagnosis.anchor}"
    else:
        change = ""
    lines = [
        f"Problem: {diagnosis.statement}",
        f"Target files: {diagnosis.target_file}",
        "Suggested changes:",
        f"- {change}",
        "Regression tests:",
        *[f"- {item}" for item in tests],
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# benchmark
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class FixtureOutcome:
    fixture_id: str
    intent: str
    expected_fault: FaultClass
    expected_action: ExpectedAction
    observation: FaultObservation
    diagnosis: FaultDiagnosis
    instruction: RepairInstruction | None
    decomposition: RepairDecomposition | None
    compilation: ChangeCompilation | None
    loop: AutonomousRepairResult | None
    changed_files: tuple[str, ...]
    probe_passed_after: bool
    baseline_passed_after: bool
    attempts: int

    @property
    def detected(self) -> bool:
        return self.observation.detected

    @property
    def diagnosed(self) -> bool:
        return self.diagnosis.fault_class is self.expected_fault

    @property
    def repaired(self) -> bool:
        return bool(self.loop and self.loop.outcome == "accepted")

    @property
    def recovered(self) -> bool:
        return self.repaired and self.probe_passed_after and self.baseline_passed_after

    @property
    def abstained(self) -> bool:
        """Handed the problem back without touching the workspace."""
        return not self.repaired and not self.changed_files

    @property
    def false_repair(self) -> bool:
        """Edited a workspace it had no business editing."""
        if self.expected_action is ExpectedAction.IGNORE:
            return bool(self.changed_files) or self.detected
        if self.expected_action is ExpectedAction.ABSTAIN:
            return bool(self.changed_files)
        return self.repaired and not self.probe_passed_after

    @property
    def decision_correct(self) -> bool:
        """Did it make the right call: repair, abstain, or leave alone?"""
        if self.expected_action is ExpectedAction.REPAIR:
            return self.recovered
        if self.expected_action is ExpectedAction.ABSTAIN:
            return self.detected and self.abstained
        return not self.detected and not self.changed_files

    @property
    def escalated(self) -> bool:
        """Ended up back with a human: a fault was seen but not verifiably fixed."""
        if self.expected_action is ExpectedAction.IGNORE:
            return self.detected
        return not self.recovered

    @property
    def unnecessary_edit(self) -> bool:
        planned = {item.file for item in self.compilation.compiled} if self.compilation else set()
        return bool(set(self.changed_files) - planned)


@dataclass(frozen=True)
class BenchmarkMetrics:
    """Repair coverage and decision accuracy are reported separately.

    Coverage answers "how far can it get on its own"; decision accuracy answers
    "does it repair when it should and keep its hands off when it should not".
    An unsupported fault lowers coverage by design while leaving decision
    accuracy at 1.0, because abstaining there is the correct behaviour.
    """

    repairable: int
    unsupported: int
    healthy: int
    attempted: int
    fault_detection_rate: float
    diagnosis_accuracy: float
    repair_grounding_rate: float
    semantic_compile_rate: float
    patch_success_rate: float
    repair_success_rate: float
    safe_abstention_rate: float
    no_false_repair_rate: float
    behavioral_decision_accuracy: float
    automatic_repair_coverage: float
    mean_attempts_to_recovery: float
    regression_introduction_rate: float
    unnecessary_edit_rate: float
    required_escalation_accuracy: float
    unnecessary_escalation_rate: float
    false_repair_count: int


@dataclass(frozen=True)
class BenchmarkResult:
    outcomes: tuple[FixtureOutcome, ...]
    metrics: BenchmarkMetrics
    notes: tuple[str, ...] = field(default=())


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 3) if denominator else 0.0


def run_fixture(root: Path, fixture: FaultFixture, verifier: RegressionVerifier | None = None,
                max_attempts: int = 2) -> FixtureOutcome:
    build_fixture(root, fixture)
    tests = (PROBE, BASELINE)
    observation = observe_fault(root, (PROBE,))
    diagnosis = diagnose_workspace(root, observation, tests)
    if not observation.detected or diagnosis.fault_class in {FaultClass.HEALTHY, FaultClass.UNKNOWN}:
        healthy = observe_fault(root, tests)
        return FixtureOutcome(fixture.id, fixture.intent, fixture.expected_fault, fixture.expected_action,
                              observation, diagnosis, None, None, None, None, (),
                              not healthy.detected, not healthy.detected, 0)

    proposal = compose_repair_proposal(diagnosis, tests)
    instruction = parse_repair_proposal(proposal, source="l86_self_diagnosis")
    plan = plan_repair(root, instruction)
    decomposition = decompose_repair(instruction, plan, root)
    compilation = compile_semantic_changes(root, instruction, plan)
    loop = run_autonomous_repair_loop(root, instruction, verifier=verifier, max_attempts=max_attempts)
    changed = loop.final_patch.patch.changed_files if loop.final_patch else ()
    after_probe = observe_fault(root, (PROBE,))
    after_baseline = observe_fault(root, (BASELINE,))
    return FixtureOutcome(fixture.id, fixture.intent, fixture.expected_fault, fixture.expected_action,
                          observation, diagnosis, instruction, decomposition, compilation, loop, changed,
                          not after_probe.detected, not after_baseline.detected, len(loop.attempts))


def summarize(outcomes: tuple[FixtureOutcome, ...]) -> BenchmarkMetrics:
    repairable = [item for item in outcomes if item.expected_action is ExpectedAction.REPAIR]
    unsupported = [item for item in outcomes if item.expected_action is ExpectedAction.ABSTAIN]
    healthy = [item for item in outcomes if item.expected_action is ExpectedAction.IGNORE]
    faulty = repairable + unsupported
    detected = [item for item in faulty if item.detected]
    attempted = [item for item in detected if item.loop is not None]
    repaired = [item for item in attempted if item.repaired]
    recovered = [item for item in repairable if item.recovered]
    grounded = [item for item in detected if item.compilation and item.compilation.compiled]
    should_not_escalate = repairable + healthy
    return BenchmarkMetrics(
        repairable=len(repairable),
        unsupported=len(unsupported),
        healthy=len(healthy),
        attempted=len(attempted),
        fault_detection_rate=_ratio(len(detected), len(faulty)),
        diagnosis_accuracy=_ratio(len([item for item in detected if item.diagnosed]), len(detected)),
        repair_grounding_rate=_ratio(len([item for item in detected if item.diagnosis.target_file and item.loop]), len(detected)),
        semantic_compile_rate=_ratio(len(grounded), len(detected)),
        patch_success_rate=_ratio(len(repaired), len(attempted)),
        repair_success_rate=_ratio(len(recovered), len(repairable)),
        safe_abstention_rate=_ratio(len([item for item in unsupported if item.decision_correct]), len(unsupported)),
        no_false_repair_rate=_ratio(len([item for item in healthy if item.decision_correct]), len(healthy)),
        behavioral_decision_accuracy=_ratio(len([item for item in outcomes if item.decision_correct]), len(outcomes)),
        automatic_repair_coverage=_ratio(len(recovered), len(faulty)),
        mean_attempts_to_recovery=round(sum(item.attempts for item in recovered) / len(recovered), 2) if recovered else 0.0,
        regression_introduction_rate=_ratio(len([item for item in repaired if not item.baseline_passed_after]), len(repaired)),
        unnecessary_edit_rate=_ratio(len([item for item in repaired if item.unnecessary_edit]), len(repaired)),
        required_escalation_accuracy=_ratio(len([item for item in unsupported if item.escalated]), len(unsupported)),
        unnecessary_escalation_rate=_ratio(len([item for item in should_not_escalate if item.escalated]), len(should_not_escalate)),
        false_repair_count=len([item for item in outcomes if item.false_repair]),
    )


def run_self_modification_benchmark(workspace: Path, fixtures: tuple[FaultFixture, ...] = FAULT_FIXTURES,
                                    verifier: RegressionVerifier | None = None) -> BenchmarkResult:
    outcomes = tuple(run_fixture(Path(workspace) / fixture.id, fixture, verifier) for fixture in fixtures)
    notes = (
        "fixture は過去に実際に発生した故障の『形』を最小 workspace で再現したもので、実コードそのものではありません。",
        "diagnose_workspace は fixture id も期待故障も受け取らず、repo 証拠と probe 出力のみで分類します。",
        "repair coverage と decision accuracy は別指標です。語彙外故障で着手しないことは coverage を下げますが decision としては正解です。",
        "fixture と診断ヒューリスティクスは同時に設計したため、高スコアは既知故障形での閉ループ成立の証拠であり、未知故障への一般化の証拠ではありません。hold-out 評価は L8.6.1 を参照。",
    )
    return BenchmarkResult(outcomes, summarize(outcomes), notes)


def _show(value: float, denominator: int) -> str:
    """A rate over an empty set is `n/a`, never 0.0 -- an absent control is not a failure."""
    return f"`{value}`" if denominator else "`n/a` (該当 fixture なし)"


def format_self_modification_report(result: BenchmarkResult, title: str = "Self-Modification Benchmark (L8.6)") -> str:
    metrics = result.metrics
    lines = [f"## {title}", "", "【Fixture 結果】", "",
             "| fixture | intent | expected | detected | diagnosed | repaired | decision |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for item in result.outcomes:
        lines.append(
            f"| {item.fixture_id} | {item.intent} | {item.expected_action.value} | "
            f"{'✓' if item.detected else '-'} | {'✓' if item.diagnosed else '-'} | "
            f"{'✓' if item.repaired else '-'} | {'✓' if item.decision_correct else '✗'} |"
        )
    lines.extend(["", "【修復能力 — どこまで自力で直せるか】", "",
                  f"- repairable / unsupported / healthy: `{metrics.repairable}` / `{metrics.unsupported}` / `{metrics.healthy}`",
                  f"- fault_detection_rate: `{metrics.fault_detection_rate}`",
                  f"- diagnosis_accuracy: `{metrics.diagnosis_accuracy}`",
                  f"- repair_grounding_rate: `{metrics.repair_grounding_rate}`",
                  f"- semantic_compile_rate: `{metrics.semantic_compile_rate}`",
                  f"- patch_success_rate: `{metrics.patch_success_rate}` (attempted `{metrics.attempted}`)",
                  f"- repair_success_rate: {_show(metrics.repair_success_rate, metrics.repairable)}",
                  f"- automatic_repair_coverage: `{metrics.automatic_repair_coverage}`",
                  f"- mean_attempts_to_recovery: `{metrics.mean_attempts_to_recovery}`",
                  "", "【判断能力 — 直す/触らないを正しく選べるか】", "",
                  f"- safe_abstention_rate: {_show(metrics.safe_abstention_rate, metrics.unsupported)}",
                  f"- no_false_repair_rate: {_show(metrics.no_false_repair_rate, metrics.healthy)}",
                  f"- behavioral_decision_accuracy: `{metrics.behavioral_decision_accuracy}`",
                  f"- required_escalation_accuracy: {_show(metrics.required_escalation_accuracy, metrics.unsupported)}",
                  f"- unnecessary_escalation_rate: `{metrics.unnecessary_escalation_rate}`",
                  "", "【安全性】", "",
                  f"- regression_introduction_rate: `{metrics.regression_introduction_rate}`",
                  f"- unnecessary_edit_rate: `{metrics.unnecessary_edit_rate}`",
                  f"- false_repair_count: `{metrics.false_repair_count}`",
                  "", "【エスカレーションした fixture】"])
    escalated = [item for item in result.outcomes if item.escalated]
    lines.extend([f"- {item.fixture_id} ({item.intent}): expected=`{item.expected_action.value}` "
                  f"diagnosis=`{item.diagnosis.fault_class.value}` outcome=`{item.loop.outcome if item.loop else 'not_attempted'}` "
                  f"{'正しいエスカレーション' if item.expected_action is ExpectedAction.ABSTAIN else '本来は自動修復すべき'}"
                  for item in escalated] or ["- なし"])
    lines.extend(["", "【解釈上の注意】", *[f"- {item}" for item in result.notes]])
    return "\n".join(lines)


def main() -> None:  # pragma: no cover - manual entry point
    import tempfile

    with tempfile.TemporaryDirectory(prefix="l86-benchmark-") as directory:
        print(format_self_modification_report(run_self_modification_benchmark(Path(directory))))


if __name__ == "__main__":  # pragma: no cover
    main()
