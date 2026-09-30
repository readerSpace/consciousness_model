"""L8.1: turn external review proposals into grounded repair workflows."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import re
import subprocess
from typing import Callable, Protocol

from .process_execution import run_text
from .repository_semantic_l58_experiment import RepositoryGraph, index_repository


class ProposalStatus(Enum):
    INVALID = "proposal_invalid"
    PARTIAL = "proposal_partially_applicable"
    APPLICABLE = "proposal_applicable"


@dataclass(frozen=True)
class RepairInstruction:
    problem: str
    target_files: tuple[str, ...]
    target_symbols: tuple[str, ...]
    requested_changes: tuple[str, ...]
    constraints: tuple[str, ...]
    acceptance_tests: tuple[str, ...]
    source: str = "chatgpt"


@dataclass(frozen=True)
class RepairPlan:
    status: ProposalStatus
    resolved_files: tuple[str, ...]
    resolved_symbols: tuple[str, ...]
    unresolved_targets: tuple[str, ...]
    changes: tuple[str, ...]
    regression_tests: tuple[str, ...]
    rationale: tuple[str, ...]


@dataclass(frozen=True)
class PatchResult:
    applied: bool
    changed_files: tuple[str, ...]
    message: str


@dataclass(frozen=True)
class VerificationResult:
    passed: bool
    command: tuple[str, ...]
    output: str
    return_code: int


@dataclass(frozen=True)
class RepairWorkflowResult:
    instruction: RepairInstruction
    plan: RepairPlan
    patch: PatchResult
    verification: VerificationResult | None


@dataclass(frozen=True)
class IntentEvidence:
    intent: str
    score: float
    reasons: tuple[str, ...]


class ProposalSentenceRole(Enum):
    DESCRIPTION = "description"
    MOTIVATION = "motivation"
    IMPLEMENTATION_REQUIREMENT = "implementation_requirement"
    EXAMPLE = "example"
    METRIC = "metric"
    CONSTRAINT = "constraint"
    HEADING = "heading"


@dataclass(frozen=True)
class ExternalRepairRequest:
    instruction: str
    proposal_body: str
    implementation_items: tuple[str, ...]


@dataclass(frozen=True)
class ProposalSentence:
    text: str
    role: ProposalSentenceRole
    reason: str


@dataclass(frozen=True)
class RecognizedRepairTaskSpec:
    intent: str
    proposal_topics: tuple[str, ...]
    implementation_items: tuple[str, ...]
    requires_code_change: bool
    requires_execution: bool
    requires_regression_tests: bool
    memory_policy: str


@dataclass(frozen=True)
class RepositoryImpact:
    requirement: str
    capability: str
    likely_files: tuple[str, ...]
    likely_symbols: tuple[str, ...]
    strategy: str
    evidence: tuple[str, ...]


@dataclass(frozen=True)
class RequirementSlot:
    requirement_id: str
    source_text: str
    trigger: str
    action: str
    object: str
    effect: str
    constraints: tuple[str, ...]
    metrics: tuple[str, ...]
    confidence: float


@dataclass(frozen=True)
class RequirementEdge:
    source: str
    relation: str
    target: str


@dataclass(frozen=True)
class RequirementGraph:
    requirements: tuple[RequirementSlot, ...]
    edges: tuple[RequirementEdge, ...]


REPAIR_WRAPPER_PATTERNS = (
    "以下の提案から",
    "この提案を実装",
    "提案内容を実装",
    "修正案を実装",
    "以下を修正して",
    "提案から実装項目を",
)


_IMPLEMENTATION_PATTERNS = (
    r"(?:を|へ|に).*(?:追加|実装|導入|接続|変更|修正|生成|作成|抽出|比較|判定|分類|再計画|反復|保存|表示|記録|抑制|フィルタ|実行|圧縮|配置|挿入|置く)",
    r"(?:add|implement|introduce|connect|modify|change|replace|generate|create|extract|compare|detect|classify|replan|retry|persist|render|record|suppress|filter|route|forbid|compress|execute|run|use)\b",
    r"(?:max steps|budget|loop detection|premature termination|missing contract|next action|taskgraph|goalcontract|expectedbehavior|observedbehavior)",
)

_DESCRIPTION_ENDINGS = (
    "になります",
    "です",
    "ます",
    "と思います",
    "できます",
    "重要です",
    "自然です",
)


def _section(text: str, names: tuple[str, ...]) -> str:
    pattern = "|".join(re.escape(name) for name in names)
    match = re.search(rf"(?:^|\n)\s*(?:{pattern})\s*:\s*(.*?)(?=\n\s*[A-Za-z][^\n:]*:\s*|\Z)", text, re.I | re.S)
    return match.group(1).strip() if match else ""


def _items(value: str) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(item.strip(" -*\t") for item in re.split(r"\n+|;", value) if item.strip(" -*\t"))


def parse_repair_proposal(text: str, source: str = "chatgpt") -> RepairInstruction:
    problem = _section(text, ("Problem", "問題", "Issue"))
    files = _items(_section(text, ("Target files", "対象ファイル", "Files")))
    symbols = _items(_section(text, ("Target symbols", "対象symbol", "Symbols")))
    changes = extract_implementation_requirements(_section(text, ("Suggested changes", "Requested changes", "変更案", "Changes")))
    constraints = _items(_section(text, ("Constraints", "制約")))
    tests = _items(_section(text, ("Regression tests", "Acceptance tests", "回帰テスト", "Tests")))
    if not changes and text:
        changes = extract_implementation_requirements(text)
    return RepairInstruction(problem, files, symbols, changes, constraints, tests, source)


def is_repair_proposal_request(text: str) -> bool:
    return detect_repair_intent(text).score >= 0.72


def detect_repair_intent(text: str) -> IntentEvidence:
    lower = text.lower()
    score = 0.0
    reasons: list[str] = []
    if "/repair-from-review" in lower or "chatgptの提案" in text:
        score += 0.45
        reasons.append("explicit external-review marker")
    if any(pattern in text for pattern in REPAIR_WRAPPER_PATTERNS):
        score += 0.45
        reasons.append("contains implementation wrapper for an external proposal")
    if any(token in lower for token in ("repairplan", "patchintent", "suggested changes", "regression tests", "fallback", "router")):
        score += 0.25
        reasons.append("contains patch-oriented proposal terminology")
    if any(token in lower for token in ("taskspec", "expectedbehavior", "observedbehavior", "behavior monitor", "file_modification")):
        score += 0.35
        reasons.append("contains task-contract or behavior-monitor terminology")
    if "semantic repair decomposition" in lower or re.search(r"\bl8\.[234]\b", lower):
        score += 0.20
        reasons.append("contains explicit repair-pipeline stage")
    if any(token in text for token in ("修正提案", "変更案", "回帰テスト", "追加", "修正", "変更")):
        score += 0.18
        reasons.append("contains explicit modification language")
    if any(token in lower for token in (" add ", " change ", " modify ", " fix ")):
        score += 0.18
        reasons.append("contains English modification verbs")
    if all(re.search(rf"(?:^|\n)\s*{label}\s*:", text, re.I) for label in ("Problem", "Suggested changes", "Regression tests")):
        score += 0.30
        reasons.append("contains structured problem/change/test sections")
    if "```" in text or re.search(r"(?:^|\n)\s*(?:Problem|Suggested changes|Regression tests)\s*:", text, re.I):
        score += 0.18
        reasons.append("contains proposal structure or code block")
    return IntentEvidence("EXTERNAL_REPAIR_PROPOSAL", min(score, 1.0), tuple(reasons))


def split_proposal_sentences(proposal_body: str) -> tuple[str, ...]:
    sentences: list[str] = []
    for raw in proposal_body.splitlines():
        line = raw.strip()
        if not line:
            continue
        line = re.sub(r"^(?:\d+[.)]|[-*・])\s+", "", line).strip()
        if not line:
            continue
        if re.match(r"^L\d+(?:\.\d+)?[^:：]*[:：]", line, re.I):
            sentences.append(line)
            continue
        if re.match(r"^[A-Za-z][A-Za-z ]{2,30}:\s*$", line):
            sentences.append(line)
            continue
        parts = [part.strip() for part in re.split(r"(?<=[。.!?])\s+", line) if part.strip()]
        sentences.extend(parts or [line])
    return tuple(dict.fromkeys(sentences))


def classify_proposal_sentence(sentence: str) -> ProposalSentence:
    text = sentence.strip()
    lower = text.lower()
    if re.match(r"^L\d+(?:\.\d+)?[^:：]*[:：]", text, re.I):
        return ProposalSentence(text, ProposalSentenceRole.HEADING, "level heading")
    if re.match(r"^(?:example|例|例えば|input|expected|observed)[:：]", text, re.I):
        return ProposalSentence(text, ProposalSentenceRole.EXAMPLE, "example marker")
    if re.search(r"(?:accuracy|rate|ratio|coverage|指標|評価指標|metric|mean_|_rate)", lower):
        return ProposalSentence(text, ProposalSentenceRole.METRIC, "evaluation metric")
    if re.search(r"(?:must|never|禁止|制約|constraint|budget|上限)", lower) and not re.search("|".join(_IMPLEMENTATION_PATTERNS), lower):
        return ProposalSentence(text, ProposalSentenceRole.CONSTRAINT, "constraint wording")
    if re.search(r"(?:問題|原因|ボトルネック|弱い|不足|motivation|because|なぜなら)", text, re.I):
        return ProposalSentence(text, ProposalSentenceRole.MOTIVATION, "problem or motivation wording")
    if any(text.endswith(ending) for ending in _DESCRIPTION_ENDINGS) and not re.search(r"(?:追加する|実装する|導入する|接続する|変更する|修正する|抽出する|判定する|テストする)$", text):
        return ProposalSentence(text, ProposalSentenceRole.DESCRIPTION, "descriptive ending")
    if re.search("|".join(_IMPLEMENTATION_PATTERNS), text, re.I):
        return ProposalSentence(text, ProposalSentenceRole.IMPLEMENTATION_REQUIREMENT, "actionable implementation wording")
    if re.search(r"(?:する|せる|できるようにする)$", text) and re.search(r"[A-Za-z_][A-Za-z0-9_]*|Goal|Task|Behavior|Router|Parser", text):
        return ProposalSentence(text, ProposalSentenceRole.IMPLEMENTATION_REQUIREMENT, "action verb with implementation target")
    return ProposalSentence(text, ProposalSentenceRole.DESCRIPTION, "no actionable change verb")


def classify_proposal_sentences(proposal_body: str) -> tuple[ProposalSentence, ...]:
    return tuple(classify_proposal_sentence(sentence) for sentence in split_proposal_sentences(proposal_body))


def extract_implementation_requirements(proposal_body: str) -> tuple[str, ...]:
    requirements: list[str] = []
    for sentence in classify_proposal_sentences(proposal_body):
        if sentence.role is ProposalSentenceRole.IMPLEMENTATION_REQUIREMENT:
            requirements.append(sentence.text.strip(" -*・\t"))
    return tuple(dict.fromkeys(item for item in requirements if item))


def extract_implementation_items(proposal_body: str) -> tuple[str, ...]:
    headings = tuple(sentence.text for sentence in classify_proposal_sentences(proposal_body) if sentence.role is ProposalSentenceRole.HEADING)
    return tuple(dict.fromkeys((*headings, *extract_implementation_requirements(proposal_body))))


def extract_requirement_slots(requirements: tuple[str, ...]) -> tuple[RequirementSlot, ...]:
    slots: list[RequirementSlot] = []
    for requirement in requirements:
        trigger = _slot_trigger(requirement)
        action = _slot_action(requirement)
        target = _slot_object(requirement, action)
        effect = _slot_effect(requirement, action)
        constraints = _slot_constraints(requirement)
        metrics = _slot_metrics(requirement)
        key = _requirement_key(trigger, action, target)
        confidence = 0.9 if action != "unspecified_change" and target != "unspecified_object" else 0.45
        slots.append(RequirementSlot(key, requirement, trigger, action, target, effect, constraints, metrics, confidence))
    return tuple(dict.fromkeys(slots))


def build_requirement_graph(proposal_body: str | tuple[str, ...]) -> RequirementGraph:
    requirements = proposal_body if isinstance(proposal_body, tuple) else extract_implementation_requirements(proposal_body)
    slots = extract_requirement_slots(requirements)
    by_action = {slot.action: slot for slot in slots}
    edges: list[RequirementEdge] = []
    sequence = (
        ("derive_missing_contract", "enables", "plan_next_action"),
        ("plan_next_action", "feeds", "replan_taskgraph"),
        ("replan_taskgraph", "feeds", "iterate_until_goal_achieved"),
        ("add_loop_guard", "bounds", "iterate_until_goal_achieved"),
        ("add_regression_test", "verifies", "iterate_until_goal_achieved"),
    )
    for source_action, relation, target_action in sequence:
        if source_action in by_action and target_action in by_action:
            edges.append(RequirementEdge(by_action[source_action].requirement_id, relation, by_action[target_action].requirement_id))
    return RequirementGraph(slots, tuple(edges))


def _slot_trigger(text: str) -> str:
    lower = text.lower()
    if re.search(r"goal.*(?:未達|unsatisfied|not satisfied|not achieved)|(?:未達|満たされていな|未充足).*goal", text, re.I):
        return "goal_unsatisfied"
    if re.search(r"expectedbehavior|observedbehavior|expected\s*/\s*observed|不一致|mismatch", lower):
        return "behavior_mismatch"
    if re.search(r"premature termination|早期終了", lower):
        return "premature_termination"
    return "proposal_requirement"


def _slot_action(text: str) -> str:
    lower = text.lower()
    if re.search(r"next action|次のaction|次のアクション|次に実行|次の行動", lower):
        return "plan_next_action"
    if re.search(r"missing contract|欠けたpostcondition|不足条件|未充足部分|完了条件の未充足|missing condition", lower):
        return "derive_missing_contract"
    if re.search(r"taskgraph.*(?:再計画|replan)|途中状態から再計画|replan.*taskgraph", lower):
        return "replan_taskgraph"
    if re.search(r"execute.*observe.*evaluate|反復|繰り返|until goal|goal達成まで|満たすまで継続", lower):
        return "iterate_until_goal_achieved"
    if re.search(r"max steps|budget|loop detection|bounded|上限|ループ検出", lower):
        return "add_loop_guard"
    if re.search(r"premature termination.*(?:test|テスト)|(?:test|テスト).*premature termination|回帰テスト|regression", lower):
        return "add_regression_test"
    if re.search(r"比較|compare", lower):
        return "compare_expected_observed"
    return "unspecified_change"


def _slot_object(text: str, action: str) -> str:
    lower = text.lower()
    if "goalcontract" in lower or "missing contract" in lower:
        return "GoalContract"
    if "taskgraph" in lower:
        return "TaskGraph"
    if "expectedbehavior" in lower or "observedbehavior" in lower:
        return "BehaviorContract"
    if "postcondition" in lower or "完了条件" in text:
        return "GoalContract"
    if action == "plan_next_action":
        return "TaskGraph"
    if action == "iterate_until_goal_achieved":
        return "ExecutionLoop"
    if action == "add_loop_guard":
        return "RepairLoopGuard"
    if action == "add_regression_test":
        return "RegressionSuite"
    return "unspecified_object"


def _slot_effect(text: str, action: str) -> str:
    lower = text.lower()
    if "enable_replanning" in lower or "再計画" in text or action in {"derive_missing_contract", "plan_next_action", "replan_taskgraph"}:
        return "enable_replanning"
    if action == "iterate_until_goal_achieved":
        return "prevent_premature_success"
    if action == "add_loop_guard":
        return "bound_repair_loop"
    if action == "add_regression_test":
        return "detect_premature_termination"
    if action == "compare_expected_observed":
        return "detect_behavior_mismatch"
    return "support_repair_pipeline"


def _slot_constraints(text: str) -> tuple[str, ...]:
    constraints: list[str] = []
    lower = text.lower()
    if re.search(r"max steps|budget|bounded|上限", lower):
        constraints.append("bounded_iterations")
    if re.search(r"loop detection|ループ検出|同じ失敗", lower):
        constraints.append("loop_detection")
    if re.search(r"premature termination|早期終了", lower):
        constraints.append("no_premature_termination")
    return tuple(dict.fromkeys(constraints))


def _slot_metrics(text: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(re.findall(r"[A-Za-z][A-Za-z0-9_]*(?:_accuracy|_rate|_coverage|_consistency)", text)))


def _requirement_key(trigger: str, action: str, target: str) -> str:
    return f"{trigger}:{action}:{target}".lower()


def analyze_repository_impact(root: Path, requirements: tuple[str, ...], graph: RepositoryGraph | None = None) -> tuple[RepositoryImpact, ...]:
    graph = graph or index_repository(root)
    files = tuple(sorted({symbol.file for symbol in graph.symbols}))
    symbol_names = tuple(sorted({symbol.name for symbol in graph.symbols}))
    impacts: list[RepositoryImpact] = []
    slots_by_text = {slot.source_text: slot for slot in extract_requirement_slots(requirements)}
    for requirement in requirements:
        slot = slots_by_text.get(requirement)
        lower = requirement.lower()
        likely_files = tuple(file for file in files if any(token in file.lower() for token in _impact_file_tokens(lower)))
        likely_symbols = tuple(symbol for symbol in symbol_names if symbol.lower() in lower or any(token in symbol.lower() for token in _impact_symbol_tokens(lower)))
        capability = _required_capability(requirement, slot)
        strategy = _change_strategy(capability, likely_files, likely_symbols)
        evidence = tuple(filter(None, (
            f"slot:{slot.requirement_id}" if slot else "",
            f"files:{', '.join(likely_files[:3])}" if likely_files else "",
            f"symbols:{', '.join(likely_symbols[:3])}" if likely_symbols else "",
        )))
        impacts.append(RepositoryImpact(requirement, capability, likely_files[:5], likely_symbols[:5], strategy, evidence))
    return tuple(impacts)


def _required_capability(requirement: str, slot: RequirementSlot | None = None) -> str:
    if slot and slot.action in {"derive_missing_contract", "plan_next_action", "replan_taskgraph", "iterate_until_goal_achieved", "add_loop_guard"}:
        return "GOAL_DIRECTED_REPLANNING"
    lower = requirement.lower()
    if "goal" in lower and re.search(r"未達|missing|next action|replan|再計画|継続|反復", requirement, re.I):
        return "GOAL_DIRECTED_REPLANNING"
    if "behavior" in lower or "expectedbehavior" in lower or "observedbehavior" in lower:
        return "BEHAVIOR_MONITORING"
    if "memory" in lower or "知識" in requirement:
        return "MEMORY_POLICY"
    if "router" in lower or "route" in lower or "ルータ" in requirement:
        return "REQUEST_ROUTING"
    if "test" in lower or "テスト" in requirement or "回帰" in requirement:
        return "REGRESSION_GUARD"
    return "CODE_CHANGE"


def _impact_file_tokens(lower: str) -> tuple[str, ...]:
    tokens: list[str] = []
    if "goal" in lower or "missing contract" in lower or "postcondition" in lower:
        tokens.append("goal")
    if "behavior" in lower or "expectedbehavior" in lower or "observedbehavior" in lower:
        tokens.append("behavior")
    if "router" in lower or "route" in lower:
        tokens.append("app")
    if "test" in lower or "regression" in lower:
        tokens.append("test")
    return tuple(tokens)


def _impact_symbol_tokens(lower: str) -> tuple[str, ...]:
    tokens: list[str] = []
    if "goal" in lower or "missing contract" in lower or "postcondition" in lower:
        tokens.extend(("goal", "contract"))
    if "behavior" in lower:
        tokens.append("behavior")
    if "replan" in lower or "next action" in lower:
        tokens.extend(("plan", "task"))
    return tuple(tokens)


def _change_strategy(capability: str, files: tuple[str, ...], symbols: tuple[str, ...]) -> str:
    if files or symbols:
        return f"reuse existing {capability.lower()} symbols before selecting operation_type"
    return f"inspect repository for {capability.lower()} support before selecting operation_type"


def format_repository_impact(impacts: tuple[RepositoryImpact, ...]) -> str:
    lines = ["## Repository Impact", ""]
    if not impacts:
        lines.append("- 実装要求がないため、影響分析は行いません。")
        return "\n".join(lines)
    for index, impact in enumerate(impacts, 1):
        files = ", ".join(impact.likely_files) if impact.likely_files else "未解決"
        symbols = ", ".join(impact.likely_symbols) if impact.likely_symbols else "未解決"
        lines.extend((
            f"{index}. `{impact.capability}`",
            f"   - requirement: {impact.requirement}",
            f"   - likely_files: {files}",
            f"   - likely_symbols: {symbols}",
            f"   - strategy: {impact.strategy}",
        ))
    return "\n".join(lines)


def format_requirement_graph(graph: RequirementGraph) -> str:
    lines = ["## Requirement Graph", ""]
    if not graph.requirements:
        lines.append("- 実装要求に対応するsemantic slotsはありません。")
        return "\n".join(lines)
    for index, slot in enumerate(graph.requirements, 1):
        constraints = ", ".join(slot.constraints) if slot.constraints else "none"
        metrics = ", ".join(slot.metrics) if slot.metrics else "none"
        lines.extend((
            f"{index}. `{slot.requirement_id}`",
            f"   - trigger: `{slot.trigger}`",
            f"   - action: `{slot.action}`",
            f"   - object: `{slot.object}`",
            f"   - effect: `{slot.effect}`",
            f"   - constraints: {constraints}",
            f"   - metrics: {metrics}",
        ))
    if graph.edges:
        lines.extend(("", "【Edges】"))
        lines.extend(f"- `{edge.source}` {edge.relation} `{edge.target}`" for edge in graph.edges)
    return "\n".join(lines)


def parse_external_repair_request(text: str) -> ExternalRepairRequest:
    wrapper = next((pattern for pattern in REPAIR_WRAPPER_PATTERNS if pattern in text), None)
    if wrapper:
        body = text[text.find(wrapper) + len(wrapper):].lstrip(" ：:。\n")
        return ExternalRepairRequest("implement", body, extract_implementation_items(body))
    return ExternalRepairRequest("repair", text, extract_implementation_items(text))


def recognized_repair_task_spec(request: ExternalRepairRequest, instruction: RepairInstruction) -> RecognizedRepairTaskSpec:
    text = request.proposal_body
    lower = text.lower()
    topics: list[str] = []
    if "taskspec" in lower or "task spec" in lower:
        topics.append("task contract visualization")
    if "expectedbehavior" in lower or "expected:" in lower:
        topics.append("expected behavior generation")
    if "observedbehavior" in lower or "observed:" in lower:
        topics.append("observed behavior collection")
    if "behavior monitor" in lower or "不一致" in text:
        topics.append("behavior mismatch detection")
    if "ui" in lower or "画面" in text or "表示" in text:
        topics.append("ui task preview")
    if not topics:
        topics.append("external repair proposal")
    items = request.implementation_items or instruction.requested_changes
    tests_text = " ".join(instruction.acceptance_tests).lower()
    return RecognizedRepairTaskSpec(
        intent="IMPLEMENT_PROPOSAL" if request.instruction == "implement" else "EXTERNAL_REPAIR_PROPOSAL",
        proposal_topics=tuple(dict.fromkeys(topics)),
        implementation_items=items,
        requires_code_change=bool(items or instruction.target_files or instruction.target_symbols),
        requires_execution=True,
        requires_regression_tests=bool(instruction.acceptance_tests or "test" in lower or "回帰" in text or "regression" in tests_text),
        memory_policy="task_contract_only" if any("behavior" in topic or "task" in topic for topic in topics) else "repair_related_only",
    )


def format_recognized_task_spec(spec: RecognizedRepairTaskSpec) -> str:
    items = [f"{index}. {item}" for index, item in enumerate(spec.implementation_items, 1)] or ["- 抽出できる実装項目はありません。"]
    topics = [f"- {item}" for item in spec.proposal_topics]
    return "\n".join((
        "## Recognized TaskSpec",
        "",
        f"- intent: `{spec.intent}`",
        "- proposal topic:",
        *topics,
        "- implementation items:",
        *items,
        f"- requires_code_change: `{spec.requires_code_change}`",
        f"- requires_execution: `{spec.requires_execution}`",
        f"- requires_regression_tests: `{spec.requires_regression_tests}`",
        f"- memory_policy: `{spec.memory_policy}`",
    ))


def plan_repair(root: Path, instruction: RepairInstruction, graph: RepositoryGraph | None = None) -> RepairPlan:
    graph = graph or index_repository(root)
    resolved_files: set[str] = set()
    resolved_symbols: set[str] = set()
    unresolved: list[str] = []
    all_targets = instruction.target_files + instruction.target_symbols
    for target in all_targets:
        normalized = target.strip().strip("`'\"")
        file_matches = [symbol.file for symbol in graph.symbols if Path(symbol.file).name.lower() == Path(normalized).name.lower() or normalized.lower() in symbol.file.lower()]
        symbol_matches = [symbol.name for symbol in graph.symbols if normalized.lower() in symbol.name.lower()]
        if file_matches:
            resolved_files.update(file_matches)
        if symbol_matches:
            resolved_symbols.update(symbol_matches)
        if not file_matches and not symbol_matches:
            unresolved.append(normalized)
    if not all_targets and graph.symbols:
        unresolved.append("no explicit target file or symbol")
    if not instruction.requested_changes:
        unresolved.append("no requested changes")
    status = ProposalStatus.INVALID if not instruction.problem or not instruction.requested_changes else (ProposalStatus.PARTIAL if unresolved else ProposalStatus.APPLICABLE)
    rationale = ["実repoのRepositoryGraphで対象を再解決しました。", "提案文のファイル名・symbol名はそのまま信用せず、存在確認を行いました。"]
    if unresolved:
        rationale.append("未解決: " + "; ".join(unresolved))
    return RepairPlan(status, tuple(sorted(resolved_files)), tuple(sorted(resolved_symbols)), tuple(unresolved), instruction.requested_changes, instruction.acceptance_tests, tuple(rationale))


class PatchExecutor(Protocol):
    def apply(self, root: Path, instruction: RepairInstruction, plan: RepairPlan) -> PatchResult: ...


class NoOpPatchExecutor:
    def apply(self, root: Path, instruction: RepairInstruction, plan: RepairPlan) -> PatchResult:
        if plan.status is not ProposalStatus.APPLICABLE:
            return PatchResult(False, (), "提案の対象または変更内容が未解決のため適用せず、RepairPlanのみ生成しました。")
        return PatchResult(False, (), "パッチ本文がないため、適用前のRepairPlanまで生成しました。")


CommandRunner = Callable[[list[str], Path], subprocess.CompletedProcess[str]]


class RegressionVerifier:
    def __init__(self, runner: CommandRunner | None = None):
        self.runner = runner or (lambda command, cwd: run_text(command, cwd=cwd, timeout=120))

    def verify(self, root: Path, tests: tuple[str, ...]) -> VerificationResult:
        paths = [test for test in tests if test.endswith(".py") and not test.startswith("http")]
        command = ["python", "-m", "pytest", *paths] if paths else ["python", "-m", "pytest", "-q"]
        try:
            completed = self.runner(command, root)
            return VerificationResult(completed.returncode == 0, tuple(command), (completed.stdout or "")[-4000:], completed.returncode)
        except (OSError, subprocess.TimeoutExpired) as error:
            return VerificationResult(False, tuple(command), str(error), 1)


def run_repair_workflow(root: Path, proposal: str, executor: PatchExecutor | None = None, verifier: RegressionVerifier | None = None) -> RepairWorkflowResult:
    instruction = parse_repair_proposal(proposal)
    plan = plan_repair(root, instruction)
    patch = (executor or NoOpPatchExecutor()).apply(root, instruction, plan)
    verification = (verifier or RegressionVerifier()).verify(root, instruction.acceptance_tests) if patch.applied else None
    return RepairWorkflowResult(instruction, plan, patch, verification)


def format_repair_workflow(result: RepairWorkflowResult) -> str:
    plan = result.plan
    lines = ["## External Repair Workflow", "", f"【判定】`{plan.status.value}`", "", "【問題】", result.instruction.problem or "未記載", "", "【解決した対象】"]
    resolved = [f"- `{item}`" for item in plan.resolved_files + plan.resolved_symbols] or ["- なし"]
    changes = [f"- {item}" for item in plan.changes] or ["- なし"]
    tests = [f"- {item}" for item in plan.regression_tests] or ["- なし"]
    lines.extend(resolved)
    lines.extend(["", "【変更案】", *changes, "", "【回帰テスト】", *tests, "", "【適用】", result.patch.message])
    if result.verification:
        lines.extend(["", "【検証】", f"- passed: `{result.verification.passed}`", f"- return code: `{result.verification.return_code}`", "```text", result.verification.output, "```"])
    lines.extend(["", "【判断根拠】", *[f"- {item}" for item in plan.rationale]])
    return "\n".join(lines)
