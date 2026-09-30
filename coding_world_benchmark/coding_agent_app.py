"""Local desktop UI for the consciousness-model coding agent.

Run with ``python -m coding_world_benchmark.coding_agent_app``.  The UI uses
only the Python standard library.  Workspace operations are intentionally
small and inspectable; natural-language requests are resolved by local
repository analysis rather than by an LLM API.
"""
from __future__ import annotations

from dataclasses import dataclass
import gzip
import json
from pathlib import Path
import re
import subprocess
import tempfile
import threading
try:  # The desktop window needs a toolkit; the agent and the HTTP bridge do not.
    import tkinter as tk
    import tkinter.font as tkfont
    from tkinter import filedialog, messagebox, ttk
except ImportError:  # pragma: no cover - exercised on headless machines
    # Importing this module headlessly gives the same LocalWorkspaceAgent the
    # bridge serves; only CodingAgentApp needs the toolkit, and it says so when
    # it is built rather than at import time. Annotations are strings here
    # (``from __future__ import annotations``), so the names are only needed at
    # call time.
    tk = tkfont = filedialog = messagebox = ttk = None

from .experiment_semantic_l66_experiment import (
    extract_experiment_spec,
    format_experiment_report,
    is_experiment_configuration_request,
)
from .execution_router_l65_experiment import execute_request, format_execution_result
from .grounded_repository_explanation_l59_experiment import explain_repository_question
from .math_algorithm_l67_experiment import extract_mathematical_model, format_math_report, is_math_explanation_request
from .scientific_experiment_l68_experiment import (
    design_relativistic_garage_experiment,
    format_verification_report,
    is_design_and_verify_request,
    run_experiment,
)
from .open_research_l69_experiment import (
    design_open_research_plan,
    format_open_research_report,
    is_open_research_request,
    run_open_research_plan,
)
from .scientific_knowledge_l70_experiment import (
    build_research_plan,
    derive_1d_wave_equation,
    format_derivation_report,
    format_research_report,
    is_math_derivation_request,
    is_scientific_research_request,
)
from .knowledge_retrieval_l71_experiment import ScientificRetriever, format_retrieval_outcome
from .live_scientific_adapters_l72_experiment import HttpPageFetcher, HttpSearchAdapter, live_web_enabled
from .conversation_memory_l73_experiment import ConversationKnowledgeMemory
from .open_scientific_compiler_l74_experiment import (
    compile_pairwise_distance_experiment,
    format_compiled_experiment_report,
    is_pairwise_geometry_request,
    run_compiled_experiment,
)
from .workspace_operations_l75_experiment import (
    execute_delete_operation,
    execute_move_operation,
    format_operation_report,
    is_delete_directory_request,
    is_move_directories_request,
    parse_delete_operation,
    parse_move_operation,
)
from .validation_memory_l76_experiment import ValidationCompressor, ValidationMemoryRetriever, format_validation_memory
from .local_paper_l77_experiment import extract_local_paper, format_paper_knowledge, is_local_paper_request
from .paper_knowledge_quality_l78_experiment import (
    assess_paper_quality,
    format_integrated_evidence,
    format_paper_quality_report,
    paper_to_scientific_evidence,
)
from .algorithm_summary_l80_experiment import format_algorithm_summary, is_algorithm_summary_request, summarize_algorithm
from .external_repair_l81_experiment import detect_repair_intent, format_repair_workflow, is_repair_proposal_request, parse_external_repair_request, run_repair_workflow
from .external_repair_l81_experiment import parse_repair_proposal, plan_repair
from .external_repair_l81_experiment import analyze_repository_impact, build_requirement_graph, format_recognized_task_spec, format_repository_impact, format_requirement_graph, recognized_repair_task_spec
from .autonomous_patch_l82_experiment import (
    CompositePatchSynthesizer,
    MinimalPatchSynthesizer,
    format_autonomous_patch,
    run_autonomous_patch_workflow,
)
from .semantic_repair_l83_experiment import decompose_repair, format_repair_decomposition
from .autonomous_repair_l84_experiment import format_autonomous_repair, run_autonomous_repair_loop
from .semantic_change_compiler_l85_experiment import SemanticPatchSynthesizer, format_change_compilation
from .self_modification_l86_experiment import format_self_modification_report, run_self_modification_benchmark
from .holdout_faults_l861_experiment import compare_known_and_holdout, format_holdout_comparison
from .behavior_monitor_l87_experiment import DEFAULT_AUDIT_REQUESTS, audit_agent, format_behavior_audit
from .scientific_simulation_l88_experiment import (
    format_simulation_implementation_report,
    implement_scientific_simulation,
    is_scientific_simulation_implementation_request,
)
from .goal_semantics_l89_experiment import extract_goal
from .goal_requirement_l811_experiment import compile_goal_driven_plan, format_goal_driven_plan
from .process_execution import combined_output, run_text
from .structured_repository_report_l62_experiment import RequestType, classify_request, structured_repository_report
from .semantic_service import SEMANTIC_FEATURE_COMMANDS, SemanticSession, handle_command, render_state


@dataclass(frozen=True)
class ChatResult:
    text: str
    provider: str


AGENT_FEATURE_COMMANDS: tuple[tuple[str, str], ...] = (
    ("/inspect", "workspace内のファイル概観"),
    ("/tests", "unittest discoveryを実行"),
    ("/self-benchmark", "L8.6 自己修正ベンチマーク"),
    ("/holdout-benchmark", "L8.6.1 hold-out自己修復評価"),
    ("/behavior-audit", "L8.7 行動契約監査"),
    ("/goal <request>", "L8.9/L8.11 GoalIR / RequirementGraph / TaskGraphを表示"),
) + SEMANTIC_FEATURE_COMMANDS


def memory_policy_for_message(message: str) -> str:
    lower = message.lower()
    if any(token in lower for token in ("taskspec", "expectedbehavior", "observedbehavior", "behavior monitor", "file_modification")):
        return "task_contract_only"
    goal = extract_goal(message)
    if "artifact_created" in goal.completion_conditions:
        return "scientific_modeling_only" if "simulation" in goal.desired_properties else "repair_related_only"
    intent = detect_repair_intent(message)
    if is_scientific_simulation_implementation_request(message):
        return "scientific_modeling_only"
    if intent.score >= 0.72:
        return "repair_related_only"
    is_workspace_action = any(word in message for word in ("移動", "削除", "消して", "フォルダ")) and any(word in message for word in ("SU2", "SU3", "directory", "folder", "検証"))
    return "path_resolution_only" if is_workspace_action else "all"


def save_session(path: Path, messages: list[dict[str, str]], workspace: Path | None = None) -> None:
    """Persist only conversation/workspace metadata, never API credentials."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"workspace": str(workspace) if workspace else None, "messages": messages}
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False)


def load_session(path: Path) -> tuple[Path | None, list[dict[str, str]]]:
    if not path.is_file():
        return None, []
    try:
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            payload = json.load(stream)
        workspace = Path(payload["workspace"]) if payload.get("workspace") else None
        messages = [item for item in payload.get("messages", [])
                    if isinstance(item, dict) and isinstance(item.get("speaker"), str) and isinstance(item.get("text"), str)]
        return workspace, messages
    except (OSError, EOFError, json.JSONDecodeError, TypeError, KeyError):
        return None, []


def save_work_log(workspace: Path, messages: list[dict[str, str]]) -> Path:
    folder = workspace / ".conscious_coding_agent"
    folder.mkdir(exist_ok=True)
    path = folder / "work_log.md"
    lines = ["# Conscious Coding Agent 作業ログ", ""]
    for item in messages:
        lines.extend([f"## {item['speaker']}", "", item["text"], ""])
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


_INLINE_MARKDOWN = re.compile(r"(`[^`]+`|\*\*[^*]+\*\*|\[[^]]+\]\([^)]*\))")


def _inline_tokens(text: str) -> list[tuple[str, str]]:
    """Compile the small Markdown inline subset used by agent responses."""
    tokens: list[tuple[str, str]] = []
    cursor = 0
    for match in _INLINE_MARKDOWN.finditer(text):
        if match.start() > cursor:
            tokens.append((text[cursor:match.start()], "body"))
        raw = match.group(0)
        if raw.startswith("`"):
            tokens.append((raw[1:-1], "inline_code"))
        elif raw.startswith("**"):
            tokens.append((raw[2:-2], "strong"))
        else:
            label, _, url = raw[1:-1].partition("](")
            tokens.append((label, "link"))
            if url:
                tokens.append((f" ({url})", "link_url"))
        cursor = match.end()
    if cursor < len(text):
        tokens.append((text[cursor:], "body"))
    return tokens or [("", "body")]


def compile_markdown(markdown: str) -> list[tuple[str, str]]:
    """Compile readable Markdown into Text-widget text/tag pairs."""
    compiled: list[tuple[str, str]] = []
    in_code = False
    for raw_line in markdown.replace("\r\n", "\n").split("\n"):
        fence = raw_line.strip().startswith("```")
        if fence:
            in_code = not in_code
            if not in_code:
                compiled.append(("\n", "code"))
            continue
        if in_code:
            compiled.append((raw_line + "\n", "code"))
            continue
        if not raw_line.strip():
            compiled.append(("\n", "blank"))
            continue
        heading = re.match(r"^(#{1,6})\s+(.+)$", raw_line)
        if heading:
            compiled.extend(_inline_tokens(heading.group(2)))
            compiled.append(("\n", f"heading{len(heading.group(1))}"))
            continue
        bullet = re.match(r"^\s*([-*])\s+(.+)$", raw_line)
        numbered = re.match(r"^\s*(\d+)[.)]\s+(.+)$", raw_line)
        quote = re.match(r"^\s*>\s?(.*)$", raw_line)
        if bullet:
            compiled.append(("• ", "bullet"))
            compiled.extend(_inline_tokens(bullet.group(2)))
        elif numbered:
            compiled.append((f"{numbered.group(1)}. ", "bullet"))
            compiled.extend(_inline_tokens(numbered.group(2)))
        elif quote:
            compiled.append(("│ ", "quote_marker"))
            compiled.extend(_inline_tokens(quote.group(1)))
        else:
            compiled.extend(_inline_tokens(raw_line))
        compiled.append(("\n", "body"))
    return compiled


def configure_markdown_tags(widget: tk.Text) -> None:
    base = tkfont.nametofont("TkDefaultFont").copy()
    base.configure(size=11)
    mono = tkfont.nametofont("TkFixedFont").copy()
    mono.configure(size=10)
    widget.configure(font=base, padx=16, pady=12, spacing1=2, spacing3=4,
                     background="#f7f8fa", foreground="#20242a",
                     insertbackground="#20242a", selectbackground="#cfe3ff")
    widget.tag_configure("speaker_user", foreground="#1261a0", font=(base.actual("family"), 10, "bold"), spacing1=12)
    widget.tag_configure("speaker_agent", foreground="#157347", font=(base.actual("family"), 10, "bold"), spacing1=12)
    for level, size in ((1, 18), (2, 15), (3, 13), (4, 12), (5, 11), (6, 11)):
        widget.tag_configure(f"heading{level}", font=(base.actual("family"), size, "bold"), foreground="#182230", spacing1=10, spacing3=5)
    widget.tag_configure("strong", font=(base.actual("family"), 11, "bold"))
    widget.tag_configure("inline_code", font=mono, foreground="#8a3b12", background="#eceff3")
    widget.tag_configure("code", font=mono, foreground="#e8edf2", background="#20252b", lmargin1=14, lmargin2=14, spacing1=0, spacing3=0)
    widget.tag_configure("bullet", foreground="#42617a", font=(base.actual("family"), 11, "bold"))
    widget.tag_configure("quote_marker", foreground="#4e7d8a", font=(base.actual("family"), 11, "bold"))
    widget.tag_configure("link", foreground="#1261a0", underline=True)
    widget.tag_configure("link_url", foreground="#6b7280")
    widget.tag_configure("blank", spacing1=3, spacing3=3)


def windows_path_candidates(message: str) -> tuple[str, ...]:
    """Windows directory paths a request may point at, longest first.

    Split out from the lookup so the parsing half is testable on any platform;
    only the existence check that consumes it is Windows-dependent, which keeps
    the surface that cannot be verified off Windows down to one call.
    """
    candidates: list[str] = []
    for raw in sorted(re.findall(r"[A-Za-z]:\\[^\"\r\n]+", message), key=len, reverse=True):
        cleaned = raw.rstrip(" 。、，:：」』")
        parts = cleaned.split("\\")
        for end in range(len(parts), 1, -1):
            candidate = "\\".join(parts[:end])
            if candidate not in candidates:
                candidates.append(candidate)
    return tuple(candidates)


class LocalWorkspaceAgent:
    #: Built on first use: the semantic session loads the frozen hold-out
    #: episodes, and a workspace-only session should not pay for that.
    _semantics: SemanticSession | None = None

    @property
    def semantics(self) -> SemanticSession:
        if self._semantics is None:
            self._semantics = SemanticSession()
        return self._semantics

    def __init__(self, workspace: Path | None = None):
        self.workspace = workspace.resolve() if workspace else None
        self.validation_knowledge = ()
        self.validation_memory = ValidationMemoryRetriever()
        if self.workspace:
            self._refresh_validation_memory()

    def _refresh_validation_memory(self) -> None:
        if self.workspace is None:
            self.validation_knowledge = ()
            self.validation_memory = ValidationMemoryRetriever()
            return
        store = self.workspace / ".conscious_coding_agent" / "validation_knowledge.jsonl.gz"
        self.validation_knowledge = ValidationCompressor().compress(self.workspace, store)
        self.validation_memory = ValidationMemoryRetriever(self.validation_knowledge)

    def set_workspace(self, workspace: Path) -> str:
        resolved = workspace.expanduser().resolve()
        if not resolved.is_dir():
            raise ValueError("作業フォルダが存在しません")
        self.workspace = resolved
        self._refresh_validation_memory()
        return f"作業フォルダを設定しました: {resolved}"

    def _workspace_in_message(self, message: str) -> Path | None:
        """Find an existing Windows directory embedded in a natural-language request."""
        for candidate in windows_path_candidates(message):
            path = Path(candidate)
            if path.is_dir():
                return path
        return None

    def _local_investigation(self, message: str) -> ChatResult:
        selected = self._workspace_in_message(message)
        if selected is not None:
            self.set_workspace(selected)
        if is_repair_proposal_request(message):
            if self.workspace is None:
                return ChatResult("外部修正提案を適用するため、先に作業フォルダを指定してください。", "local")
            repair_request = parse_external_repair_request(message)
            instruction = parse_repair_proposal(repair_request.proposal_body)
            task_spec = recognized_repair_task_spec(repair_request, instruction)
            root = self._root()
            plan = plan_repair(root, instruction)
            requirement_graph = build_requirement_graph(repair_request.implementation_items or instruction.requested_changes)
            impact = analyze_repository_impact(root, repair_request.implementation_items or instruction.requested_changes)
            # L8.5 compiles the semantic requirements down to CodeChangeIR first,
            # so the decomposition and the patch loop share one grounded view.
            semantic = SemanticPatchSynthesizer()
            compilation = semantic.compile(root, instruction, plan)
            synthesizer = CompositePatchSynthesizer((MinimalPatchSynthesizer(), semantic))
            decomposition = decompose_repair(instruction, plan, root)
            loop = run_autonomous_repair_loop(root, instruction, synthesizer=synthesizer)
            patch_report = format_autonomous_patch(loop.final_patch) if loop.final_patch else ""
            item_lines = ["## Implementation Items", ""] + ([f"{index}. {item}" for index, item in enumerate(repair_request.implementation_items, 1)] or ["- 抽出できる実装項目はありません。"])
            sections = [
                format_recognized_task_spec(task_spec),
                "\n".join(item_lines),
                format_requirement_graph(requirement_graph),
                format_repository_impact(impact),
                format_change_compilation(compilation),
                format_repair_decomposition(decomposition),
                format_autonomous_repair(loop),
            ]
            if patch_report:
                sections.append(patch_report)
            return ChatResult("\n\n".join(sections), "local")
        if is_local_paper_request(message):
            if self.workspace is None:
                return ChatResult("ローカルPDFを調査するため、先に作業フォルダを指定してください。", "local")
            pdfs = sorted(
                path for path in self.workspace.rglob("*.pdf")
                if path.is_file() and ".git" not in path.parts
            )
            if not pdfs:
                return ChatResult("作業フォルダ内にPDFが見つかりませんでした。", "local")
            knowledge = extract_local_paper(pdfs[0])
            suffix = ""
            if len(pdfs) > 1:
                suffix = f"\n\n候補 {len(pdfs)} 件中、先頭のPDFを解析しました: {pdfs[0].name}"
            quality = assess_paper_quality(knowledge)
            evidence = paper_to_scientific_evidence(knowledge)
            report = "\n\n".join((
                format_paper_knowledge(knowledge),
                format_paper_quality_report(quality),
                format_integrated_evidence(evidence),
            ))
            return ChatResult(report + suffix, "local")
        # Explicit filesystem actions must run before validation-memory recall.
        if is_move_directories_request(message):
            if self.workspace is None:
                return ChatResult("移動対象を解決するため、先に作業フォルダを指定してください。", "local")
            operation = parse_move_operation(message)
            results = execute_move_operation(self._root(), operation)
            return ChatResult(format_operation_report(operation, results), "local")
        if is_delete_directory_request(message):
            if self.workspace is None:
                return ChatResult("削除対象を解決するため、先に作業フォルダを指定してください。", "local")
            operation = parse_delete_operation(message)
            return ChatResult(format_operation_report(operation, execute_delete_operation(self._root(), operation)), "local")
        if is_math_derivation_request(message):
            return ChatResult(format_derivation_report(derive_1d_wave_equation(message)), "local")
        if is_pairwise_geometry_request(message):
            result = run_compiled_experiment(compile_pairwise_distance_experiment())
            return ChatResult(format_compiled_experiment_report(compile_pairwise_distance_experiment(), result), "local")
        if is_scientific_research_request(message):
            plan = build_research_plan(message)
            retriever = ScientificRetriever(HttpSearchAdapter(), HttpPageFetcher()) if live_web_enabled() else ScientificRetriever()
            retrieval = retriever.resolve_task(message)
            evidence_text = "\n\n【検索・Evidence統合】\n" + "\n\n".join(format_retrieval_outcome(item) for item in retrieval)
            return ChatResult(format_research_report(plan) + evidence_text, "local")
        if is_open_research_request(message):
            plan = design_open_research_plan(message)
            result = run_open_research_plan(plan)
            return ChatResult(format_open_research_report(plan, result), "local")
        if is_design_and_verify_request(message):
            plan = design_relativistic_garage_experiment(message)
            result = run_experiment(plan)
            return ChatResult(format_verification_report(plan, result), "local")
        if self.workspace is None:
            return self._answer_without_workspace(message)
        if is_scientific_simulation_implementation_request(message):
            result = implement_scientific_simulation(self._root(), message)
            return ChatResult(format_simulation_implementation_report(result), "local")
        intent = classify_request(message)
        if intent.request_type is RequestType.TEST:
            return ChatResult(self.tests(), "local")
        if intent.request_type is RequestType.EXECUTE:
            result = execute_request(self._root(), message)
            return ChatResult(format_execution_result(result), "local")
        if is_math_explanation_request(message):
            model = extract_mathematical_model(self._root(), message)
            return ChatResult("ローカル調査結果です。\n\n" + format_math_report(model), "local")
        if is_experiment_configuration_request(message):
            spec = extract_experiment_spec(self._root(), message)
            return ChatResult("ローカル調査結果です。\n\n" + format_experiment_report(spec), "local")
        if is_algorithm_summary_request(message):
            summary = summarize_algorithm(self._root(), message)
            return ChatResult(format_algorithm_summary(summary), "local")
        if self.validation_knowledge and any(word in message.lower() for word in ("検証", "validation", "保存則", "不変", "robustness", "audit")):
            reusable = self.validation_memory.retrieve(message)
            if reusable:
                return ChatResult(format_validation_memory(reusable), "local")
        if intent.target and intent.request_type.value in {"summarize", "explain"}:
            report = structured_repository_report(self._root(), message)
            return ChatResult("ローカル調査結果です。\n\n" + report, "local")
        if self._is_explanation_request(message):
            grounded = explain_repository_question(self._root(), self._semantic_question(message), max_claims=8)
            if grounded != "Code Context Graphから質問に対応するsymbolを特定できませんでした。":
                commands = "\n\n続けて実行できます: /inspect /tests /read <path> /find <term>"
                return ChatResult("ローカル調査結果です。\n\n" + grounded + commands, "local")
        summary = self.summarize_compact(message)
        commands = "\n\n続けて実行できます: /inspect /tests /read <path> /find <term>"
        return ChatResult("ローカル調査結果です。\n\n" + summary + commands, "local")

    def _answer_without_workspace(self, message: str) -> ChatResult:
        """Answer from the request itself when no local workspace is selected."""
        plan = compile_goal_driven_plan(message)
        text = (
            "ローカルフォルダ未指定のため、ファイル一覧・検索・コード読取は行っていません。\n\n"
            "フォルダに依存しない範囲で、指示を GoalIR / RequirementGraph / TaskGraph として整理します。\n\n"
            f"{format_goal_driven_plan(plan)}"
        )
        return ChatResult(text, "no-workspace")

    @staticmethod
    def _is_explanation_request(message: str) -> bool:
        lower = message.lower()
        return any(term in message for term in ("解説", "説明", "どのように", "どう作", "どこから")) or any(
            term in lower for term in ("explain", "how is", "where does", "call chain", "data flow")
        )

    @staticmethod
    def _semantic_question(message: str) -> str:
        lower = message.lower()
        if any(term in message for term in ("初期", "量子", "状態")) or "initial" in lower or "quantum" in lower:
            # Keep the query centered on the lifecycle concept; adding quantum
            # over-ranks unrelated quantum-state experiments in large repos.
            return "initial state"
        if "呼" in message or "call" in lower:
            return "call chain entry point"
        if "流" in message or "data flow" in lower or "どこから" in message:
            return "data flow variable"
        return message

    def _root(self) -> Path:
        if self.workspace is None:
            raise ValueError("先に作業フォルダを指定してください")
        return self.workspace

    def _safe_path(self, relative: str) -> Path:
        root = self._root()
        target = (root / relative).resolve()
        if target != root and root not in target.parents:
            raise ValueError("workspace外のパスにはアクセスできません")
        return target

    def inspect(self) -> str:
        root = self._root()
        files = [str(path.relative_to(root)) for path in root.rglob("*") if path.is_file() and ".git" not in path.parts]
        files.sort()
        shown = files[:80]
        suffix = f"\n... 他 {len(files) - len(shown)} 件" if len(files) > len(shown) else ""
        return f"{root}\nファイル {len(files)} 件:\n" + "\n".join(shown) + suffix

    def read(self, relative: str, limit: int = 12000) -> str:
        path = self._safe_path(relative)
        if not path.is_file():
            raise ValueError("ファイルが見つかりません")
        return f"[{path.relative_to(self._root())}]\n{path.read_text(encoding='utf-8', errors='replace')[:limit]}"

    def find(self, term: str) -> str:
        root = self._root()
        matches = []
        for path in root.rglob("*"):
            if not path.is_file() or ".git" in path.parts:
                continue
            try:
                for line_no, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
                    if term.lower() in line.lower():
                        matches.append(f"{path.relative_to(root)}:{line_no}: {line.strip()[:180]}")
            except OSError:
                continue
        return "\n".join(matches[:100]) or "該当なし"

    def tests(self) -> str:
        root = self._root()
        command = ("python", "-m", "unittest", "discover")
        completed = run_text(list(command), cwd=root, timeout=60)
        output = combined_output(completed).strip()
        return f"$ {' '.join(command)}\n終了コード: {completed.returncode}\n{output[-8000:]}"

    def self_benchmark(self) -> str:
        with tempfile.TemporaryDirectory(prefix="l86-ui-benchmark-") as directory:
            return format_self_modification_report(run_self_modification_benchmark(Path(directory)))

    def holdout_benchmark(self) -> str:
        with tempfile.TemporaryDirectory(prefix="l861-ui-holdout-") as directory:
            return format_holdout_comparison(compare_known_and_holdout(Path(directory)))

    @staticmethod
    def _build_behavior_audit_fixture(root: Path) -> None:
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

    def behavior_audit(self) -> str:
        with tempfile.TemporaryDirectory(prefix="l87-ui-audit-") as directory:
            root = Path(directory)
            self._build_behavior_audit_fixture(root)
            return format_behavior_audit(audit_agent(root, DEFAULT_AUDIT_REQUESTS))

    def summarize(self, request: str) -> str:
        root = self._root()
        normalized = request.lower().replace("pre big ban", "big bang").replace("pre-big-ban", "big bang")
        if "big bang" in normalized:
            terms = ["big bang", "big-bang", "big_bang", "pre big ban", "pre-big-ban", "pre_big_ban",
                     "singularity", "pre-geometry", "初期宇宙", "初期量子状態", "量子状態",
                     "initial quantum state", "initial_quantum_state", "quantum_state"]
        else:
            terms = [term for term in re.findall(r"[a-z][a-z0-9_-]{2,}|[一-龥ぁ-んァ-ヶ]{2,}", normalized)
                     if term not in {"まとめて", "内容", "検証", "について"}]
        extensions = {".md", ".txt", ".json", ".py", ".tex"}
        excluded = {".git", ".venv", "__pycache__", ".pytest_cache", "node_modules"}
        matches: list[tuple[Path, list[str], list[str]]] = []
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in extensions or excluded.intersection(path.parts):
                continue
            try:
                lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
            except OSError:
                continue
            hits = [f"{index}: {line.strip()[:240]}" for index, line in enumerate(lines, 1)
                    if any(term in line.lower() for term in terms)]
            if hits:
                headings = [line.strip() for line in lines if line.lstrip().startswith("#")][:8]
                matches.append((path, headings, hits[:8]))
        matches.sort(key=lambda item: (-len(item[2]), str(item[0])))
        if not matches:
            return "ローカル要約: 指定語に一致する資料が見つかりませんでした。"
        sections = ["ローカル要約"]
        for path, headings, hits in matches[:12]:
            sections.append(f"\n[{path.relative_to(root)}]")
            if headings:
                sections.append("見出し: " + " / ".join(headings))
            sections.append("該当箇所:\n" + "\n".join(hits))
        return "\n".join(sections)

    def summarize_compact(self, request: str, max_chars: int = 4500) -> str:
        """Return a concise, API-free investigation digest for the transcript."""
        root = self._root()
        normalized = request.lower().replace("pre big ban", "big bang").replace("pre-big-ban", "big bang")
        if "big bang" in normalized:
            terms = ["big bang", "big-bang", "big_bang", "pre big ban", "pre-big-ban", "pre_big_ban",
                     "singularity", "pre-geometry", "初期宇宙", "初期量子状態", "量子状態",
                     "initial quantum state", "initial_quantum_state", "quantum_state"]
        else:
            terms = [term for term in re.findall(r"[a-z][a-z0-9_-]{2,}|[一-龥ぁ-んァ-ヶ]{2,}", normalized)
                     if term not in {"まとめて", "内容", "検証", "について", "説明", "解説"}]
        extensions = {".md", ".txt", ".json", ".py", ".tex"}
        excluded = {".git", ".venv", "__pycache__", ".pytest_cache", "node_modules"}
        matches: list[tuple[int, Path, list[str]]] = []
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in extensions or excluded.intersection(path.parts):
                continue
            try:
                lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
            except OSError:
                continue
            hits = [f"{index}: {line.strip()[:180]}" for index, line in enumerate(lines, 1)
                    if any(term in line.lower() for term in terms)]
            if hits:
                matches.append((len(hits), path, hits[:2]))
        matches.sort(key=lambda item: (-item[0], str(item[1])))
        if not matches:
            candidates = [
                path for path in sorted(root.rglob("*"))
                if path.is_file() and path.suffix.lower() in extensions and not excluded.intersection(path.parts)
            ][:8]
            if not candidates:
                return "ローカル要約: 調査対象にできるテキスト/コードファイルが見つかりませんでした。"
            lines = ["ローカル要約（検索語一致なし・ファイル概観）", f"対象: {root}", f"候補ファイル: {len(candidates)}件"]
            for path in candidates:
                try:
                    line_count = len(path.read_text(encoding="utf-8", errors="ignore").splitlines())
                except OSError:
                    line_count = 0
                lines.append(f"- {path.relative_to(root)} ({line_count}行)")
            lines.append("次の操作: /read <path>  /find <term>  /tests")
            return "\n".join(lines)[:max_chars]
        lines = ["ローカル要約（API不要・圧縮表示）", f"対象: {root}", f"一致ファイル: {len(matches)}件"]
        for count, path, hits in matches[:5]:
            lines.append(f"- {path.relative_to(root)} ({count}件): " + " / ".join(hits))
        lines.append("次の操作: /read <path>  /find <term>  /tests")
        return "\n".join(lines)[:max_chars]

    def handle(self, message: str) -> ChatResult:
        try:
            # A pending question owns the next utterance (L8.33): the reply is
            # only interpretable against the question it answers, so it is bound
            # there before anything else looks at it.
            spoken = handle_command(self.semantics, message)
            if spoken is None and self._semantics is not None \
                    and self._semantics.dialogue.pending is not None:
                state = self._semantics.ask(message)
                spoken = f"{state['reply']}\n\n" + render_state(state)
            if spoken is not None:
                return ChatResult(spoken, "semantics")
            if message == "/help":
                feature_lines = "\n".join(f"{command}  {description}" for command, description in AGENT_FEATURE_COMMANDS)
                return ChatResult(feature_lines + "\n/read <path>  ファイル表示\n/find <term>  workspace検索\n自然文  ローカル調査", "local")
            if message == "/inspect":
                return ChatResult(self.inspect(), "local")
            if message == "/tests":
                return ChatResult(self.tests(), "local")
            if message == "/self-benchmark":
                return ChatResult(self.self_benchmark(), "local")
            if message == "/holdout-benchmark":
                return ChatResult(self.holdout_benchmark(), "local")
            if message == "/behavior-audit":
                return ChatResult(self.behavior_audit(), "local")
            if message.startswith("/goal "):
                return ChatResult(format_goal_driven_plan(compile_goal_driven_plan(message[6:].strip())), "local")
            if message.startswith("/read "):
                return ChatResult(self.read(message[6:].strip()), "local")
            if message.startswith("/find "):
                return ChatResult(self.find(message[6:].strip()), "local")
            selected = self._workspace_in_message(message)
            if selected is not None:
                self.set_workspace(selected)
            return self._local_investigation(message)
        except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
            return ChatResult(f"操作を実行できませんでした: {exc}", "local")


class CodingAgentApp:
    def __init__(self, root: tk.Tk, agent: LocalWorkspaceAgent | None = None):
        if tk is None:  # pragma: no cover - exercised on headless machines
            raise RuntimeError(
                "tkinter がこの環境にありません。デスクトップ画面のかわりに "
                "coding_agent_bridge（HTTP API）と web UI を使ってください。")
        self.root = root
        self.agent = agent or LocalWorkspaceAgent()
        self.messages: list[dict[str, str]] = []
        self.session_path = Path(__file__).with_name(".conversation.json.gz")
        self.memory = ConversationKnowledgeMemory(Path(__file__).with_name(".knowledge_memory.json.gz"))
        self.root.title("Conscious Coding Agent")
        self.root.geometry("1000x700")
        self.root.minsize(720, 480)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self._build()
        self._restore_session()

    def _build(self) -> None:
        toolbar = ttk.Frame(self.root, padding=8)
        toolbar.pack(fill="x")
        ttk.Button(toolbar, text="作業フォルダを指定", command=self.choose_workspace).pack(side="left")
        self.workspace_label = ttk.Label(toolbar, text=str(self.agent.workspace or "未指定"))
        self.workspace_label.pack(side="left", padx=12)
        self.status = ttk.Label(toolbar, text="待機中")
        self.status.pack(side="right")
        transcript_frame = ttk.Frame(self.root)
        transcript_frame.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self.transcript = tk.Text(transcript_frame, wrap="word", state="disabled", height=30,
                                  undo=False, borderwidth=0, relief="flat")
        configure_markdown_tags(self.transcript)
        scrollbar = ttk.Scrollbar(transcript_frame, orient="vertical", command=self.transcript.yview)
        self.transcript.configure(yscrollcommand=scrollbar.set)
        self.transcript.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        bottom = ttk.Frame(self.root, padding=(8, 4, 8, 8))
        bottom.pack(fill="x")
        ttk.Label(bottom, text="指示 / Instruction").pack(side="left", padx=(0, 8))
        self.entry = ttk.Entry(bottom, width=80)
        self.entry.pack(side="left", fill="x", expand=True, ipady=5)
        self.entry.bind("<Return>", lambda _event: self.send())
        ttk.Button(bottom, text="保存", command=self.save).pack(side="left", padx=(8, 4))
        ttk.Button(bottom, text="送信", command=self.send).pack(side="left")
        self._append("Agent", "作業フォルダを指定してください。`/help`で操作を確認できます。")

    def _append(self, speaker: str, text: str) -> None:
        self.messages.append({"speaker": speaker, "text": text})
        self.transcript.configure(state="normal")
        speaker_tag = "speaker_user" if speaker == "You" else "speaker_agent"
        self.transcript.insert("end", f"\n{speaker}\n", speaker_tag)
        for value, tag in compile_markdown(text):
            self.transcript.insert("end", value, tag)
        self.transcript.configure(state="disabled")
        self.transcript.see("end")

    def _restore_session(self) -> None:
        workspace, messages = load_session(self.session_path)
        if workspace and workspace.is_dir() and self.agent.workspace is None:
            self.agent.set_workspace(workspace)
            self.workspace_label.configure(text=str(self.agent.workspace))
        if not messages:
            return
        self.transcript.configure(state="normal")
        self.transcript.delete("1.0", "end")
        self.transcript.configure(state="disabled")
        self.messages = []
        for item in messages:
            self._append(item["speaker"], item["text"])
        self._append("Agent", "前回の会話を復元しました。続きの指示を入力できます。")

    def save(self) -> None:
        save_session(self.session_path, self.messages, self.agent.workspace)
        if self.agent.workspace:
            log_path = save_work_log(self.agent.workspace, self.messages)
            self.status.configure(text=f"保存済み: {log_path.name}")
        else:
            self.status.configure(text="会話を保存しました")

    def close(self) -> None:
        self.memory.save()
        self.save()
        self.root.destroy()

    def choose_workspace(self) -> None:
        selected = filedialog.askdirectory(title="作業フォルダを選択")
        if not selected:
            return
        try:
            message = self.agent.set_workspace(Path(selected))
            self.workspace_label.configure(text=str(self.agent.workspace))
            self._append("Agent", message)
        except ValueError as exc:
            messagebox.showerror("作業フォルダ", str(exc))

    def send(self) -> None:
        message = self.entry.get().strip()
        if not message:
            return
        self.entry.delete(0, "end")
        self._append("You", message)
        self.status.configure(text="実行中...")

        def work() -> None:
            result = self.agent.handle(message)
            policy = memory_policy_for_message(message)
            remembered = self.memory.context(message, policy=policy)
            if remembered and policy == "all":
                result = ChatResult(result.text + "\n\n【再利用された知識】\n" + remembered, result.provider)
            self.memory.learn(message, result.text)
            self.memory.save()
            self.root.after(0, lambda: (self.workspace_label.configure(text=str(self.agent.workspace or "未指定")),
                                        self._append(f"Agent [{result.provider}]", result.text),
                                        self.status.configure(text="待機中")))

        threading.Thread(target=work, daemon=True).start()


def main() -> None:
    root = tk.Tk()
    CodingAgentApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
