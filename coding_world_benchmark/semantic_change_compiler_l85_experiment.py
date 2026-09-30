"""L8.5: compile semantic repair requirements into concrete code changes.

L8.1 grounds an external proposal, L8.3 orders it, L8.2 applies minimal
diffs.  The missing layer was the translation between them: a proposal
sentence such as ``既存Routerの前にSemantic Parserを置く`` carries a semantic
requirement, not an edit.  This module performs

    RepairInstruction
      -> SemanticRequirement   (verb / artifact / anchor / members)
      -> repository grounding  (which real file and symbol is meant)
      -> CodeChangeIR          (file, symbol, operation, anchor, behavior)
      -> PatchIntent / PatchCandidate for the L8.2 synthesizer.

Every stage refuses to guess.  When the artifact has no identifier, or the
anchor cannot be found in the repository, the requirement is emitted as an
*unresolved* CodeChangeIR with a reason instead of a fabricated patch.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
import difflib
from enum import Enum
import os
from pathlib import Path
import re

from .autonomous_patch_l82_experiment import PatchCandidate, PatchIntent
from .external_repair_l81_experiment import RepairInstruction, RepairPlan
from .repository_semantic_l58_experiment import RepositoryGraph, index_repository


class ChangeOperation(Enum):
    """Formalized edit vocabulary understood by the L8.2 synthesizer."""

    ADD_DATACLASS = "ADD_DATACLASS"
    ADD_CLASS = "ADD_CLASS"
    ADD_FUNCTION = "ADD_FUNCTION"
    ADD_METHOD = "ADD_METHOD"
    MODIFY_ENUM = "MODIFY_ENUM"
    INSERT_ROUTE = "INSERT_ROUTE"
    REPLACE_BRANCH = "REPLACE_BRANCH"
    ADD_TEST = "ADD_TEST"
    ADD_IMPORT = "ADD_IMPORT"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class SemanticRequirement:
    """What a single proposal sentence asks for, before repository grounding."""

    text: str
    operation: ChangeOperation
    artifact: str
    target_file_hint: str
    anchor: str
    members: tuple[str, ...]
    expected_behavior: str
    requires_execution: bool


@dataclass(frozen=True)
class CodeChangeIR:
    """A grounded, executable description of one code edit."""

    file: str
    symbol: str | None
    operation: ChangeOperation
    anchor: str | None
    behavior: str
    members: tuple[str, ...]
    creates_file: bool
    status: str
    reason: str
    evidence: tuple[str, ...]
    confidence: float
    requirement: SemanticRequirement


@dataclass(frozen=True)
class ChangeCompilation:
    changes: tuple[CodeChangeIR, ...]
    rationale: tuple[str, ...]

    @property
    def compiled(self) -> tuple[CodeChangeIR, ...]:
        return tuple(item for item in self.changes if item.status == "compiled")

    @property
    def unresolved(self) -> tuple[CodeChangeIR, ...]:
        return tuple(item for item in self.changes if item.status != "compiled")


# --------------------------------------------------------------------------
# stage 1: natural language -> SemanticRequirement
# --------------------------------------------------------------------------

_KEYWORD_STOP = {
    "add", "added", "adds", "new", "class", "classes", "function", "functions", "method",
    "methods", "dataclass", "enum", "test", "tests", "file", "files", "module", "modules",
    "the", "a", "an", "to", "in", "into", "of", "for", "and", "or", "with", "before",
    "after", "front", "insert", "inserts", "route", "routes", "routing", "replace",
    "branch", "condition", "return", "returns", "type", "types", "field", "fields",
    "regression", "implement", "implements", "implementation", "support", "supports",
    "at", "on", "from", "by", "then", "when", "if",
    "定義", "追加", "実装", "変更", "修正", "配置", "降格", "作成",
}

_MEMBER_LABELS = ("fields", "field", "members", "member", "attributes", "フィールド", "属性", "メンバー", "要素")


def _first_python_file(text: str) -> str:
    match = re.search(r"([A-Za-z_][A-Za-z0-9_\-]*(?:[/\\][A-Za-z0-9_\-]+)*\.py)", text)
    return match.group(1).replace("\\", "/") if match else ""


# Japanese text has no ASCII word boundary (``TaskIR型`` keeps \b from matching),
# so identifier scanning brackets the ASCII character class explicitly.
_L = r"(?<![A-Za-z0-9_])"
_R = r"(?![A-Za-z0-9_])"


def _identifiers(text: str) -> list[str]:
    """Identifier-shaped tokens, richest first (backticked > CamelCase > snake)."""
    backticked = re.findall(r"`([A-Za-z_][A-Za-z0-9_.]*)`", text)
    camel = re.findall(rf"{_L}([A-Z][A-Za-z0-9]*[a-z][A-Za-z0-9]*){_R}", text)
    upper = re.findall(rf"{_L}([A-Z][A-Z0-9_]{{2,}}){_R}", text)
    snake = re.findall(rf"{_L}([a-z_][a-z0-9_]{{2,}}){_R}", text)
    ordered: list[str] = []
    for token in (*backticked, *camel, *upper, *snake):
        clean = token.strip(".")
        if clean and clean.lower() not in _KEYWORD_STOP and clean not in ordered:
            ordered.append(clean)
    return ordered


def _detect_operation(text: str) -> ChangeOperation:
    lower = text.lower()
    has_anchor = bool(_detect_anchor(text))
    if re.search(r"\btests?\b|テスト|回帰", lower) and re.search(r"\badd|追加|作成|書く", lower):
        return ChangeOperation.ADD_TEST
    if re.search(r"test_[\w]*\.py", lower):
        return ChangeOperation.ADD_TEST
    if re.search(r"\bdataclass\b|データクラス|\bir型|型を(?:追加|定義|新設)|型定義", lower):
        return ChangeOperation.ADD_DATACLASS
    if re.search(r"\benum\b|列挙", lower):
        return ChangeOperation.MODIFY_ENUM if re.search(r"member|メンバー|値を|に追加|へ追加", lower) else ChangeOperation.ADD_CLASS
    if re.search(r"\bimport\b|インポート", lower):
        return ChangeOperation.ADD_IMPORT
    if has_anchor and re.search(r"insert|配置|置く|挿入|route|dispatch|前に|後に|先に", lower):
        return ChangeOperation.INSERT_ROUTE
    if re.search(r"replace|置換|差し替え|書き換え", lower) and re.search(r"branch|condition|分岐|条件|if", lower):
        return ChangeOperation.REPLACE_BRANCH
    if re.search(r"\bmethod\b|メソッド", lower):
        return ChangeOperation.ADD_METHOD
    if re.search(r"\bfunctions?\b|\bdef\b|関数", lower):
        return ChangeOperation.ADD_FUNCTION
    if re.search(r"\bclass(?:es)?\b|クラス|parser|synthesizer|router|compiler", lower) and re.search(r"add|追加|新設|導入|実装", lower):
        return ChangeOperation.ADD_CLASS
    return ChangeOperation.UNKNOWN


def _detect_anchor(text: str) -> str:
    english = re.search(rf"{_L}(?:before|ahead of|in front of)\s+`?([A-Za-z_][A-Za-z0-9_.]*)`?", text, re.I)
    if english:
        return english.group(1)
    japanese = re.search(r"`?([A-Za-z_][A-Za-z0-9_.]*)`?\s*(?:の)?\s*(?:直前|手前|前)\s*に", text)
    if japanese:
        return japanese.group(1)
    # ``<anchor> -> <new condition>`` names the thing being replaced on the left.
    arrow = re.search(r"`?([A-Za-z_][A-Za-z0-9_.]*)`?\s*(?:→|->|=>)", text)
    if arrow and arrow.group(1).lower() not in _KEYWORD_STOP:
        return arrow.group(1)
    branch = re.search(rf"(?:branch|condition|分岐|条件)\s+`?([A-Za-z_][A-Za-z0-9_.]*)`?", text, re.I)
    if branch and branch.group(1).lower() not in _KEYWORD_STOP:
        return branch.group(1)
    return ""


def _detect_members(text: str) -> tuple[str, ...]:
    chunks: list[str] = []
    for match in re.finditer(r"\(([^)]*)\)", text):
        chunks.append(match.group(1))
    for label in _MEMBER_LABELS:
        match = re.search(rf"{re.escape(label)}\s*[:：]?\s*(.+)$", text, re.I)
        if match:
            chunks.append(match.group(1))
    members: list[str] = []
    for chunk in chunks:
        for token in re.split(r"[,、\s/]+", chunk):
            token = token.strip("`'\"")
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", token) and token.lower() not in _KEYWORD_STOP and token not in members:
                members.append(token)
    return tuple(members)


def _is_bare_member_line(text: str) -> bool:
    """A continuation bullet such as ``- requires_execution`` under a type."""
    stripped = text.strip().strip("-*・ ")
    return bool(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", stripped)) and stripped.lower() not in _KEYWORD_STOP


def extract_semantic_requirement(change: str) -> SemanticRequirement:
    text = change.strip()
    operation = _detect_operation(text)
    anchor = _detect_anchor(text)
    members = _detect_members(text)
    file_hint = _first_python_file(text)
    candidates = [item for item in _identifiers(text) if item != anchor and not item.endswith(".py")]
    candidates = [item for item in candidates if item not in members] or candidates
    if file_hint:
        stem = Path(file_hint).stem
        candidates = [item for item in candidates if item != stem] or candidates
    artifact = candidates[0] if candidates else ""
    requires_execution = bool(re.search(r"実行|run|execute|verify|検証|テスト|test", text, re.I))
    return SemanticRequirement(text, operation, artifact, file_hint, anchor, members, text, requires_execution)


def extract_semantic_requirements(changes: tuple[str, ...]) -> tuple[SemanticRequirement, ...]:
    """Parse each change, folding bare continuation bullets into the previous type."""
    requirements: list[SemanticRequirement] = []
    for change in changes:
        if requirements and _is_bare_member_line(change):
            previous = requirements[-1]
            if previous.operation in {ChangeOperation.ADD_DATACLASS, ChangeOperation.MODIFY_ENUM, ChangeOperation.ADD_CLASS}:
                member = change.strip().strip("-*・ ")
                requirements[-1] = SemanticRequirement(
                    previous.text, previous.operation, previous.artifact, previous.target_file_hint,
                    previous.anchor, previous.members + (member,), previous.expected_behavior,
                    previous.requires_execution,
                )
                continue
        requirements.append(extract_semantic_requirement(change))
    return tuple(requirements)


# --------------------------------------------------------------------------
# stage 2: repository grounding -> CodeChangeIR
# --------------------------------------------------------------------------

def _repository_files(root: Path) -> tuple[str, ...]:
    return tuple(
        str(path.relative_to(root)).replace("\\", "/")
        for path in sorted(root.rglob("*.py"))
        if not {".git", ".venv", "__pycache__", ".pytest_cache"}.intersection(path.parts)
    )


def _match_file(files: tuple[str, ...], hint: str) -> str:
    if not hint:
        return ""
    name = Path(hint).name.lower()
    exact = [item for item in files if item.lower() == hint.lower()]
    by_name = [item for item in files if Path(item).name.lower() == name]
    return (exact or by_name or [""])[0]


def _find_anchor(root: Path, files: tuple[str, ...], anchor: str, preferred: str = "") -> tuple[str, int]:
    """Locate an anchor, preferring a use site inside a function over its definition."""
    if not anchor:
        return "", 0
    pattern = re.compile(rf"{_L}{re.escape(anchor)}{_R}")
    definition = re.compile(rf"^\s*(?:class|def|async def)\s+{re.escape(anchor)}{_R}")
    ordered = ([preferred] if preferred in files else []) + [item for item in files if item != preferred]
    best: tuple[int, str, int] | None = None
    for relative in ordered:
        try:
            source = (root / relative).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for index, line in enumerate(source.splitlines(), 1):
            stripped = line.strip()
            if not pattern.search(line) or stripped.startswith(("#", "import ", "from ")):
                continue
            inside = _enclosing_function(source, index) is not None
            rank = 0 if inside and not definition.match(line) else 1 if inside else 2
            if best is None or rank < best[0]:
                best = (rank, relative, index)
            if rank == 0:
                break
        if best and best[0] == 0:
            break
    return (best[1], best[2]) if best else ("", 0)


def _enclosing_function(source: str, line: int) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    best: ast.FunctionDef | ast.AsyncFunctionDef | None = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.lineno <= line <= (node.end_lineno or node.lineno):
            if best is None or node.lineno > best.lineno:
                best = node
    return best


def _class_node(source: str, name: str) -> ast.ClassDef | None:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    return next((node for node in ast.walk(tree) if isinstance(node, ast.ClassDef) and node.name == name), None)


def _function_arity(source: str, name: str) -> int | None:
    """Positional arity of a top-level function, or None when it is not defined."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return len([arg for arg in node.args.args if arg.arg not in {"self", "cls"}])
    return None


def _symbol_defined(source: str, name: str) -> bool:
    return bool(re.search(rf"^\s*(?:class|def|async def)\s+{re.escape(name)}{_R}", source, re.M))


def _unresolved(requirement: SemanticRequirement, reason: str, file: str = "") -> CodeChangeIR:
    return CodeChangeIR(file, None, requirement.operation, requirement.anchor or None, requirement.expected_behavior,
                        requirement.members, False, "unresolved", reason, (), 0.0, requirement)


def compile_semantic_changes(
    root: Path,
    instruction: RepairInstruction,
    plan: RepairPlan,
    graph: RepositoryGraph | None = None,
) -> ChangeCompilation:
    """Ground every requested change against the real repository."""
    root = Path(root)
    graph = graph or index_repository(root)
    files = _repository_files(root)
    requirements = extract_semantic_requirements(instruction.requested_changes)
    default_file = plan.resolved_files[0] if plan.resolved_files else (_match_file(files, instruction.target_files[0]) if instruction.target_files else "")
    rationale = [
        "提案文をSemanticRequirement(動詞・対象・anchor・メンバー)へ分解しました。",
        "対象ファイル/symbolはRepositoryGraphと実ファイル走査で再解決しました。",
        "識別子またはanchorが実リポジトリに接地しない要求はunresolvedとして残し、推測生成しません。",
    ]
    changes: list[CodeChangeIR] = []
    created_symbols: list[tuple[str, str]] = []

    for requirement in requirements:
        operation = requirement.operation
        hinted = _match_file(files, requirement.target_file_hint)
        creates_file = bool(requirement.target_file_hint) and not hinted
        target = hinted or (requirement.target_file_hint if creates_file else default_file)

        if operation is ChangeOperation.UNKNOWN:
            changes.append(_unresolved(requirement, "変更内容を定型operationへ写像できませんでした。", target))
            continue

        if operation is ChangeOperation.ADD_TEST:
            source_change = next((item for item in reversed(changes) if item.status == "compiled" and item.symbol), None)
            if source_change is None and not created_symbols:
                changes.append(_unresolved(requirement, "検証対象のsymbolが先行stepに存在しません。", target))
                continue
            subject_file, subject_symbol = (source_change.file, source_change.symbol) if source_change else created_symbols[-1]
            test_file = requirement.target_file_hint or f"test_{Path(subject_file).stem}.py"
            if not creates_file:
                test_file = _match_file(files, test_file) or test_file
            changes.append(CodeChangeIR(
                test_file, f"test_{_snake(subject_symbol)}_is_defined", operation, subject_symbol,
                requirement.expected_behavior, (subject_file,), test_file not in files, "compiled", "",
                (f"{subject_file}:{subject_symbol}",), 0.8, requirement,
            ))
            continue

        if not target:
            changes.append(_unresolved(requirement, "対象ファイルを解決できませんでした。"))
            continue

        if operation in {ChangeOperation.INSERT_ROUTE, ChangeOperation.REPLACE_BRANCH}:
            anchor_file, anchor_line = _find_anchor(root, files, requirement.anchor, target)
            if not anchor_file:
                changes.append(_unresolved(requirement, f"anchor `{requirement.anchor or '-'}` を実リポジトリで特定できませんでした。", target))
                continue
            source = (root / anchor_file).read_text(encoding="utf-8", errors="replace")
            function = _enclosing_function(source, anchor_line)
            if function is None:
                changes.append(_unresolved(requirement, "anchorが関数本体の内側にないため安全に挿入できません。", anchor_file))
                continue
            if not requirement.artifact:
                changes.append(_unresolved(requirement, "挿入する処理の識別子を特定できませんでした。", anchor_file))
                continue
            wiring = "hook:existing" if _function_arity(source, requirement.artifact) is not None else "hook:generated"
            changes.append(CodeChangeIR(
                anchor_file, function.name, operation, requirement.anchor, requirement.expected_behavior,
                requirement.members, False, "compiled", "",
                (f"{anchor_file}:{anchor_line}", f"enclosing:{function.name}", wiring),
                0.85 if wiring == "hook:existing" else 0.72, requirement,
            ))
            created_symbols.append((anchor_file, requirement.artifact))
            continue

        if not requirement.artifact:
            changes.append(_unresolved(requirement, "追加する識別子を提案文から特定できませんでした。", target))
            continue

        if operation is ChangeOperation.MODIFY_ENUM:
            source = (root / target).read_text(encoding="utf-8", errors="replace") if target in files else ""
            enum_name = next((item for item in _identifiers(requirement.text) if _class_node(source, item)), "")
            if not enum_name:
                changes.append(_unresolved(requirement, "追加先のEnumクラスを実リポジトリで特定できませんでした。", target))
                continue
            members = tuple(item for item in (requirement.members or (requirement.artifact,)) if item != enum_name)
            if not members:
                changes.append(_unresolved(requirement, "追加するEnumメンバーを特定できませんでした。", target))
                continue
            changes.append(CodeChangeIR(target, enum_name, operation, None, requirement.expected_behavior, members,
                                        False, "compiled", "", (f"{target}:{enum_name}",), 0.8, requirement))
            continue

        if operation is ChangeOperation.ADD_METHOD:
            source = (root / target).read_text(encoding="utf-8", errors="replace") if target in files else ""
            owner = next((item for item in _identifiers(requirement.text) if item != requirement.artifact and _class_node(source, item)), "")
            if not owner:
                changes.append(_unresolved(requirement, "メソッドを追加するクラスを特定できませんでした。", target))
                continue
            changes.append(CodeChangeIR(target, f"{owner}.{requirement.artifact}", operation, owner,
                                        requirement.expected_behavior, requirement.members, False, "compiled", "",
                                        (f"{target}:{owner}",), 0.75, requirement))
            created_symbols.append((target, requirement.artifact))
            continue

        source = (root / target).read_text(encoding="utf-8", errors="replace") if target in files else ""
        if source and _symbol_defined(source, requirement.artifact):
            changes.append(CodeChangeIR(target, requirement.artifact, operation, None, requirement.expected_behavior,
                                        requirement.members, False, "already_satisfied",
                                        f"`{requirement.artifact}` は既に定義済みです。", (f"{target}:{requirement.artifact}",),
                                        0.9, requirement))
            continue
        changes.append(CodeChangeIR(target, requirement.artifact, operation, None, requirement.expected_behavior,
                                    requirement.members, creates_file, "compiled", "",
                                    (f"{target}:{requirement.artifact}",), 0.85, requirement))
        created_symbols.append((target, requirement.artifact))

    return ChangeCompilation(tuple(changes), tuple(rationale))


def to_patch_intents(compilation: ChangeCompilation, constraints: tuple[str, ...] = ()) -> tuple[PatchIntent, ...]:
    return tuple(
        PatchIntent(item.file, item.symbol, item.operation.value, item.behavior, constraints, item.evidence)
        for item in compilation.compiled
    )


# --------------------------------------------------------------------------
# stage 3: CodeChangeIR -> source text
# --------------------------------------------------------------------------

def _snake(name: str) -> str:
    spaced = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name.replace(".", "_"))
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", spaced).lower()


def _docstring(behavior: str) -> str:
    return behavior.replace('"""', "'''").replace("\n", " ").strip()[:200]


def _ensure_import(source: str, statement: str) -> str:
    if re.search(rf"^{re.escape(statement)}\s*$", source, re.M):
        return source
    lines = source.splitlines()
    insert_at = 0
    try:
        tree = ast.parse(source)
    except SyntaxError:
        tree = None
    if tree is not None:
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                insert_at = node.end_lineno or node.lineno
            elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str) and insert_at == 0:
                insert_at = node.end_lineno or node.lineno
    lines.insert(insert_at, statement)
    return "\n".join(lines) + ("\n" if source.endswith("\n") or not source else "")


def _append_block(source: str, block: str) -> str:
    base = source.rstrip("\n")
    prefix = f"{base}\n\n\n" if base else ""
    return f"{prefix}{block.rstrip()}\n"


def _insert_lines(source: str, index: int, block: str) -> str:
    lines = source.splitlines()
    lines[index:index] = block.rstrip("\n").split("\n")
    return "\n".join(lines) + "\n"


def _class_body_end(source: str, name: str) -> tuple[int, str] | None:
    node = _class_node(source, name)
    if node is None or not node.body:
        return None
    end = max((item.end_lineno or item.lineno) for item in node.body)
    indent = " " * node.body[0].col_offset
    return end, indent


def _render_dataclass(change: CodeChangeIR) -> str:
    fields = change.members or ()
    body = [f"    {member}: str = \"\"" for member in fields] or ["    pass"]
    return "\n".join([
        "@dataclass(frozen=True)",
        f"class {change.symbol}:",
        f'    """{_docstring(change.behavior)}"""',
        "",
        *body,
    ])


def _render_class(change: CodeChangeIR) -> str:
    return "\n".join([
        f"class {change.symbol}:",
        f'    """{_docstring(change.behavior)}"""',
    ])


def _render_function(name: str, params: tuple[str, ...], behavior: str) -> str:
    signature = ", ".join(params)
    return "\n".join([
        f"def {name}({signature}):",
        f'    """{_docstring(behavior)}',
        "",
        "    L8.5 generated seam: returns None until implemented, so the change is",
        "    structurally in place without altering existing behaviour.",
        '    """',
        "    return None",
    ])


def _render_route_block(indent: str, hook: str, argument: str, behavior: str) -> str:
    call = f"{hook}({argument})" if argument else f"{hook}()"
    return "\n".join([
        f"{indent}# L8.5 semantic route: {_docstring(behavior)}",
        f"{indent}_l85_routed = {call}",
        f"{indent}if _l85_routed is not None:",
        f"{indent}    return _l85_routed",
    ])


def _render_test(change: CodeChangeIR) -> str:
    subject_file = change.members[0] if change.members else ""
    relative = os.path.relpath(subject_file or ".", str(Path(change.file).parent)).replace("\\", "/")
    marker = f"class {change.anchor}" if change.anchor and change.anchor[:1].isupper() else f"def {change.anchor}"
    return "\n".join([
        "from pathlib import Path",
        "",
        "",
        f'TARGET = Path(__file__).resolve().parent / "{relative}"',
        "",
        "",
        f"def {change.symbol}():",
        f'    """Structural regression guard generated by L8.5."""',
        f'    assert "{marker}" in TARGET.read_text(encoding="utf-8")',
    ])


def _anchor_line_in(source: str, anchor: str) -> int:
    """Re-locate the anchor inside a pending buffer, preferring the use site."""
    if not anchor:
        return 0
    pattern = re.compile(rf"{_L}{re.escape(anchor)}{_R}")
    definition = re.compile(rf"^\s*(?:class|def|async def)\s+{re.escape(anchor)}{_R}")
    fallback = 0
    for index, line in enumerate(source.splitlines(), 1):
        stripped = line.strip()
        if not pattern.search(line) or stripped.startswith(("#", "import ", "from ")):
            continue
        if _enclosing_function(source, index) is not None and not definition.match(line):
            return index
        fallback = fallback or index
    return fallback


def _new_condition(text: str) -> str:
    match = re.search(r"(?:→|->|=>|条件\s*[:：]|condition\s*[:：])\s*(.+)$", text)
    if not match:
        return ""
    candidate = match.group(1).strip().strip("`。 ")
    try:
        ast.parse(candidate, mode="eval")
    except SyntaxError:
        return ""
    return candidate


def render_change(root: Path, change: CodeChangeIR, source: str) -> tuple[str, str]:
    """Return (updated_source, reason).  An empty update means 'not rendered'."""
    operation = change.operation
    if operation is ChangeOperation.ADD_DATACLASS:
        updated = _ensure_import(source, "from dataclasses import dataclass") if source.strip() else "from dataclasses import dataclass\n"
        return _append_block(updated, _render_dataclass(change)), ""
    if operation is ChangeOperation.ADD_CLASS:
        return _append_block(source, _render_class(change)), ""
    if operation is ChangeOperation.ADD_FUNCTION:
        return _append_block(source, _render_function(change.symbol or "generated", (), change.behavior)), ""
    if operation is ChangeOperation.ADD_IMPORT:
        statement = change.requirement.text.strip() if change.requirement.text.strip().startswith(("import ", "from ")) else f"import {change.symbol}"
        return _ensure_import(source, statement), ""
    if operation is ChangeOperation.ADD_TEST:
        return _append_block(source, _render_test(change)) if source.strip() else _render_test(change) + "\n", ""
    if operation is ChangeOperation.MODIFY_ENUM:
        location = _class_body_end(source, change.symbol or "")
        if location is None:
            return "", "Enumクラス本体を特定できませんでした。"
        end, indent = location
        block = "\n".join(f'{indent}{member} = "{member}"' for member in change.members if not re.search(rf"^\s*{re.escape(member)}\s*=", source, re.M))
        return (_insert_lines(source, end, block), "") if block else ("", "追加すべき未定義メンバーがありません。")
    if operation is ChangeOperation.ADD_METHOD:
        owner = change.anchor or ""
        location = _class_body_end(source, owner)
        if location is None:
            return "", "メソッド追加先のクラス本体を特定できませんでした。"
        end, indent = location
        name = (change.symbol or "").split(".")[-1]
        block = "\n".join([
            "",
            f"{indent}def {name}(self):",
            f'{indent}    """{_docstring(change.behavior)}"""',
            f"{indent}    return None",
        ])
        return _insert_lines(source, end, block), ""
    if operation is ChangeOperation.INSERT_ROUTE:
        anchor_line = _anchor_line_in(source, change.anchor or "")
        if not anchor_line:
            return "", "anchor行を再特定できませんでした。"
        function = _enclosing_function(source, anchor_line)
        if function is None:
            return "", "anchorの外側関数を特定できませんでした。"
        raw = source.splitlines()[anchor_line - 1]
        indent = raw[: len(raw) - len(raw.lstrip())]
        argument = next((arg.arg for arg in function.args.args if arg.arg not in {"self", "cls"}), "")
        artifact = change.requirement.artifact
        # An artifact that already exists is wired up directly; only a genuinely
        # new capability gets a behaviour-preserving generated seam.
        existing = _function_arity(source, artifact)
        hook = artifact if existing is not None else f"{_snake(artifact)}_route"
        if existing == 0:
            argument = ""
        updated = _insert_lines(source, anchor_line - 1, _render_route_block(indent, hook, argument, change.behavior))
        if existing is None and not _symbol_defined(updated, hook):
            updated = _append_block(updated, _render_function(hook, (argument,) if argument else (), change.behavior))
        return updated, ""
    if operation is ChangeOperation.REPLACE_BRANCH:
        condition = _new_condition(change.requirement.text)
        if not condition:
            return "", "置き換え後の条件式が提案文に含まれていません。"
        pattern = re.compile(rf"^(\s*)(if|elif)\s+.*{_L}{re.escape(change.anchor or '')}{_R}.*:\s*$", re.M)
        match = pattern.search(source)
        if not match:
            return "", "置き換え対象の分岐行を特定できませんでした。"
        replacement = f"{match.group(1)}{match.group(2)} {condition}:"
        return source[: match.start()] + replacement + source[match.end():], ""
    return "", "未対応のoperationです。"


def _unified_diff(path: str, original: str, updated: str) -> str:
    return "".join(difflib.unified_diff(original.splitlines(True), updated.splitlines(True), fromfile=path, tofile=path))


class SemanticPatchSynthesizer:
    """L8.5 synthesizer: semantic requirement -> CodeChangeIR -> PatchCandidate."""

    def __init__(self, graph: RepositoryGraph | None = None):
        self.graph = graph
        self.last_compilation: ChangeCompilation | None = None

    def compile(self, root: Path, instruction: RepairInstruction, plan: RepairPlan) -> ChangeCompilation:
        compilation = compile_semantic_changes(Path(root), instruction, plan, self.graph)
        self.last_compilation = compilation
        return compilation

    def synthesize(self, root: Path, instruction: RepairInstruction, plan: RepairPlan) -> tuple[PatchCandidate, ...]:
        root = Path(root)
        compilation = self.compile(root, instruction, plan)
        pending: dict[str, str] = {}
        candidates: list[PatchCandidate] = []
        for change in compilation.compiled:
            path = root / change.file
            if change.file in pending:
                source = pending[change.file]
            elif path.is_file():
                source = path.read_text(encoding="utf-8", errors="replace")
            else:
                source = ""
            updated, _reason = render_change(root, change, source)
            if not updated or updated == source:
                continue
            intent = PatchIntent(change.file, change.symbol, change.operation.value, change.behavior,
                                 instruction.constraints, change.evidence)
            candidates.append(PatchCandidate(intent, source, updated, _unified_diff(change.file, source, updated)))
            pending[change.file] = updated
        return tuple(candidates)


def format_change_compilation(compilation: ChangeCompilation) -> str:
    lines = ["## Semantic-to-Code Change Compiler (L8.5)", "",
             f"【compiled】`{len(compilation.compiled)}` / 【unresolved】`{len(compilation.unresolved)}`", "",
             "【Code Change IR】"]
    if not compilation.changes:
        lines.append("- 変換対象の変更案がありません。")
    for change in compilation.changes:
        lines.append(
            f"- `{change.operation.value}` file=`{change.file or '-'}` symbol=`{change.symbol or '-'}` "
            f"anchor=`{change.anchor or '-'}` status=`{change.status}` confidence=`{change.confidence:.2f}`"
        )
        if change.members:
            lines.append(f"    - members: {', '.join(change.members)}")
        if change.creates_file:
            lines.append("    - 新規ファイルを作成します。")
        if change.reason:
            lines.append(f"    - 理由: {change.reason}")
        lines.append(f"    - behavior: {_docstring(change.behavior)}")
    lines.extend(["", "【判断根拠】", *[f"- {item}" for item in compilation.rationale]])
    return "\n".join(lines)
