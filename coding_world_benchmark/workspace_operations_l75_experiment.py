"""L7.5: safe workspace action routing, resolution, execution, and verification."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import re
import shutil
from pathlib import Path
from uuid import uuid4


class WorkspaceAction(Enum):
    LIST = "list"
    READ = "read"
    DELETE_DIRECTORY = "delete_directory"
    MOVE_DIRECTORIES = "move_directories"


@dataclass(frozen=True)
class WorkspaceOperation:
    action: WorkspaceAction
    target_descriptions: tuple[str, ...]
    destination_name: str | None = None
    target_mode: str = "one"


@dataclass(frozen=True)
class ResolvedPath:
    path: Path
    confidence: float
    evidence: tuple[str, ...]
    kind: str


@dataclass(frozen=True)
class ResolvedTargetGroup:
    concept: str
    paths: tuple[Path, ...]
    confidence: float
    resolution_mode: str


@dataclass(frozen=True)
class DeleteImpact:
    target: str
    imported_by: tuple[str, ...]
    referenced_by: tuple[str, ...]
    tests_affected: tuple[str, ...]


@dataclass(frozen=True)
class OperationResult:
    success: bool
    status: str
    target: str
    trash_path: str | None = None
    impact: DeleteImpact | None = None
    destination_path: str | None = None


# Japanese particles are word characters, so ``\bSU2\b`` never matches in
# "SU2とSU3の検証フォルダ": the boundary between "2" and "と" does not exist.
# Bracket the ASCII class explicitly instead.
_SU_LABEL = re.compile(r"(?<![A-Za-z0-9_])SU[0-9]+(?![A-Za-z0-9_])", re.I)


def is_delete_directory_request(text: str) -> bool:
    has_delete = any(word in text for word in ("削除して", "消して", "消去して", "delete", "remove"))
    has_directory = any(word in text for word in ("フォルダ", "ディレクトリ", "directory", "folder"))
    return has_delete and (has_directory or bool(_SU_LABEL.search(text)))


def is_move_directories_request(text: str) -> bool:
    has_move = any(word in text for word in ("移動させて", "移動して", "まとめて", "move", "relocate"))
    has_directory = any(word in text for word in ("フォルダ", "ディレクトリ", "directory", "folder"))
    return has_move and has_directory and bool(_SU_LABEL.search(text))


def parse_delete_operation(text: str) -> WorkspaceOperation:
    descriptions = tuple(dict.fromkeys(re.findall(r"SU[0-9]+[^、,。]*?(?:フォルダ|ディレクトリ)?", text, re.I)))
    if not descriptions:
        descriptions = (text,)
    return WorkspaceOperation(WorkspaceAction.DELETE_DIRECTORY, descriptions)


def parse_move_operation(text: str) -> WorkspaceOperation:
    labels = tuple(dict.fromkeys(label.upper() for label in re.findall(r"SU[0-9]+", text, re.I)))
    descriptions = tuple(f"{label} validation" for label in labels)
    if not descriptions:
        descriptions = (text,)
    named_destination = re.search(r"新規(?:フォルダ|ディレクトリ)\s*([A-Za-z][A-Za-z0-9_-]*)", text, re.I)
    destination = named_destination.group(1) if named_destination else (
        "su2_su3" if {label.lower() for label in labels} == {"su2", "su3"} else "combined_validation"
    )
    all_matching = any(word in text for word in ("まとめて", "すべて", "全て", "all", "each"))
    return WorkspaceOperation(WorkspaceAction.MOVE_DIRECTORIES, descriptions, destination, "all_matching" if all_matching else "one")


class WorkspacePathResolver:
    def __init__(self, root: Path):
        self.root = root.resolve()

    def resolve_safe(self, value: str | Path) -> Path:
        candidate = Path(value)
        target = (candidate if candidate.is_absolute() else self.root / candidate).resolve()
        if target != self.root and self.root not in target.parents:
            raise PermissionError("path escapes workspace")
        return target

    def find_directories(self, description: str) -> tuple[ResolvedPath, ...]:
        tokens = [token.lower() for token in re.findall(r"[a-z][a-z0-9_-]*|su\d+", description.lower())]
        candidates: list[ResolvedPath] = []
        for path in self.root.rglob("*"):
            if not path.is_dir() or ".git" in path.parts or ".agent_trash" in path.parts:
                continue
            name = path.name.lower()
            identity_tokens = [token for token in tokens if token not in {"validation", "verification", "folder", "directory"}]
            score = sum(token in name or token in str(path.relative_to(self.root)).lower() for token in identity_tokens)
            validation = any(word in name for word in ("validation", "verification", "検証"))
            if score and (validation or any(token in name for token in identity_tokens)):
                confidence = 0.97 if score == len(identity_tokens) else 0.72
                candidates.append(ResolvedPath(path, confidence, (f"directory name matched: {path.name}",), "directory"))
        return tuple(sorted(candidates, key=lambda item: (-item.confidence, str(item.path))))

    def resolve_target_group(self, concept: str, mode: str = "one") -> ResolvedTargetGroup:
        """Resolve one concept to one or all matching directories."""
        candidates = self.find_directories(concept)
        if mode == "all_matching":
            paths = tuple(dict.fromkeys(item.path for item in candidates if item.confidence >= 0.9))
            return ResolvedTargetGroup(concept, paths, 0.97 if paths else 0.0, mode)
        if len(candidates) == 1 and candidates[0].confidence >= 0.9:
            return ResolvedTargetGroup(concept, (candidates[0].path,), candidates[0].confidence, mode)
        return ResolvedTargetGroup(concept, (), 0.0, mode)


class WorkspaceExecutor:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.resolver = WorkspacePathResolver(self.root)

    def impact(self, target: Path) -> DeleteImpact:
        target_text = str(target.relative_to(self.root))
        imported_by: list[str] = []
        referenced_by: list[str] = []
        tests: list[str] = []
        for path in self.root.rglob("*"):
            if not path.is_file() or ".git" in path.parts or ".agent_trash" in path.parts or path == target:
                continue
            try:
                content = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if target.name in content or target_text in content:
                referenced_by.append(str(path.relative_to(self.root)))
                if "import " in content or "from " in content:
                    imported_by.append(str(path.relative_to(self.root)))
                if path.name.startswith("test_") or "test" in path.name.lower():
                    tests.append(str(path.relative_to(self.root)))
        return DeleteImpact(target_text, tuple(imported_by), tuple(referenced_by), tuple(tests))

    def trash_directory(self, target: Path) -> OperationResult:
        safe_target = self.resolver.resolve_safe(target)
        if safe_target == self.root:
            return OperationResult(False, "REFUSED_ROOT", str(safe_target))
        if not safe_target.exists():
            return OperationResult(False, "NOT_FOUND", str(safe_target))
        if not safe_target.is_dir():
            return OperationResult(False, "NOT_A_DIRECTORY", str(safe_target))
        impact = self.impact(safe_target)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        trash = self.root / ".agent_trash" / f"{stamp}_{uuid4().hex[:8]}" / safe_target.name
        trash.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(safe_target), str(trash))
        verified = not safe_target.exists() and trash.is_dir()
        return OperationResult(verified, "TRASHED" if verified else "VERIFY_FAILED", str(safe_target), str(trash), impact)


def execute_delete_operation(root: Path, operation: WorkspaceOperation) -> tuple[OperationResult, ...]:
    executor = WorkspaceExecutor(root)
    results: list[OperationResult] = []
    for description in operation.target_descriptions:
        candidates = executor.resolver.find_directories(description)
        if len(candidates) != 1 or candidates[0].confidence < 0.9:
            results.append(OperationResult(False, "AMBIGUOUS_TARGET", description))
            continue
        results.append(executor.trash_directory(candidates[0].path))
    return tuple(results)


def execute_move_operation(root: Path, operation: WorkspaceOperation) -> tuple[OperationResult, ...]:
    executor = WorkspaceExecutor(root)
    groups = [executor.resolver.resolve_target_group(description, operation.target_mode) for description in operation.target_descriptions]
    resolved: list[Path] = [path for group in groups for path in group.paths]
    failures: list[OperationResult] = []
    for group in groups:
        if not group.paths:
            failures.append(OperationResult(False, "AMBIGUOUS_TARGET", group.concept))
    if failures:
        return tuple(failures)
    if not resolved or len(set(resolved)) != len(resolved):
        return (OperationResult(False, "AMBIGUOUS_TARGET", "duplicate move target"),)
    destination_name = operation.destination_name or "combined_validation"
    destination = executor.resolver.resolve_safe(destination_name)
    if destination.exists():
        return tuple(OperationResult(False, "DESTINATION_EXISTS", str(path)) for path in resolved)
    if any(path == destination or destination in path.parents for path in resolved):
        return tuple(OperationResult(False, "INVALID_DESTINATION", str(path)) for path in resolved)
    destination.mkdir(parents=False)
    results: list[OperationResult] = []
    try:
        for source in resolved:
            target = destination / source.name
            shutil.move(str(source), str(target))
            verified = not source.exists() and target.is_dir()
            results.append(OperationResult(verified, "MOVED" if verified else "VERIFY_FAILED", str(source), str(destination), None, str(target)))
    except OSError:
        for source in resolved:
            if source.exists():
                continue
        results.extend(OperationResult(False, "MOVE_FAILED", str(source), str(destination)) for source in resolved[len(results):])
    return tuple(results)


def format_operation_report(operation: WorkspaceOperation, results: tuple[OperationResult, ...]) -> str:
    lines = ["## Workspace Operation Report", "", f"【操作】`{operation.action.name}`", "", "【対象】"]
    for result in results:
        lines.append(f"- `{result.target}`: `{result.status}`")
        if result.destination_path:
            lines.append(f"  - destination: `{result.destination_path}`")
        if result.trash_path:
            label = "destination" if operation.action is WorkspaceAction.MOVE_DIRECTORIES else "trash"
            lines.append(f"  - {label}: `{result.trash_path}`")
        if result.impact and result.impact.referenced_by:
            lines.append(f"  - references: {len(result.impact.referenced_by)}")
    success_count = sum(result.success for result in results)
    expected_count = len(results)
    if operation.action is WorkspaceAction.DELETE_DIRECTORY:
        note = "- 完全削除ではなく`.agent_trash`へ移動しました。"
    elif success_count == expected_count and expected_count > 0:
        note = "- 移動元が消え、移動先のディレクトリが存在することを確認しました。"
    elif success_count == 0:
        note = "- 実際の移動は行われていません。"
    else:
        note = "- 一部のみ移動されました。未完了の対象を確認してください。"
    lines.extend(["", "【検証】", f"- 成功: `{success_count}/{expected_count}`", note])
    if any(result.status == "AMBIGUOUS_TARGET" for result in results):
        lines.append("- 曖昧な候補は安全のため操作していません。対象pathを指定してください。")
    return "\n".join(lines)
