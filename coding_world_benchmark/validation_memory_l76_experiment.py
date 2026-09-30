"""L7.6: compress validation files into reusable knowledge patterns."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import gzip
import json
from pathlib import Path
import re


@dataclass(frozen=True)
class ProvenanceLink:
    file: str
    line: int
    role: str
    excerpt: str


@dataclass
class ValidationKnowledge:
    concept: str
    problem_type: str
    hypothesis: str | None
    setup: dict[str, str]
    parameters: dict[str, str]
    procedure: list[str]
    success_condition: list[str]
    failure_modes: list[str]
    fixes: list[str]
    reusable_operators: list[str]
    evidence_files: list[str]
    provenance: list[ProvenanceLink]
    confidence: float
    uses: int = 0


def _validation_file(path: Path) -> bool:
    name = path.name.lower()
    return path.suffix.lower() in {".py", ".md", ".json", ".txt"} and any(
        word in name or word in "/".join(path.parts).lower()
        for word in ("validation", "verify", "verification", "benchmark", "test", "audit")
    )


def _concept(path: Path, text: str) -> tuple[str, str]:
    lowered = (str(path) + " " + text).lower()
    if any(word in lowered for word in ("su2", "su3", "gauge", "wilson")):
        return "gauge_theory_validation", "invariance_or_criticality_validation"
    if any(word in lowered for word in ("finite_size", "finite-size", "scaling")):
        return "finite_size_scaling", "robustness_validation"
    if any(word in lowered for word in ("conservation", "invariant", "energy")):
        return "conservation_audit", "invariant_validation"
    if any(word in lowered for word in ("counterexample", "falsif")):
        return "counterexample_search", "falsification_validation"
    stem = re.sub(r"_(?:validation|verification|benchmark|test|audit).*", "", path.stem.lower())
    return stem or "validation_pattern", "general_validation"


def _unique(values: list[str], limit: int = 20) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))[:limit]


def _extract_knowledge(path: Path, root: Path) -> ValidationKnowledge:
    text = path.read_text(encoding="utf-8", errors="ignore")
    concept, problem_type = _concept(path, text)
    relative = path.relative_to(root).as_posix()
    procedures: list[str] = []
    success: list[str] = []
    failures: list[str] = []
    fixes: list[str] = []
    operators: list[str] = []
    provenance: list[ProvenanceLink] = []
    parameters: dict[str, str] = {}
    for line_no, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        lower = line.lower()
        if not line:
            continue
        if re.search(r"\bdef\s+([A-Za-z_]\w*)", line):
            operators.append(re.search(r"\bdef\s+([A-Za-z_]\w*)", line).group(1))
            provenance.append(ProvenanceLink(relative, line_no, "operator", line[:240]))
        if any(word in lower for word in ("initialize", "setup", "transform", "evolve", "simulate", "measure", "sweep", "fit", "compare")):
            procedures.append(line[:240])
            provenance.append(ProvenanceLink(relative, line_no, "procedure", line[:240]))
        if any(word in lower for word in ("assert", "passed", "success", "verified", "condition")):
            success.append(line[:240])
            provenance.append(ProvenanceLink(relative, line_no, "success_condition", line[:240]))
        if any(word in lower for word in ("fail", "failure", "error", "unstable", "insufficient", "nan", "flaky")):
            failures.append(line[:240])
            provenance.append(ProvenanceLink(relative, line_no, "failure_mode", line[:240]))
        if any(word in lower for word in ("fix", "repair", "increase", "initialize", "seed", "retry")):
            fixes.append(line[:240])
            provenance.append(ProvenanceLink(relative, line_no, "fix", line[:240]))
        match = re.search(r"(?:PARAM|parameter|dt|beta|size|steps|nodes|samples)\s*[=:]\s*([^,#]+)", line, re.I)
        if match:
            key = re.split(r"\s*[=:]", line, maxsplit=1)[0].strip(" #")
            parameters[key] = match.group(1).strip()
    return ValidationKnowledge(
        concept, problem_type, next((line for line in success if "hypothesis" in line.lower()), None),
        {"source_kind": path.suffix.lower()}, parameters, _unique(procedures), _unique(success),
        _unique(failures), _unique(fixes), _unique(operators), [relative], provenance[:80],
        min(0.99, 0.45 + 0.05 * len(provenance)),
    )


class ValidationCompressor:
    def compress(self, root: Path, store_path: Path | None = None) -> tuple[ValidationKnowledge, ...]:
        root = root.resolve()
        grouped: dict[str, ValidationKnowledge] = {}
        for path in root.rglob("*"):
            if not path.is_file() or ".git" in path.parts or ".agent_trash" in path.parts or ".conscious_coding_agent" in path.parts or not _validation_file(path):
                continue
            try:
                item = _extract_knowledge(path, root)
            except OSError:
                continue
            existing = grouped.get(item.concept)
            if existing is None:
                grouped[item.concept] = item
            else:
                existing.procedure = _unique(existing.procedure + item.procedure)
                existing.success_condition = _unique(existing.success_condition + item.success_condition)
                existing.failure_modes = _unique(existing.failure_modes + item.failure_modes)
                existing.fixes = _unique(existing.fixes + item.fixes)
                existing.reusable_operators = _unique(existing.reusable_operators + item.reusable_operators)
                existing.evidence_files = _unique(existing.evidence_files + item.evidence_files)
                existing.provenance = (existing.provenance + item.provenance)[:100]
                existing.confidence = max(existing.confidence, item.confidence)
        result = tuple(grouped.values())
        if store_path:
            store_path.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(store_path, "wt", encoding="utf-8") as stream:
                for item in result:
                    json.dump(asdict(item), stream, ensure_ascii=False)
                    stream.write("\n")
        return result


class ValidationMemoryRetriever:
    def __init__(self, knowledge: tuple[ValidationKnowledge, ...] = ()):
        self.knowledge = knowledge

    def retrieve(self, query: str, limit: int = 5) -> tuple[ValidationKnowledge, ...]:
        tokens = set(re.findall(r"[a-z][a-z0-9_-]{2,}|[一-龥ぁ-んァ-ヶ]{2,}", query.lower()))
        scored = []
        for item in self.knowledge:
            haystack = " ".join([item.concept, item.problem_type, *item.evidence_files, *item.procedure, *item.failure_modes, *item.reusable_operators]).lower()
            score = sum(token in haystack for token in tokens)
            if score:
                scored.append((score + item.uses * 0.1, item))
        scored.sort(key=lambda row: (-row[0], row[1].concept))
        for _, item in scored[:limit]:
            item.uses += 1
        return tuple(item for _, item in scored[:limit])


def format_validation_memory(knowledge: tuple[ValidationKnowledge, ...]) -> str:
    lines = ["## Reusable Validation Knowledge", ""]
    for item in knowledge:
        lines.extend([f"【{item.concept}】", f"- problem type: `{item.problem_type}`", f"- confidence: `{item.confidence:.2f}`"])
        if item.procedure:
            lines.append("- procedure: " + " → ".join(item.procedure[:5]))
        if item.failure_modes:
            lines.append("- failure modes: " + " / ".join(item.failure_modes[:4]))
        if item.fixes:
            lines.append("- fixes: " + " / ".join(item.fixes[:4]))
        lines.append("- evidence: " + ", ".join(item.evidence_files[:4]))
    return "\n".join(lines)
