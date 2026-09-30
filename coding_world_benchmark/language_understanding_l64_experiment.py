"""L6.4: machine-graded language understanding benchmark.

This layer separates language understanding from Japanese response generation.
It evaluates whether a natural-language work request becomes an executable
semantic task representation with the right intent, target, context flags,
ambiguity handling, and multi-step operation sequence.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from enum import Enum
import json
import re
from typing import Callable, Mapping, Sequence

from .canonical_ir_l63_experiment import CanonicalConcept, canonicalize_request


class SemanticIntent(Enum):
    SUMMARIZE = "SUMMARIZE"
    EXPLAIN = "EXPLAIN"
    INSPECT = "INSPECT"
    VERIFY = "VERIFY"
    IMPLEMENT = "IMPLEMENT"
    MODIFY = "MODIFY"
    FIND_DEFINITION = "FIND_DEFINITION"
    RUN = "RUN"


class TargetType(Enum):
    FILE = "FILE"
    EXPERIMENT = "EXPERIMENT"
    SYMBOL = "SYMBOL"
    REFERENCE = "REFERENCE"
    REPOSITORY = "REPOSITORY"
    MATH_RELATION = "MATH_RELATION"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class LanguageContext:
    previous_experiment: str | None = None
    repository_targets: tuple[str, ...] = ()


@dataclass(frozen=True)
class SemanticTask:
    intent: SemanticIntent
    target_type: TargetType
    target: str
    requires_execution: bool
    requires_modification: bool
    requires_repository_context: bool
    requires_reference_resolution: bool
    ambiguous: bool
    operation_sequence: tuple[str, ...]


@dataclass(frozen=True)
class LanguageCase:
    case_id: str
    level: int
    text: str
    expected: SemanticTask
    context: LanguageContext = LanguageContext()
    pair_id: str | None = None


@dataclass(frozen=True)
class CaseResult:
    case_id: str
    expected: SemanticTask
    predicted: SemanticTask
    field_matches: Mapping[str, bool]


def _language(text: str) -> str:
    has_japanese = bool(re.search(r"[一-龥ぁ-んァ-ヶ]", text))
    has_latin = bool(re.search(r"[A-Za-z]", text))
    if has_japanese and has_latin:
        return "mixed"
    return "ja" if has_japanese else "en"


def _operations(text: str, base_intent: SemanticIntent) -> tuple[str, ...]:
    lower = text.lower()
    operations: list[str] = []
    if base_intent is SemanticIntent.IMPLEMENT:
        operations.extend(["hypothesis_extraction", "experiment_design", "implementation"])
    if base_intent is SemanticIntent.SUMMARIZE or "要約" in text:
        operations.append("report")
    if base_intent is SemanticIntent.IMPLEMENT and "結果" in text:
        operations.append("execution")
    if any(term in text for term in ("実行", "走らせ", "もう一度実行")) or any(term in lower for term in ("run", "execute")):
        operations.append("execution")
    if any(term in text for term in ("確認", "評価", "検証")) or any(term in lower for term in ("evaluate", "verify", "check")):
        operations.append("evaluation")
    if any(term in text for term in ("まとめ", "報告")) or any(term in lower for term in ("summarize", "report")):
        operations.append("report")
    if not operations:
        operations.append(base_intent.value.lower())
    return tuple(dict.fromkeys(operations))


def _target_from_text(text: str, context: LanguageContext) -> tuple[TargetType, str, bool]:
    lower = text.lower()
    file_match = re.search(r"([A-Za-z0-9_.-]+\.py|README(?:\.md)?)", text)
    if file_match:
        return TargetType.FILE, file_match.group(1), False
    experiment_match = re.search(r"(exp[A-Za-z0-9_]*)", text, re.IGNORECASE)
    if experiment_match:
        return TargetType.EXPERIMENT, experiment_match.group(1), False
    if any(term in text for term in ("さっき", "前回", "直前")) or "previous" in lower:
        return TargetType.REFERENCE, context.previous_experiment or "previous_experiment", True
    if "rho" in lower and ("a^-3" in lower or "a**-3" in lower or "inverse cube" in lower):
        return TargetType.MATH_RELATION, "rho(a) = rho0 * a^(-3)", False
    if "initial" in lower or any(term in text for term in ("初期状態", "初期条件")):
        return TargetType.SYMBOL, "initial state", False
    return TargetType.REPOSITORY, "current repository", False


def parse_semantic_task(text: str, context: LanguageContext = LanguageContext()) -> SemanticTask:
    canonical = canonicalize_request(text)
    lower = text.lower()
    if any(term in text for term in ("どこで設定", "定義", "場所")) or any(term in lower for term in ("find where", "defined", "definition")):
        intent = SemanticIntent.FIND_DEFINITION
    elif any(term in text for term in ("実行", "動か", "走らせ", "試して", "もう一度実行")) or any(term in lower for term in ("run", "execute")):
        intent = SemanticIntent.RUN
    else:
        intent = SemanticIntent(canonical.action.upper())
    target_type, target, reference = _target_from_text(text, context)
    if reference and context.previous_experiment:
        target_type = TargetType.EXPERIMENT
        target = context.previous_experiment
    has_initial_state = any(term.concept is CanonicalConcept.INITIAL_STATE for term in canonical.concepts)
    ambiguous = (
        has_initial_state
        and target_type is TargetType.SYMBOL
        and len(context.repository_targets) > 1
        and not any(term in lower for term in ("exp", ".py"))
    )
    requires_repository_context = (
        target_type in {TargetType.SYMBOL, TargetType.REPOSITORY, TargetType.MATH_RELATION}
        or "repository" in lower
        or any(term in text for term in ("このモデル", "リポジトリ"))
    )
    operations = _operations(text, intent)
    return SemanticTask(
        intent=intent,
        target_type=target_type,
        target=target,
        requires_execution="execution" in operations or intent in {SemanticIntent.RUN, SemanticIntent.VERIFY},
        requires_modification=intent in {SemanticIntent.IMPLEMENT, SemanticIntent.MODIFY},
        requires_repository_context=requires_repository_context,
        requires_reference_resolution=reference,
        ambiguous=ambiguous,
        operation_sequence=operations,
    )


def keyword_parse_semantic_task(text: str, _context: LanguageContext = LanguageContext()) -> SemanticTask:
    lower = text.lower()
    intent = SemanticIntent.SUMMARIZE
    if "実装" in text or "implement" in lower:
        intent = SemanticIntent.IMPLEMENT
    elif any(term in text for term in ("実行", "動か", "走らせ", "試して")) or "run" in lower:
        intent = SemanticIntent.RUN
    elif "説明" in text or "explain" in lower:
        intent = SemanticIntent.EXPLAIN
    target_type, target, _reference = _target_from_text(text, LanguageContext())
    return SemanticTask(
        intent=intent,
        target_type=target_type,
        target=target,
        requires_execution=intent is SemanticIntent.RUN,
        requires_modification=intent is SemanticIntent.IMPLEMENT,
        requires_repository_context=False,
        requires_reference_resolution=False,
        ambiguous=False,
        operation_sequence=(intent.value.lower(),),
    )


def default_language_cases() -> tuple[LanguageCase, ...]:
    ambiguous_context = LanguageContext(repository_targets=("exp445.initial_state", "exp464.initial_state", "expA1.initial_state"))
    previous_context = LanguageContext(previous_experiment="exp465")
    return (
        LanguageCase(
            "l1-file-summary-ja", 1, "READMEを要約して",
            SemanticTask(SemanticIntent.SUMMARIZE, TargetType.FILE, "README", False, False, False, False, False, ("report",)),
            pair_id="readme-summary",
        ),
        LanguageCase(
            "l1-file-summary-en", 1, "Summarize README.",
            SemanticTask(SemanticIntent.SUMMARIZE, TargetType.FILE, "README", False, False, False, False, False, ("report",)),
            pair_id="readme-summary",
        ),
        LanguageCase(
            "l1-exp-run-ja", 1, "exp465を実行して結果を確認して",
            SemanticTask(SemanticIntent.RUN, TargetType.EXPERIMENT, "exp465", True, False, False, False, False, ("execution", "evaluation")),
        ),
        LanguageCase(
            "l2-implement-evaluate-ja", 2, "この仮説を検証する実験を実装して結果をまとめて",
            SemanticTask(
                SemanticIntent.IMPLEMENT, TargetType.REPOSITORY, "current repository", True, True, True, False, False,
                ("hypothesis_extraction", "experiment_design", "implementation", "execution", "evaluation", "report"),
            ),
        ),
        LanguageCase(
            "l3-reference-ja", 3, "さっき追加した実験をもう一度実行して",
            SemanticTask(SemanticIntent.RUN, TargetType.EXPERIMENT, "exp465", True, False, False, True, False, ("execution",)),
            previous_context,
        ),
        LanguageCase(
            "l4-repository-initial-state-ja", 4, "このモデルの初期状態はどうなってる？",
            SemanticTask(SemanticIntent.SUMMARIZE, TargetType.SYMBOL, "initial state", False, False, True, False, False, ("report",)),
        ),
        LanguageCase(
            "l5-ambiguous-initial-state-ja", 5, "初期状態を説明して",
            SemanticTask(SemanticIntent.EXPLAIN, TargetType.SYMBOL, "initial state", False, False, True, False, True, ("explain",)),
            ambiguous_context,
        ),
        LanguageCase(
            "l6-math-relation-ja", 6, "この式からrhoがa^-3になる理由を説明して",
            SemanticTask(SemanticIntent.EXPLAIN, TargetType.MATH_RELATION, "rho(a) = rho0 * a^(-3)", False, False, True, False, False, ("explain",)),
        ),
    )


def evaluate_cases(
    cases: Sequence[LanguageCase],
    parser: Callable[[str, LanguageContext], SemanticTask] = parse_semantic_task,
) -> tuple[CaseResult, ...]:
    results = []
    for case in cases:
        predicted = parser(case.text, case.context)
        matches = {field.name: getattr(predicted, field.name) == getattr(case.expected, field.name) for field in fields(SemanticTask)}
        results.append(CaseResult(case.case_id, case.expected, predicted, matches))
    return tuple(results)


def summarize_language_results(cases: Sequence[LanguageCase], results: Sequence[CaseResult]) -> Mapping[str, float]:
    if not results:
        raise ValueError("results must not be empty")

    def field_rate(name: str) -> float:
        return sum(result.field_matches[name] for result in results) / len(results)

    reference_results = [result for result in results if result.expected.requires_reference_resolution]
    ambiguous_results = [result for result in results if result.expected.ambiguous or result.predicted.ambiguous]
    pairs: dict[str, list[CaseResult]] = {}
    for case, result in zip(cases, results):
        if case.pair_id:
            pairs.setdefault(case.pair_id, []).append(result)
    pair_matches = [
        len(pair) > 1 and all(pair[0].predicted == item.predicted for item in pair[1:])
        for pair in pairs.values()
    ]
    return {
        "cases": float(len(results)),
        "semantic_task_accuracy": sum(all(result.field_matches.values()) for result in results) / len(results),
        "intent_accuracy": field_rate("intent"),
        "target_resolution": field_rate("target"),
        "target_type_accuracy": field_rate("target_type"),
        "execution_flag_accuracy": field_rate("requires_execution"),
        "modification_flag_accuracy": field_rate("requires_modification"),
        "repository_context_accuracy": field_rate("requires_repository_context"),
        "operation_sequence_accuracy": field_rate("operation_sequence"),
        "reference_resolution": (
            sum(result.field_matches["target"] and result.field_matches["requires_reference_resolution"] for result in reference_results)
            / len(reference_results)
            if reference_results else 1.0
        ),
        "ambiguity_detection": (
            sum(result.field_matches["ambiguous"] for result in ambiguous_results) / len(ambiguous_results)
            if ambiguous_results else 1.0
        ),
        "cross_language_consistency": sum(pair_matches) / len(pair_matches) if pair_matches else 1.0,
    }


def compare_language_policies(cases: Sequence[LanguageCase] = ()) -> Mapping[str, Mapping[str, float]]:
    selected = tuple(cases) or default_language_cases()
    policies = {
        "keyword_parser": keyword_parse_semantic_task,
        "canonical_ir_context": parse_semantic_task,
    }
    return {
        name: summarize_language_results(selected, evaluate_cases(selected, parser))
        for name, parser in policies.items()
    }


if __name__ == "__main__":
    print(json.dumps(compare_language_policies(), ensure_ascii=False, indent=2, sort_keys=True))
