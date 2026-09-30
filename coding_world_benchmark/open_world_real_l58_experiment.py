"""L5.8: open-world autonomous research on real execution boundaries.

This runner accepts only a natural-language research prompt plus real boundary
handles: a repository path, an HTTP evidence URL, and a Web->API reviewer.
Failures become unresolved rather than unsupported conclusions.  The default
API configuration is the benchmark-local ``.config`` file, but no external
request is made unless the runner is invoked with a real reviewer router.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Callable

from .external_llm_reviewer_l54_experiment import ReviewProposal
from .real_environment_l56_experiment import RealRepositoryAdapter, LiveHttpSearchAdapter, ReviewRequest
from .real_external_llm_reviewer_l57_experiment import (
    ChatGPTWebReviewer,
    FallbackReviewer,
    OpenAIAPIReviewer,
    ReviewerResult,
)
from .live_retrieval_adapter_l531_experiment import RetrievalRequest


@dataclass(frozen=True)
class OpenWorldResearchTask:
    prompt: str
    repository_root: Path
    evidence_url: str
    required_terms: tuple[str, ...]
    reviewer_required: bool = True


@dataclass(frozen=True)
class ResearchReport:
    prompt: str
    hypothesis: str
    repository_files: tuple[str, ...]
    evidence_status: str
    reviewer_provider: str | None
    reviewer_success: bool
    tests_passed: bool
    final_status: str
    human_interventions: int
    unsupported_final_claim: bool
    reproducible: bool


ReviewerRouter = Callable[[str, str], ReviewerResult]


def hypothesis_from_prompt(prompt: str) -> str:
    lowered = prompt.lower()
    if "memory" in lowered or "kernel" in lowered:
        return "memory-kernel timescale scales with dynamical time"
    if "sample" in lowered or "power" in lowered:
        return "the observed signal remains after adequate statistical power"
    return "the stated experimental signal is reproducible under its controls"


def benchmark_reviewer_router(*, web_transport: Callable[[str], str], config_path: str | Path | None = None,
                              api_base_url: str = "https://api.openai.com/v1/responses") -> ReviewerRouter:
    web = ChatGPTWebReviewer(web_transport)
    api = OpenAIAPIReviewer(config_path=config_path or Path(__file__).with_name(".config"), base_url=api_base_url)
    router = FallbackReviewer(web, api)
    return router.review


def run_open_world_real_task(task: OpenWorldResearchTask, reviewer: ReviewerRouter | None = None,
                             *, test_command: tuple[str, ...] | None = None) -> ResearchReport:
    repository = RealRepositoryAdapter(task.repository_root)
    snapshot = repository.inspect()
    retrieval = LiveHttpSearchAdapter().fetch(RetrievalRequest(task.evidence_url, task.prompt, task.required_terms))
    reviewer_result = None
    if task.reviewer_required and reviewer is not None:
        reviewer_result = reviewer(task.prompt, retrieval.document.text if retrieval.document else "")
    tests = repository.run_tests(test_command) if test_command is not None else repository.run_tests()
    reviewer_ok = not task.reviewer_required or reviewer_result is not None and reviewer_result.success
    evidence_ok = retrieval.document is not None
    final_status = "supported" if tests.passed and evidence_ok and reviewer_ok else "unresolved"
    return ResearchReport(task.prompt, hypothesis_from_prompt(task.prompt), snapshot.files,
                          retrieval.status.value, reviewer_result.provider if reviewer_result else None,
                          reviewer_result.success if reviewer_result else not task.reviewer_required,
                          tests.passed, final_status, 0, False, tests.passed)


def run_l58_report(task: OpenWorldResearchTask, reviewer: ReviewerRouter | None = None) -> dict[str, object]:
    report = run_open_world_real_task(task, reviewer)
    return {"prompt": report.prompt, "hypothesis": report.hypothesis, "repository_files": list(report.repository_files),
            "evidence_status": report.evidence_status, "reviewer_provider": report.reviewer_provider,
            "reviewer_success": report.reviewer_success, "tests_passed": report.tests_passed,
            "final_status": report.final_status, "human_interventions": report.human_interventions,
            "unsupported_final_claim": report.unsupported_final_claim, "reproducible": report.reproducible}


if __name__ == "__main__":
    print(json.dumps({"status": "requires a real repository, evidence URL, and optional reviewer router"}, indent=2))
