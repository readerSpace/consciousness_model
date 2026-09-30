"""L5.6: real-boundary integration harness.

Unlike the earlier sealed layers, this module operates on a caller-provided
repository, performs an actual subprocess test run, and fetches an actual HTTP
URL.  The reviewer remains an explicit callable boundary so a real LLM client
can be injected in production while tests use a deterministic reviewer.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import sys

from .process_execution import combined_output, run_text
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .external_llm_reviewer_l54_experiment import ReviewProposal
from .live_retrieval_adapter_l531_experiment import (
    AdapterSource,
    RawResponse,
    RetrievalRequest,
    RetrievalResult,
    normalize_response,
)


@dataclass(frozen=True)
class RepositorySnapshot:
    root: str
    files: tuple[str, ...]
    source_excerpt: str


@dataclass(frozen=True)
class TestRunResult:
    command: tuple[str, ...]
    return_code: int
    passed: bool
    output: str


@dataclass(frozen=True)
class ReviewRequest:
    prompt: str
    repository_files: tuple[str, ...]
    evidence_text: str


@dataclass(frozen=True)
class RealEnvironmentTask:
    prompt: str
    repository_root: Path
    source_url: str
    required_terms: tuple[str, ...]
    expected_review: bool


class RealRepositoryAdapter:
    def __init__(self, root: Path):
        self.root = root.resolve()

    def inspect(self) -> RepositorySnapshot:
        files = tuple(sorted(str(path.relative_to(self.root)) for path in self.root.rglob("*") if path.is_file()))
        excerpts = []
        for relative in files[:8]:
            path = self.root / relative
            try:
                excerpts.append(path.read_text(encoding="utf-8")[:500])
            except UnicodeDecodeError:
                continue
        return RepositorySnapshot(str(self.root), files, "\n".join(excerpts))

    def run_tests(self, command: tuple[str, ...] = (sys.executable, "-m", "unittest", "discover"), *, timeout: int = 20) -> TestRunResult:
        completed = run_text(command, cwd=self.root, timeout=timeout)
        output = combined_output(completed, 4000)
        return TestRunResult(command, completed.returncode, completed.returncode == 0, output)


class LiveHttpSearchAdapter:
    def __init__(self, *, timeout: float = 5.0):
        self.timeout = timeout

    def fetch(self, request: RetrievalRequest) -> RetrievalResult:
        http_request = Request(request.url, headers={"User-Agent": "coding-world-benchmark/5.6"})
        try:
            with urlopen(http_request, timeout=self.timeout) as response:
                raw = RawResponse(response.status, dict(response.headers.items()), response.read().decode("utf-8", errors="replace"), response.geturl())
        except HTTPError as exc:
            raw = RawResponse(exc.code, dict(exc.headers.items()), "", request.url)
        except TimeoutError:
            raw = RawResponse(None, {}, "", request.url, "timeout")
        except URLError as exc:
            reason = str(exc.reason).lower()
            raw = RawResponse(None, {}, "", request.url, "timeout" if "timed out" in reason else str(exc.reason))
        return normalize_response(request, raw, AdapterSource.HTTP)


ExternalReviewer = Callable[[ReviewRequest], ReviewProposal]


def run_real_environment_task(task: RealEnvironmentTask, reviewer: ExternalReviewer | None = None,
                              *, test_command: tuple[str, ...] = (sys.executable, "-m", "unittest", "discover")) -> dict[str, object]:
    repository = RealRepositoryAdapter(task.repository_root)
    snapshot = repository.inspect()
    retrieval = LiveHttpSearchAdapter().fetch(RetrievalRequest(task.source_url, task.prompt, task.required_terms))
    review_request = ReviewRequest(task.prompt, snapshot.files, retrieval.document.text if retrieval.document else "")
    proposal = reviewer(review_request) if reviewer is not None and task.expected_review else None
    tests = repository.run_tests(test_command)
    return {
        "prompt": task.prompt,
        "repository_file_count": len(snapshot.files),
        "repository_files": list(snapshot.files),
        "http_status": retrieval.status.value,
        "http_transport_success": retrieval.transport_success,
        "content_extracted": retrieval.content_extracted,
        "review_called": proposal is not None,
        "review_proposal_type": proposal.proposal_type.value if proposal else None,
        "tests_passed": tests.passed,
        "test_return_code": tests.return_code,
        "test_output": tests.output,
        "integration_success": tests.passed and retrieval.document is not None and (not task.expected_review or proposal is not None),
    }


def deterministic_reviewer(request: ReviewRequest) -> ReviewProposal:
    from .external_llm_reviewer_l54_experiment import ProposalType
    return ReviewProposal(ProposalType.METHODOLOGICAL_RISK, "Check that the fetched claim matches the repository measurement.",
                          "The repository and external source should be compared.", ("measurement validation",), ("repo-claim",), .74)


def run_l56_smoke(task: RealEnvironmentTask) -> dict[str, object]:
    return run_real_environment_task(task, deterministic_reviewer)


if __name__ == "__main__":
    print(json.dumps({"status": "requires a caller-provided repository and URL"}, indent=2))
