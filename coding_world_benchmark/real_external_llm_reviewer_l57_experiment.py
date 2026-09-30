"""L5.7: real external reviewer routing with Web -> API fallback.

The reviewer remains a proposal generator.  Its output is parsed into the
L5.4 ``ReviewProposal`` type and must still be verified by the citation layers.
The Web client is injected so a browser controller can be used in production;
when it fails, ``OpenAIAPIReviewer`` uses a key discovered from ``.config``.
Secrets are never included in results or logs.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path
import time
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .external_llm_reviewer_l54_experiment import ProposalType, ReviewProposal


class ReviewerFailureType(Enum):
    WEB_UNAVAILABLE = "web_unavailable"
    API_CONFIG_MISSING = "api_config_missing"
    API_HTTP_FAILURE = "api_http_failure"
    API_TIMEOUT = "api_timeout"
    RESPONSE_PARSE_FAILURE = "response_parse_failure"


@dataclass(frozen=True)
class ReviewerResult:
    success: bool
    proposals: tuple[ReviewProposal, ...]
    raw_response: str
    failure_type: ReviewerFailureType | None
    latency_s: float | None
    provider: str


def load_chatgpt_api_key(config_path: str | Path | None = None) -> str | None:
    candidates = [Path(config_path)] if config_path else [Path.cwd() / ".config", Path.home() / ".config"]
    for candidate in candidates:
        paths = [candidate] if candidate.is_file() else sorted(candidate.rglob("*") if candidate.is_dir() else ())
        for path in paths:
            if not path.is_file() or path.name.lower() not in {".config", "config", "chatgpt", "chatgpt.env", ".env"}:
                continue
            for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
                key, separator, value = line.partition("=")
                if not separator:
                    key, separator, value = line.partition(":")
                if separator and key.strip().upper() in {"OPENAI_API_KEY", "CHATGPT_API_KEY"}:
                    value = value.strip().strip("\"'")
                    if value:
                        return value
    return None


def build_review_prompt(prompt: str, evidence_chain: str) -> str:
    return ("Review the experiment and citation chain below. Return JSON array only. "
            "Propose missing evidence, counter-hypotheses, alternative explanations, "
            "additional search queries, or methodological risks. Do not add evidence "
            "or decide the final conclusion.\n\nTASK:\n" + prompt + "\n\nEVIDENCE:\n" + evidence_chain)


def parse_review_proposals(raw: str) -> tuple[ReviewProposal, ...]:
    payload = json.loads(raw)
    if isinstance(payload, dict):
        payload = payload.get("proposals", [])
    if not isinstance(payload, list):
        raise ValueError("review response must be a JSON array")
    proposals = []
    for item in payload:
        kind = ProposalType(item["proposal_type"])
        queries = item.get("suggested_search_queries", item.get("suggested_queries", []))
        targets = item.get("target_claim_ids", item.get("target_claim", []))
        if isinstance(queries, str):
            queries = [queries]
        if isinstance(targets, str):
            targets = [targets]
        proposals.append(ReviewProposal(kind, str(item["statement"]), str(item.get("rationale", item.get("reason", ""))),
                                         tuple(queries), tuple(targets), float(item.get("reviewer_confidence", 0.0))))
    return tuple(proposals)


class ChatGPTWebReviewer:
    def __init__(self, transport: Callable[[str], str]):
        self.transport = transport

    def review(self, prompt: str, evidence_chain: str) -> ReviewerResult:
        started = time.perf_counter()
        try:
            raw = self.transport(build_review_prompt(prompt, evidence_chain))
            proposals = parse_review_proposals(raw)
            return ReviewerResult(True, proposals, raw, None, time.perf_counter() - started, "chatgpt_web")
        except Exception as exc:
            return ReviewerResult(False, (), str(exc), ReviewerFailureType.WEB_UNAVAILABLE, time.perf_counter() - started, "chatgpt_web")


class OpenAIAPIReviewer:
    def __init__(self, api_key: str | None = None, *, config_path: str | Path | None = None,
                 model: str = "gpt-5", base_url: str = "https://api.openai.com/v1/responses", timeout: float = 20.0):
        self.api_key = api_key or load_chatgpt_api_key(config_path)
        self.model = model
        self.base_url = base_url
        self.timeout = timeout

    def review(self, prompt: str, evidence_chain: str) -> ReviewerResult:
        if not self.api_key:
            return ReviewerResult(False, (), "", ReviewerFailureType.API_CONFIG_MISSING, None, "openai_api")
        started = time.perf_counter()
        body = json.dumps({"model": self.model, "input": build_review_prompt(prompt, evidence_chain)}, ensure_ascii=False).encode("utf-8")
        request = Request(self.base_url, data=body, headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}, method="POST")
        try:
            with urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
            raw = payload.get("output_text", "")
            proposals = parse_review_proposals(raw)
            return ReviewerResult(True, proposals, raw, None, time.perf_counter() - started, "openai_api")
        except TimeoutError:
            failure = ReviewerFailureType.API_TIMEOUT
            raw = ""
        except (HTTPError, URLError, OSError) as exc:
            failure = ReviewerFailureType.API_HTTP_FAILURE
            raw = str(exc)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            failure = ReviewerFailureType.RESPONSE_PARSE_FAILURE
            raw = str(exc)
        return ReviewerResult(False, (), raw, failure, time.perf_counter() - started, "openai_api")


class FallbackReviewer:
    def __init__(self, web: ChatGPTWebReviewer, api: OpenAIAPIReviewer):
        self.web = web
        self.api = api

    def review(self, prompt: str, evidence_chain: str) -> ReviewerResult:
        web_result = self.web.review(prompt, evidence_chain)
        if web_result.success:
            return web_result
        return self.api.review(prompt, evidence_chain)


def run_l57_case(prompt: str, evidence_chain: str, reviewer: FallbackReviewer) -> dict[str, object]:
    result = reviewer.review(prompt, evidence_chain)
    return {
        "success": result.success,
        "provider": result.provider,
        "proposal_count": len(result.proposals),
        "failure_type": result.failure_type.value if result.failure_type else None,
        "latency_recorded": result.latency_s is not None,
        "api_key_exposed": "sk-" in result.raw_response,
    }


if __name__ == "__main__":
    print(json.dumps({"status": "requires a Web reviewer or .config API key"}, indent=2))
