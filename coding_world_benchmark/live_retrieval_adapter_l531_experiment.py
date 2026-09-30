"""L5.3.1: HTTP/browser retrieval adapters and deterministic replay.

This layer stops transport success from being confused with usable evidence.
It normalizes raw responses into documents and maps both transport and content
failures to the L5.3 status vocabulary.  The benchmark uses injected
transports, so its behavior is reproducible without network access.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import json
import re
from typing import Callable, Mapping

from .live_web_retrieval_l53_experiment import LiveRetrievalStatus


class AdapterSource(Enum):
    HTTP = "http"
    BROWSER = "browser"


@dataclass(frozen=True)
class RetrievalRequest:
    url: str
    query: str
    required_terms: tuple[str, ...]
    max_age_days: int = 3650


@dataclass(frozen=True)
class RawResponse:
    status_code: int | None
    headers: Mapping[str, str]
    body: str
    final_url: str
    error: str | None = None


@dataclass(frozen=True)
class RawDocument:
    title: str
    url: str
    text: str
    fetched_at: str
    published_at: str | None
    source: AdapterSource


@dataclass(frozen=True)
class RetrievalResult:
    status: LiveRetrievalStatus
    source: AdapterSource
    document: RawDocument | None
    transport_success: bool
    content_extracted: bool
    used_browser_fallback: bool = False
    replayed: bool = False
    detail: str = ""


HttpTransport = Callable[[str], RawResponse]
BrowserTransport = Callable[[str], RawResponse]


def _now() -> datetime:
    return datetime(2026, 9, 14, tzinfo=timezone.utc)


def _html_to_text(body: str) -> tuple[str, str]:
    title_match = re.search(r"<title[^>]*>(.*?)</title>", body, re.I | re.S)
    title = re.sub(r"\s+", " ", title_match.group(1)).strip() if title_match else ""
    body_without_scripts = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", body, flags=re.I | re.S)
    text = re.sub(r"<[^>]+>", " ", body_without_scripts)
    return title, re.sub(r"\s+", " ", text).strip()


def normalize_response(request: RetrievalRequest, response: RawResponse,
                       source: AdapterSource, *, replayed: bool = False) -> RetrievalResult:
    if response.error == "timeout":
        return RetrievalResult(LiveRetrievalStatus.RETRIEVAL_TIMEOUT, source, None, False, False, replayed=replayed, detail=response.error)
    if response.error or response.status_code in {403, 429} or (response.status_code is not None and response.status_code >= 500):
        return RetrievalResult(LiveRetrievalStatus.HTTP_OR_BROWSER_FAILURE, source, None, False, False, replayed=replayed, detail=response.error or str(response.status_code))
    if response.status_code == 404:
        return RetrievalResult(LiveRetrievalStatus.SOURCE_NOT_FOUND, source, None, True, False, replayed=replayed, detail="404")
    if response.status_code != 200:
        return RetrievalResult(LiveRetrievalStatus.HTTP_OR_BROWSER_FAILURE, source, None, False, False, replayed=replayed, detail=str(response.status_code))
    content_type = next((value for key, value in response.headers.items() if key.lower() == "content-type"), "")
    if "html" not in content_type.lower() and "text" not in content_type.lower():
        return RetrievalResult(LiveRetrievalStatus.CONTENT_PARSE_FAILURE, source, None, True, False, replayed=replayed, detail="content-type")
    title, text = _html_to_text(response.body)
    lowered = text.lower()
    if not text or "enable javascript" in lowered or "accept cookies" in lowered or "checking your browser" in lowered:
        return RetrievalResult(LiveRetrievalStatus.CONTENT_PARSE_FAILURE, source, None, True, False, replayed=replayed, detail="empty-or-gated")
    if not all(term.lower() in lowered for term in request.required_terms):
        return RetrievalResult(LiveRetrievalStatus.INSUFFICIENT_EVIDENCE, source, None, True, True, replayed=replayed, detail="required-term-missing")
    published_match = re.search(r"data-published=\"([^\"]+)\"", response.body, re.I)
    published_at = published_match.group(1) if published_match else None
    if published_at:
        published = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
        if published.tzinfo is None:
            published = published.replace(tzinfo=timezone.utc)
        if (_now() - published).days > request.max_age_days:
            document = RawDocument(title, response.final_url, text, _now().isoformat(), published_at, source)
            return RetrievalResult(LiveRetrievalStatus.SOURCE_STALE, source, document, True, True, replayed=replayed, detail="age-limit")
    document = RawDocument(title, response.final_url, text, _now().isoformat(), published_at, source)
    return RetrievalResult(LiveRetrievalStatus.SUCCESS, source, document, True, True, replayed=replayed)


class HttpRetrievalAdapter:
    def __init__(self, transport: HttpTransport):
        self.transport = transport

    def fetch(self, request: RetrievalRequest, *, replayed: bool = False) -> RetrievalResult:
        try:
            response = self.transport(request.url)
        except TimeoutError:
            response = RawResponse(None, {}, "", request.url, "timeout")
        except OSError as exc:
            response = RawResponse(None, {}, "", request.url, str(exc))
        return normalize_response(request, response, AdapterSource.HTTP, replayed=replayed)


class BrowserRetrievalAdapter:
    def __init__(self, transport: BrowserTransport):
        self.transport = transport

    def open(self, request: RetrievalRequest, *, replayed: bool = False) -> RetrievalResult:
        try:
            response = self.transport(request.url)
        except TimeoutError:
            response = RawResponse(None, {}, "", request.url, "timeout")
        except OSError as exc:
            response = RawResponse(None, {}, "", request.url, str(exc))
        return normalize_response(request, response, AdapterSource.BROWSER, replayed=replayed)


def fetch_with_browser_fallback(request: RetrievalRequest, http: HttpRetrievalAdapter,
                                browser: BrowserRetrievalAdapter) -> RetrievalResult:
    result = http.fetch(request)
    if result.status in {LiveRetrievalStatus.RETRIEVAL_TIMEOUT, LiveRetrievalStatus.HTTP_OR_BROWSER_FAILURE,
                         LiveRetrievalStatus.CONTENT_PARSE_FAILURE, LiveRetrievalStatus.SOURCE_NOT_FOUND}:
        fallback = browser.open(request)
        if fallback.status is LiveRetrievalStatus.SOURCE_NOT_FOUND and result.status is not LiveRetrievalStatus.SOURCE_NOT_FOUND:
            return result
        return RetrievalResult(fallback.status, fallback.source, fallback.document, fallback.transport_success,
                                fallback.content_extracted, used_browser_fallback=True,
                                replayed=fallback.replayed, detail=fallback.detail)
    return result


class RecordReplayTransport:
    def __init__(self, transport: HttpTransport | BrowserTransport | None = None,
                 records: list[dict[str, object]] | None = None):
        self.transport = transport
        self.records = records if records is not None else []

    def __call__(self, url: str) -> RawResponse:
        if self.transport is None:
            record = next(item for item in self.records if item["url"] == url)
            return RawResponse(record["status_code"], record["headers"], record["body"], record["final_url"], record["error"])
        response = self.transport(url)
        self.records.append({"url": url, "status_code": response.status_code, "headers": dict(response.headers),
                             "body": response.body, "final_url": response.final_url, "error": response.error})
        return response

    def dumps(self) -> str:
        return json.dumps(self.records, ensure_ascii=False, sort_keys=True)

    @classmethod
    def loads(cls, payload: str) -> "RecordReplayTransport":
        return cls(records=json.loads(payload))


@dataclass(frozen=True)
class AdapterCase:
    name: str
    request: RetrievalRequest
    http_response: RawResponse
    browser_response: RawResponse | None
    expected_status: LiveRetrievalStatus
    expected_browser_fallback: bool = False


def _html(text: str, *, title: str = "Reference", published: str | None = "2026-09-01") -> str:
    marker = f' data-published="{published}"' if published else ""
    return f"<html><title>{title}</title><body{marker}>{text}</body></html>"


def cases() -> tuple[AdapterCase, ...]:
    def request(name: str, terms: tuple[str, ...] = ("effect size",), age: int = 3650) -> RetrievalRequest:
        return RetrievalRequest(f"https://adapter.example.test/{name}", "sample size", terms, age)
    valid = RawResponse(200, {"Content-Type": "text/html"}, _html("Use effect size and target power."), "https://adapter.example.test/valid")
    return (
        AdapterCase("http_valid", request("valid"), valid, None, LiveRetrievalStatus.SUCCESS),
        AdapterCase("http_irrelevant", request("irrelevant"), RawResponse(200, {"Content-Type": "text/html"}, _html("A history of astronomy."), "https://adapter.example.test/irrelevant"), None, LiveRetrievalStatus.INSUFFICIENT_EVIDENCE),
        AdapterCase("http_stale", request("stale", age=30), RawResponse(200, {"Content-Type": "text/html"}, _html("Use effect size and target power.", published="2020-01-01"), "https://adapter.example.test/stale"), None, LiveRetrievalStatus.SOURCE_STALE),
        AdapterCase("http_parse_failure", request("parse"), RawResponse(200, {"Content-Type": "text/html"}, _html("Enable JavaScript to continue."), "https://adapter.example.test/parse"), None, LiveRetrievalStatus.CONTENT_PARSE_FAILURE),
        AdapterCase("http_404", request("404"), RawResponse(404, {"Content-Type": "text/html"}, "", "https://adapter.example.test/404"), None, LiveRetrievalStatus.SOURCE_NOT_FOUND),
        AdapterCase("http_403", request("403"), RawResponse(403, {"Content-Type": "text/html"}, "", "https://adapter.example.test/403"), None, LiveRetrievalStatus.HTTP_OR_BROWSER_FAILURE),
        AdapterCase("http_timeout", request("timeout"), RawResponse(None, {}, "", "https://adapter.example.test/timeout", "timeout"), None, LiveRetrievalStatus.RETRIEVAL_TIMEOUT),
        AdapterCase("http_5xx", request("5xx"), RawResponse(503, {}, "", "https://adapter.example.test/5xx"), None, LiveRetrievalStatus.HTTP_OR_BROWSER_FAILURE),
        AdapterCase("browser_fallback", request("browser"), RawResponse(200, {"Content-Type": "text/html"}, _html("Enable JavaScript to continue."), "https://adapter.example.test/browser"), RawResponse(200, {"Content-Type": "text/html"}, _html("Browser rendered effect size evidence."), "https://adapter.example.test/browser"), LiveRetrievalStatus.SUCCESS, True),
        AdapterCase("redirect_loop", request("redirect"), RawResponse(None, {}, "", "https://adapter.example.test/redirect", "redirect loop"), None, LiveRetrievalStatus.HTTP_OR_BROWSER_FAILURE),
    )


def run_adapter_experiment() -> dict[str, object]:
    rows = []
    for case in cases():
        http = HttpRetrievalAdapter(lambda _url, response=case.http_response: response)
        browser = BrowserRetrievalAdapter(lambda _url, response=case.browser_response: response or RawResponse(404, {}, "", case.request.url))
        result = fetch_with_browser_fallback(case.request, http, browser)
        rows.append({"case": case.name, "status": result.status.value, "expected_status": case.expected_status.value,
                     "status_correct": result.status is case.expected_status, "transport_success": result.transport_success,
                     "content_extracted": result.content_extracted, "browser_fallback": result.used_browser_fallback,
                     "false_success": result.status is LiveRetrievalStatus.SUCCESS and result.document is None})
    return {"rows": rows, "metrics": {
        "transport_success_rate": sum(row["transport_success"] for row in rows) / len(rows),
        "status_mapping_accuracy": sum(row["status_correct"] for row in rows) / len(rows),
        "http_to_browser_fallback_rate": sum(row["browser_fallback"] for row in rows) / len(rows),
        "false_success_rate": sum(row["false_success"] for row in rows) / len(rows),
        "content_extraction_accuracy": sum(row["content_extracted"] == (row["status"] in {
            LiveRetrievalStatus.SUCCESS.value,
            LiveRetrievalStatus.SOURCE_STALE.value,
            LiveRetrievalStatus.INSUFFICIENT_EVIDENCE.value,
        }) for row in rows) / len(rows),
        "adapter_recovery_rate": sum(row["status_correct"] and row["browser_fallback"] for row in rows) / max(1, sum(row["browser_fallback"] for row in rows)),
    }}


if __name__ == "__main__":
    print(json.dumps(run_adapter_experiment(), ensure_ascii=False, indent=2, sort_keys=True))
