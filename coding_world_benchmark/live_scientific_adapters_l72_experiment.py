"""L7.2: live Web adapters for the scientific retrieval boundary.

The adapters use DuckDuckGo's HTML endpoint for search and urllib for page
fetching. They are intentionally separate from ScientificRetriever so tests
can inject deterministic transports and the app can fall back to offline
replay when the network is unavailable.
"""
from __future__ import annotations

from html import unescape
from html.parser import HTMLParser
import os
import re
from typing import Callable
from urllib.parse import parse_qs, quote_plus, unquote, urljoin, urlparse
from urllib.request import Request, urlopen

from .knowledge_retrieval_l71_experiment import SearchResult


HttpTransport = Callable[[str], str]


class _SearchParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.results: list[tuple[str, str, str]] = []
        self._url = ""
        self._title: list[str] = []
        self._snippet: list[str] = []
        self._in_title = False
        self._in_snippet = False

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        classes = set(attributes.get("class", "").split())
        if tag == "a" and "result__a" in classes:
            self._url = self._normalize_url(attributes.get("href", ""))
            self._title = []
            self._in_title = True
        elif "result__snippet" in classes:
            self._snippet = []
            self._in_snippet = True

    def handle_endtag(self, tag):
        if tag == "a" and self._in_title:
            self._in_title = False
        if self._in_snippet:
            self._in_snippet = False
            if self._url and self._title:
                self.results.append((self._url, "".join(self._title).strip(), "".join(self._snippet).strip()))

    def handle_data(self, data):
        if self._in_title:
            self._title.append(data)
        if self._in_snippet:
            self._snippet.append(data)

    @staticmethod
    def _normalize_url(value: str) -> str:
        if not value:
            return ""
        parsed = urlparse(value)
        if parsed.path.startswith("/l/") and "uddg" in parse_qs(parsed.query):
            return unquote(parse_qs(parsed.query)["uddg"][0])
        return value if parsed.scheme else urljoin("https://html.duckduckgo.com", value)


class HttpSearchAdapter:
    def __init__(self, transport: HttpTransport | None = None, endpoint: str | None = None, timeout: float = 10.0):
        self.transport = transport or self._request
        self.endpoint = endpoint or "https://html.duckduckgo.com/html/"
        self.timeout = timeout

    def _request(self, url: str) -> str:
        request = Request(url, headers={"User-Agent": "ConsciousCodingAgent/0.1 scientific-research"})
        with urlopen(request, timeout=self.timeout) as response:
            return response.read().decode("utf-8", errors="replace")

    def search(self, query: str) -> list[SearchResult]:
        url = f"{self.endpoint}?q={quote_plus(query)}"
        try:
            parser = _SearchParser()
            parser.feed(self.transport(url))
        except (OSError, TimeoutError, ValueError):
            return []
        results = []
        for target, title, snippet in parser.results[:8]:
            source_type = "paper" if "arxiv.org" in target or "doi.org" in target else "web"
            authority = 0.95 if source_type == "paper" else _domain_authority(target)
            results.append(SearchResult(unescape(title), target, unescape(snippet), source_type, authority, None))
        return results


class _TextParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts: list[str] = []
        self.hidden = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "noscript"}:
            self.hidden += 1

    def handle_endtag(self, tag):
        if tag in {"script", "style", "noscript"} and self.hidden:
            self.hidden -= 1

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)


class HttpPageFetcher:
    def __init__(self, transport: HttpTransport | None = None, timeout: float = 10.0, max_chars: int = 30000):
        self.transport = transport or self._request
        self.timeout = timeout
        self.max_chars = max_chars

    def _request(self, url: str) -> str:
        request = Request(url, headers={"User-Agent": "ConsciousCodingAgent/0.1 scientific-research"})
        with urlopen(request, timeout=self.timeout) as response:
            return response.read().decode("utf-8", errors="replace")

    def fetch(self, url: str) -> str:
        try:
            body = self.transport(url)
        except (OSError, TimeoutError, ValueError):
            return ""
        parser = _TextParser()
        parser.feed(body)
        text = " ".join(" ".join(parser.parts).split())
        text = re.sub(r"\s+([,.;:!?])", r"\1", text)
        return text[: self.max_chars]


def _domain_authority(url: str) -> float:
    host = urlparse(url).netloc.lower()
    if host.endswith((".gov", ".edu", ".ac.jp")):
        return 0.9
    if host in {"wikipedia.org", "en.wikipedia.org"}:
        return 0.75
    return 0.55


def live_web_enabled() -> bool:
    return os.getenv("CODING_AGENT_LIVE_WEB", "").lower() in {"1", "true", "yes", "on"}
