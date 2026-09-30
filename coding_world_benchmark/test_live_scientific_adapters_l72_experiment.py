from coding_world_benchmark.knowledge_retrieval_l71_experiment import SearchResult
from coding_world_benchmark.live_scientific_adapters_l72_experiment import (
    HttpPageFetcher,
    HttpSearchAdapter,
    live_web_enabled,
)


SEARCH_HTML = '''
<html><a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Farxiv.org%2Fabs%2F1234">Paper title</a>
<a class="result__snippet">Entanglement geometry evidence.</a></html>
'''
PAGE_HTML = "<html><script>ignore()</script><body>Useful evidence <b>and equation</b>.</body></html>"


def test_http_search_adapter_normalizes_search_results_without_network():
    adapter = HttpSearchAdapter(transport=lambda url: SEARCH_HTML)
    results = adapter.search("quantum information")

    assert len(results) == 1
    assert isinstance(results[0], SearchResult)
    assert results[0].url == "https://arxiv.org/abs/1234"
    assert results[0].source_type == "paper"
    assert results[0].authority == 0.95


def test_http_page_fetcher_extracts_visible_text_without_network():
    fetcher = HttpPageFetcher(transport=lambda url: PAGE_HTML)
    assert fetcher.fetch("https://example.test") == "Useful evidence and equation."


def test_live_web_feature_flag_is_explicit(monkeypatch):
    monkeypatch.delenv("CODING_AGENT_LIVE_WEB", raising=False)
    assert live_web_enabled() is False
    monkeypatch.setenv("CODING_AGENT_LIVE_WEB", "true")
    assert live_web_enabled() is True
