from coding_world_benchmark.live_retrieval_adapter_l531_experiment import (
    AdapterSource,
    BrowserRetrievalAdapter,
    HttpRetrievalAdapter,
    RawResponse,
    RecordReplayTransport,
    RetrievalRequest,
    cases,
    fetch_with_browser_fallback,
    run_adapter_experiment,
)
from coding_world_benchmark.live_web_retrieval_l53_experiment import LiveRetrievalStatus


REPORT = run_adapter_experiment()


def test_adapter_cases_cover_http_and_browser_failure_boundaries():
    assert len(cases()) == 10
    assert {case.expected_status for case in cases()} == {
        LiveRetrievalStatus.SUCCESS,
        LiveRetrievalStatus.INSUFFICIENT_EVIDENCE,
        LiveRetrievalStatus.SOURCE_STALE,
        LiveRetrievalStatus.CONTENT_PARSE_FAILURE,
        LiveRetrievalStatus.SOURCE_NOT_FOUND,
        LiveRetrievalStatus.HTTP_OR_BROWSER_FAILURE,
        LiveRetrievalStatus.RETRIEVAL_TIMEOUT,
    }


def test_http_200_is_not_automatically_success():
    for name in {"http_irrelevant", "http_stale", "http_parse_failure"}:
        row = next(row for row in REPORT["rows"] if row["case"] == name)
        assert row["status"] != LiveRetrievalStatus.SUCCESS.value
        assert row["transport_success"]


def test_browser_fallback_recovers_rendered_content():
    case = next(case for case in cases() if case.name == "browser_fallback")
    http = HttpRetrievalAdapter(lambda _url: case.http_response)
    browser = BrowserRetrievalAdapter(lambda _url: case.browser_response)

    result = fetch_with_browser_fallback(case.request, http, browser)

    assert result.status is LiveRetrievalStatus.SUCCESS
    assert result.source is AdapterSource.BROWSER
    assert result.used_browser_fallback
    assert result.document is not None
    assert "effect size" in result.document.text.lower()


def test_record_replay_preserves_raw_response_and_normalized_status():
    response = RawResponse(200, {"Content-Type": "text/html"}, "<title>x</title><body>effect size evidence</body>", "https://replay.test/x")
    recorder = RecordReplayTransport(lambda _url: response)
    request = RetrievalRequest("https://replay.test/x", "sample size", ("effect size",))
    recorded = HttpRetrievalAdapter(recorder).fetch(request)
    replay = RecordReplayTransport.loads(recorder.dumps())
    replayed = HttpRetrievalAdapter(replay).fetch(request, replayed=True)

    assert recorded.status is LiveRetrievalStatus.SUCCESS
    assert replayed.status is recorded.status
    assert replayed.document == recorded.document
    assert replayed.replayed


def test_adapter_has_no_false_success_and_maps_all_cases_correctly():
    metrics = REPORT["metrics"]
    assert metrics["status_mapping_accuracy"] == 1.0
    assert metrics["false_success_rate"] == 0.0
    assert metrics["content_extraction_accuracy"] == 1.0
    assert metrics["adapter_recovery_rate"] == 1.0

