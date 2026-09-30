from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import threading

from coding_world_benchmark.external_llm_reviewer_l54_experiment import ProposalType
from coding_world_benchmark.real_external_llm_reviewer_l57_experiment import (
    ChatGPTWebReviewer,
    FallbackReviewer,
    OpenAIAPIReviewer,
    ReviewerFailureType,
    build_review_prompt,
    load_chatgpt_api_key,
    parse_review_proposals,
    run_l57_case,
)


REVIEW_JSON = json.dumps({"proposals": [{
    "proposal_type": "counter_hypothesis",
    "statement": "Shared drift may explain the signal.",
    "rationale": "The control is incomplete.",
    "suggested_search_queries": ["shared drift negative control"],
    "target_claim_ids": ["cite-1"],
    "reviewer_confidence": 0.7,
}]})


class _APIHandler(BaseHTTPRequestHandler):
    auth_seen = False

    def do_POST(self):
        _APIHandler.auth_seen = bool(self.headers.get("Authorization", "").startswith("Bearer "))
        length = int(self.headers["Content-Length"])
        json.loads(self.rfile.read(length))
        body = json.dumps({"output_text": REVIEW_JSON}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args):
        pass


def _server():
    server = HTTPServer(("127.0.0.1", 0), _APIHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def test_structured_review_proposals_are_parsed_without_promoting_them_to_evidence():
    proposals = parse_review_proposals(REVIEW_JSON)
    prompt = build_review_prompt("Test the signal.", "cite-1: positive")

    assert len(proposals) == 1
    assert proposals[0].proposal_type is ProposalType.COUNTER_HYPOTHESIS
    assert proposals[0].suggested_queries == ("shared drift negative control",)
    assert "Do not add evidence" in prompt


def test_web_reviewer_success_is_used_first():
    calls = []
    reviewer = ChatGPTWebReviewer(lambda prompt: calls.append(prompt) or REVIEW_JSON)
    result = reviewer.review("Review task", "cite-1")

    assert result.success
    assert result.provider == "chatgpt_web"
    assert len(result.proposals) == 1
    assert calls


def test_api_fallback_uses_config_key_without_returning_the_secret(tmp_path: Path):
    config = tmp_path / ".config"
    config.write_text("CHATGPT_API_KEY=sk-test-secret\n", encoding="utf-8")
    server, thread = _server()
    try:
        api = OpenAIAPIReviewer(config_path=config, base_url=f"http://127.0.0.1:{server.server_port}/v1/responses")
        fallback = FallbackReviewer(ChatGPTWebReviewer(lambda _prompt: (_ for _ in ()).throw(ConnectionError("web down"))), api)
        result = fallback.review("Review task", "cite-1")
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()

    assert result.success
    assert result.provider == "openai_api"
    assert len(result.proposals) == 1
    assert _APIHandler.auth_seen
    assert "sk-test-secret" not in result.raw_response


def test_missing_config_is_a_classified_failure():
    result = OpenAIAPIReviewer(config_path="C:/path/that/does/not/exist").review("Review", "evidence")

    assert not result.success
    assert result.failure_type is ReviewerFailureType.API_CONFIG_MISSING


def test_fallback_result_has_no_secret_exposure_and_records_latency(tmp_path: Path):
    config = tmp_path / ".config"
    config.write_text("OPENAI_API_KEY=sk-another-secret\n", encoding="utf-8")
    reviewer = FallbackReviewer(ChatGPTWebReviewer(lambda _prompt: REVIEW_JSON), OpenAIAPIReviewer(config_path=config))
    result = run_l57_case("Review", "citation", reviewer)

    assert result["success"]
    assert result["provider"] == "chatgpt_web"
    assert result["latency_recorded"]
    assert not result["api_key_exposed"]
    assert load_chatgpt_api_key(config) == "sk-another-secret"
