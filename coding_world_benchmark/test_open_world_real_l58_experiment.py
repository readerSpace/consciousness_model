from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import threading

from coding_world_benchmark.external_llm_reviewer_l54_experiment import ProposalType, ReviewProposal
from coding_world_benchmark.live_retrieval_adapter_l531_experiment import LiveRetrievalStatus
from coding_world_benchmark.open_world_real_l58_experiment import (
    OpenWorldResearchTask,
    benchmark_reviewer_router,
    hypothesis_from_prompt,
    run_l58_report,
    run_open_world_real_task,
)
from coding_world_benchmark.real_external_llm_reviewer_l57_experiment import ReviewerResult


REVIEW_JSON = json.dumps({"proposals": [{
    "proposal_type": "methodological_risk",
    "statement": "Check the measurement assumption.",
    "rationale": "The evidence should match the repository measurement.",
    "suggested_search_queries": ["measurement validation"],
    "target_claim_ids": ["repo-claim"],
    "reviewer_confidence": 0.74,
}]})


class _ResearchHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/missing":
            self.send_response(404)
            self.end_headers()
            return
        body = b"<html><title>Research Evidence</title><body>Claim: effect size evidence</body></html>"
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
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
    server = HTTPServer(("127.0.0.1", 0), _ResearchHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def _repo(tmp_path: Path) -> Path:
    (tmp_path / "measurement.py").write_text("VALUE = 3\n", encoding="utf-8")
    (tmp_path / "test_measurement.py").write_text("import unittest\nimport measurement\n\nclass T(unittest.TestCase):\n    def test_value(self):\n        self.assertEqual(measurement.VALUE, 3)\n", encoding="utf-8")
    return tmp_path


def test_natural_language_research_prompt_generates_domain_hypothesis():
    assert "memory-kernel" in hypothesis_from_prompt("Test whether the memory kernel scales with dynamical time.")
    assert "reproducible" in hypothesis_from_prompt("Test the signal.")


def test_l58_connects_real_repo_http_and_api_fallback(tmp_path: Path):
    config = tmp_path / ".config"
    config.write_text("OPENAI_API_KEY=sk-local-test\n", encoding="utf-8")
    server, thread = _server()
    try:
        router = benchmark_reviewer_router(
            web_transport=lambda _prompt: (_ for _ in ()).throw(ConnectionError("web unavailable")),
            config_path=config,
            api_base_url=f"http://127.0.0.1:{server.server_port}/v1/responses",
        )
        task = OpenWorldResearchTask("Test the effect size evidence in this repository.", _repo(tmp_path),
                                     f"http://127.0.0.1:{server.server_port}/evidence", ("effect size",), True)
        report = run_l58_report(task, router)
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()

    assert report["evidence_status"] == LiveRetrievalStatus.SUCCESS.value
    assert report["reviewer_provider"] == "openai_api"
    assert report["reviewer_success"]
    assert report["tests_passed"]
    assert report["human_interventions"] == 0
    assert report["unsupported_final_claim"] is False
    assert report["final_status"] == "supported"


def test_l58_stops_unresolved_when_http_evidence_is_missing(tmp_path: Path):
    server, thread = _server()
    try:
        task = OpenWorldResearchTask("Test missing evidence.", _repo(tmp_path),
                                     f"http://127.0.0.1:{server.server_port}/missing", ("effect size",), False)
        report = run_l58_report(task)
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()

    assert report["evidence_status"] == LiveRetrievalStatus.SOURCE_NOT_FOUND.value
    assert report["final_status"] == "unresolved"
    assert not report["unsupported_final_claim"]


def test_l58_can_use_a_deterministic_reviewer_at_the_same_boundary(tmp_path: Path):
    proposal = ReviewProposal(ProposalType.METHODOLOGICAL_RISK, "Check assumptions.", "risk", ("query",), ("claim",), .5)
    reviewer = lambda _prompt, _evidence: ReviewerResult(True, (proposal,), "{}", None, 0.01, "deterministic")
    server, thread = _server()
    try:
        task = OpenWorldResearchTask("Test effect size evidence.", _repo(tmp_path),
                                     f"http://127.0.0.1:{server.server_port}/evidence", ("effect size",), True)
        report = run_open_world_real_task(task, reviewer)
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()

    assert report.final_status == "supported"
    assert report.reviewer_provider == "deterministic"
