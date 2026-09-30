from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import threading

from coding_world_benchmark.external_llm_reviewer_l54_experiment import ProposalType
from coding_world_benchmark.real_environment_l56_experiment import (
    RealEnvironmentTask,
    RealRepositoryAdapter,
    deterministic_reviewer,
    run_real_environment_task,
)
from coding_world_benchmark.live_web_retrieval_l53_experiment import LiveRetrievalStatus


class _EvidenceHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/missing":
            self.send_response(404)
            self.end_headers()
            return
        body = b"<html><title>Local Evidence</title><body>Claim: effect size evidence</body></html>"
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args):
        pass


def _server():
    server = HTTPServer(("127.0.0.1", 0), _EvidenceHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def _repo(tmp_path: Path) -> Path:
    (tmp_path / "scientific_code.py").write_text("VALUE = 2\n", encoding="utf-8")
    (tmp_path / "test_scientific_code.py").write_text("import unittest\nimport scientific_code\n\nclass ScientificTest(unittest.TestCase):\n    def test_value(self):\n        self.assertEqual(scientific_code.VALUE, 2)\n", encoding="utf-8")
    return tmp_path


def test_real_repository_adapter_inspects_files_and_runs_real_tests(tmp_path):
    root = _repo(tmp_path)
    adapter = RealRepositoryAdapter(root)

    snapshot = adapter.inspect()
    result = adapter.run_tests()

    assert "scientific_code.py" in snapshot.files
    assert "test_scientific_code.py" in snapshot.files
    assert result.passed
    assert result.return_code == 0


def test_real_http_and_reviewer_boundaries_are_connected(tmp_path):
    root = _repo(tmp_path)
    server, thread = _server()
    try:
        task = RealEnvironmentTask("Review the effect size evidence.", root, f"http://127.0.0.1:{server.server_port}/evidence", ("effect size",), True)
        result = run_real_environment_task(task, deterministic_reviewer)
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()

    assert result["http_status"] == LiveRetrievalStatus.SUCCESS.value
    assert result["http_transport_success"]
    assert result["content_extracted"]
    assert result["review_called"]
    assert result["review_proposal_type"] == ProposalType.METHODOLOGICAL_RISK.value
    assert result["tests_passed"]
    assert result["integration_success"]


def test_reviewer_is_not_called_when_workspace_marks_review_unnecessary(tmp_path):
    root = _repo(tmp_path)
    server, thread = _server()
    calls = []

    def reviewer(request):
        calls.append(request)
        return deterministic_reviewer(request)

    try:
        task = RealEnvironmentTask("The evidence chain is sufficient.", root, f"http://127.0.0.1:{server.server_port}/evidence", ("effect size",), False)
        result = run_real_environment_task(task, reviewer)
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()

    assert not calls
    assert not result["review_called"]
    assert result["integration_success"]


def test_http_404_does_not_become_an_integration_success(tmp_path):
    root = _repo(tmp_path)
    server, thread = _server()
    try:
        task = RealEnvironmentTask("Fetch missing evidence.", root, f"http://127.0.0.1:{server.server_port}/missing", ("effect size",), False)
        result = run_real_environment_task(task)
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()

    assert result["http_status"] == LiveRetrievalStatus.SOURCE_NOT_FOUND.value
    assert not result["content_extracted"]
    assert not result["integration_success"]
