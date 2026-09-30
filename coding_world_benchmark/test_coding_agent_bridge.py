import json
from http.client import HTTPConnection
from pathlib import Path
from threading import Thread
from http.server import ThreadingHTTPServer

from coding_world_benchmark.coding_agent_app import ChatResult
from coding_world_benchmark.coding_agent_bridge import BridgeHandler
import coding_world_benchmark.coding_agent_bridge as bridge


def test_bridge_validates_workspace_and_returns_normalized_path(tmp_path: Path):
    original = BridgeHandler.agent
    BridgeHandler.agent = type("Agent", (), {
        "workspace": None,
        "set_workspace": lambda self, path: setattr(self, "workspace", path.resolve()) or f"作業フォルダを設定しました: {self.workspace}",
        "handle": lambda self, message: ChatResult(f"received: {message}", "local"),
    })()
    server = ThreadingHTTPServer(("127.0.0.1", 0), BridgeHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        connection = HTTPConnection("127.0.0.1", server.server_port)
        body = json.dumps({"path": str(tmp_path)}).encode()
        connection.request("POST", "/api/workspace", body, {"Content-Type": "application/json", "Origin": "http://127.0.0.1:5173"})
        response = connection.getresponse()
        payload = json.loads(response.read())
        assert response.status == 200
        assert response.getheader("Access-Control-Allow-Origin") == "http://127.0.0.1:5173"
        assert payload["workspace"] == str(tmp_path.resolve())
        connection.close()
    finally:
        server.shutdown()
        server.server_close()
        BridgeHandler.agent = original


def test_bridge_rejects_missing_workspace(tmp_path: Path):
    original = BridgeHandler.agent
    BridgeHandler.agent = type("Agent", (), {
        "workspace": None,
        "set_workspace": lambda self, path: (_ for _ in ()).throw(ValueError("作業フォルダが存在しません")),
    })()
    server = ThreadingHTTPServer(("127.0.0.1", 0), BridgeHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        connection = HTTPConnection("127.0.0.1", server.server_port)
        connection.request("POST", "/api/workspace", json.dumps({"path": str(tmp_path / "missing")}).encode(), {"Content-Type": "application/json"})
        response = connection.getresponse()
        assert response.status == 400
        assert "存在しません" in json.loads(response.read())["error"]
        connection.close()
    finally:
        server.shutdown()
        server.server_close()
        BridgeHandler.agent = original


def test_bridge_exposes_integrated_feature_commands():
    server = ThreadingHTTPServer(("127.0.0.1", 0), BridgeHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        connection = HTTPConnection("127.0.0.1", server.server_port)
        connection.request("GET", "/api/features")
        response = connection.getresponse()
        payload = json.loads(response.read())
        commands = {item["command"] for item in payload["features"]}
        assert response.status == 200
        assert "/self-benchmark" in commands
        assert "/holdout-benchmark" in commands
        assert "/behavior-audit" in commands
        connection.close()
    finally:
        server.shutdown()
        server.server_close()


def test_bridge_chat_without_workspace_answers_without_local_investigation():
    original = BridgeHandler.agent
    BridgeHandler.agent = bridge.LocalWorkspaceAgent()
    server = ThreadingHTTPServer(("127.0.0.1", 0), BridgeHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        connection = HTTPConnection("127.0.0.1", server.server_port)
        body = json.dumps({"message": "一様磁場中で荷電粒子が運動するシミュレーションを作成して"}).encode()
        connection.request("POST", "/api/chat", body, {"Content-Type": "application/json"})
        response = connection.getresponse()
        payload = json.loads(response.read())
        assert response.status == 200
        assert payload["provider"] == "no-workspace"
        assert payload["workspace"] is None
        assert "ファイル一覧・検索・コード読取は行っていません" in payload["text"]
        assert "GoalIR" in payload["text"]
        connection.close()
    finally:
        server.shutdown()
        server.server_close()
        BridgeHandler.agent = original


def test_bridge_uses_task_contract_memory_policy_for_behavior_monitor_proposals():
    assert bridge.memory_policy_for_message("TaskSpec ExpectedBehavior ObservedBehavior Behavior Monitorを実装して") == "task_contract_only"


def test_bridge_uses_scientific_modeling_policy_for_new_simulation_requests():
    assert bridge.memory_policy_for_message("一様磁場中で荷電粒子が運動するシミュレーションを作成して") == "scientific_modeling_only"


def test_bridge_native_picker_sets_selected_workspace(tmp_path: Path, monkeypatch):
    original = BridgeHandler.agent
    BridgeHandler.agent = type("Agent", (), {
        "workspace": None,
        "set_workspace": lambda self, path: setattr(self, "workspace", path.resolve()) or f"selected: {self.workspace}",
    })()
    monkeypatch.setattr(bridge, "choose_local_folder", lambda: str(tmp_path))
    server = ThreadingHTTPServer(("127.0.0.1", 0), BridgeHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        connection = HTTPConnection("127.0.0.1", server.server_port)
        connection.request("POST", "/api/workspace/pick/?source=ui")
        response = connection.getresponse()
        payload = json.loads(response.read())
        assert response.status == 200
        assert payload["workspace"] == str(tmp_path.resolve())
        connection.close()
    finally:
        server.shutdown()
        server.server_close()
        BridgeHandler.agent = original
