"""Small local HTTP bridge between the React UI and LocalWorkspaceAgent."""
from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import subprocess
import sys

from .coding_agent_app import AGENT_FEATURE_COMMANDS, LocalWorkspaceAgent
from .dialogue_script import looks_like_script, run as run_script
from .physics_script import looks_like_physics, run as run_physics
from . import theorem_service
from .semantic_service import (
    SEMANTIC_FEATURE_COMMANDS, SemanticSession, handle_command, render_state,
    templates_payload,
)
from .conversation_memory_l73_experiment import ConversationKnowledgeMemory
from .external_repair_l81_experiment import detect_repair_intent
from .scientific_simulation_l88_experiment import is_scientific_simulation_implementation_request
from .goal_semantics_l89_experiment import extract_goal


def memory_policy_for_message(message: str) -> str:
    lower = message.lower()
    if any(token in lower for token in ("taskspec", "expectedbehavior", "observedbehavior", "behavior monitor", "file_modification")):
        return "task_contract_only"
    goal = extract_goal(message)
    if "artifact_created" in goal.completion_conditions:
        return "scientific_modeling_only" if "simulation" in goal.desired_properties else "repair_related_only"
    repair_intent = detect_repair_intent(message)
    if is_scientific_simulation_implementation_request(message):
        return "scientific_modeling_only"
    if repair_intent.score >= 0.72:
        return "repair_related_only"
    is_workspace_action = any(word in message for word in ("移動", "削除", "消して", "フォルダ")) and any(word in message for word in ("SU2", "SU3", "directory", "folder", "検証"))
    return "path_resolution_only" if is_workspace_action else "all"


def choose_local_folder() -> str | None:
    """Open the picker in a dedicated Python/Tk main process.

    The HTTP server handles requests on worker threads; Tk must own its main
    thread on Windows, so it is deliberately isolated in this subprocess.
    """
    picker = (
        "import tkinter as tk; from tkinter import filedialog; "
        "root=tk.Tk(); root.title('Coding Agent'); root.withdraw(); "
        "root.attributes('-topmost', True); root.update(); "
        "selected=filedialog.askdirectory(parent=root, title='コーディング作業フォルダを選択'); "
        "print(selected, flush=True); root.destroy()"
    )
    pythonw = Path(sys.executable).with_name("pythonw.exe")
    executable = str(pythonw) if pythonw.is_file() else sys.executable
    completed = subprocess.run([executable, "-c", picker], capture_output=True,
                               text=True, encoding="utf-8", timeout=120,
                               check=False)
    if completed.returncode != 0:
        raise OSError(completed.stderr.strip() or "Pythonフォルダ選択ダイアログを起動できませんでした")
    selected = completed.stdout.strip().splitlines()
    return selected[-1] if selected and selected[-1] else None


class BridgeHandler(BaseHTTPRequestHandler):
    agent = LocalWorkspaceAgent()
    memory = ConversationKnowledgeMemory(Path(__file__).with_name(".knowledge_memory.json.gz"))
    #: The interpretation state (L8.27-L8.42). One per process, like the agent:
    #: it is a person's session, not a request, and the whole point of the layers
    #: is that it survives across turns.
    semantics = SemanticSession()

    def _send(self, status: int, payload: dict[str, object]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        origin = self.headers.get("Origin", "")
        if origin in {"http://localhost:5173", "http://127.0.0.1:5173"}:
            self.send_header("Access-Control-Allow-Origin", origin)
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        origin = self.headers.get("Origin", "")
        if origin in {"http://localhost:5173", "http://127.0.0.1:5173"}:
            self.send_header("Access-Control-Allow-Origin", origin)
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0].rstrip("/") or "/"
        if path == "/api/health":
            self._send(200, {"ok": True, "workspace": str(self.agent.workspace) if self.agent.workspace else None})
            return
        if path == "/api/theorem/status":
            self._send(200, {"ok": True, "status": theorem_service.status()})
            return
        if path == "/api/semantic/templates":
            # Static, so it is its own endpoint rather than riding along on
            # every state snapshot: the panel fetches it once.
            self._send(200, {"ok": True, "templates": templates_payload()})
            return
        if path == "/api/semantic/state":
            self._send(200, {"ok": True, "state": self.semantics.state()})
            return
        if path == "/api/features":
            self._send(200, {"ok": True, "features": [
                {"command": command, "description": description}
                for command, description in AGENT_FEATURE_COMMANDS + SEMANTIC_FEATURE_COMMANDS
            ]})
            return
        self._send(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
            path = self.path.split("?", 1)[0].rstrip("/")
            if path == "/api/workspace":
                workspace = self.agent.set_workspace(Path(str(payload.get("path", ""))))
                self._send(200, {"ok": True, "message": workspace, "workspace": str(self.agent.workspace)})
                return
            if path == "/api/workspace/pick":
                selected = choose_local_folder()
                if not selected:
                    self._send(200, {"ok": True, "cancelled": True})
                    return
                workspace = self.agent.set_workspace(Path(selected))
                self._send(200, {"ok": True, "message": workspace, "workspace": str(self.agent.workspace)})
                return
            if path.startswith("/api/theorem"):
                self._send(200, self._theorem(path, payload))
                return
            if path.startswith("/api/semantic"):
                self._send(200, {"ok": True,
                                 "state": self._semantic(path, payload)})
                return
            if self.path == "/api/chat":
                message = str(payload.get("message", ""))
                # A pending question owns the next utterance: answering it is
                # what L8.33 binds the reply to, and routing it to the ordinary
                # agent instead would lose the question it was an answer to.
                spoken = handle_command(self.semantics, message)
                if spoken is None and self.semantics.dialogue.pending is not None:
                    state = self.semantics.ask(message)
                    spoken = f"{state['reply']}\n\n" + render_state(state)
                if spoken is not None:
                    self._send(200, {"ok": True, "text": spoken, "provider": "semantics",
                                     "semantic": self.semantics.state(),
                                     "workspace": str(self.agent.workspace) if self.agent.workspace else None})
                    return
                result = self.agent.handle(message)
                policy = memory_policy_for_message(message)
                remembered = self.memory.context(message, policy=policy)
                if remembered and policy == "all":
                    result = type(result)(result.text + "\n\n【再利用された知識】\n" + remembered, result.provider)
                self.memory.learn(message, result.text)
                self.memory.save()
                self._send(200, {"ok": True, "text": result.text, "provider": result.provider,
                                 "workspace": str(self.agent.workspace) if self.agent.workspace else None})
                return
            self._send(404, {"error": "not found"})
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            self._send(400, {"ok": False, "error": str(exc)})

    def _theorem(self, path: str, payload: dict) -> dict:
        """The discovery bundle. Each call is a subprocess with a timeout."""
        if path == "/api/theorem/search":
            query = str(payload.get("query", ""))
            found = theorem_service.search(query)
            return {"ok": True, "results": found,
                    "output": theorem_service.format_search(query, found)}
        if path in ("/api/theorem/run", "/api/theorem/obligations"):
            cycles = int(payload.get("cycles") or theorem_service.DEFAULT_CYCLES)
            outcome = theorem_service.run_discovery(cycles)
            if not outcome.ok:
                return {"ok": False, "error": outcome.detail, "output":
                        f"**回せませんでした**: {outcome.detail}"}
            body = {"ok": True, "summary": outcome.payload.get("summary", {}),
                    "asserted": outcome.payload.get("asserted", []),
                    "output": theorem_service.format_run(outcome.payload)}
            if path.endswith("obligations"):
                body["obligations"] = outcome.payload.get("obligations", "")
            return body
        if path == "/api/theorem/verify":
            outcome = theorem_service.verify()
            return {"ok": outcome.ok, "detail": outcome.detail,
                    "counts": outcome.payload.get("counts", {}),
                    "output": f"`{outcome.detail}`"}
        raise ValueError(f"unknown theorem endpoint: {path}")

    def _semantic(self, path: str, payload: dict) -> dict:
        """The interpretation state, and the few actions that change it."""
        session = self.semantics
        if path == "/api/semantic/ask":
            return session.ask(str(payload.get("message", "")))
        if path == "/api/semantic/answer":
            return session.answer(str(payload.get("text", "")))
        if path == "/api/semantic/open":
            session.open_episode(str(payload.get("episode", "")))
            return session.state()
        if path == "/api/semantic/episode":
            session.use_episode(str(payload.get("episode", "")))
            return session.state()
        if path == "/api/semantic/script":
            # The same envelope as every other action -- a state, with what the
            # script did attached -- so the panel has one shape to render.
            source = str(payload.get("source", ""))
            # One box, two languages: which one this is follows from whether it
            # parses, not from a mode the person has to remember to set.
            if looks_like_physics(source) and not looks_like_script(source):
                found = run_physics(source)
                state = session.state()
                state["script"] = {"ok": found["ok"], "error": found["error"],
                                   "results": found["results"],
                                   "needs": found["needs"],
                                   "output": found["output"],
                                   "language": "physics"}
            else:
                found = run_script(session, source)
                state = found["state"]
                state["script"] = {"ok": found["ok"], "error": found["error"],
                                   "results": found["results"], "needs": [],
                                   "output": found["output"],
                                   "language": "dialogue"}
            state["reply"] = found["output"]
            return state
        if path == "/api/semantic/model":
            return session.answer_structural(str(payload.get("label", "")))
        if path == "/api/semantic/reset":
            session.reset()
            return session.state()
        if path == "/api/semantic/rerun":
            state = session.state()
            state["rerun"] = session.rerun(str(payload.get("request", "")))
            return state
        raise ValueError(f"unknown semantic endpoint: {path}")

    def log_message(self, _format: str, *_args: object) -> None:
        return


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8787
    server = ThreadingHTTPServer(("127.0.0.1", port), BridgeHandler)
    print(f"Coding agent bridge listening on http://127.0.0.1:{port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
