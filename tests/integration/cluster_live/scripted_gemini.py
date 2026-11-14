"""A scripted model behind Gemini's HTTP API: the provider boundary, for a real `ucx worker`.

The worker runs its real `GeminiConnector` against this server (`GEMINI_BASE_URL`), so a call
crosses a real socket, its counts come from a real response, and a held call is a worker really
waiting in the middle of a model call when the test kills it. No test hook exists in the worker.

Gemini is the connector to use here because it counts as paid whatever its address, so its
calls pass the usage gate; the OpenAI connector treats a loopback endpoint as free.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


@dataclass
class ScriptedTurn:
    """How the model answers a request whose latest user text holds `marker`."""

    marker: str
    #: Answered in order, one per call; the last repeats. A dict is a function call.
    answers: list[str | dict[str, Any]]
    input_tokens: int = 137
    output_tokens: int = 41
    hold: threading.Event | None = None
    entered: threading.Event = field(default_factory=threading.Event)
    calls: int = 0


class ScriptedGemini:
    """The server; `script` what to answer, read `requests` for what was asked."""

    def __init__(self) -> None:
        self._turns: list[ScriptedTurn] = []
        self._lock = threading.Lock()
        self.requests: list[dict[str, Any]] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, format: str, *args: Any) -> None:
                return None

            def _send(self, status: int, content_type: str, body: bytes) -> None:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:
                self._send(200, "application/json", b'{"models": []}')

            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", "0"))
                request: dict[str, Any] = json.loads(self.rfile.read(length) or b"{}")
                body = outer.answer(request)
                try:
                    if "streamGenerateContent" in self.path:
                        self._send(
                            200,
                            "text/event-stream",
                            b"data: " + json.dumps(body).encode() + b"\n\n",
                        )
                    else:
                        self._send(200, "application/json", json.dumps(body).encode())
                except (BrokenPipeError, ConnectionResetError):
                    pass  # the worker was killed while it waited

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._server.server_address[1]}/v1beta"

    def script(self, marker: str, *answers: str | dict[str, Any], **fields: Any) -> ScriptedTurn:
        turn = ScriptedTurn(marker=marker, answers=list(answers), **fields)
        with self._lock:
            self._turns.append(turn)
        return turn

    def close(self) -> None:
        for turn in self._turns:
            if turn.hold is not None:
                turn.hold.set()
        self._server.shutdown()
        self._server.server_close()

    def requests_for(self, marker: str) -> list[dict[str, Any]]:
        return [r for r in self.requests if marker in _latest_user_text(r)]

    def answer(self, request: dict[str, Any]) -> dict[str, Any]:
        asked = _latest_user_text(request)
        with self._lock:
            self.requests.append(request)
            turn = next((t for t in reversed(self._turns) if t.marker in asked), None)
            if turn is None:
                return _body("Unscripted.", 1, 1)
            turn.calls += 1
            position = min(turn.calls, len(turn.answers)) - 1
        turn.entered.set()
        if turn.hold is not None:
            turn.hold.wait()
        return _body(turn.answers[position], turn.input_tokens, turn.output_tokens)


def _latest_user_text(request: dict[str, Any]) -> str:
    for content in reversed(request.get("contents", [])):
        if content.get("role") == "user":
            parts = [p.get("text", "") for p in content.get("parts", []) if "text" in p]
            if parts:
                return " ".join(parts)
    return ""


def _body(answer: str | dict[str, Any], input_tokens: int, output_tokens: int) -> dict[str, Any]:
    part: dict[str, Any] = {"text": answer} if isinstance(answer, str) else {"functionCall": answer}
    return {
        "candidates": [{"content": {"role": "model", "parts": [part]}, "finishReason": "STOP"}],
        "usageMetadata": {
            "promptTokenCount": input_tokens,
            "candidatesTokenCount": output_tokens,
            "totalTokenCount": input_tokens + output_tokens,
        },
        "modelVersion": "scripted-model",
    }
