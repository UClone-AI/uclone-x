"""What a provider's error leaves for whoever debugs it, and what U0 sees (#2227).

On 2026-10-10 every tool turn on `gemini-3.8-flash` failed with "Google returned an error
UClone-X didn't recognise", and Google's own text -- "Function call is missing a
thought_signature" -- was recorded nowhere: not in the session's event log, not in
`ucx.log`, which the dashboard server and its reload worker never wrote to. These tests
hold that the status and a capped, redacted excerpt of the body reach the failed step's
event and the server's log, that a key never does, and that U0's sentence stays plain.
"""

from __future__ import annotations

import json
import logging
import multiprocessing
import os
from pathlib import Path
from typing import Any

import httpx
import pytest

from uclone_x.agent.base import BaseAgent
from uclone_x.agent.models import AgentConfig, AgentContext, AgentLLMConfig
from uclone_x.agent.session import SessionStore
from uclone_x.core.logging_setup import server_log_config
from uclone_x.errors import ProviderResponseError
from uclone_x.llm.connectors.failures import failed_status
from uclone_x.llm.connectors.gemini import GeminiConnector
from uclone_x.log.reader import read_session_log

#: The shape of a Google API key, made up: never a real one.
FAKE_KEY = "AIza" + "SyD-fake0123456789abcdefghijklmnopq"
GOOGLE_400 = json.dumps(
    {
        "error": {
            "code": 400,
            "message": "Function call is missing a thought_signature in functionCall parts. "
            f"key={FAKE_KEY}",
            "status": "INVALID_ARGUMENT",
        }
    }
)


def test_an_unrecognised_error_keeps_its_status_and_a_redacted_capped_body() -> None:
    """
    Killed by: src/uclone_x/llm/connectors/failures.py :: return _answered(ProviderResponseError(provider=provider, model=model), status_code, body)
    Becomes: return ProviderResponseError(provider=provider, model=model)
    """
    error = failed_status(provider="Google", model="m", status_code=400, body=GOOGLE_400 * 50)
    assert isinstance(error, ProviderResponseError)
    assert error.status_code == 400
    assert error.body_excerpt is not None
    assert "missing a thought_signature" in error.body_excerpt
    assert FAKE_KEY not in error.body_excerpt
    assert len(error.body_excerpt) <= 2000
    # U0's sentence carries none of it.
    assert "400" not in str(error) and "thought_signature" not in str(error)


def test_a_google_key_is_redacted_from_text() -> None:
    r"""
    Killed by: src/uclone_x/core/secrets.py :: re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b"),
    Becomes: re.compile(r"\bAIza[0-9A-Za-z_-]{99}\b"),
    """
    from uclone_x.core.secrets import redact_credentials

    # Bare, as a provider quotes it back: no "api_key=" for the generic rule to catch.
    assert FAKE_KEY not in redact_credentials(f"request from {FAKE_KEY} was refused")


@pytest.mark.asyncio
async def test_the_failed_steps_event_keeps_googles_answer_and_the_turn_says_it_plainly(
    tmp_path: Path,
) -> None:
    """
    Killed by: src/uclone_x/agent/turn_executor.py :: **_provider_answer(invoke_exc),
    Becomes: **{},
    """

    def google(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, text=GOOGLE_400)

    llm = GeminiConnector(
        model="gemini-3.8-flash",
        api_key="key",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(google)),
    )
    store = SessionStore(storage_dir=tmp_path / "sessions")
    agent = BaseAgent(
        config=AgentConfig(
            agent_id="a", name="A", llm_config=AgentLLMConfig(model_name="gemini-3.8-flash")
        ),
        llm=llm,
        store=store,
        context=AgentContext(session_id="sess_detail", agent_id="a"),
    )
    await agent.start()
    streamed: list[Any] = []
    result = await agent.execute_turn("hi", stream_callback=lambda e, d: streamed.append(e))
    assert not result.is_completed
    assert result.provider_failure is not None
    assert result.provider_failure.message == (
        "Google returned an error UClone-X didn't recognise. If it happens again, check the "
        "model name and any custom endpoint."
    )
    agent.persist_session()
    [log] = list((tmp_path / "sessions").rglob("*.jsonl"))
    [failed] = [e for e in read_session_log(log) if e.get("type") == "MODEL_RESPONSE"]
    assert failed["error"]["provider_status"] == 400
    assert "missing a thought_signature" in failed["error"]["provider_body"]
    assert FAKE_KEY not in json.dumps(failed)


def _spawned_server_logs(log_dir: str) -> None:
    """What a uvicorn reload worker does first (`uvicorn._subprocess.subprocess_started`):
    configure logging from the config it was given, in a fresh interpreter; then a provider
    error and an unhandled exception, as the server would log them."""
    os.environ["UCX_LOG_DIR"] = log_dir
    import uvicorn

    from uclone_x.core.logging_setup import server_log_config as config
    from uclone_x.llm.connectors.failures import failed_status as failed

    uvicorn.Config("x:y", log_config=config()).configure_logging()
    failed(provider="Google", model="m", status_code=400, body=GOOGLE_400)
    try:
        raise RuntimeError("boom in a route")
    except RuntimeError:
        logging.getLogger("uvicorn.error").exception("Exception in ASGI application")


def test_a_spawned_server_worker_writes_the_log_without_the_key(tmp_path: Path) -> None:
    """The `--dev` reload worker is a spawned interpreter: the parent's handler is not in
    it, and it wrote nothing (2026-10-10). The log config uvicorn hands it is ours.

    Killed by: src/uclone_x/core/logging_setup.py :: config["root"] = {"handlers": ["ucx_file"], "level": "WARNING"}
    Becomes: config["root"] = {"handlers": [], "level": "WARNING"}
    Killed by: src/uclone_x/core/logging_setup.py :: config["loggers"]["uclone_x"] = {"level": "INFO"}
    Becomes: config["loggers"]["uclone_x"] = {"level": "WARNING"}
    Killed by: src/uclone_x/core/logging_setup.py :: config["loggers"]["uvicorn"]["handlers"].append("ucx_file")
    Becomes: pass
    """
    worker = multiprocessing.get_context("spawn").Process(
        target=_spawned_server_logs, args=(str(tmp_path),)
    )
    worker.start()
    worker.join(timeout=60)
    assert worker.exitcode == 0
    records = [
        json.loads(line) for line in (tmp_path / "ucx.log").read_text(encoding="utf-8").splitlines()
    ]
    text = json.dumps(records)
    assert any(
        "answered 400" in r["message"] and "thought_signature" in r["message"] for r in records
    )
    assert any("boom in a route" in r.get("exception", "") for r in records)
    assert FAKE_KEY not in text


def test_both_ways_of_serving_hand_uvicorn_the_log_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`ucx ui` and `ucx ui --dev` (and `ucx start`, which serves as `ucx ui` does).

    Killed by: src/uclone_x/ui/server.py :: log_config = server_log_config()
    Becomes: log_config = None
    """
    import subprocess

    from uclone_x.ui import server

    monkeypatch.setenv("UCX_LOG_DIR", str(tmp_path))
    calls: list[dict[str, Any]] = []

    def run(*args: object, **kwargs: Any) -> None:
        calls.append(kwargs)

    def app(**kwargs: object) -> _Stub:
        return _Stub()

    def vite_answers(host: str, port: int) -> None:
        return None

    def vite(*args: object, **kwargs: object) -> _Vite:
        return _Vite()

    monkeypatch.setattr(server.uvicorn, "run", run)
    monkeypatch.setattr(server, "create_ui_app", app)
    monkeypatch.setattr(server, "_diagnose_vite", vite_answers)
    monkeypatch.setattr(subprocess, "Popen", vite)
    for dev in (False, True):
        server._serve(  # pyright: ignore[reportPrivateUsage]
            port=0,
            dev=dev,
            host="127.0.0.1",
            vite_port=0,
            storage_dir=tmp_path,
            workspace_dir=None,
            timeout_graceful_shutdown=1,
        )
    assert [c["log_config"] for c in calls] == [server_log_config(), server_log_config()]


class _Stub:
    class state:  # noqa: N801 -- stands in for FastAPI's `app.state`
        person_gate = None


class _Vite:
    """A Vite that has already exited, so nothing is stopped or waited on."""

    def poll(self) -> int:
        return 0


def _spawned_server_terminal(log_dir: str, stderr_path: str) -> None:
    """A server process configured from the log config, its stderr written to a file: a
    WARNING and an INFO from UClone-X's own code, and a WARNING from uvicorn."""
    import sys

    os.environ["UCX_LOG_DIR"] = log_dir
    sys.stderr = open(stderr_path, "w", encoding="utf-8")  # noqa: SIM115 -- kept for the run
    import uvicorn

    from uclone_x.core.logging_setup import server_log_config as config

    uvicorn.Config("x:y", log_config=config()).configure_logging()
    logging.getLogger("uclone_x.room.service").warning("a turn failed for room r1")
    logging.getLogger("uclone_x.room.service").info("a turn began in room r1")
    logging.getLogger("uvicorn.error").warning("uvicorn says this once")
    sys.stderr.flush()


def _spawned_server_secret(log_dir: str, stderr_path: str) -> None:
    """A server process configured from the log config logs a WARNING carrying a key and a
    URL password, its stderr written to a file."""
    import sys

    os.environ["UCX_LOG_DIR"] = log_dir
    sys.stderr = open(stderr_path, "w", encoding="utf-8")  # noqa: SIM115 -- kept for the run
    import uvicorn

    from uclone_x.core.logging_setup import server_log_config as config

    uvicorn.Config("x:y", log_config=config()).configure_logging()
    logging.getLogger("uclone_x.llm").warning(
        "provider said: bad key %s at %s",
        "sk-ant-" + "A" * 30,
        "https://u:p4ssw0rd@example.com",
    )
    sys.stderr.flush()


def test_the_server_terminal_and_file_never_show_a_key_or_url_password(tmp_path: Path) -> None:
    """The terminal handler formats through the same redaction as the file (#2319 review).

    Killed by: src/uclone_x/core/logging_setup.py :: "()": "uclone_x.core.logging_setup.RedactingFormatter",
    Becomes: "()": "logging.Formatter",
    """
    stderr_path = tmp_path / "stderr.txt"
    worker = multiprocessing.get_context("spawn").Process(
        target=_spawned_server_secret, args=(str(tmp_path), str(stderr_path))
    )
    worker.start()
    worker.join(timeout=60)
    assert worker.exitcode == 0
    terminal = stderr_path.read_text(encoding="utf-8")
    logged = (tmp_path / "ucx.log").read_text(encoding="utf-8")
    for shown in (terminal, logged):
        assert "provider said: bad key" in shown, shown
        assert "sk-ant-AAAA" not in shown, shown
        assert "p4ssw0rd" not in shown, shown


def _spawned_uvicorn_lines(log_dir: str, stderr_path: str) -> None:
    """A server process configured from the log config prints what uvicorn itself prints:
    an ASGI exception's traceback holding a key, and an access line carrying a token and a
    URL with a password."""
    import sys

    os.environ["UCX_LOG_DIR"] = log_dir
    sys.stderr = open(stderr_path, "w", encoding="utf-8")  # noqa: SIM115 -- kept for the run
    sys.stdout = sys.stderr
    import uvicorn

    from uclone_x.core.logging_setup import server_log_config as config

    uvicorn.Config("x:y", log_config=config()).configure_logging()
    try:
        raise RuntimeError("upstream refused key " + "sk-ant-" + "B" * 30)
    except RuntimeError:
        logging.getLogger("uvicorn.error").exception("Exception in ASGI application")
    logging.getLogger("uvicorn.access").info(
        '%s - "%s %s HTTP/%s" %d',
        "127.0.0.1:5000",
        "GET",
        "/cb?token=abcdefghijklmnop1234&next=https://u:p4ssw0rd@example.com/",
        "1.1",
        200,
    )
    sys.stderr.flush()


def test_uvicorns_own_lines_reach_the_terminal_redacted(tmp_path: Path) -> None:
    """uvicorn's traceback and access lines go through the same redaction as ours (#2327).

    Killed by: src/uclone_x/core/logging_setup.py :: _redact_uvicorn_formatters(config)
    Becomes: pass
    """
    stderr_path = tmp_path / "stderr.txt"
    worker = multiprocessing.get_context("spawn").Process(
        target=_spawned_uvicorn_lines, args=(str(tmp_path), str(stderr_path))
    )
    worker.start()
    worker.join(timeout=60)
    assert worker.exitcode == 0
    terminal = stderr_path.read_text(encoding="utf-8")
    assert "Exception in ASGI application" in terminal, terminal
    assert "GET /cb?token=" in terminal, terminal
    for secret in ("sk-ant-BBBB", "abcdefghijklmnop1234", "p4ssw0rd"):
        assert secret not in terminal, (secret, terminal)


def test_the_server_terminal_still_shows_warnings_once(tmp_path: Path) -> None:
    """The file handler on the root logger (#2231) took the place of Python's last-resort
    printer, so the server's own WARNINGs reached the launching terminal no more (#2234).
    They do again, INFO stays off it, uvicorn's lines are not doubled, and the file keeps
    what it kept.

    Killed by: src/uclone_x/core/logging_setup.py :: cast(list[str], config["root"]["handlers"]).append("ucx_terminal")
    Becomes: pass
    """
    stderr_path = tmp_path / "stderr.txt"
    worker = multiprocessing.get_context("spawn").Process(
        target=_spawned_server_terminal, args=(str(tmp_path), str(stderr_path))
    )
    worker.start()
    worker.join(timeout=60)
    assert worker.exitcode == 0
    terminal = stderr_path.read_text(encoding="utf-8")
    assert terminal.count("a turn failed for room r1") == 1
    assert "a turn began" not in terminal
    assert terminal.count("uvicorn says this once") == 1
    logged = [
        json.loads(line)["message"]
        for line in (tmp_path / "ucx.log").read_text(encoding="utf-8").splitlines()
    ]
    assert logged == [
        "a turn failed for room r1",
        "a turn began in room r1",
        "uvicorn says this once",
    ]
