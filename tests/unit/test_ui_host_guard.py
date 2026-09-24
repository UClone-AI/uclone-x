"""The dashboard refuses a request not addressed to this machine by name (#1413).

A DNS-rebinding page -- `http://attacker.example:5180`, its name re-pointed at 127.0.0.1 --
reaches a loopback-bound dashboard with `Origin` and `Host` both naming its own domain. The
origin check admits that, so the `Host` name is the only thing that tells it apart.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock

import httpx
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from tests.support.vite_diagnosis import answering_as
from uclone_x.llm import MockLLMConnector
from uclone_x.shells import ui_process
from uclone_x.shells.ui_process import UI_BIND_HOST_ENV_VAR
from uclone_x.ui.app import _is_loopback_bind, create_ui_app  # pyright: ignore[reportPrivateUsage]
from uclone_x.ui.server import VITE_IDENTITY_MARKER, start_ui_server

REBOUND = {"Origin": "http://attacker.example:5180", "Host": "attacker.example:5180"}


def _client(tmp_path: Path, **kwargs: object) -> TestClient:
    app = create_ui_app(static_dir=tmp_path, storage_dir=tmp_path / "store", **kwargs)  # pyright: ignore[reportArgumentType]
    return TestClient(app)


def test_a_rebound_page_cannot_read_the_dashboard(tmp_path: Path) -> None:
    """A read route with no check of its own is refused, and the reason names the remedy.

    Killed by: src/uclone_x/ui/app.py :: if _host_header_hostname(host) in _LOOPBACK_HOSTS:
    Becomes: if True:
    """
    client = _client(tmp_path)

    res = client.get("/api/sessions", headers=REBOUND)

    assert res.status_code == 403
    detail = res.json()["detail"]
    assert "http://localhost:5180" in detail
    assert "http://127.0.0.1:5180" in detail


def test_a_rebound_page_cannot_post_a_chat_turn(tmp_path: Path) -> None:
    """The consequence that made this a security issue: a turn runs tools in the workspace.

    The refused turn reaches no model: the first reply is still waiting for the next turn.

    Killed by: src/uclone_x/ui/app.py :: app.add_middleware(LoopbackHostGuard)
    Becomes: pass
    """
    llm = MockLLMConnector(responses=["first", "second"])
    client = _client(tmp_path, llm=llm)
    turn = {"message": "run bash_run", "agent_id": "agent-a", "session_id": "sess_rebound"}

    refused = client.post("/api/turn", json=turn, headers=REBOUND)
    admitted = client.post("/api/turn", json=turn)

    assert refused.status_code == 403
    assert admitted.status_code == 200
    assert admitted.json()["response"] == "first"


@pytest.mark.parametrize("host", ["localhost:5180", "127.0.0.1:5180", "[::1]:5180", "LOCALHOST"])
def test_loopback_names_are_admitted_at_any_port(tmp_path: Path, host: str) -> None:
    """Every name a browser on this machine can use for the dashboard.

    Killed by: src/uclone_x/ui/app.py :: _LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "[::1]"})
    Becomes: _LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "[::1]"})
    """
    client = _client(tmp_path)

    assert client.get("/api/health", headers={"Host": host}).status_code == 200


def test_a_deliberately_exposed_dashboard_is_not_refused(tmp_path: Path) -> None:
    """`ucx ui --host 0.0.0.0` is reached by names this cannot know; refusing them breaks it.

    Killed by: src/uclone_x/ui/app.py :: if _is_loopback_bind(resolved_bind_host):
    Becomes: if True:
    """
    client = _client(tmp_path, bind_host="0.0.0.0")

    assert client.get("/api/health", headers={"Host": "my-desktop.lan:5180"}).status_code == 200


def test_the_bind_host_reaches_a_factory_built_app_through_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reload mode builds the app through uvicorn's factory, which passes no arguments.

    Killed by: src/uclone_x/ui/app.py :: else os.environ.get(UI_BIND_HOST_ENV_VAR, DEFAULT_UI_BIND_HOST)
    Becomes: else DEFAULT_UI_BIND_HOST
    """
    monkeypatch.setenv(UI_BIND_HOST_ENV_VAR, "0.0.0.0")
    client = _client(tmp_path)

    assert client.get("/api/health", headers={"Host": "my-desktop.lan:5180"}).status_code == 200


@pytest.mark.usefixtures("frontend_build_suppressed")
def test_reload_mode_exports_the_bind_host_for_the_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without it a reload-mode dashboard shared on the network would refuse its users.

    Killed by: src/uclone_x/ui/server.py :: os.environ[UI_BIND_HOST_ENV_VAR] = host
    Becomes: pass
    """
    monkeypatch.setattr("uvicorn.run", MagicMock())
    identity = ui_process.process_identity(os.getpid())
    monkeypatch.setattr(ui_process, "process_identity", MagicMock(return_value=identity))
    monkeypatch.setattr("subprocess.Popen", MagicMock())
    monkeypatch.setattr(httpx, "get", answering_as(VITE_IDENTITY_MARKER))

    start_ui_server(port=5180, dev=True, host="0.0.0.0", vite_port=5173)

    assert os.environ.get(UI_BIND_HOST_ENV_VAR) == "0.0.0.0"


@pytest.mark.parametrize(
    ("bind_host", "loopback"),
    [
        ("127.0.0.1", True),
        ("127.0.0.2", True),
        ("localhost", True),
        ("::1", True),
        ("[::1]", True),
        ("0.0.0.0", False),
        ("::", False),
        ("192.168.1.20", False),
        ("my-desktop.lan", False),
    ],
)
def test_which_bind_addresses_are_loopback(bind_host: str, loopback: bool) -> None:
    """`0.0.0.0` and `::` listen on every interface, so they are an exposure, not loopback.

    Killed by: src/uclone_x/ui/app.py :: return ipaddress.ip_address(name).is_loopback
    Becomes: return not ipaddress.ip_address(name).is_unspecified
    """
    assert _is_loopback_bind(bind_host) is loopback


def test_a_page_opened_by_the_wrong_name_reads_the_reason_as_text(tmp_path: Path) -> None:
    """A person who typed a LAN name sees a sentence, not a JSON document.

    Killed by: src/uclone_x/ui/app.py :: if str(scope.get("path", "")).startswith("/api"):
    Becomes: if True:
    """
    client = _client(tmp_path)

    res = client.get("/", headers={"Host": "my-desktop.lan:5180"})

    assert res.status_code == 403
    assert res.headers["content-type"].startswith("text/plain")
    assert "http://localhost:5180" in res.text


def test_an_unparseable_host_is_refused_not_a_server_error(tmp_path: Path) -> None:
    """`[::1` makes `urlparse` raise; that must be a refusal, not a 500.

    Killed by: src/uclone_x/ui/app.py :: except ValueError:  # e.g. an unclosed `[::1`
    Becomes: except KeyError:  # e.g. an unclosed `[::1`
    """
    client = _client(tmp_path)

    assert client.get("/api/health", headers={"Host": "[::1"}).status_code == 403


def test_a_rebound_page_cannot_open_a_websocket(tmp_path: Path) -> None:
    """Streams are refused the same way: closed with a policy-violation code before accept.

    Killed by: src/uclone_x/ui/app.py :: await WebSocketClose(code=1008)(scope, receive, send)
    Becomes: await WebSocketClose(code=1000)(scope, receive, send)
    """
    client = _client(tmp_path)

    with pytest.raises(WebSocketDisconnect) as closed:
        with client.websocket_connect("/api/anything", headers=REBOUND):
            pass

    assert closed.value.code == 1008
