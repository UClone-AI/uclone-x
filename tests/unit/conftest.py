"""Unit test environment guards and fixtures."""

from __future__ import annotations

import socket
from collections.abc import Iterator
from typing import Any

import pytest

_BOUND_LOCAL_PORTS: set[int] = set()
_ORIG_SOCKET_BIND = socket.socket.bind
_ORIG_SOCKET_CONNECT = socket.socket.connect


def _is_loopback_address(host: str) -> bool:
    return host in {"127.0.0.1", "localhost", "::1", "0.0.0.0", "testserver"}


def _recording_bind(self: socket.socket, address: Any) -> Any:  # pyright: ignore[reportUnknownParameterType]
    # Record bound local ports
    res = _ORIG_SOCKET_BIND(self, address)  # pyright: ignore[reportUnknownArgumentType]
    try:
        sock_name = self.getsockname()
        if isinstance(sock_name, tuple) and len(sock_name) >= 2:  # pyright: ignore[reportUnknownArgumentType]
            port = sock_name[1]  # pyright: ignore[reportUnknownVariableType]
            if isinstance(port, int) and port > 0:
                _BOUND_LOCAL_PORTS.add(port)
    except Exception:
        pass
    return res


def _guarded_connect(self: socket.socket, address: Any) -> Any:  # pyright: ignore[reportUnknownParameterType]
    # Allow UNIX sockets
    if getattr(socket, "AF_UNIX", None) is not None and self.family == socket.AF_UNIX:
        return _ORIG_SOCKET_CONNECT(self, address)  # pyright: ignore[reportUnknownArgumentType]

    # Address is usually (host, port)
    if isinstance(address, tuple) and len(address) >= 2:  # pyright: ignore[reportUnknownArgumentType]
        host_obj = address[0]  # pyright: ignore[reportUnknownVariableType]
        port_obj = address[1]  # pyright: ignore[reportUnknownVariableType]
        host = str(host_obj)  # pyright: ignore[reportUnknownArgumentType]
        port = port_obj if isinstance(port_obj, int) else None
        if _is_loopback_address(host) and port in _BOUND_LOCAL_PORTS:
            return _ORIG_SOCKET_CONNECT(self, address)  # pyright: ignore[reportUnknownArgumentType]
        raise RuntimeError(
            f"Outbound network connection blocked in unit test: {host}:{port_obj}. "
            f"Unit tests must not make external network calls or connect to unmanaged ports (P8). "
            f"Use httpx.MockTransport, MockLLMConnector, or bind an in-process ephemeral test server."
        )

    return _ORIG_SOCKET_CONNECT(self, address)  # pyright: ignore[reportUnknownArgumentType]


# Install recording bind unconditionally on conftest import
socket.socket.bind = _recording_bind


@pytest.fixture(autouse=True)
def _enforce_no_outbound_network(  # pyright: ignore[reportUnusedFunction]
    request: pytest.FixtureRequest,
) -> Iterator[None]:
    """Block outbound socket connections for unit tests unless marked with allow_network."""
    if "allow_network" in request.keywords:
        yield
        return

    socket.socket.connect = _guarded_connect
    try:
        yield
    finally:
        socket.socket.connect = _ORIG_SOCKET_CONNECT


def _no_frontend_build() -> None:
    """Stand in for `uclone_x.ui.server._ensure_frontend_built`: the bundle is committed."""


@pytest.fixture
def frontend_build_suppressed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Required by every test that calls the real `start_ui_server` (#1067).

    `start_ui_server` calls `_ensure_frontend_built`, which runs a real `npm run build`
    whenever the committed `src/uclone_x/ui_static` is not what `frontend/` builds — so a
    test that does not replace it rewrites tracked files and races every other worker
    reading the bundle. Since #1075 that question is a digest over the build inputs rather
    than an mtime comparison, so it is `False` on a pristine checkout; the fixture stays
    because a test must not depend on the checkout being pristine. `tests/unit/test_cli_ui_stop.py` had been doing this by hand, in
    its subprocess dashboard, since #927; this is that same substitution in one place, so
    the next test to reach for the real launcher inherits the answer (R12).

    **Named at each call site rather than autouse**, because the dependency is real: a
    reader of `test_start_ui_server_production` should see that the launcher builds a
    frontend and that this test does not. Forgetting it is not silent — the session-wide
    guard in `tests/support/frontend_build_guard.py` turns the omission into a failure of
    that test instead of a rewritten bundle.

    Imported inside the fixture so that only the tests requesting it pay for importing
    `uvicorn` and the FastAPI application.
    """
    from uclone_x.ui import server

    monkeypatch.setattr(server, "_ensure_frontend_built", _no_frontend_build)
