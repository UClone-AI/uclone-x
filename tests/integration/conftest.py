"""L2 Integration test environment guards and fixtures."""

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
    if getattr(socket, "AF_UNIX", None) is not None and self.family == socket.AF_UNIX:
        return _ORIG_SOCKET_CONNECT(self, address)  # pyright: ignore[reportUnknownArgumentType]

    if isinstance(address, tuple) and len(address) >= 2:  # pyright: ignore[reportUnknownArgumentType]
        host_obj = address[0]  # pyright: ignore[reportUnknownVariableType]
        port_obj = address[1]  # pyright: ignore[reportUnknownVariableType]
        host = str(host_obj)  # pyright: ignore[reportUnknownArgumentType]
        port = port_obj if isinstance(port_obj, int) else None
        if _is_loopback_address(host) and port in _BOUND_LOCAL_PORTS:
            return _ORIG_SOCKET_CONNECT(self, address)  # pyright: ignore[reportUnknownArgumentType]
        raise RuntimeError(
            f"Outbound network connection blocked in integration test: {host}:{port_obj}. "
            f"L2 Integration tests must not make external network calls (P8). "
            f"Use in-memory EventBus, MockLLMConnector, and local components."
        )

    return _ORIG_SOCKET_CONNECT(self, address)  # pyright: ignore[reportUnknownArgumentType]


socket.socket.bind = _recording_bind


@pytest.fixture(autouse=True)
def _enforce_no_outbound_network(  # pyright: ignore[reportUnusedFunction]
    request: pytest.FixtureRequest,
) -> Iterator[None]:
    """Block outbound socket connections for L2 integration tests unless allow_network."""
    if "allow_network" in request.keywords:
        yield
        return

    socket.socket.connect = _guarded_connect
    try:
        yield
    finally:
        socket.socket.connect = _ORIG_SOCKET_CONNECT
