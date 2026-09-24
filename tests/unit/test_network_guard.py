"""Unit tests for the unit gate network guard (conftest.py)."""

import socket

import pytest

from tests.unit.conftest import _BOUND_LOCAL_PORTS  # pyright: ignore[reportPrivateUsage]


def _find_unbound_port(start: int = 10000, end: int = 40000) -> int:
    """Find a port in the non-ephemeral range that is not in _BOUND_LOCAL_PORTS."""
    for port in range(start, end):
        if port not in _BOUND_LOCAL_PORTS:
            return port
    raise RuntimeError("No unbound port found in range")


def test_network_guard_blocks_unbound_connect() -> None:
    """An unmanaged socket connect in a unit test raises RuntimeError."""
    port = _find_unbound_port()
    assert port not in _BOUND_LOCAL_PORTS

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        with pytest.raises(RuntimeError, match="Outbound network connection blocked in unit test"):
            sock.connect(("127.0.0.1", port))
    finally:
        sock.close()


def test_network_guard_allows_bound_loopback_connect() -> None:
    """A socket connect to a port bound in this process succeeds."""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]

    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        client.connect(("127.0.0.1", port))
        assert client.fileno() > 0
    finally:
        client.close()
        server.close()


def test_network_guard_blocks_unbound_port_with_other_ports_bound() -> None:
    """Network guard strictly blocks connection attempts when port is not in _BOUND_LOCAL_PORTS."""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    bound_port = server.getsockname()[1]

    try:
        assert bound_port in _BOUND_LOCAL_PORTS

        unbound_port = _find_unbound_port()
        assert unbound_port != bound_port
        assert unbound_port not in _BOUND_LOCAL_PORTS

        # Connecting to unbound port must raise RuntimeError even though bound_port is registered
        client_unbound = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            with pytest.raises(
                RuntimeError, match="Outbound network connection blocked in unit test"
            ):
                client_unbound.connect(("127.0.0.1", unbound_port))
        finally:
            client_unbound.close()

        # Connecting to the bound port must succeed through the guard
        client_bound = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            client_bound.connect(("127.0.0.1", bound_port))
            assert client_bound.fileno() > 0
        finally:
            client_bound.close()
    finally:
        server.close()


@pytest.mark.parametrize(
    ("host", "port"),
    [
        ("8.8.8.8", 80),
        ("example.com", 443),
        ("1.1.1.1", 53),
        ("192.168.1.1", 8080),
    ],
)
def test_network_guard_blocks_external_outbound_addresses(host: str, port: int) -> None:
    """Non-loopback / external outbound addresses are unconditionally blocked."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        with pytest.raises(RuntimeError, match="Outbound network connection blocked in unit test"):
            sock.connect((host, port))
    finally:
        sock.close()


def test_network_guard_blocks_external_address_even_if_port_in_bound_ports() -> None:
    """External destination is blocked even if port matches a bound local port."""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    bound_port = server.getsockname()[1]
    try:
        assert bound_port in _BOUND_LOCAL_PORTS
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            with pytest.raises(
                RuntimeError, match="Outbound network connection blocked in unit test"
            ):
                sock.connect(("8.8.8.8", bound_port))
        finally:
            sock.close()
    finally:
        server.close()


@pytest.mark.allow_network
def test_network_guard_allow_network_marker_bypasses() -> None:
    """The allow_network marker bypasses the guard."""
    port = _find_unbound_port()
    assert port not in _BOUND_LOCAL_PORTS

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        # Connecting to a closed port will raise ConnectionRefusedError or OSError from the OS, not RuntimeError
        with pytest.raises((ConnectionRefusedError, OSError)) as excinfo:
            sock.connect(("127.0.0.1", port))
        assert not isinstance(excinfo.value, RuntimeError)
    finally:
        sock.close()
