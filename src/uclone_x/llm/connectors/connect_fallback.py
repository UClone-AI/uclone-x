"""Why a request to a provider never got an answer, and one try on the other address family.

#2424: one Composer turn on Gemini failed with a bare `ConnectError`. The machine's own log
showed why: the connection to Google was made over IPv6 in 21 ms, and then the path dropped
every packet for 27 s (13 retransmissions, 0 bytes back) until the kernel gave up
(`ETIMEDOUT`). The app logged only the class name. The owner ruled that what matters is why
it failed even once, and that a retry should almost never happen. So:

* **Each request records its own connection** through httpx's `trace` extension: the peer
  address and its family once the TCP connection is made, which phase failed (`connect_tcp`,
  `start_tls`), how long it took, and whether any response arrived. A failure is logged with
  all of it and with the exception chain (`describe_transport_cause`).
* **The connect and TLS phase is bounded on its own** (`CONNECT_PHASE_TIMEOUT_S`), apart from
  the read timeout, so a stalled path fails in seconds rather than half a minute.
* **One fallback, to the other family, and only for that stall.** When the TCP connection
  was made over one family and failed before any response, the request is sent once more
  over the other family (`local_address`). There is no retry loop: a failure before TCP
  connected (the OS already tried every address), a failure after the request body was
  sent or the response started, or a second failure is not tried again. Every fallback is counted (`family_fallbacks`) and
  logged at WARNING, so it can be seen to stay rare.
"""

from __future__ import annotations

import logging
import socket
import time
from collections import Counter
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, cast

import httpx

from uclone_x.llm.connectors.failures import describe_transport_cause

__all__ = [
    "CONNECT_PHASE_TIMEOUT_S",
    "family_fallbacks",
    "post_with_fallback",
    "stream_with_fallback",
]

_logger = logging.getLogger(__name__)

#: Seconds the TCP connect and the TLS handshake may take together, apart from the read
#: timeout (httpcore applies `connect` to both). The stall in #2424 lasted 27 s.
CONNECT_PHASE_TIMEOUT_S = 10.0

#: Raised before any response could arrive.
_BEFORE_RESPONSE: tuple[type[httpx.RequestError], ...] = (httpx.ConnectError, httpx.ConnectTimeout)

#: The address to bind to for each family, which makes the OS connect over that family.
_BIND: dict[socket.AddressFamily, str] = {socket.AF_INET: "0.0.0.0", socket.AF_INET6: "::"}
_NAME: dict[socket.AddressFamily, str] = {socket.AF_INET: "IPv4", socket.AF_INET6: "IPv6"}

#: Fallbacks made since this process started, by host.
_FALLBACKS: Counter[str] = Counter()


def family_fallbacks() -> dict[str, int]:
    """How many requests fell back to the other address family, by host, since start."""
    return dict(_FALLBACKS)


@dataclass
class _Attempt:
    """What one request's connection did, from httpx's `trace` events."""

    started: float = field(default_factory=time.monotonic)
    family: socket.AddressFamily | None = None
    peer: str | None = None
    phase: str | None = None
    body_sent: bool = False
    response_started: bool = False

    async def __call__(self, event: str, info: dict[str, Any]) -> None:
        name = event.removeprefix("connection.").removeprefix("http11.").removeprefix("http2.")
        if name.endswith(".started"):
            self.phase = name.removesuffix(".started")
        if name == "connect_tcp.complete":
            stream = info.get("return_value")
            sock: object = stream.get_extra_info("socket") if stream is not None else None
            found: object = stream.get_extra_info("server_addr") if stream is not None else None
            addr = cast("tuple[object, ...]", found) if isinstance(found, tuple) else ()
            host = str(addr[0]) if addr else ""
            if isinstance(sock, socket.socket):
                self.family = sock.family
            elif host:
                self.family = socket.AF_INET6 if ":" in host else socket.AF_INET
            if len(addr) >= 2:
                self.peer = f"[{host}]:{addr[1]}" if ":" in host else f"{host}:{addr[1]}"
        if name == "send_request_body.complete":
            self.body_sent = True
        if name.startswith("receive_response_headers.complete"):
            self.response_started = True

    def describe(self, exc: BaseException) -> str:
        where = f" during {self.phase}" if self.phase is not None else ""
        if self.family is None:
            over = "before any connection was made"
        else:
            over = f"over {_NAME.get(self.family, str(self.family))}"
            if self.peer is not None:
                over += f" {self.peer}"
        received = (
            "the response had started" if self.response_started else "no response bytes received"
        )
        elapsed = time.monotonic() - self.started
        return f"{describe_transport_cause(exc)}{where} after {elapsed:.2f}s, {over}; {received}"


def _bounded(kwargs: dict[str, Any]) -> dict[str, Any]:
    """`kwargs` with the connect/TLS phase bounded apart from the read timeout."""
    timeout = kwargs.get("timeout")
    if isinstance(timeout, (int, float)):
        kwargs["timeout"] = httpx.Timeout(
            float(timeout), connect=min(float(timeout), CONNECT_PHASE_TIMEOUT_S)
        )
    return kwargs


def _transport_for(family: socket.AddressFamily) -> httpx.AsyncBaseTransport:
    """A transport that connects over ``family`` only; a test replaces it."""
    return httpx.AsyncHTTPTransport(local_address=_BIND[family])


#: The family a stalled request falls back to.
_OTHER: dict[socket.AddressFamily, socket.AddressFamily] = {
    socket.AF_INET6: socket.AF_INET,
    socket.AF_INET: socket.AF_INET6,
}


def _other(family: socket.AddressFamily | None) -> socket.AddressFamily | None:
    return _OTHER.get(family) if family is not None else None


async def _sending(
    send: Callable[[httpx.AsyncClient, _Attempt], Awaitable[httpx.Response]],
    client: httpx.AsyncClient,
    url: str,
    *,
    keep_fallback_open: bool = False,
) -> httpx.Response:
    attempt = _Attempt()
    try:
        return await send(client, attempt)
    except _BEFORE_RESPONSE as exc:
        host = httpx.URL(url).host
        _logger.warning("Could not reach %s: %s", host, attempt.describe(exc))
        other = _other(attempt.family)
        # Sent once the body has gone out, whatever the error is called: the provider may
        # have acted on it.
        if other is None or attempt.body_sent or attempt.response_started:
            raise
        _FALLBACKS[host] += 1
        _logger.warning(
            "Trying %s once over %s instead of %s (fallback %d for this host since start)",
            host,
            _NAME[other],
            _NAME.get(attempt.family, "?") if attempt.family else "?",
            _FALLBACKS[host],
        )
        second = _Attempt()
        fallback = httpx.AsyncClient(transport=_transport_for(other))
        close = True  # a failed fallback is closed here; the caller closes a stream it reads
        try:
            response = await send(fallback, second)
            close = not keep_fallback_open
            return response
        except _BEFORE_RESPONSE as again:
            _logger.warning(
                "Could not reach %s over %s either: %s", host, _NAME[other], second.describe(again)
            )
            raise
        finally:
            if close:
                await fallback.aclose()


async def post_with_fallback(client: httpx.AsyncClient, url: str, **kwargs: Any) -> httpx.Response:
    """`client.post(url, **kwargs)`, recorded, bounded, and sent once over the other address
    family when the first stalled before any response (module doc)."""
    bounded = _bounded(kwargs)

    async def send(c: httpx.AsyncClient, attempt: _Attempt) -> httpx.Response:
        return await c.post(url, extensions={"trace": attempt}, **bounded)

    return await _sending(send, client, url)


@asynccontextmanager
async def stream_with_fallback(
    client: httpx.AsyncClient, method: str, url: str, **kwargs: Any
) -> AsyncGenerator[httpx.Response]:
    """`client.stream(method, url, **kwargs)`, with the same rule. Only the opening falls
    back: once the response has started, nothing is sent again."""
    bounded = _bounded(kwargs)
    clients: list[httpx.AsyncClient] = []

    async def send(c: httpx.AsyncClient, attempt: _Attempt) -> httpx.Response:
        clients.append(c)
        request = c.build_request(method, url, extensions={"trace": attempt}, **bounded)
        return await c.send(request, stream=True)

    # The fallback client stays open while the stream is read, and is closed below.
    response = await _sending(send, client, url, keep_fallback_open=True)
    try:
        yield response
    finally:
        await response.aclose()
        for opened in clients[1:]:
            await opened.aclose()
