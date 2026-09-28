"""One live connection from this machine to uClone2, for one link.

`LinkSession` dials the link's `ws_url`, answers the server's heartbeat, reconnects after a
drop, and stops for good when the server says the link is over. It is the WebSocket half of
the jointly owned Linked Runtime contract (uClone2 `docs/contracts/linked-runtime/`,
`frames.schema.json`); `client.py` is the REST half.

The session runs no tasks yet -- the executor is a later change. It tells the server so
with `ready{task_types: []}`, and answers any offer that still arrives with a retryable
`task.fail`, so nothing offered is dropped without a word.

Two kinds of text leave this module. `STATE_TEXT` is what the user reads: plain sentences,
no close code, no frame name. `LinkSession.diagnostic` is for the Diagnostics tab and may
carry protocol detail. Neither ever holds the token, and neither does a log line: the socket
library logs every request header at DEBUG, so it is given a logger that cannot log below
INFO (`_WireLogger`).
"""

from __future__ import annotations

import asyncio
import contextlib
import ipaddress
import json
import logging
import random
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum, StrEnum
from typing import Any, Final
from urllib.parse import urlsplit

from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import ConnectionClosed, InvalidHandshake, InvalidStatus, InvalidURI

from uclone_x import __version__
from uclone_x.link.uclone2.client import MESSAGES, LinkError, LinkFailure, Uclone2LinkClient
from uclone_x.link.uclone2.frames import (
    PROTOCOL,
    Bye,
    ErrorFrame,
    FrameError,
    Hello,
    Ping,
    TaskCancel,
    TaskOffer,
    ack_frame,
    bye_logout_frame,
    error_frame,
    fail_frame,
    parse_frame,
    pong_frame,
    ready_frame,
)
from uclone_x.link.uclone2.models import LinkRecord
from uclone_x.link.uclone2.service import refresh_profile
from uclone_x.link.uclone2.store import LinkStore

__all__ = [
    "STATE_TEXT",
    "LinkSession",
    "LinkSessionState",
    "SessionTiming",
    "backoff_delay",
]

logger = logging.getLogger(__name__)


class LinkSessionState(StrEnum):
    """Where one link's session is. Each maps to exactly one sentence in `STATE_TEXT`."""

    #: Dialling, and uClone2 has not seen this runtime within the presence grace.
    CONNECTING = "connecting"
    ONLINE = "online"
    #: The socket dropped and the session is dialling again inside the server's presence
    #: grace, so visitors in uClone2 still see the clone online.
    RECONNECTING = "reconnecting"
    #: The user switched the clone offline; no session runs until they switch it back.
    PAUSED = "paused"
    #: The dashboard stopped the session on its way out.
    OFFLINE = "offline"
    #: Close 4409: another runtime took this clone. Not retried, which would fight it.
    REPLACED = "replaced"
    #: HTTP 401 on dial or close 4401: the token is revoked. The record is disabled.
    ENDED = "ended"
    #: HTTP 503 on dial or close 4503: uClone2's kill switch. Retried at the backoff cap.
    SERVER_PAUSED = "server_paused"
    #: HTTP 426 on dial: this runtime's protocol is no longer served.
    UPDATE_REQUIRED = "update_required"
    #: Close 4400: the server found a frame from this runtime it could not use.
    PROTOCOL_ERROR = "protocol_error"


#: What the Settings section says for each state: plain sentences a user can act on.
STATE_TEXT: Final[Mapping[LinkSessionState, str]] = {
    LinkSessionState.CONNECTING: "연결 중…",
    LinkSessionState.ONLINE: "온라인 — uClone2에서 활동 중",
    LinkSessionState.RECONNECTING: "다시 연결하는 중… (uClone2에는 잠시 온라인으로 보입니다)",
    LinkSessionState.PAUSED: "오프라인으로 전환함",
    LinkSessionState.OFFLINE: "오프라인 — UClone-X가 실행되면 다시 연결됩니다",
    LinkSessionState.REPLACED: "이 클론은 다른 기기에서 연결되었습니다",
    LinkSessionState.ENDED: MESSAGES[LinkFailure.TOKEN_INVALID],
    LinkSessionState.SERVER_PAUSED: "uClone2가 연결을 잠시 멈췄습니다",
    LinkSessionState.UPDATE_REQUIRED: "UClone-X를 업데이트해야 연결할 수 있습니다",
    LinkSessionState.PROTOCOL_ERROR: "연결에 문제가 생겼습니다. UClone-X를 업데이트하십시오",
}

#: States the session does not leave on its own. Everything else is on its way to ONLINE.
TERMINAL_STATES: Final = frozenset(
    {
        LinkSessionState.PAUSED,
        LinkSessionState.OFFLINE,
        LinkSessionState.REPLACED,
        LinkSessionState.ENDED,
        LinkSessionState.UPDATE_REQUIRED,
        LinkSessionState.PROTOCOL_ERROR,
    }
)


@dataclass(frozen=True)
class SessionTiming:
    """The contract's clocks. Tests shrink them; nothing else should."""

    #: The server pings every 20 s and drops a socket silent for 60 s; the runtime treats
    #: its own 60 s without a frame the same way.
    heartbeat_timeout_s: float = 60.0
    backoff_base_s: float = 1.0
    backoff_cap_s: float = 300.0
    #: `LINKED_PRESENCE_GRACE`: how long uClone2 keeps showing a dropped clone online.
    presence_grace_s: float = 90.0
    #: How long a deliberate stop waits for `bye{logout}` and the close to be written.
    logout_wait_s: float = 2.0
    open_timeout_s: float = 15.0


class _Next(Enum):
    """What the run loop does after one socket ends."""

    BACKOFF = "backoff"
    AT_CAP = "at_cap"
    STOP = "stop"


@dataclass(frozen=True)
class _Ending:
    next: _Next
    state: LinkSessionState | None = None
    disable: bool = False


#: HTTP statuses on dial, per the contract's handshake refusals.
_DIAL_REFUSALS: Final[Mapping[int, _Ending]] = {
    401: _Ending(_Next.STOP, LinkSessionState.ENDED, disable=True),
    426: _Ending(_Next.STOP, LinkSessionState.UPDATE_REQUIRED),
    503: _Ending(_Next.AT_CAP, LinkSessionState.SERVER_PAUSED),
}

#: Server closes. Each is preceded by a `bye` whose reason names the same case; the close
#: code wins when both arrive, and the bye stands in when the close frame is lost.
_CLOSES: Final[Mapping[int, _Ending]] = {
    4401: _Ending(_Next.STOP, LinkSessionState.ENDED, disable=True),
    4409: _Ending(_Next.STOP, LinkSessionState.REPLACED),
    4503: _Ending(_Next.AT_CAP, LinkSessionState.SERVER_PAUSED),
    4400: _Ending(_Next.STOP, LinkSessionState.PROTOCOL_ERROR),
    1001: _Ending(_Next.BACKOFF),
}
_BYE_CLOSE_CODES: Final[Mapping[str, int]] = {
    "revoked": 4401,
    "replaced": 4409,
    "kill_switch": 4503,
    "protocol_error": 4400,
    "shutdown": 1001,
}
_DROP: Final = _Ending(_Next.BACKOFF)

#: The owner reads this in uClone2 when an offer reaches a runtime that cannot run it.
_CANNOT_RUN = "이 클론의 UClone-X가 아직 이 작업을 처리할 수 없습니다"

_USER_AGENT: Final = f"uclone-x/{__version__}"
#: The contract bounds a frame's context; 1 MiB is far above any offer it describes.
_MAX_FRAME_BYTES: Final = 1 << 20
#: Past this many doublings every base delay is far above any cap; `2**attempt` stays small.
_MAX_BACKOFF_EXPONENT: Final = 30


class _WireLogger(logging.LoggerAdapter[logging.Logger]):
    """The socket library's logger, pinned above DEBUG.

    `websockets` logs each handshake header at DEBUG, the `Authorization` header included.
    A level on a named logger can be lowered by anyone configuring logging; this adapter
    cannot be, so the bearer never reaches a handler whatever the configuration.
    """

    def isEnabledFor(self, level: int) -> bool:
        return level > logging.DEBUG and self.logger.isEnabledFor(level)

    def log(self, level: int, msg: object, *args: object, **kwargs: Any) -> None:
        if level > logging.DEBUG:
            super().log(level, msg, *args, **kwargs)

    def debug(self, msg: object, *args: object, **kwargs: Any) -> None:
        return None


_WIRE_LOG: Final = _WireLogger(logging.getLogger("uclone_x.link.uclone2.wire"))


def _is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _dial_url(ws_url: str) -> tuple[str, bool] | None:
    """`ws_url?protocol=1`, and whether it is a loopback host; `None` if it may not be dialled.

    `wss` only, except `ws` to this machine -- the same rule `client._origin` applies to the
    REST origin, because the bearer rides on this handshake too.
    """
    try:
        parts = urlsplit(ws_url)
        host = parts.hostname or ""
    except ValueError:
        return None
    loopback = _is_loopback(host)
    if not host or parts.scheme not in {"wss", "ws"} or (parts.scheme == "ws" and not loopback):
        return None
    joiner = "&" if parts.query else "?"
    return f"{ws_url}{joiner}protocol={PROTOCOL}", loopback


def backoff_delay(attempt: int, timing: SessionTiming, rng: random.Random) -> float:
    """Exponential from `backoff_base_s` to the cap, with equal jitter.

    The exponent is clamped: `attempt` resets only on `ready`, so a dashboard left offline
    for days counts past the point where a float can hold `2**attempt`.
    """
    doublings = 2 ** min(attempt, _MAX_BACKOFF_EXPONENT)
    ceiling = min(timing.backoff_cap_s, timing.backoff_base_s * doublings)
    return ceiling / 2 + rng.uniform(0, ceiling / 2)


class LinkSession:
    """The socket for one `LinkRecord`, kept up until `stop()` or the server ends the link.

    Conforms to `engine.mattermost_bridge.TransportProtocol` (`send_payload`,
    `is_connected`), which is how later steps hand it outcomes to send.
    """

    def __init__(
        self,
        record: LinkRecord,
        *,
        store: LinkStore,
        client: Uclone2LinkClient,
        timing: SessionTiming | None = None,
        rng: random.Random | None = None,
        on_state: Callable[[LinkSessionState], None] | None = None,
    ) -> None:
        self._record = record
        self._store = store
        self._client = client
        self._timing = timing if timing is not None else SessionTiming()
        self._rng = rng if rng is not None else random.Random()
        self._on_state = on_state
        self._state = LinkSessionState.CONNECTING
        self._diagnostic = ""
        self._limits: dict[str, int] = {}
        self._profile_changed = False
        self._ws: ClientConnection | None = None
        self._ready = False
        #: The current socket reached `ready`; a later drop starts the backoff over.
        self._ready_seen = False
        #: The server's `bye.reason` on the current socket; its close follows.
        self._bye_reason: str | None = None
        #: The server's last `error` on the current socket; a 4400 close reports it.
        self._server_error: str | None = None
        self._stopping = False
        self._wake = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._refresh_task: asyncio.Task[None] | None = None
        #: `time.monotonic()` of the last frame from the server on a ready socket.
        self._last_contact: float | None = None

    def __repr__(self) -> str:
        # The record's own repr masks the token; this names the link and nothing more.
        return f"LinkSession(link_id={self._record.link_id!r}, state={self._state.value!r})"

    # --- what heads read --------------------------------------------------------------

    @property
    def link_id(self) -> str:
        return self._record.link_id

    @property
    def state(self) -> LinkSessionState:
        return self._state

    @property
    def state_text(self) -> str:
        return STATE_TEXT[self._state]

    @property
    def diagnostic(self) -> str:
        """The last protocol event, for the Diagnostics tab: a close code, a status, a frame."""
        return self._diagnostic

    @property
    def limits(self) -> dict[str, int]:
        """`hello.limits` from the current socket: field name -> max code points."""
        return dict(self._limits)

    @property
    def profile_changed(self) -> bool:
        """uClone2's persona or avatar changed since the record last read it."""
        return self._profile_changed

    @property
    def is_connected(self) -> bool:
        return self._ws is not None and self._ready

    async def send_payload(self, payload: dict[str, Any]) -> None:
        """Send one frame on the ready socket; raises `ConnectionError` if there is none."""
        ws = self._ws
        if ws is None or not self._ready:
            raise ConnectionError("the uClone2 link is not connected")
        await ws.send(json.dumps(payload, ensure_ascii=False))

    def listen(self, on_state: Callable[[LinkSessionState], None]) -> None:
        """Be told each state change from now on (replaces the constructor's `on_state`)."""
        self._on_state = on_state

    # --- lifecycle --------------------------------------------------------------------

    def start(self) -> None:
        """Begin dialling in the background. Calling it twice does nothing."""
        if self._task is None:
            self._task = asyncio.create_task(
                self._run(), name=f"uclone2-link-{self._record.link_id}"
            )

    async def wait(self) -> None:
        """Until the session has stopped for good (a terminal state)."""
        if self._task is not None:
            await asyncio.shield(self._task)

    async def stop(self, *, logout: bool, final: LinkSessionState) -> None:
        """End the session, sending `bye{logout}` first when `logout` is true.

        `logout` is for the deliberate stops -- the user going offline, the dashboard
        closing -- after which uClone2 shows the clone offline at once. It waits at most
        `logout_wait_s` for the bye and the close to be written, then gives up on them.
        """
        self._stopping = True
        self._wake.set()
        ws = self._ws
        if ws is not None:
            with contextlib.suppress(Exception):
                async with asyncio.timeout(self._timing.logout_wait_s):
                    if logout:
                        await ws.send(json.dumps(bye_logout_frame()))
                    await ws.close()
            ws.transport.abort()
        for task in (self._task, self._refresh_task):
            if task is not None and not task.done():
                task.cancel()
                with contextlib.suppress(BaseException):
                    await task
        self._ws = None
        self._ready = False
        self._set_state(final)

    # --- the loop ---------------------------------------------------------------------

    def _set_state(self, state: LinkSessionState) -> None:
        if state is self._state:
            return
        self._state = state
        logger.info("uClone2 link %s is %s", self._record.link_id, state.value)
        if self._on_state is not None:
            try:
                self._on_state(state)
            except Exception:
                logger.exception("uClone2 link state listener failed")

    def _persist(self, **changes: Any) -> None:
        """Write to the link record; a store that cannot be written does not end the socket."""
        try:
            self._store.update(self._record.link_id, **changes)
        except Exception as err:
            self._note(f"link record not updated: {type(err).__name__}")

    def _note(self, diagnostic: str) -> None:
        self._diagnostic = diagnostic
        logger.info("uClone2 link %s: %s", self._record.link_id, diagnostic)

    def _backoff(self, attempt: int) -> float:
        return backoff_delay(attempt, self._timing, self._rng)

    def _at_cap(self) -> float:
        cap = self._timing.backoff_cap_s
        return cap / 2 + self._rng.uniform(0, cap / 2)

    def _waiting_state(self) -> LinkSessionState:
        """Between sockets: still online to visitors inside the grace, otherwise connecting."""
        if (
            self._last_contact is not None
            and time.monotonic() - self._last_contact < self._timing.presence_grace_s
        ):
            return LinkSessionState.RECONNECTING
        return LinkSessionState.CONNECTING

    async def _run(self) -> None:
        attempt = 0
        while not self._stopping:
            if self._state is not LinkSessionState.SERVER_PAUSED:
                self._set_state(self._waiting_state())
            ending = await self._one_socket()
            if self._stopping:
                return
            if ending.disable:
                self._persist(enabled=False)
            if ending.state is not None:
                self._set_state(ending.state)
            if ending.next is _Next.STOP:
                return
            if self._ready_seen:
                attempt = 0
            if ending.next is _Next.AT_CAP:
                delay = self._at_cap()
            else:
                delay = self._backoff(attempt)
                attempt += 1
                self._set_state(self._waiting_state())
            with contextlib.suppress(TimeoutError):
                async with asyncio.timeout(delay):
                    await self._wake.wait()

    async def _one_socket(self) -> _Ending:
        """Dial once and serve the socket until it ends; how it ended."""
        self._ready_seen = False
        dial = _dial_url(self._record.ws_url)
        if dial is None:
            self._note("ws_url is neither wss nor ws to loopback")
            return _Ending(_Next.STOP, LinkSessionState.PROTOCOL_ERROR)
        url, loopback = dial
        # The one place the token leaves this process on the socket: a header, never the URL.
        headers = {"Authorization": f"Bearer {self._record.token.get_secret_value()}"}
        try:
            async with connect(
                url,
                additional_headers=headers,
                user_agent_header=_USER_AGENT,
                # A proxy is never right for this machine's own loopback server; the system
                # proxy is honoured for the real one.
                proxy=None if loopback else True,
                open_timeout=self._timing.open_timeout_s,
                # The server drives the heartbeat with its own `ping` frames.
                ping_interval=None,
                close_timeout=self._timing.logout_wait_s,
                max_size=_MAX_FRAME_BYTES,
                logger=_WIRE_LOG,
            ) as ws:
                self._ws = ws
                try:
                    return await self._serve(ws)
                finally:
                    self._ws = None
                    self._ready = False
        except InvalidStatus as err:
            status = err.response.status_code
            self._note(f"HTTP {status} on dial")
            return _DIAL_REFUSALS.get(status, _DROP)
        except ConnectionClosed as err:
            return self._closed(err)
        except (OSError, TimeoutError, InvalidHandshake, InvalidURI) as err:
            # The class name only: a library message can quote the URL.
            self._note(f"dial failed: {type(err).__name__}")
            return _DROP
        except Exception as err:
            # Anything else -- a proxy the library cannot use, a SOCKS proxy with its extra
            # missing -- must not end the loop, or the session sits in its last state for
            # good with nothing to say why. The class name only, as above.
            self._note(f"dial failed unexpectedly: {type(err).__name__}")
            return _DROP

    def _closed(self, err: ConnectionClosed) -> _Ending:
        code = err.rcvd.code if err.rcvd is not None else None
        bye = self._bye_reason
        if code is not None and code in _CLOSES:
            after = f" after {self._server_error}" if code == 4400 and self._server_error else ""
            self._note(f"closed {code}" + (f" after bye {bye}" if bye else "") + after)
            return _CLOSES[code]
        if bye is not None and bye in _BYE_CLOSE_CODES:
            self._note(f"bye {bye}, closed {code if code is not None else 'without a close frame'}")
            return _CLOSES[_BYE_CLOSE_CODES[bye]]
        self._note(f"closed {code}" if code is not None else "socket dropped")
        return _DROP

    async def _serve(self, ws: ClientConnection) -> _Ending:
        self._bye_reason = None
        self._server_error = None
        while True:
            try:
                async with asyncio.timeout(self._timing.heartbeat_timeout_s):
                    raw = await ws.recv()
            except TimeoutError:
                # The server pings every 20 s; a minute of silence is a dead socket that
                # the OS has not noticed. Abort rather than close: the peer is not there to
                # answer a close handshake, and no bye goes out for a drop.
                self._note(f"no frame for {self._timing.heartbeat_timeout_s:g} s")
                ws.transport.abort()
                return _DROP
            except ConnectionClosed as err:
                return self._closed(err)
            if self._ready:
                self._last_contact = time.monotonic()
            await self._handle(ws, raw)

    async def _send(self, ws: ClientConnection, frame: Mapping[str, object]) -> None:
        with contextlib.suppress(ConnectionClosed):
            await ws.send(json.dumps(frame, ensure_ascii=False))

    async def _handle(self, ws: ClientConnection, raw: str | bytes) -> None:
        try:
            frame = parse_frame(raw)
        except FrameError as err:
            self._note(f"sent error {err.code}: {err.detail}")
            await self._send(ws, error_frame(err.code, err.detail))
            return
        if isinstance(frame, Ping):
            await self._send(ws, pong_frame(frame.ts))
        elif isinstance(frame, Hello):
            await self._on_hello(ws, frame)
        elif isinstance(frame, Bye):
            # The close follows; remember why, in case its frame is lost.
            self._bye_reason = frame.reason
        elif isinstance(frame, ErrorFrame):
            self._server_error = f"server error {frame.code}: {frame.message}"
            self._note(self._server_error)
        elif isinstance(frame, TaskOffer):
            await self._decline(ws, frame)
        elif isinstance(frame, TaskCancel):
            self._note(f"task.cancel {frame.reason} for a task this runtime was not running")

    async def _on_hello(self, ws: ClientConnection, hello: Hello) -> None:
        if self._ready:
            self._note("second hello on one socket ignored")
            return
        self._limits = hello.length_limits()
        self._maybe_refresh(hello.clone_updated_at)
        await self._send(ws, ready_frame(max_concurrency=1, task_types=[]))
        self._ready = True
        self._ready_seen = True
        self._last_contact = time.monotonic()
        self._persist(last_connected_at=datetime.now(UTC))
        self._set_state(LinkSessionState.ONLINE)
        self._note(f"online, {hello.pending} pending")

    async def _decline(self, ws: ClientConnection, offer: TaskOffer) -> None:
        """An offer this runtime cannot run. `ready` listed no task types, so none should come.

        The contract has no code for an unlisted type; `runtime_error` is its catch-all, and
        retryable, so the server offers the task again rather than counting it as failed.
        """
        self._note(f"declined task.offer {offer.task_type} {offer.delivery_id}")
        await self._send(ws, ack_frame(offer.delivery_id))
        await self._send(
            ws,
            fail_frame(offer.delivery_id, code="runtime_error", retryable=True, reason=_CANNOT_RUN),
        )

    def _maybe_refresh(self, clone_updated_at: datetime) -> None:
        known = self._record.clone_updated_at
        if known is not None and clone_updated_at <= known:
            return
        if self._refresh_task is not None and not self._refresh_task.done():
            return
        self._refresh_task = asyncio.create_task(self._refresh())

    async def _refresh(self) -> None:
        try:
            result = await refresh_profile(
                self._record.link_id, store=self._store, client=self._client
            )
        except LinkError as err:
            self._note(f"profile refresh failed: {err.diagnostic or err.failure.value}")
            return
        except Exception as err:
            self._note(f"profile refresh failed: {type(err).__name__}")
            return
        self._record = result.record
        self._profile_changed = self._profile_changed or result.profile_changed
