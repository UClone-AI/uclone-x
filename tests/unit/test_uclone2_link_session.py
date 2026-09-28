"""The uClone2 link session: handshake, heartbeat, reconnect, close codes, logout vs. drop.

Every test dials a real socket on 127.0.0.1 (`FakeLinkedSocket`), so what is checked is
what goes over the wire. The contract's clocks are shrunk through `SessionTiming`; the
logic that reads them is the shipped one.
"""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import AsyncIterator
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import SecretStr

from tests.support.uclone2_fake import (
    BOT_ID,
    DELETE_SELF,
    EDITED_CLONE,
    GET_SELF,
    ORIGIN,
    TOKEN,
    FakeUclone2,
    internals_in,
    self_body,
)
from tests.support.uclone2_ws_fake import (
    HELLO_LIMITS,
    WS_PATH,
    FakeLinkedSocket,
    eventually,
    hello_frame,
)
from uclone_x.link.uclone2.client import Uclone2LinkClient
from uclone_x.link.uclone2.models import LinkRecord
from uclone_x.link.uclone2.session import (
    STATE_TEXT,
    LinkSession,
    LinkSessionState,
    SessionTiming,
    backoff_delay,
)
from uclone_x.link.uclone2.store import LinkStore, LinkStoreError
from uclone_x.link.uclone2.supervisor import LinkSupervisor

State = LinkSessionState

#: Fast clocks: a missed heartbeat in 0.3 s, backoff capped at 50 ms.
FAST = SessionTiming(
    heartbeat_timeout_s=0.3,
    backoff_base_s=0.01,
    backoff_cap_s=0.05,
    presence_grace_s=90.0,
    logout_wait_s=1.0,
    open_timeout_s=2.0,
)

#: Long enough for several backoff rounds at `FAST`: a stopped session would have redialled.
SEVERAL_ROUNDS_S = 0.4


def _record(ws_url: str, **changes: object) -> LinkRecord:
    base = LinkRecord(
        link_id="lnk_1",
        server_url=ORIGIN,
        bot_id=BOT_ID,
        remote_username="haru",
        remote_display_name="Haru",
        local_agent_id="haru",
        token=SecretStr(TOKEN),
        ws_url=ws_url,
        created_at=datetime(2026, 9, 27, 9, 0, tzinfo=UTC),
        clone_updated_at=datetime(2026, 9, 26, 22, 40, tzinfo=UTC),
    )
    return base.model_copy(update=changes)


@pytest.fixture
def store(tmp_path: Path) -> LinkStore:
    return LinkStore(tmp_path / "links" / "uclone2.json")


@pytest.fixture
def rest() -> FakeUclone2:
    return FakeUclone2()


@pytest.fixture
async def server() -> AsyncIterator[FakeLinkedSocket]:
    async with FakeLinkedSocket() as fake:
        yield fake


class Harness:
    def __init__(self, store: LinkStore, rest: FakeUclone2) -> None:
        self.store = store
        self.rest = rest
        self.sessions: list[LinkSession] = []
        self.seen: list[LinkSessionState] = []

    def session(self, record: LinkRecord, timing: SessionTiming = FAST) -> LinkSession:
        self.store.put(record)
        session = LinkSession(
            record,
            store=self.store,
            client=Uclone2LinkClient(transport=self.rest.transport),
            timing=timing,
            rng=random.Random(7),
            on_state=self.seen.append,
        )
        self.sessions.append(session)
        session.start()
        return session


@pytest.fixture
async def harness(store: LinkStore, rest: FakeUclone2) -> AsyncIterator[Harness]:
    h = Harness(store, rest)
    yield h
    for s in h.sessions:
        await s.stop(logout=False, final=State.OFFLINE)


async def _online(
    harness: Harness, server: FakeLinkedSocket, timing: SessionTiming = FAST, **changes: object
) -> LinkSession:
    session = harness.session(_record(server.ws_url, **changes), timing)
    await eventually(lambda: session.state is State.ONLINE, what="the session to go online")
    return session


# --- the handshake --------------------------------------------------------------------


async def test_dials_the_link_ws_url_with_protocol_1_and_the_bearer_header(
    harness: Harness, server: FakeLinkedSocket
) -> None:
    await _online(harness, server)

    conn = server.latest
    assert conn.path == f"{WS_PATH}?protocol=1"
    assert conn.headers["Authorization"] == f"Bearer {TOKEN}"
    assert conn.headers["User-Agent"].startswith("uclone-x/")
    assert TOKEN not in conn.path


async def test_hello_is_answered_with_ready_and_the_link_reads_online(
    harness: Harness, server: FakeLinkedSocket, store: LinkStore
) -> None:
    session = await _online(harness, server)

    await eventually(lambda: bool(server.latest.frames("ready")), what="ready")
    assert server.latest.frames()[0] == {
        "type": "ready",
        "protocol": 1,
        "max_concurrency": 1,
        "task_types": [],
    }
    assert session.is_connected
    assert session.limits == HELLO_LIMITS
    stored = store.get("lnk_1")
    assert stored is not None and stored.last_connected_at is not None


async def test_a_ping_is_answered_with_a_pong_echoing_its_ts(
    harness: Harness, server: FakeLinkedSocket
) -> None:
    await _online(harness, server)
    await server.latest.send({"type": "ping", "ts": "2026-09-27T09:00:20Z"})

    await eventually(lambda: bool(server.latest.frames("pong")), what="a pong")
    assert server.latest.frames("pong") == [{"type": "pong", "ts": "2026-09-27T09:00:20Z"}]


async def test_a_newer_clone_updated_at_refreshes_the_profile(
    harness: Harness, rest: FakeUclone2, store: LinkStore
) -> None:
    rest.self_answer = self_body(clone=EDITED_CLONE, updated_at="2026-09-27T08:00:00Z")
    async with FakeLinkedSocket(hello=hello_frame(clone_updated_at="2026-09-27T08:00:00Z")) as srv:
        session = await _online(harness, srv)

        await eventually(lambda: session.profile_changed, what="the profile refresh")
    assert len(rest.calls(GET_SELF)) == 1
    stored = store.get("lnk_1")
    assert stored is not None and stored.remote_display_name == "Haru (봄)"


async def test_an_unchanged_clone_updated_at_reads_nothing(
    harness: Harness, rest: FakeUclone2, server: FakeLinkedSocket
) -> None:
    session = await _online(harness, server)
    await asyncio.sleep(0.05)

    assert rest.calls(GET_SELF) == []
    assert not session.profile_changed


async def test_a_hello_without_the_required_limit_is_not_readied(
    harness: Harness,
) -> None:
    hello = hello_frame(limits={"post_title_max_chars": 80})
    async with FakeLinkedSocket(hello=hello) as srv:
        harness.session(_record(srv.ws_url))
        await eventually(lambda: bool(srv.connections and srv.latest.frames("error")), what="error")

        assert srv.latest.frames("ready") == []
        assert srv.latest.frames("error")[0]["code"] == "invalid_frame"


# --- frames the session does not run --------------------------------------------------


async def test_an_offer_is_acked_then_failed_retryably(
    harness: Harness, server: FakeLinkedSocket
) -> None:
    await _online(harness, server)
    await server.latest.send(
        {
            "type": "task.offer",
            "delivery_id": "dlv_1",
            "task_id": "tsk_1",
            "task_type": "hompy_reply",
            "obligation": True,
            "attempt": 1,
            "context": {},
            "subject_url": "https://staging.uclone.test/haru",
        }
    )

    await eventually(lambda: bool(server.latest.frames("task.fail")), what="task.fail")
    sent = [f for f in server.latest.frames() if f["type"] in {"task.ack", "task.fail"}]
    assert [f["type"] for f in sent] == ["task.ack", "task.fail"]
    fail = sent[1]
    assert fail["delivery_id"] == "dlv_1"
    assert fail["code"] == "runtime_error"
    assert fail["retryable"] is True
    assert internals_in(str(fail["reason"])) == []


@pytest.mark.parametrize(
    ("frame", "code"),
    [
        ('{"type": "square.gossip"}', "unknown_type"),
        ("not json at all", "invalid_frame"),
        ('{"type": "ping"}', "invalid_frame"),
    ],
)
async def test_an_unusable_frame_gets_an_error_and_the_socket_stays(
    harness: Harness, server: FakeLinkedSocket, frame: str, code: str
) -> None:
    session = await _online(harness, server)
    await server.latest.send(frame)

    await eventually(lambda: bool(server.latest.frames("error")), what="an error frame")
    assert server.latest.frames("error")[0]["code"] == code
    assert session.state is State.ONLINE
    assert len(server.connections) == 1


# --- drops and the heartbeat ----------------------------------------------------------


async def test_a_drop_reconnects_and_sends_no_bye(
    harness: Harness, server: FakeLinkedSocket
) -> None:
    session = await _online(harness, server)
    first = server.latest

    first.drop()

    await eventually(lambda: len(server.connections) == 2, what="a reconnect")
    await eventually(lambda: session.state is State.ONLINE, what="online again")
    assert first.frames("bye") == []
    # Inside the presence grace: the section says the clone still shows online.
    assert State.RECONNECTING in harness.seen
    assert STATE_TEXT[State.RECONNECTING].startswith("다시 연결하는 중")


async def test_a_silent_socket_is_given_up_after_the_heartbeat_timeout(
    harness: Harness,
) -> None:
    """The server's pings stop; after `heartbeat_timeout_s` the session redials.

    Killed by: src/uclone_x/link/uclone2/session.py :: async with asyncio.timeout(self._timing.heartbeat_timeout_s):
    Becomes: async with asyncio.timeout(None):
    """
    async with FakeLinkedSocket() as srv:
        await _online(harness, srv)
        first = srv.latest

        await eventually(
            lambda: len(srv.connections) >= 2, timeout_s=3.0, what="a redial after silence"
        )
        await first.closed.wait()
        assert first.frames("bye") == []


async def test_pings_keep_the_socket_open_past_the_heartbeat_timeout(
    harness: Harness,
) -> None:
    # A 2 s timeout against a ping every 0.1 s: twenty pings of margin per window, so a slow
    # machine delays a ping far less than it would need to for the timeout to fire.
    timing = replace(FAST, heartbeat_timeout_s=2.0)
    async with FakeLinkedSocket(ping_every_s=0.1) as srv:
        session = await _online(harness, srv, timing)
        # Counted, not slept: 30 pongs cannot come back in less than 3 s of pings, which is
        # past the heartbeat timeout however the scheduler runs.
        await eventually(
            lambda: len(srv.latest.frames("pong")) >= 30, timeout_s=15.0, what="30 pongs"
        )

        assert len(srv.connections) == 1
        assert session.state is State.ONLINE


# --- the server ending a socket -------------------------------------------------------


@pytest.mark.parametrize(
    ("code", "reason", "state"),
    [
        (4409, "replaced", State.REPLACED),
        (4400, "protocol_error", State.PROTOCOL_ERROR),
    ],
)
async def test_a_close_that_stops_the_session_is_not_redialled(
    harness: Harness,
    server: FakeLinkedSocket,
    store: LinkStore,
    code: int,
    reason: str,
    state: LinkSessionState,
) -> None:
    session = await _online(harness, server)
    await server.latest.close_with(code, reason)

    await eventually(lambda: session.state is state, what=f"state {state}")
    await asyncio.sleep(SEVERAL_ROUNDS_S)
    assert len(server.connections) == 1
    stored = store.get("lnk_1")
    assert stored is not None and stored.enabled


async def test_close_4400_reports_the_preceding_error_to_diagnostics(
    harness: Harness, server: FakeLinkedSocket
) -> None:
    session = await _online(harness, server)
    await server.latest.send(
        {"type": "error", "code": "invalid_frame", "message": "pong.ts is not a time"}
    )
    await server.latest.close_with(4400, "protocol_error")

    await eventually(lambda: session.state is State.PROTOCOL_ERROR, what="protocol error")
    assert "4400" in session.diagnostic
    assert "pong.ts is not a time" in session.diagnostic
    assert "4400" not in session.state_text


async def test_close_4401_stops_and_disables_the_record(
    harness: Harness, server: FakeLinkedSocket, store: LinkStore
) -> None:
    """Killed by: src/uclone_x/link/uclone2/session.py :: 4401: _Ending(_Next.STOP, LinkSessionState.ENDED, disable=True),
    Becomes: 4401: _Ending(_Next.BACKOFF),
    """
    session = await _online(harness, server)
    await server.latest.close_with(4401, "revoked")

    await eventually(lambda: session.state is State.ENDED, what="the link to end")
    await asyncio.sleep(SEVERAL_ROUNDS_S)
    assert len(server.connections) == 1
    stored = store.get("lnk_1")
    assert stored is not None and not stored.enabled


async def test_a_bye_stands_in_for_a_lost_close_frame(
    harness: Harness, server: FakeLinkedSocket, store: LinkStore
) -> None:
    session = await _online(harness, server)
    await server.latest.send({"type": "bye", "reason": "revoked"})
    # Frames are handled in order: once the ping after the bye is answered, the bye was read.
    await server.latest.send({"type": "ping", "ts": "2026-09-27T09:00:30Z"})
    await eventually(lambda: bool(server.latest.frames("pong")), what="the pong after the bye")
    server.latest.drop()

    await eventually(lambda: session.state is State.ENDED, what="the link to end")
    stored = store.get("lnk_1")
    assert stored is not None and not stored.enabled


@pytest.mark.parametrize(
    ("code", "reason", "state"),
    [
        (4503, "kill_switch", State.SERVER_PAUSED),
        (1001, "shutdown", State.ONLINE),
    ],
)
async def test_a_close_that_is_retried_redials(
    harness: Harness,
    server: FakeLinkedSocket,
    code: int,
    reason: str,
    state: LinkSessionState,
) -> None:
    session = await _online(harness, server)
    await server.latest.close_with(code, reason)

    await eventually(lambda: len(server.connections) >= 2, what="a redial")
    if state is State.SERVER_PAUSED:
        assert State.SERVER_PAUSED in harness.seen
    await eventually(lambda: session.state is State.ONLINE, what="online again")


# --- the server refusing the dial -----------------------------------------------------


@pytest.mark.parametrize(
    ("status", "state", "disabled"),
    [(401, State.ENDED, True), (426, State.UPDATE_REQUIRED, False)],
)
async def test_a_refused_dial_that_stops_is_tried_once(
    harness: Harness, store: LinkStore, status: int, state: LinkSessionState, disabled: bool
) -> None:
    async with FakeLinkedSocket(refuse_status=status) as srv:
        session = harness.session(_record(srv.ws_url))
        await eventually(lambda: session.state is state, what=f"state {state}")
        await asyncio.sleep(SEVERAL_ROUNDS_S)

        assert len(srv.requests) == 1
    stored = store.get("lnk_1")
    assert stored is not None and stored.enabled is not disabled
    assert f"HTTP {status}" in session.diagnostic


@pytest.mark.parametrize(("status", "state"), [(503, State.SERVER_PAUSED), (500, State.CONNECTING)])
async def test_a_refused_dial_that_is_retried_keeps_dialling(
    harness: Harness, status: int, state: LinkSessionState
) -> None:
    async with FakeLinkedSocket(refuse_status=status) as srv:
        session = harness.session(_record(srv.ws_url))
        await eventually(lambda: len(srv.requests) >= 2, what="a second dial")
        assert session.state is state


async def test_an_insecure_ws_url_is_not_dialled(harness: Harness) -> None:
    session = harness.session(_record("ws://uclone.test/api/v4/linked/ws"))

    await eventually(lambda: session.state is State.PROTOCOL_ERROR, what="refusal")


# --- going offline on purpose ---------------------------------------------------------


async def test_logout_sends_bye_logout_before_closing(
    harness: Harness, server: FakeLinkedSocket
) -> None:
    """Killed by: src/uclone_x/link/uclone2/session.py :: await ws.send(json.dumps(bye_logout_frame()))
    Becomes: pass
    """
    session = await _online(harness, server)
    conn = server.latest

    await session.stop(logout=True, final=State.PAUSED)

    await asyncio.wait_for(conn.closed.wait(), 2.0)
    assert conn.frames()[-1] == {"type": "bye", "reason": "logout"}
    assert conn.client_close_code == 1000
    assert session.state is State.PAUSED
    await asyncio.sleep(SEVERAL_ROUNDS_S)
    assert len(server.connections) == 1


async def test_a_stop_without_logout_sends_no_bye(
    harness: Harness, server: FakeLinkedSocket
) -> None:
    session = await _online(harness, server)
    conn = server.latest

    await session.stop(logout=False, final=State.OFFLINE)

    await asyncio.wait_for(conn.closed.wait(), 2.0)
    assert conn.frames("bye") == []


# --- the supervisor -------------------------------------------------------------------


def _supervisor(store: LinkStore, rest: FakeUclone2) -> tuple[LinkSupervisor, list[LinkRecord]]:
    dialled: list[LinkRecord] = []

    def factory(record: LinkRecord, st: LinkStore, client: Uclone2LinkClient) -> LinkSession:
        dialled.append(record)
        return LinkSession(record, store=st, client=client, timing=FAST)

    return (
        LinkSupervisor(
            store=store, client=Uclone2LinkClient(transport=rest.transport), session_factory=factory
        ),
        dialled,
    )


async def test_no_link_record_means_no_session_and_no_dial(
    store: LinkStore, rest: FakeUclone2
) -> None:
    supervisor, dialled = _supervisor(store, rest)

    await supervisor.start()

    assert dialled == []
    assert supervisor.states() == {}
    assert rest.requests == []
    await supervisor.shutdown()


@pytest.mark.parametrize("change", [{"enabled": False}, {"paused": True}])
async def test_a_link_that_should_not_run_is_not_dialled(
    store: LinkStore, rest: FakeUclone2, server: FakeLinkedSocket, change: dict[str, object]
) -> None:
    store.put(_record(server.ws_url, **change))
    supervisor, dialled = _supervisor(store, rest)

    await supervisor.start()
    await asyncio.sleep(0.1)

    assert dialled == []
    assert server.requests == []
    await supervisor.shutdown()


async def test_start_retries_pending_unlinks(store: LinkStore, rest: FakeUclone2) -> None:
    store.put(_record("wss://staging.uclone.test/api/v4/linked/ws", unlink_pending=True))
    supervisor, dialled = _supervisor(store, rest)

    await supervisor.start()

    await eventually(lambda: store.get("lnk_1") is None, what="the pending unlink")
    assert len(rest.calls(DELETE_SELF)) == 1
    assert dialled == []
    await supervisor.shutdown()


async def test_going_offline_logs_out_and_going_online_redials(
    store: LinkStore, rest: FakeUclone2, server: FakeLinkedSocket
) -> None:
    store.put(_record(server.ws_url))
    supervisor, _ = _supervisor(store, rest)
    await supervisor.start()
    await eventually(lambda: supervisor.states().get("lnk_1") is State.ONLINE, what="online")
    first = server.latest

    await supervisor.set_online("lnk_1", False)

    assert first.frames()[-1] == {"type": "bye", "reason": "logout"}
    stored = store.get("lnk_1")
    assert stored is not None and stored.paused

    await supervisor.set_online("lnk_1", True)
    await eventually(lambda: supervisor.states().get("lnk_1") is State.ONLINE, what="online again")
    assert len(server.connections) == 2
    await supervisor.shutdown()


async def test_shutdown_logs_every_session_out(
    store: LinkStore, rest: FakeUclone2, server: FakeLinkedSocket
) -> None:
    store.put(_record(server.ws_url))
    store.put(_record(server.ws_url, link_id="lnk_2", bot_id="bot_9c1e", local_agent_id="mina"))
    supervisor, _ = _supervisor(store, rest)
    await supervisor.start()
    await eventually(
        lambda: list(supervisor.states().values()) == [State.ONLINE, State.ONLINE],
        what="both online",
    )

    await supervisor.shutdown()

    for conn in server.connections:
        await asyncio.wait_for(conn.closed.wait(), 2.0)
        assert conn.frames()[-1] == {"type": "bye", "reason": "logout"}
    assert set(supervisor.states().values()) == {State.OFFLINE}


async def test_the_dashboard_lifespan_starts_and_logs_out_the_sessions(
    store: LinkStore, rest: FakeUclone2, server: FakeLinkedSocket, tmp_path: Path
) -> None:
    from uclone_x.ui.app import create_ui_app

    store.put(_record(server.ws_url))
    supervisor, _ = _supervisor(store, rest)
    app = create_ui_app(storage_dir=tmp_path / "sessions", link_supervisor=supervisor)

    async with app.router.lifespan_context(app):
        assert app.state.link_supervisor is supervisor
        await eventually(lambda: supervisor.states().get("lnk_1") is State.ONLINE, what="online")

    assert server.latest.frames()[-1] == {"type": "bye", "reason": "logout"}
    assert supervisor.states() == {"lnk_1": State.OFFLINE}


# --- the loop does not die --------------------------------------------------------------


def test_the_backoff_stays_under_the_cap_after_days_of_failed_dials() -> None:
    """A dashboard offline for days: `attempt` climbs past where `2**attempt` fits a float."""
    timing = SessionTiming()
    rng = random.Random(7)
    for attempt in (0, 30, 1_024, 10**6):
        delay = backoff_delay(attempt, timing, rng)
        assert 0 < delay <= timing.backoff_cap_s, attempt
    assert backoff_delay(10**6, timing, rng) >= timing.backoff_cap_s / 2


async def test_an_unexpected_dial_error_is_noted_and_the_session_redials(
    harness: Harness,
    server: FakeLinkedSocket,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    import uclone_x.link.uclone2.session as session_module

    real_connect = session_module.connect
    calls: list[str] = []

    def connect_once_broken(*args: object, **kwargs: object) -> object:
        calls.append("dial")
        if len(calls) == 1:
            raise RuntimeError("proxy said something with a secret in it")
        return real_connect(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(session_module, "connect", connect_once_broken)
    caplog.set_level(logging.INFO, logger="uclone_x.link.uclone2.session")

    session = harness.session(_record(server.ws_url))
    await eventually(lambda: session.state is State.ONLINE, what="online after the failed dial")

    assert len(calls) == 2
    assert "dial failed unexpectedly: RuntimeError" in caplog.text
    assert "secret" not in caplog.text


async def test_a_store_that_cannot_be_written_does_not_end_the_socket(
    harness: Harness,
    server: FakeLinkedSocket,
    store: LinkStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def refuse(link_id: str, **changes: object) -> LinkRecord | None:
        raise LinkStoreError("disk full")

    session = harness.session(_record(server.ws_url))
    monkeypatch.setattr(store, "update", refuse)
    await eventually(lambda: session.state is State.ONLINE, what="online")
    await server.latest.send({"type": "ping", "ts": "2026-09-27T09:00:30Z"})
    await eventually(lambda: bool(server.latest.frames("pong")), what="a pong")

    assert len(server.connections) == 1
    assert session.state is State.ONLINE


async def test_switching_an_ended_link_on_keeps_its_ended_state(
    store: LinkStore, rest: FakeUclone2, server: FakeLinkedSocket
) -> None:
    store.put(_record(server.ws_url))
    supervisor, _ = _supervisor(store, rest)
    await supervisor.start()
    await eventually(lambda: supervisor.states().get("lnk_1") is State.ONLINE, what="online")
    await server.latest.close_with(4401, "revoked")
    await eventually(lambda: supervisor.states().get("lnk_1") is State.ENDED, what="ended")

    await supervisor.set_online("lnk_1", True)

    assert supervisor.states() == {"lnk_1": State.ENDED}
    assert len(server.connections) == 1
    await supervisor.shutdown()


async def test_shutdown_stops_a_session_the_server_ended_without_a_logout(
    store: LinkStore, rest: FakeUclone2, server: FakeLinkedSocket
) -> None:
    stops: list[tuple[bool, LinkSessionState]] = []

    class Recording(LinkSession):
        async def stop(self, *, logout: bool, final: LinkSessionState) -> None:
            stops.append((logout, final))
            await super().stop(logout=logout, final=final)

    def factory(record: LinkRecord, st: LinkStore, client: Uclone2LinkClient) -> LinkSession:
        return Recording(record, store=st, client=client, timing=FAST)

    store.put(_record(server.ws_url))
    supervisor = LinkSupervisor(
        store=store, client=Uclone2LinkClient(transport=rest.transport), session_factory=factory
    )
    await supervisor.start()
    await eventually(lambda: supervisor.states().get("lnk_1") is State.ONLINE, what="online")
    await server.latest.close_with(4401, "revoked")
    await eventually(lambda: supervisor.states().get("lnk_1") is State.ENDED, what="ended")

    await supervisor.shutdown()

    # Stopped -- so a profile refresh still in flight is cancelled -- but not logged out,
    # and the state the user reads is left as the server set it.
    assert stops == [(False, State.ENDED)]
    assert supervisor.states() == {"lnk_1": State.ENDED}


# --- what reaches the user, the logs and the repr -------------------------------------


def test_every_state_has_plain_copy() -> None:
    assert set(STATE_TEXT) == set(LinkSessionState)
    for state, text in STATE_TEXT.items():
        assert internals_in(text) == [], state
        assert state.value not in text


async def test_the_token_reaches_no_log_repr_or_error(
    harness: Harness, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    caplog.set_level(logging.DEBUG, logger="websockets")
    caplog.set_level(logging.DEBUG, logger="uclone_x.link.uclone2.wire")

    async with FakeLinkedSocket() as srv:
        session = await _online(harness, srv)
        srv.latest.drop()
        await eventually(lambda: len(srv.connections) == 2, what="a reconnect")
        await session.stop(logout=True, final=State.OFFLINE)
    async with FakeLinkedSocket(refuse_status=401) as srv:
        refused = harness.session(_record(srv.ws_url, link_id="lnk_2"))
        await eventually(lambda: refused.state is State.ENDED, what="refusal")

    with pytest.raises(ConnectionError) as raised:
        await session.send_payload({"type": "pong", "ts": "2026-09-27T09:00:20Z"})

    assert caplog.records, "the capture saw nothing, so the check below would be vacuous"
    assert TOKEN not in caplog.text
    assert TOKEN not in repr(session)
    assert TOKEN not in str(raised.value)
    for s in (session, refused):
        assert TOKEN not in s.diagnostic
