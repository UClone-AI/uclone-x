"""``ucx link run``: the links online without a dashboard, and one runtime per links file.

The signal tests run the command's coroutine in this test's event loop against the loopback
socket fake (`FakeLinkedSocket`) and send the process a real SIGINT or SIGTERM, so what is
checked is what goes over the wire when the user stops it: `bye{logout}`, then the close.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import signal
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import SecretStr
from typer.testing import CliRunner

import uclone_x.cli.commands.link as link_cli
from tests.support.uclone2_fake import BOT_ID, ORIGIN, TOKEN, FakeUclone2, internals_in
from tests.support.uclone2_ws_fake import FakeLinkedSocket, eventually
from uclone_x.cli.main import app
from uclone_x.link.uclone2.client import LinkError, LinkFailure, Uclone2LinkClient
from uclone_x.link.uclone2.models import LinkRecord
from uclone_x.link.uclone2.runtime_lock import RuntimeLock, RuntimeRole, runtime_lock_path
from uclone_x.link.uclone2.session import STATE_TEXT, LinkSessionState
from uclone_x.link.uclone2.store import LINKS_DIR_ENV_VAR, LinkStore
from uclone_x.link.uclone2.supervisor import LinkSupervisor
from uclone_x.ui.links import ELSEWHERE_STATE, link_state

State = LinkSessionState
runner = CliRunner()


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
        # Equal to the fake's `hello.clone_updated_at`: no profile refresh to wait on.
        clone_updated_at=datetime(2026, 9, 26, 22, 40, tzinfo=UTC),
    )
    return base.model_copy(update=changes)


@pytest.fixture
def rest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeUclone2:
    server = FakeUclone2()
    monkeypatch.setenv(LINKS_DIR_ENV_VAR, str(tmp_path / "links"))
    monkeypatch.setattr(
        link_cli, "make_client", lambda: Uclone2LinkClient(transport=server.transport)
    )
    return server


@pytest.fixture
async def server() -> AsyncIterator[FakeLinkedSocket]:
    async with FakeLinkedSocket() as fake:
        yield fake


def _run() -> tuple[int, str]:
    result = runner.invoke(app, ["link", "run"])
    assert result.exception is None or isinstance(result.exception, SystemExit), repr(
        result.exception
    )
    return result.exit_code, result.output


def _plain(text: str) -> None:
    assert "ucl_" not in text
    assert internals_in(text) == []
    for frame_name in ("bye", "hello", "ready", "logout", "4409", "ws://"):
        assert frame_name not in text


# --- nothing to run ---------------------------------------------------------------------


@pytest.mark.parametrize("change", [None, {"paused": True}, {"enabled": False}])
def test_with_no_enabled_link_it_says_so_and_dials_nothing(
    rest: FakeUclone2, change: dict[str, object] | None
) -> None:
    """Killed by: src/uclone_x/cli/commands/link.py :: if not runnable:
    Becomes: if False:
    """
    if change is not None:
        LinkStore().put(_record("ws://127.0.0.1:9/api/v4/linked/ws", **change))

    code, out = _run()

    assert code == 0
    assert "켜 둘 uClone2 연결이 없습니다" in out
    assert rest.requests == []
    assert "Ctrl-C" not in out
    _plain(out)


# --- a deliberate stop logs out ---------------------------------------------------------


@pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGTERM])
async def test_a_signal_logs_every_link_out_before_closing(
    rest: FakeUclone2,
    server: FakeLinkedSocket,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
    sig: signal.Signals,
) -> None:
    """Killed by: src/uclone_x/link/uclone2/supervisor.py :: else s.stop(logout=True, final=LinkSessionState.OFFLINE)
    Becomes: else s.stop(logout=False, final=LinkSessionState.OFFLINE)
    """
    caplog.set_level(logging.DEBUG)
    before = signal.getsignal(sig)
    store = LinkStore()
    store.put(_record(server.ws_url))
    running = asyncio.create_task(link_cli._serve_links(store, link_cli.make_client()))  # pyright: ignore[reportPrivateUsage]
    await eventually(lambda: bool(server.connections), what="the dial")
    conn = server.latest
    await eventually(lambda: bool(conn.frames("ready")), what="ready")
    await eventually(
        lambda: STATE_TEXT[State.ONLINE] in capsys.readouterr().out, what="the online line"
    )

    # The command's own handler, not the event loop runner's or the default one, which
    # would interrupt or kill the run instead of stopping it cleanly.
    assert signal.getsignal(sig) is not before
    os.kill(os.getpid(), sig)
    code = await asyncio.wait_for(running, 5.0)

    await asyncio.wait_for(conn.closed.wait(), 2.0)
    assert code == 0
    assert conn.frames()[-1] == {"type": "bye", "reason": "logout"}
    assert conn.client_close_code == 1000
    assert len(server.connections) == 1
    out = capsys.readouterr().out
    assert "오프라인으로 전환하고 멈췄습니다" in out
    for text in [out, *(r.getMessage() for r in caplog.records)]:
        assert TOKEN not in text
    _plain(out)


async def test_state_lines_are_the_ui_sentences(
    rest: FakeUclone2, server: FakeLinkedSocket, capsys: pytest.CaptureFixture[str]
) -> None:
    store = LinkStore()
    store.put(_record(server.ws_url))
    running = asyncio.create_task(link_cli._serve_links(store, link_cli.make_client()))  # pyright: ignore[reportPrivateUsage]
    seen = ""

    def online() -> bool:
        nonlocal seen
        seen += capsys.readouterr().out
        return STATE_TEXT[State.ONLINE] in seen

    await eventually(online, what="the online line")
    assert "uClone2 @haru (haru): " + STATE_TEXT[State.ONLINE] in seen
    os.kill(os.getpid(), signal.SIGTERM)
    assert await asyncio.wait_for(running, 5.0) == 0
    _plain(seen)


async def test_it_exits_nonzero_when_the_server_ends_every_link(
    rest: FakeUclone2, server: FakeLinkedSocket, capsys: pytest.CaptureFixture[str]
) -> None:
    store = LinkStore()
    store.put(_record(server.ws_url))
    running = asyncio.create_task(link_cli._serve_links(store, link_cli.make_client()))  # pyright: ignore[reportPrivateUsage]
    await eventually(
        lambda: bool(server.connections and server.latest.frames("ready")), what="ready"
    )

    await server.latest.close_with(4409, "replaced")

    assert await asyncio.wait_for(running, 5.0) == 1
    out = capsys.readouterr().out
    assert STATE_TEXT[State.REPLACED] in out
    assert server.latest.frames("bye") == []
    _plain(out)


# --- one runtime per links file ---------------------------------------------------------


async def test_link_run_is_refused_while_the_dashboard_holds_the_links(
    rest: FakeUclone2, server: FakeLinkedSocket, capsys: pytest.CaptureFixture[str]
) -> None:
    """Killed by: src/uclone_x/link/uclone2/runtime_lock.py :: fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    Becomes: pass
    """
    store = LinkStore()
    store.put(_record(server.ws_url))
    dashboard = LinkSupervisor(store=store, client=link_cli.make_client())
    await dashboard.start()
    await eventually(lambda: dashboard.states().get("lnk_1") is State.ONLINE, what="online")

    running = asyncio.create_task(link_cli._serve_links(store, link_cli.make_client()))  # pyright: ignore[reportPrivateUsage]
    done, _ = await asyncio.wait({running}, timeout=3.0)

    if not done:  # it went on to serve the links: stop it before failing
        running.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await running
    assert done, "link run served the links while the dashboard held them"
    code = running.result()
    assert code == 1
    out = capsys.readouterr().out
    assert "대시보드가 이미 uClone2 연결을 맡고 있습니다" in out
    assert len(server.requests) == 1  # the dashboard's dial only
    assert server.latest.frames("bye") == []
    _plain(out)
    await dashboard.shutdown()


async def test_a_dashboard_dials_nothing_while_link_run_holds_the_links_and_takes_over_after(
    rest: FakeUclone2, server: FakeLinkedSocket
) -> None:
    """Killed by: src/uclone_x/link/uclone2/supervisor.py :: if not self.claim():  # another runtime
    Becomes: if False:  # another runtime
    """
    store = LinkStore()
    store.put(_record(server.ws_url))
    headless = LinkSupervisor(store=store, client=link_cli.make_client(), role=RuntimeRole.LINK_RUN)
    await headless.start()
    await eventually(lambda: headless.states().get("lnk_1") is State.ONLINE, what="online")

    dashboard = LinkSupervisor(store=store, client=link_cli.make_client(), claim_retry_s=0.05)
    await dashboard.start()
    await asyncio.sleep(0.2)

    assert len(server.requests) == 1
    assert dashboard.states() == {}
    assert dashboard.held_elsewhere
    assert dashboard.holder() is RuntimeRole.LINK_RUN
    assert link_state(_record(server.ws_url), dashboard.states(), elsewhere=True) == ELSEWHERE_STATE
    with pytest.raises(LinkError) as refused:
        await dashboard.set_online("lnk_1", False)
    assert refused.value.failure is LinkFailure.ELSEWHERE
    stored = store.get("lnk_1")
    assert stored is not None and not stored.paused

    await headless.shutdown()
    await eventually(lambda: dashboard.states().get("lnk_1") is State.ONLINE, what="take-over")
    assert not dashboard.held_elsewhere
    assert server.connections[0].frames()[-1] == {"type": "bye", "reason": "logout"}
    await dashboard.shutdown()


async def test_linking_from_a_dashboard_that_does_not_hold_the_links_spends_no_code(
    rest: FakeUclone2, server: FakeLinkedSocket
) -> None:
    store = LinkStore()
    store.put(_record(server.ws_url))
    headless = LinkSupervisor(store=store, client=link_cli.make_client(), role=RuntimeRole.LINK_RUN)
    await headless.start()
    await eventually(lambda: headless.states().get("lnk_1") is State.ONLINE, what="online")
    dashboard = LinkSupervisor(store=store, client=link_cli.make_client())

    with pytest.raises(LinkError) as refused:
        await dashboard.link("https://staging.uclone.test/link/ABCD", choose_local=lambda _: "haru")

    assert refused.value.failure is LinkFailure.ELSEWHERE
    assert rest.requests == []
    await dashboard.shutdown()
    await headless.shutdown()


async def test_the_links_are_let_go_only_after_every_logout_is_written(
    rest: FakeUclone2, server: FakeLinkedSocket
) -> None:
    """A runtime that let go first could be replaced (4409) by the next one before its
    `bye{logout}` went out; the lock must still be held when uClone2 reads the bye.

    Killed by: src/uclone_x/link/uclone2/supervisor.py :: # A session the server stopped keeps its state, but its background work
    Becomes: self._lock.release() if self._lock else None  # A session the server stopped keeps its state, but its background work
    """
    store = LinkStore()
    store.put(_record(server.ws_url))
    headless = LinkSupervisor(store=store, client=link_cli.make_client(), role=RuntimeRole.LINK_RUN)
    await headless.start()
    await eventually(lambda: headless.states().get("lnk_1") is State.ONLINE, what="online")
    probe = RuntimeLock(runtime_lock_path(store.path), RuntimeRole.DASHBOARD)
    taken_at_bye: list[bool] = []

    class _ProbingFrames(list[object]):
        def append(self, frame: object) -> None:
            if frame == {"type": "bye", "reason": "logout"}:
                taken = probe.acquire()
                if taken:
                    probe.release()
                taken_at_bye.append(taken)
            super().append(frame)

    conn = server.latest
    conn.received = _ProbingFrames(conn.received)  # pyright: ignore[reportAttributeAccessIssue]

    await headless.shutdown()

    assert conn.frames()[-1] == {"type": "bye", "reason": "logout"}
    assert taken_at_bye == [False]
    assert probe.acquire()  # and let go once the logouts are done
    probe.release()
