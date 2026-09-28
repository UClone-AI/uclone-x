"""Settings → 연결 → uClone2: the `/api/links*` routes (uclone2-link.md §3.6).

The routes run over a real `LinkSupervisor`, `LinkStore` and service, against the
in-process uClone2 fake; only the socket session is a stub, so a test can set the state a
card must word without dialling anything.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pytest
from fastapi import FastAPI, HTTPException, Request
from fastapi.testclient import TestClient
from pydantic import SecretStr

from tests.support.uclone2_fake import (
    CONNECT,
    CONNECT_URL,
    DELETE_SELF,
    ORIGIN,
    TOKEN,
    FakeUclone2,
    internals_in,
)
from uclone_x.link.uclone2.client import MESSAGES, LinkFailure, Uclone2LinkClient
from uclone_x.link.uclone2.models import LinkRecord
from uclone_x.link.uclone2.runtime_lock import RuntimeLock, RuntimeRole, runtime_lock_path
from uclone_x.link.uclone2.session import STATE_TEXT, LinkSession, LinkSessionState
from uclone_x.link.uclone2.store import LinkStore
from uclone_x.link.uclone2.supervisor import LinkSupervisor
from uclone_x.ui.links import register_link_routes

State = LinkSessionState

_REPO = Path(__file__).resolve().parents[2]
_KO_CATALOG = _REPO / "frontend" / "src" / "i18n" / "locales" / "ko" / "links.json"


class _StubSession:
    """A session that dials nothing; its state is whatever the test sets."""

    def __init__(self, record: LinkRecord) -> None:
        self.link_id = record.link_id
        self.state = State.CONNECTING
        self.started = 0
        self.stops: list[tuple[bool, State]] = []

    def start(self) -> None:
        self.started += 1

    async def stop(self, *, logout: bool, final: State) -> None:
        self.stops.append((logout, final))
        self.state = final


class _Harness:
    def __init__(self, tmp_path: Path, known: list[str]) -> None:
        self.fake = FakeUclone2()
        self.store = LinkStore(tmp_path / "links" / "uclone2.json")
        self.sessions: list[_StubSession] = []
        self.known = known

        def factory(record: LinkRecord, _s: LinkStore, _c: Uclone2LinkClient) -> LinkSession:
            session = _StubSession(record)
            self.sessions.append(session)
            return cast(LinkSession, session)

        self.supervisor = LinkSupervisor(
            store=self.store,
            client=Uclone2LinkClient(transport=self.fake.transport),
            session_factory=factory,
        )
        app = FastAPI()

        def refuse_cross_origin(request: Request) -> None:
            if request.headers.get("origin") not in (None, "http://localhost"):
                raise HTTPException(status_code=403, detail="cross-origin")

        register_link_routes(
            app,
            supervisor=self.supervisor,
            local_clone_names=lambda: list(self.known),
            refuse_cross_origin=refuse_cross_origin,
        )
        self.client = TestClient(app)

    def link(self, **body: Any) -> dict[str, Any]:
        res = self.client.post("/api/links/uclone2", json={"connect": CONNECT_URL, **body})
        assert res.status_code == 200, res.text
        return cast(dict[str, Any], res.json())

    def cards(self) -> list[dict[str, Any]]:
        res = self.client.get("/api/links")
        assert res.status_code == 200
        return cast(list[dict[str, Any]], res.json()["links"])


@pytest.fixture
def h(tmp_path: Path) -> Iterator[_Harness]:
    harness = _Harness(tmp_path, known=["clone", "haru"])
    with harness.client:
        yield harness


def _record(link_id: str = "lnk_1", **changes: Any) -> LinkRecord:
    base = LinkRecord(
        link_id=link_id,
        server_url=ORIGIN,
        bot_id=f"bot_{link_id}",
        remote_username="haru",
        remote_display_name="Haru",
        local_agent_id="haru",
        token=SecretStr(TOKEN),
        ws_url="wss://staging.uclone.test/api/v4/linked/ws",
        created_at=datetime(2026, 9, 27, tzinfo=UTC),
    )
    return base.model_copy(update=changes)


# --- the token ---------------------------------------------------------------------------


def test_no_response_or_log_carries_the_token(
    h: _Harness, caplog: pytest.LogCaptureFixture
) -> None:
    """Not the token, and not its masked hint either: nothing starting `ucl_` at all.

    Killed by: src/uclone_x/ui/links.py :: "page_url": _page_url(record),
    Becomes: "page_url": _page_url(record), "token_hint": record.view().token_hint,
    """
    caplog.set_level(logging.DEBUG)
    bodies = [json.dumps(h.link())]
    link_id = h.cards()[0]["link_id"]
    bodies.append(h.client.get("/api/links").text)
    bodies.append(h.client.post(f"/api/links/{link_id}/enabled", json={"enabled": False}).text)
    bodies.append(h.client.post(f"/api/links/{link_id}/enabled", json={"enabled": True}).text)
    h.fake.unreachable.add(DELETE_SELF)
    bodies.append(h.client.delete(f"/api/links/{link_id}").text)
    bodies.append(h.client.delete(f"/api/links/{link_id}?local=true").text)
    for body in bodies:
        assert "ucl_" not in body
        assert TOKEN[-4:] not in body
    assert TOKEN not in caplog.text


# --- connect -----------------------------------------------------------------------------


def test_connect_starts_the_session_at_once(h: _Harness) -> None:
    card = h.link()

    assert card["local_agent_id"] == "haru"
    assert card["remote_username"] == "haru"
    assert card["page_url"] == f"{ORIGIN}/hompy/haru"
    assert [s.started for s in h.sessions] == [1]
    assert card["state"] == "connecting"
    h.sessions[0].state = State.ONLINE
    assert [c["state"] for c in h.cards()] == ["online"]


def test_connect_uses_the_chosen_local_clone(h: _Harness) -> None:
    assert h.link(local_agent_id="clone")["local_agent_id"] == "clone"


def test_linking_the_same_clone_again_stops_the_replaced_session(h: _Harness) -> None:
    first = h.link()
    second = h.link()

    assert first["link_id"] != second["link_id"]
    assert [c["link_id"] for c in h.cards()] == [second["link_id"]]
    assert h.sessions[0].stops == [(False, State.OFFLINE)]
    assert h.sessions[1].started == 1


@pytest.mark.parametrize(
    ("status", "failure"),
    [
        (400, LinkFailure.CODE_INVALID),
        (409, LinkFailure.CAP_REACHED),
        (429, LinkFailure.RATE_LIMITED),
        (503, LinkFailure.DISABLED),
    ],
)
def test_a_refusal_is_a_code_and_a_plain_sentence(
    h: _Harness, status: int, failure: LinkFailure
) -> None:
    h.fake.answer(CONNECT, status)
    res = h.client.post("/api/links/uclone2", json={"connect": CONNECT_URL})

    assert res.json() == {"code": failure.value, "message": MESSAGES[failure]}
    assert internals_in(res.json()["message"]) == []
    assert h.cards() == []


def test_unreachable_uclone2_is_a_plain_sentence(h: _Harness) -> None:
    h.fake.unreachable.add(CONNECT)
    res = h.client.post("/api/links/uclone2", json={"connect": CONNECT_URL})

    assert res.json()["code"] == "unreachable"
    assert internals_in(res.text) == []


def test_bad_input_and_an_unreadable_body_send_nothing(h: _Harness) -> None:
    for res in (
        h.client.post("/api/links/uclone2", json={"connect": "not a code"}),
        h.client.post("/api/links/uclone2", content=b"{"),
        h.client.post("/api/links/uclone2", json={}),
    ):
        assert res.status_code == 400
        assert res.json()["code"] == "bad_input"
        assert internals_in(res.json()["message"]) == []
    assert h.fake.requests == []


def test_a_named_clone_that_is_not_here_costs_no_code(h: _Harness) -> None:
    res = h.client.post(
        "/api/links/uclone2", json={"connect": CONNECT_URL, "local_agent_id": "nobody"}
    )

    assert res.json()["code"] == "no_local_clone"
    assert h.fake.requests == []


def test_no_local_clone_at_all_costs_no_code(h: _Harness) -> None:
    h.known = []
    res = h.client.post("/api/links/uclone2", json={"connect": CONNECT_URL})

    assert res.json()["code"] == "no_local_clone"
    assert h.fake.requests == []


def test_the_fallback_clone_must_exist_or_the_link_is_undone(h: _Harness) -> None:
    """#1853 follow-up (b): `clone` was used without checking it is a local clone.

    Killed by: src/uclone_x/link/uclone2/service.py :: if FALLBACK_CLONE in known:
    Becomes: if True:
    """
    h.known = ["mina"]
    res = h.client.post("/api/links/uclone2", json={"connect": CONNECT_URL})

    assert res.json()["code"] == "no_matching_clone"
    assert internals_in(res.json()["message"]) == []
    assert len(h.fake.calls(DELETE_SELF)) == 1
    assert h.cards() == []
    assert h.sessions == []


def test_the_same_name_matches_whatever_its_case(h: _Harness) -> None:
    h.known = ["Haru"]
    assert h.link()["local_agent_id"] == "Haru"


# --- state -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("changes", "session_state", "expected"),
    [
        ({}, State.ONLINE, "online"),
        ({}, State.RECONNECTING, "reconnecting"),
        ({}, State.REPLACED, "replaced"),
        ({}, State.SERVER_PAUSED, "server_paused"),
        ({}, None, "offline"),
        ({"paused": True}, None, "paused"),
        ({"enabled": False}, State.ENDED, "ended"),
        ({"enabled": False}, None, "ended"),
        ({"unlink_pending": True}, None, "unlink_pending"),
        ({"unlink_pending": True, "enabled": False}, None, "unlink_pending"),
    ],
)
def test_each_card_names_its_state(
    h: _Harness, changes: dict[str, Any], session_state: State | None, expected: str
) -> None:
    """Killed by: src/uclone_x/ui/links.py :: if record.paused:
    Becomes: if False:
    """
    record = _record(**changes)
    h.store.put(record)
    if session_state is not None:
        h.supervisor._start(record)  # pyright: ignore[reportPrivateUsage]
        h.sessions[-1].state = session_state
    assert [c["state"] for c in h.cards()] == [expected]


def test_the_korean_catalog_says_the_design_sentence_for_each_state() -> None:
    """The section words each state in the head's catalog; Korean must be `STATE_TEXT`."""
    catalog = json.loads(_KO_CATALOG.read_text(encoding="utf-8"))["state"]
    for state, sentence in STATE_TEXT.items():
        assert catalog[state.value] == sentence


# --- another runtime holds the links ------------------------------------------------------


def test_while_link_run_holds_the_links_the_section_says_so_and_changes_nothing(
    h: _Harness,
) -> None:
    """`./ucx link run` on this machine: no session here, no switch, no code spent.

    Killed by: src/uclone_x/ui/links.py :: if record.link_id not in session_states and elsewhere:
    Becomes: if False:
    """
    other = RuntimeLock(runtime_lock_path(h.store.path), RuntimeRole.LINK_RUN)
    assert other.acquire()
    try:
        h.store.put(_record())
        h.client.portal.call(h.supervisor.start)  # pyright: ignore[reportOptionalMemberAccess]

        assert h.sessions == []
        assert [c["state"] for c in h.cards()] == ["elsewhere"]
        toggle = h.client.post("/api/links/lnk_1/enabled", json={"enabled": False})
        linked = h.client.post("/api/links/uclone2", json={"connect": CONNECT_URL})
        for res in (toggle, linked):
            assert res.status_code == 409
            assert res.json()["code"] == "elsewhere"
            assert internals_in(res.text) == []
        assert h.fake.requests == []
        stored = h.store.get("lnk_1")
        assert stored is not None and not stored.paused
    finally:
        other.release()


def test_a_dashboard_opened_with_no_links_sees_a_link_run_started_after_it(
    h: _Harness,
) -> None:
    """The dashboard had nothing to dial, so no retry of its own: the lock is the answer.

    Opened empty; then `ucx link uclone2` adds a link and `ucx link run` takes the links.
    The card must say *elsewhere* at once, not *offline* until something is clicked.

    Killed by: src/uclone_x/link/uclone2/supervisor.py :: return self._runtime_lock().held_by_another()
    Becomes: return False
    """
    h.client.portal.call(h.supervisor.start)  # pyright: ignore[reportOptionalMemberAccess]
    other = RuntimeLock(runtime_lock_path(h.store.path), RuntimeRole.LINK_RUN)
    h.store.put(_record())
    assert other.acquire()
    try:
        assert [c["state"] for c in h.cards()] == ["elsewhere"]
        toggle = h.client.post("/api/links/lnk_1/enabled", json={"enabled": False})
        assert toggle.status_code == 409
        assert toggle.json()["code"] == "elsewhere"
        assert h.sessions == []
    finally:
        other.release()
    # Once `link run` stops, the card is this head's again: nothing runs it here yet.
    assert [c["state"] for c in h.cards()] != ["elsewhere"]


# --- online / offline --------------------------------------------------------------------


def test_the_toggle_logs_out_and_is_kept(h: _Harness) -> None:
    """Killed by: src/uclone_x/ui/links.py :: record = await supervisor.set_online(link_id, body.enabled)
    Becomes: record = await supervisor.set_online(link_id, True)
    """
    link_id = h.link()["link_id"]
    h.sessions[0].state = State.ONLINE

    off = h.client.post(f"/api/links/{link_id}/enabled", json={"enabled": False})
    assert off.json()["state"] == "paused"
    assert h.sessions[0].stops == [(True, State.PAUSED)]
    stored = h.store.get(link_id)
    assert stored is not None and stored.paused

    on = h.client.post(f"/api/links/{link_id}/enabled", json={"enabled": True})
    assert on.json()["state"] == "connecting"
    assert len(h.sessions) == 2 and h.sessions[1].started == 1
    stored = h.store.get(link_id)
    assert stored is not None and not stored.paused


def test_the_toggle_refuses_an_unknown_link_and_a_bad_body(h: _Harness) -> None:
    missing = h.client.post("/api/links/lnk_none/enabled", json={"enabled": False})
    assert missing.status_code == 404
    assert missing.json()["code"] == "not_found"
    link_id = h.link()["link_id"]
    bad = h.client.post(f"/api/links/{link_id}/enabled", json={"enabled": "maybe"})
    assert bad.json()["code"] == "bad_input"


# --- unlink ------------------------------------------------------------------------------


def test_unlink_removes_the_record_and_stops_the_session(h: _Harness) -> None:
    link_id = h.link()["link_id"]
    res = h.client.delete(f"/api/links/{link_id}")

    assert res.json() == {"outcome": "removed"}
    assert h.cards() == []
    # uClone2 revokes and closes the socket itself; no logout of our own.
    assert h.sessions[0].stops == [(False, State.OFFLINE)]


def test_an_unlink_uclone2_does_not_confirm_is_kept_pending(h: _Harness) -> None:
    link_id = h.link()["link_id"]
    h.fake.answer(DELETE_SELF, 503)
    res = h.client.delete(f"/api/links/{link_id}")

    assert res.json() == {"outcome": "pending"}
    assert [c["state"] for c in h.cards()] == ["unlink_pending"]
    assert h.sessions[0].stops == [(True, State.OFFLINE)]


def test_a_pending_record_can_be_removed_from_this_machine_only(h: _Harness) -> None:
    """#1853 follow-up (a): a DELETE uClone2 keeps refusing left the record forever."""
    link_id = h.link()["link_id"]
    h.fake.answer(DELETE_SELF, 503)
    h.client.delete(f"/api/links/{link_id}")
    calls = len(h.fake.requests)

    res = h.client.delete(f"/api/links/{link_id}?local=true")

    assert res.json() == {"outcome": "forgotten"}
    assert h.cards() == []
    assert len(h.fake.requests) == calls  # uClone2 was not asked


def test_only_a_pending_record_can_be_removed_locally(h: _Harness) -> None:
    link_id = h.link()["link_id"]
    res = h.client.delete(f"/api/links/{link_id}?local=true")

    assert res.status_code == 409
    assert res.json()["code"] == "not_pending"
    assert [c["link_id"] for c in h.cards()] == [link_id]


# --- the store and the origin ------------------------------------------------------------


def test_an_unreadable_links_file_is_said_not_raised(h: _Harness) -> None:
    path = h.store.path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json", encoding="utf-8")

    assert h.client.get("/api/links").json() == {"links": [], "unreadable": True}


def test_every_route_refuses_another_origin(h: _Harness) -> None:
    evil = {"origin": "https://evil.example"}
    assert h.client.get("/api/links", headers=evil).status_code == 403
    assert (
        h.client.post("/api/links/uclone2", json={"connect": CONNECT_URL}, headers=evil).status_code
        == 403
    )
    assert h.client.delete("/api/links/lnk_1", headers=evil).status_code == 403
    assert (
        h.client.post("/api/links/lnk_1/enabled", json={"enabled": False}, headers=evil).status_code
        == 403
    )
    assert h.fake.requests == []


async def test_the_dashboard_mounts_the_routes_over_its_own_supervisor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from uclone_x.ui.app import create_ui_app

    monkeypatch.setenv("UCLONE_SESSION_DIR", str(tmp_path / "sessions"))
    monkeypatch.setenv("UCLONE_DIAGNOSTICS_DIR", str(tmp_path / "diagnostics"))
    store = LinkStore(tmp_path / "links" / "uclone2.json")
    store.put(_record(paused=True))
    supervisor = LinkSupervisor(
        store=store, client=Uclone2LinkClient(transport=FakeUclone2().transport)
    )
    app = create_ui_app(
        static_dir=tmp_path, storage_dir=tmp_path / "sessions", link_supervisor=supervisor
    )

    with TestClient(app) as client:
        body = client.get("/api/links").json()
    assert [(c["link_id"], c["state"]) for c in body["links"]] == [("lnk_1", "paused")]
