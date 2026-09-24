"""A request that names no agent is refused, not answered by one we picked (#1125)."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from uclone_x.engine.event_bus import AgentEvent
from uclone_x.llm import MockLLMConnector
from uclone_x.ui.app import create_ui_app


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    return TestClient(create_ui_app(static_dir=tmp_path, llm=MockLLMConnector()))


def test_a_chat_request_that_names_no_agent_is_refused(client: TestClient) -> None:
    """Answering it meant inventing the answerer.

    These routes read a built-in persona's name when the body carried no `agent_id`. On
    an install that has no agent by that name -- which is any install whose agents are
    its own -- the turn still ran: it was executed under a config assembled for a name
    the caller never wrote, answered 200, and was written into the session record as
    that agent's. Nothing reported the substitution, which is what P6 forbids; the user
    saw a reply from somebody they had not addressed.

    Killed by: src/uclone_x/ui/app.py :: if not isinstance(raw, str) or not raw.strip():
    Becomes: if False:
    """
    refused = client.post("/api/turn", json={"message": "hello"})

    assert refused.status_code == 400, refused.text
    assert "agent_id" in refused.json()["detail"]


def test_the_refusal_says_where_to_find_the_names(client: TestClient) -> None:
    """P0: the reader is not assumed to know the API.

    A bare "field required" leaves a non-expert with no next step, so the refusal names
    the endpoint that lists this install's agents rather than only the missing field.
    """
    detail = client.post("/api/turn", json={"message": "hello"}).json()["detail"]

    assert "/api/agents" in detail
    assert "no default agent" in detail


def test_truncating_a_conversation_names_the_agent_whose_it_is(client: TestClient) -> None:
    """Truncation reached for a default through a chain of `or`s.

    `str(body.get("agent_id") or agent_id or "champion")` let a truncate request with no
    name cut a *different* agent's transcript back to an index the caller chose for
    theirs -- a destructive edit, attributed to the wrong agent and not reported.

    Killed by: src/uclone_x/ui/app.py :: eff_agent_id = _required_agent_id(body.get("agent_id") or agent_id)
    Becomes: eff_agent_id = str(body.get("agent_id") or agent_id or "champion")
    """
    refused = client.post("/api/session/history/truncate", json={"index": 0})

    assert refused.status_code == 400, refused.text
    assert "agent_id" in refused.json()["detail"]


@pytest.mark.parametrize("method", ["GET", "DELETE"])
def test_reading_or_clearing_history_without_a_name_is_refused(
    client: TestClient, method: str
) -> None:
    """The two query-parameter routes refuse with FastAPI's own 422.

    No `Killed by:` declaration: requiredness here is the *absence* of a default in the
    signature (`agent_id: str`), and that line is not unique in the file, so no
    single-line mutation names it. The test still pins the observable behaviour --
    without it, re-adding `= "champion"` to either signature would make both routes read
    and *delete* a named agent's transcript for a request that named nobody.
    """
    refused = client.request(method, "/api/session/history")

    assert refused.status_code == 422, refused.text


def test_a_dispatch_without_a_recipient_is_refused(client: TestClient) -> None:
    """A spawn was published to an agent named after the clock, and called dispatched.

    `str(req.get("agent_id", f"agent_{now_ts}"))` made up a recipient, published
    `SUBAGENT_SPAWN` to it and answered `{"status": "dispatched"}`. No such agent exists,
    so nothing ever ran the task; the caller was told one had been handed over (P6).

    Killed by: src/uclone_x/ui/app.py :: recipient_id = _required_agent_id(req.get("agent_id"))
    Becomes: recipient_id = str(req.get("agent_id", f"agent_{now_ts}"))
    """
    refused = client.post("/api/dispatch", json={"task": "summarize the log"})

    assert refused.status_code == 400, refused.text
    assert "agent_id" in refused.json()["detail"]


@pytest.mark.asyncio
async def test_a_dispatch_is_published_on_the_apps_own_bus(tmp_path: Path) -> None:
    """The route reached for the process-wide bus, not the one the app was built with.

    An app given `bus=` streams that bus to its heads, so a spawn published elsewhere was
    answered `dispatched` and never reached a listener of this app.

    Killed by: src/uclone_x/ui/app.py :: dispatch_bus = active_bus
    Becomes: dispatch_bus = get_ui_event_bus()
    """
    import httpx

    from uclone_x.engine.event_bus import EventBus  # noqa: PLC0415

    bus = EventBus()
    seen: list[AgentEvent] = []

    async def _record(event: AgentEvent) -> None:
        seen.append(event)

    bus.subscribe_callback("swarm.dispatch", _record)
    app = create_ui_app(static_dir=tmp_path, bus=bus, llm=MockLLMConnector())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
    ) as client:
        answered = await client.post(
            "/api/dispatch", json={"task": "summarize the log", "agent_id": "champion"}
        )
    await bus.wait_until_idle()

    assert answered.status_code == 200, answered.text
    assert [e.payload["task"] for e in seen] == ["summarize the log"]
