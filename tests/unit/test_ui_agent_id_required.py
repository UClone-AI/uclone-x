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


def test_the_refusal_says_where_to_find_the_names(client: TestClient) -> None:
    """P0: the reader is not assumed to know the API.

    A bare "field required" leaves a non-expert with no next step, so the refusal names
    the endpoint that lists this install's agents rather than only the missing field. A
    blank name is no name: answering it would mean inventing the answerer (P6).

    Killed by: src/uclone_x/ui/app.py :: if not isinstance(raw, str) or not raw.strip():
    Becomes: if False:
    """
    refused = client.post("/api/dispatch", json={"task": "summarize", "agent_id": "  "})
    assert refused.status_code == 400, refused.text
    detail = refused.json()["detail"]

    assert "/api/clones" in detail
    assert "no default agent" in detail


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
