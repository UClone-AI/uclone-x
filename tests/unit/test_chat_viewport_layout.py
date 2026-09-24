"""Unit tests for the modern chat layout, static assets, and session endpoints (#339)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import httpx
import pytest

from uclone_x.llm import MockLLMConnector
from uclone_x.ui.app import create_ui_app


@pytest.mark.asyncio
async def test_ui_static_assets_exist_and_served(tmp_path: Path) -> None:
    """Verify that static bundle files generated for the modern layout are properly served."""
    static_dir = Path(__file__).resolve().parents[2] / "src" / "uclone_x" / "ui_static"
    assert (static_dir / "index.html").exists()

    app = create_ui_app(static_dir=static_dir)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
    ) as client:
        res = await client.get("/")
        assert res.status_code == 200
        assert "text/html" in res.headers["content-type"]
        assert "<!doctype html>" in res.text.lower() or "<html" in res.text.lower()


@pytest.mark.asyncio
async def test_ui_chat_multi_turn_session_lifecycle(tmp_path: Path) -> None:
    """Verify chat endpoint creates sessions and retains turn provenance across multi-turn exchanges (#339)."""
    mock_llm = MockLLMConnector(
        responses=[
            "Hello! I am ready to assist with multi-agent orchestration.",
            "Turn 2: FastPath A2A is fully verified with P6 provenance.",
        ]
    )
    app = create_ui_app(static_dir=tmp_path, llm=mock_llm)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1"
    ) as client:
        # Turn 1
        res1 = await client.post(
            "/api/turn",
            json={
                "message": "Start turn 1",
                "agent_id": "agent-general",
                "session_id": "sess_viewport_test",
            },
        )
        assert res1.status_code == 200
        data1 = cast(dict[str, Any], res1.json())
        assert data1["status"] == "success"
        assert "Hello! I am ready" in data1["response"]
        assert data1["turn_count"] == 1
        assert "provenance" in data1
        assert data1["provenance"]["component"] == "uclone_x.llm.orchestrator"

        # Turn 2
        res2 = await client.post(
            "/api/turn",
            json={
                "message": "Start turn 2",
                "agent_id": "agent-general",
                "session_id": "sess_viewport_test",
            },
        )
        assert res2.status_code == 200
        data2 = cast(dict[str, Any], res2.json())
        assert data2["status"] == "success"
        assert data2["turn_count"] == 2
        assert "Turn 2:" in data2["response"]

        # Fetch session list
        res_sessions = await client.get("/api/sessions")
        assert res_sessions.status_code == 200
        sess_data = cast(dict[str, Any], res_sessions.json())
        assert "sessions" in sess_data
        session_ids = [s["session_id"] for s in sess_data["sessions"]]
        assert "sess_viewport_test" in session_ids

        # Fetch history
        res_history = await client.get(
            "/api/session/history",
            params={"agent_id": "agent-general", "session_id": "sess_viewport_test"},
        )
        assert res_history.status_code == 200
        hist_data = cast(dict[str, Any], res_history.json())
        assert "messages" in hist_data
        assert len(hist_data["messages"]) == 4  # 2 user + 2 assistant
