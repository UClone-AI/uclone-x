"""Turn aid and hook unit tests for the browser state section."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from uclone_x.agent.models import ToolExecutionRecord
from uclone_x.browser.service import BrowserService, BrowserTab, BrowserWindow
from uclone_x.browser.turn import BrowserTurnHook, browser_state_section
from uclone_x.tools.models import ToolContext


def test_browser_state_section_single_tab() -> None:
    """Killed by: src/uclone_x/browser/turn.py :: return f"[Browser: 1 open tab — {tabs_desc[0]}]"
    Becomes: return ""
    """
    service = BrowserService(AsyncMock())
    tab = BrowserTab(
        target="t1",
        events=asyncio.Queue(),
        url="https://example.com",
        title="Example Domain",
        controller="scout",
    )
    service._windows[("room-1", "scout")] = BrowserWindow(tabs=[tab], active=0)  # pyright: ignore[reportPrivateUsage]

    section = browser_state_section(service, "room-1")
    assert (
        section
        == '[Browser: 1 open tab — Tab 1: "Example Domain" (https://example.com), controller: scout]'
    )


def test_browser_state_section_multiple_tabs() -> None:
    """Killed by: src/uclone_x/browser/turn.py :: return f"[Browser: {len(tabs_desc)} open tabs — {joined}]"
    Becomes: return ""
    """
    service = BrowserService(AsyncMock())
    tab1 = BrowserTab(
        target="t1",
        events=asyncio.Queue(),
        url="https://a.com",
        title="A",
        controller="user",
    )
    tab2 = BrowserTab(
        target="t2",
        events=asyncio.Queue(),
        url="https://b.com",
        title="B",
        controller="scout",
    )
    service._windows[("room-1", "scout")] = BrowserWindow(tabs=[tab1, tab2], active=0)  # pyright: ignore[reportPrivateUsage]

    section = browser_state_section(service, "room-1")
    assert (
        section
        == '[Browser: 2 open tabs — Tab 1: "A" (https://a.com), controller: user; Tab 2: "B" (https://b.com), controller: scout]'
    )


def test_browser_turn_hook_produces_turn_aid() -> None:
    service = BrowserService(AsyncMock())
    tab = BrowserTab(
        target="t1",
        events=asyncio.Queue(),
        url="https://example.com",
        title="Example",
        controller="scout",
    )
    service._windows[("room-1", "scout")] = BrowserWindow(tabs=[tab], active=0)  # pyright: ignore[reportPrivateUsage]

    hook = BrowserTurnHook(service=service)
    aid = hook.turn_aid(
        message="Hello",
        story_id=None,
        room_id="room-1",
        workspace_root=None,
        tool_names=frozenset({"browser"}),
    )
    assert aid is not None
    assert (
        aid.section
        == '[Browser: 1 open tab — Tab 1: "Example" (https://example.com), controller: scout]'
    )
    assert aid.reply_lines([], korean=False) == []


def test_browser_turn_hook_no_tabs_returns_none() -> None:
    service = BrowserService(AsyncMock())
    hook = BrowserTurnHook(service=service)
    assert (
        hook.turn_aid(
            message="Hello",
            story_id=None,
            room_id="room-empty",
            workspace_root=None,
            tool_names=frozenset({"browser"}),
        )
        is None
    )
    assert (
        hook.turn_aid(
            message="Hello",
            story_id=None,
            room_id=None,
            workspace_root=None,
            tool_names=frozenset({"browser"}),
        )
        is None
    )


def test_browser_turn_hook_passes_a_tool_step_context_through() -> None:
    """The agent calls `after_tool_step` on every lifecycle hook after every tool step (#2159).

    Without it every tool step of every clone turn raised AttributeError.

    Killed by: src/uclone_x/browser/turn.py :: return context
    Becomes: return context.model_copy()
    """
    context = ToolContext(
        agent_id="scout", session_id="s1", workspace_root=Path("."), story_id="story-1"
    )
    records: list[ToolExecutionRecord] = []

    assert (
        BrowserTurnHook(service=BrowserService(AsyncMock())).after_tool_step(records, context)
        is context
    )


_BROWSER_MODULES = (
    "uclone_x.browser.cdp",
    "uclone_x.browser.chrome",
    "uclone_x.browser.extension",
    "uclone_x.browser.link",
    "uclone_x.browser.live",
    "uclone_x.browser.service",
    "uclone_x.browser.tool",
    "uclone_x.browser.vision",
)


def test_browser_turn_hook_on_an_install_without_websockets_adds_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A base or `[cli]` install has no `websockets` (it is the `http` extra's), so no tab (#2159).

    Killed by: src/uclone_x/browser/turn.py :: return None  # no websockets, so no browser on this install
    Becomes: raise
    """
    # Every cached `websockets.*` too: a cached submodule would import without its parent.
    for name in [m for m in sys.modules if m.partition(".")[0] == "websockets"]:
        monkeypatch.setitem(sys.modules, name, None)
    monkeypatch.setitem(sys.modules, "websockets", None)
    for name in _BROWSER_MODULES:
        monkeypatch.delitem(sys.modules, name, raising=False)

    aid = BrowserTurnHook().turn_aid(
        message="Hello",
        story_id=None,
        room_id="room-1",
        workspace_root=None,
        tool_names=frozenset(),
    )

    assert aid is None
    assert "uclone_x.browser.tool" not in sys.modules
