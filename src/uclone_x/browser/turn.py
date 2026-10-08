"""Turn aid for the browser: gives every turn the conversation's browser state (design §3.4)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from uclone_x.agent.models import ToolExecutionRecord
    from uclone_x.browser.service import BrowserService
    from uclone_x.tools.models import ToolContext

__all__ = ["BrowserTurnAid", "BrowserTurnHook", "browser_state_section"]


def browser_state_section(service: BrowserService, conversation: str) -> str | None:
    """One-line summary of open browser tabs for the conversation (design §3.4)."""
    tabs_desc: list[str] = []
    tab_num = 1
    for (r_id, cl), win in service._windows.items():  # pyright: ignore[reportPrivateUsage]
        if r_id == conversation:
            for tab in win.tabs:
                ctrl = (
                    "user"
                    if tab.controller == "user"
                    else (tab.controller if tab.controller != "clone" else cl)
                )
                url = tab.url or "about:blank"
                title = tab.title.strip() if tab.title else url
                tabs_desc.append(f'Tab {tab_num}: "{title}" ({url}), controller: {ctrl}')
                tab_num += 1
    if not tabs_desc:
        return None
    if len(tabs_desc) == 1:
        return f"[Browser: 1 open tab — {tabs_desc[0]}]"
    joined = "; ".join(tabs_desc)
    return f"[Browser: {len(tabs_desc)} open tabs — {joined}]"


class BrowserTurnAid:
    """The turn aid holding the browser state section (`TurnAidProtocol`)."""

    def __init__(self, section_text: str) -> None:
        self._section = section_text

    @property
    def section(self) -> str:
        return self._section

    def reply_lines(self, records: Sequence[object], *, korean: bool) -> list[str]:
        del records, korean
        return []


def _default_service() -> BrowserService | None:
    """The Core's browser service, or `None` on an install that cannot run a browser.

    Every clone composes `BrowserTurnHook` (`agent/clone_builder.py`), so this module loads
    on a base or `[cli]` install too. The browser's socket library, `websockets`, comes with
    the `http` extra; without it no tab can be open, so there is no state to report (#2159).
    """
    try:
        from uclone_x.browser.tool import default_browser_service
    except ModuleNotFoundError as exc:
        if (exc.name or "").partition(".")[0] != "websockets":
            raise
        return None  # no websockets, so no browser on this install
    return default_browser_service()


class BrowserTurnHook:
    """The lifecycle hook that gives a turn the conversation's browser state (`TurnAidHookProtocol`).

    It moves nothing between steps: `after_tool_step` returns the context as it is.
    """

    def __init__(self, service: BrowserService | None = None) -> None:
        self._service = service

    def after_tool_step(
        self, records: Sequence[ToolExecutionRecord], context: ToolContext
    ) -> ToolContext:
        """`context`, unchanged."""
        del records
        return context

    def turn_aid(
        self,
        *,
        message: str,
        story_id: str | None,
        room_id: str | None,
        workspace_root: Path | None,
        tool_names: frozenset[str],
    ) -> BrowserTurnAid | None:
        del message, story_id, workspace_root, tool_names
        if room_id is None:
            return None
        service = self._service or _default_service()
        if service is None:
            return None
        section = browser_state_section(service, room_id)
        if section is None:
            return None
        return BrowserTurnAid(section)
