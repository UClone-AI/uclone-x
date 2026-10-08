"""The `browser` tool's arguments and which tab a call reaches (design §3.10)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from tests.support.browser_budget import (
    BROWSER_DEFINITION_TOKEN_CEILING,
    browser_definition_tokens,
)
from uclone_x.browser.service import BrowserService, TabKey
from uclone_x.browser.tool import HELP_TEXT, BrowserParams, BrowserTool
from uclone_x.errors import PlainRefusalError
from uclone_x.tools.models import ToolContext


class _Recording(BrowserService):
    """Records which service method each call reaches, with its arguments."""

    def __init__(self) -> None:
        async def _no_link() -> Any:
            raise AssertionError("the tool must not reach Chrome in this test")

        super().__init__(_no_link)
        self.calls: list[tuple[str, TabKey, object]] = []

    async def open(self, key: TabKey, url: str) -> dict[str, Any]:
        self.calls.append(("open", key, url))
        return {}

    async def ask_user(
        self, key: TabKey, kind: str = "sign_in", message: str | None = None
    ) -> dict[str, Any]:
        self.calls.append(("ask_user", key, (kind, message)))
        return {}

    async def snapshot(self, key: TabKey) -> dict[str, Any]:
        self.calls.append(("snapshot", key, None))
        return {}

    async def read(self, key: TabKey, max_length: int) -> dict[str, Any]:
        self.calls.append(("read", key, max_length))
        return {}

    async def find(self, key: TabKey, query: str) -> dict[str, Any]:
        self.calls.append(("find", key, query))
        return {}

    async def click(self, key: TabKey, ref: str, **flags: bool) -> dict[str, Any]:
        self.calls.append(("click", key, (ref, flags)))
        return {}

    async def click_at(
        self,
        key: TabKey,
        x: int,
        y: int,
        *,
        double: bool = False,
        right: bool = False,
        hover: bool = False,
    ) -> dict[str, Any]:
        self.calls.append(
            ("click_at", key, (x, y, {"double": double, "right": right, "hover": hover}))
        )
        return {}

    async def type_text(
        self, key: TabKey, ref: str | None, text: str, **flags: bool
    ) -> dict[str, Any]:
        self.calls.append(("type", key, (ref, text, flags)))
        return {}

    async def press(self, key: TabKey, chord: str) -> dict[str, Any]:
        self.calls.append(("press", key, chord))
        return {}

    async def upload(self, key: TabKey, ref: str, files: Sequence[Path]) -> dict[str, Any]:
        self.calls.append(("upload", key, (ref, list(files))))
        return {}

    async def wait(self, key: TabKey, **what: object) -> dict[str, Any]:
        self.calls.append(("wait", key, what))
        return {}

    async def tab(self, key: TabKey, **what: object) -> dict[str, Any]:
        self.calls.append(("tab", key, what))
        return {}


def test_open_needs_a_url_and_find_needs_a_query() -> None:
    with pytest.raises(ValidationError, match="open needs a url"):
        BrowserParams(action="open")
    with pytest.raises(ValidationError, match="find needs a query"):
        BrowserParams(action="find", query="  ")
    with pytest.raises(ValidationError):
        BrowserParams(action="drag")  # pyright: ignore[reportArgumentType]


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"action": "click"}, "click needs a ref"),
        ({"action": "check", "ref": " "}, "check needs a ref"),
        ({"action": "type", "ref": "e1"}, "type needs text"),
        ({"action": "select", "ref": "e1", "options": []}, "select needs options"),
        ({"action": "press"}, "press needs a key"),
        ({"action": "upload", "ref": "e1"}, "upload needs paths"),
        ({"action": "wait"}, "wait needs one of text, ref or ms"),
        ({"action": "wait", "text": "Done", "ms": 500}, "wait needs one of text, ref or ms"),
    ],
)
def test_each_action_names_what_it_is_missing(arguments: dict[str, Any], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        BrowserParams(**arguments)


def test_waits_are_capped_at_ten_seconds() -> None:
    with pytest.raises(ValidationError):
        BrowserParams(action="wait", ms=10_001)
    assert BrowserParams(action="wait", ms=10_000).ms == 10_000


async def test_a_call_reaches_the_tab_of_its_conversation_and_clone() -> None:
    service = _Recording()
    tool = BrowserTool(service=service)
    in_room = ToolContext(agent_id="scout", session_id="s-1", room_id="room-1")
    alone = ToolContext(agent_id="scout", session_id="s-2")

    await tool.run(BrowserParams(action="open", url="https://example.com"), in_room)
    await tool.run(BrowserParams(action="read", max_length=500), in_room)
    await tool.run(BrowserParams(action="find", query="로그인"), alone)
    await tool.run(BrowserParams(action="snapshot"), alone)

    assert service.calls == [
        ("open", ("room-1", "scout"), "https://example.com"),
        ("read", ("room-1", "scout"), 500),
        ("find", ("s-2", "scout"), "로그인"),
        ("snapshot", ("s-2", "scout"), None),
    ]


async def test_actions_reach_the_service_with_their_arguments() -> None:
    service = _Recording()
    tool = BrowserTool(service=service)
    context = ToolContext(agent_id="scout", session_id="s-1")
    key = ("s-1", "scout")

    await tool.run(BrowserParams(action="click", ref="e3", double=True), context)
    await tool.run(BrowserParams(action="type", ref="e4", text="부산", submit=True), context)
    await tool.run(BrowserParams(action="press", key="Escape"), context)
    await tool.run(BrowserParams(action="wait", text="결과"), context)
    await tool.run(BrowserParams(action="tab", index=2, close=True), context)

    assert service.calls == [
        ("click", key, ("e3", {"double": True, "right": False, "hover": False})),
        ("type", key, ("e4", "부산", {"submit": True, "append": False})),
        ("press", key, "Escape"),
        ("wait", key, {"text": "결과", "ref": None, "ms": None}),
        ("tab", key, {"index": 2, "close": True}),
    ]


async def test_upload_hands_over_workspace_files_only(tmp_path: Path) -> None:
    service = _Recording()
    tool = BrowserTool(service=service)
    (tmp_path / "cv.pdf").write_bytes(b"%PDF")
    context = ToolContext(agent_id="scout", session_id="s-1", workspace_root=tmp_path)

    await tool.run(BrowserParams(action="upload", ref="e2", paths=["cv.pdf"]), context)
    assert service.calls == [
        ("upload", ("s-1", "scout"), ("e2", [(tmp_path / "cv.pdf").resolve()]))
    ]

    with pytest.raises(PlainRefusalError) as missing:
        await tool.run(BrowserParams(action="upload", ref="e2", paths=["nope.pdf"]), context)
    assert str(missing.value) == "There is no file nope.pdf in the workspace."

    with pytest.raises(PlainRefusalError) as outside:
        await tool.run(BrowserParams(action="upload", ref="e2", paths=["../etc/passwd"]), context)
    assert str(outside.value) == "../etc/passwd is outside the files this clone can read."

    bare = ToolContext(agent_id="scout", session_id="s-1")
    with pytest.raises(PlainRefusalError, match="no workspace"):
        await tool.run(BrowserParams(action="upload", ref="e2", paths=["cv.pdf"]), bare)


def test_the_tool_stays_usable_by_read_only_personas() -> None:
    # scout runs with write tools off; the browser must not count as one (§3.10).
    assert BrowserTool.writes_files is False


async def test_help_action_returns_documentation() -> None:
    """The help action returns human-readable documentation without reaching Chrome.

    Killed by: src/uclone_x/browser/tool.py :: if params.action == "help":
    Becomes: if False:
    """
    service = _Recording()
    tool = BrowserTool(service=service)
    context = ToolContext(agent_id="scout", session_id="s-1")

    result = await tool.run(BrowserParams(action="help"), context)
    assert isinstance(result, dict)
    assert result == {"help": HELP_TEXT}
    assert service.calls == []
    for action in ("open", "click", "type", "press", "scroll", "upload", "wait", "tab"):
        assert action in HELP_TEXT


def test_browser_tool_schema_and_description_size_budget() -> None:
    """The browser's advertised definition stays under its ceiling for 8K-window models
    (#2116, #2164).

    Counted in tokens the way the compactor counts a request's tools, not in characters,
    because tokens are what the 8K window is spent in. The 8K-window ingest test pads the
    browser to this same ceiling, so a schema that grows past it fails here, by name, and
    not there.

    Killed by: src/uclone_x/browser/tool.py :: description: str = "Real Chrome browser. Action 'help'."
    Becomes: description: str = "Real Chrome browser. Action 'help'." + "x" * 200
    """
    tokens = browser_definition_tokens(BrowserTool())
    assert tokens <= BROWSER_DEFINITION_TOKEN_CEILING, (
        f"the browser tool's advertised definition costs {tokens} tokens, over its ceiling "
        f"of {BROWSER_DEFINITION_TOKEN_CEILING}. Every request to an 8K-window model carries "
        f"it; make room by trimming the schema, or raise the ceiling in "
        f"tests/support/browser_budget.py and keep the 8K-window ingest test passing."
    )


def test_click_refuses_when_both_ref_and_coordinates_are_passed() -> None:
    """Killed by: src/uclone_x/browser/tool.py :: raise ValueError("click takes either ref or x, y, not both")
    Becomes: pass
    """
    with pytest.raises(ValidationError, match="click takes either ref or x, y, not both"):
        BrowserParams(action="click", ref="e1", x=10, y=20)


async def test_click_at_dispatches_with_flags() -> None:
    """Killed by: src/uclone_x/browser/tool.py :: double=params.double,  # coordinate click double
    Becomes: double=False,  # coordinate click double
    """
    service = _Recording()
    tool = BrowserTool(service=service)
    context = ToolContext(agent_id="scout", session_id="s-1")
    await tool.run(
        BrowserParams(action="click", x=10, y=20, double=True, right=True, hover=True), context
    )
    assert service.calls == [
        ("click_at", ("s-1", "scout"), (10, 20, {"double": True, "right": True, "hover": True}))
    ]


async def test_ask_user_hands_the_reason_in_text_to_the_service() -> None:
    """`ask_user` takes its reason in `text`: no field of its own, for the 8K budget (#2159).

    Killed by: src/uclone_x/browser/tool.py :: return await service.ask_user(key, message=params.text)
    Becomes: return await service.ask_user(key)
    """
    service = _Recording()
    tool = BrowserTool(service=service)
    context = ToolContext(agent_id="scout", session_id="s-1")

    await tool.run(BrowserParams(action="ask_user", text="Please sign in to the bank"), context)

    assert service.calls == [
        ("ask_user", ("s-1", "scout"), ("sign_in", "Please sign in to the bank"))
    ]


async def test_read_without_a_max_length_reads_up_to_the_stated_default() -> None:
    """`max_length` advertises no default (8K budget, #2159); `read` still applies 20000.

    Killed by: src/uclone_x/browser/tool.py :: READ_MAX_LENGTH: Final = 20000
    Becomes: READ_MAX_LENGTH: Final = 2000
    """
    service = _Recording()
    tool = BrowserTool(service=service)
    context = ToolContext(agent_id="scout", session_id="s-1")

    await tool.run(BrowserParams(action="read"), context)

    assert service.calls == [("read", ("s-1", "scout"), 20000)]
