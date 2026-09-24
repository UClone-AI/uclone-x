"""End-to-end: a tool server added in Settings connects, lends its tools, and can be removed.

The unit tests (`tests/unit/test_mcp_manager.py`, `tests/unit/test_tools_mcp_http.py`) pin the
manager, the routes and both transports. What only the assembled product can show is the loop
through the shipped bundle: the form reaches the route, a real local process is started and
answers, the row the page shows says so with the server's own tool, the tool is offered to
clones, and removing it takes the tool away again.

The server is a local command (`sys.executable` running a small script), so nothing leaves the
machine.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
from playwright.async_api import async_playwright

from tests.e2e.conftest import running_ui
from uclone_x.llm.connectors.mock import MockLLMConnector

pytestmark = pytest.mark.e2e

_ECHO_SERVER = r"""
import json, sys
for line in sys.stdin:
    req = json.loads(line)
    if "id" not in req:
        continue
    if req["method"] == "initialize":
        result = {"protocolVersion": "2024-11-05"}
    elif req["method"] == "tools/list":
        result = {"tools": [{"name": "echo", "description": "Says it back"}]}
    else:
        result = {"content": [{"type": "text", "text": "echoed"}]}
    sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": req["id"], "result": result}) + "\n")
    sys.stdout.flush()
"""


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    path = tmp_path / "workspace"
    path.mkdir()
    return path


@pytest.fixture
def server(tmp_path: Path, workspace: Path) -> Iterator[str]:
    llm = MockLLMConnector(default_model="mock-gpt-4o", default_response="ok")
    with running_ui(storage_dir=tmp_path / "sessions", llm=llm, workspace_dir=workspace) as url:
        yield url


@pytest.mark.asyncio
async def test_a_local_tool_server_added_in_settings_connects_and_can_be_removed(
    server: str, tmp_path: Path
) -> None:
    """Add a command server through the form; it connects, lends `echoer__echo`, and goes.

    Killed by: src/uclone_x/tools/mcp_manager.py :: self._registry.unregister(name)
    Becomes: pass
    """
    script = tmp_path / "echo_server.py"
    script.write_text(_ECHO_SERVER)

    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        try:
            page = await browser.new_page(viewport={"width": 1600, "height": 900})
            await page.goto(server, wait_until="networkidle")

            await page.get_by_test_id("open-settings").click()
            section = page.get_by_test_id("settings-mcp")
            await section.get_by_test_id("settings-mcp-empty").wait_for(timeout=10_000)

            await section.get_by_role("button", name="Add a server").click()
            form = section.get_by_test_id("mcp-add-form")
            await form.get_by_label("On this computer (command)").check()
            await page.locator("#mcp-name").fill("echoer")
            await page.locator("#mcp-command").fill(sys.executable)
            await page.locator("#mcp-args").fill(str(script))
            await form.get_by_role("button", name="Add server").click()

            row = section.get_by_test_id("mcp-server-echoer")
            await row.get_by_text("Connected · 1 tool").wait_for(timeout=20_000)
            assert await section.get_by_test_id("settings-mcp-empty").count() == 0

            await row.get_by_text("Show tools (1)").click()
            tools = row.get_by_role("list", name="Tools from echoer")
            await tools.get_by_text("echoer__echo").wait_for(timeout=5_000)

            personas = await page.request.get(f"{server}/api/personas")
            assert personas.ok, await personas.text()
            assert "echoer__echo" in await personas.text(), "clones were not offered the tool"

            await row.get_by_role("button", name="Remove echoer").click()
            await (
                row.get_by_test_id("mcp-remove-confirm")
                .get_by_role("button", name="Remove", exact=True)
                .click()
            )
            await section.get_by_test_id("settings-mcp-empty").wait_for(timeout=10_000)

            personas = await page.request.get(f"{server}/api/personas")
            assert "echoer__echo" not in await personas.text(), "the tool outlived its server"
        finally:
            await browser.close()
