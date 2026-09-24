"""End-to-end: an agent made in Settings is saved, listed, and answers as itself (#892).

The unit tests (`tests/unit/test_persona_write.py`) pin the route and the file. What only the
assembled product can show is the whole loop through the shipped bundle: the Settings form
reaches the route, the file lands in the workspace the loader reads, the list the page shows
afterwards names it, and the next turn addressed to it is sent with its instructions.

The second test is #1167's browser acceptance: the shipped `critic` persona, whose YAML sets
both capability flags to `false`, starts no helper and writes no file even when the model asks
for both on every turn.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from playwright.async_api import async_playwright

from tests.e2e.conftest import running_ui
from uclone_x.llm.connectors.mock import MockLLMConnector
from uclone_x.llm.models import LLMRequest, MessageRole, ModelResponse, ToolCallRequest

pytestmark = pytest.mark.e2e

_PROMPT = "You survey the site and report what you map."


class _RecordingConnector(MockLLMConnector):
    """Answers normally, and keeps every request it was sent."""

    def __init__(self) -> None:
        super().__init__(default_model="mock-gpt-4o", default_response="Surveyed.")
        self.requests: list[LLMRequest] = []

    async def generate(self, request: LLMRequest) -> ModelResponse:
        self.requests.append(request)
        return await super().generate(request)


@pytest.fixture
def llm() -> _RecordingConnector:
    return _RecordingConnector()


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    path = tmp_path / "workspace"
    path.mkdir()
    return path


@pytest.fixture
def server(tmp_path: Path, workspace: Path, llm: _RecordingConnector) -> Iterator[str]:
    with running_ui(storage_dir=tmp_path / "sessions", llm=llm, workspace_dir=workspace) as url:
        yield url


@pytest.mark.asyncio
async def test_an_agent_created_in_settings_is_listed_and_answers_with_its_instructions(
    server: str, workspace: Path, llm: _RecordingConnector
) -> None:
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        try:
            page = await browser.new_page(viewport={"width": 1600, "height": 900})
            await page.goto(server, wait_until="networkidle")

            await page.get_by_test_id("open-settings").click()
            section = page.get_by_test_id("settings-personas")
            await section.get_by_role("button", name="New clone").click()
            await page.locator("#persona-name").fill("surveyor")
            await page.locator("#persona-role").fill("Site Surveyor")
            await page.locator("#persona-prompt").fill(_PROMPT)
            await page.get_by_test_id("persona-editor").get_by_role("button", name="Save").click()

            await section.get_by_test_id("persona-row-surveyor").wait_for(timeout=10_000)
            await section.get_by_role("status").wait_for(timeout=5_000)

            written = workspace / ".uclone" / "personas" / "surveyor.yaml"
            assert written.is_file(), "the save did not land in the directory the loader reads"

            turn = await page.request.post(
                f"{server}/api/turn",
                data={"message": "go", "agent_id": "surveyor", "session_id": "sess_e2e_892"},
            )
            assert turn.ok, await turn.text()
        finally:
            await browser.close()

    assert llm.requests, "the turn never reached the model"
    systems = [m.content or "" for m in llm.requests[-1].messages if m.role is MessageRole.SYSTEM]
    assert systems and systems[0].startswith(_PROMPT), systems


class _DelegatingConnector(MockLLMConnector):
    """Asks to delegate and to write on every turn, whether or not it was offered either.

    A small model asks for a tool it was never given; the point of #1167 is that asking is
    not enough. Recording the requests is what lets the assertion be "no helper was started"
    rather than "the reply did not mention one".
    """

    def __init__(self) -> None:
        super().__init__(
            default_model="mock-gpt-4o",
            default_response="Reviewed.",
            tool_calls=(
                ToolCallRequest(
                    id="call_delegate",
                    name="delegate_subagent",
                    arguments={"role": "helper", "goal": "help", "prompt": "go"},
                ),
                ToolCallRequest(
                    id="call_write",
                    name="file_write",
                    arguments={"path": "verdict.md", "content": "rewritten"},
                ),
            ),
        )
        self.requests: list[LLMRequest] = []

    async def generate(self, request: LLMRequest) -> ModelResponse:
        self.requests.append(request)
        return await super().generate(request)


@pytest.mark.asyncio
async def test_the_shipped_guardian_starts_no_helper_and_writes_no_file_through_the_browser(
    tmp_path: Path,
) -> None:
    """Acceptance 2 of #1167, against the assembled product and the shipped `guardian`.

    `guardian`'s YAML has declared `enable_write_tools: false` and `enable_subagent_tools:
    false` all along, and until #1167 both were decoration: the delegation tool sits in the
    default registry, and an empty `allowed_tools` hands an agent the whole registry. The
    issue named the `default` persona; the owner ruling corrects that to `guardian`, because
    there is no runtime `default` persona
    ().

    The browser is what makes this the product's answer and not the route's: the page is
    loaded from the shipped bundle, `guardian` is chosen in the rail's Clones list, and the turn
    is the one the page sends.

    Verified by hand: `spawn_subagent`'s own guard is *not* what pins this test --
    `_capability_refusal` withholds `delegate_subagent` in `_execute_single_tool` before
    that guard is ever reached, so neutering it leaves this test green. The refusal branch
    below is the line that actually stands between this persona and both capabilities.

    Killed by: src/uclone_x/agent/base.py :: elif (refusal := self._capability_refusal(tool_inst)) is not None:
    Becomes: elif False:
    """
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    llm = _DelegatingConnector()

    with running_ui(storage_dir=tmp_path / "sessions", llm=llm, workspace_dir=workspace) as server:
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(headless=True)
            try:
                page = await browser.new_page(viewport={"width": 1600, "height": 900})
                await page.goto(server, wait_until="networkidle")

                turn = await page.request.post(
                    f"{server}/api/turn",
                    data={
                        "message": "review this",
                        "agent_id": "guardian",
                        "session_id": "sess_e2e_1167",
                    },
                )
                assert turn.ok, await turn.text()
            finally:
                await browser.close()

    assert llm.requests, "the turn never reached the model"

    # No helper: a sub-agent's first turn carries the prompt `spawn_subagent` writes, so its
    # absence from every request the model ever saw is the whole claim.
    started = [
        m.content or ""
        for request in llm.requests
        for m in request.messages
        if m.role is MessageRole.SYSTEM and "specialized sub-agent" in (m.content or "")
    ]
    assert started == [], started

    # No write: the tool the model asked for put no bytes on the host.
    assert not (workspace / "verdict.md").exists()

    # Neither tool was even offered, so the refusal is not the only thing standing between
    # the model and the capability.
    offered = {t.name for t in llm.requests[-1].tools}
    assert "delegate_subagent" not in offered and "file_write" not in offered, sorted(offered)


@pytest.mark.asyncio
async def test_generate_fills_the_instructions_from_the_model_and_can_be_undone(
    server: str, llm: _RecordingConnector
) -> None:
    """The instructions-draft button reaches the connected model, and says it did.

    It used to fill the field from a fixed English template while labelled "Generate with
    AI", and replaced whatever the person had typed with no way back.

    Killed by: src/uclone_x/ui/app.py :: return {"system_prompt": drafted, "source": "llm", "model": response.model_name}
    Becomes: raise ValueError("no model")
    """
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        try:
            page = await browser.new_page(viewport={"width": 1600, "height": 900})
            await page.goto(server, wait_until="networkidle")
            await page.get_by_test_id("room-composer").wait_for(timeout=20_000)

            await page.get_by_role("button", name="New clone").first.click()
            await page.locator("#persona-description").fill("Maps the site and reports on it.")
            instructions = page.locator("#persona-prompt")
            await instructions.fill("My own words.")
            await page.get_by_test_id("persona-synthesize-prompt").click()

            notice = page.get_by_test_id("persona-draft-notice")
            await notice.wait_for(timeout=10_000)
            assert "Drafted by mock-gpt-4o" in await notice.inner_text()
            assert await instructions.input_value() == "Surveyed."

            await page.get_by_test_id("persona-draft-undo").click()
            assert await instructions.input_value() == "My own words."
        finally:
            await browser.close()

    drafts = [
        r for r in llm.requests if "Description: Maps the site" in (r.messages[-1].content or "")
    ]
    assert len(drafts) == 1, [r.messages[-1].content for r in llm.requests]
