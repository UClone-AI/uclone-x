"""A persona asked for directly -- not seated in a room -- calls its peers over A2A (#1659).

`a2a_call` is offered only to an agent holding a transport (#1558), and until #1659 only a
room's seats were given one. The Writer opened as a plain chat therefore had no way to ask
the Artist and could only promise to; the pictures it promised were never drawn.

Pinned here, through `AgentSessionManager.get_or_create_agent` -- the path a chat takes:

* a chat persona naming `a2a_peers` is given a transport, and `a2a_call` is offered to it;
* a persona naming no peers is given none, so `a2a_call` stays hidden from it;
* the Writer's call reaches the Artist, which runs its own tool, and the file it drew comes
  back in the Writer's tool result;
* the Artist answering that call is built saying nobody answers approvals, as the chat
  agent is, and refuses a call that needs one at once (#1692).

Offline: the model is a scripted mock and `generate_image` is a fake that writes bytes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, Field

from uclone_x.llm.connectors.mock import MockLLMConnector
from uclone_x.llm.models import LLMRequest, MessageRole, ModelResponse, ToolCallRequest
from uclone_x.tools.base import BaseTool
from uclone_x.tools.builtin.a2a import A2A_CALL_TOOL_NAME
from uclone_x.tools.models import ToolContext
from uclone_x.tools.registry import create_default_registry
from uclone_x.ui.app import AgentSessionManager


class _ImageParams(BaseModel):
    prompt: str = Field(default="")


class _FakeImage(BaseTool[_ImageParams]):
    """Stands in for `generate_image`: writes a file under the workspace and names it."""

    name = "generate_image"
    description = "Generates an image."
    writes_files = True

    def __init__(self, workspace: Path) -> None:
        super().__init__()
        self.workspace = workspace
        self.runs = 0

    def run(self, params: _ImageParams, context: ToolContext) -> dict[str, Any]:
        self.runs += 1
        target = self.workspace / "artifacts" / "images" / "hero.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"\x89PNG fake")
        return {"path": "artifacts/images/hero.png"}


class _ByOfferedTool(MockLLMConnector):
    """One model for both personas: it calls whichever of the scripted tools it is offered.

    The Writer is offered `a2a_call` and the Artist `generate_image`, so the same connector
    plays each part. A call is made once, until its own result is in the request.
    """

    def __init__(self) -> None:
        super().__init__(default_response="Done.")

    async def generate(self, request: LLMRequest) -> ModelResponse:
        offered = {tool.name for tool in request.tools}
        spent = {m.tool_call_id for m in request.messages if m.role is MessageRole.TOOL}
        if A2A_CALL_TOOL_NAME in offered:
            self._tool_calls = [
                ToolCallRequest(
                    id="w1",
                    name=A2A_CALL_TOOL_NAME,
                    arguments={"agent": "artist", "task": "Draw the hero."},
                )
            ]
        elif "generate_image" in offered:
            self._tool_calls = [
                ToolCallRequest(id="d1", name="generate_image", arguments={"prompt": "hero"})
            ]
        else:
            self._tool_calls = []
        self._tool_calls = [c for c in self._tool_calls if c.id not in spent]
        return await super().generate(request)


def _manager(tmp_path: Path, llm: MockLLMConnector, image: _FakeImage) -> AgentSessionManager:
    # The real registry, so the built-in personas load, with the image model swapped out.
    tools = create_default_registry(workspace_root=tmp_path, enable_mcp=False)
    tools.unregister(image.name)
    tools.register(image)
    return AgentSessionManager(
        storage_dir=tmp_path / "sessions", llm=llm, tools=tools, workspace_dir=tmp_path
    )


def _offered(agent: Any) -> set[str]:
    return {tool.name for tool in agent.available_tools()}


@pytest.mark.asyncio
async def test_a_chat_writer_is_offered_a2a_call(tmp_path: Path) -> None:
    """The Writer opened as a chat can ask the Artist: it holds a transport and the tool.

    Killed by: src/uclone_x/agent/clone_builder.py :: and persona_def.a2a_peers
    Becomes: and not persona_def.a2a_peers
    """
    manager = _manager(tmp_path, MockLLMConnector(), _FakeImage(tmp_path))

    writer = await manager.get_or_create_agent("writer")

    assert writer.a2a_transport is not None
    assert A2A_CALL_TOOL_NAME in _offered(writer)


@pytest.mark.asyncio
async def test_a_chat_persona_with_no_peers_gets_no_transport(tmp_path: Path) -> None:
    """The Artist names no peers, so it is given no one to call and no `a2a_call`."""
    manager = _manager(tmp_path, MockLLMConnector(), _FakeImage(tmp_path))

    artist = await manager.get_or_create_agent("artist")

    assert artist.a2a_transport is None
    assert A2A_CALL_TOOL_NAME not in _offered(artist)


@pytest.mark.asyncio
async def test_a_chat_writers_call_is_done_by_the_artist(tmp_path: Path) -> None:
    """Acceptance (#1659): the call is dispatched and executed, not left as a promise --
    the Artist's own tool runs once and the file comes back in the Writer's tool result.

    Killed by: src/uclone_x/agent/clone_builder.py :: personas=persona.a2a_peers,
    Becomes: personas=(),
    """
    llm = _ByOfferedTool()
    image = _FakeImage(tmp_path)
    manager = _manager(tmp_path, llm, image)
    writer = await manager.get_or_create_agent("writer", session_id="sess_1659")

    turn = await writer.execute_turn("Draw me the hero.")

    assert image.runs == 1
    assert (tmp_path / "artifacts" / "images" / "hero.png").is_file()
    calls = [e for e in turn.tool_executions if e.tool_name == A2A_CALL_TOOL_NAME]
    assert len(calls) == 1
    output = calls[0].output
    assert isinstance(output, dict)
    assert output["paths"] == ["artifacts/images/hero.png"]


def test_clone_may_ask_every_other_built_in_persona() -> None:
    """Clone is the user's general partner: its `a2a_peers` names every other built-in
    persona, so a persona added later without a line there is caught here."""
    from uclone_x.agent.persona_registry import PersonaRegistry

    builtin = {p.name for p in PersonaRegistry(include_defaults=True).list_personas()}
    clone = PersonaRegistry(include_defaults=True).get_persona("clone")

    assert clone is not None
    assert set(clone.a2a_peers) == builtin - {"clone"}


@pytest.mark.asyncio
async def test_a_chat_clones_call_is_done_by_the_artist(tmp_path: Path) -> None:
    """Clone opened as a chat asks the Artist, and the Artist's own tool draws the picture.

    Killed by: src/uclone_x/personas/clone.yaml :: - artist
    Becomes: - nobody
    """
    llm = _ByOfferedTool()
    image = _FakeImage(tmp_path)
    manager = _manager(tmp_path, llm, image)
    clone = await manager.get_or_create_agent("clone", session_id="sess_clone_a2a")

    turn = await clone.execute_turn("Ask the artist to draw the hero.")

    calls = [e for e in turn.tool_executions if e.tool_name == A2A_CALL_TOOL_NAME]
    assert len(calls) == 1
    output = calls[0].output
    assert isinstance(output, dict)
    assert output["paths"] == ["artifacts/images/hero.png"]
    assert image.runs == 1


class _ApprovalImage(_FakeImage):
    """`generate_image` with an action that runs only once a person approves it."""

    approval_actions = frozenset({"draw"})


class _DrawsWithApproval(_ByOfferedTool):
    """As `_ByOfferedTool`, but the Artist's call is the one that needs approval."""

    async def generate(self, request: LLMRequest) -> ModelResponse:
        response = await super().generate(request)
        return response.model_copy(
            update={
                "tool_calls": tuple(
                    call.model_copy(update={"arguments": {**call.arguments, "action": "draw"}})
                    if call.name == "generate_image"
                    else call
                    for call in response.tool_calls
                )
            }
        )


@pytest.mark.asyncio
async def test_a_chat_writers_callee_refuses_an_approval_call_at_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The Artist a chat Writer calls is built from the desktop's host, which says nobody
    answers an approval request (owner decision 2026-09-26), so it refuses at once.

    Killed by: src/uclone_x/ui/app.py :: scope, live_host=lambda: dataclasses.replace(scope.host, llm=self._llm or llm)
    Becomes: scope, live_host=lambda: dataclasses.replace(scope.host, llm=self._llm or llm, approvals_answered=True)
    """
    from uclone_x.room.a2a_handlers import PersonaTaskHandler

    callees: list[Any] = []
    compose = PersonaTaskHandler._compose  # pyright: ignore[reportPrivateUsage]

    def recording(self: PersonaTaskHandler, *args: Any, **kwargs: Any) -> Any:
        callee = compose(self, *args, **kwargs)
        callees.append(callee)
        return callee

    monkeypatch.setattr(PersonaTaskHandler, "_compose", recording)
    image = _ApprovalImage(tmp_path)
    manager = _manager(tmp_path, _DrawsWithApproval(), image)
    writer = await manager.get_or_create_agent("writer", session_id="sess_1692")

    turn = await writer.execute_turn("Draw me the hero.")

    assert [c.agent_id for c in callees] == ["artist"]
    assert callees[0].approvals_answered is False
    assert image.runs == 0
    assert not (tmp_path / "artifacts" / "images").exists()
    calls = [e for e in turn.tool_executions if e.tool_name == A2A_CALL_TOOL_NAME]
    assert len(calls) == 1
    assert calls[0].error == "artist needed approval to use generate_image, so it did not run."
