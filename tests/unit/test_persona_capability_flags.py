# pyright: reportPrivateUsage=false
"""A persona's `enable_write_tools` and `enable_subagent_tools` are enforced (#1167).

Until #1167 both flags were accepted, stored and copied onto every agent's config, and
nothing read them: the shipped `critic` persona (`false`, `false`) was offered and ran
`file_write`, `bash_run` and `delegate_subagent`. What these tests pin is the side effect of
the rule, never a model's prose: a refused tool raises or yields an error record, and the
tool's own code does not run.

The classification and the precedence are the owner-delegated ruling recorded at


* a tool writes files when it declares `writes_files`, and a tool that does not declare it
  (an MCP server's tool, a `LocalTool` handler, an undeclared `BaseTool` subclass) is
  treated as writing;
* the flags only remove tools. A `false` from the agent's config (the operator's, or the
  parent's for a sub-agent) or from the persona in force wins, and naming a tool in
  `allowed_tools` does not lift either flag;
* a sub-agent inherits its parent's effective flags.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel, Field

from uclone_x.agent.base import BaseAgent
from uclone_x.agent.models import AgentConfig, PersonaDefinition
from uclone_x.core.provenance import Provenance
from uclone_x.llm.connectors.mock import MockLLMConnector
from uclone_x.llm.models import LLMRequest, MessageRole, ModelResponse, ToolCallRequest
from uclone_x.memory.tools import (
    QueryMemoryFactsTool,
    RecordMemoryFactTool,
    RetractMemoryFactTool,
)
from uclone_x.tools.base import BaseTool, tool_writes_files
from uclone_x.tools.builtin.comfy_image_tool import ComfyImageGenTool
from uclone_x.tools.builtin.shell import BashRunTool
from uclone_x.tools.builtin.skill_loader import LoadSkillTool
from uclone_x.tools.builtin.subagent import SubagentDelegationTool
from uclone_x.tools.client import MCPTool
from uclone_x.tools.models import ToolContext, ToolResult, ToolResultStatus
from uclone_x.tools.protocols import MCPClientProtocol
from uclone_x.tools.registry import LocalTool, ToolRegistry, create_default_registry
from uclone_x.ui.app import AgentSessionManager, create_ui_app


class _TextParams(BaseModel):
    text: str = Field(default="", description="Anything")


class _Writer(BaseTool[_TextParams]):
    """Declares that it writes. Counts its runs, so "not executed" is observed, not assumed."""

    name = "scribble"
    description = "Writes a file."
    writes_files = True

    def __init__(self) -> None:
        super().__init__()
        self.runs = 0

    def run(self, params: _TextParams, context: ToolContext) -> str:
        self.runs += 1
        return "written"


class _Reader(BaseTool[_TextParams]):
    name = "peek"
    description = "Reads something."
    writes_files = False

    def __init__(self) -> None:
        super().__init__()
        self.runs = 0

    def run(self, params: _TextParams, context: ToolContext) -> str:
        self.runs += 1
        return "read"


class _Undeclared(BaseTool[_TextParams]):
    """A `BaseTool` subclass that says nothing about writing."""

    name = "mystery"
    description = "Does something."

    def __init__(self) -> None:
        super().__init__()
        self.runs = 0

    def run(self, params: _TextParams, context: ToolContext) -> str:
        self.runs += 1
        return "done"


class _RecordingConnector(MockLLMConnector):
    """Answers as a `MockLLMConnector` does, and keeps every request it was sent."""

    def __init__(self, tool_calls: tuple[ToolCallRequest, ...] = ()) -> None:
        super().__init__(default_response="ok", tool_calls=tool_calls)
        self.requests: list[LLMRequest] = []

    async def generate(self, request: LLMRequest) -> ModelResponse:
        self.requests.append(request)
        return await super().generate(request)


def _agent(
    *tools: object,
    persona: PersonaDefinition | None = None,
    llm: MockLLMConnector | None = None,
    **config: Any,
) -> BaseAgent:
    registry = ToolRegistry()
    for tool in tools:
        registry.register(cast(Any, tool))
    agent = BaseAgent(
        config=AgentConfig(agent_id="flagged", name="Flagged", **config),
        tools=registry,
        llm=llm,
    )
    if persona is not None:
        agent.define_persona(persona)
        agent.persona = persona.name
    return agent


def _persona(**flags: Any) -> PersonaDefinition:
    return PersonaDefinition(name="careful", role="Careful", system_prompt="Be careful.", **flags)


# --- enable_write_tools ------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("allowed_tools", [(), ("scribble", "peek")])
async def test_write_flag_off_refuses_a_writing_tool_by_raising_and_never_runs_it(
    allowed_tools: tuple[str, ...],
) -> None:
    """Refused on `execute_tool_call`'s `PermissionError` path, whether or not a list names it.

    An empty `allowed_tools` is "no restriction", and a list naming the tool is the
    operator's permission for the *name*: neither lifts the flag, which is a separate
    restriction a tool must also pass.

    Killed by: src/uclone_x/agent/base.py :: raise PermissionError(refusal)
    Becomes: pass
    """
    writer, reader = _Writer(), _Reader()
    agent = _agent(writer, reader, enable_write_tools=False, allowed_tools=allowed_tools)

    with pytest.raises(PermissionError, match="enable_write_tools"):
        await agent.execute_tool_call("scribble", {"text": "x"})

    assert writer.runs == 0
    assert (await agent.execute_tool_call("peek", {"text": "x"})).status is (
        ToolResultStatus.SUCCESS
    )
    assert reader.runs == 1


@pytest.mark.asyncio
async def test_write_flag_off_does_not_offer_a_writing_tool_to_the_model() -> None:
    """The advertisement drops what the flags refuse, and keeps what they do not.

    Killed by: src/uclone_x/agent/base.py :: if self._capability_refusal(t) is not None:
    Becomes: if False:
    """
    llm = _RecordingConnector()
    agent = _agent(_Writer(), _Reader(), llm=llm, enable_write_tools=False)

    await agent.execute_turn("go")

    assert [t.name for t in llm.requests[-1].tools] == ["peek"]


@pytest.mark.asyncio
async def test_a_persona_s_false_wins_over_the_operator_s_true_and_its_own_tool_list() -> None:
    """The persona in force can withhold writes the agent's config allows.

    The persona names the writing tool in its own `allowed_tools`, which is the strongest
    permission a persona can give a name, and it is still refused.

    Killed by: src/uclone_x/agent/base.py :: persona_allows = persona is None or persona.enable_write_tools
    Becomes: persona_allows = True
    """
    writer = _Writer()
    agent = _agent(writer, persona=_persona(allowed_tools=("scribble",), enable_write_tools=False))
    assert agent.config.enable_write_tools is True

    with pytest.raises(PermissionError, match="enable_write_tools"):
        await agent.execute_tool_call("scribble", {"text": "x"})
    assert writer.runs == 0


@pytest.mark.asyncio
async def test_the_operator_s_false_wins_over_a_persona_s_true() -> None:
    """A persona cannot widen what the agent's own config withheld.

    Killed by: src/uclone_x/agent/base.py :: operator_allows = self._config.enable_write_tools
    Becomes: operator_allows = True
    """
    writer = _Writer()
    agent = _agent(writer, persona=_persona(enable_write_tools=True), enable_write_tools=False)

    with pytest.raises(PermissionError, match="enable_write_tools"):
        await agent.execute_tool_call("scribble", {"text": "x"})
    assert writer.runs == 0


# --- what counts as writing: undeclared means writing ------------------------------------


@pytest.mark.asyncio
async def test_an_undeclared_base_tool_subclass_is_treated_as_writing() -> None:
    """A tool that does not say it is read-only is refused, not trusted.

    Killed by: src/uclone_x/tools/base.py :: writes_files: ClassVar[bool] = True
    Becomes: writes_files: ClassVar[bool] = False
    """
    mystery = _Undeclared()
    agent = _agent(mystery, enable_write_tools=False)

    with pytest.raises(PermissionError, match="enable_write_tools"):
        await agent.execute_tool_call("mystery", {"text": "x"})
    assert mystery.runs == 0


@pytest.mark.asyncio
async def test_a_local_tool_handler_is_treated_as_writing_unless_declared_read_only() -> None:
    """A `LocalTool` wraps an opaque handler, so it defaults to writing.

    Killed by: src/uclone_x/tools/registry.py :: writes_files: bool = True,
    Becomes: writes_files: bool = False,
    """
    calls: list[str] = []

    def handler_for(name: str) -> Any:
        async def handle(params: dict[str, Any], context: ToolContext) -> ToolResult:
            calls.append(name)
            return ToolResult(success=True, output=name, provenance=Provenance.primary(name))

        return handle

    opaque = LocalTool(name="opaque", description="?", handler=handler_for("opaque"))
    declared = LocalTool(
        name="declared", description="?", handler=handler_for("declared"), writes_files=False
    )
    agent = _agent(opaque, declared, enable_write_tools=False)

    with pytest.raises(PermissionError, match="enable_write_tools"):
        await agent.execute_tool_call("opaque", {})
    await agent.execute_tool_call("declared", {})
    assert calls == ["declared"]


class _CountingMCPClient:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def call_tool(
        self, name: str, arguments: dict[str, Any], context: ToolContext
    ) -> ToolResult:
        self.calls.append(name)
        return ToolResult(success=True, output="called", provenance=Provenance.primary(name))


@pytest.mark.asyncio
async def test_an_mcp_server_s_tool_is_refused_when_writes_are_off() -> None:
    """Fail closed: this proxy cannot know what a server's tool does to the host.

    Killed by: src/uclone_x/tools/client.py :: writes_files: ClassVar[bool] = True
    Becomes: writes_files: ClassVar[bool] = False
    """
    client = _CountingMCPClient()
    tool = MCPTool(
        name="remote_thing",
        description="A tool some server offers.",
        parameters_schema={"type": "object", "properties": {}},
        client=cast(MCPClientProtocol, client),
    )
    agent = _agent(tool, enable_write_tools=False)

    with pytest.raises(PermissionError, match="enable_write_tools"):
        await agent.execute_tool_call("remote_thing", {})
    assert client.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["bash_run", "run_command"])
async def test_the_shell_is_a_writing_tool_under_both_of_its_names(
    name: str, tmp_path: Path
) -> None:
    """`echo > f` writes a file, so the shell is refused -- under either registered name.

    The approval hook's name prefixes catch `run_command` and miss `bash_run`, the same
    class; a capability read from the tool cannot disagree with itself that way.

    Killed by: src/uclone_x/tools/builtin/shell.py :: writes_files: ClassVar[bool] = True
    Becomes: writes_files: ClassVar[bool] = False
    """
    marker = tmp_path / "marker"
    agent = _agent(BashRunTool(name=name), enable_write_tools=False, workspace_dir=str(tmp_path))

    with pytest.raises(PermissionError, match="enable_write_tools"):
        await agent.execute_tool_call(name, {"command": f"touch {marker}"})
    assert not marker.exists()


def test_the_ruling_s_classification_of_every_shipped_tool() -> None:
    """Which shipped tools write, as ruled -- a change to this set is a decision, not a refactor.

    The shells, both file writers, both image generators and the package installer can put
    bytes on the host. The memory tools, the plan and the skill loader write only the
    agent's own state, and the rest read.

    Killed by: src/uclone_x/tools/builtin/filesystem.py :: writes_files: ClassVar[bool] = True  # modifies a file on the host in place (#1167)
    Becomes: writes_files: ClassVar[bool] = False
    """
    registry = create_default_registry(enable_mcp=False)
    classified = {t.name: tool_writes_files(t) for t in registry.list_tools()}

    assert classified == {
        "file_read": False,
        "file_write": True,
        "file_edit": True,
        "file_search": False,
        "directory_list": False,
        "bash_run": True,
        "run_command": True,
        "web_fetch": False,
        "web_search": False,
        "generate_image": True,
        "character_sheet": True,
        "story_library": True,
        "muse_spark": False,
        "story_outline": True,
        "story_manuscript": True,
        # `propose` and `apply` write the story's proposals and codex (#1557).
        "story_codex": True,
        "story_context": False,
        "story_audit": False,
        "install_package": True,
        "update_plan": False,
        "delegate_subagent": False,
        # A peer asked through it writes with its own tools, and those files are the
        # call's result; declared, so a persona that may not write cannot have a peer
        # write for it (#1558).
        "a2a_call": True,
        "tool_result_read": False,
    }
    agent_local = {
        cls.__name__: tool_writes_files(cls)
        for cls in (
            RecordMemoryFactTool,
            RetractMemoryFactTool,
            QueryMemoryFactsTool,
            LoadSkillTool,
            ComfyImageGenTool,
        )
    }
    assert agent_local == {
        "RecordMemoryFactTool": False,
        "RetractMemoryFactTool": False,
        "QueryMemoryFactsTool": False,
        "LoadSkillTool": False,
        "ComfyImageGenTool": True,
    }


# --- enable_subagent_tools ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_subagent_flag_off_refuses_delegate_subagent_and_starts_no_agent() -> None:
    """The delegation tool is refused on the `PermissionError` path, and no child exists.

    Killed by: src/uclone_x/agent/base.py :: if tool_spawns_subagents(tool) and not self.subagent_tools_enabled:
    Becomes: if False:
    """
    llm = _RecordingConnector()
    agent = _agent(SubagentDelegationTool(), llm=llm, enable_subagent_tools=False)
    args = {"role": "helper", "goal": "help", "prompt": "go"}

    with pytest.raises(PermissionError, match="enable_subagent_tools"):
        await agent.execute_tool_call("delegate_subagent", args)
    assert llm.requests == []


@pytest.mark.asyncio
async def test_spawn_subagent_itself_refuses_under_a_persona_that_forbids_sub_agents() -> None:
    """Refused where agents are made, so a caller that bypasses the tool is stopped too.

    Killed by: src/uclone_x/agent/base.py :: if not self.subagent_tools_enabled:
    Becomes: if False:
    """
    agent = _agent(_Reader(), persona=_persona(enable_subagent_tools=False))
    assert agent.config.enable_subagent_tools is True

    with pytest.raises(PermissionError, match="enable_subagent_tools"):
        await agent.spawn_subagent("helper", "help")


# --- inheritance -------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_child_inherits_its_parent_s_persona_write_restriction() -> None:
    """A child has no persona of its own, so it carries the parent's effective flag.

    Before #1167 the child's config took the `AgentConfig` default, `True`, so a sub-agent
    of a read-only persona could write: the delegation was an escalation.

    Killed by: src/uclone_x/agent/base.py :: enable_write_tools=self.write_tools_enabled,
    Becomes: enable_write_tools=True,
    """
    writer = _Writer()
    parent = _agent(writer, persona=_persona(enable_write_tools=False, enable_subagent_tools=True))

    child = await parent.spawn_subagent("helper", "help")

    assert child.write_tools_enabled is False
    with pytest.raises(PermissionError, match="enable_write_tools"):
        await child.execute_tool_call("scribble", {"text": "x"})
    assert writer.runs == 0


# --- the UI path -------------------------------------------------------------------------


_EDITABLE_KEYS = (
    "name",
    "role",
    "description",
    "system_prompt",
    "append_default_prompt",
    "allowed_tools",
    "model_name",
    "model_tier",
    "temperature",
    "max_tokens",
    "enable_write_tools",
    "enable_subagent_tools",
)


def test_the_shipped_guardian_neither_writes_nor_delegates_until_its_settings_allow_it(
    tmp_path: Path,
) -> None:
    """Through `/api/turn`, with the default tools and the shipped `guardian` persona.

    The model asks for `file_write` and `delegate_subagent` whether or not it was offered
    them, which is what a small model does. Under guardian's shipped flags (`false`, `false`)
    the file is not written and no helper agent is started -- no request carries a
    sub-agent's prompt. Turning on "Allow file-writing tools" from the editor
    (`PUT /api/personas/guardian`) reaches the running agent, and the next turn writes.

    Killed by: src/uclone_x/agent/base.py :: elif (refusal := self._capability_refusal(tool_inst)) is not None:
    Becomes: elif False:
    """
    workspace = tmp_path / "ws"
    workspace.mkdir()
    target = workspace / "verdict.md"
    calls = (
        ToolCallRequest(
            id="call_write",
            name="file_write",
            arguments={"path": "verdict.md", "content": "rewritten"},
        ),
        ToolCallRequest(
            id="call_delegate",
            name="delegate_subagent",
            arguments={"role": "helper", "goal": "help", "prompt": "go"},
        ),
    )
    llm = _RecordingConnector(calls)
    app = create_ui_app(
        static_dir=workspace / "static",
        storage_dir=workspace / "sessions",
        llm=llm,
        workspace_dir=workspace,
    )
    manager = cast(AgentSessionManager, app.state.session_manager)

    def a_helper_was_started() -> bool:
        return any(
            "specialized sub-agent" in (m.content or "")
            for request in llm.requests
            for m in request.messages
            if m.role is MessageRole.SYSTEM
        )

    with TestClient(app) as client:
        turn = {"message": "review this", "agent_id": "guardian", "session_id": "s1"}
        assert client.post("/api/turn", json=turn).status_code == 200
        guardian = manager.get_agent("guardian", "s1")
        assert guardian is not None
        assert (guardian.write_tools_enabled, guardian.subagent_tools_enabled) == (False, False)

        assert not target.exists()
        assert not a_helper_was_started()

        listed = {p["name"]: p for p in client.get("/api/personas").json()["personas"]}
        edit = {k: v for k, v in listed["guardian"].items() if k in _EDITABLE_KEYS}
        edit["enable_write_tools"] = True
        res = client.put("/api/personas/guardian", json=edit)
        assert res.status_code == 200, res.text

        turn = {"message": "now fix it", "agent_id": "guardian", "session_id": "s2"}
        assert client.post("/api/turn", json=turn).status_code == 200

        assert target.read_text(encoding="utf-8") == "rewritten"
        assert not a_helper_was_started()


# --- the declarations reach the record a room reads (#1354, #1355) ------------------------


@pytest.mark.asyncio
async def test_an_executed_tools_declarations_are_carried_on_its_record() -> None:
    """A room records a written file only from a tool that declares it writes.

    The record is the only thing the room sees of a call, so the flag has to travel on it:
    without it, the room would have to guess from an output's shape, and `file_read`
    returns a `path` too.

    Killed by: src/uclone_x/agent/base.py :: declared_writes = tool_call_writes_files(tool_inst, unwrapped_args)
    Becomes: declared_writes = False
    """
    agent = _agent(_Writer(), _Reader(), enable_write_tools=True)

    wrote = await agent.execute_tool_call("scribble", {"text": "x"})
    read = await agent.execute_tool_call("peek", {"text": "x"})

    assert wrote.writes_files is True
    assert read.writes_files is False
    assert wrote.spawns_subagents is False


class _Spawner(BaseTool[_TextParams]):
    name = "spawn"
    description = "Starts a helper."
    writes_files = False
    spawns_subagents = True

    def run(self, params: _TextParams, context: ToolContext) -> str:
        return "started"


@pytest.mark.asyncio
async def test_a_spawning_tools_declaration_is_carried_on_its_record() -> None:
    """P4: the room's topology draws a sub-agent only from a tool that declares it spawns.

    Killed by: src/uclone_x/agent/base.py :: spawns_subagents=tool_spawns_subagents(tool_inst),
    Becomes: spawns_subagents=False,
    """
    agent = _agent(_Spawner(), enable_subagent_tools=True)

    record = await agent.execute_tool_call("spawn", {"text": "x"})

    assert record.status is ToolResultStatus.SUCCESS, record.error
    assert record.spawns_subagents is True
