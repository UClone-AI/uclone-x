"""Every persona is given the base tools: its own memory and read-only workspace access.

Owner ruling on #1402: an agent is given the tools it basically needs by default. Before
it, `scout`, `pioneer`, `guardian`, `writer` and `artist` each listed `allowed_tools`
without `record_memory_fact`, so a call to it was refused as not allowed and -- since
#1400 -- the row said nothing was saved. Only `clone`, which lists no tools and so is not restricted, could remember.

`BASE_PERSONA_TOOLS` is the one place the base set is named, and
`PersonaDefinition.granted_tools` is the one place it is added to a persona's list. What is
pinned here is that every way an agent takes on a persona reads the union: a chat agent, a
room seat, a bootstrapped config and a workspace persona.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from uclone_x.agent.base import BaseAgent
from uclone_x.agent.bootstrap import agent_config_for_persona
from uclone_x.agent.composition import HostDependencies
from uclone_x.agent.models import (
    BASE_MEMORY_TOOLS,
    BASE_PERSONA_TOOLS,
    AgentConfig,
    AgentContext,
    AgentLLMConfig,
    PersonaDefinition,
)
from uclone_x.agent.persona_registry import PersonaRegistry
from uclone_x.agent.session import SessionStore
from uclone_x.engine.event_bus import EventBus
from uclone_x.llm.connectors.mock import MockLLMConnector
from uclone_x.llm.models import LLMRequest, ModelResponse, ToolCallRequest
from uclone_x.memory.store import CrossSessionMemory
from uclone_x.memory.tools import QueryMemoryFactsTool, RecordMemoryFactTool, RetractMemoryFactTool
from uclone_x.room.models import Participant, ParticipantKind
from uclone_x.room.resolver import RoomAgentResolver
from uclone_x.telemetry.tracer import TelemetryTracer
from uclone_x.tools.base import tool_spawns_subagents, tool_writes_files
from uclone_x.tools.builtin.filesystem import DirectoryListTool, FileReadTool, FileSearchTool
from uclone_x.tools.builtin.skill_loader import LoadSkillTool
from uclone_x.tools.builtin.tool_results import ToolResultReadTool
from uclone_x.tools.models import ToolResultStatus
from uclone_x.tools.registry import ToolRegistry, create_default_registry

#: The built-ins that carry an `allowed_tools` list, and so were the ones that could not
#: remember. `clone` lists none and is covered separately: it must stay unrestricted.
RESTRICTED_BUILTINS = ("scout", "pioneer", "guardian", "writer", "artist")

_BASE_CLASSES = (
    RecordMemoryFactTool,
    QueryMemoryFactsTool,
    RetractMemoryFactTool,
    FileReadTool,
    FileSearchTool,
    DirectoryListTool,
    ToolResultReadTool,
    LoadSkillTool,
)


def _host(tmp_path: Path, llm: MockLLMConnector | None = None) -> HostDependencies:
    return HostDependencies(
        bus=EventBus(),
        llm=llm or MockLLMConnector(),
        tools=create_default_registry(enable_mcp=False),
        tracer=TelemetryTracer(),
        store=SessionStore(tmp_path / "sessions"),
    )


class _RecordingConnector(MockLLMConnector):
    """Answers every request with plain text, and keeps each request it was sent."""

    def __init__(self) -> None:
        super().__init__(default_response="ok")
        self.requests: list[LLMRequest] = []

    async def generate(self, request: LLMRequest) -> ModelResponse:
        self.requests.append(request)
        return await super().generate(request)


def _seat(participant_id: str, persona: str) -> Participant:
    return Participant(
        id=participant_id,
        kind=ParticipantKind.AGENT,
        display_name=participant_id,
        session_id=f"sess_room__r1__{participant_id}",
        persona=persona,
    )


def _memory_factory(tmp_path: Path, stores: dict[str, CrossSessionMemory]):  # noqa: ANN202
    def memory_for(agent_id: str) -> CrossSessionMemory:
        store = CrossSessionMemory(storage_path=tmp_path / f"{agent_id}-memory.json")
        stores[agent_id] = store
        return store

    return memory_for


class TestTheBaseSet:
    def test_each_name_is_the_name_a_tool_class_registers_under(self) -> None:
        """The set is spelled as names, so a renamed tool would leave a dead entry.

        Killed by: src/uclone_x/agent/models.py :: "query_memory_facts",
        Becomes: "query_memory_fact",
        """
        assert set(BASE_PERSONA_TOOLS) == {cls.name for cls in _BASE_CLASSES}
        assert len(BASE_PERSONA_TOOLS) == len(set(BASE_PERSONA_TOOLS))

    def test_nothing_in_it_writes_a_file_or_starts_an_agent(self) -> None:
        """The rule for membership: no effect outside the agent's own memory."""
        for cls in _BASE_CLASSES:
            assert not tool_writes_files(cls), cls.name
            assert not tool_spawns_subagents(cls), cls.name

    def test_of_the_default_tools_only_the_read_only_tools_are_in_it(self) -> None:
        """Shell, write, install, network, image, plan and sub-agent tools stay per persona.

        The read-only four: the three file tools, and `tool_result_read`, which reads back
        this conversation's own shortened results (#1422).
        """
        default_names = {
            tool.name for tool in create_default_registry(enable_mcp=False).list_tools()
        }
        assert default_names & set(BASE_PERSONA_TOOLS) == {
            "file_read",
            "file_search",
            "directory_list",
            "tool_result_read",
        }


class TestGrantedTools:
    def test_a_personas_own_list_is_kept_and_the_base_follows_once(self) -> None:
        """Killed by: src/uclone_x/agent/models.py :: return own + tuple(name for name in BASE_PERSONA_TOOLS if name not in own)
        Becomes: return own
        """
        persona = PersonaDefinition(
            name="p",
            role="r",
            system_prompt="s",
            allowed_tools=("web_search", "file_read"),
        )
        granted = persona.granted_tools
        assert granted[:2] == ("web_search", "file_read")
        assert set(granted) == {"web_search", *BASE_PERSONA_TOOLS}
        assert len(granted) == len(set(granted))
        # The persona's own list is not rewritten: its file and its editor keep what it said.
        assert persona.allowed_tools == ("web_search", "file_read")

    def test_a_persona_with_no_list_stays_unrestricted(self) -> None:
        """An empty list is every tool; the base set added to it would be a restriction.

        Killed by: src/uclone_x/agent/models.py :: if not self.allowed_tools:
        Becomes: if False:
        """
        persona = PersonaDefinition(name="p", role="r", system_prompt="s")
        assert persona.granted_tools == ()


class TestEveryWayToTakeOnAPersona:
    @pytest.mark.parametrize("name", RESTRICTED_BUILTINS)
    def test_a_builtin_persona_agent_is_scoped_to_its_list_plus_the_base(self, name: str) -> None:
        """The chat head: an agent resolving its persona's list itself (#892).

        Killed by: src/uclone_x/agent/base.py :: resolved = persona.granted_tools
        Becomes: resolved = persona.allowed_tools
        """
        persona = PersonaRegistry().get_persona(name)
        assert persona is not None and persona.allowed_tools
        agent = BaseAgent(config=AgentConfig(agent_id=name, name=name, persona=name))

        scope = set(agent.config.allowed_tools)
        assert set(BASE_PERSONA_TOOLS) <= scope
        assert set(persona.allowed_tools) <= scope
        assert scope == set(persona.allowed_tools) | set(BASE_PERSONA_TOOLS)

    def test_clone_is_still_unrestricted(self) -> None:
        agent = BaseAgent(config=AgentConfig(agent_id="clone", name="clone", persona="clone"))
        assert agent.config.allowed_tools == ()

    def test_an_operators_list_is_taken_as_written(self) -> None:
        """An operator naming tools names exactly what the agent may run; no base is added."""
        agent = BaseAgent(
            config=AgentConfig(
                agent_id="scout", name="scout", persona="scout", allowed_tools=("web_search",)
            )
        )
        assert agent.config.allowed_tools == ("web_search",)

    @pytest.mark.parametrize("name", RESTRICTED_BUILTINS)
    def test_a_bootstrapped_config_carries_the_base(self, name: str) -> None:
        """A config from `agent_config_for_persona` gives its agent the persona's list and the base.

        The config names the persona and leaves `allowed_tools` to the agent (#892, #1448), so
        the union is read where the agent resolves it.

        Killed by: src/uclone_x/agent/bootstrap.py :: persona=persona.name,
        Becomes: persona=None,
        """
        persona = PersonaRegistry().get_persona(name)
        assert persona is not None
        agent = BaseAgent(config=agent_config_for_persona(persona))
        scope = set(agent.config.allowed_tools)
        assert scope == set(persona.allowed_tools) | set(BASE_PERSONA_TOOLS)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("name", RESTRICTED_BUILTINS)
    async def test_a_room_seat_is_offered_the_memory_tools(self, name: str, tmp_path: Path) -> None:
        """A seat's scope carries the base, and its request offers the memory tools.

        Killed by: src/uclone_x/agent/models.py :: return own + tuple(name for name in BASE_PERSONA_TOOLS if name not in own)
        Becomes: return own
        """
        llm = _RecordingConnector()
        resolver = RoomAgentResolver(
            _host(tmp_path, llm),
            memory_factory=_memory_factory(tmp_path, {}),
            persona_registry=PersonaRegistry(),
        )
        agent = cast("BaseAgent", await resolver.resolve(_seat(name, name)))

        assert set(BASE_PERSONA_TOOLS) <= set(agent.config.allowed_tools)
        await agent.execute_turn("Hello")
        offered = {tool.name for tool in llm.requests[-1].tools}
        assert {"record_memory_fact", "query_memory_facts"} <= offered

    def test_a_workspace_persona_is_given_the_base_too(self, tmp_path: Path) -> None:
        """A persona the user wrote is a `PersonaDefinition` like a built-in, and reads the same union."""
        personas = tmp_path / ".uclone" / "personas"
        personas.mkdir(parents=True)
        (personas / "notes.yaml").write_text(
            "name: notes\nrole: Note taker\nsystem_prompt: Take notes.\n"
            "allowed_tools:\n  - web_search\n",
            encoding="utf-8",
        )
        persona = PersonaRegistry(workspace_root=tmp_path, include_defaults=False).get_persona(
            "notes"
        )
        assert persona is not None
        assert persona.allowed_tools == ("web_search",)
        assert set(persona.granted_tools) == {"web_search", *BASE_PERSONA_TOOLS}

    @pytest.mark.asyncio
    async def test_a_sub_agent_inherits_the_base_less_the_memory_tools(self) -> None:
        """A child holds its parent's resolved list, less the memory tools (#1431)."""
        parent = BaseAgent(
            config=AgentConfig(agent_id="lead", name="lead", enable_subagent_tools=True),
            tools=ToolRegistry(),
        )
        parent.define_persona(
            PersonaDefinition(
                name="lead",
                role="Lead",
                system_prompt="Lead.",
                allowed_tools=("web_search",),
                enable_subagent_tools=True,
            )
        )
        parent.persona = "lead"
        child = await parent.spawn_subagent(role="helper", goal="help")
        assert set(child.config.allowed_tools) == {
            "web_search",
            *(name for name in BASE_PERSONA_TOOLS if name not in BASE_MEMORY_TOOLS),
        }


class TestABuiltinPersonaCanRemember:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("name", RESTRICTED_BUILTINS)
    async def test_a_seated_persona_saves_a_fact_in_a_turn(self, name: str, tmp_path: Path) -> None:
        """End to end: the model asks to remember, the real tool runs, the fact is in the store.

        The turn goes through the same allowlist check that refused the call before #1402
        (`_execute_single_tool`), against the seat's real store -- nothing pre-approves it.
        """
        llm = MockLLMConnector(
            responses=["", "I'll remember that."],
            tool_calls=[
                ToolCallRequest(
                    id="c1",
                    name="record_memory_fact",
                    arguments={
                        "subject": "user",
                        "predicate": "favourite colour",
                        "object_value": "teal",
                    },
                )
            ],
        )
        stores: dict[str, CrossSessionMemory] = {}
        resolver = RoomAgentResolver(
            _host(tmp_path, llm),
            llm_config=AgentLLMConfig(model_name="mock-model"),
            memory_factory=_memory_factory(tmp_path, stores),
            persona_registry=PersonaRegistry(),
        )
        agent = cast("BaseAgent", await resolver.resolve(_seat(name, name)))

        result = await agent.run_turn("Remember that my favourite colour is teal.")

        saves = [r for r in result.tool_executions if r.tool_name == "record_memory_fact"]
        assert [r.status for r in saves] == [ToolResultStatus.SUCCESS], [r.error for r in saves]
        assert [(f.subject, f.object_value) for f in stores[name].list_facts()] == [
            ("user", "teal")
        ]


class TestTheWorkspaceSectionFollowsARegisteredFileTool:
    """Every persona is now *permitted* the read-only file tools, registered or not.

    The `[Workspace]` prompt section explains how the file tools resolve paths, and was
    shown whenever the allowlist named one. With the base set in every list, that would
    describe file tools to an agent whose registry has none.
    """

    def _agent(self, tmp_path: Path, tools: ToolRegistry) -> BaseAgent:
        agent = BaseAgent(
            config=AgentConfig(agent_id="lead", name="lead"),
            tools=tools,
            context=AgentContext(session_id="s", agent_id="lead", workspace_root=tmp_path),
        )
        agent.define_persona(
            PersonaDefinition(
                name="lead", role="Lead", system_prompt="Lead.", allowed_tools=("web_search",)
            )
        )
        agent.persona = "lead"
        return agent

    def test_no_section_when_no_file_tool_is_registered(self, tmp_path: Path) -> None:
        """Killed by: src/uclone_x/agent/base.py :: name in _FILE_TOOL_NAMES and self._tools is not None and self._tools.get(name)
        Becomes: name in _FILE_TOOL_NAMES
        """
        agent = self._agent(tmp_path, ToolRegistry())
        assert "file_read" in agent.config.allowed_tools
        assert agent._get_workspace_prompt_section() is None  # pyright: ignore[reportPrivateUsage]

    def test_the_section_when_a_permitted_file_tool_is_registered(self, tmp_path: Path) -> None:
        agent = self._agent(tmp_path, ToolRegistry(tools=[FileReadTool()]))
        section = agent._get_workspace_prompt_section()  # pyright: ignore[reportPrivateUsage]
        assert section is not None and section.startswith("[Workspace]")
