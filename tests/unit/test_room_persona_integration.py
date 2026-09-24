"""Tests for room persona integration: declarative personas, scoped tools, and room orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
from pydantic import BaseModel

from uclone_x.agent.base import BaseAgent
from uclone_x.agent.composition import HostDependencies
from uclone_x.agent.models import BASE_PERSONA_TOOLS, PersonaDefinition
from uclone_x.agent.persona_registry import PersonaRegistry
from uclone_x.agent.session import SessionStore
from uclone_x.engine.event_bus import EventBus
from uclone_x.llm.connectors.mock import MockLLMConnector
from uclone_x.llm.models import LLMRequest, ModelResponse, ToolCallRequest
from uclone_x.room.models import Participant, ParticipantKind
from uclone_x.room.orchestrator import RoomOrchestrator
from uclone_x.room.resolver import RoomAgentResolver
from uclone_x.room.selectors import MentionSelector
from uclone_x.room.service import RoomService
from uclone_x.room.store import RoomStore
from uclone_x.telemetry.tracer import TelemetryTracer
from uclone_x.tools.base import BaseTool
from uclone_x.tools.models import ToolContext, ToolResultStatus
from uclone_x.tools.registry import ToolRegistry


class DummyParams(BaseModel):
    pass


class DummyTool(BaseTool[DummyParams]):
    def __init__(self, name: str) -> None:
        super().__init__(name=name, description=f"Dummy tool {name}", params_type=DummyParams)

    async def run(self, params: DummyParams, context: ToolContext) -> str:
        return f"Executed {self.name}"


@pytest.fixture
def persona_registry(tmp_path: Path) -> PersonaRegistry:
    reg = PersonaRegistry(workspace_root=tmp_path, include_defaults=False)
    reg.register_persona(
        PersonaDefinition(
            name="novelist",
            role="Creative Fiction Writer",
            description="Specialist in narrative prose and world-building.",
            system_prompt="You are a novelist. Focus on vivid sensory details and emotional stakes.",
            allowed_tools=("draft_chapter", "read_outline"),
        )
    )
    reg.register_persona(
        PersonaDefinition(
            name="critic",
            role="Editorial Reviewer",
            description="Specialist in narrative critique, plot holes, and structure.",
            system_prompt="You are a literary critic. Be analytical, incisive, and rigorous.",
            allowed_tools=("read_outline", "check_pacing"),
        )
    )
    return reg


class _ReadOnlyTool(DummyTool):
    """A dummy that declares it writes no file, so a persona's write switch does not hide it."""

    writes_files = False


class _RecordingConnector(MockLLMConnector):
    """Answers every request with plain text, and keeps each request it was sent."""

    def __init__(self) -> None:
        super().__init__(default_response="ok")
        self.requests: list[LLMRequest] = []

    async def generate(self, request: LLMRequest) -> ModelResponse:
        self.requests.append(request)
        return await super().generate(request)


def _own_tools_offered(llm: _RecordingConnector) -> set[str]:
    """The tools the latest request offered, less the base set every persona is given."""
    assert llm.requests, "no request reached the model"
    return {t.name for t in llm.requests[-1].tools} - set(BASE_PERSONA_TOOLS)


@pytest.fixture
def host(tmp_path: Path) -> HostDependencies:
    tools = ToolRegistry()
    tools.register(_ReadOnlyTool("draft_chapter"))
    tools.register(_ReadOnlyTool("read_outline"))
    tools.register(_ReadOnlyTool("check_pacing"))
    tools.register(_ReadOnlyTool("admin_shell"))

    return HostDependencies(
        bus=EventBus(),
        llm=_RecordingConnector(),
        tools=tools,
        tracer=TelemetryTracer(),
        store=SessionStore(tmp_path / "sessions"),
    )


class TestRoomPersonaResolution:
    @pytest.mark.asyncio
    async def test_resolved_agent_has_composed_system_prompt(
        self, host: HostDependencies, persona_registry: PersonaRegistry
    ) -> None:
        resolver = RoomAgentResolver(host, persona_registry=persona_registry)
        participant = Participant(
            id="author",
            kind=ParticipantKind.AGENT,
            display_name="Author",
            persona="novelist",
            session_id="sess_room__r1__author",
        )

        agent = await resolver.resolve(participant)
        assert agent.config.persona == "novelist"
        assert isinstance(agent, BaseAgent)
        prompt = agent.effective_system_prompt
        assert (
            "You are Author (author), one participant in a shared multi-agent conversation."
            in prompt
        )
        assert "[Persona Instructions: Creative Fiction Writer]" in prompt
        assert "You are a novelist. Focus on vivid sensory details" in prompt

    @pytest.mark.asyncio
    async def test_resolved_agent_receives_scoped_tools(
        self, host: HostDependencies, persona_registry: PersonaRegistry
    ) -> None:
        """The persona's allowlist decides both what the seat is offered and what it may run.

        `config.allowed_tools` is the half that refuses a call at execution time (#909), so it
        is asserted directly; what the model is offered is read from the request it was sent.
        Both are the agent's own resolution of the persona the resolver defines on it: the
        seat's registry is the host's, unscoped, since #1448.

        Killed by: src/uclone_x/agent/models.py :: return own + tuple(name for name in BASE_PERSONA_TOOLS if name not in own)
        Becomes: return ()
        """
        resolver = RoomAgentResolver(host, persona_registry=persona_registry)
        participant = Participant(
            id="author",
            kind=ParticipantKind.AGENT,
            display_name="Author",
            persona="novelist",
            session_id="sess_room__r1__author",
        )

        agent = await resolver.resolve(participant)
        assert isinstance(agent, BaseAgent)
        await agent.execute_turn("Go")

        # The persona's own list plus the base set every persona is given (#1402). None of
        # the base tools is registered on this host, so none is offered.
        assert _own_tools_offered(cast("_RecordingConnector", host.llm)) == {
            "draft_chapter",
            "read_outline",
        }
        # What the agent will actually refuse. The request above only shows what is offered.
        assert agent.config.allowed_tools == ("draft_chapter", "read_outline", *BASE_PERSONA_TOOLS)

    @pytest.mark.asyncio
    async def test_fallback_to_participant_id_when_persona_field_empty(
        self, host: HostDependencies, persona_registry: PersonaRegistry
    ) -> None:
        """A persona found through the participant id carries its allowlist into the seat.

        Pinned through `granted_tools`, for the reason given on the test above.

        Killed by: src/uclone_x/agent/models.py :: return own + tuple(name for name in BASE_PERSONA_TOOLS if name not in own)
        Becomes: return ()
        """
        resolver = RoomAgentResolver(host, persona_registry=persona_registry)
        participant = Participant(
            id="critic",
            kind=ParticipantKind.AGENT,
            display_name="Reviewer",
            persona="",
            session_id="sess_room__r1__critic",
        )

        agent = await resolver.resolve(participant)
        assert agent.config.persona == "critic"
        assert isinstance(agent, BaseAgent)
        assert "[Persona Instructions: Editorial Reviewer]" in agent.effective_system_prompt
        await agent.execute_turn("Go")
        assert _own_tools_offered(cast("_RecordingConnector", host.llm)) == {
            "read_outline",
            "check_pacing",
        }
        assert agent.config.allowed_tools == ("read_outline", "check_pacing", *BASE_PERSONA_TOOLS)

    @pytest.mark.asyncio
    async def test_room_resolved_agent_does_not_run_a_tool_outside_its_persona_allowlist(
        self, tmp_path: Path
    ) -> None:
        """A room persona agent refuses a tool call its persona does not allow, mid-turn.

        Since #909 the scoped registry's `get` finds a withheld tool, so what stands between
        the model's request and the tool running is `config.allowed_tools`. The assertions on
        the registry above cannot see it. It is pinned through `granted_tools`, for the reason
        given on `test_resolved_agent_receives_scoped_tools`.

        The persona name is deliberately one nothing ships: `BaseAgent.__init__` backfills an
        empty `allowed_tools` from a same-named shipped persona, and with a shipped name the
        mutated run is refused by that list instead -- passing for the wrong reason.

        Killed by: src/uclone_x/agent/models.py :: return own + tuple(name for name in BASE_PERSONA_TOOLS if name not in own)
        Becomes: return ()
        """
        persona_name = "room_scoped_scribe"
        # Guard the precondition the docstring depends on, so a future shipped persona of
        # this name turns this test red instead of silently hiding the gap again.
        assert PersonaRegistry().get_persona(persona_name) is None

        executed: list[str] = []

        class RecordingTool(BaseTool[DummyParams]):
            def __init__(self) -> None:
                super().__init__(
                    name="admin_shell", description="Records that it ran.", params_type=DummyParams
                )

            async def run(self, params: DummyParams, context: ToolContext) -> str:
                executed.append(self.name)
                return "ran"

        tools = ToolRegistry()
        tools.register(DummyTool("draft_chapter"))
        tools.register(RecordingTool())
        scripted = MockLLMConnector(
            default_response="Done.",
            tool_calls=[ToolCallRequest(id="call_1", name="admin_shell", arguments={})],
        )
        host = HostDependencies(
            bus=EventBus(),
            llm=scripted,
            tools=tools,
            tracer=TelemetryTracer(),
            store=SessionStore(tmp_path / "sessions"),
        )
        registry = PersonaRegistry(workspace_root=tmp_path, include_defaults=False)
        registry.register_persona(
            PersonaDefinition(
                name=persona_name,
                role="Scribe",
                description="Drafts chapters and nothing else.",
                system_prompt="You draft chapters.",
                allowed_tools=("draft_chapter",),
            )
        )
        resolver = RoomAgentResolver(host, persona_registry=registry)
        participant = Participant(
            id="scribe",
            kind=ParticipantKind.AGENT,
            display_name="Scribe",
            persona=persona_name,
            session_id="sess_room__r1__scribe",
        )

        agent = await resolver.resolve(participant)
        assert isinstance(agent, BaseAgent)
        result = await agent.execute_turn("Go")

        assert executed == []
        assert len(result.tool_executions) == 1
        record = result.tool_executions[0]
        assert record.tool_name == "admin_shell"
        assert record.status is ToolResultStatus.ERROR
        assert "allowed_tools" in (record.error or "")


class TestAPersonaEditReachesARoomSeat:
    @pytest.mark.asyncio
    async def test_an_edited_tool_list_reaches_a_seated_agent_as_it_does_a_chat_agent(
        self, host: HostDependencies, persona_registry: PersonaRegistry
    ) -> None:
        """A seat handed an edited persona offers and allows the edited tools, not the old ones.

        `define_persona` is the call the chat head makes to put an edit in force on a live
        agent (`AgentSessionManager.apply_persona`, #892). A seat whose config carried the
        persona's list as the *operator's* list kept it through that call, because an
        operator's list wins over any persona's; and a registry proxy fixed at seating hid a
        tool the edit added. Either one left the seat on the old tools (#1448).

        Asserted on what the seat does -- the tools its next request offers, and a direct call
        to a tool the edit removed -- not on how it is built.

        Killed by: src/uclone_x/agent/bootstrap.py :: allowed_tools=(),  # the persona's list, resolved by the agent
        Becomes: allowed_tools=persona.granted_tools,
        """
        llm = cast("_RecordingConnector", host.llm)
        resolver = RoomAgentResolver(host, persona_registry=persona_registry)
        participant = Participant(
            id="author",
            kind=ParticipantKind.AGENT,
            display_name="Author",
            persona="novelist",
            session_id="sess_room__r1__author",
        )
        agent = await resolver.resolve(participant)
        assert isinstance(agent, BaseAgent)
        await agent.execute_turn("Go")
        assert _own_tools_offered(llm) == {"draft_chapter", "read_outline"}

        seated = agent.get_persona("novelist")
        assert seated is not None
        agent.define_persona(seated.model_copy(update={"allowed_tools": ("check_pacing",)}))
        await agent.execute_turn("Again")

        assert _own_tools_offered(llm) == {"check_pacing"}
        assert agent.config.allowed_tools == ("check_pacing", *BASE_PERSONA_TOOLS)
        with pytest.raises(PermissionError):
            await agent.execute_tool_call("draft_chapter", {})


class TestRoomServicePersonaAutoHydration:
    def test_service_add_participant_auto_hydrates_persona_summary(
        self, tmp_path: Path, persona_registry: PersonaRegistry
    ) -> None:
        store = RoomStore(tmp_path / "rooms")
        service = RoomService(store)
        service.create("Novel Collab", room_id="collab_1")

        state = service.add_participant(
            "collab_1",
            "writer",
            persona="writer",
        )

        writer = next(p for p in state.participants if p.id == "writer")
        assert writer.persona == "writer"
        assert (
            "prose" in writer.persona_summary.lower()
            or "narrative" in writer.persona_summary.lower()
        )


class TestRoomPersonaCollaborationOrchestration:
    @pytest.mark.asyncio
    async def test_room_orchestration_with_persona_agents(
        self, tmp_path: Path, host: HostDependencies, persona_registry: PersonaRegistry
    ) -> None:
        store = RoomStore(tmp_path / "rooms")
        service = RoomService(store)
        service.create("Story Room", room_id="room_story")

        service.add_participant("room_story", "human_user", kind=ParticipantKind.HUMAN)
        service.add_participant("room_story", "writer", persona="novelist")
        service.add_participant("room_story", "editor", persona="critic", aliases=("reviewer",))

        state = service.get("room_story")
        assert len(state.participants) == 3

        resolver = RoomAgentResolver(host, persona_registry=persona_registry)
        selectors = (MentionSelector(),)
        orchestrator = RoomOrchestrator(
            store=store,
            selectors=selectors,
            resolver=resolver,
        )

        post_state = await orchestrator.post(
            room_id="room_story",
            sender_id="human_user",
            content="Hello @writer, please draft the opening scene!",
        )

        utterances = [m for m in post_state.transcript if m.is_utterance]
        assert len(utterances) == 2
        assert utterances[0].sender_id == "human_user"
        assert utterances[1].sender_id == "writer"
