"""Tests for room persona integration: declarative personas, scoped tools, and room orchestration."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import BaseModel

from uclone_x.agent import session_lifecycle
from uclone_x.agent.base import BaseAgent
from uclone_x.agent.composition import HostDependencies
from uclone_x.agent.models import BASE_PERSONA_TOOLS, PersonaDefinition, TurnResult
from uclone_x.agent.persona_registry import PersonaRegistry
from uclone_x.agent.session import SessionStore
from uclone_x.agent.turn_executor import PERSONA_EDIT_NOT_APPLIED
from uclone_x.agent.turn_trace import trace_turn
from uclone_x.core.context_state import EPOCH_PERSONA_EDITED
from uclone_x.engine.event_bus import EventBus
from uclone_x.llm.connectors.mock import MockLLMConnector
from uclone_x.llm.models import LLMRequest, ModelResponse, ToolCallRequest
from uclone_x.log.reader import read_session_log
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


class _HeldConnector(_RecordingConnector):
    """A recording connector whose next request can be held, so a turn is in flight."""

    def __init__(self) -> None:
        super().__init__()
        self.hold_next = False
        self.reached = asyncio.Event()
        self.release = asyncio.Event()

    def ask_for_tool(self, call: ToolCallRequest) -> None:
        """Ask for `call` until its result is in a request (the mock's own rule)."""
        self._tool_calls = [call]

    async def generate(self, request: LLMRequest) -> ModelResponse:
        if self.hold_next:
            self.hold_next = False
            self.reached.set()
            await self.release.wait()
        return await super().generate(request)


def _assert_told_plainly(failed: TurnResult, cause: str) -> None:
    """The failed turn names a dropped edit, in the fixed sentence and never the cause (#1904)."""
    assert failed.stop_reason == "persona_edit_failed"
    assert failed.error == PERSONA_EDIT_NOT_APPLIED
    assert cause not in (failed.error or "")
    assert "RuntimeError" not in (failed.error or "")


def _own_tools_offered(llm: _RecordingConnector) -> set[str]:
    """The tools the latest request offered, less the base set every persona is given."""
    assert llm.requests, "no request reached the model"
    return {t.name for t in llm.requests[-1].tools} - set(BASE_PERSONA_TOOLS)


@pytest.fixture
def host(tmp_path: Path) -> HostDependencies:
    return _host(tmp_path, _RecordingConnector())


def _host(tmp_path: Path, llm: _RecordingConnector) -> HostDependencies:
    tools = ToolRegistry()
    tools.register(_ReadOnlyTool("draft_chapter"))
    tools.register(_ReadOnlyTool("read_outline"))
    tools.register(_ReadOnlyTool("check_pacing"))
    tools.register(_ReadOnlyTool("admin_shell"))

    return HostDependencies(
        bus=EventBus(),
        llm=llm,
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
            id="novelist",
            kind=ParticipantKind.AGENT,
            display_name="Author",
            session_id="sess_room__r1__novelist",
        )

        agent = await resolver.resolve(participant)
        assert agent.config.persona == "novelist"
        assert isinstance(agent, BaseAgent)
        prompt = agent.effective_system_prompt
        assert (
            "You are Author (novelist), one participant in a shared multi-agent conversation."
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

        Killed by: src/uclone_x/core/models.py :: return own + tuple(name for name in BASE_PERSONA_TOOLS if name not in own)
        Becomes: return ()
        """
        resolver = RoomAgentResolver(host, persona_registry=persona_registry)
        participant = Participant(
            id="novelist",
            kind=ParticipantKind.AGENT,
            display_name="Author",
            session_id="sess_room__r1__novelist",
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
    async def test_a_seat_finds_its_persona_by_its_id(
        self, host: HostDependencies, persona_registry: PersonaRegistry
    ) -> None:
        """A persona found through the seat's id carries its allowlist into the seat.

        A seat names no persona of its own: it is its clone, and the persona is looked up
        by the seat id (clone-data-scopes §4 step 3).

        Pinned through `granted_tools`, for the reason given on the test above.

        Killed by: src/uclone_x/core/models.py :: return own + tuple(name for name in BASE_PERSONA_TOOLS if name not in own)
        Becomes: return ()
        """
        resolver = RoomAgentResolver(host, persona_registry=persona_registry)
        participant = Participant(
            id="critic",
            kind=ParticipantKind.AGENT,
            display_name="Reviewer",
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

        Killed by: src/uclone_x/core/models.py :: return own + tuple(name for name in BASE_PERSONA_TOOLS if name not in own)
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
            id=persona_name,
            kind=ParticipantKind.AGENT,
            display_name="Scribe",
            session_id=f"sess_room__r1__{persona_name}",
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

        `define_persona` is the call that puts an edit in force on a live agent -- made by
        the seat's resolver at its next turn (`RoomAgentResolver._take_persona_edit`,
        #1899; before rooms, `AgentSessionManager.apply_persona`, #892). A seat whose config carried the
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
            id="novelist",
            kind=ParticipantKind.AGENT,
            display_name="Author",
            session_id="sess_room__r1__novelist",
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

    @pytest.mark.asyncio
    async def test_a_saved_persona_edit_reaches_a_live_seat_at_its_next_turn(
        self, host: HostDependencies, persona_registry: PersonaRegistry
    ) -> None:
        """A persona saved while a seat is live is in force from that seat's next turn (#1899).

        The seat registers the definition it was built with, which pins it: saving the
        persona changed the registry and nothing the seat reads. The edit is held by the
        seat (`stage_persona_edit`) and applied when its next turn starts -- never to the
        live agent at save time, since a seat may be mid-turn and within a turn the context
        only appends (llm-request-layering §5.8 Rule 1).

        Killed by: src/uclone_x/agent/turn_executor.py :: self._take_staged_persona()
        Becomes: None
        """
        llm = cast("_RecordingConnector", host.llm)
        resolver = RoomAgentResolver(host, persona_registry=persona_registry)
        participant = Participant(
            id="novelist",
            kind=ParticipantKind.AGENT,
            display_name="Author",
            session_id="sess_room__r1__novelist",
        )
        agent = await resolver.resolve(participant)
        assert isinstance(agent, BaseAgent)
        await agent.execute_turn("Go")
        assert _own_tools_offered(llm) == {"draft_chapter", "read_outline"}

        saved = persona_registry.get_persona("novelist")
        assert saved is not None
        edited = saved.model_copy(
            update={
                "allowed_tools": ("check_pacing",),
                "system_prompt": "You are a poet. Answer in short verse.",
            }
        )
        persona_registry.register_persona(edited)

        assert resolver.persona_edited(edited) == 1, "the seat speaking as it is not counted"
        # Not applied at save time: the live agent is untouched until its next turn begins.
        assert agent.config.allowed_tools != ("check_pacing", *BASE_PERSONA_TOOLS)

        again = await resolver.resolve(participant)
        assert again is agent, "the edit rebuilt the seat instead of reaching it"
        await agent.execute_turn("Again")

        assert _own_tools_offered(llm) == {"check_pacing"}
        sent = " ".join(str(m.content) for m in llm.requests[-1].messages)
        assert "short verse" in sent
        with pytest.raises(PermissionError):
            await agent.execute_tool_call("draft_chapter", {})

    @pytest.mark.asyncio
    async def test_a_persona_edit_opens_an_epoch_and_both_turns_still_trace(
        self, host: HostDependencies, persona_registry: PersonaRegistry
    ) -> None:
        """The turn that takes a saved edit opens a `persona_edited` epoch (#1899 review).

        The history only grew between the two turns, so without a declared boundary the
        second turn's requests extend the first epoch and the log shows no point where the
        identity changed. Both turns still verify against the log (`trace_turn`): each step
        is a prefix of its own epoch's conversation.

        A save that changes nothing (the same definition again) marks no boundary.

        Killed by: src/uclone_x/agent/turn_executor.py :: self._active_session.declare_new_epoch(EPOCH_PERSONA_EDITED)
        Becomes: None
        """
        store = cast("SessionStore", host.store)
        resolver = RoomAgentResolver(host, persona_registry=persona_registry)
        participant = Participant(
            id="novelist",
            kind=ParticipantKind.AGENT,
            display_name="Author",
            session_id="sess_room__r1__novelist",
        )
        agent = await resolver.resolve(participant)
        assert isinstance(agent, BaseAgent)
        assert (await agent.execute_turn("Go", caller_turn_id="t1")).is_completed

        saved = persona_registry.get_persona("novelist")
        assert saved is not None
        edited = saved.model_copy(update={"system_prompt": "You are a poet."})
        persona_registry.register_persona(edited)
        assert resolver.persona_edited(edited) == 1
        assert (await agent.execute_turn("Again", caller_turn_id="t2")).is_completed
        # Saved again unchanged: nothing moved, so no boundary is marked.
        assert resolver.persona_edited(edited) == 1
        assert (await agent.execute_turn("Once more", caller_turn_id="t3")).is_completed
        agent.persist_session()

        state = store.load(agent.session_id)
        assert state is not None
        assert [(e.turn, e.opened_by) for e in state.context_epochs] == [
            (1, ("start",)),
            (2, (EPOCH_PERSONA_EDITED,)),
        ]
        log_path = store.event_log_path(agent.session_id)
        assert log_path is not None
        events = [dict(e) for e in read_session_log(log_path)]
        for turn_id in ("t1", "t2", "t3"):
            trace = trace_turn(store, state, events, caller_turn_id=turn_id)
            assert trace.steps, turn_id
            assert all(s.verified for s in trace.steps), turn_id
            assert all(s.from_log for s in trace.steps), turn_id

    @pytest.mark.asyncio
    async def test_an_edit_saved_mid_turn_waits_for_the_seats_next_turn(
        self, tmp_path: Path, persona_registry: PersonaRegistry
    ) -> None:
        """An edit saved while a seat's turn runs, and a compaction resolve, change nothing yet.

        Two seats speak as one persona -- the one clone seated in two rooms, since a seat
        is its clone's id. Seat B's turn is held at its first request; the
        persona is saved and both seats are resolved, as `/compact` resolves the seat it
        summarizes while another seat's turn runs (#1899 review). B's in-flight turn
        finishes under the persona it began with -- its second step offers the old tools --
        and A, idle, is not changed by the resolve either. Each takes the edit when its
        own next turn starts.

        Killed by: src/uclone_x/room/resolver.py :: agent.stage_persona_edit(persona)
        Becomes: agent.define_persona(persona)
        """
        llm = _HeldConnector()
        resolver = RoomAgentResolver(_host(tmp_path, llm), persona_registry=persona_registry)
        seat_a, seat_b = (
            Participant(
                id="novelist",
                kind=ParticipantKind.AGENT,
                display_name=name.title(),
                session_id=f"sess_room__{room}__novelist",
            )
            for name, room in (("ann", "r1"), ("ben", "r2"))
        )
        a = await resolver.resolve(seat_a)
        b = await resolver.resolve(seat_b)
        assert isinstance(a, BaseAgent) and isinstance(b, BaseAgent)
        before = b.persona_definition
        assert before is not None

        llm.ask_for_tool(ToolCallRequest(id="c1", name="read_outline", arguments={}))
        llm.hold_next = True
        running = asyncio.create_task(b.execute_turn("Go"))
        await asyncio.wait_for(llm.reached.wait(), 2.0)

        edited = before.model_copy(update={"allowed_tools": ("check_pacing",)})
        persona_registry.register_persona(edited)
        assert resolver.persona_edited(edited) == 2
        assert await resolver.resolve(seat_a) is a
        assert await resolver.resolve(seat_b) is b
        assert a.persona_definition == before, "a resolve applied the edit to an idle seat"
        assert b.persona_definition == before, "the edit reached a seat mid-turn"

        llm.release.set()
        assert (await asyncio.wait_for(running, 5.0)).is_completed
        assert len(llm.requests) == 2, "the held turn did not take its tool step"
        assert _own_tools_offered(llm) == {"draft_chapter", "read_outline"}
        assert b.persona_definition == before

        assert (await b.execute_turn("Again")).is_completed
        assert _own_tools_offered(llm) == {"check_pacing"}
        assert b.persona_definition == edited
        assert (await a.execute_turn("Hello")).is_completed
        assert a.persona_definition == edited

    @pytest.mark.asyncio
    async def test_an_edit_that_fails_to_apply_is_a_failed_turn_not_an_exception(
        self,
        host: HostDependencies,
        persona_registry: PersonaRegistry,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A staged edit whose apply raises ends that turn as a failed result (#1904).

        The apply sat above the turn's `try`, so a raise escaped `execute_turn` with the
        agent left mid-transition and no `TurnResult` for the head to show. The edit is
        dropped with the failed turn rather than failing every turn after it; here the
        apply raises before registering the edit, so the next turn runs under the
        previous definition.

        The mutation below is the revert: the apply runs again ahead of the `try`.

        Killed by: src/uclone_x/agent/turn_executor.py :: self._turn_counter += 1
        Becomes: self._take_staged_persona(); self._turn_counter += 1

        The failure is named `persona_edit_failed` and says so in a fixed sentence, so the
        person who saved the edit is told it was dropped; the cause goes to the log only.
        The mutations below re-raise the cause as it was, or stop naming it.

        Killed by: src/uclone_x/agent/turn_executor.py :: raise _PersonaEditNotApplied from exc
        Becomes: raise
        Killed by: src/uclone_x/agent/turn_executor.py :: if isinstance(exc, _PersonaEditNotApplied):
        Becomes: if False:
        """
        resolver = RoomAgentResolver(host, persona_registry=persona_registry)
        participant = Participant(
            id="novelist",
            kind=ParticipantKind.AGENT,
            display_name="Author",
            session_id="sess_room__r1__novelist",
        )
        agent = await resolver.resolve(participant)
        assert isinstance(agent, BaseAgent)
        assert (await agent.execute_turn("Go")).is_completed
        before = agent.persona_definition
        assert before is not None

        edited = before.model_copy(update={"system_prompt": "You are a poet."})
        assert resolver.persona_edited(edited) == 1

        def _refuse(_persona: PersonaDefinition) -> None:
            raise RuntimeError("the edit could not be applied")

        monkeypatch.setattr(agent, "_persona_definition_commit", _refuse)
        failed = await agent.execute_turn("Again")

        assert not failed.is_completed
        _assert_told_plainly(failed, "the edit could not be applied")
        assert "the edit could not be applied" in caplog.text, "the cause was not logged"
        monkeypatch.undo()
        assert (await agent.execute_turn("Once more")).is_completed
        assert agent.persona_definition == before

    async def _seat_with_a_failing_edit(
        self, host: HostDependencies, persona_registry: PersonaRegistry
    ) -> tuple[BaseAgent, PersonaDefinition, tuple[str, ...]]:
        """A seat after one turn, with an edit to its prompt and tools staged but not taken."""
        resolver = RoomAgentResolver(host, persona_registry=persona_registry)
        participant = Participant(
            id="novelist",
            kind=ParticipantKind.AGENT,
            display_name="Author",
            session_id="sess_room__r1__novelist",
        )
        agent = await resolver.resolve(participant)
        assert isinstance(agent, BaseAgent)
        assert (await agent.execute_turn("Go")).is_completed
        before = agent.persona_definition
        assert before is not None
        edited = before.model_copy(
            update={"system_prompt": "You are a poet.", "allowed_tools": ("check_pacing",)}
        )
        assert resolver.persona_edited(edited) == 1
        return agent, before, agent.config.allowed_tools

    async def _assert_wholly_on_the_old_definition(
        self,
        host: HostDependencies,
        agent: BaseAgent,
        before: PersonaDefinition,
        tools_before: tuple[str, ...],
    ) -> None:
        """Neither half of the edit is in force, and no epoch was opened for it."""
        llm = cast("_RecordingConnector", host.llm)
        assert agent.persona_definition == before, "the new definition is half in force"
        assert agent.config.allowed_tools == tools_before, "the tool scope moved"
        assert (await agent.execute_turn("Once more")).is_completed
        sent = " ".join(str(m.content) for m in llm.requests[-1].messages)
        assert "You are a poet." not in sent, "the new prompt went out"
        assert _own_tools_offered(llm) == {"draft_chapter", "read_outline"}
        agent.persist_session()
        state = cast("SessionStore", host.store).load(agent.session_id)
        assert state is not None
        assert all(EPOCH_PERSONA_EDITED not in e.opened_by for e in state.context_epochs)

    @pytest.mark.asyncio
    async def test_an_edit_whose_tool_scope_fails_leaves_the_prompt_as_it_was(
        self,
        host: HostDependencies,
        persona_registry: PersonaRegistry,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A raise while the new tool scope is worked out applies none of the edit (#1904).

        The apply wrote the definition into the store and then recomputed the tool scope.
        The prompt is read from the store, so a raise in between left the seat speaking
        with the new prompt under the old tools, with no `persona_edited` epoch. Now the
        store and the scope are worked out first and assigned together.

        The mutation below writes the store before the scope is worked out.

        Killed by: src/uclone_x/agent/base.py :: store = {**self._persona_store, persona.name: persona}
        Becomes: store = self._persona_store; store[persona.name] = persona
        """
        agent, before, tools_before = await self._seat_with_a_failing_edit(host, persona_registry)

        def _refuse(_store: object) -> None:
            raise RuntimeError("the tool scope could not be resolved")

        monkeypatch.setattr(agent, "_persona_scoped_config", _refuse)
        failed = await agent.execute_turn("Again")
        monkeypatch.undo()

        assert not failed.is_completed
        _assert_told_plainly(failed, "the tool scope could not be resolved")
        await self._assert_wholly_on_the_old_definition(host, agent, before, tools_before)

    @pytest.mark.asyncio
    async def test_an_edit_whose_epoch_fails_to_open_is_not_applied(
        self,
        host: HostDependencies,
        persona_registry: PersonaRegistry,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A raise while the edit's epoch is declared leaves the seat on its old definition.

        The owner's ruling ties the two together: a persona edit opens a new epoch. An edit
        applied before its epoch is declared would, on a raise there, put the new identity
        inside the old epoch -- a context that changed without a boundary (#1904).

        The mutation below applies the edit before the epoch is declared.

        Killed by: src/uclone_x/agent/turn_executor.py :: self._active_session.declare_new_epoch(EPOCH_PERSONA_EDITED)
        Becomes: apply_edit(); self._active_session.declare_new_epoch(EPOCH_PERSONA_EDITED)
        """
        agent, before, tools_before = await self._seat_with_a_failing_edit(host, persona_registry)
        # The live session class is module-private; reached by name, as `monkeypatch` does.
        live = getattr(session_lifecycle, "_LiveSession")  # noqa: B009
        declare = live.declare_new_epoch

        def _refuse(session: Any, cause: str) -> None:
            if cause == EPOCH_PERSONA_EDITED:
                raise RuntimeError("the epoch could not be opened")
            declare(session, cause)

        monkeypatch.setattr(live, "declare_new_epoch", _refuse)
        failed = await agent.execute_turn("Again")
        monkeypatch.undo()

        assert not failed.is_completed
        _assert_told_plainly(failed, "the epoch could not be opened")
        await self._assert_wholly_on_the_old_definition(host, agent, before, tools_before)

    @pytest.mark.asyncio
    async def test_a_held_edit_never_overrides_a_later_registry_change(
        self, host: HostDependencies, persona_registry: PersonaRegistry
    ) -> None:
        """A seat built after a save takes the save only while the registry lags it (#1904).

        The resolver holds each saved edit for seats built later, since its registry may
        not have been the one the save wrote. It never let go of one, so a definition put
        in the registry afterwards by another path lost to the older saved copy on every
        seat built from then on.

        Killed by: src/uclone_x/room/resolver.py :: if self._app.persona_registry.get_persona(edited.name) != registry_at_save:
        Becomes: if False:
        """
        resolver = RoomAgentResolver(host, persona_registry=persona_registry)
        original = persona_registry.get_persona("novelist")
        assert original is not None

        def _seat(room: str) -> Participant:
            # The one clone seated in another room: a seat is its clone's id.
            return Participant(
                id="novelist",
                kind=ParticipantKind.AGENT,
                display_name="Novelist",
                session_id=f"sess_room__{room}__novelist",
            )

        # Saved where this resolver's registry does not see it: a seat built now takes it.
        saved = original.model_copy(update={"system_prompt": "You are a poet."})
        assert resolver.persona_edited(saved) == 0
        first = await resolver.resolve(_seat("r1"))
        assert isinstance(first, BaseAgent)
        assert first.persona_definition == saved

        # The registry then moves on by another path: the held copy is older than it.
        later = original.model_copy(update={"system_prompt": "You are an essayist."})
        persona_registry.register_persona(later)
        second = await resolver.resolve(_seat("r2"))
        assert isinstance(second, BaseAgent)
        assert second.persona_definition == later


class TestRoomServicePersonaAutoHydration:
    def test_service_add_participant_auto_hydrates_persona_summary(
        self, tmp_path: Path, persona_registry: PersonaRegistry
    ) -> None:
        store = RoomStore(tmp_path / "rooms")
        service = RoomService(store)
        service.create("Novel Collab", room_id="collab_1")

        state = service.add_participant("collab_1", "writer")

        writer = next(p for p in state.participants if p.id == "writer")
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
        service.add_participant("room_story", "novelist")
        service.add_participant("room_story", "critic", aliases=("reviewer",))

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
            content="Hello @novelist, please draft the opening scene!",
        )

        utterances = [m for m in post_state.transcript if m.is_utterance]
        assert len(utterances) == 2
        assert utterances[0].sender_id == "human_user"
        assert utterances[1].sender_id == "novelist"
