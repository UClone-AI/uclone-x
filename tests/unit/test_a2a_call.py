"""Writer asks Artist for a picture over in-memory A2A (#1558).

What is pinned here, against the issue's acceptance list:

* Artist, called by Writer, makes the image with its own tool, and the file lands in the
  conversation's file list.
* A peer not in the caller's `a2a_peers`, a call from inside a call (depth 2), and a tool
  that needs a person's approval are all refused -- the last without the tool running.
* Artist's steps are charged to Writer's budget.
* Artist keeps no memory: it is not given the memory store or the memory tools (owner
  decision, 2026-09-24; the rule `spawn_subagent` applies to a child, #1431).
* Artist works in Writer's conversation and story (#1555): the same story folder, and a
  story write checked against the lease as Writer's conversation -- refused, in plain
  words, when that conversation is not the one writing the story.

Everything is offline: the models are `MockLLMConnector`s and the "image" tool is a fake
that writes a few bytes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, Field

from uclone_x.a2a.in_memory import A2AInMemoryTransport
from uclone_x.a2a.models import TaskMessage, TaskResult, TaskStatus
from uclone_x.agent.base import BaseAgent
from uclone_x.agent.composition import HostDependencies
from uclone_x.agent.hooks.models import HookAction, HookContext, HookDecision, HookEvent
from uclone_x.agent.models import (
    BASE_MEMORY_TOOLS,
    AgentConfig,
    AgentContext,
    AgentLLMConfig,
    PersonaDefinition,
)
from uclone_x.agent.persona_registry import PersonaRegistry
from uclone_x.agent.session import SessionStore
from uclone_x.core.provenance import Provenance
from uclone_x.engine.event_bus import EventBus
from uclone_x.llm.connectors.mock import MockLLMConnector
from uclone_x.llm.models import LLMRequest, MessageRole, ModelResponse, ToolCallRequest
from uclone_x.memory.store import CrossSessionMemory
from uclone_x.room.a2a_handlers import PersonaTaskHandler, register_persona_handlers
from uclone_x.room.models import Participant, ParticipantKind, RoomPolicy, RoomState
from uclone_x.story.library import StoryLibrary
from uclone_x.telemetry.tracer import TelemetryTracer
from uclone_x.tools.base import BaseTool
from uclone_x.tools.builtin.a2a import (
    A2A_CALL_TOOL_NAME,
    A2A_CALLER_SESSION_KEY,
    A2A_DEPTH_KEY,
    A2A_ROOM_KEY,
    A2A_STORY_KEY,
    A2ACallTool,
)
from uclone_x.tools.models import ToolContext, ToolResultStatus
from uclone_x.tools.registry import ToolRegistry

# --------------------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------------------


class _DrawParams(BaseModel):
    subject: str = Field(default="")


class _Draw(BaseTool[_DrawParams]):
    """Stands in for `generate_image`: writes a file and names it. Counts its runs."""

    name = "draw"
    description = "Draws a picture."
    writes_files = True

    def __init__(self, workspace: Path) -> None:
        super().__init__()
        self.workspace = workspace
        self.runs = 0

    def run(self, params: _DrawParams, context: ToolContext) -> dict[str, Any]:
        self.runs += 1
        target = self.workspace / "images" / f"{params.subject or 'picture'}.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"\x89PNG fake")
        return {"path": f"images/{target.name}"}


class _Sketch(BaseTool[_DrawParams]):
    """Stands in for an Artist tool that saves into the open story, through the lease.

    Records the story folder and conversation its `ToolContext` named, and writes with
    `StoryLibrary.write_for` -- the one check every story write goes through.
    """

    name = "sketch"
    description = "Saves a sketch into the story."
    writes_files = True

    def __init__(self) -> None:
        super().__init__()
        self.roots: list[Path] = []
        self.rooms: list[str | None] = []

    def run(self, params: _DrawParams, context: ToolContext) -> dict[str, Any]:
        library = StoryLibrary(context.require_workspace())
        self.rooms.append(context.room_id)
        if context.story_id is not None:
            self.roots.append(library.root(context.story_id))
        relative = f"art/{params.subject or 'sketch'}.txt"
        library.write_for(context, relative, "a sketch", expected_digest=None)
        return {"path": f"stories/{context.story_id}/{relative}"}


class _Listening(MockLLMConnector):
    """A mock model that keeps every tool result it was shown."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.tool_results: list[str] = []

    async def generate(self, request: LLMRequest) -> ModelResponse:
        self.tool_results.extend(
            m.content or "" for m in request.messages if m.role is MessageRole.TOOL
        )
        return await super().generate(request)


class _AskFor:
    """A hook that puts one tool to a person, as `HumanApprovalHook` does for a write."""

    def __init__(self, tool: str) -> None:
        self.tool = tool

    @property
    def name(self) -> str:
        return "ask_for_draw"

    @property
    def failure_policy(self) -> str:
        return "fail_closed"

    async def dispatch(self, context: HookContext) -> HookDecision:
        if (
            context.event_type == HookEvent.PRE_TOOL_USE
            and context.payload.get("tool_name") == self.tool
        ):
            return HookDecision(action=HookAction.ASK)
        return HookDecision(action=HookAction.ALLOW)


class _Recorder:
    """A peer handler that answers from a script and keeps every message it was sent."""

    def __init__(self, result: TaskResult | None = None) -> None:
        self.messages: list[TaskMessage] = []
        self._result = result

    async def __call__(self, message: TaskMessage) -> TaskResult:
        self.messages.append(message)
        if self._result is not None:
            return self._result.model_copy(update={"task_id": message.task_id})
        return TaskResult(
            task_id=message.task_id,
            status=TaskStatus.COMPLETED,
            output_data={"steps": 3, "paths": ["images/hero.png"], "response": "Done."},
            provenance=Provenance.primary(provider="mock", model="mock-model"),
        )


WRITER = PersonaDefinition(
    name="writer",
    role="Writer",
    system_prompt="You write.",
    allowed_tools=(A2A_CALL_TOOL_NAME,),
    enable_write_tools=True,
    a2a_peers=("artist",),
)


def _artist(**update: Any) -> PersonaDefinition:
    persona = PersonaDefinition(
        name="artist",
        role="Artist",
        system_prompt="You draw.",
        allowed_tools=("draw",),
        enable_write_tools=True,
    )
    return persona.model_copy(update=update)


def _writer(
    tmp_path: Path,
    transport: A2AInMemoryTransport | None,
    *,
    persona: PersonaDefinition = WRITER,
    depth: int = 0,
    llm: MockLLMConnector | None = None,
    session_id: str = "sess_writer",
) -> BaseAgent:
    agent = BaseAgent(
        config=AgentConfig(
            agent_id="writer",
            name="writer",
            enable_write_tools=True,
            llm_config=AgentLLMConfig(model_name="mock-model"),
        ),
        llm=llm or MockLLMConnector(),
        tools=ToolRegistry([A2ACallTool()]),
        context=AgentContext(
            session_id=session_id, agent_id="writer", workspace_root=tmp_path, depth=depth
        ),
        a2a_transport=transport,
    )
    agent.define_persona(persona)
    agent.persona = persona.name
    return agent


def _host(tmp_path: Path, llm: MockLLMConnector, tools: ToolRegistry, **extra: Any) -> Any:
    def build() -> HostDependencies:
        return HostDependencies(
            bus=EventBus(),
            llm=llm,
            tools=tools,
            tracer=TelemetryTracer(),
            store=SessionStore(tmp_path / "sessions"),
            **extra,
        )

    return build


def _registry(*personas: PersonaDefinition) -> PersonaRegistry:
    registry = PersonaRegistry(include_defaults=False)
    for persona in personas:
        registry.register_persona(persona)
    return registry


def _draw_llm(subject: str = "hero") -> MockLLMConnector:
    return MockLLMConnector(
        default_response=f"I drew the {subject}.",
        tool_calls=[ToolCallRequest(id="d1", name="draw", arguments={"subject": subject})],
    )


def _handler(
    tmp_path: Path,
    llm: MockLLMConnector,
    tools: ToolRegistry,
    *,
    persona: PersonaDefinition | None = None,
    **extra: Any,
) -> PersonaTaskHandler:
    return PersonaTaskHandler(
        "artist",
        host_factory=_host(tmp_path, llm, tools, **extra),
        persona_registry=_registry(persona or _artist()),
        workspace_root=tmp_path,
        llm_config=AgentLLMConfig(model_name="mock-model"),
    )


def _message(**metadata: str) -> TaskMessage:
    return TaskMessage(
        task_id="a2a_test",
        session_id="sess_writer",
        input_data={"task": "Draw the hero.", "input": {"subject": "hero"}},
        sender_agent_id="writer",
        target_agent_id="artist",
        metadata={A2A_DEPTH_KEY: "1", **metadata},
    )


# --------------------------------------------------------------------------------------
# The tool: who may be called, how deep, and what it costs the caller
# --------------------------------------------------------------------------------------


class TestWhoMayBeCalled:
    @pytest.mark.asyncio
    async def test_a_peer_not_in_a2a_peers_is_refused_and_never_asked(self, tmp_path: Path) -> None:
        """The allowlist is the caller's own `a2a_peers`, checked before anything is sent.

        Killed by: src/uclone_x/tools/builtin/a2a.py :: if call.agent not in peers:
        Becomes: if False:
        """
        transport = A2AInMemoryTransport()
        critic = _Recorder()
        transport.register_handler("critic", critic)
        writer = _writer(tmp_path, transport)

        record = await writer.execute_tool_call(
            A2A_CALL_TOOL_NAME, {"agent": "critic", "task": "Review this."}
        )

        assert record.status is ToolResultStatus.ERROR
        assert record.error is not None
        assert "not allowed to ask 'critic'" in record.error
        assert "artist" in record.error
        assert critic.messages == []

    @pytest.mark.asyncio
    async def test_a_call_from_inside_a_call_is_refused(self, tmp_path: Path) -> None:
        """Depth 1: an agent answering a peer call cannot call on, even holding a transport.

        Killed by: src/uclone_x/tools/builtin/a2a.py :: if transport is None or agent.context.depth >= 1:
        Becomes: if transport is None:
        """
        transport = A2AInMemoryTransport()
        artist = _Recorder()
        transport.register_handler("artist", artist)
        nested = _writer(tmp_path, transport, depth=1)

        record = await nested.execute_tool_call(
            A2A_CALL_TOOL_NAME, {"agent": "artist", "task": "Draw."}
        )

        assert record.status is ToolResultStatus.ERROR
        assert record.error is not None
        assert "cannot ask another persona" in record.error
        assert artist.messages == []

    @pytest.mark.asyncio
    async def test_a_handler_refuses_a_message_deeper_than_one_level(self, tmp_path: Path) -> None:
        """The handler checks depth too, so the rule does not rest on the tool alone.

        Killed by: src/uclone_x/room/a2a_handlers.py :: if depth != "1":
        Becomes: if False:
        """
        draw = _Draw(tmp_path)
        handler = _handler(tmp_path, _draw_llm(), ToolRegistry([draw]))

        result = await handler(_message(**{A2A_DEPTH_KEY: "2"}))

        assert result.status is TaskStatus.REJECTED
        assert result.error == "A task from another persona cannot be passed on to a third one."
        assert draw.runs == 0

    def test_a2a_call_is_offered_only_to_an_agent_with_someone_to_call(
        self, tmp_path: Path
    ) -> None:
        """Its schema stays out of the turns of every agent that could not use it.

        Killed by: src/uclone_x/agent/base.py :: return persona is not None and bool(persona.a2a_peers)
        Becomes: return persona is not None
        """
        transport = A2AInMemoryTransport()
        offered = [t.name for t in _writer(tmp_path, transport).available_tools()]
        lonely = _writer(tmp_path, transport, persona=WRITER.model_copy(update={"a2a_peers": ()}))

        assert A2A_CALL_TOOL_NAME in offered
        assert A2A_CALL_TOOL_NAME not in [t.name for t in lonely.available_tools()]

    def test_the_called_persona_is_left_no_one_to_call(self) -> None:
        """The persona the called agent runs has no `a2a_call` and no peers.

        Killed by: src/uclone_x/room/a2a_handlers.py :: return persona.model_copy(update={"allowed_tools": tools, "a2a_peers": ()})
        Becomes: return persona.model_copy(update={"allowed_tools": tools})
        """
        chatty = _artist(allowed_tools=("draw", A2A_CALL_TOOL_NAME), a2a_peers=("writer",))

        callee = PersonaTaskHandler._callee_persona(chatty)  # pyright: ignore[reportPrivateUsage]

        assert callee.a2a_peers == ()
        assert A2A_CALL_TOOL_NAME not in callee.allowed_tools


class TestWhatTheCallerSends:
    @pytest.mark.asyncio
    async def test_the_message_carries_the_callers_session_and_its_budget(
        self, tmp_path: Path
    ) -> None:
        """The caller's session goes with the task, so writes can later be checked against
        the conversation that asked for them; the peer's step ceiling is what the caller
        has left.

        Killed by: src/uclone_x/tools/builtin/a2a.py :: A2A_CALLER_SESSION_KEY: ctx.session_id,
        Becomes: A2A_CALLER_SESSION_KEY: "",
        """
        transport = A2AInMemoryTransport()
        artist = _Recorder()
        transport.register_handler("artist", artist)
        writer = _writer(tmp_path, transport, session_id="sess_room__r1__writer")
        left = writer.steps_remaining

        await writer.execute_tool_call(A2A_CALL_TOOL_NAME, {"agent": "artist", "task": "Draw."})

        (sent,) = artist.messages
        assert sent.metadata[A2A_CALLER_SESSION_KEY] == "sess_room__r1__writer"
        assert sent.metadata[A2A_DEPTH_KEY] == "1"
        assert sent.metadata["step_budget"] == str(left)


class TestTheCallersBudget:
    @pytest.mark.asyncio
    async def test_the_peers_steps_are_charged_to_the_caller(self, tmp_path: Path) -> None:
        """P4: three steps the peer took are three steps fewer for the caller.

        Killed by: src/uclone_x/tools/builtin/a2a.py :: agent.consume_steps(_reported_steps(output))
        Becomes: pass
        """
        transport = A2AInMemoryTransport()
        transport.register_handler("artist", _Recorder())
        writer = _writer(tmp_path, transport)
        before = writer.run_steps

        record = await writer.execute_tool_call(
            A2A_CALL_TOOL_NAME, {"agent": "artist", "task": "Draw."}
        )

        assert record.status is ToolResultStatus.SUCCESS
        assert writer.run_steps - before == 3

    @pytest.mark.asyncio
    async def test_a_peer_that_stopped_still_costs_what_it_spent(self, tmp_path: Path) -> None:
        """A failed or stopped peer's steps are charged too, not only a finished one's."""
        transport = A2AInMemoryTransport()
        transport.register_handler(
            "artist",
            _Recorder(
                TaskResult(
                    task_id="x",
                    status=TaskStatus.FAILED,
                    output_data={"steps": 2, "paths": []},
                    error="it ran out of steps",
                    provenance=Provenance.primary(provider="mock", model="mock-model"),
                )
            ),
        )
        writer = _writer(tmp_path, transport)
        before = writer.run_steps

        record = await writer.execute_tool_call(
            A2A_CALL_TOOL_NAME, {"agent": "artist", "task": "Draw."}
        )

        assert record.status is ToolResultStatus.ERROR
        assert writer.run_steps - before == 2

    @pytest.mark.asyncio
    async def test_the_handler_reports_the_steps_its_agent_took(self, tmp_path: Path) -> None:
        """Drawing and then answering is two steps, and the result says so.

        Killed by: src/uclone_x/room/a2a_handlers.py :: "steps": steps,
        Becomes: "steps": 0,
        """
        handler = _handler(tmp_path, _draw_llm(), ToolRegistry([_Draw(tmp_path)]))

        result = await handler(_message())

        assert result.status is TaskStatus.COMPLETED
        assert result.output_data["steps"] == 2


# --------------------------------------------------------------------------------------
# The called agent: its own tools, no memory, no unapproved tool
# --------------------------------------------------------------------------------------


class TestTheCalledAgent:
    @pytest.mark.asyncio
    async def test_writer_gets_the_picture_artist_drew_with_its_own_tool(
        self, tmp_path: Path
    ) -> None:
        """The whole path: tool -> transport -> handler -> a one-off Artist -> its tool.

        Killed by: src/uclone_x/room/a2a_handlers.py :: paths = _written_paths(turn)
        Becomes: paths = []
        """
        draw = _Draw(tmp_path)
        transport = A2AInMemoryTransport()
        names = register_persona_handlers(
            transport,
            host_factory=_host(tmp_path, _draw_llm(), ToolRegistry([draw])),
            persona_registry=_registry(WRITER, _artist()),
            workspace_root=tmp_path,
            llm_config=AgentLLMConfig(model_name="mock-model"),
        )
        writer = _writer(tmp_path, transport)
        before = writer.run_steps

        record = await writer.execute_tool_call(
            A2A_CALL_TOOL_NAME, {"agent": "artist", "task": "Draw the hero."}
        )

        assert names == ("artist",)
        assert record.status is ToolResultStatus.SUCCESS, record.error
        assert draw.runs == 1
        assert isinstance(record.output, dict)
        assert record.output["paths"] == ["images/hero.png"]
        assert record.output["unnamed_writes"] is False
        assert (tmp_path / "images" / "hero.png").is_file()
        assert writer.run_steps - before == 2

    @pytest.mark.asyncio
    async def test_a_tool_needing_approval_is_refused_and_never_runs(self, tmp_path: Path) -> None:
        """Nobody can approve during a peer call, so the task stops as INPUT_REQUIRED,
        naming the tool, and the tool does not run.

        Killed by: src/uclone_x/room/a2a_handlers.py :: self.pending.append(tool)
        Becomes: pass
        """
        draw = _Draw(tmp_path)
        handler = _handler(tmp_path, _draw_llm(), ToolRegistry([draw]), hooks=(_AskFor("draw"),))

        result = await handler(_message())

        assert result.status is TaskStatus.INPUT_REQUIRED
        assert result.error == "artist needed approval to use draw, so it did not run."
        assert draw.runs == 0
        assert not (tmp_path / "images").exists()

    @pytest.mark.asyncio
    async def test_the_caller_hears_which_tool_needed_approval(self, tmp_path: Path) -> None:
        """The refusal reaches Writer in plain words, and nothing is resumed."""
        draw = _Draw(tmp_path)
        transport = A2AInMemoryTransport()
        transport.register_handler(
            "artist",
            _handler(tmp_path, _draw_llm(), ToolRegistry([draw]), hooks=(_AskFor("draw"),)),
        )
        writer = _writer(tmp_path, transport)

        record = await writer.execute_tool_call(
            A2A_CALL_TOOL_NAME, {"agent": "artist", "task": "Draw the hero."}
        )

        assert record.status is ToolResultStatus.ERROR
        assert record.error == (
            "'artist' stopped before finishing: "
            "artist needed approval to use draw, so it did not run."
        )
        assert draw.runs == 0

    @pytest.mark.asyncio
    async def test_the_called_artist_cannot_record_a_memory(self, tmp_path: Path) -> None:
        """No memory between calls: the host's store is not handed on, so an Artist asked
        over A2A that tries `record_memory_fact` records nothing (#1431's rule for a child).

        An unrestricted persona on purpose: with no tool list of its own, the store being
        withheld is the only thing between it and the memory tools.

        Killed by: src/uclone_x/room/a2a_handlers.py :: memory=None,
        Becomes: memory=base.memory,
        """
        memory = CrossSessionMemory(storage_path=tmp_path / "artist-memory.json")
        llm = MockLLMConnector(
            default_response="Noted.",
            tool_calls=[
                ToolCallRequest(
                    id="m1",
                    name="record_memory_fact",
                    arguments={
                        "subject": "hero",
                        "predicate": "hair_color",
                        "object_value": "red",
                    },
                )
            ],
        )
        handler = _handler(
            tmp_path, llm, ToolRegistry(), persona=_artist(allowed_tools=()), memory=memory
        )

        await handler(_message())

        assert memory.list_facts() == []

    def test_the_called_artist_is_held_to_its_tools_less_memory(self) -> None:
        """Its persona's list always carries the memory tools; the list it runs under
        does not, and has no `a2a_call` either.

        Killed by: src/uclone_x/room/a2a_handlers.py :: if name not in BASE_MEMORY_TOOLS and name != A2A_CALL_TOOL_NAME
        Becomes: if name != A2A_CALL_TOOL_NAME
        """
        persona = _artist(allowed_tools=("draw", A2A_CALL_TOOL_NAME))
        assert set(BASE_MEMORY_TOOLS) <= set(persona.granted_tools)

        tools = PersonaTaskHandler._callee_tools(persona)  # pyright: ignore[reportPrivateUsage]

        assert "draw" in tools
        assert not set(BASE_MEMORY_TOOLS) & set(tools)
        assert A2A_CALL_TOOL_NAME not in tools


# --------------------------------------------------------------------------------------
# The caller's conversation and story (#1555)
# --------------------------------------------------------------------------------------


def _asks_artist(subject: str = "hero") -> MockLLMConnector:
    return MockLLMConnector(
        default_response="Here it is.",
        tool_calls=[
            ToolCallRequest(
                id="w1",
                name=A2A_CALL_TOOL_NAME,
                arguments={"agent": "artist", "task": f"Sketch the {subject}."},
            )
        ],
    )


def _sketching_artist(tmp_path: Path, sketch: _Sketch, llm: MockLLMConnector) -> Any:
    transport = A2AInMemoryTransport()
    register_persona_handlers(
        transport,
        host_factory=_host(tmp_path, llm, ToolRegistry([sketch])),
        persona_registry=_registry(WRITER, _artist(allowed_tools=("sketch",))),
        workspace_root=tmp_path,
        llm_config=AgentLLMConfig(model_name="mock-model"),
    )
    return transport


def _sketch_llm() -> _Listening:
    return _Listening(
        default_response="I sketched the hero.",
        tool_calls=[ToolCallRequest(id="s1", name="sketch", arguments={"subject": "hero"})],
    )


class TestTheCallersStory:
    @pytest.mark.asyncio
    async def test_the_message_carries_the_callers_conversation_and_story(
        self, tmp_path: Path
    ) -> None:
        """The conversation and the story a seat's turn runs in go with its call.

        Killed by: src/uclone_x/tools/builtin/a2a.py :: metadata[A2A_ROOM_KEY] = ctx.room_id
        Becomes: pass
        Killed by: src/uclone_x/tools/builtin/a2a.py :: metadata[A2A_STORY_KEY] = ctx.story_id
        Becomes: pass
        """
        transport = A2AInMemoryTransport()
        artist = _Recorder()
        transport.register_handler("artist", artist)
        writer = _writer(tmp_path, transport, llm=_asks_artist())

        await writer.execute_turn("Sketch the hero.", room_id="r1", story_id="the-hero")

        (sent,) = artist.messages
        assert sent.metadata[A2A_ROOM_KEY] == "r1"
        assert sent.metadata[A2A_STORY_KEY] == "the-hero"

    @pytest.mark.asyncio
    async def test_a_call_outside_a_conversation_sends_neither(self, tmp_path: Path) -> None:
        """No conversation, no story: nothing is sent in their place."""
        transport = A2AInMemoryTransport()
        artist = _Recorder()
        transport.register_handler("artist", artist)
        writer = _writer(tmp_path, transport)

        await writer.execute_tool_call(A2A_CALL_TOOL_NAME, {"agent": "artist", "task": "Draw."})

        (sent,) = artist.messages
        assert A2A_ROOM_KEY not in sent.metadata
        assert A2A_STORY_KEY not in sent.metadata

    @pytest.mark.asyncio
    async def test_artist_saves_into_the_story_writer_has_open(self, tmp_path: Path) -> None:
        """Same story folder as Writer's, and the write goes through the lease Writer's
        conversation holds.

        Killed by: src/uclone_x/room/a2a_handlers.py :: room_id=_forwarded(message, A2A_ROOM_KEY),
        Becomes: room_id=None,
        Killed by: src/uclone_x/room/a2a_handlers.py :: story_id=_forwarded(message, A2A_STORY_KEY),
        Becomes: story_id=None,
        """
        library = StoryLibrary(tmp_path)
        story = library.create("The Hero", "r1")
        sketch = _Sketch()
        transport = _sketching_artist(tmp_path, sketch, _sketch_llm())
        writer = _writer(tmp_path, transport, llm=_asks_artist())

        turn = await writer.execute_turn("Sketch the hero.", room_id="r1", story_id=story.story_id)

        (call,) = [e for e in turn.tool_executions if e.tool_name == A2A_CALL_TOOL_NAME]
        assert call.status is ToolResultStatus.SUCCESS, call.error
        assert sketch.roots == [library.root(story.story_id)]
        assert sketch.rooms == ["r1"]
        assert library.read_file(story.story_id, "art/hero.txt").text == "a sketch"
        assert isinstance(call.output, dict)
        assert call.output["paths"] == [f"stories/{story.story_id}/art/hero.txt"]

    @pytest.mark.asyncio
    async def test_asked_from_a_conversation_not_writing_the_story_artist_is_read_only(
        self, tmp_path: Path
    ) -> None:
        """Another conversation holds the lease, so Artist -- working for Writer's -- is
        refused as Writer's conversation would be, in plain words, and nothing is written.

        Killed by: src/uclone_x/room/a2a_handlers.py :: room_id=_forwarded(message, A2A_ROOM_KEY),
        Becomes: room_id="r2",
        """
        library = StoryLibrary(tmp_path)
        story = library.create("The Hero", "r2")
        sketch = _Sketch()
        artist_llm = _sketch_llm()
        transport = _sketching_artist(tmp_path, sketch, artist_llm)
        writer = _writer(tmp_path, transport, llm=_asks_artist())

        turn = await writer.execute_turn("Sketch the hero.", room_id="r1", story_id=story.story_id)

        assert sketch.rooms == ["r1"]
        assert not (library.root(story.story_id) / "art").exists()
        (refusal,) = [r for r in artist_llm.tool_results if "nothing was written" in r]
        assert "Another conversation has been writing 'The Hero'" in refusal
        assert "Error" not in refusal and "Traceback" not in refusal
        (call,) = [e for e in turn.tool_executions if e.tool_name == A2A_CALL_TOOL_NAME]
        assert isinstance(call.output, dict)
        assert call.output["paths"] == []


# --------------------------------------------------------------------------------------
# Through a room: the picture is in the conversation's file list
# --------------------------------------------------------------------------------------


class _Seats:
    def __init__(self, agents: dict[str, Any]) -> None:
        self._agents = agents

    async def resolve(self, participant: Participant) -> Any:
        return self._agents[participant.id]


@pytest.mark.asyncio
async def test_the_picture_artist_drew_is_in_the_rooms_file_list(tmp_path: Path) -> None:
    """Acceptance (#1558): Writer, seated in a room, asks Artist -- who is not seated --
    and the file Artist drew is one the conversation wrote, attributed to Writer.

    Killed by: src/uclone_x/tools/builtin/a2a.py :: "paths": list(paths),
    Becomes: "paths": [],
    """
    from uclone_x.room.models import SelectionVerdict, SpeakerDecision
    from uclone_x.room.orchestrator import RoomOrchestrator
    from uclone_x.room.store import RoomStore

    writer_seat = Participant(
        id="writer",
        kind=ParticipantKind.AGENT,
        display_name="Writer",
        session_id="sess_room__r1__writer",
    )
    alice = Participant(id="alice", kind=ParticipantKind.HUMAN, display_name="Alice")
    store = RoomStore(tmp_path / "rooms")
    store.save(RoomState(room_id="r1", participants=(alice, writer_seat), policy=RoomPolicy()))

    draw = _Draw(tmp_path)
    transport = A2AInMemoryTransport()
    register_persona_handlers(
        transport,
        host_factory=_host(tmp_path, _draw_llm(), ToolRegistry([draw])),
        persona_registry=_registry(WRITER, _artist()),
        workspace_root=tmp_path,
        llm_config=AgentLLMConfig(model_name="mock-model"),
    )
    writer = _writer(
        tmp_path,
        transport,
        session_id=writer_seat.session_id,
        llm=MockLLMConnector(
            default_response="Here is the hero.",
            tool_calls=[
                ToolCallRequest(
                    id="w1",
                    name=A2A_CALL_TOOL_NAME,
                    arguments={"agent": "artist", "task": "Draw the hero."},
                )
            ],
        ),
    )

    class _Once:
        name = "once"

        def __init__(self) -> None:
            self.done = False

        async def select(self, request: Any) -> SpeakerDecision:
            if self.done:
                return SpeakerDecision(verdict=SelectionVerdict.ABSTAIN, selector="once")
            self.done = True
            return SpeakerDecision(
                verdict=SelectionVerdict.SPEAK, speaker_id="writer", selector="once"
            )

    orchestrator = RoomOrchestrator(
        store=store, selectors=[_Once()], resolver=_Seats({"writer": writer})
    )
    saved = await orchestrator.post("r1", "alice", "Draw me the hero.")

    assert draw.runs == 1
    assert [(f.path, f.participant_id, f.tool_name) for f in saved.written_files] == [
        ("images/hero.png", "writer", A2A_CALL_TOOL_NAME)
    ]
    assert saved.file_record.unattributed_writes == 0
