"""The §7 conformance check, on a scripted room (design doc `llm-request-layering.md`, #1641).

A room with one human and two seats runs through the real orchestrator, resolver and
`BaseAgent`, with the standard selector chain, and a recording connector captures every
request. The assertions are §7's, over those captured requests:

*   **Prefix**: each request of a seat extends that seat's previous request. The turn
    context is set aside for this check, because its position is a divergence §5.5
    declares (F2); the F2 break itself is pinned below as a strict xfail.
*   **Well-formedness**: no empty message, no two consecutive USER messages, and every
    `tool_calls` answered by TOOL messages before the next ASSISTANT message.
*   **Determinism**: the same inputs give byte-identical requests across two runs, with
    the tool registry filled in a different order the second time.
*   **Selector order** (§5.7): consecutive selector requests share everything up to the
    volatile floor-state header, which comes last.

Known breaks are strict xfails naming the finding, so the day one is fixed the test
says so rather than passing silently: F2 (the turn context is merged on step 1 and a
separate message after a tool step) and the §5.3 slow-context rebuild (teaching an
invariant mid-session rewrites the system message instead of appending an update).
"""

from __future__ import annotations

import json
import re
import shutil
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, Field

from uclone_x.agent.composition import HostDependencies
from uclone_x.agent.models import AgentLLMConfig
from uclone_x.agent.persona_registry import PersonaRegistry
from uclone_x.agent.session import SessionStore
from uclone_x.core.provenance import ExecutionPath, Provenance, ServiceRef
from uclone_x.engine.event_bus import EventBus
from uclone_x.llm.connectors.base import BaseLLMConnector
from uclone_x.llm.models import (
    ChatMessage,
    FinishReason,
    LLMRequest,
    MessageRole,
    ModelResponse,
    StreamChunk,
    TokenUsage,
    ToolCallRequest,
)
from uclone_x.memory.store import CrossSessionMemory
from uclone_x.ontology.engine import OntologyEngine
from uclone_x.room.models import ParticipantKind
from uclone_x.room.orchestrator import RoomOrchestrator
from uclone_x.room.resolver import RoomAgentResolver
from uclone_x.room.selectors import build_selector_chain
from uclone_x.room.service import RoomService
from uclone_x.room.store import RoomStore
from uclone_x.telemetry.tracer import TelemetryTracer
from uclone_x.tools.base import BaseTool
from uclone_x.tools.models import ToolContext
from uclone_x.tools.registry import ToolRegistry

_MODEL = "scripted"
_PROV = Provenance(
    path=ExecutionPath.PRIMARY,
    requested=ServiceRef(provider="scripted", model=_MODEL),
    served_by=ServiceRef(provider="scripted", model=_MODEL),
    attempts=(),
)
_USAGE = TokenUsage(provider="scripted", model=_MODEL, input_tokens=0, output_tokens=0)

_SELECTOR_SYSTEM = "You allocate the floor"
_TURN_CONTEXT = "[Turn Context]"
_ROOM = "room_conformance"

#: Each human message names the seat the scripted selector gives it to, and whether that
#: seat calls a tool. Turns 2 and 3 call one, so both seats run a two-step turn.
_SCRIPT = (
    "turn 1 for scout: where do we stand on the cache?",
    "turn 2 for critic: run lookup on alpha before you answer.",
    "turn 3 for scout: run echo on beta please.",
    "turn 4 for critic: what are the risks?",
    "turn 5 for scout: wrap up.",
)

#: Taught mid-session, before this 1-based script turn, by the §5.3 case only.
_AXIOM_BEFORE_TURN = 4


class _TopicParams(BaseModel):
    topic: str = Field(default="alpha")


class _LookupTool(BaseTool[_TopicParams]):
    name = "lookup"
    description = "Looks a topic up"

    def run(self, params: _TopicParams, context: ToolContext) -> dict[str, Any]:
        return {"topic": params.topic, "status": "nominal"}


class _EchoTool(BaseTool[_TopicParams]):
    name = "echo"
    description = "Echoes the topic back"

    def run(self, params: _TopicParams, context: ToolContext) -> dict[str, Any]:
        return {"echo": params.topic}


def _answer(text: str) -> ModelResponse:
    return ModelResponse(
        finish_reason=FinishReason.STOP, content=text, tool_calls=(), usage=_USAGE, provenance=_PROV
    )


def _call(call_id: str, name: str, topic: str) -> ModelResponse:
    return ModelResponse(
        finish_reason=FinishReason.TOOL_CALLS,
        content=None,
        tool_calls=(ToolCallRequest(id=call_id, name=name, arguments={"topic": topic}),),
        usage=_USAGE,
        provenance=_PROV,
    )


#: A line of conversation in a rendered selector window, whoever said it.
_CONVERSATION_LINE = re.compile(r"^\[(?:user|scout|critic)\]: ", re.MULTILINE)


def _conversation_of(window: str) -> list[str]:
    """The conversation lines of a rendered selector window, in order."""
    return [
        m.group(0) + window[m.end() :].split("\n", 1)[0]
        for m in _CONVERSATION_LINE.finditer(window)
    ]


def _latest_human_line(text: str) -> str:
    """The last `[user]: ...` line of a rendered span or selector window."""
    lines = [line for line in text.splitlines() if line.startswith("[user]: ")]
    return lines[-1] if lines else ""


class _ScriptedRoomLLM(BaseLLMConnector):
    """Answers the selector and both seats, as a function of the request alone.

    Stateless on purpose: a reply that depended on call order would let the determinism
    check pass by construction.
    """

    def __init__(self) -> None:
        super().__init__()
        self.requests: list[LLMRequest] = []

    @property
    def provider_name(self) -> str:
        return "scripted"

    async def generate(self, request: LLMRequest) -> ModelResponse:
        self.requests.append(request)
        system = request.messages[0].content or ""
        if system.startswith(_SELECTOR_SYSTEM):
            return self._select(request.messages[-1].content or "")
        return self._speak(request.messages)

    @staticmethod
    def _select(window: str) -> ModelResponse:
        # One seat answers each human message, then the floor goes quiet.
        if "AGENT TURNS SINCE THE LAST HUMAN MESSAGE: 0" not in window:
            return _answer(json.dumps({"speaker_id": None, "confidence": 1.0}))
        line = _latest_human_line(window)
        seat = line.split(" for ", 1)[1].split(":", 1)[0] if " for " in line else None
        return _answer(json.dumps({"speaker_id": seat, "confidence": 1.0}))

    @staticmethod
    def _speak(messages: Sequence[ChatMessage]) -> ModelResponse:
        if any(m.role is MessageRole.TOOL for m in messages[-2:]):
            return _answer("The tool answered; that settles it.")
        last_user = next(
            (m.content or "" for m in reversed(messages) if m.role is MessageRole.USER), ""
        )
        line = _latest_human_line(last_user.split(_TURN_CONTEXT, 1)[0])
        turn = line.split("turn ", 1)[1].split(" ", 1)[0] if "turn " in line else "0"
        if "run lookup" in line:
            return _call(f"call_{turn}", "lookup", "alpha")
        if "run echo" in line:
            return _call(f"call_{turn}", "echo", "beta")
        return _answer(f"My answer to turn {turn}.")

    async def stream(self, request: LLMRequest) -> AsyncIterator[StreamChunk]:  # pragma: no cover
        yield StreamChunk(delta_content="")


async def _run_room(
    tmp: Path,
    *,
    reverse_tools: bool = False,
    axiom_mid_session: bool = False,
    transcript_window: int | None = None,
) -> list[LLMRequest]:
    """Run `_SCRIPT` through a fresh room under `tmp` and return every request sent."""
    ws = tmp / "ws"
    ws.mkdir(parents=True, exist_ok=True)
    tools: list[BaseTool[Any]] = [_LookupTool(), _EchoTool()]
    registry = ToolRegistry(tools=list(reversed(tools)) if reverse_tools else tools)

    onto = OntologyEngine(agent_id="room")
    onto.teach_axiom(
        name="invoice_positive", subject_entity="Invoice", rule_expression="amount > 0"
    )
    memory = CrossSessionMemory(storage_path=tmp / "memory.json")
    # A fact, so the requests carry a turn-context block for the F2 seam to be about.
    memory.record_fact(
        subject="project",
        predicate="uses",
        object_value="postgres",
        provenance=_PROV,
        source_session_id="sess_earlier",
    )

    llm = _ScriptedRoomLLM()
    host = HostDependencies(
        bus=EventBus(),
        llm=llm,
        tools=registry,
        tracer=TelemetryTracer(),
        store=SessionStore(tmp / "sessions"),
        memory=memory,
        # Off: every seat pins every tool it holds. Host binding appends to the tools layer
        # once per user message (design §5.1), so with it on the prefix check would be
        # measuring binding rather than the layering. test_tool_binder.py covers binding.
        tool_binder=None,
    )

    room_store = RoomStore(tmp / "rooms")
    service = RoomService(room_store)
    state = service.create("Conformance", room_id=_ROOM)
    service.add_participant(_ROOM, "user", kind=ParticipantKind.HUMAN)
    service.add_participant(
        _ROOM, "scout", display_name="Scout", persona_summary="Explores the codebase"
    )
    service.add_participant(
        _ROOM, "critic", display_name="Critic", persona_summary="Reviews adversarially"
    )
    if transcript_window is not None:
        state = room_store.load(_ROOM)
        assert state is not None
        policy = state.policy.model_copy(update={"transcript_window": transcript_window})
        state = room_store.save(state.model_copy(update={"policy": policy}))
    resolver = RoomAgentResolver(
        host,
        llm_config=AgentLLMConfig(model_name=_MODEL, context_limit=65_536),
        ontology_factory=lambda _ns: onto,
        persona_registry=PersonaRegistry(workspace_root=tmp, include_defaults=False),
        workspace_root=ws,
    )
    orch = RoomOrchestrator(
        store=room_store,
        selectors=build_selector_chain(state.policy, provider=llm),
        resolver=resolver,
    )
    for turn, text in enumerate(_SCRIPT, start=1):
        if axiom_mid_session and turn == _AXIOM_BEFORE_TURN:
            onto.teach_axiom(
                name="tax_nonnegative", subject_entity="Invoice", rule_expression="tax >= 0"
            )
        await orch.post(_ROOM, "user", text)
    return llm.requests


def _seat_of(request: LLMRequest) -> str | None:
    """The seat a request was sent for, read off its identity; `None` for the selector."""
    system = request.messages[0].content or ""
    for seat, name in (("scout", "Scout"), ("critic", "Critic")):
        if f"You are {name} ({seat})" in system:
            return seat
    return None


def _selector_requests(requests: list[LLMRequest]) -> list[LLMRequest]:
    return [r for r in requests if (r.messages[0].content or "").startswith(_SELECTOR_SYSTEM)]


def _by_seat(requests: list[LLMRequest]) -> dict[str, list[LLMRequest]]:
    seats: dict[str, list[LLMRequest]] = {"scout": [], "critic": []}
    for request in requests:
        seat = _seat_of(request)
        if seat is not None:
            seats[seat].append(request)
    return seats


def _dump(message: ChatMessage) -> str:
    return message.model_dump_json()


def _without_turn_context(messages: Sequence[ChatMessage]) -> list[ChatMessage]:
    """`messages` with the turn-context block removed, in whichever form it was placed."""
    out: list[ChatMessage] = []
    for message in messages:
        content = message.content or ""
        if message.role is MessageRole.USER and content.startswith(_TURN_CONTEXT):
            continue
        marker = f"\n\n{_TURN_CONTEXT}"
        if message.role is MessageRole.USER and marker in content:
            message = message.model_copy(update={"content": content.split(marker, 1)[0]})
        out.append(message)
    return out


def _is_prefix(shorter: Sequence[ChatMessage], longer: Sequence[ChatMessage]) -> bool:
    return len(shorter) <= len(longer) and all(
        _dump(a) == _dump(b) for a, b in zip(shorter, longer, strict=False)
    )


def _tools_of(request: LLMRequest) -> str:
    return json.dumps([t.model_dump(mode="json") for t in request.tools or ()], sort_keys=True)


@pytest.fixture
async def recorded(tmp_path: Path) -> list[LLMRequest]:
    return await _run_room(tmp_path)


class TestTheScriptRanAsWritten:
    """Every other assertion is vacuous if the room did not do what the script says."""

    @pytest.mark.asyncio
    async def test_both_seats_spoke_and_both_ran_a_tool_step(
        self, recorded: list[LLMRequest]
    ) -> None:
        seats = _by_seat(recorded)
        for seat, requests in seats.items():
            assert len(requests) >= 3, (seat, len(requests))
            assert any(m.role is MessageRole.TOOL for r in requests for m in r.messages), (
                f"{seat} never ran a tool step"
            )
            assert any(_TURN_CONTEXT in (m.content or "") for r in requests for m in r.messages), (
                f"{seat} never carried a turn-context block"
            )
        assert len(_selector_requests(recorded)) >= len(_SCRIPT)


class TestPrefix:
    """§7 Prefix: each request of a seat extends that seat's previous request."""

    @pytest.mark.asyncio
    async def test_each_seat_request_extends_the_previous_one(
        self, recorded: list[LLMRequest]
    ) -> None:
        for seat, requests in _by_seat(recorded).items():
            for i in range(1, len(requests)):
                before, after = requests[i - 1], requests[i]
                assert _tools_of(before) == _tools_of(after), f"{seat} request {i}: tools changed"
                assert _dump(before.messages[0]) == _dump(after.messages[0]), (
                    f"{seat} request {i}: the system message changed"
                )
                assert _is_prefix(
                    _without_turn_context(before.messages), _without_turn_context(after.messages)
                ), f"{seat} request {i} does not extend request {i - 1}"

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "F2 (llm-request-layering.md §5.5): the turn context is merged into the span "
            "message on step 1 and is a separate message after a tool step, so step 2 "
            "does not extend step 1."
        ),
    )
    @pytest.mark.asyncio
    async def test_a_tool_step_extends_the_step_before_it_turn_context_included(
        self, recorded: list[LLMRequest]
    ) -> None:
        for seat, requests in _by_seat(recorded).items():
            for i in range(1, len(requests)):
                before, after = requests[i - 1], requests[i]
                if before.messages[-1].role is MessageRole.TOOL:
                    continue
                if not any(
                    m.role is MessageRole.TOOL for m in after.messages[len(before.messages) - 1 :]
                ):
                    continue
                assert _is_prefix(before.messages, after.messages), (
                    f"{seat}: step request {i} does not extend request {i - 1}"
                )

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "§5.3 slow context: an invariant taught mid-session rewrites the system "
            "message instead of appending a [Context Update] delta to history."
        ),
    )
    @pytest.mark.asyncio
    async def test_a_mid_session_invariant_is_appended_not_rewritten(self, tmp_path: Path) -> None:
        requests = await _run_room(tmp_path, axiom_mid_session=True)
        for seat, seat_requests in _by_seat(requests).items():
            for i in range(1, len(seat_requests)):
                before, after = seat_requests[i - 1], seat_requests[i]
                assert _dump(before.messages[0]) == _dump(after.messages[0]), (
                    f"{seat} request {i}: the system message was rewritten"
                )
        assert any("[Context Update]" in (m.content or "") for r in requests for m in r.messages), (
            "no request carried the invariant as an appended update"
        )


class TestWellFormedness:
    """§7 Well-formedness, over every request the room sent."""

    @pytest.mark.asyncio
    async def test_no_message_is_empty(self, recorded: list[LLMRequest]) -> None:
        for n, request in enumerate(recorded):
            for m in request.messages:
                assert (m.content or "").strip() or m.tool_calls, (n, m.role)

    @pytest.mark.asyncio
    async def test_no_two_user_messages_are_adjacent(self, recorded: list[LLMRequest]) -> None:
        for n, request in enumerate(recorded):
            roles = [m.role for m in request.messages]
            for a, b in zip(roles, roles[1:], strict=False):
                assert not (a is MessageRole.USER and b is MessageRole.USER), (n, roles)

    @pytest.mark.asyncio
    async def test_every_tool_call_is_answered_before_the_next_assistant(
        self, recorded: list[LLMRequest]
    ) -> None:
        for n, request in enumerate(recorded):
            owed: set[str] = set()
            for m in request.messages:
                if m.role is MessageRole.ASSISTANT:
                    assert not owed, (n, "an assistant message before results for", owed)
                    owed = {call.id for call in m.tool_calls or ()}
                elif m.role is MessageRole.TOOL:
                    assert m.tool_call_id in owed, (n, "a result for no open call", m.tool_call_id)
                    owed.discard(m.tool_call_id)
                elif m.role is MessageRole.USER:
                    assert not owed, (n, "a user message before results for", owed)


class TestDeterminism:
    """§7 Determinism: the same inputs give byte-identical requests, tool order included."""

    @pytest.mark.asyncio
    async def test_two_runs_send_the_same_bytes(self, tmp_path: Path) -> None:
        # The same directory both times: the workspace path is in the prompt, and a
        # different path is a different input.
        run_dir = tmp_path / "run"
        first = await _run_room(run_dir)
        shutil.rmtree(run_dir)
        second = await _run_room(run_dir, reverse_tools=True)

        assert len(first) == len(second)
        for n, (a, b) in enumerate(zip(first, second, strict=True)):
            assert a.model_dump_json() == b.model_dump_json(), f"request {n} differs"


class TestSelectorRequests:
    """§5.7: instructions, roster, transcript window, then the volatile header last."""

    @pytest.mark.asyncio
    async def test_the_volatile_header_is_last(self, recorded: list[LLMRequest]) -> None:
        for request in _selector_requests(recorded):
            window = request.messages[-1].content or ""
            # The last line of conversation, whoever said it: anchoring on the last
            # `[user]:` line let a seat's line after the header pass (#1661).
            last_line = max(m.start() for m in _CONVERSATION_LINE.finditer(window))
            assert window.index("ROSTER:") < window.index("CONVERSATION:") < last_line
            assert window.index("AGENT TURNS SINCE THE LAST HUMAN MESSAGE") > last_line
            assert window.index("LAST SPEAKER") > last_line

    @pytest.mark.asyncio
    async def test_each_selection_extends_the_one_before_it(
        self, recorded: list[LLMRequest]
    ) -> None:
        requests = _selector_requests(recorded)
        for i in range(1, len(requests)):
            before, after = requests[i - 1], requests[i]
            assert _dump(before.messages[0]) == _dump(after.messages[0])
            window = before.messages[-1].content or ""
            # The blank line that sets the floor state off is where the next line of
            # conversation goes, so the shared part ends at the last line before it.
            stable = (
                window[: window.index("AGENT TURNS SINCE THE LAST HUMAN MESSAGE")].rstrip("\n")
                + "\n"
            )
            assert "CONVERSATION:" in stable, "the floor state comes before the conversation"
            assert (after.messages[-1].content or "").startswith(stable), (
                f"selector request {i} does not extend request {i - 1}"
            )


#: Smaller than the script's ten utterances, and above the policy's turn ceiling of 3.
_SLID_WINDOW = 4


class TestSelectorRequestsOnceTheWindowSlides:
    """§5.7 once the conversation outgrows `transcript_window` (#1661).

    The script alone never outgrew the default window of 15, so every selector request
    above saw the whole conversation. Once the window slides, a request no longer extends
    the one before it -- its first line of conversation is gone -- and what is still
    shared is the instructions and the roster, with the floor state still last.
    """

    @pytest.fixture
    async def slid(self, tmp_path: Path) -> list[LLMRequest]:
        return _selector_requests(await _run_room(tmp_path, transcript_window=_SLID_WINDOW))

    @pytest.mark.asyncio
    async def test_the_window_slid_and_holds_at_most_its_size(self, slid: list[LLMRequest]) -> None:
        """Killed by: src/uclone_x/room/orchestrator.py :: windowed = self.window_over_utterances(state.transcript, state.policy.transcript_window)
        Becomes: windowed = state.transcript
        """
        windows = [_conversation_of(r.messages[-1].content or "") for r in slid]
        assert all(len(lines) <= _SLID_WINDOW for lines in windows), [len(w) for w in windows]
        assert any(
            before and after[: len(before)] != before
            for before, after in zip(windows, windows[1:], strict=False)
        ), "the window never slid, so nothing below is about a slid window"

    @pytest.mark.asyncio
    async def test_the_instructions_and_roster_stay_shared_and_the_header_last(
        self, slid: list[LLMRequest]
    ) -> None:
        stable = {
            (
                r.messages[0].model_dump_json(),
                (r.messages[-1].content or "").split("CONVERSATION:")[0],
            )
            for r in slid
        }
        assert len(stable) == 1, "the instructions or the roster changed between selections"
        for request in slid:
            window = request.messages[-1].content or ""
            last_line = max(m.start() for m in _CONVERSATION_LINE.finditer(window))
            assert window.index("AGENT TURNS SINCE THE LAST HUMAN MESSAGE") > last_line
            assert window.index("LAST SPEAKER") > last_line

    @pytest.mark.asyncio
    async def test_every_selection_still_sees_the_message_it_answers(
        self, slid: list[LLMRequest]
    ) -> None:
        latest = [_latest_human_line(r.messages[-1].content or "") for r in slid]
        assert {line.split(" for ", 1)[0] for line in latest} >= {
            f"[user]: turn {n}" for n in range(1, len(_SCRIPT) + 1)
        }
