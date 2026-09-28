"""§7 parity: a clone's request from a 1:1 chat equals its request from a room seat.

Design doc `llm-request-layering.md` §5.9 and §7 (owner ruling 2026-09-27): a 1:1 chat is a
room with one seat, and a clone builds its request one way wherever it is seated. So for
one clone and one person's message, the request sent from a one-seat room and the request
sent from a seat in a room with another clone differ in two places only: the seat framing
that leads the room seat's identity layer, which a one-seat room does not send (§5.9.3),
and the rendering of the span in the conversation layer.

Both arms run through the desktop head as it is shipped -- `AgentSessionManager` for the
app scope, `RoomStack` for the room, the real resolver, `build_clone` and `BaseAgent` --
with a connector that records every request. Nothing about a request is built by hand
here. The two arms run one after the other in the same fresh directory, so paths that
reach the prompt are the same in both.

Everything but the two named parts is compared whole: the rest of the system message, the
turn context (memory lives there), the tool schemas, the model and the sampling settings.
A feature wired into the 1:1 head only, or into the multi-seat room only, shows up as a
difference in one of those.
"""

from __future__ import annotations

import asyncio
import json
import shutil
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from uclone_x.core.provenance import ExecutionPath, Provenance, ServiceRef
from uclone_x.llm.connectors.base import BaseLLMConnector
from uclone_x.llm.models import (
    FinishReason,
    LLMRequest,
    MessageRole,
    ModelResponse,
    StreamChunk,
    TokenUsage,
)
from uclone_x.memory.extractor import INSTRUCTIONS_OPENING
from uclone_x.room.models import ParticipantKind
from uclone_x.room.resolver import room_participant_system_prompt

_MODEL = "parity-model"
_PROV = Provenance(
    path=ExecutionPath.PRIMARY,
    requested=ServiceRef(provider="parity", model=_MODEL),
    served_by=ServiceRef(provider="parity", model=_MODEL),
    attempts=(),
)
_USAGE = TokenUsage(provider="parity", model=_MODEL, input_tokens=0, output_tokens=0)

_SELECTOR_SYSTEM = "You allocate the floor"
_TURN_CONTEXT = "[Turn Context]"
_ROOM = "room_parity"
_HUMAN = "user"
_CLONE = "writer"
_OTHER = "champion"
#: The one person's message both arms answer. It names the clone, so the multi-seat room
#: gives it the floor without a routing call; the one-seat room gives it the floor anyway.
_MESSAGE = f"@{_CLONE} how should chapter one open?"
#: What the other seat is asked first, so the clone's span in the room holds more than
#: the person's message -- otherwise the two spans would be equal and prove nothing.
_OTHER_FIRST = f"@{_OTHER} give us a one-line premise."
#: What every seat answers.
_REPLY = "A line of my own."


class _RecordingLLM(BaseLLMConnector):
    """Answers every seat with a fixed line and every routing call with silence.

    Records every request but the clone learning from its turn afterwards (#1404), which
    is not part of the turn and is answered with no facts.
    """

    def __init__(self) -> None:
        super().__init__()
        self.requests: list[LLMRequest] = []

    @property
    def provider_name(self) -> str:
        return "parity"

    async def generate(self, request: LLMRequest) -> ModelResponse:
        system = request.messages[0].content or "" if request.messages else ""
        if system.startswith(INSTRUCTIONS_OPENING):
            content = "[]"
        else:
            self.requests.append(request)
            content = (
                json.dumps({"speaker_id": None, "confidence": 1.0})
                if system.startswith(_SELECTOR_SYSTEM)
                else _REPLY
            )
        return ModelResponse(
            finish_reason=FinishReason.STOP,
            content=content,
            tool_calls=(),
            usage=_USAGE,
            provenance=_PROV,
        )

    async def stream(self, request: LLMRequest) -> AsyncIterator[StreamChunk]:
        # A room turn streams (design D2), so a seat's request arrives here.
        response = await self.generate(request)
        yield StreamChunk(
            delta_content=response.content or "",
            finish_reason=response.finish_reason,
            usage=response.usage,
        )


def _clone_request(tmp: Path, *, other_seat: bool) -> tuple[LLMRequest, str]:
    """The clone's request answering `_MESSAGE`, and the multi-agent framing of its seat.

    The framing is what `room_participant_system_prompt` renders for the seat, returned
    for both arms: the room arm must lead with it, the one-seat arm must not carry it.

    `other_seat=False` is the 1:1 chat: a room with the person and the clone. `True` seats
    a second clone, which speaks first, so the clone's span carries its line too.
    """
    from uclone_x.core.agent_home import AGENTS_DIR_ENV_VAR
    from uclone_x.ui.app import AgentSessionManager
    from uclone_x.ui.rooms import RoomStack

    llm = _RecordingLLM()
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv(AGENTS_DIR_ENV_VAR, str(tmp / "agents"))
        mgr = AgentSessionManager(
            storage_dir=tmp / "sessions", llm=llm, workspace_dir=tmp / "workspace"
        )
        # Clone-scope memory, so the turn context carries something to compare. It is about
        # chapter one because recall ranks facts against the message (clone-knowledge-graph
        # §3.5), and a fact the message does not touch is counted, not sent.
        mgr.memory_for(_CLONE).record_fact(
            subject="chapter_one",
            predicate="opens_in",
            object_value="Busan",
            provenance=_PROV,
            source_session_id="sess_earlier",
        )
        stack = RoomStack(mgr)
        stack.service.create(title="Chapter one", room_id=_ROOM)
        stack.service.add_participant(_ROOM, _HUMAN, kind=ParticipantKind.HUMAN)
        stack.service.add_participant(_ROOM, _CLONE, kind=ParticipantKind.AGENT)
        if other_seat:
            stack.service.add_participant(_ROOM, _OTHER, kind=ParticipantKind.AGENT)
        state = stack.store.load(_ROOM)
        assert state is not None
        orchestrator = stack.orchestrator(state)

        async def converse() -> None:
            if other_seat:
                await orchestrator.post(_ROOM, _HUMAN, _OTHER_FIRST)
            await orchestrator.post(_ROOM, _HUMAN, _MESSAGE)

        asyncio.run(converse())
        seat = next(p for p in state.participants if p.id == _CLONE)
        framing = room_participant_system_prompt(seat)

    seats = [
        r
        for r in llm.requests
        if r.messages and not (r.messages[0].content or "").startswith(_SELECTOR_SYSTEM)
    ]
    # The room arm has two seat requests, one per clone; the other clone's does not lead
    # with this seat's framing. The one-seat arm has only the clone's.
    ours = [r for r in seats if (r.messages[0].content or "").startswith(framing)]
    if not other_seat:
        assert not ours, "a one-seat room sent the multi-agent seat framing (§5.9.3)"
        ours = seats
    assert len(ours) == 1, f"expected one request from {_CLONE}, got {len(ours)}"
    return ours[0], framing


def _span_and_rest(request: LLMRequest, framing: str | None) -> tuple[str, dict[str, Any]]:
    """The span the request carries, and the request with the framing and span set aside.

    Framing (`None` for the one-seat arm, which has none to set aside): the prefix of the
    system message, with the header `compose_identity_prompt` puts between it and the
    persona's instructions. What is left is what a one-seat room sends, so it is compared
    as is. Span: the last USER message's text up to the turn-context block, which stays in
    the comparison.
    """
    dumped = request.model_dump(mode="json")
    messages: list[dict[str, Any]] = dumped["messages"]
    if framing is not None:
        system = messages[0]["content"] or ""
        header = f"{framing}\n\n[Persona Instructions: "
        assert system.startswith(header), "the seat framing is not where §5.2 puts it"
        end = system.index("]\n", len(header)) + len("]\n")
        messages[0]["content"] = system[end:]

    last_user = max(i for i, m in enumerate(request.messages) if m.role is MessageRole.USER)
    content = messages[last_user]["content"] or ""
    span, marker, context = content.partition(f"\n\n{_TURN_CONTEXT}")
    messages[last_user]["content"] = "<span>" + marker + context
    return span, dumped


def test_a_one_seat_rooms_request_equals_a_room_seats_but_for_framing_and_span(
    tmp_path: Path,
) -> None:
    """One clone, one message: 1:1 and room seat send the same request, bar two parts.

    Killed by: src/uclone_x/ui/rooms.py :: self._session_mgr.app_scope(),
    Becomes: (self._session_mgr.app_scope() if len(state.participants) < 3 else __import__("dataclasses").replace(self._session_mgr.app_scope(), memory_for=None)),
    """
    one_seat, _ = _clone_request(tmp_path, other_seat=False)
    shutil.rmtree(tmp_path)
    tmp_path.mkdir()
    room_seat, room_framing = _clone_request(tmp_path, other_seat=True)

    one_span, one_rest = _span_and_rest(one_seat, None)
    room_span, room_rest = _span_and_rest(room_seat, room_framing)

    # The two parts set aside are the ones that are allowed to differ, so each is held to
    # exactly what it must be: the 1:1 span is the person's message and nothing else; the
    # room span is the exchange the clone missed, then the message. Anything a head adds to
    # the span in one arm only -- a rule, a note, a reminder -- fails here, not nowhere.
    assert one_span == f"[{_HUMAN}]: {_MESSAGE}", one_span
    assert room_span == (
        f"[{_HUMAN}]: {_OTHER_FIRST}\n[{_OTHER}]: {_REPLY}\n[{_HUMAN}]: {_MESSAGE}"
    ), room_span
    # What is compared is not empty: the memory fact reaches the turn context, and the
    # clone is offered tools.
    assert "Busan" in json.dumps(one_rest["messages"])
    assert one_rest["tools"], "no tool schemas were sent, so the tools layer went unchecked"

    assert room_rest == one_rest


def test_the_seat_framing_follows_the_roster_across_joins_leaves_and_restarts(
    tmp_path: Path,
) -> None:
    """A clone alone is framed as a 1:1 chat; a second clone frames it, and leaving unframes.

    §5.9.3 (owner ruling 2026-09-27): the anchor is written once and not rewritten, so a
    roster that crosses between one clone and two changes what the turn sends -- one
    prompt-cache miss -- and leaves the stored anchor as it was. A process that restarts
    into the changed roster reads the framing back from the anchor and does the same.

    Killed by: src/uclone_x/agent/base.py :: return recorded is not None and recorded != canonical(self._config.seat_framing)
    Becomes: return False
    """
    from uclone_x.core.agent_home import AGENTS_DIR_ENV_VAR
    from uclone_x.ui.app import AgentSessionManager
    from uclone_x.ui.rooms import RoomStack

    llm = _RecordingLLM()

    def seat_systems() -> list[str]:
        return [
            r.messages[0].content or ""
            for r in llm.requests
            if r.messages and not (r.messages[0].content or "").startswith(_SELECTOR_SYSTEM)
        ]

    with pytest.MonkeyPatch.context() as mp:
        mp.setenv(AGENTS_DIR_ENV_VAR, str(tmp_path / "agents"))

        def open_room() -> RoomStack:
            mgr = AgentSessionManager(
                storage_dir=tmp_path / "sessions", llm=llm, workspace_dir=tmp_path / "workspace"
            )
            return RoomStack(mgr)

        stack = open_room()
        stack.service.create(title="Chapter one", room_id=_ROOM)
        stack.service.add_participant(_ROOM, _HUMAN, kind=ParticipantKind.HUMAN)
        state = stack.service.add_participant(_ROOM, _CLONE, kind=ParticipantKind.AGENT)
        seat = next(p for p in state.participants if p.id == _CLONE)
        framing = room_participant_system_prompt(seat)
        orchestrator = stack.orchestrator(state)

        asyncio.run(orchestrator.post(_ROOM, _HUMAN, _MESSAGE))
        alone = seat_systems()[-1]
        assert framing not in alone

        stack.service.add_participant(_ROOM, _OTHER, kind=ParticipantKind.AGENT)
        asyncio.run(orchestrator.post(_ROOM, _HUMAN, f"@{_CLONE} and now?"))
        joined = seat_systems()[-1]
        assert joined.startswith(f"{framing}\n\n[Persona Instructions: ")
        # The recomposed identity is what is sent; the record keeps the anchor it had.
        agent = stack.live_agent(_ROOM, seat.session_id)
        assert agent is not None
        anchor = agent.get_session(seat.session_id).messages[0].content or ""
        assert framing not in anchor

        stack.service.remove_participant(_ROOM, _OTHER)
        asyncio.run(orchestrator.post(_ROOM, _HUMAN, f"@{_CLONE} once more."))
        assert seat_systems()[-1] == alone

        # A new process, into a room that has a second clone again.
        stack.service.add_participant(_ROOM, _OTHER, kind=ParticipantKind.AGENT)
        restarted = open_room()
        reopened = restarted.store.load(_ROOM)
        assert reopened is not None
        asyncio.run(restarted.orchestrator(reopened).post(_ROOM, _HUMAN, f"@{_CLONE} again."))
        assert seat_systems()[-1] == joined
