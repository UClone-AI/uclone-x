"""A model chosen in Settings reaches the conversations already open (#1446).

A conversation's seats and its routing model are built once per room and cached, from the
connector of that moment. Settings replaced the session manager's connector and reloaded
the chat agents, and said "applied", while every open conversation kept the old one --
`None`, for a room first used before a model was chosen, so it kept failing with "a
problem in the agent runtime" after the user had done exactly what would fix it.

Each test goes through `AgentSessionManager.update_settings`, the path the Settings route
calls, with the `mock` provider so that no model server is needed.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, cast

import pytest

from uclone_x.agent.base import BaseAgent
from uclone_x.agent.composition import MissingCapabilityError
from uclone_x.llm import MockLLMConnector
from uclone_x.room.models import ParticipantKind, RoomPolicy
from uclone_x.ui.app import create_ui_app
from uclone_x.ui.rooms import (
    NO_MODEL_REASON,
    RoomStack,
    _http_error,  # pyright: ignore[reportPrivateUsage]
    reader_facing_reason,
)

#: Every variable `update_settings` writes for these arguments. Touched through
#: `monkeypatch` first so that it puts back whatever the machine had.
_SETTINGS_ENV = ("LLM_PROVIDER", "LLM_MODEL")


@pytest.fixture(autouse=True)
def _restore_settings_env(  # pyright: ignore[reportUnusedFunction]
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in _SETTINGS_ENV:
        monkeypatch.delenv(name, raising=False)


def _stack(tmp_path: Path, llm: Any) -> tuple[RoomStack, Any]:
    app = create_ui_app(static_dir=tmp_path, storage_dir=tmp_path / "sessions", llm=llm)
    return cast(RoomStack, app.state.room_stack), app.state.session_manager


def _room(stack: RoomStack, policy: RoomPolicy | None = None) -> str:
    room_id = stack.service.create("Index tuning", policy=policy).room_id
    stack.service.add_participant(room_id, "user", kind=ParticipantKind.HUMAN)
    stack.service.add_participant(room_id, "scout")
    return room_id


def _seat(stack: RoomStack, room_id: str) -> BaseAgent:
    state = stack.store.load(room_id)
    assert state is not None
    participant = next(p for p in state.participants if p.id == "scout")
    return asyncio.run(stack.resolve_agent(state, participant))


def test_a_conversation_first_used_with_no_model_answers_once_one_is_chosen(
    tmp_path: Path,
) -> None:
    """The reproduction in #1446: fail with no model, choose one, send again in the same room.

    Killed by: src/uclone_x/agent/composition.py :: missing=tuple(missing),
    Becomes:
    Killed by: src/uclone_x/ui/app.py :: listener(new_llm)
    Becomes: pass
    Killed by: src/uclone_x/room/resolver.py :: self._host = dataclasses.replace(self._host, llm=llm)
    Becomes: pass
    """
    stack, mgr = _stack(tmp_path, llm=None)
    room_id = _room(stack)

    with pytest.raises(MissingCapabilityError) as refused:
        _seat(stack, room_id)
    assert reader_facing_reason(refused.value) == NO_MODEL_REASON

    mgr.update_settings(llm_provider="mock", llm_model="mock-model")

    agent = _seat(stack, room_id)
    assert agent.llm is mgr.llm
    assert agent.llm is not None


def test_a_seat_already_speaking_moves_to_the_new_connector(tmp_path: Path) -> None:
    """A seat built before the change is the same live agent afterwards, on the new connector.

    Killed by: src/uclone_x/room/resolver.py :: agent.hot_reload_llm(llm)
    Becomes: pass
    """
    first = MockLLMConnector()
    stack, mgr = _stack(tmp_path, llm=first)
    room_id = _room(stack)
    agent = _seat(stack, room_id)
    assert agent.llm is first

    mgr.update_settings(llm_provider="mock", llm_model="mock-model")

    assert mgr.llm is not first
    assert _seat(stack, room_id) is agent, "the seat was rebuilt rather than reloaded"
    assert agent.llm is mgr.llm


def test_a_conversation_that_routes_by_model_routes_with_the_new_one(tmp_path: Path) -> None:
    """The routing model is built from the connector too, and has to follow it.

    Killed by: src/uclone_x/ui/rooms.py :: build_selector_chain(state.policy, provider=llm)
    Becomes: build_selector_chain(state.policy, provider=None)
    """
    first = MockLLMConnector()
    stack, mgr = _stack(tmp_path, llm=first)
    room_id = _room(stack, RoomPolicy(auto_routing=True))
    _seat(stack, room_id)
    orchestrator = stack.orchestrator(cast(Any, stack.store.load(room_id)))

    def providers() -> list[Any]:
        # The chain is private and has no reader; which connector a selector asks is the
        # whole question, and asking it through a routed turn needs two agents and a
        # scripted reply from each connector for no more information than this.
        chain = cast(tuple[Any, ...], orchestrator._selectors)  # pyright: ignore[reportPrivateUsage]
        return [s._provider for s in chain if hasattr(s, "_provider")]

    assert providers() == [first]

    mgr.update_settings(llm_provider="mock", llm_model="mock-model")

    assert providers() == [mgr.llm]


class TestNoModelRefusal:
    def test_it_names_the_cause_and_the_remedy_in_plain_words(self) -> None:
        """
        Killed by: src/uclone_x/ui/rooms.py :: return NO_MODEL_REASON
        Becomes: pass
        """
        reason = reader_facing_reason(MissingCapabilityError("…: llm", missing=("llm",)))

        assert "No model is selected" in reason
        assert "Settings" in reason
        for internal in ("llm", "capabilit", "composition", "MissingCapabilityError"):
            assert internal not in reason

    def test_a_capability_the_reader_cannot_supply_stays_a_runtime_fault(self) -> None:
        reason = reader_facing_reason(MissingCapabilityError("…: tracer", missing=("tracer",)))

        assert reason != NO_MODEL_REASON
        assert "tracer" not in reason

    def test_a_retry_that_meets_it_is_answered_the_same_way(self) -> None:
        """
        Killed by: src/uclone_x/ui/rooms.py :: return HTTPException(status_code=409, detail=NO_MODEL_REASON)
        Becomes: pass
        """
        error = _http_error(MissingCapabilityError("…: llm", missing=("llm",)))

        assert (error.status_code, error.detail) == (409, NO_MODEL_REASON)
