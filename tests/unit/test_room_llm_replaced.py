"""A model chosen in Settings reaches the conversations already open (#1446, model-gateway §3.3).

A conversation's seats and its routing model are built once per room and cached. Settings
used to replace the session manager's one connector while every open conversation kept the
old one. With the gateway, a change to the connections or default models re-binds every
open seat and the routing chain (`AgentSessionManager.models_changed`).

Each test saves through `AgentSessionManager.save_default_models`, the path `POST
/api/settings` takes, over `mock` connections so that no model server is needed.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, cast

import pytest

from uclone_x.agent.base import BaseAgent
from uclone_x.agent.composition import MissingCapabilityError
from uclone_x.llm import MockLLMConnector
from uclone_x.llm.connections import ModelRef
from uclone_x.room.models import ParticipantKind, RoomPolicy
from uclone_x.ui.app import create_ui_app
from uclone_x.ui.rooms import (
    NO_MODEL_REASON,
    RoomStack,
    _http_error,  # pyright: ignore[reportPrivateUsage]
    reader_facing_reason,
)

_SETTINGS_ENV = ("LLM_PROVIDER", "LLM_MODEL")

#: Two connections of the `mock` kind: each builds its own `MockLLMConnector`.
_CONNECTIONS = {"connections": [{"id": "mock", "kind": "mock"}, {"id": "box", "kind": "mock"}]}


@pytest.fixture(autouse=True)
def _restore_settings_env(  # pyright: ignore[reportUnusedFunction]
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in _SETTINGS_ENV:
        monkeypatch.delenv(name, raising=False)


def _stack(tmp_path: Path, llm: Any = None, settings: Any = None) -> tuple[RoomStack, Any]:
    storage = tmp_path / "sessions"
    storage.mkdir(parents=True, exist_ok=True)
    (storage / "settings.json").write_text(json.dumps(settings or _CONNECTIONS), encoding="utf-8")
    app = create_ui_app(static_dir=tmp_path, storage_dir=storage, llm=llm)
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


def _connector(mgr: Any, ref: str) -> Any:
    return mgr.gateway.connector_for(ModelRef.parse(ref))


def test_a_conversation_first_used_with_no_model_answers_once_one_is_chosen(
    tmp_path: Path,
) -> None:
    """The reproduction in #1446: fail with no model, choose one, send again in the same room.

    Killed by: src/uclone_x/agent/composition.py :: missing=tuple(missing),
    Becomes:
    Killed by: src/uclone_x/agent/clone_builder.py :: host = dataclasses.replace(host, llm=seat.llm)
    Becomes: pass
    """
    stack, mgr = _stack(tmp_path)
    room_id = _room(stack)

    with pytest.raises(MissingCapabilityError) as refused:
        _seat(stack, room_id)
    assert reader_facing_reason(refused.value) == NO_MODEL_REASON

    mgr.save_default_models({"deep": "mock/mock-model"})

    agent = _seat(stack, room_id)
    assert agent.llm is _connector(mgr, "mock/mock-model")
    assert agent.config.llm_config.model_name == "mock-model"


def test_a_seat_already_speaking_moves_to_the_new_default(tmp_path: Path) -> None:
    """A seat built before the change is the same live agent afterwards, on the new connection.

    Killed by: src/uclone_x/room/resolver.py :: seat = gateway.bind(own)
    Becomes: continue
    """
    stack, mgr = _stack(tmp_path, settings={**_CONNECTIONS, "default_models": {"deep": "mock/a"}})
    room_id = _room(stack)
    agent = _seat(stack, room_id)
    assert agent.llm is _connector(mgr, "mock/a")

    mgr.save_default_models({"deep": "box/b"})

    assert _seat(stack, room_id) is agent, "the seat was rebuilt rather than reloaded"
    assert agent.llm is _connector(mgr, "box/b")
    assert agent.config.llm_config.model_name == "b"


def test_routing_runs_on_the_default_fast_model_and_follows_it(tmp_path: Path) -> None:
    """Room routing belongs to no clone: always the default fast model (§3.4, decision 5).

    Killed by: src/uclone_x/ui/rooms.py :: provider, model = self._session_mgr.gateway.default_fast()
    Becomes: provider, model = self._session_mgr.gateway.default_deep()
    """
    stack, mgr = _stack(
        tmp_path,
        settings={**_CONNECTIONS, "default_models": {"deep": "mock/deep", "fast": "box/fast"}},
    )
    room_id = _room(stack, RoomPolicy(auto_routing=True))
    _seat(stack, room_id)
    orchestrator = stack.orchestrator(cast(Any, stack.store.load(room_id)))

    def providers() -> list[Any]:
        # The chain is private and has no reader; which connector a selector asks is the
        # whole question.
        chain = cast(tuple[Any, ...], orchestrator._selectors)  # pyright: ignore[reportPrivateUsage]
        return [s._provider for s in chain if hasattr(s, "_provider")]

    assert providers() == [_connector(mgr, "box/fast")]

    mgr.save_default_models({"fast": "mock/quick"})

    assert providers() == [_connector(mgr, "mock/quick")]


def test_a_given_connector_answers_a_clone_that_follows_the_default(tmp_path: Path) -> None:
    """A head that hands in its own connector (a test's fake) still gets it for followers."""
    given = MockLLMConnector()
    stack, _ = _stack(tmp_path, llm=given)
    agent = _seat(stack, _room(stack))
    assert agent.llm is given


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
