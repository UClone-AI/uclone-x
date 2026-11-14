"""A model chosen in the clone editor reaches that clone's open conversations (#2418).

Owner rulings 2026-10-29: a clone's model can be changed in the middle of a conversation,
taking effect from the next turn, with no restart and no new conversation; the change is
made in the clone editor, never by a picker beside the composer; and a clone with no fast
model of its own uses its own conversation model for its fast work.

A save used to stage only the persona on each open seat (#1899), so the seat kept the
connector and model ids it was built with. Each test here saves through the path the
editor takes (`PUT /api/clones/{ref}`, or `RoomStack.persona_edited`, which that route
calls), over `mock` connections so that no model server is needed. A 1:1 chat is a
one-seat room (§5.9.3), so the same resolver serves both; both are exercised.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, cast

import pytest
from fastapi.testclient import TestClient

from uclone_x.agent.base import BaseAgent
from uclone_x.agent.models import TurnResult
from uclone_x.core.models import AgentLLMConfig
from uclone_x.llm import MockLLMConnector
from uclone_x.llm.connections import ModelRef
from uclone_x.llm.gateway import DefaultBinding, ModelGateway
from uclone_x.llm.models import LLMRequest, ModelResponse
from uclone_x.room.models import ParticipantKind
from uclone_x.ui.app import create_ui_app
from uclone_x.ui.rooms import RoomStack

_SETTINGS_ENV = ("LLM_PROVIDER", "LLM_MODEL")

#: Two connections of the `mock` kind, each with its own `MockLLMConnector`; the default
#: conversation model is on one and the default fast model on the other.
_SETTINGS = {
    "connections": [{"id": "mock", "kind": "mock"}, {"id": "box", "kind": "mock"}],
    "default_models": {"deep": "mock/a", "fast": "mock/quick"},
}


@pytest.fixture(autouse=True)
def _restore_settings_env(  # pyright: ignore[reportUnusedFunction]
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in _SETTINGS_ENV:
        monkeypatch.delenv(name, raising=False)


class _Held(MockLLMConnector):
    """A connector whose next request is held, so a turn is in flight while a save lands."""

    def __init__(self) -> None:
        super().__init__(default_response="ok")
        self.reached = asyncio.Event()
        self.release = asyncio.Event()

    async def generate(self, request: LLMRequest) -> ModelResponse:
        self.reached.set()
        await self.release.wait()
        return await super().generate(request)


def _app(tmp_path: Path) -> Any:
    workspace = tmp_path / "ws"
    storage = workspace / "sessions"
    storage.mkdir(parents=True)
    (storage / "settings.json").write_text(json.dumps(_SETTINGS), encoding="utf-8")
    return create_ui_app(
        static_dir=workspace / "static", storage_dir=storage, workspace_dir=workspace
    )


def _stack(app: Any) -> RoomStack:
    return cast(RoomStack, app.state.room_stack)


def _room(stack: RoomStack, *, one_seat: bool) -> str:
    room_id = stack.service.create("Index tuning").room_id
    stack.service.add_participant(room_id, "user", kind=ParticipantKind.HUMAN)
    stack.service.add_participant(room_id, "helper")
    if not one_seat:
        stack.service.add_participant(room_id, "scout")
    return room_id


def _seat(stack: RoomStack, room_id: str) -> BaseAgent:
    state = stack.store.load(room_id)
    assert state is not None
    participant = next(p for p in state.participants if p.id == "helper")
    return asyncio.run(stack.resolve_agent(state, participant))


def _connector(app: Any, ref: str) -> Any:
    return app.state.session_manager.gateway.connector_for(ModelRef.parse(ref))


def _served(result: TurnResult) -> tuple[str, str | None]:
    assert result.is_completed, result.error
    assert result.provenance is not None
    return result.provenance.served_by.provider, result.provenance.served_by.model


def _create(client: TestClient, **models: Any) -> dict[str, Any]:
    res = client.post(
        "/api/clones",
        json={"name": "helper", "role": "Helper", "system_prompt": "Help.", **models},
    )
    assert res.status_code in (200, 201), res.text
    return res.json()["persona"]


def _edit(client: TestClient, **changes: Any) -> int:
    entry = {p["name"]: p for p in client.get("/api/clones").json()["clones"]}["helper"]
    body = {
        k: v
        for k, v in entry.items()
        if k in ("name", "role", "description", "system_prompt", "allowed_tools")
    }
    body.update(changes)
    res = client.put("/api/clones/helper", json=body)
    assert res.status_code == 200, res.text
    return int(res.json()["live_agents_updated"])


@pytest.mark.parametrize("one_seat", [True, False], ids=["one_to_one", "room"])
def test_the_next_turn_after_a_save_runs_on_the_new_conversation_model(
    tmp_path: Path, one_seat: bool
) -> None:
    """The owner's report: change the Conversation model, send again in the open conversation.

    The seat is the same live agent afterwards, and its next turn is served by the model
    the clone editor saved, on that model's connection.

    Killed by: src/uclone_x/room/resolver.py :: self._stage_model_change(cache_key, agent, persona.llm_config)
    Becomes: pass
    """
    app = _app(tmp_path)
    stack = _stack(app)
    client = TestClient(app)
    _create(client)
    room_id = _room(stack, one_seat=one_seat)
    agent = _seat(stack, room_id)
    assert _served(asyncio.run(agent.execute_turn("Go"))) == ("mock", "a")

    assert _edit(client, model_name="box/flash") == 1

    assert _seat(stack, room_id) is agent, "the seat was rebuilt rather than rebound"
    assert _served(asyncio.run(agent.execute_turn("Again"))) == ("mock", "flash")
    assert agent.llm is _connector(app, "box/flash")
    assert agent.config.llm_config.model_name == "flash"


def test_a_turn_already_running_finishes_on_its_old_model(tmp_path: Path) -> None:
    """A save that lands mid-turn changes nothing until that turn ends (#1899's rule).

    Killed by: src/uclone_x/agent/base.py :: self._staged_model = change
    Becomes: self._apply_model_change(change)
    """
    app = _app(tmp_path)
    stack = _stack(app)
    client = TestClient(app)
    _create(client)
    room_id = _room(stack, one_seat=True)
    agent = _seat(stack, room_id)
    held = _Held()
    agent.hot_reload_llm(llm=held)

    async def scenario() -> tuple[TurnResult, TurnResult]:
        running = asyncio.create_task(agent.execute_turn("Go"))
        await asyncio.wait_for(held.reached.wait(), 5.0)
        persona = agent.persona_definition
        assert persona is not None
        edited = persona.model_copy(
            update={"llm_config": persona.llm_config.model_copy(update={"model_name": "box/flash"})}
        )
        assert stack.persona_edited(edited) == 1
        assert agent.llm is held, "the connector changed under a running turn"
        assert agent.config.llm_config.model_name == "a", "the model changed mid-turn"
        held.release.set()
        first = await asyncio.wait_for(running, 5.0)
        return first, await agent.execute_turn("Again")

    first, second = asyncio.run(scenario())
    assert _served(first) == ("mock", "a")
    assert _served(second) == ("mock", "flash")


def test_a_save_that_changes_no_model_does_not_rebind_the_seat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A role edit reaches the seat as #1899 has it, and leaves its connector alone.

    Killed by: src/uclone_x/room/resolver.py :: if _model_slots(saved) == _model_slots(own):
    Becomes: if False:
    """
    app = _app(tmp_path)
    stack = _stack(app)
    client = TestClient(app)
    _create(client, model_name="box/flash")
    room_id = _room(stack, one_seat=True)
    agent = _seat(stack, room_id)
    before = agent.llm
    binds: list[AgentLLMConfig] = []
    gateway = app.state.session_manager.gateway
    real_bind = gateway.bind

    def counting_bind(own: AgentLLMConfig) -> Any:
        binds.append(own)
        return real_bind(own)

    monkeypatch.setattr(gateway, "bind", counting_bind)

    assert _edit(client, role="Assistant", model_name="box/flash") == 1
    assert _served(asyncio.run(agent.execute_turn("Go"))) == ("mock", "flash")

    assert binds == [], "a save that changed no model bound the seat again"
    assert agent.llm is before
    assert agent.persona_definition is not None and agent.persona_definition.role == "Assistant"


def test_an_unset_fast_model_follows_a_changed_conversation_model(tmp_path: Path) -> None:
    """Owner ruling 2026-10-29: a clone with no fast model uses its conversation model.

    So when the conversation model changes, the open seat's fast model moves with it.

    Killed by: src/uclone_x/llm/gateway.py :: fast_model = deep_model
    Becomes: fast_model = None
    """
    app = _app(tmp_path)
    stack = _stack(app)
    client = TestClient(app)
    _create(client, model_name="mock/a")
    agent = _seat(stack, _room(stack, one_seat=True))
    assert agent.config.llm_config.fast_model == "a"

    _edit(client, model_name="box/flash")
    assert _served(asyncio.run(agent.execute_turn("Go"))) == ("mock", "flash")

    assert agent.config.llm_config.fast_model == "flash"


def test_a_changed_picture_model_reaches_the_open_seat_at_its_next_turn(tmp_path: Path) -> None:
    """The picture model is read off the seat's config by every drawing tool call.

    Killed by: src/uclone_x/agent/base.py :: update={"image_model": change.image_model}
    Becomes: update={}
    """
    app = _app(tmp_path)
    stack = _stack(app)
    client = TestClient(app)
    _create(client)
    agent = _seat(stack, _room(stack, one_seat=True))
    assert agent.config.llm_config.image_model is None

    persona = agent.persona_definition
    assert persona is not None
    edited = persona.model_copy(
        update={"llm_config": persona.llm_config.model_copy(update={"image_model": "gpu/auto"})}
    )
    assert stack.persona_edited(edited) == 1
    assert agent.config.llm_config.image_model is None, "the change did not wait for a turn"
    assert asyncio.run(agent.execute_turn("Go")).is_completed

    assert agent.config.llm_config.image_model == "gpu/auto"


class TestUnsetFastModel:
    """Owner ruling 2026-10-29: no fast model set means the clone's own conversation model."""

    @staticmethod
    def _gateway(tmp_path: Path, fast: str) -> ModelGateway:
        return ModelGateway(
            tmp_path / "settings.json",
            default_binding=DefaultBinding(connector=MockLLMConnector(), deep="d", fast=fast),
        )

    def test_it_resolves_to_the_conversation_model_on_the_same_connection(
        self, tmp_path: Path
    ) -> None:
        """The system fast default is on the clone's connection and is still not used.

        Killed by: src/uclone_x/llm/gateway.py :: fast_model = deep_model
        Becomes: fast_model = self.resolve(None, "fast").model
        """
        app = _app(tmp_path)
        gateway = app.state.session_manager.gateway
        app.state.session_manager.save_default_models({"deep": "mock/a", "fast": "box/quick"})

        seat = gateway.bind(AgentLLMConfig(model_name="box/flash"))

        assert seat.llm_config.model_name == "flash"
        assert seat.llm_config.fast_model == "flash"

    def test_it_resolves_to_the_conversation_model_on_another_connection(
        self, tmp_path: Path
    ) -> None:
        """The system fast default sits on another connection: fast was dropped to none.

        Killed by: src/uclone_x/llm/gateway.py :: fast_model = deep_model
        Becomes: fast_model = None
        """
        app = _app(tmp_path)
        gateway = app.state.session_manager.gateway

        seat = gateway.bind(AgentLLMConfig(model_name="box/flash"))

        assert seat.llm_config.model_name == "flash"
        assert seat.llm_config.fast_model == "flash"

    def test_a_clone_following_the_default_uses_the_default_conversation_model(
        self, tmp_path: Path
    ) -> None:
        """Its conversation model is the default one, so that is its fast model too."""
        app = _app(tmp_path)
        seat = app.state.session_manager.gateway.bind(AgentLLMConfig())

        assert seat.llm_config.model_name == "a"
        assert seat.llm_config.fast_model == "a"

    def test_its_own_fast_model_still_wins(self, tmp_path: Path) -> None:
        app = _app(tmp_path)
        seat = app.state.session_manager.gateway.bind(
            AgentLLMConfig(model_name="box/flash", fast_model="box/tiny")
        )

        assert seat.llm_config.fast_model == "tiny"

    def test_a_bare_model_id_on_the_given_connector_is_its_own_fast_model(
        self, tmp_path: Path
    ) -> None:
        """A terminal command's `--model` on the head's connector (no connection named)."""
        seat = self._gateway(tmp_path, "f").bind(AgentLLMConfig(model_name="bare"))

        assert seat.llm_config.fast_model == "bare"


#: The owner's Composer (2026-10-29): its seat on `remote-gpu/qwen3:14b`, the clone then
#: set to `gemini/gemini-flash-latest`, and the `remote-gpu` connection deleted (#2422).
_COMPOSER = {
    "connections": [{"id": "remote-gpu", "kind": "mock"}, {"id": "gemini", "kind": "mock"}],
    "default_models": {"deep": "gemini/gemini-flash-latest"},
}


def _composer_app(workspace: Path) -> Any:
    storage = workspace / "sessions"
    storage.mkdir(parents=True, exist_ok=True)
    settings = storage / "settings.json"
    if not settings.exists():
        settings.write_text(json.dumps(_COMPOSER), encoding="utf-8")
    return create_ui_app(
        static_dir=workspace / "static", storage_dir=storage, workspace_dir=workspace
    )


def test_a_seat_whose_connection_was_removed_takes_the_clones_new_model(tmp_path: Path) -> None:
    """Owner ruling 2026-10-29: never force a new conversation; it recovers in place.

    The rebind resolves only the *new* choice, so the old model's connection being gone
    does not stand in its way.

    Killed by: src/uclone_x/room/resolver.py :: self._stage_model_change(cache_key, agent, persona.llm_config)
    Becomes: pass
    """
    app = _composer_app(tmp_path / "ws")
    stack = _stack(app)
    client = TestClient(app)
    _create(client, model_name="remote-gpu/qwen3:14b")
    room_id = _room(stack, one_seat=True)
    agent = _seat(stack, room_id)
    assert _served(asyncio.run(agent.execute_turn("Go"))) == ("mock", "qwen3:14b")

    assert client.delete("/api/connections/remote-gpu").status_code == 200
    _edit(client, model_name="gemini/gemini-flash-latest")

    assert _seat(stack, room_id) is agent, "the conversation was rebuilt rather than recovered"
    assert _served(asyncio.run(agent.execute_turn("Again"))) == ("mock", "gemini-flash-latest")


def _post(stack: RoomStack, room_id: str, text: str) -> Any:
    state = stack.store.load(room_id)
    assert state is not None
    asyncio.run(stack.orchestrator(state).post(room_id, "user", text))
    landed = cast(Any, stack.store.load(room_id)).transcript[-1]
    assert landed.error is None, landed.error
    return landed


def test_a_seat_rebuilt_from_its_record_takes_the_clones_saved_model(tmp_path: Path) -> None:
    """After a restart the seat is built again from its session record: nothing recorded
    there (the anchor's persona stamp, the room record) keeps the model it used before.
    """
    workspace = tmp_path / "ws"
    app = _composer_app(workspace)
    stack = _stack(app)
    client = TestClient(app)
    _create(client, model_name="remote-gpu/qwen3:14b")
    room_id = _room(stack, one_seat=True)
    first = _post(stack, room_id, "Go")
    assert first.provenance.served_by.model == "qwen3:14b"
    _edit(client, model_name="gemini/gemini-flash-latest")
    assert client.delete("/api/connections/remote-gpu").status_code == 200

    restarted = _stack(_composer_app(workspace))
    again = _post(restarted, room_id, "Again")

    assert again.provenance.served_by.model == "gemini-flash-latest"
    seat = _seat(restarted, room_id)
    assert any("Go" in str(m.content) for m in seat.history), "the seat did not resume"
