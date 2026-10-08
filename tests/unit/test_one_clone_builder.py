"""One clone, one builder: a 1:1 chat and a room seat build the same agent (#1731).

Owner ruling 2026-09-27: a clone builds its request one way whether it is opened as a
chat or seated in a room; the room adds only its shared context. Both heads now go
through `build_clone`, so the parity is pinned on what the two built agents hold --
config, model, memory, tool binder, offered tools -- and the differences are listed
rather than implied: the seat's framing, its display name, its ontology and its
transport.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any, cast

import pytest

from tests.support.app_clone import app_clone
from tests.support.clones import make_clones
from uclone_x.agent.base import BaseAgent
from uclone_x.llm import MockLLMConnector
from uclone_x.room.models import ParticipantKind

SRC = Path(__file__).resolve().parents[2] / "src" / "uclone_x"

# What a room seat is allowed to differ in from the same clone opened as a chat.
ROOM_ONLY_CONFIG_FIELDS = {"seat_framing", "name"}


def _chat_and_seat(tmp_path: Path, clone_id: str) -> tuple[BaseAgent, BaseAgent]:
    from uclone_x.core.agent_home import AGENTS_DIR_ENV_VAR
    from uclone_x.ui.app import AgentSessionManager
    from uclone_x.ui.rooms import RoomStack

    with pytest.MonkeyPatch.context() as mp:
        mp.setenv(AGENTS_DIR_ENV_VAR, str(tmp_path / "agents"))
        # A persona-less clone of that handle, which is what an unknown name was before
        # clones were stored (clone-data-scopes §3.4); `writer` is installed at start.
        if clone_id != "writer":
            make_clones(clone_id)
        mgr = AgentSessionManager(
            storage_dir=tmp_path / "sessions",
            llm=MockLLMConnector(),
            workspace_dir=tmp_path / "workspace",
        )
        chat = app_clone(mgr, clone_id)
        stack = RoomStack(mgr)
        state = stack.service.create(title="Parity")
        stack.service.add_participant(
            state.room_id, participant_id=clone_id, kind=ParticipantKind.AGENT
        )
        seated = stack.store.load(state.room_id)
        assert seated is not None
        resolver = stack.orchestrator(seated)._resolver  # pyright: ignore[reportPrivateUsage]
        participant = next(p for p in seated.participants if p.id == clone_id)
        seat = cast("BaseAgent", asyncio.run(resolver.resolve(participant)))
    return chat, seat


def _config_without_room_fields(agent: BaseAgent) -> dict[str, Any]:
    dumped = agent.config.model_dump()
    return {k: v for k, v in dumped.items() if k not in ROOM_ONLY_CONFIG_FIELDS}


@pytest.mark.parametrize(
    ("clone_id", "persona"),
    [("writer", "writer"), ("champion", None)],
    ids=["a persona", "no persona"],
)
def test_a_chat_and_a_seat_of_one_clone_are_built_alike(
    tmp_path: Path, clone_id: str, persona: str | None
) -> None:
    """Same config (bar framing and name), model, memory store, binder and tool offer.

    The no-persona case caught the drift this builder removes: the chat named its own
    fallback prompt and capped `max_tokens` at 2048, and the seat did neither.
    """
    chat, seat = _chat_and_seat(tmp_path, clone_id)

    assert _config_without_room_fields(seat) == _config_without_room_fields(chat)
    assert (seat.persona, chat.persona) == (persona, persona)
    assert seat._llm is chat._llm  # pyright: ignore[reportPrivateUsage]
    assert seat._memory is not None  # pyright: ignore[reportPrivateUsage]
    assert seat._memory is chat._memory  # pyright: ignore[reportPrivateUsage]
    # `None` for both under the mock provider, which binds nothing; the manager caches one
    # binder per provider, so under Ollama this is one object too.
    assert seat._tool_invoker.binder is chat._tool_invoker.binder  # pyright: ignore[reportPrivateUsage]
    assert {t.name for t in seat.available_tools()} == {t.name for t in chat.available_tools()}


def test_the_room_adds_its_seat_framing(tmp_path: Path) -> None:
    """The framing is the room's addition; the chat carries none."""
    chat, seat = _chat_and_seat(tmp_path, "writer")

    assert chat.config.seat_framing == ""
    assert seat.config.seat_framing != ""


def test_no_head_builds_its_own_host() -> None:
    """Every head reaches `build_clone`; a `HostDependencies(` in ui/ or cli/ is a second
    builder, which is how the chat and the room drifted apart before #1731."""
    offenders = [
        f"{path.relative_to(SRC)}:{n}"
        for head in ("ui", "cli")
        for path in sorted((SRC / head).rglob("*.py"))
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if re.search(r"\bHostDependencies\(", line)
    ]
    assert offenders == []


@pytest.mark.parametrize(
    ("clone_id", "asked", "expected"),
    [
        ("scribe", None, "persona-model"),
        ("scribe", "asked-model", "asked-model"),
        ("nobody", None, "saved-model"),
    ],
    ids=["persona keeps its own", "--model wins", "no persona follows the saved choice"],
)
def test_a_cli_clone_picks_its_model_as_the_app_does(
    tmp_path: Path, clone_id: str, asked: str | None, expected: str
) -> None:
    """The saved model fills only what a persona leaves empty, as Settings does in the app.

    `ucx run` used to pass its saved model as `model_name`, which wins over the persona's
    own: the same clone ran on a different model in the terminal than in the desktop app.

    `run` and `loop` now pass `--model` alone as `model_name` and their connector and the
    saved choice as the gateway's default binding (`command_gateway`); a persona's own
    model ref is served from its connection (model-gateway §3.4). This pins what the
    builder does with the three.

    Killed by: src/uclone_x/agent/clone_builder.py :: return ModelGateway(default_binding=DefaultBinding(llm, model))
    Becomes: return ModelGateway()
    """
    from uclone_x.agent.clone_builder import (
        build_clone,
        command_gateway,
        local_app_scope,
        memory_map,
    )
    from uclone_x.agent.models import AgentLLMConfig, PersonaDefinition
    from uclone_x.agent.persona_registry import PersonaRegistry
    from uclone_x.agent.session import SessionStore
    from uclone_x.engine.event_bus import EventBus
    from uclone_x.llm.connectors.saved_choice import settings_file
    from uclone_x.memory.store import CrossSessionMemory
    from uclone_x.telemetry import TelemetryTracer
    from uclone_x.tools.registry import ToolRegistry

    registry = PersonaRegistry(include_defaults=False)
    registry.register_persona(
        PersonaDefinition(
            name="scribe",
            role="Scribe",
            system_prompt="You write.",
            llm_config=AgentLLMConfig(model_name="box/persona-model"),
        )
    )
    settings = settings_file()
    settings.parent.mkdir(parents=True, exist_ok=True)
    settings.write_text('{"connections": [{"id": "box", "kind": "mock"}]}', encoding="utf-8")
    command_llm = MockLLMConnector()
    app = local_app_scope(
        workspace_root=tmp_path,
        llm=command_llm,
        tools=ToolRegistry(),
        persona_registry=registry,
        memory_for=memory_map(lambda cid: CrossSessionMemory(storage_path=tmp_path / cid)),
        gateway=command_gateway(command_llm, "saved-model"),
        bus=EventBus(),
        tracer=TelemetryTracer(),
        store=SessionStore(tmp_path / "sessions"),
    )

    agent = build_clone(app, clone_id=clone_id, session_id="s", model_name=asked).agent

    assert agent.config.llm_config.model_name == expected
    # The persona's own ref runs on its own connection; everything else on the command's.
    assert (agent.llm is command_llm) is (expected != "persona-model")


@pytest.mark.parametrize(
    ("sent", "saved", "said"),
    [
        ("persona-model", "saved-model", True),
        ("saved-model", "saved-model", False),
        ("asked-model", None, False),
    ],
    ids=["persona's own wins", "saved is sent", "no saved choice"],
)
def test_the_cli_says_when_a_persona_runs_on_its_own_model(
    sent: str, saved: str | None, said: bool
) -> None:
    """The saved-choice notice names the saved model; a persona's own is said too (P6).

    Killed by: src/uclone_x/cli/commands/run.py :: if saved is None or sent is None or sent == saved:
    Becomes: if True:
    """
    from uclone_x.cli.commands.run import own_model_notice

    notice = own_model_notice("writer", sent, saved)

    assert (notice is not None) is said
    if notice is not None:
        # Both named: `saved` may be an `OLLAMA_MODEL` value rather than the saved choice.
        assert sent in notice
        assert saved is not None and saved in notice
