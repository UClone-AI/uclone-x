"""ACP CLI commands: start ACP server over stdio for editor-to-agent integration (Issue #649)."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.markup import escape

from uclone_x.agent.base import BaseAgent
from uclone_x.agent.clone_builder import AppScope, build_clone, local_app_scope, memory_map
from uclone_x.agent.models import TurnResult
from uclone_x.agent.persona_registry import get_default_persona_registry
from uclone_x.agent.persona_store import DEFAULT_PERSONA_NAME
from uclone_x.agent.session import SessionStore
from uclone_x.cli.agent_memory import memory_for_agent_id
from uclone_x.engine.event_bus import EventBus
from uclone_x.llm.connectors.factory import create_llm_connector, saved_choice_notice
from uclone_x.room.one_seat import (
    HeadTurn,
    conversation_room_id,
    head_room_person_names,
    record_head_turn,
    resolve_one_seat_room,
)
from uclone_x.room.service import RoomService, participant_session_id
from uclone_x.room.store import RoomStore
from uclone_x.shells.acp import ACPServer
from uclone_x.shells.acp.server import (
    DEFAULT_MAX_LIVE_AGENTS,
    ACPTurnFailure,
    PersonNames,
    SeatSessionName,
    SessionAgentFactory,
    TurnRecorder,
)
from uclone_x.telemetry import TelemetryTracer
from uclone_x.tools.registry import create_default_registry

acp_app = typer.Typer(
    name="acp",
    help="Manage Agent Client Protocol (ACP) server and editor integration",
    no_args_is_help=True,
)
console = Console()
# ACP speaks over stdout, so anything said to the person goes to stderr.
err_console = Console(stderr=True)


def acp_seat_session(clone_id: str) -> SeatSessionName:
    """The seat session `clone_id` keeps for an ACP session id: its one-seat room's (§5.9).

    Keyed by clone and ACP session id (owner ruling 2026-09-27), so two clones served
    under one ACP session id never share a room. Pure: naming a session creates no room,
    so `load_session` on an id never started finds nothing, as before.
    """

    def seat(session_id: str) -> str:
        return participant_session_id(conversation_room_id("acp", clone_id, session_id), clone_id)

    return seat


def session_agent_factory(
    app: AppScope, *, clone_id: str, room_store: RoomStore | None = None, **clone: Any
) -> SessionAgentFactory:
    """Build each ACP session's agent: one clone, built per session (#1454, #1731).

    Each ACP session is a one-seat room (§5.9): the room `acp_seat_session` names is
    checked -- a stored one must seat the clone alone -- and the agent is built on the
    clone's seat session in it. Nothing is written here: the room is stored with the
    session's first turn (`acp_turn_recorder`), so a `session/new` whose save fails, or a
    session never prompted, leaves no room behind (#1846). The room is kept in
    `room_store` (the Core default).

    Every session gets an agent of its own, so no session's history reaches another's
    request, while everything that is the *agent's* rather than the conversation's -- the
    store, the bus, the connector, the tools and the cross-session memory -- is shared
    through `app`. Memory in particular is one store per clone id, from the scope's map:
    two stores over one file would each drop the facts the other recorded. `clone` is
    what `build_clone` takes besides the session.
    """

    def build(session_id: str) -> BaseAgent:
        room = resolve_one_seat_room(
            RoomService(room_store if room_store is not None else RoomStore()),
            room_id=conversation_room_id("acp", clone_id, session_id),
            clone_id=clone_id,
            head="acp",
        )
        return build_clone(app, clone_id=clone_id, session_id=room.session_id, **clone).agent

    return build


def acp_turn_recorder(clone_id: str, room_store: RoomStore | None = None) -> TurnRecorder:
    """Record each ACP turn in its session's one-seat room transcript (#1837).

    The room is created with the session's first turn. A turn the room could not take
    raises here, which the server logs; the client's answer is unaffected.
    """
    store = room_store if room_store is not None else RoomStore()

    def record(
        session_id: str,
        prompt: str,
        outcome: TurnResult | ACPTurnFailure,
        session_set_aside: bool = False,
    ) -> None:
        turn = (
            HeadTurn.failed(prompt, outcome.cause, completed=outcome.completed)
            if isinstance(outcome, ACPTurnFailure)
            else HeadTurn.from_result(prompt, outcome)
        )
        turn = replace(turn, session_set_aside=session_set_aside)
        record_head_turn(
            store,
            room_id=conversation_room_id("acp", clone_id, session_id),
            clone_id=clone_id,
            turn=turn,
            head="acp",
        )

    return record


def acp_person_names(clone_id: str, room_store: RoomStore | None = None) -> PersonNames:
    """The names the person goes by in an ACP session's one-seat room (#1893 item 1)."""
    rooms = RoomService(room_store if room_store is not None else RoomStore())

    def names(session_id: str) -> tuple[str, ...]:
        return head_room_person_names(rooms, conversation_room_id("acp", clone_id, session_id))

    return names


def one_seat_acp_server(
    app: AppScope,
    *,
    clone_id: str,
    bus: EventBus | None = None,
    room_store: RoomStore | None = None,
    max_live_agents: int = DEFAULT_MAX_LIVE_AGENTS,
    **clone: Any,
) -> ACPServer:
    """An ACP server answering as `clone_id`, each ACP session a one-seat room (§5.9).

    The agent factory and the server's session names are made together here, because
    one without the other would build every agent on a session the server never reads.
    `bus` is the one in `app`, which the server subscribes to for a turn's events.
    """
    return ACPServer(
        agent_factory=session_agent_factory(app, clone_id=clone_id, room_store=room_store, **clone),
        seat_session=acp_seat_session(clone_id),
        turn_recorder=acp_turn_recorder(clone_id, room_store),
        person_names=acp_person_names(clone_id, room_store),
        bus=bus,
        store=app.host.store,
        max_live_agents=max_live_agents,
    )


#: What `--agent-id` says it does. Shared by `ucx acp serve` and `ucx acp-server`.
AGENT_ID_HELP: str = (
    "Agent identifier the clone's memory is kept under. Defaults to the persona's name, "
    "which is the id the desktop app uses for the same clone"
)


def start_acp_server(agent_id: str | None = None, persona: str = DEFAULT_PERSONA_NAME) -> None:
    """Start inbound ACP server over stdio, answering as `persona`.

    An unset `agent_id` means the persona's name (#1494). The desktop app keys each agent
    by its persona's name, and the agent id names the memory store, so a clone reached
    from an editor remembers what the same clone learned in the app and the other way
    round. The old default, `default`, kept the editor's memory apart from the app's.
    """
    # Deferred for the reason `run` defers its imports: `ucx --help` loads this module.
    from uclone_x.skills.auditor import load_runtime_skill_registry

    tools = create_default_registry()
    # The persona's config is built where the chat head and room seats build theirs
    # (#1452), so the clone an editor talks to is given its tools by the same rule. This
    # used to be a hand-written `AgentConfig` no other head produced.
    registry = get_default_persona_registry(tool_names=[tool.name for tool in tools.list_tools()])
    persona_def = registry.get_persona(persona)
    if persona_def is None:
        console.print(f"[bold red]Unknown --persona:[/bold red] {escape(persona)}")
        raise typer.Exit(code=2)
    # Named after the persona when no id is given, the id the desktop app uses.
    clone_id = agent_id or persona_def.name
    bus = EventBus()
    saved_notice = saved_choice_notice()
    if saved_notice is not None:
        err_console.print(saved_notice, markup=False, highlight=False)
    llm = create_llm_connector(fallback_to_mock=True)
    store = SessionStore()

    # One store per clone id, opened now so a bad `--agent-id` exits 2 before the server
    # starts rather than when the first session is built.
    memory_for = memory_map(memory_for_agent_id)
    memory_for(clone_id)

    # Built where the chat head and room seats build theirs (#1452, #1731), so the clone
    # an editor talks to is the one the desktop app answers as.
    app = local_app_scope(
        workspace_root=Path.cwd().resolve(),
        llm=llm,
        tools=tools,
        persona_registry=registry,
        memory_for=memory_for,
        bus=bus,
        tracer=TelemetryTracer(),
        store=store,
        # P9: the approved skills in the runtime store; without them there is no `load_skill`.
        skills=asyncio.run(load_runtime_skill_registry()),
    )

    server = one_seat_acp_server(app, clone_id=clone_id, bus=bus, persona=persona_def.name)

    asyncio.run(server.run_stdio())


@acp_app.command("serve")
def acp_serve(
    agent_id: Annotated[str | None, typer.Option("--agent-id", help=AGENT_ID_HELP)] = None,
    persona: Annotated[
        str, typer.Option("--persona", help="Clone to answer as")
    ] = DEFAULT_PERSONA_NAME,
) -> None:
    """Start standalone ACP stdio server."""
    start_acp_server(agent_id=agent_id, persona=persona)
