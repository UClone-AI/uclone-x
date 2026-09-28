"""A2A CLI commands: serve standalone A2A gateway and manage federated nodes (Issue #69)."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import typer
from rich.console import Console

from uclone_x.a2a.models import AgentCard
from uclone_x.engine.event_bus import EventBus

if TYPE_CHECKING:
    from uclone_x.agent.base import BaseAgent
    from uclone_x.agent.clone_builder import AppScope
    from uclone_x.agent.models import TurnResult
    from uclone_x.room.store import RoomStore
    from uclone_x.shells.a2a_server import (
        A2APersonNames,
        A2ATurnFailure,
        A2ATurnRecorder,
        ContextAgentFactory,
    )

a2a_app = typer.Typer(
    name="a2a",
    help="Manage Agent-to-Agent (A2A) federated gateways and nodes",
    no_args_is_help=True,
)
console = Console()


def a2a_room_id(clone_id: str, context_id: str) -> str:
    """The one-seat room `clone_id` keeps for A2A conversation `context_id` (#1836).

    Keyed by both (owner ruling 2026-09-27), so two clones called under one `contextId`
    never share a room.
    """
    from uclone_x.room.one_seat import conversation_room_id

    return conversation_room_id("a2a", clone_id, context_id)


def context_agent_factory(
    app: AppScope, *, clone_id: str, room_store: RoomStore | None = None, **clone: Any
) -> ContextAgentFactory:
    """Build each A2A conversation's agent on the clone's seat session in its room (§5.9).

    The room is checked, not written: it is stored with the conversation's first turn
    (`a2a_turn_recorder`). The agent's saved history is restored, so a known `contextId`
    continues where it left off. `clone` is what `build_clone` takes besides the session.
    """
    from uclone_x.agent.clone_builder import build_clone
    from uclone_x.room.one_seat import resolve_one_seat_room
    from uclone_x.room.service import RoomService
    from uclone_x.room.store import RoomStore

    rooms = RoomService(room_store if room_store is not None else RoomStore())

    def build(context_id: str) -> BaseAgent:
        room = resolve_one_seat_room(
            rooms, room_id=a2a_room_id(clone_id, context_id), clone_id=clone_id, head="a2a"
        )
        agent = build_clone(app, clone_id=clone_id, session_id=room.session_id, **clone).agent
        agent.hydrate_session()
        return agent

    return build


def a2a_turn_recorder(clone_id: str, room_store: RoomStore | None = None) -> A2ATurnRecorder:
    """Record each A2A turn in its conversation's one-seat room transcript (#1837)."""
    from uclone_x.room.one_seat import HeadTurn, record_head_turn
    from uclone_x.room.store import RoomStore
    from uclone_x.shells.a2a_server import A2ATurnFailure

    store = room_store if room_store is not None else RoomStore()

    def record(context_id: str, prompt: str, outcome: TurnResult | A2ATurnFailure) -> None:
        turn = (
            HeadTurn.failed(prompt, outcome.cause, completed=outcome.completed)
            if isinstance(outcome, A2ATurnFailure)
            else HeadTurn.from_result(prompt, outcome)
        )
        record_head_turn(
            store,
            room_id=a2a_room_id(clone_id, context_id),
            clone_id=clone_id,
            turn=turn,
            head="a2a",
        )

    return record


def a2a_person_names(clone_id: str, room_store: RoomStore | None = None) -> A2APersonNames:
    """The names the person goes by in an A2A conversation's one-seat room (#1893 item 1)."""
    from uclone_x.room.one_seat import head_room_person_names
    from uclone_x.room.service import RoomService
    from uclone_x.room.store import RoomStore

    rooms = RoomService(room_store if room_store is not None else RoomStore())

    def names(context_id: str) -> tuple[str, ...]:
        return head_room_person_names(rooms, a2a_room_id(clone_id, context_id))

    return names


def start_a2a_server(
    host: str = "127.0.0.1",
    port: int = 8080,
    agent_id: str = "default",
    agent_card_path: Path | None = None,
    dev: bool = False,
) -> None:
    """Start standalone A2A HTTP/SSE gateway and federated node."""
    if agent_card_path is not None:
        if not agent_card_path.is_file():
            console.print(
                f"[bold red]Error:[/bold red] Agent card file not found: {agent_card_path}"
            )
            raise typer.Exit(code=1)
        try:
            card_content = agent_card_path.read_text(encoding="utf-8")
            card = AgentCard.model_validate_json(card_content)
        except Exception as exc:
            console.print(f"[bold red]Error parsing agent card:[/bold red] {exc}")
            raise typer.Exit(code=1) from exc
    else:
        card = AgentCard(
            name=f"agent-{agent_id}",
            description=f"UClone-X A2A Node for agent '{agent_id}'",
            version="1.0.1",
            skills=("general_reasoning", "task_execution"),
            endpoints={"http": f"http://{host}:{port}"},
        )

    bus = EventBus()
    # Deferred, with the rest of this function's imports: `uvicorn` and the A2A
    # shell belong to the `http` extra, and `cli/main.py` imports this module at
    # module scope. Imported at the top, they made every `ucx` invocation --
    # `--help` included -- require a dependency only this command uses, so a
    # `uclone-x[cli]` install died on a raw ModuleNotFoundError before typer ever
    # ran.
    #
    # Guarded rather than bare, matching `shells/a2a_server.py`: the error a user
    # gets has to name the extra they are missing, and an import sorter is free
    # to move a bare `import uvicorn` above the shell import whose own guard
    # would otherwise have said it.
    try:
        import uvicorn
    except ImportError as exc:
        from uclone_x.errors import MissingDependencyError

        raise MissingDependencyError(
            extra="http",
            package="uvicorn",
            feature="A2A gateway server (ucx a2a serve)",
        ) from exc

    from uclone_x.agent.clone_builder import local_app_scope, memory_map
    from uclone_x.agent.session import SessionStore
    from uclone_x.cli.agent_memory import memory_for_agent_id
    from uclone_x.llm.connectors.factory import create_llm_connector, saved_choice_notice
    from uclone_x.shells.a2a_server import A2AServer
    from uclone_x.skills.auditor import load_runtime_skill_registry
    from uclone_x.telemetry import TelemetryTracer
    from uclone_x.tools.registry import create_default_registry

    saved_notice = saved_choice_notice()
    if saved_notice is not None:
        # stderr, as ACP does: what the server says to its caller stays on its own channel.
        Console(stderr=True).print(saved_notice, markup=False, highlight=False)
    llm = create_llm_connector(fallback_to_mock=True)
    # One store per clone id, opened now so a bad `--agent-id` exits 2.
    memory_for = memory_map(memory_for_agent_id)
    memory_for(agent_id)

    # Built as the desktop app builds the same clone (#1731): an id that names a persona
    # answers as it. Any other id is a generic federated node.
    app = local_app_scope(
        workspace_root=Path.cwd().resolve(),
        llm=llm,
        tools=create_default_registry(),
        memory_for=memory_for,
        bus=bus,
        tracer=TelemetryTracer(),
        store=SessionStore(),
        # P9: the approved skills in the runtime store; without them there is no `load_skill`.
        skills=asyncio.run(load_runtime_skill_registry()),
    )
    node = (
        None
        if app.persona_registry.get_persona(agent_id) is not None
        else {
            "name": f"Agent-{agent_id}",
            "role": "A2A Federated Agent",
            "description": f"Autonomous A2A agent node ({agent_id})",
        }
    )
    # Every A2A conversation (`contextId`) is a one-seat room of its own (§5.9, owner
    # ruling 2026-09-27); the session this command kept before, `sess_<clone>`, is left
    # on disk and not resumed.
    server = A2AServer(
        agent_card=card,
        bus=bus,
        host=host,
        port=port,
        context_agent_factory=context_agent_factory(
            app,
            clone_id=agent_id,
            fallback_prompt="You are a federated UClone-X A2A agent node.",
            config_update=node,
        ),
        turn_recorder=a2a_turn_recorder(agent_id),
        person_names=a2a_person_names(agent_id),
    )

    if dev:
        console.print(
            f"[bold yellow]🔥 Starting UClone-X A2A Gateway (Dev Mode) on[/bold yellow] "
            f"[cyan]http://{host}:{port}[/cyan]"
        )
        src_dir = str(Path(__file__).resolve().parents[2])
        uvicorn.run(
            server.app,
            host=host,
            port=port,
            log_level="debug",
            reload=True,
            reload_dirs=[src_dir],
        )
    else:
        console.print(
            f"[bold green]🚀 Launching UClone-X A2A Gateway on[/bold green] "
            f"[cyan]http://{host}:{port}[/cyan]"
        )
        uvicorn.run(
            server.app,
            host=host,
            port=port,
            log_level="info",
        )


@a2a_app.command("serve")
def a2a_serve(
    host: Annotated[str, typer.Option("--host", "-h", help="Host address to bind")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port", "-p", help="Port to bind")] = 8080,
    agent_id: Annotated[str, typer.Option("--agent-id", help="Local agent identifier")] = "default",
    agent_card: Annotated[
        Path | None,
        typer.Option("--agent-card", help="Path to JSON AgentCard configuration file"),
    ] = None,
    dev: Annotated[
        bool,
        typer.Option("--dev", "-d", help="Enable developer hot-reload / debug mode"),
    ] = False,
) -> None:
    """Start standalone A2A HTTP/SSE gateway and federated node."""
    start_a2a_server(
        host=host,
        port=port,
        agent_id=agent_id,
        agent_card_path=agent_card,
        dev=dev,
    )
