"""A2A CLI commands: serve standalone A2A gateway and manage federated nodes (Issue #69)."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from uclone_x.a2a.models import AgentCard
from uclone_x.agent.models import AgentConfig
from uclone_x.engine.event_bus import EventBus

a2a_app = typer.Typer(
    name="a2a",
    help="Manage Agent-to-Agent (A2A) federated gateways and nodes",
    no_args_is_help=True,
)
console = Console()


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

    agent_config = AgentConfig(
        agent_id=agent_id,
        name=f"Agent-{agent_id}",
        role="A2A Federated Agent",
        description=f"Autonomous A2A agent node ({agent_id})",
        system_prompt="You are a federated UClone-X A2A agent node.",
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

    from uclone_x.agent.composition import HostDependencies, compose_agent
    from uclone_x.agent.session import SessionStore
    from uclone_x.cli.agent_memory import memory_for_agent_id
    from uclone_x.llm.connectors.factory import create_llm_connector
    from uclone_x.shells.a2a_server import A2AServer
    from uclone_x.telemetry import TelemetryTracer
    from uclone_x.tools.registry import create_default_registry

    llm = create_llm_connector(fallback_to_mock=True)
    memory = memory_for_agent_id(agent_config.agent_id)

    host_deps = HostDependencies(
        bus=bus,
        llm=llm,
        tools=create_default_registry(),
        tracer=TelemetryTracer(),
        store=SessionStore(),
        memory=memory,
    )
    agent = compose_agent(config=agent_config, host=host_deps)

    server = A2AServer(
        agent_card=card,
        agent=agent,
        bus=bus,
        host=host,
        port=port,
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
