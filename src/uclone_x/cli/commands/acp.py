"""ACP CLI commands: start ACP server over stdio for editor-to-agent integration (Issue #649)."""

from __future__ import annotations

import asyncio
from typing import Annotated

import typer
from rich.console import Console
from rich.markup import escape

from uclone_x.agent.base import BaseAgent
from uclone_x.agent.bootstrap import agent_config_for_persona
from uclone_x.agent.composition import HostDependencies, compose_agent
from uclone_x.agent.models import AgentConfig, AgentContext
from uclone_x.agent.persona_registry import get_default_persona_registry
from uclone_x.agent.persona_store import DEFAULT_PERSONA_NAME
from uclone_x.agent.session import SessionStore
from uclone_x.cli.agent_memory import memory_for_agent_id
from uclone_x.engine.event_bus import EventBus
from uclone_x.llm.connectors.factory import create_llm_connector
from uclone_x.shells.acp import ACPServer
from uclone_x.shells.acp.server import SessionAgentFactory
from uclone_x.telemetry import TelemetryTracer
from uclone_x.tools.registry import create_default_registry

acp_app = typer.Typer(
    name="acp",
    help="Manage Agent Client Protocol (ACP) server and editor integration",
    no_args_is_help=True,
)
console = Console()


def session_agent_factory(config: AgentConfig, host: HostDependencies) -> SessionAgentFactory:
    """Build each ACP session's agent: one `config`, one `host`, a context per session (#1454).

    Every session gets an agent of its own, so no session's history reaches another's
    request, while everything that is the *agent's* rather than the conversation's -- the
    store, the bus, the connector, the tools and the cross-session memory -- is shared
    through `host`. Memory in particular must be one store per agent id: two stores over
    one file would each drop the facts the other recorded (see `HostDependencies.memory`).
    """

    def build(session_id: str) -> BaseAgent:
        return compose_agent(
            config=config,
            host=host,
            context=AgentContext(session_id=session_id, agent_id=config.agent_id),
        )

    return build


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
    tools = create_default_registry()
    # The persona's config is built where the chat head and room seats build theirs
    # (#1452), so the clone an editor talks to is given its tools by the same rule. This
    # used to be a hand-written `AgentConfig` no other head produced.
    registry = get_default_persona_registry(tool_names=[tool.name for tool in tools.list_tools()])
    persona_def = registry.get_persona(persona)
    if persona_def is None:
        console.print(f"[bold red]Unknown --persona:[/bold red] {escape(persona)}")
        raise typer.Exit(code=2)
    # `agent_config_for_persona` names the agent after the persona when no id is given.
    agent_config = agent_config_for_persona(persona_def, agent_id=agent_id)

    bus = EventBus()
    llm = create_llm_connector(fallback_to_mock=True)
    store = SessionStore()
    tracer = TelemetryTracer()

    memory = memory_for_agent_id(agent_config.agent_id)

    host_deps = HostDependencies(
        bus=bus,
        llm=llm,
        tools=tools,
        tracer=tracer,
        store=store,
        memory=memory,
    )

    server = ACPServer(
        agent_factory=session_agent_factory(agent_config, host_deps),
        bus=bus,
        store=store,
    )

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
