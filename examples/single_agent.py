"""Run one agent turn with the base install alone: `pip install uclone-x`, no extras.

The agent is composed from the five things a host must provide -- an event bus, a model,
a tool registry, a tracer and a session store. The model here is the deterministic mock,
so this runs offline; for a real one, pass another connector from
`uclone_x.llm.connectors` (Ollama, vLLM, OpenAI, Anthropic, Gemini). Every connector
speaks HTTP through `httpx`, so none needs a provider SDK.

    python examples/single_agent.py
"""

from __future__ import annotations

import asyncio
import tempfile

from uclone_x.agent import (
    AgentConfig,
    AgentContext,
    HostDependencies,
    SessionStore,
    compose_agent,
)
from uclone_x.engine.event_bus import EventBus
from uclone_x.llm.connectors.mock import MockLLMConnector
from uclone_x.telemetry.tracer import TelemetryTracer
from uclone_x.tools.registry import ToolRegistry


async def main() -> None:
    with tempfile.TemporaryDirectory() as sessions:
        host = HostDependencies(
            bus=EventBus(),
            llm=MockLLMConnector(default_response="Hello from a single agent."),
            tools=ToolRegistry(),
            tracer=TelemetryTracer(),
            store=SessionStore(storage_dir=sessions),
        )
        agent = compose_agent(
            AgentConfig(agent_id="demo", name="Demo", system_prompt="Answer briefly."),
            host,
            AgentContext(agent_id="demo", session_id="demo-session"),
        )
        result = await agent.execute_turn("Say hello.")
        print(result.content)


if __name__ == "__main__":
    asyncio.run(main())
