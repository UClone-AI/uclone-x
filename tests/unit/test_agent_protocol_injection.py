from typing import Any, cast
from unittest.mock import MagicMock

import pytest

from uclone_x.agent.base import BaseAgent
from uclone_x.agent.models import AgentConfig, AgentContext
from uclone_x.core.session_store import SessionStoreProtocol
from uclone_x.engine.protocols import EventBusProtocol
from uclone_x.llm.protocols import LLMProviderProtocol
from uclone_x.telemetry.protocols import TracerProtocol


class _MockTracer:
    @property
    def trace_id(self) -> str:
        return "mock"

    def start_span(self, *args: Any, **kwargs: Any) -> str:
        return "mock"

    def end_span(self, *args: Any, **kwargs: Any) -> Any:
        pass

    async def span(self, *args: Any, **kwargs: Any) -> Any:
        class _Ctx:
            async def __aenter__(self):
                return "mock"

            async def __aexit__(self, *a: Any):
                pass

        return _Ctx()


@pytest.mark.asyncio
async def test_agent_protocol_injection_c1_c2_c6() -> None:
    """Assert BaseAgent can be constructed with custom substitute implementations of bus, tracer, and session store.
    Killed by: src/uclone_x/agent/base.py :: tracer: TracerProtocol | None
    Killed by: src/uclone_x/agent/base.py :: store: SessionStoreProtocol | None
    Killed by: src/uclone_x/agent/base.py :: bus: EventBusProtocol | None
    """
    config = AgentConfig(name="test_agent", role="test", agent_id="test_id")
    context = AgentContext(session_id="test_session", agent_id="test_id")

    mock_bus = cast(EventBusProtocol, MagicMock())
    mock_tracer = cast(TracerProtocol, _MockTracer())
    mock_store = cast(SessionStoreProtocol, MagicMock())
    mock_llm = cast(LLMProviderProtocol, MagicMock())

    agent = BaseAgent(
        config=config,
        context=context,
        bus=mock_bus,
        tracer=mock_tracer,
        store=mock_store,
        llm=mock_llm,
    )

    assert agent.tracer is mock_tracer
    assert agent.store is mock_store

    # execute turn mock
    agent.execute_turn = MagicMock()  # type: ignore
