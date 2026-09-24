"""Unit tests for agent composition root.

The `Killed by:` line that named `MissingCapabilityError` moved into the test it pins
Killed by: src/uclone_x/agent/composition.py :: missing.append("bus")
Killed by: src/uclone_x/agent/composition.py :: missing.append("llm")
Killed by: src/uclone_x/agent/composition.py :: missing.append("tools")
Killed by: src/uclone_x/agent/composition.py :: missing.append("tracer")
Killed by: src/uclone_x/agent/composition.py :: missing.append("store")
Killed by: src/uclone_x/agent/composition.py :: raise MissingCapabilityError
Killed by: src/uclone_x/agent/composition.py :: BaseAgent(
"""

import pytest

from uclone_x.agent.base import BaseAgent
from uclone_x.agent.composition import HostDependencies, MissingCapabilityError, compose_agent
from uclone_x.agent.models import AgentConfig, AgentContext
from uclone_x.agent.session import SessionStore
from uclone_x.engine.event_bus import EventBus
from uclone_x.llm.connectors.mock import MockLLMConnector
from uclone_x.ontology.engine import OntologyEngine
from uclone_x.skills.auditor import SkillRegistry
from uclone_x.telemetry.tracer import TelemetryTracer
from uclone_x.tools.registry import ToolRegistry


@pytest.fixture
def base_config() -> AgentConfig:
    return AgentConfig(
        agent_id="test_agent",
        name="Test Agent",
        system_prompt="Test prompt",
    )


def test_valid_composition_across_shell_configurations(base_config: AgentConfig):
    """Test composition with all required dependencies."""
    host = HostDependencies(
        bus=EventBus(),
        llm=MockLLMConnector(),
        tools=ToolRegistry(),
        tracer=TelemetryTracer(),
        store=SessionStore(),
    )
    context = AgentContext(agent_id="test_agent", session_id="test_sess")

    agent = compose_agent(config=base_config, host=host, context=context)

    assert isinstance(agent, BaseAgent)
    assert agent.config == base_config
    assert agent.context.session_id == context.session_id
    assert agent.context.agent_id == context.agent_id
    assert host.tracer is not None
    assert agent.context.trace_id == host.tracer.trace_id
    assert agent._bus is host.bus  # pyright: ignore[reportPrivateUsage]
    assert agent.llm is host.llm
    assert agent._tools is host.tools  # pyright: ignore[reportPrivateUsage]
    assert agent.tracer is host.tracer
    assert agent.store is host.store
    # Optional capabilities defaults
    assert agent.ontology is None


def test_missing_capabilities_raises_named_error(base_config: AgentConfig):
    """Missing required capabilities raise `MissingCapabilityError` naming every one of them.

    The bare name `MissingCapabilityError` matched both the class statement and the raise,
    so it named no one edit (#1080); and the class statement is not usable as an anchor at
    all. Both edits to it measured through `run_isolated_mutation` apply cleanly and neither
    kills: mutating the **name** exits 4 — `SESSION_ERROR`, `scored=False`, `selected=0` —
    because this module imports it, so that run is voided; changing the base to `ValueError`
    scores and ESCAPES. The message is what the five assertions below pin.

    Killed by: src/uclone_x/agent/composition.py :: f"Missing required capabilities for agent composition: {missing_str}"
    Becomes: f"Missing required capabilities for agent composition: {len(missing_str)}"
    """
    host = HostDependencies(
        bus=None,
        llm=None,
        tools=None,
        tracer=None,
        store=None,
    )

    with pytest.raises(MissingCapabilityError) as exc_info:
        compose_agent(config=base_config, host=host)

    err_msg = str(exc_info.value)
    assert "bus" in err_msg
    assert "llm" in err_msg
    assert "tools" in err_msg
    assert "tracer" in err_msg
    assert "store" in err_msg


def test_explicit_handling_of_optional_capabilities(base_config: AgentConfig):
    """Test that optional capabilities are passed correctly to BaseAgent."""
    skill_registry = SkillRegistry()
    ontology_engine = OntologyEngine()

    host = HostDependencies(
        bus=EventBus(),
        llm=MockLLMConnector(),
        tools=ToolRegistry(),
        tracer=TelemetryTracer(),
        store=SessionStore(),
        skills=skill_registry,
        ontology=ontology_engine,
    )

    agent = compose_agent(config=base_config, host=host)

    assert agent.skills is skill_registry
    assert agent.ontology is ontology_engine


def test_equivalence_assertions_between_shells(base_config: AgentConfig):
    """Test that equivalent inputs produce equivalently configured agents across multiple shell paradigms."""
    # Simulating UI shell config
    ui_host = HostDependencies(
        bus=EventBus(),
        llm=MockLLMConnector(),
        tools=ToolRegistry(),
        tracer=TelemetryTracer(),
        store=SessionStore(),
        skills=SkillRegistry(),
    )
    ui_agent = compose_agent(config=base_config, host=ui_host)

    # Simulating CLI shell config
    cli_host = HostDependencies(
        bus=EventBus(),
        llm=MockLLMConnector(),
        tools=ToolRegistry(),
        tracer=TelemetryTracer(),
        store=SessionStore(),
    )
    cli_agent = compose_agent(config=base_config, host=cli_host)

    assert ui_agent.config == cli_agent.config
    # Test they received the correct injected dependencies
    assert ui_agent.skills is not None
    assert cli_agent.skills is None
