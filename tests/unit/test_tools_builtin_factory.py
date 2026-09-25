"""Unit tests for built-in tool suite factory, default registry auto-wiring, UI, and CLI integration."""

from __future__ import annotations

import json
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from unittest.mock import patch

import pytest

from uclone_x.cli.commands.run import run_agent_repl_async
from uclone_x.core.provenance import Provenance
from uclone_x.engine.event_bus import EventBus
from uclone_x.llm.models import (
    FinishReason,
    LLMRequest,
    ModelResponse,
    StreamChunk,
    TokenUsage,
)
from uclone_x.tools import (
    ToolRegistry,
    create_default_registry,
    create_default_tool_registry,
)
from uclone_x.ui.app import AgentSessionManager, create_ui_app

MOCK_MCP_FACTORY_SERVER_SCRIPT = """
import sys, json

while True:
    line = sys.stdin.readline()
    if not line:
        break
    try:
        req = json.loads(line.strip())
    except Exception:
        continue

    method = req.get("method")
    req_id = req.get("id")

    if method == "initialize":
        resp = {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "factory_test_server", "version": "1.0.0"}
            }
        }
        sys.stdout.write(json.dumps(resp) + "\\n")
        sys.stdout.flush()
    elif method == "notifications/initialized":
        pass
    elif method == "tools/list":
        resp = {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "tools": [
                    {
                        "name": "custom_mcp_action",
                        "description": "Custom action via MCP",
                        "inputSchema": {"type": "object", "properties": {}}
                    }
                ]
            }
        }
        sys.stdout.write(json.dumps(resp) + "\\n")
        sys.stdout.flush()
    elif method == "tools/call":
        resp = {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {"content": [{"type": "text", "text": "executed"}], "isError": False}
        }
        sys.stdout.write(json.dumps(resp) + "\\n")
        sys.stdout.flush()
"""


class DummyLLMConnector:
    """Mock LLM connector conforming to LLMProviderProtocol for testing."""

    @property
    def provider_name(self) -> str:
        return "mock_factory_llm"

    async def generate(self, request: LLMRequest) -> ModelResponse:
        return ModelResponse(
            finish_reason=FinishReason.STOP,
            content="Hello from test runner",
            usage=TokenUsage(
                input_tokens=10,
                output_tokens=5,
                provider="mock_factory_llm",
                model="test-model",
            ),
            provenance=Provenance.primary(provider="mock_factory_llm", model="test-model"),
        )

    async def stream(self, request: LLMRequest) -> AsyncIterator[StreamChunk]:
        yield StreamChunk(
            delta_content="Hello",
            usage=TokenUsage(
                input_tokens=10,
                output_tokens=5,
                provider="mock_factory_llm",
                model="test-model",
            ),
        )


# ======================================================================================
# 1. create_default_registry and Factory Tests
# ======================================================================================


def test_create_default_registry_contains_all_core_tools() -> None:
    """create_default_registry populates 4 filesystem tools, 2 shell tools, and 2 web tools."""
    registry = create_default_registry(enable_mcp=False)
    tools = registry.list_tools()
    tool_names = {t.name for t in tools}

    expected_names = {
        "file_read",
        "file_write",
        "file_edit",
        "file_search",
        "directory_list",
        "bash_run",
        "run_command",
        "web_fetch",
        "web_search",
        "generate_image",
        "character_sheet",
        "story_library",
        "muse_spark",
        "story_outline",
        "story_codex",
        "story_manuscript",
        "story_context",
        "install_package",
    }
    assert expected_names.issubset(tool_names)
    assert "delegate_subagent" in tool_names
    assert "a2a_call" in tool_names
    assert "tool_result_read" in tool_names
    assert len(tools) == 22

    for name in expected_names:
        assert registry.get(name) is not None

    # Test alias
    reg_alias = create_default_tool_registry(enable_mcp=False)
    assert len(reg_alias.list_tools()) == 22

    # Test classmethod factories
    reg_builtins = ToolRegistry.with_builtins(enable_mcp=False)
    assert len(reg_builtins.list_tools()) == 22
    reg_default = ToolRegistry.default(enable_mcp=False)
    assert len(reg_default.list_tools()) == 22


def test_create_default_registry_with_mcp_auto_wiring(tmp_path: Path) -> None:
    """create_default_registry auto-discovers and registers MCP tools when configured."""
    mcp_cfg_file = tmp_path / "mcp.json"
    mcp_cfg_file.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "factory_server": {
                        "command": sys.executable,
                        "args": ["-c", MOCK_MCP_FACTORY_SERVER_SCRIPT],
                        "workspace_root": str(tmp_path),
                    }
                }
            }
        )
    )

    registry = create_default_registry(
        workspace_root=tmp_path,
        enable_mcp=True,
        mcp_config_path=mcp_cfg_file,
    )

    tool_names = {t.name for t in registry.list_tools()}
    assert "file_read" in tool_names
    assert "bash_run" in tool_names
    assert "web_fetch" in tool_names
    assert "web_search" in tool_names
    assert "custom_mcp_action" in tool_names
    assert len(registry.list_tools()) == 23


def test_create_default_registry_graceful_on_broken_mcp(tmp_path: Path) -> None:
    """create_default_registry does not crash when an MCP server fails to connect."""
    bad_cfg_file = tmp_path / "bad_mcp.json"
    bad_cfg_file.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "broken_server": {
                        "command": sys.executable,
                        "args": ["-c", "import sys; sys.exit(1)"],
                        "workspace_root": str(tmp_path),
                    }
                }
            }
        )
    )

    registry = create_default_registry(
        workspace_root=tmp_path,
        enable_mcp=True,
        mcp_config_path=bad_cfg_file,
    )

    # All built-in tools remain registered despite MCP failure
    assert len(registry.list_tools()) == 22
    assert registry.get("file_read") is not None
    assert registry.get("bash_run") is not None
    assert registry.get("web_fetch") is not None
    assert registry.get("web_search") is not None


# ======================================================================================
# 2. UI AgentSessionManager Default Registry Wiring Tests
# ======================================================================================


@pytest.mark.asyncio
async def test_ui_session_manager_wires_default_registry() -> None:
    """AgentSessionManager defaults to create_default_registry() when tools is None."""
    bus = EventBus()
    llm = DummyLLMConnector()

    session_mgr = AgentSessionManager(bus=bus, llm=llm, tools=None)
    assert session_mgr.tools is not None

    tool_names = {t.name for t in session_mgr.tools.list_tools()}
    assert "file_read" in tool_names
    assert "bash_run" in tool_names
    assert len(session_mgr.tools.list_tools()) >= 8

    # Test agent instantiated with default tools
    agent = await session_mgr.get_or_create_agent("test-ui-agent")
    assert agent._tools is session_mgr.tools  # pyright: ignore[reportPrivateUsage]
    assert session_mgr.tools.get("file_read") is not None

    await session_mgr.clear()


def test_create_ui_app_default_registry() -> None:
    """create_ui_app sets up AgentSessionManager with default registry."""
    app = create_ui_app(tools=None)
    mgr = app.state.session_manager
    assert isinstance(mgr, AgentSessionManager)
    assert mgr.tools.get("file_read") is not None
    assert mgr.tools.get("bash_run") is not None


# ======================================================================================
# 3. CLI REPL Tool Registry Wiring Tests
# ======================================================================================


@pytest.mark.asyncio
async def test_cli_run_agent_repl_wires_default_registry() -> None:
    """run_agent_repl_async wires default tool registry and makes tools accessible in turns."""
    mock_llm = DummyLLMConnector()

    with patch(
        "uclone_x.cli.commands.run.create_llm_connector",
        return_value=mock_llm,
    ):
        # Run single-shot turn
        await run_agent_repl_async(
            agent_name="test_cli_agent",
            provider="mock",
            prompt="Perform a test action",
            tools=None,
        )


@pytest.mark.asyncio
async def test_cli_run_agent_repl_custom_tools() -> None:
    """run_agent_repl_async accepts an explicit custom tool registry."""
    mock_llm = DummyLLMConnector()
    custom_reg = ToolRegistry()

    with patch(
        "uclone_x.cli.commands.run.create_llm_connector",
        return_value=mock_llm,
    ):
        await run_agent_repl_async(
            agent_name="test_custom_agent",
            provider="mock",
            prompt="Test with custom registry",
            tools=custom_reg,
        )
