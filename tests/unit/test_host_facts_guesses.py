# pyright: ignore[reportPrivateUsage]
from unittest.mock import MagicMock

import pytest

from uclone_x.agent.base import BaseAgent
from uclone_x.agent.models import AgentConfig
from uclone_x.errors import SandboxViolationError
from uclone_x.sandbox.models import IsolationLevel, WorkspaceIsolation


def test_agent_workspace_fallback_is_none() -> None:
    """Verify that constructing an agent with no workspace supplies `workspace_root=None` and never `Path.cwd()`.

    Killed by: src/uclone_x/agent/base.py :: # If the host supplies a workspace, use it
    Becomes: return Path.cwd()
    """
    agent = BaseAgent(
        config=AgentConfig(agent_id="test", name="test", workspace_dir=None), host=None
    )
    assert agent._resolve_workspace_root() is None  # pyright: ignore[reportPrivateUsage]


def test_agent_tool_isolation_honors_host_floor() -> None:
    """Verify `_resolve_tool_isolation` honors `isolation_floor=None` and host-supplied floors.

    Killed by: src/uclone_x/agent/base.py :: floor = self._host.isolation_floor
    """
    config = AgentConfig(
        agent_id="test",
        name="test",
        # Default config requests WorkspaceIsolation
    )

    # 1. Test host with no execution runner (isolation_floor=None)
    mock_host_no_runner = MagicMock()
    mock_host_no_runner.isolation_floor = None
    mock_host_no_runner.available_isolation = frozenset([IsolationLevel.WORKSPACE])

    agent_no_runner = BaseAgent(config=config, host=mock_host_no_runner)

    with pytest.raises(SandboxViolationError, match="Host provides no execution runner"):
        agent_no_runner._resolve_tool_isolation()  # pyright: ignore[reportPrivateUsage]

    # 2. Test host with CONTAINER floor
    mock_host_container = MagicMock()
    mock_host_container.isolation_floor = IsolationLevel.CONTAINER
    mock_host_container.available_isolation = frozenset(
        [IsolationLevel.WORKSPACE, IsolationLevel.CONTAINER]
    )

    agent_container = BaseAgent(config=config, host=mock_host_container)

    with pytest.raises(
        NotImplementedError, match="Cannot auto-upgrade to ContainerIsolation without an image"
    ):
        agent_container._resolve_tool_isolation()  # pyright: ignore[reportPrivateUsage]

    # 3. Test host with WORKSPACE floor (should return WorkspaceIsolation)
    mock_host_workspace = MagicMock()
    mock_host_workspace.isolation_floor = IsolationLevel.WORKSPACE
    mock_host_workspace.available_isolation = frozenset([IsolationLevel.WORKSPACE])

    agent_workspace = BaseAgent(config=config, host=mock_host_workspace)

    policy = agent_workspace._resolve_tool_isolation()  # pyright: ignore[reportPrivateUsage]
    assert isinstance(policy, WorkspaceIsolation)
