"""Tests for ScopedToolRegistry (Principle 4).

Verifies whitelist scoping over backing tool registries to prevent cognitive
tool overload on local models.
"""

from __future__ import annotations

from uclone_x.tools.protocols import ToolRegistryProtocol
from uclone_x.tools.registry import LocalTool, ToolRegistry
from uclone_x.tools.scoped_registry import ScopedToolRegistry


def _build_sample_registry() -> ToolRegistryProtocol:
    reg = ToolRegistry()
    reg.register(LocalTool(name="write_to_file", description="write file"))
    reg.register(LocalTool(name="file_read", description="read file"))
    reg.register(LocalTool(name="bash_run", description="run shell"))
    reg.register(LocalTool(name="web_search", description="search web"))
    return reg


def test_scoped_tool_registry_whitelisting() -> None:
    backing = _build_sample_registry()
    scoped = ScopedToolRegistry(backing, allowed_tools=("write_to_file", "file_read"))

    assert scoped.allowed_tools == {"write_to_file", "file_read"}

    tools = scoped.list_tools()
    tool_names = {t.name for t in tools}
    assert tool_names == {"write_to_file", "file_read"}

    # `get` answers existence, not permission: a scoped-out but registered tool is found
    # here, and refusing the call is the agent's job. Amends assertions that pinned `None`
    # for a scoped-out tool, which made a withheld tool indistinguishable from an absent
    # one at `BaseAgent.execute_tool_call`.
    assert scoped.get("write_to_file") is not None
    assert scoped.get("file_read") is not None
    assert scoped.get("bash_run") is not None
    assert scoped.get("unregistered_tool") is None


def test_scoped_tool_registry_unconstrained_when_none_or_wildcard() -> None:
    backing = _build_sample_registry()

    scoped_none = ScopedToolRegistry(backing, allowed_tools=None)
    assert scoped_none.allowed_tools is None
    assert len(scoped_none.list_tools()) == 4
    assert scoped_none.get("bash_run") is not None

    scoped_empty = ScopedToolRegistry(backing, allowed_tools=())
    assert scoped_empty.allowed_tools is None
    assert len(scoped_empty.list_tools()) == 4

    scoped_wildcard = ScopedToolRegistry(backing, allowed_tools=("*", "write_to_file"))
    assert scoped_wildcard.allowed_tools is None
    assert len(scoped_wildcard.list_tools()) == 4


def test_scoped_tool_registry_filter_intersection() -> None:
    backing = _build_sample_registry()
    scoped = ScopedToolRegistry(backing, allowed_tools=("write_to_file", "file_read"))

    # Caller passes a filter including both an allowed and disallowed tool
    filtered = scoped.list_tools(filter_names=("file_read", "bash_run"))
    assert len(filtered) == 1
    assert filtered[0].name == "file_read"

    # Caller passes a filter with completely disallowed tools
    disallowed = scoped.list_tools(filter_names=("bash_run", "web_search"))
    assert len(disallowed) == 0


def test_scoped_tool_registry_register_forwards_to_backing() -> None:
    backing = ToolRegistry()
    scoped = ScopedToolRegistry(backing, allowed_tools=("new_tool",))

    new_tool = LocalTool(name="new_tool", description="dynamically added")
    scoped.register(new_tool)

    assert backing.get("new_tool") is not None
    assert scoped.get("new_tool") is not None
