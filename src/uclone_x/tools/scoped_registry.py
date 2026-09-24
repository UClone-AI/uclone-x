"""Scoped tool registry proxy filtering tools by an allowed whitelist (Principle 4)."""

from __future__ import annotations

from collections.abc import Sequence

from uclone_x.tools.protocols import ToolProtocol, ToolRegistryProtocol


class ScopedToolRegistry(ToolRegistryProtocol):
    """View over a backing ToolRegistryProtocol restricted to an allowed tools whitelist.

    Prevents cognitive tool overload on smaller local models (such as 7B-14B parameter models)
    by exposing only the tool schemas declared in the active persona's `allowed_tools`.
    """

    def __init__(
        self,
        backing: ToolRegistryProtocol,
        allowed_tools: Sequence[str] | None = None,
    ) -> None:
        self._backing = backing
        # If allowed_tools is None, empty, or contains "*", all tools in backing are permitted
        self._allowed: set[str] | None = (
            None
            if (allowed_tools is None or not allowed_tools or "*" in allowed_tools)
            else set(allowed_tools)
        )

    @property
    def allowed_tools(self) -> set[str] | None:
        """The whitelist set of tool names, or None if unconstrained."""
        return set(self._allowed) if self._allowed is not None else None

    def register(self, tool: ToolProtocol) -> None:
        """Register tool into the backing registry."""
        self._backing.register(tool)

    def get(self, name: str) -> ToolProtocol | None:
        """Look up a tool by name. `None` means *not registered*, never *not permitted*.

        This deliberately ignores the whitelist. `get` answers an existence question, and
        returning `None` for a scoped-out tool answers it with a policy fact instead --
        which made `BaseAgent.execute_tool_call` raise `KeyError("not registered")` for a
        tool that is registered and merely withheld, collapsing the distinction its two
        exception types exist to draw.

        Scoping is what this class does at *advertisement* time, through `list_tools`: the
        model is shown only the permitted names. Refusing an impermissible call is the
        agent's job and is done on both of its paths -- `_execute_single_tool` returns an
        error record mid-turn, `execute_tool_call` raises `PermissionError` -- so nothing
        depends on `get` doing it too, and having it do so silently changed what the
        refusal said.
        """
        return self._backing.get(name)

    def unregister(self, name: str) -> bool:
        """Remove a tool from the backing registry."""
        return self._backing.unregister(name)

    def list_tools(
        self,
        filter_names: tuple[str, ...] | None = None,
    ) -> list[ToolProtocol]:
        """List registered tools matching whitelist and optional caller filter."""
        if self._allowed is None:
            return self._backing.list_tools(filter_names=filter_names)

        effective_filters: tuple[str, ...]
        if filter_names is None:
            effective_filters = tuple(self._allowed)
        else:
            effective_filters = tuple(set(filter_names) & self._allowed)
        return self._backing.list_tools(filter_names=effective_filters)
