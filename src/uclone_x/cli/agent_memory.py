"""Build a shell's cross-session memory, reporting a bad agent id as a usage error.

`ucx a2a serve` and `ucx acp serve` both take an `--agent-id` and both hand it straight to
`default_cross_session_memory`, where it becomes the name of that agent's home directory.
An id no directory can carry is a mistake in the argument, and `ucx run` already reports
that class of mistake as exit 2 with a message rather than as a framed traceback (P0: the
person who typed it has to be able to act on it). One helper, because two copies of a
refusal drift into two different answers to the same question.
"""

from __future__ import annotations

import typer
from rich.console import Console
from rich.markup import escape

from uclone_x.core.agent_home import AgentHomeError
from uclone_x.memory.store import CrossSessionMemory, default_cross_session_memory

__all__ = ["memory_for_agent_id"]

_console = Console()


def memory_for_agent_id(agent_id: str) -> CrossSessionMemory:
    """Open `agent_id`'s memory, or exit 2 naming the id and the rule it broke.

    Raises:
        typer.Exit: The id cannot be an agent home directory name.
    """
    try:
        return default_cross_session_memory(agent_id)
    except AgentHomeError as exc:
        _console.print(f"[bold red]Invalid --agent-id:[/bold red] {escape(str(exc))}")
        raise typer.Exit(code=2) from exc
