"""What a host can do, named so a tool can require it and a registry can refuse it.

A capability is a fact about the process the kernel runs in, not a preference. The point
of naming them is that a tool which cannot work is refused **when it is registered or when
a session opens**, rather than failing on the turn that first calls it. That is P6 applied
to composition: the alternative is a tool whose schema the model can see and whose every
invocation fails.

Design reference: the core/shell architecture note, §6.6 (development repository).
"""

from __future__ import annotations

from enum import StrEnum

__all__ = ["Capability"]


class Capability(StrEnum):
    """A capability a host either provides or does not.

    Read and write are separate because they genuinely separate: a host may expose a
    read-only checkout. Workspace access and process execution are separate for the same
    reason — a host may read and write files and never execute, or execute in a container
    against a scratch volume with no user files at all. Review finding 2026-09-02-019
    recorded that collapsing those two axes was already causing confusion in the
    vocabulary; here the separation is structural. (The Markdown findings register that
    held it is retired; the snapshot survives as an eval fixture.)
    """

    WORKSPACE_READ = "workspace.read"
    WORKSPACE_WRITE = "workspace.write"
    PROCESS_EXEC = "process.exec"
    NETWORK_EGRESS = "network.egress"
    SUBAGENT_SPAWN = "subagent.spawn"
    PERSONA_MUTATE = "persona.mutate"
