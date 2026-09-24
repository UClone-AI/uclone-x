"""The Host: everything the kernel may need from the process it runs in.

The kernel describes what it needs and never reaches for it. Today it reaches: it derives a
workspace from `Path.cwd()`, pins an isolation floor, and names concrete classes for its bus,
tracer and session store (the core/shell architecture note C1–C6). Each of those is a fact
about the *process*, and a process that cannot supply it — a worker with no user filesystem,
no shell, and storage in a database — gets a plausible wrong answer instead of a refusal.

A shell builds one Host and hands it over. After that the kernel asks nothing else about the
world.

**`None` is a statement, not a default.** An optional member set to `None` means the host
does not have that capability. Anything that needs it fails at composition with a named
error (§6.6), never by substituting a result at turn time (P6). That is the difference this
protocol exists to create: "no workspace" becomes a value the kernel can read, rather than a
question it answers on the host's behalf.

**Two things are deliberately absent.**

* **Configuration.** `AgentConfig`, personas and budgets describe the *agent*, not the
  process, and stay where they are.
* **Secrets.** The kernel never sees an API key. The shell constructs connectors with their
  credentials and hands over a ready `LLMProviderProtocol`. The one kernel-side secret
  concern is recognising a credential *name* in order to redact it, and that carries no
  values — see `uclone_x.core.secrets`.

Design reference: the core/shell architecture note §6.1.
"""

from __future__ import annotations

from typing import Protocol

from uclone_x.core.capability import Capability
from uclone_x.core.session_store import SessionStoreProtocol
from uclone_x.core.workspace import WorkspaceProtocol
from uclone_x.engine.protocols import EventBusProtocol
from uclone_x.llm.protocols import LLMProviderProtocol
from uclone_x.sandbox.models import IsolationLevel
from uclone_x.sandbox.protocols import SandboxRunnerProtocol
from uclone_x.telemetry.protocols import TraceRecorderProtocol
from uclone_x.tools.protocols import ToolRegistryProtocol

__all__ = ["HostProtocol"]


class HostProtocol(Protocol):
    """The capabilities of one process, as the kernel sees them."""

    @property
    def llm(self) -> LLMProviderProtocol:
        """A ready connector. The shell supplied its credentials; the kernel never sees them."""
        ...

    @property
    def tools(self) -> ToolRegistryProtocol: ...

    @property
    def sessions(self) -> SessionStoreProtocol:
        """Durable session storage. The kernel decides when to save (D4); this is where."""
        ...

    @property
    def tracer(self) -> TraceRecorderProtocol: ...

    @property
    def bus(self) -> EventBusProtocol: ...

    @property
    def workspace(self) -> WorkspaceProtocol | None:
        """The user's files, or `None` when this process has none.

        `None` is why this protocol exists. Nothing may infer a workspace from the
        process's own working directory: on a worker that would hand the agent the
        worker's cwd, which is a real directory and the wrong answer (C3).
        """
        ...

    @property
    def sandbox(self) -> SandboxRunnerProtocol | None:
        """The execution runner, or `None` when this process cannot execute anything.

        Independent of `workspace`: a host may read and write files and never execute, or
        execute against a scratch volume with no user files.
        """
        ...

    @property
    def isolation_floor(self) -> IsolationLevel | None:
        """The weakest isolation this host permits, or `None` when it cannot execute.

        `None` is stricter than any level rather than a new one. With `available_isolation`
        it supplies the two parameters `effective_isolation_level` already accepts and the
        agent does not yet pass (C4).
        """
        ...

    @property
    def available_isolation(self) -> frozenset[IsolationLevel]: ...

    @property
    def capabilities(self) -> frozenset[Capability]:
        """What this host can do, derived from the members above.

        The intent (§6.6) is that a registry refuse a tool whose requirements this set does
        not cover, so a tool the host cannot honour is never registered and the model never
        sees its schema. **No registry consults this yet** — `tools/registry.py` does not
        read capabilities — so this member describes what a host must be able to answer, not
        a check that currently runs.
        """
        ...
