"""The session-storage seam: what the kernel needs, without saying where it is kept.

`agent/session.py` defines `SessionStore`, and `BaseAgent` names that concrete class. It
performs `expanduser`, `read_text` and `os.replace`, so a host that keeps sessions anywhere
but the local filesystem has no seam at all (the core/shell architecture note C6).

This protocol lifts the backend-neutral half of that class's public surface. Deliberately
**not** on it: `storage_dir`, `session_path`, `reap_orphaned_temp_files` and the
`artifacts_dir` parameter. Each is a fact about keeping sessions in files — a temporary
file to reap, a directory to name — and putting them here would make every future backend
implement a filesystem vocabulary in order to satisfy an interface that claims not to
require one.

**The kernel owns *when*, the host owns *where*.** Decision D4 (§11) settled the first
half: `BaseAgent` saves after every completed turn and on `stop()`, and no shell chooses.
This protocol is the second half.

**Concurrency is part of the contract, not an implementation detail.** `save` takes the
state whose `revision` is the precondition and returns the state stamped with the next one,
so a caller that keeps the return value can write again. A backend that ignores the
precondition turns two writers into one silently discarded update — the failure #219 was
filed for. The protocol therefore documents the refusal rather than leaving it to each
backend to invent.

**Where this diverges from §6.4, stated rather than left to be discovered.**

* **Sync, where the design says `async`.** The methods here match `SessionStore`'s existing
  synchronous surface. That keeps this extraction behaviour-free, and it is a real cost:
  §6.4's table names a Postgres adapter as the cloud backend, and a synchronous `load`/`save`
  either blocks the event loop there or grows an async twin later — a protocol change after
  consumers exist, which is the failure this interface-first ordering exists to avoid. The
  choice is deliberate and is the first thing to revisit when a non-file backend is written.
* **`list_session_ids()` rather than `list_sessions(agent_id=None)`.** The `agent_id` filter
  is dropped because `SessionStore` has no such method, so "list this agent's sessions" is
  not expressible through the seam today.
* **`save` returns the stamped state**, where §6.4 shows `-> None`. The return value carries
  the incremented revision, without which a caller cannot write twice.

Design reference: the core/shell architecture note §6.4, and #462 which proposed this
extraction.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from uclone_x.agent.session import SessionState

__all__ = ["SessionStoreProtocol"]


# `runtime_checkable` is kept here and deliberately not on `HostProtocol` or
# `WorkspaceProtocol`. `isinstance` against a runtime-checkable protocol uses `hasattr`,
# which *invokes* properties — so on a protocol whose members are properties it raises out of
# the stub rather than returning `False`, which is an answer nobody asked for. Every member
# here is a method, so the check behaves.
#
# This deviates from the convention the other protocol modules state — `runtime_checkable`
# only where an `isinstance` check is actually performed — and no such check exists yet.
# Kept deliberately: the seam's first job is to let a substitute store be verified as one,
# and a protocol that cannot be checked at the moment someone writes that substitute is a
# seam that arrives one step late.
@runtime_checkable
class SessionStoreProtocol(Protocol):
    """Durable storage for sessions, as the kernel sees it.

    Typed over the existing `SessionState` rather than a new record model. §6.4 asks for
    the two to be reconciled rather than placed beside each other, and a second model would
    give the repository two answers to "what is a session" with no rule for which wins.
    """

    def load(self, session_id: str) -> SessionState | None:
        """Return the stored session, or `None` when there is none.

        `None` means "no session to load", and the shipped backend uses it for both absent
        and unreadable. That is deliberate on its part and argued at length in
        `agent/session.py`: a truncated or unparseable record is treated as absent rather
        than raised, so a damaged file does not make a session unopenable.

        Stated here because the consequence belongs to every backend: a caller reading
        `None` seeds a fresh session and overwrites whatever was there. A backend that can
        distinguish the two cases — a database that can tell "no row" from "connection
        refused" — must raise for the second, because returning `None` there converts a
        transport fault into data loss. The shipped file backend cannot make that
        distinction reliably and does not claim to.
        """
        ...

    def save(
        self, state: SessionState, pending_events: Sequence[Any] | None = None
    ) -> SessionState:
        """Persist `state` and return exactly what a subsequent `load` will yield.

        Refuses when the stored record has moved past the state's precondition (a version or revision), so a lost update
        is an error rather than a silent overwrite. The returned state carries the
        incremented version.
        """
        ...

    def delete(self, session_id: str, artifacts_dir: Path | None = None) -> bool:
        """Remove the session; return whether one was there to remove."""
        ...

    def list_session_ids(self) -> tuple[str, ...]:
        """Every session id this store holds."""
        ...

    def clear_event_log(self, session_id: str) -> None:
        """Remove the session's durable turn events, so the next ones start a new log.

        Called when a session's history is cleared, which clears its events too, and the
        context-snapshot bodies those events refer to. A backend with no event log of its
        own does nothing for the log.
        """
        ...

    def save_context_body(self, session_id: str, digest: str, body: str) -> None:
        """Store a context-snapshot layer body under its SHA-256 `digest`, once."""
        ...

    def load_context_body(self, session_id: str, digest: str) -> str | None:
        """The layer body stored under `digest`, or `None` when there is none."""
        ...
