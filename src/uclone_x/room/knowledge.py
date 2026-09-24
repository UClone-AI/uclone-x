"""What the room needs from a store of each seat's knowledge (#1367).

The orchestrator saves a seat's ontology engine after every turn, beside its session
(#1361), and the resolver loads it into a seat's new engine. Both are kernel code and see
only this protocol; the file-backed implementation is `room.knowledge_store`, an adapter,
composed in by the head that owns the storage root.

**A record that cannot be read is set aside, not refused and not overwritten.** Refusing
the seat left a non-expert with a clone that could never speak again in that conversation
and no way out of it from the app; overwriting it would lose what was there with nothing
saying so. `load_into` renames the record aside -- it is never deleted -- and answers
`SET_ASIDE`; the resolver then builds the seat over a fresh engine, and the orchestrator
states on the seat's next row that its record was set aside (P6). A GET never sets a
record aside: the knowledge read reports it as `unreadable` and changes nothing.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Protocol

from uclone_x.ontology.protocols import OntologyEngineProtocol

__all__ = ["SEAT_KNOWLEDGE_SUBDIR", "KnowledgeLoad", "SeatKnowledgeProtocol"]

#: Where a head keeps seat knowledge, under its storage root: a sibling of the session
#: records (`core/`) and the rooms (`rooms/`).
SEAT_KNOWLEDGE_SUBDIR = "knowledge"


class KnowledgeLoad(StrEnum):
    """What `SeatKnowledgeProtocol.load_into` found."""

    #: The saved knowledge is in the engine.
    LOADED = "loaded"
    #: Nothing was saved for this seat; the engine is as it was.
    ABSENT = "absent"
    #: A record was there and could not be read. It has been renamed aside, kept, and the
    #: engine may be partly loaded: the caller builds the seat over a fresh one.
    SET_ASIDE = "set_aside"


class SeatKnowledgeProtocol(Protocol):
    """Reads and writes one room seat's knowledge, keyed by the seat's session id."""

    def load_into(self, session_id: str, engine: OntologyEngineProtocol) -> KnowledgeLoad:
        """Load the saved knowledge into `engine`, setting an unreadable record aside.

        Raises:
            SeatKnowledgeUnreadableError: A record is there, cannot be read, and could not
                be set aside either; the seat cannot be built without losing it.
        """
        ...

    def read(self, session_id: str, namespace: str) -> OntologyEngineProtocol | None:
        """The saved knowledge in a detached engine, or `None` when none has been saved.

        Changes nothing on disk, whatever it finds.

        Raises:
            SeatKnowledgeUnreadableError: A record is there and cannot be read.
        """
        ...

    def save(self, session_id: str, engine: OntologyEngineProtocol) -> None:
        """Write `engine` as this seat session's knowledge, replacing any earlier record."""
        ...

    def take_set_aside(self, session_id: str) -> bool:
        """Whether a record of this seat was set aside since this was last asked.

        Answers `True` once per set-aside, so the notice lands on one row.
        """
        ...
