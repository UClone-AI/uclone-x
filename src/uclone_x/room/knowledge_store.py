"""A room seat's knowledge, kept on disk beside its session (#1367): the file adapter.

Each seat is composed with an ontology engine of its own (P7), and that engine used to live
only in the running agent: a restart lost everything the seat had learned, and the room's
knowledge read could only say the seat was not running. The orchestrator now writes the
engine here after every turn, in the same step that writes the seat's session (#1361), and
two readers load it back without composing anything else:

* `RoomAgentResolver.resolve` loads it into the seat's new engine before the seat's first
  turn, so the seat continues from what it knew;
* the room's knowledge read loads it into a detached engine when no agent is running, so
  the dock can show what a seat remembers without building the seat to find out.

**The format is the engine's own.** `OntologyEngine.save_to_yaml` / `load_from_yaml` are
what `ucx ontology` already writes and reads; 1:1 chat keeps its one shared engine in
memory and has no persistence to reuse. A second serialisation would be a second thing to
keep in step with the engine's models.

**No import of the engine.** This module writes files, so it is an adapter, and an adapter
imports only the kernel; `OntologyEngine` is itself an adapter. The head passes the factory
that builds a detached engine for `read`, and the engine's YAML methods are reached through
`_YamlBacked`.

**One file per seat session**, named by the session id, which is already unique per
`(room, participant)` and already validated as a file name. Not by the ontology namespace:
that is an IRI and holds separators.

**An absent record and an unreadable one are different answers.** `OntologyEngine.load_from_yaml`
on its own returns quietly on a document that is not a mapping, which would present a
damaged record as an empty memory (P6); both paths here check first.

**An unreadable record is set aside, never deleted.** `load_into` -- the seat is about to
take a turn -- renames it to `<session_id>.unreadable-<UTC time>.yaml` beside where it was,
logs where it went and why, and answers `SET_ASIDE`; `take_set_aside` then tells the turn
once, so its row can say the record was set aside. `read` -- a GET from the dock -- changes
nothing and raises instead. Neither puts the file's location or the parser's words in the
message a person is shown: those are on the exception's `path` and `cause`, and in the log.
"""

from __future__ import annotations

import logging
import os
import tempfile
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import yaml

from uclone_x.agent.session import validate_session_id
from uclone_x.errors import SeatKnowledgeUnreadableError
from uclone_x.ontology.protocols import OntologyEngineProtocol
from uclone_x.room.knowledge import KnowledgeLoad

__all__ = ["SeatKnowledgeStore"]

logger = logging.getLogger(__name__)


@runtime_checkable
class _YamlBacked(Protocol):
    """The engine's own YAML round trip, which `ucx ontology` also uses."""

    def save_to_yaml(self, file_path: Path) -> None: ...

    def load_from_yaml(self, file_path: Path) -> None: ...


class SeatKnowledgeStore:
    """Reads and writes each room seat's ontology engine, one YAML file per seat session.

    Implements `uclone_x.room.knowledge.SeatKnowledgeProtocol`.
    """

    def __init__(self, root: Path, engine_factory: Callable[[str], OntologyEngineProtocol]) -> None:
        self._root = root
        #: Builds the detached engine `read` loads into, from the seat's namespace.
        self._engine_factory = engine_factory
        #: Seat sessions whose record was set aside and whose turn has not yet said so.
        self._set_aside: set[str] = set()

    @property
    def root(self) -> Path:
        return self._root

    def path(self, session_id: str) -> Path:
        """The file holding this seat session's knowledge, whether or not it exists yet."""
        validate_session_id(session_id)
        return self._root / f"{session_id}.yaml"

    def load_into(self, session_id: str, engine: OntologyEngineProtocol) -> KnowledgeLoad:
        """Load the saved knowledge into `engine`, setting an unreadable record aside.

        `ABSENT` when none has been saved. `SET_ASIDE` when a record is there and cannot be
        read -- it is not valid YAML, is not a knowledge record, or holds an entry the
        engine's models refuse: it has been renamed aside and kept, and `engine` may be
        partly loaded, so the caller builds the seat over a fresh one.

        Raises:
            SeatKnowledgeUnreadableError: The record cannot be read and the rename failed
                too; it is where it was. Building the seat over an empty engine would save
                that emptiness over it after the turn.
            TypeError: `engine` has no YAML round trip, which is the only format this
                store writes.
        """
        path = self.path(session_id)
        if not path.exists():
            return KnowledgeLoad.ABSENT
        cause = self._load(path, engine)
        if cause is None:
            return KnowledgeLoad.LOADED
        aside = self._aside_path(path)
        try:
            os.rename(path, aside)
        except OSError as exc:
            logger.warning(
                "Room seat knowledge at %s could not be read (%s) and could not be set aside "
                "(%s); the seat is refused so that its next save does not overwrite it.",
                path,
                cause,
                exc,
            )
            unreadable = SeatKnowledgeUnreadableError(
                "This clone's knowledge record for this conversation could not be read, and "
                "could not be set aside.",
                path=path,
                cause=cause,
            )
            raise unreadable from exc
        logger.warning(
            "Room seat knowledge at %s could not be read (%s); it was set aside, unchanged, as %s.",
            path,
            cause,
            aside,
        )
        self._set_aside.add(session_id)
        return KnowledgeLoad.SET_ASIDE

    def take_set_aside(self, session_id: str) -> bool:
        """Whether this seat's record was set aside since this was last asked; once each."""
        if session_id not in self._set_aside:
            return False
        self._set_aside.discard(session_id)
        return True

    def read(self, session_id: str, namespace: str) -> OntologyEngineProtocol | None:
        """The saved knowledge in an engine of its own, or `None` when none has been saved.

        For a reader that has no running agent to ask. The engine is detached: nothing
        writes it back, and nothing on disk is changed, whatever is found.

        Raises:
            SeatKnowledgeUnreadableError: A record is there and cannot be read. Its message
                is plain; `path` and `cause` carry the rest.
        """
        path = self.path(session_id)
        if not path.exists():
            return None
        engine = self._engine_factory(namespace)
        cause = self._load(path, engine)
        if cause is not None:
            logger.warning("Room seat knowledge at %s could not be read: %s", path, cause)
            raise SeatKnowledgeUnreadableError(
                "This clone's knowledge record for this conversation could not be read.",
                path=path,
                cause=cause,
            )
        return engine

    @staticmethod
    def _load(path: Path, engine: OntologyEngineProtocol) -> str | None:
        """Load `path` into `engine`; why it could not be read, for the log, or `None`."""
        if not isinstance(engine, _YamlBacked):
            raise TypeError(
                f"Seat knowledge is stored in the engine's own YAML format; a "
                f"{type(engine).__name__} cannot load it."
            )
        try:
            document: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
            if not isinstance(document, dict):
                raise ValueError(f"it holds a {type(document).__name__}, not a knowledge record")
            engine.load_from_yaml(path)
        except Exception as exc:
            return f"{type(exc).__name__}: {exc}"
        return None

    @staticmethod
    def _aside_path(path: Path) -> Path:
        """A name beside `path` that nothing holds yet; the record is never written over."""
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        aside = path.with_name(f"{path.stem}.unreadable-{stamp}.yaml")
        n = 1
        while aside.exists():
            aside = path.with_name(f"{path.stem}.unreadable-{stamp}-{n}.yaml")
            n += 1
        return aside

    def save(self, session_id: str, engine: OntologyEngineProtocol) -> None:
        """Write `engine` as this seat session's knowledge, replacing the file atomically.

        Written to a temporary file in the same directory and renamed over the record, so
        a reader -- or a crash -- sees the old record or the new one, never half of one.

        Raises:
            TypeError: `engine` has no YAML round trip.
            OSError: The directory or the file could not be written.
        """
        if not isinstance(engine, _YamlBacked):
            raise TypeError(
                f"Seat knowledge is stored in the engine's own YAML format; a "
                f"{type(engine).__name__} cannot be saved in it."
            )
        path = self.path(session_id)
        self._root.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self._root, prefix=f".{session_id}.", suffix=".tmp")
        os.close(fd)
        try:
            engine.save_to_yaml(Path(tmp))
            with open(tmp, "rb") as written:
                os.fsync(written.fileno())
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
