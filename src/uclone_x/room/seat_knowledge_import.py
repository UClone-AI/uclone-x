"""The one-time import of retired seat knowledge records into clone facts (step 6).

Before clone-knowledge-graph step 6 each room seat kept its own rules engine, saved after
every turn to `knowledge/<session_id>.yaml` (#1367). Step 6 retired that engine: what a
clone learned is its facts, in its one memory (§3.1), and no seat has a graph of its own.
A record may still hold relations, so this moves them where the clone now keeps what it
learned, once, and keeps the file (§3.7):

* **What moves.** Each relation becomes a fact of the seat's clone with origin `saved`,
  its session the record's session and its conversation the record's room -- the same
  shape a `record_memory_fact` call in that conversation would have left. A relation the
  clone already holds (same subject, predicate and value) is not saved twice, and a
  *retracted* fact counts as held: a relation the person forgot, or corrected to another
  value, is never brought back by the import (#1872). Nothing a person holds is
  superseded: an imported fact is added beside a fact it conflicts with, never over it, so
  a correction made since outranks the import.
* **What does not.** Concepts and axioms. A clone's rules come from a person (§3.1); a
  seat's engine held them only as the regex inducer's guesses, which step 7 removes. They
  are counted in the log, and stay in the kept file.
* **The file is renamed, never deleted**, to `<name>.imported-<UTC time>` beside it
  (`core/set_aside.set_aside`), once every relation in it is saved. A file that cannot be
  read is renamed `<name>.unreadable-<UTC time>` instead, as every unreadable record is
  (#1844). A file whose facts could not all be saved keeps its name, so the next start
  tries again; the duplicate check keeps that retry from saving a fact twice.
* **A file whose name is not a room seat's session** (`sess_room__<room>__<clone>`) is
  left where it is and logged: nothing says which clone it belonged to.
* **Two processes may import at once** (two heads started together). Each imported fact's
  id is derived from its clone and relation (`imported_fact_id`), so both save the *same*
  fact and the store's merge by id leaves one. A record that another process renamed
  first is logged as taken, not as a failed rename (#1872).
* **The log never carries a record's text.** A failure is logged by its error type and
  where in the record it was (line and column, or the field path), never the exception's
  message, which for a validation error quotes the record's values (#1872).
* **The clone's memory is opened through the app's store, unreadable or not.** A store
  that cannot be read is set aside by the store itself when it is opened, exactly as the
  clone's first turn or the Remembers tab would set it aside; the import saves nothing
  into it and keeps the record (author's choice in #1872: checking the file first would
  be a second reader of `memory.json` with its own idea of "readable").
* **A memory with a set-aside copy waits, once** (#1879, #1892). Once the store has set an
  unreadable `memory.json` aside, the next start opens a fresh, readable one, and the facts
  the person forgot inside the copy are not in it to count as held. So when a start finds
  a `memory.json.unreadable-<time>` copy the import has not waited on before, a record
  with relations is kept under its name and logged, and the copy's name is written to the
  import's pending record (`WAITED_RECORD`, beside the records). A copy that was already
  set aside when that record was written no longer blocks: the next start goes ahead, and
  a fact forgotten only inside the copy can come back as an imported fact the person can
  forget again. Before this the wait never ended, because retention (#1867) always keeps
  the newest copies and this build cannot read them to restore (author's choice in #1892:
  one start's wait, over an explicit action no surface offers). A copy set aside later is
  new, and waits once in turn. Only the copies' names are read, never their content
  (author's choice: reading the copy would be the second reader #1872 declined). A
  pending record that cannot be read or written counts as empty, so the import waits
  again rather than going ahead unannounced.

An adapter: it reads and renames files and writes through the clone's memory store.
"""

from __future__ import annotations

import errno
import json
import logging
import os
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, cast

from pydantic import ValidationError

from uclone_x.core.provenance import Provenance
from uclone_x.core.set_aside import set_aside, set_aside_copies, set_aside_unreadable
from uclone_x.memory.store import CrossSessionMemory
from uclone_x.ontology.engine import OntologyEngine
from uclone_x.room.service import SESSION_ID_PREFIX, SESSION_ID_SEPARATOR

__all__ = [
    "IMPORTED_MARKER",
    "SEAT_KNOWLEDGE_SUBDIR",
    "WAITED_RECORD",
    "SeatKnowledgeImport",
    "import_seat_knowledge",
    "imported_fact_id",
]

logger = logging.getLogger(__name__)

#: Where the retired records were written, under the app's storage directory.
SEAT_KNOWLEDGE_SUBDIR: Final = "knowledge"

#: What an imported record's kept name carries between its own name and the time.
IMPORTED_MARKER: Final = ".imported-"

#: The import's pending record, beside the records: per clone, the names of the set-aside
#: memory copies the import has already waited one start on (#1892). Not `*.yaml`, so it
#: is never read as a seat record.
WAITED_RECORD: Final = "set-aside-waited.json"


@dataclass(frozen=True)
class SeatKnowledgeImport:
    """What one import pass did, for its caller and its tests."""

    #: Records whose relations are all in their clone's facts now, and were renamed.
    imported: int = 0
    #: Facts saved, over every record imported.
    facts: int = 0
    #: Records that could not be read, renamed `.unreadable-`.
    unreadable: int = 0
    #: Records left under their own name: an unparseable name, or a save that failed.
    left: int = 0
    #: Records kept under their own name for one start, because their clone's memory has
    #: a set-aside copy the import had not waited on before (#1892).
    waiting: int = 0


def _provenance() -> Provenance:
    """Where an imported fact came from: this import, not a model call or a person."""
    return Provenance.primary(provider="uclone.import", model="seat-knowledge")


#: The namespace `imported_fact_id` derives ids in. Fixed: changing it would let a second
#: import beside an older one save each relation again.
_IMPORT_NAMESPACE: Final = uuid.UUID("6f1c0a52-3d0e-4b8e-9a51-1872c10e5eed")


def imported_fact_id(clone_id: str, subject: str, predicate: str, object_value: str) -> str:
    """The one id every import gives this relation of this clone, in any process."""
    name = "\x1f".join((clone_id, subject, predicate, object_value))
    return f"mem_{uuid.uuid5(_IMPORT_NAMESPACE, name).hex[:12]}"


def _where(exc: BaseException) -> str:
    """An exception as its type and where it happened, never its message.

    A pydantic error's message quotes the offending value (`input_value=...`) and a YAML
    error's quotes the offending line: both are the record's own text, which a log must
    not carry (#1872).
    """
    name = type(exc).__name__
    if isinstance(exc, ValidationError):
        locations = sorted({".".join(str(part) for part in err["loc"]) for err in exc.errors()})
        return f"{name} at {', '.join(locations)}" if locations else name
    mark = getattr(exc, "problem_mark", None)
    if mark is not None:
        return f"{name} at line {mark.line + 1}, column {mark.column + 1}"
    if isinstance(exc, OSError) and exc.errno is not None:
        return f"{name} ({errno.errorcode.get(exc.errno, exc.errno)})"
    return name


def _seat_of(session_id: str) -> tuple[str, str] | None:
    """`(room id, clone id)` from a room seat's session id, or `None` if it is not one."""
    parts = session_id.split(SESSION_ID_SEPARATOR)
    if len(parts) != 3 or parts[0] != SESSION_ID_PREFIX or not parts[1] or not parts[2]:
        return None
    return parts[1], parts[2]


def import_seat_knowledge(
    knowledge_dir: Path,
    memory_for: Callable[[str], CrossSessionMemory],
) -> SeatKnowledgeImport:
    """Move every retired seat record's relations under `knowledge_dir` into clone facts.

    `memory_for` is the app's one store per clone id, so the import is not a second
    whole-document writer beside a running clone. Never raises for one record's failure:
    each is logged and the pass goes on to the next.
    """
    try:
        if not knowledge_dir.is_dir():
            return SeatKnowledgeImport()
        records = sorted(knowledge_dir.glob("*.yaml"))
    except OSError as unlisted:
        # The app still starts: the records stay where they are for the next start (#1872).
        logger.warning(
            "Seat knowledge in %s could not be listed (%s); nothing was imported, and it "
            "will be tried again on the next start",
            knowledge_dir,
            _where(unlisted),
        )
        return SeatKnowledgeImport()
    waited = _read_waited(knowledge_dir)
    newly_waited: dict[str, set[str]] = {}
    imported = facts = unreadable = left = waiting = 0
    for path in records:
        session_id = path.stem
        seat = _seat_of(session_id)
        if seat is None:
            logger.warning(
                "Seat knowledge %s is not named by a room seat's session, so which clone it "
                "belonged to is not known; it was left where it is",
                path,
            )
            left += 1
            continue
        room_id, clone_id = seat
        engine = OntologyEngine()
        try:
            engine.load_from_yaml(path)
        except FileNotFoundError:
            _log_taken(path)
            continue
        except Exception as exc:  # any parse or validation failure: the file is unreadable
            why = _where(exc)  # never the message: it can quote the record (#1872)
            try:
                aside = set_aside_unreadable(path)
            except FileNotFoundError:
                _log_taken(path)
                continue
            except OSError as rename_exc:
                logger.warning(
                    "Seat knowledge %s could not be read (%s) or set aside (%s); left in place",
                    path,
                    why,
                    _where(rename_exc),
                )
                left += 1
                continue
            logger.warning(
                "Seat knowledge %s could not be read (%s); kept as %s, nothing imported",
                path,
                why,
                aside.name,
            )
            unreadable += 1
            continue
        saved = _save_relations(
            engine, path, session_id, room_id, clone_id, memory_for, waited, newly_waited
        )
        if isinstance(saved, _Waiting):
            waiting += 1
            continue
        if saved is None:
            left += 1
            continue
        try:
            kept = set_aside(path, IMPORTED_MARKER)
        except FileNotFoundError:
            # Another process imported the same record and renamed it first. The facts
            # both saved carry the same ids, so the clone holds each once.
            _log_taken(path)
            continue
        except OSError as exc:
            # The facts are saved; the next pass finds them held and saves none again.
            logger.warning(
                "Seat knowledge %s was imported but could not be renamed (%s); it will be "
                "read again on the next start and nothing saved twice",
                path,
                _where(exc),
            )
            left += 1
            facts += saved
            continue
        logger.info(
            "Imported seat knowledge of %r in conversation %s: %d relation(s) now facts of "
            "the clone, %d already held; %d concept(s) and %d axiom(s) not imported (a "
            "clone's rules come from a person). Kept as %s",
            clone_id,
            room_id,
            saved,
            len(engine.list_relations()) - saved,
            len(engine.list_concepts()),
            len(engine.list_axioms()),
            kept.name,
        )
        imported += 1
        facts += saved
    if newly_waited:
        _write_waited(knowledge_dir, waited, newly_waited)
    return SeatKnowledgeImport(
        imported=imported, facts=facts, unreadable=unreadable, left=left, waiting=waiting
    )


class _Waiting:
    """`_save_relations`'s answer for a record held back one start by a set-aside copy."""


_WAITING: Final = _Waiting()


def _read_waited(knowledge_dir: Path) -> dict[str, frozenset[str]]:
    """The pending record: per clone, the set-aside copies already waited on (#1892).

    Absent, unreadable or of the wrong shape, it is empty: every copy then waits once more,
    which is the safe direction.
    """
    path = knowledge_dir / WAITED_RECORD
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as exc:
        logger.warning(
            "The seat knowledge pending record %s could not be read (%s)", path, _where(exc)
        )
        return {}
    clones: object = cast(dict[str, Any], raw).get("clones") if isinstance(raw, dict) else None
    if not isinstance(clones, dict):
        return {}
    waited: dict[str, frozenset[str]] = {}
    for clone, names in cast(dict[str, object], clones).items():
        if isinstance(names, list):
            waited[clone] = frozenset(n for n in cast(list[object], names) if isinstance(n, str))
    return waited


def _write_waited(
    knowledge_dir: Path,
    waited: dict[str, frozenset[str]],
    newly_waited: dict[str, set[str]],
) -> None:
    """Add `newly_waited` to the pending record, written whole through a temporary file.

    A copy retention has deleted since stays listed; its name is never reused (the stamp
    is in it), so it can block nothing. Two processes writing at once keep the last one's
    record: the other's copies then wait one more start.
    """
    clones = {clone: set(names) for clone, names in waited.items()}
    for clone, names in newly_waited.items():
        clones.setdefault(clone, set()).update(names)
    body = {"clones": {clone: sorted(names) for clone, names in sorted(clones.items())}}
    path = knowledge_dir / WAITED_RECORD
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    except OSError as exc:
        temporary.unlink(missing_ok=True)
        logger.warning(
            "The seat knowledge pending record %s could not be written (%s); the import "
            "will wait on the same set-aside copies again on the next start",
            path,
            _where(exc),
        )


def _log_taken(path: Path) -> None:
    """A record that left its name while this pass held it: another import took it."""
    logger.info("Seat knowledge %s was taken by another import; nothing to do here", path)


def _save_relations(
    engine: OntologyEngine,
    path: Path,
    session_id: str,
    room_id: str,
    clone_id: str,
    memory_for: Callable[[str], CrossSessionMemory],
    waited: dict[str, frozenset[str]],
    newly_waited: dict[str, set[str]],
) -> int | _Waiting | None:
    """Save `engine`'s relations as `clone_id`'s facts; the count saved, or `None` on failure.

    `_WAITING` when the clone's memory has a set-aside copy not in `waited` (read at the
    start of this pass, so a copy first seen in this pass waits for every record of its
    clone); its name is added to `newly_waited` for the pending record (#1892).
    """
    relations = engine.list_relations()
    if not relations:
        return 0
    try:
        memory = memory_for(clone_id)
        memory.refresh()
    except Exception as exc:  # an id with no home, or a store that will not open
        logger.warning(
            "Seat knowledge %s was not imported: the memory of %r could not be opened (%s); "
            "it will be tried again on the next start",
            path,
            clone_id,
            _where(exc),
        )
        return None
    if memory.load_failure is not None:
        logger.warning(
            "Seat knowledge %s was not imported: the memory of %r could not be read; it "
            "will be tried again on the next start",
            path,
            clone_id,
        )
        return None
    try:
        copies: set[str] = (
            {c.name for c in set_aside_copies(memory.storage_path)}
            if memory.storage_path is not None
            else set()
        )
    except OSError as exc:
        # Unknown is treated as set aside: the record waits, and nothing is marked waited.
        logger.warning(
            "Seat knowledge %s was not imported: the folder of the memory of %r could not "
            "be listed (%s); it will be tried again on the next start",
            path,
            clone_id,
            _where(exc),
        )
        return None
    unseen = copies - waited.get(clone_id, frozenset())
    if unseen:
        # A fact forgotten inside the set-aside copy is not in the fresh store (#1879).
        newly_waited.setdefault(clone_id, set()).update(copies)
        logger.warning(
            "Seat knowledge %s was not imported yet: a copy of the memory of %r was set "
            "aside unread, and a fact forgotten in it could come back. It goes ahead on the "
            "next start; to keep what the copy holds, restore it with a build that can read "
            "it before then",
            path,
            clone_id,
        )
        return _WAITING
    # Retracted facts count as held: what the person forgot or corrected stays so (#1872).
    held = {
        (f.subject, f.predicate, f.object_value) for f in memory.list_facts(include_retracted=True)
    }
    saved = 0
    for relation in relations:
        key = (
            relation.source_entity.strip(),
            relation.predicate.strip(),
            relation.target_entity.strip(),
        )
        if key in held or not all(key):
            continue
        try:
            memory.record_fact(
                subject=key[0],
                predicate=key[1],
                object_value=key[2],
                provenance=_provenance(),
                source_session_id=session_id,
                confidence=min(1.0, max(0.0, relation.confidence)),
                auto_retract_conflicts=False,
                origin="saved",
                source_room_id=room_id,
                fact_id=imported_fact_id(clone_id, *key),
            )
        except Exception as exc:
            logger.warning(
                "Seat knowledge %s was not fully imported: a fact of %r could not be saved "
                "(%s); it will be tried again on the next start",
                path,
                clone_id,
                _where(exc),
            )
            return None
        held.add(key)
        saved += 1
    return saved
